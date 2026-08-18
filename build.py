import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Literal

# --- COMPILATION BACKEND ---
BACKEND: Literal["pyinstaller", "nuitka"] = "pyinstaller"

_start = time.perf_counter()

from src import __version__

APP_NAME = "newt"
ENTRY_POINT = "main.py"
VERSION = __version__

SYS_NAME = platform.system().lower()
MACHINE = platform.machine().lower()
if SYS_NAME == "windows":
    MACHINE = "x86_64" if MACHINE == "amd64" else MACHINE

BASE_DIR = Path(__file__).parent.resolve()
DIST_DIR = BASE_DIR / "dist"
BUILD_DIR = BASE_DIR / "build"
CACHE_DIR = BASE_DIR / ".nuitka_cache"
os.environ["NUITKA_CACHE_DIR"] = str(CACHE_DIR)

BASENAME = f"{APP_NAME}-{VERSION}-{SYS_NAME}-{MACHINE}"
STAGING_DIR = BASE_DIR / BASENAME
ARCHIVE_NAME = BASE_DIR / BASENAME

FILES_TO_INCLUDE = ["README.md", "LICENSE"]
DIRS_TO_INCLUDE = ["data"]

UNNEEDED_EXTENSIONS = {
    ".pyi",
    ".h",
    ".hpp",
    ".c",
    ".cpp",
    ".cmake",
    ".pdb",
    ".pyo",
    ".pyx",
    ".pxd",
}
UNNEEDED_DIR_NAMES = {
    "include",
    "cmake",
    "pkgconfig",
    "tests",
    "testing",
    "__pycache__",
    "site-packages/torch/test",
    "site-packages/numpy/tests",
}


def get_out_dist_dir() -> Path:
    """Returns the standalone build output directory based on current BACKEND."""
    if BACKEND == "pyinstaller":
        return DIST_DIR / APP_NAME
    return DIST_DIR / "main.dist"


def parse_args():
    parser = argparse.ArgumentParser(description=f"Newt Build Script ({BACKEND})")
    parser.add_argument(
        "--full-clean",
        action="store_true",
        help="Clean entire dist/, build/ and cache before build",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Skip creating archive (standalone dir only)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Exclude heavy NVIDIA CUDA runtime packages",
    )
    return parser.parse_args()


def clean(full: bool = False):
    print(f"[*] Cleaning build directory for backend [{BACKEND}]...")
    if full:
        if DIST_DIR.exists():
            shutil.rmtree(DIST_DIR)
            print("  Deleted: dist/")
        if BUILD_DIR.exists():
            shutil.rmtree(BUILD_DIR)
            print("  Deleted: build/")
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            print("  Deleted: .nuitka_cache/")
    else:
        out_dist = get_out_dist_dir()
        if out_dist.exists():
            shutil.rmtree(out_dist)
            print(f"  Deleted: {out_dist.relative_to(BASE_DIR)}")

    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)


def post_build_cleanup(dist_path: Path):
    print(f"[*] Performing post-build cleanup on {dist_path.name}...")
    removed_bytes = 0

    for root, _, files in os.walk(dist_path, topdown=False):
        current_path = Path(root)

        if any(bad_dir in str(current_path) for bad_dir in UNNEEDED_DIR_NAMES):
            for f in current_path.rglob("*"):
                if f.is_file():
                    removed_bytes += f.stat().st_size
            try:
                shutil.rmtree(current_path)
            except OSError:
                pass
            continue

        for file_name in files:
            file_path = current_path / file_name
            if file_path.suffix.lower() in UNNEEDED_EXTENSIONS:
                removed_bytes += file_path.stat().st_size
                try:
                    file_path.unlink()
                except OSError:
                    pass

            elif SYS_NAME == "linux" and file_path.suffix == ".so":
                try:
                    subprocess.run(
                        ["strip", "--strip-unneeded", str(file_path)],
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                except Exception:  # noqa: BLE001
                    pass

    mb_saved = removed_bytes / (1024 * 1024)
    print(f"[$] Cleanup finished! Saved ~{mb_saved:.2f} MB.")


def generate_build_info():
    git_hash = "unknown"
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        pass

    return {
        "version": VERSION,
        "commit": git_hash,
        "build_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": f"{SYS_NAME}-{MACHINE}",
        "backend": BACKEND,
        "python_version": platform.python_version(),
    }


def compile_exe(cpu_only: bool = False):
    print(f"[!] Compiling {APP_NAME} using [{BACKEND.upper()}] backend...")
    DIST_DIR.mkdir(exist_ok=True)

    if BACKEND == "pyinstaller":
        BUILD_DIR.mkdir(exist_ok=True)

        import piper

        espeak_data = Path(piper.__file__).parent / "espeak-ng-data"
        if not espeak_data.exists():
            print(
                f"[!] WARNING: espeak-ng-data not found at {espeak_data}! Audio generation WILL crash!"
            )

        cmd = [
            sys.executable,
            "-m",
            "PyInstaller",
            f"{ENTRY_POINT}",
            "--noconfirm",
            "--onedir",
            "--contents-directory=bin",
            f"--name={APP_NAME}",
            f"--distpath={DIST_DIR}",
            f"--workpath={BUILD_DIR}",
            f"--specpath={BUILD_DIR}",
            # "--clean",
            "--collect-data=silero_vad",
            f"--add-data={espeak_data}{os.pathsep}piper/espeak-ng-data",
            # "--collect-all=piper_phonemize",
            "--collect-all=piper",
            "--collect-all=onnxruntime",
            "--collect-all=textual",
            "--collect-all=rich",
            "--collect-all=llama_cpp",
            "--exclude-module=tkinter",
            "--exclude-module=matplotlib",
            "--exclude-module=IPython",
            "--exclude-module=pytest",
        ]

        if cpu_only:
            print("[*] CPU-only mode: Excluding CUDA/NVIDIA modules...")
            cmd.extend(
                [
                    "--exclude-module=nvidia",
                    "--exclude-module=triton",
                ]
            )

    elif BACKEND == "nuitka":
        cmd = [
            sys.executable,
            "-m",
            "nuitka",
            "--standalone",
            "--lto=no",
            "--assume-yes-for-downloads",
            "--plugin-enable=anti-bloat",
            "--plugin-enable=pkg-resources",
            "--module-parameter=torch-disable-jit=yes",
            "--include-module=unittest",  # Keeps C-extensions stable
            "--nofollow-import-to=IPython",
            "--nofollow-import-to=matplotlib",
            "--nofollow-import-to=sympy",
            "--nofollow-import-to=mpmath",
            "--nofollow-import-to=tensorflow",
            "--nofollow-import-to=tensorboard",
            "--nofollow-import-to=flax",
            "--nofollow-import-to=jax",
            "--nofollow-import-to=triton",
            "--nofollow-import-to=torch._dynamo",
            "--nofollow-import-to=torch.distributions",
        ]

        if cpu_only:
            print("[*] CPU-only mode: Stripping CUDA/NVIDIA dependencies...")
            cmd.extend(
                [
                    "--nofollow-import-to=nvidia",
                    "--nofollow-import-to=torch.cuda",
                    "--nofollow-import-to=cupy",
                ]
            )

        cmd.extend(
            [
                f"--output-dir={DIST_DIR}",
                f"--output-filename={APP_NAME}",
                ENTRY_POINT,
            ]
        )
    else:
        raise ValueError(f"Unknown BACKEND: {BACKEND}")

    subprocess.run(cmd, check=True)

    out_dist = get_out_dist_dir()
    if out_dist.exists():
        post_build_cleanup(out_dist)

    _compile_end = time.perf_counter()
    print(
        f"Compilation ended in [{datetime.timedelta(seconds=int(_compile_end - _start))}]"
    )


def make_compressed_archive():
    print("[*] Creating high-compression release archive...")
    STAGING_DIR.mkdir(exist_ok=True)
    out_dist = get_out_dist_dir()

    if out_dist.exists():
        shutil.copytree(out_dist, STAGING_DIR, dirs_exist_ok=True)

    info_path = STAGING_DIR / "build_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(generate_build_info(), f, indent=2)

    for file_name in FILES_TO_INCLUDE:
        file_path = BASE_DIR / file_name
        if file_path.exists():
            shutil.copy(file_path, STAGING_DIR / file_name)

    for dir_name in DIRS_TO_INCLUDE:
        dir_path = BASE_DIR / dir_name
        if dir_path.exists():
            if dir_name == "data":
                shutil.copytree(
                    dir_path,
                    STAGING_DIR / dir_name,
                    ignore=shutil.ignore_patterns(
                        "models",
                        ".cache",
                        "*.gguf",
                        "*.pt",
                        "*.bin",
                        "*.safetensors",
                        "logs",
                        "*.png",
                    ),
                )
            else:
                shutil.copytree(dir_path, STAGING_DIR / dir_name)

    # archive_file = ARCHIVE_NAME.with_suffix(".tar.xz")
    archive_file = BASE_DIR / f"{BASENAME}.tar.xz"
    compressed = False
    if shutil.which("tar") and shutil.which("xz"):
        try:
            subprocess.run(
                [
                    "tar",
                    "-I",
                    "xz -T0",  # -T0 makes ALL CPU cores to work
                    "-cf",
                    str(archive_file),
                    "-C",
                    str(STAGING_DIR.parent),
                    STAGING_DIR.name,
                ],
                check=True,
            )
            compressed = True
        except Exception as e:  # noqa: BLE001
            print(f"[!] System tar failed ({e}), falling back to python tarfile...")

    if not compressed:
        with tarfile.open(archive_file, "w:xz") as tf:
            tf.add(STAGING_DIR, arcname=STAGING_DIR.name)

    shutil.rmtree(STAGING_DIR)

    size_mb = archive_file.stat().st_size / (1024 * 1024)
    print(f"[$] Archive created successfully: {archive_file.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    args = parse_args()
    try:
        clean(full=args.full_clean)
        compile_exe(cpu_only=args.cpu_only)
        if not args.no_archive:
            make_compressed_archive()
    except subprocess.CalledProcessError as e:
        print(f"[!] Compilation error: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"[!] Unexpected error: {e}")
