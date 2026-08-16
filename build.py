import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

from src.utils import __version__

# --- Конфігурація ---
APP_NAME = "newt"
ENTRY_POINT = "main.py"
VERSION = __version__

SYS_NAME = platform.system().lower()
MACHINE = platform.machine().lower()
if SYS_NAME == "windows":
    MACHINE = "x86_64" if MACHINE == "amd64" else MACHINE

BASE_DIR = Path(__file__).parent.resolve()
DIST_DIR = BASE_DIR / "dist"
CACHE_DIR = BASE_DIR / ".nuitka_cache"
os.environ["NUITKA_CACHE_DIR"] = str(CACHE_DIR)

BASENAME = f"{APP_NAME}-{VERSION}-{SYS_NAME}-{MACHINE}"
STAGING_DIR = BASE_DIR / BASENAME
ARCHIVE_NAME = BASE_DIR / BASENAME

FILES_TO_INCLUDE = ["README.md", "LICENSE"]
DIRS_TO_INCLUDE = ["data"]

UNNEEDED_EXTENSIONS = {".pyi", ".h", ".hpp", ".c", ".cpp", ".cmake", ".pdb", ".pyo"}
UNNEEDED_DIR_NAMES = {
    "include",
    "cmake",
    "pkgconfig",
    "tests",
    "testing",
    "__pycache__",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Newt Build Script")
    parser.add_argument(
        "--full-clean",
        action="store_true",
        help="Clean entire dist/ and cache before build",
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
    print("[*] Cleaning build directory...")
    if full:
        if DIST_DIR.exists():
            shutil.rmtree(DIST_DIR)
            print("  Deleted: dist/")
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            print("  Deleted: .nuitka_cache/")
    else:
        out_dist = DIST_DIR / "main.dist"
        if out_dist.exists():
            shutil.rmtree(out_dist)
            print("  Deleted: dist/main.dist")

    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)


def post_build_cleanup(dist_path: Path):
    print("[*] Performing post-build cleanup on main.dist...")
    removed_bytes = 0

    for root, _, files in os.walk(dist_path, topdown=False):
        current_path = Path(root)

        if current_path.name in UNNEEDED_DIR_NAMES:
            for f in current_path.rglob("*"):
                if f.is_file():
                    removed_bytes += f.stat().st_size
            shutil.rmtree(current_path)
            continue

        for file_name in files:
            file_path = current_path / file_name
            if file_path.suffix.lower() in UNNEEDED_EXTENSIONS:
                removed_bytes += file_path.stat().st_size
                file_path.unlink()

            elif SYS_NAME == "linux" and file_path.suffix == ".so":
                try:
                    subprocess.run(
                        ["strip", "--strip-unneeded", str(file_path)],
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                except Exception:  # noqa: BLE001, S110
                    pass

    mb_saved = removed_bytes / (1024 * 1024)
    print(f"[$] Cleanup finished! Saved ~{mb_saved:.2f} MB.")


def generate_build_info():
    """Створює файл метаданих збірки."""
    git_hash = "unknown"
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001, S110
        pass

    return {
        "version": VERSION,
        "commit": git_hash,
        "build_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": f"{SYS_NAME}-{MACHINE}",
        "python_version": platform.python_version(),
    }


def compile_exe(cpu_only: bool = False):
    print(f"[!] Compiling {APP_NAME} with Nuitka...")
    DIST_DIR.mkdir(exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "nuitka",
        "--standalone",
        "--lto=no",
        "--assume-yes-for-downloads",
        "--module-parameter=torch-disable-jit=yes",
        "--nofollow-import-to=sympy",
        "--nofollow-import-to=mpmath",
        "--nofollow-import-to=tensorflow",
        "--nofollow-import-to=flax",
        "--nofollow-import-to=jax",
        "--nofollow-import-to=torch.testing",
        "--nofollow-import-to=torch._dynamo",
        "--nofollow-import-to=torch.distributions",
        "--nofollow-import-to=unittest",
        "--nofollow-import-to=pytest",
        "--enable-plugin=anti-bloat",
    ]

    if cpu_only:
        cmd.extend(
            [
                "--nofollow-import-to=nvidia",
                "--nofollow-import-to=torch.cuda",
            ]
        )

    cmd.extend(
        [
            f"--output-dir={DIST_DIR}",
            f"--output-filename={APP_NAME}",
            ENTRY_POINT,
        ]
    )

    subprocess.run(cmd, check=True)

    nuitka_dist = DIST_DIR / "main.dist"
    if nuitka_dist.exists():
        post_build_cleanup(nuitka_dist)


def make_compressed_archive():
    print("[*] Creating high-compression release archive...")
    STAGING_DIR.mkdir(exist_ok=True)
    nuitka_output_dir = DIST_DIR / "main.dist"

    if SYS_NAME == "linux":
        # Linux: .so and bins to bin/
        bin_dir = STAGING_DIR / "bin"
        if nuitka_output_dir.exists():
            shutil.copytree(nuitka_output_dir, bin_dir, dirs_exist_ok=True)

        launcher_path = STAGING_DIR / APP_NAME
        with open(launcher_path, "w", encoding="utf-8") as f:
            f.write("#!/bin/bash\n")
            f.write(
                'DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"\n'
            )
            f.write(f'exec "$DIR/bin/{APP_NAME}" "$@"\n')
        launcher_path.chmod(0o755)
    else:
        # Everything into root on Windows
        if nuitka_output_dir.exists():
            shutil.copytree(nuitka_output_dir, STAGING_DIR, dirs_exist_ok=True)

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

    archive_file = ARCHIVE_NAME.with_suffix(".tar.xz")
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
