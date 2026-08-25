import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from src import __version__

APP_NAME = "atlas"
ENTRY_POINT = "main.py"
VERSION = __version__

SYS_NAME = platform.system().lower()
MACHINE = platform.machine().lower()
if SYS_NAME == "windows":
    MACHINE = "x86_64" if MACHINE == "amd64" else MACHINE

BASE_DIR = Path(__file__).parent.parent.resolve()
DIST_DIR = BASE_DIR / "dist"
BUILD_DIR = BASE_DIR / "build"
OUT_DIST_DIR = DIST_DIR / APP_NAME

FILES_TO_INCLUDE = ["README.md", "LICENSE"]
DIRS_TO_INCLUDE = ["data", "config", "plugins", "runtime"]

DIR_IGNORE_PATTERNS: dict[str, list[str]] = {
    "data": [
        "models",
        ".cache",
        "*.gguf",
        "*.pt",
        "*.bin",
        "*.safetensors",
        "logs",
        "*.png",
    ],
    "plugins": [".venv", "venv", "*.pyc"],
}

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
UNNEEDED_CUDA_LIBS = {
    "libcufft.so.12",
    "libcusparse.so.12",
    "libnvJitLink.so.13",
    "libnppc.so.12",
    "libnppig.so.12",
}
UNNEEDED_DIR_NAMES = {
    "include",
    "cmake",
    "pkgconfig",
    "tests",
    "testing",
    "__pycache__",
    "torch/bin",
    "torch/include",
    "torch/share",
    "site-packages/torch/bin",
    "site-packages/torch/test",
    "site-packages/numpy/tests",
}
_SIMPLE_UNNEEDED_DIRS = {n for n in UNNEEDED_DIR_NAMES if "/" not in n}
_SUFFIX_UNNEEDED_DIRS = tuple(n for n in UNNEEDED_DIR_NAMES if "/" in n)

PROBLEMATIC_LINUX_LIBS = {"libstdc++.so.6", "libgcc_s.so.1", "libm.so.6"}


def log(message: str, level: str = "INFO"):
    stream = sys.stderr if level in ("WARN", "ERROR") else sys.stdout
    print(f"[{level}] {message}", file=stream)


def parse_args():
    parser = argparse.ArgumentParser(description="Atlas Build Script")
    parser.add_argument("--full-clean", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    return parser.parse_args()


def get_basename(cpu_only: bool) -> str:
    suffix = "-cpu-only" if cpu_only else ""
    return f"{APP_NAME}-{VERSION}-{SYS_NAME}-{MACHINE}{suffix}"


def get_exe_name() -> str:
    return f"{APP_NAME}.exe" if SYS_NAME == "windows" else APP_NAME


def clean(staging_dir: Path, full: bool = False):
    if full:
        for d in [DIST_DIR, BUILD_DIR]:
            if d.exists():
                shutil.rmtree(d)
    elif OUT_DIST_DIR.exists():
        shutil.rmtree(OUT_DIST_DIR)

    if staging_dir.exists():
        shutil.rmtree(staging_dir)


def _should_prune_dir(path: Path, name: str) -> bool:
    if name in _SIMPLE_UNNEEDED_DIRS:
        return True
    posix = path.as_posix()
    return any(posix.endswith(suffix) for suffix in _SUFFIX_UNNEEDED_DIRS)


def post_build_cleanup(dist_path: Path):
    for root, dirs, _files in os.walk(dist_path, topdown=True):
        current_path = Path(root)
        for d in list(dirs):
            if _should_prune_dir(current_path / d, d):
                shutil.rmtree(current_path / d, ignore_errors=True)
                dirs.remove(d)

    for root, _dirs, files in os.walk(dist_path):
        current_path = Path(root)
        for file_name in files:
            file_path = current_path / file_name

            if (
                file_path.suffix.lower() in UNNEEDED_EXTENSIONS
                or (SYS_NAME == "linux" and file_path.name in PROBLEMATIC_LINUX_LIBS)
                or file_path.name in UNNEEDED_CUDA_LIBS
            ):
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
                except Exception:  # noqa: BLE001, S110
                    pass

    if SYS_NAME == "linux":
        _dedupe_by_hardlink(dist_path)


def _dedupe_by_hardlink(dist_path: Path, min_size: int = 500 * 1024):
    seen_hashes: dict[str, Path] = {}
    for root, _dirs, files in os.walk(dist_path):
        for file_name in files:
            file_path = Path(root) / file_name

            if file_path.is_symlink() or not file_path.is_file():
                continue
            if file_path.stat().st_size <= min_size:
                continue

            try:
                file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            except OSError:
                continue

            original_path = seen_hashes.get(file_hash)
            if original_path is not None:
                try:
                    file_path.unlink()
                    os.link(original_path, file_path)
                except OSError:
                    pass
            else:
                seen_hashes[file_hash] = file_path


def generate_build_info(cpu_only: bool) -> dict:
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
        "cpu_only": cpu_only,
    }


def verify_compiled_output():
    exe_path = OUT_DIST_DIR / get_exe_name()
    bin_path = OUT_DIST_DIR / "bin"

    if not exe_path.exists():
        raise FileNotFoundError(f"Executable not found: {exe_path}")
    if not bin_path.exists():
        raise FileNotFoundError(f"Binary file not found: {bin_path}")


def compile_exe(cpu_only: bool = False):
    DIST_DIR.mkdir(exist_ok=True)
    BUILD_DIR.mkdir(exist_ok=True)

    import piper

    espeak_data = Path(piper.__file__).parent / "espeak-ng-data"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        ENTRY_POINT,
        "--noconfirm",
        "--onedir",
        "--contents-directory=bin",
        f"--name={APP_NAME}",
        f"--distpath={DIST_DIR}",
        f"--workpath={BUILD_DIR}",
        f"--specpath={BUILD_DIR}",
        f"--add-data={espeak_data}{os.pathsep}piper/espeak-ng-data",
        "--collect-all=piper",
        "--collect-all=onnxruntime",
        "--collect-all=textual",
        "--collect-all=rich",
        "--collect-all=llama_cpp",
        "--exclude-module=torch",
        "--exclude-module=torchvision",
        "--exclude-module=torchaudio",
        "--exclude-module=transformers",
    ]

    if cpu_only:
        cmd.extend(
            [
                "--exclude-module=nvidia",
                "--exclude-module=triton",
                "--exclude-module=torch.cuda",
            ]
        )
    subprocess.run(cmd, check=True)

    verify_compiled_output()
    post_build_cleanup(OUT_DIST_DIR)


def _copy_included_dir(name: str, staging_dir: Path):
    src = BASE_DIR / name
    if not src.exists():
        log(f"'{name}/' not found in {src}, skipping.", "WARN")
        return

    patterns = [*DIR_IGNORE_PATTERNS.get(name, []), "__pycache__", "*.pyc"]
    shutil.copytree(
        src,
        staging_dir / name,
        ignore=shutil.ignore_patterns(*patterns),
        dirs_exist_ok=True,
    )


def _copy_included_files(staging_dir: Path):
    for file_name in FILES_TO_INCLUDE:
        file_path = BASE_DIR / file_name
        if not file_path.exists():
            log(f"'{file_name}' not found in {file_path}, skipping.", "WARN")
            continue
        shutil.copy(file_path, staging_dir / file_name)


def make_compressed_archive(basename: str, staging_dir: Path, cpu_only: bool):
    staging_dir.mkdir(exist_ok=True)

    if OUT_DIST_DIR.exists():
        shutil.copytree(OUT_DIST_DIR, staging_dir, dirs_exist_ok=True)
    else:
        raise FileNotFoundError(
            f"{OUT_DIST_DIR} відсутній — компіляцію не було виконано?"
        )

    info_path = staging_dir / "build_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(generate_build_info(cpu_only), f, indent=2)

    _copy_included_files(staging_dir)
    for dir_name in DIRS_TO_INCLUDE:
        _copy_included_dir(dir_name, staging_dir)

    archive_file = BASE_DIR / f"{basename}.tar.xz"
    compressed = False

    if shutil.which("tar") and shutil.which("xz"):
        try:
            subprocess.run(
                [
                    "tar",
                    "-I",
                    "xz -T0",
                    "-cf",
                    str(archive_file),
                    "-C",
                    str(staging_dir.parent),
                    staging_dir.name,
                ],
                check=True,
            )
            compressed = True
        except Exception:  # noqa: BLE001
            log(
                "System tar/xz didn't work, using python's tarfile.",
                "WARN",
            )

    if not compressed:
        with tarfile.open(archive_file, "w:xz") as tf:
            tf.add(staging_dir, arcname=staging_dir.name)

    shutil.rmtree(staging_dir)
    log(f"Archive created: {archive_file}")


if __name__ == "__main__":
    args = parse_args()
    basename = get_basename(args.cpu_only)
    staging_dir = BASE_DIR / basename

    try:
        clean(staging_dir=staging_dir, full=args.full_clean)
        compile_exe(cpu_only=args.cpu_only)
        if not args.no_archive:
            make_compressed_archive(
                basename=basename, staging_dir=staging_dir, cpu_only=args.cpu_only
            )
    except Exception as e:  # noqa: BLE001
        log(f"Build failed: {e}", "ERROR")
        sys.exit(1)
