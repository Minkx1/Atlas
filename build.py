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
OUT_DIST_DIR = DIST_DIR / APP_NAME

BASENAME = f"{APP_NAME}-{VERSION}-{SYS_NAME}-{MACHINE}"
STAGING_DIR = BASE_DIR / BASENAME

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

PROBLEMATIC_LINUX_LIBS = {"libstdc++.so.6", "libgcc_s.so.1", "libm.so.6"}


def parse_args():
    parser = argparse.ArgumentParser(description="Newt Build Script")
    parser.add_argument("--full-clean", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    return parser.parse_args()


def clean(full: bool = False):
    if full:
        for d in [DIST_DIR, BUILD_DIR]:
            if d.exists():
                shutil.rmtree(d)
    elif OUT_DIST_DIR.exists():
        shutil.rmtree(OUT_DIST_DIR)

    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)


def post_build_cleanup(dist_path: Path):
    for root, _, files in os.walk(dist_path, topdown=False):
        current_path = Path(root)

        if any(bad_dir in str(current_path) for bad_dir in UNNEEDED_DIR_NAMES):
            try:
                shutil.rmtree(current_path)
            except OSError:
                pass
            continue

        for file_name in files:
            file_path = current_path / file_name

            if file_path.suffix.lower() in UNNEEDED_EXTENSIONS:
                try:
                    file_path.unlink()
                except OSError:
                    pass
            elif SYS_NAME == "linux" and file_path.name in PROBLEMATIC_LINUX_LIBS:
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
                except Exception:
                    pass


def generate_build_info():
    git_hash = "unknown"
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        pass

    return {
        "version": VERSION,
        "commit": git_hash,
        "build_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": f"{SYS_NAME}-{MACHINE}",
        "python_version": platform.python_version(),
    }


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
        "--collect-data=silero_vad",
        f"--add-data={espeak_data}{os.pathsep}piper/espeak-ng-data",
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
        cmd.extend(
            [
                "--exclude-module=nvidia",
                "--exclude-module=triton",
                "--exclude-module=torch.cuda",
            ]
        )
    else:
        cmd.extend(["--collect-all=nvidia", "--collect-all=torch"])

    subprocess.run(cmd, check=True)

    if OUT_DIST_DIR.exists():
        post_build_cleanup(OUT_DIST_DIR)


def make_compressed_archive():
    STAGING_DIR.mkdir(exist_ok=True)

    if OUT_DIST_DIR.exists():
        shutil.copytree(OUT_DIST_DIR, STAGING_DIR, dirs_exist_ok=True)

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

    archive_file = BASE_DIR / f"{BASENAME}.tar.xz"
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
                    str(STAGING_DIR.parent),
                    STAGING_DIR.name,
                ],
                check=True,
            )
            compressed = True
        except Exception:
            pass

    if not compressed:
        with tarfile.open(archive_file, "w:xz") as tf:
            tf.add(STAGING_DIR, arcname=STAGING_DIR.name)

    shutil.rmtree(STAGING_DIR)


if __name__ == "__main__":
    args = parse_args()
    try:
        clean(full=args.full_clean)
        compile_exe(cpu_only=args.cpu_only)
        if not args.no_archive:
            make_compressed_archive()
    except Exception as e:
        sys.exit(1)
