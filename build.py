import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from src import __version__

sys_name = platform.system().lower()  # 'linux', 'windows', 'darwin'
machine = platform.machine().lower()  # 'x86_64', 'amd64'

if sys_name == "windows":
    machine = "x86_64" if machine == "amd64" else machine

APP_NAME = "newt"
ENTRY_POINT = "main.py"
VERSION = __version__

BASE_DIR = Path(__file__).parent.resolve()
BUILD_DIR = BASE_DIR / "build"
DIST_DIR = BASE_DIR / "dist"

BASENAME = f"{APP_NAME}-{VERSION}-{sys_name}-{machine}"
STAGING_DIR = BASE_DIR / BASENAME
ARCHIVE_NAME = BASE_DIR / BASENAME

FILES_TO_INCLUDE = ["README.md", "LICENSE"]
DIRS_TO_INCLUDE = ["data"]


def clean():
    print("Cleaning previous builds...")
    for p in [BUILD_DIR, DIST_DIR, STAGING_DIR]:
        if p.exists():
            shutil.rmtree(p)
            print(f"  Deleted: {p.name}/")


def compile_exe():
    print(f"[!] Compiling {APP_NAME} with Nuitka...")
    DIST_DIR.mkdir(exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "nuitka",
        "--onefile",
        "--assume-yes-for-downloads",
        f"--output-dir={DIST_DIR}",
        f"--output-filename={APP_NAME}",
        "--remove-output",
        ENTRY_POINT,
    ]

    subprocess.run(cmd, check=True)
    print("[$] Nuitka compilation complete!")


def stage_and_archive():
    print("Making archive...")

    STAGING_DIR.mkdir(exist_ok=True)

    exe_name = f"{APP_NAME}.exe" if os.name == "nt" else APP_NAME
    shutil.copy(DIST_DIR / exe_name, STAGING_DIR / exe_name)
    print(f"  Added executable: {exe_name}")

    for file_name in FILES_TO_INCLUDE:
        file_path = BASE_DIR / file_name
        if file_path.exists():
            shutil.copy(file_path, STAGING_DIR / file_name)
            print(f"  Added file: {file_name}")
        else:
            print(f"  [WARN]: {file_name} not found!")

    for dir_name in DIRS_TO_INCLUDE:
        dir_path = BASE_DIR / dir_name
        if dir_path.exists():
            shutil.copytree(dir_path, STAGING_DIR / dir_name)
            print(f"  Added directory: {dir_name}/")
        else:
            print(f"  [WARN]: Directory {dir_name}/ not found!")

    archive_format = "zip" if os.name == "nt" else "gztar"

    shutil.make_archive(
        str(ARCHIVE_NAME), archive_format, BASE_DIR, STAGING_DIR.name
    )

    shutil.rmtree(STAGING_DIR)

    extension = ".zip" if archive_format == "zip" else ".tar.gz"
    print(f"[$] Archiving complete: {ARCHIVE_NAME.name}{extension}")


if __name__ == "__main__":
    try:
        clean()
        compile_exe()
        stage_and_archive()
    except subprocess.CalledProcessError as e:
        print(f"[!] Compilation error: {e}")
    except Exception as e:
        print(f"[!] Unpredictable error: {e}")