import os
import platform
import shutil
import subprocess
from pathlib import Path

from src import __version__

sys_name = platform.system().lower() # 'linux', 'windows', 'darwin'
machine = platform.machine().lower() # 'x86_64', 'amd64'

if sys_name == "windows":
    machine = "x86_64" if machine == "amd64" else machine # Нормалізація для Win

APP_NAME = "newt"  
ENTRY_POINT = "main.py"      
VERSION = __version__

BASE_DIR = Path(__file__).parent.resolve()
BUILD_DIR = BASE_DIR / "build"
DIST_DIR = BASE_DIR / "dist"

BASENAME = f"{APP_NAME}-{VERSION}-{sys_name}-{machine}"
STAGING_DIR = BASE_DIR / BASENAME
ARCHIVE_NAME = BASE_DIR / BASENAME

# Файли та папки, які треба додати в архів (окрім екзешника)
FILES_TO_INCLUDE = ["README.md", "LICENSE"]
DIRS_TO_INCLUDE = ["data"]

def clean():
    print("Cleaning previous builds...")
    for p in [BUILD_DIR, DIST_DIR, STAGING_DIR]:
        if p.exists():
            shutil.rmtree(p)
            print(f"  Deleted: {p.name}/")

def compile_exe():
    print(f"[!] Building {APP_NAME}...")
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--onefile",
        "--name", APP_NAME,
        ENTRY_POINT
    ]
    
    subprocess.run(cmd, check=True)
    print("[$] Building complete!")

def stage_and_archive():
    print("Making archive...")
    
    STAGING_DIR.mkdir(exist_ok=True)
    
    exe_name = f"{APP_NAME}.exe" if os.name == 'nt' else APP_NAME
    shutil.copy(DIST_DIR / exe_name, STAGING_DIR / exe_name)
    print(f"  Added: {exe_name}")

    for file_name in FILES_TO_INCLUDE:
        file_path = BASE_DIR / file_name
        if file_path.exists():
            shutil.copy(file_path, STAGING_DIR / file_name)
            print(f"  Added: {file_name}")
        else:
            print(f"  [WARN]: {file_name} not found!")

    for dir_name in DIRS_TO_INCLUDE:
        dir_path = BASE_DIR / dir_name
        if dir_path.exists():
            shutil.copytree(dir_path, STAGING_DIR / dir_name)
            print(f"  Added: {dir_name}/")
        else:
            print(f"  [WARN]: Directory {dir_name}/ not found!")


    archive_format = "zip" if os.name == 'nt' else "gztar"
    shutil.make_archive(ARCHIVE_NAME.name, archive_format, BASE_DIR, STAGING_DIR.name)
    
    shutil.rmtree(STAGING_DIR)
    
    extension = ".zip" if archive_format == "zip" else ".tar.gz"
    print(f"[$] Archiving complete: {ARCHIVE_NAME.name}{extension}")

if __name__ == "__main__":
    try:
        clean()
        compile_exe()
        stage_and_archive()
    except subprocess.CalledProcessError as e:
        print(f"[!] Compliation error: {e}")
    except Exception as e:
        print(f"[!] Unpredictable error: {e}")