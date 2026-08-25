@echo off
setlocal

cd /d "%~dp0\.."

echo [Atlas] Starting installation...

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [!] Python is not installed or not added to PATH!
    pause
    exit /b 1
)

echo [Atlas] Creating virtual environment...
python -m venv .venv

echo [Atlas] Activating environment...
call .venv\Scripts\activate.bat

echo [Atlas] Updating pip...
python -m pip install --upgrade pip


echo [Atlas] Installing required packages...
pip install . --no-cache-dir

echo.
echo [Atlas] Installation complete!
echo To start the assistant, run: call .venv\Scripts\activate.bat ^&^& python main.py
pause