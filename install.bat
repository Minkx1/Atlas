@echo off
setlocal

echo [Newt] Starting installation...

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [!] Python is not installed or not added to PATH!
    pause
    exit /b 1
)

echo [Newt] Creating virtual environment...
python -m venv .venv

echo [Newt] Activating environment...
call .venv\Scripts\activate.bat

echo [Newt] Updating pip...
python -m pip install --upgrade pip


echo [Newt] Installing required packages...
pip install --no-cache-dir -r requirements.txt

echo.
echo [Newt] Installation complete!
echo To start the assistant, run: call .venv\Scripts\activate.bat ^&^& python main.py
pause