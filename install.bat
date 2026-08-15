@echo off
echo [Newt] Creating virtual environment...
python -m venv .venv

echo [Newt] Activating and installing dependencies...
call .venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

echo [Newt] Installation complete! You can now run the assistant.
pause
