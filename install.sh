#!/bin/bash
set -e

echo -e "\033[1;36m[Newt] Starting installation...\033[0m"

echo "[Newt] Creating virtual environment..."
python3 -m venv .venv

echo "[Newt] Activating environment..."
source .venv/bin/activate
pip install --upgrade pip

echo "[Newt] Installing required packages..."
pip install --no-cache-dir -r requirements.txt

echo -e "\033[1;32m[Newt] Installation complete! 🎉\033[0m"
echo "To start the assistant, simply run: source .venv/bin/activate && python main.py"