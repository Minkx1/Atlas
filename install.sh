#!/bin/bash
set -e

echo -e "\033[1;36m[Atlas] Starting installation...\033[0m"

echo "[Atlas] Creating virtual environment..."
python3 -m venv .venv

echo "[Atlas] Activating environment..."
source .venv/bin/activate
pip install --upgrade pip

echo "[Atlas] Installing required packages..."
pip install --no-cache-dir -r requirements.txt

echo -e "\033[1;32m[Atlas] Installation complete! 🎉\033[0m"
echo "To start the assistant, simply run: source .venv/bin/activate && python main.py"