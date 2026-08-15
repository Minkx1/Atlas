#!/bin/bash
set -e

echo "[Newt] Creating virtual environment..."
python3 -m venv .venv

echo "[Newt] Activating and installing dependencies..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "[Newt] Installation complete! You can now run the assistant."
