#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# CPU wheels keep setup small and work without a matching CUDA driver.
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[test]'
# ultralytics declares GUI OpenCV; use its headless equivalent on demo servers.
.venv/bin/python -m pip uninstall -y opencv-python opencv-python-headless
.venv/bin/python -m pip install opencv-python-headless
if [ ! -f .env ]; then cp .env.example .env; fi
printf '%s\n' 'Ready. Run: .venv/bin/uvicorn agatha.main:app --host 0.0.0.0 --port 8017'
