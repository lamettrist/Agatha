#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv-reconstruction
.venv-reconstruction/bin/pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.venv-reconstruction/bin/pip install -e '.[test,reconstruction]' 'transformers==4.57.6'
.venv-reconstruction/bin/python scripts/prepare_reconstruction.py --warm
# Open3D also needs libEGL.so.1 and libGL.so.1. Install through your system package
# manager, or extract libegl1/libgl1/libglx0/libglvnd0 locally and set LD_LIBRARY_PATH.
