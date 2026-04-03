#!/usr/bin/env bash
set -euo pipefail

# Bootstrap the CUDA-side research stack in a Linux GPU environment.
# This script is intentionally not executed on macOS. It documents the
# exact upstream dependencies needed after the local prebuild phase.

python_version="${PYTHON_VERSION:-3.11}"
workspace="${1:-third_party}"
pytorch_index_url="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

mkdir -p "${workspace}"

echo "[INFO] Create uv environment with Python ${python_version}"
uv venv .venv --python "${python_version}"
source .venv/bin/activate

echo "[INFO] Sync project dependencies and export hooks"
uv sync --extra dev --extra serve --extra train --extra export --extra reference

echo "[INFO] Reinstall CUDA-enabled PyTorch wheels from ${pytorch_index_url}"
uv pip install --reinstall --index-url "${pytorch_index_url}" torch torchvision torchaudio

pushd "${workspace}" >/dev/null

if [ ! -d pytorch3d ]; then
  git clone https://github.com/facebookresearch/pytorch3d.git
fi
if [ ! -d curobo ]; then
  git clone https://github.com/NVlabs/curobo.git
fi
if [ ! -d BODex_api ]; then
  git clone https://github.com/yangsizhe/BODex_api.git
fi

echo "[INFO] Install editable CUDA-side dependencies with uv pip"
uv pip install -e pytorch3d --no-build-isolation
uv pip install -e curobo --no-build-isolation
uv pip install -e BODex_api --no-build-isolation

echo "[INFO] CUDA-side dependency bootstrap finished"

popd >/dev/null
