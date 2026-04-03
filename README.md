# MANIP-ULTRADEX — ANIMA Module

> **UltraDexGrasp: Learning Universal Dexterous Grasping for Bimanual Robots with Synthetic Data**
> Paper: [arXiv:2603.05312](https://arxiv.org/abs/2603.05312)

Part of the [ANIMA Intelligence Compiler Suite](https://github.com/RobotFlow-Labs) by AIFLOW LABS LIMITED.

## Domain
Manipulation

## Status
- [x] Paper aligned and `ASSETS.md` created
- [x] PRD-01 foundation scaffold
- [x] PRD-02 core model scaffold
- [x] PRD-03 inference pipeline scaffold
- [x] PRD-04 evaluation scaffold
- [ ] PRD-05 through PRD-07
- [x] Training and export entrypoint scaffolds
- [x] API predict surface and Docker serving scaffold
- [ ] GPU training
- [ ] Export: pth + safetensors + ONNX + TRT fp16 + TRT fp32
- [ ] Push to HuggingFace

## Quick Start
```bash
cd project_manip_ultradex
uv venv .venv --python 3.11
uv sync --extra dev --extra serve --extra mac --extra train
uv run pytest tests/ -v
uv run python scripts/train.py --dry-run
uv run python scripts/run_inference.py --use-sor
```

## Dependency Strategy

- macOS prebuild and model scaffold: `uv sync --extra dev --extra serve --extra mac --extra train`
- export extras when needed: `uv sync --extra dev --extra serve --extra mac --extra train --extra export`
- CUDA-side training build later: `uv sync --extra dev --extra serve --extra train --extra export --extra reference`
- CUDA-only third-party research stack bootstrap: `bash scripts/bootstrap_cuda_deps.sh`

## License
MIT — AIFLOW LABS LIMITED
