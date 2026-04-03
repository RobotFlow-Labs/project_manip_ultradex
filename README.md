# MANIP-ULTRADEX — ANIMA Module

> **UltraDexGrasp: Bimanual Dexterous Grasping**
> Paper: [arXiv:2503.13300](https://arxiv.org/abs/2503.13300)

Part of the [ANIMA Intelligence Compiler Suite](https://github.com/RobotFlow-Labs) by AIFLOW LABS LIMITED.

## Domain
Manipulation

## Status
- [ ] Paper read + ASSETS.md created
- [ ] PRD-01 through PRD-07
- [ ] Training pipeline
- [ ] GPU training
- [ ] Export: pth + safetensors + ONNX + TRT fp16 + TRT fp32
- [ ] Push to HuggingFace
- [ ] Docker serving

## Quick Start
```bash
cd project_manip_ultradex
uv venv .venv --python python3.11 && uv sync
uv run pytest tests/ -v
```

## License
MIT — AIFLOW LABS LIMITED
