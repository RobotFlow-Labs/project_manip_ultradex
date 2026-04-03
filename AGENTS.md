# MANIP-ULTRADEX

## Paper
**UltraDexGrasp: Bimanual Dexterous Grasping**
arXiv: https://arxiv.org/abs/2503.13300

## Module Identity
- Codename: MANIP-ULTRADEX
- Domain: Manipulation
- Part of ANIMA Intelligence Compiler Suite

## Structure
```
project_manip_ultradex/
├── pyproject.toml
├── configs/
├── src/anima_manip_ultradex/
├── tests/
├── scripts/
├── papers/          # Paper PDF
├── AGENTS.md        # This file
├── NEXT_STEPS.md
├── ASSETS.md
└── PRD.md
```

## Commands
```bash
uv sync
uv run pytest
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Conventions
- Package manager: uv (never pip)
- Build backend: hatchling
- Python: >=3.10
- Config: TOML + Pydantic BaseSettings
- Lint: ruff
- Git commit prefix: [MANIP-ULTRADEX]
