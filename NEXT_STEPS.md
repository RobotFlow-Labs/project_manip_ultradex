# MANIP-ULTRADEX — Execution Ledger

Resume rule: Read this file COMPLETELY before writing any code.
This project covers exactly ONE paper: UltraDexGrasp: Bimanual Dexterous Grasping.

## 1. Working Rules
- Work only inside `project_manip_ultradex/`
- This wave has 17 parallel projects, 17 papers, 17 agents
- Prefix every commit with `[MANIP-ULTRADEX]`
- Stage only `project_manip_ultradex/` files
- VERIFY THE PAPER BEFORE BUILDING ANYTHING

## 2. The Paper
- **Title**: UltraDexGrasp: Learning Universal Dexterous Grasping for Bimanual Robots with Synthetic Data
- **ArXiv**: 2603.05312
- **Link**: https://arxiv.org/abs/2603.05312
- **Repo**: https://github.com/InternRobotics/UltraDexGrasp
- **Compute**: GPU-NEED
- **Verification status**: Correct paper PDF ✅ | Repo ✅ | Planning PRDs ✅ | PRD-01/02/03/04 scaffolds validated locally ✅

## 3. Current Status
- **Date**: 2026-04-03
- **Phase**: PRD-05 serving hardening next
- **MVP Readiness**: 50%
- **Accomplished**:
  1. Correct UltraDexGrasp paper identified and stored locally
  2. PRD suite, task deck, and asset manifest created
  3. Autopilot Gate 1 complete
  4. Autopilot Gate 2 found training data and weights missing on this Mac
  5. Autopilot Gate 3 infra gaps closed with `anima_module.yaml`, serve Dockerfiles, train/export CLIs, and package rename to `anima_manip_ultradex`
  6. PRD-01 foundation scaffold completed and validated
  7. PRD-02 core model scaffold completed: replay schema, grasp ranking, demo generator, point encoder, action queries, decoder-only transformer, bounded Gaussian head, assembled policy
  8. PRD-03 inference scaffold completed: preprocess, checkpoint runner, postprocess, offline inference CLI, export CLI, FastAPI `/predict` surface
  9. PRD-04 evaluation scaffold completed: manifest-driven sim/real benchmark harness, subgroup metrics, markdown/json reporting, evaluation CLIs
  10. Local validation passed: `uv run ruff check src/ tests/ scripts/`, `uv run pytest -v`, `uv run python scripts/train.py --dry-run`, CLI/export/eval smoke checks
- **TODO**:
  1. Replace random-init policy usage with real checkpoint loading once dataset/weights exist
  2. Materialize dataset generation pipeline against DexGraspNet selected-1000 and UltraDexGrasp-20M replay shards
  3. Wire CUDA server bootstrap and verify BODex / cuRobo / PyTorch3D imports on Linux GPU host
  4. Start PRD-05 serving hardening and PRD-06 ROS2 integration
  5. Add PRD-07 production packaging, model artifact publishing, and deployment hardening
- **Blockers**:
  1. UltraDexGrasp-20M dataset missing locally
  2. DexGraspNet selected-1000 assets missing locally
  3. CUDA-side BODex / cuRobo / PyTorch3D stack unavailable on this Mac

## 4. Datasets
### Required for this paper
| Dataset | Size | URL | Format | Phase Needed |
|---------|------|-----|--------|-------------|
| UltraDexGrasp-20M | 20M frames | internal generation target | replay shards | Train |
| DexGraspNet selected 1000 objects | 1000 objects | DexGraspNet assets | mesh + metadata | Build + Train |
| Sim benchmark 600 | 600 objects | local benchmark manifest | eval set | Eval |
| Real benchmark 25 | 25 objects | local benchmark manifest | eval set | Eval |

### Check shared volume first
/Volumes/AIFlowDev/RobotFlowLabs/datasets

### Download
`bash scripts/download_data.sh`

### Current preflight result
- `/Volumes/AIFlowDev/RobotFlowLabs/datasets/datasets/manip-ultradex/ultradexgrasp_20m` — MISSING
- `/Volumes/AIFlowDev/RobotFlowLabs/datasets/datasets/dexgraspnet/selected_1000` — MISSING
- `/Volumes/AIFlowDev/RobotFlowLabs/datasets/datasets/manip-ultradex/benchmarks/sim_600` — MISSING
- `/Volumes/AIFlowDev/RobotFlowLabs/datasets/datasets/manip-ultradex/benchmarks/real_25` — MISSING

## 5. Hardware
- ZED 2i stereo camera: Available
- Unitree L2 3D LiDAR: Available
- Dual UR5e + dual XHand target embodiment: Planned
- Mac Studio M-series: active local prebuild target
- CUDA server / 8x RTX 6000 Pro Blackwell: post-prebuild training target

## 6. Session Log
| Date | Agent | What Happened |
|------|-------|---------------|
| 2026-04-03 | ANIMA Research Agent | Project scaffolded |
| 2026-04-03 | Codex Autopilot | Gate 0-3 run. Correct paper verified. Data and training weights missing locally. Foundation build started from PRD-01. |
| 2026-04-03 | Codex Build Pass | Completed PRD-01 foundation, PRD-02 core model scaffold, PRD-03 inference scaffold, PRD-04 evaluation scaffold, API predict route, Docker/service cleanup, UV 3.11 lock refresh, and 20 passing tests on macOS. |
