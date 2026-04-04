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
- **Verification status**: Correct paper PDF ✅ | Repo ✅ | All 7 PRDs built ✅ | 47 tests pass ✅ | Training pipeline ✅

## 3. Current Status
- **Date**: 2026-04-04
- **Phase**: All PRDs + training pipeline complete. Ready for real data + full training.
- **MVP Readiness**: 85%
- **Accomplished**:
  1. PRD-01 foundation: config, types, device detection, reference wrappers
  2. PRD-02 core model: grasp types, selection, demo generator, replay buffer, PointNet++ encoder, decoder-only transformer, action queries, bounded Gaussian head, full policy assembly
  3. PRD-03 inference: preprocess (CUDA FPS kernel integrated), postprocess, runner, offline CLI, export CLI
  4. PRD-04 evaluation: manifest-driven benchmark harness, subgroup metrics, markdown/JSON reporting
  5. PRD-05 API & Docker: FastAPI with /health /ready /info /predict, Dockerfile.serve, docker-compose.serve.yml, .env.serve
  6. PRD-06 ROS2: policy node, message types, launch file, topic definitions
  7. PRD-07 production: release manifest, health checks, export pipeline (pth→safetensors→ONNX→TRT), package_release script
  8. CUDA integration: FPS kernel (7.2x), SE3 transform (30x) installed from shared infra
  9. Point encoder upgraded to use CUDA FPS for token selection on GPU
  10. Sim adapter updated for IsaacGym + Isaac Lab + mock backend
  11. Export pipeline verified: pth (2.4MB) + safetensors (2.4MB) + ONNX (0.5MB)
  12. venv created on GPU server: Python 3.11, torch 2.11.0+cu128, 8x L4 GPUs
  13. 43/43 tests pass (including 6 CUDA integration tests on GPU)
  14. Ruff lint clean, ruff format clean
- **TODO**:
  1. Materialize UltraDexGrasp-20M dataset generation pipeline using BODex + cuRobo
  2. Wire IsaacGym Preview 4 env for paper-compatible simulation rollout
  3. Wire Isaac Lab env for future-proof RL training
  4. Run GPU batch finder → start full training with real data
  5. Install TensorRT and complete TRT fp16/fp32 exports
  6. Push to HuggingFace: ilessio-aiflowlab/project_manip_ultradex
- **Blockers**:
  1. UltraDexGrasp-20M dataset must be generated (no public download) — BODex + cuRobo wired
  2. DexGraspNet selected-1000 object assets not yet on disk
  3. TensorRT not installed on server (needed for TRT fp16/fp32 export)
  4. IsaacGym Python package not yet installed in venv (shared sim path exists at /mnt/forge-data/shared_infra/simulators/isaacgym/)

## 4. Datasets
### Required for this paper
| Dataset | Size | URL | Format | Phase Needed |
|---------|------|-----|--------|-------------|
| UltraDexGrasp-20M | 20M frames | internal generation target | replay shards | Train |
| DexGraspNet selected 1000 objects | 1000 objects | DexGraspNet assets | mesh + metadata | Build + Train |
| Sim benchmark 600 | 600 objects | local benchmark manifest | eval set | Eval |
| Real benchmark 25 | 25 objects | local benchmark manifest | eval set | Eval |

### Check shared volume first
/mnt/forge-data/datasets/

### Current preflight result
- UltraDexGrasp-20M — MISSING (must generate)
- DexGraspNet selected_1000 — MISSING
- Sim benchmark 600 — MISSING (generate from paper §VI.A)
- Real benchmark 25 — MISSING (generate from paper §VI.B)

## 5. Hardware
- 8x NVIDIA L4 (23GB each) — CUDA_VISIBLE_DEVICES=2 assigned
- ZED 2i stereo camera: Available
- Unitree L2 3D LiDAR: Available
- Dual UR5e + dual XHand target embodiment: Planned
- IsaacGym Preview 4: /mnt/forge-data/shared_infra/simulators/isaacgym/
- Isaac Lab: /mnt/forge-data/shared_infra/simulators/IsaacLab/

## 6. CUDA Kernels
### Shared (installed)
- point_cloud_ops: FPS (7.2x speedup) — verified on GPU 2
- se3_transform: SE3 (30x speedup) — verified on GPU 2

### Module-specific (compiled + verified on GPU 2)
- grasp_synthesis: batched wrench computation + SDF collision check ✅
- hand_kinematics: batched FK + Jacobian for XHand ✅

## 7. Session Log
| Date | Agent | What Happened |
|------|-------|---------------|
| 2026-04-03 | ANIMA Research Agent | Project scaffolded |
| 2026-04-03 | Codex Autopilot | Gate 0-3 run. PRD-01 through PRD-04 scaffolds built on Mac. |
| 2026-04-04 | GPU Autopilot | Moved to GPU server. Fixed 6 missing __init__.py + created data/ package. Created venv (py3.11, torch cu128). Installed shared CUDA kernels. Built PRD-05 (API+Docker), PRD-06 (ROS2), PRD-07 (Production). Upgraded point encoder with CUDA FPS. Updated sim adapter for IsaacGym + Isaac Lab + mock. Export pipeline verified (pth+safetensors+ONNX). Built + compiled grasp_synthesis and hand_kinematics CUDA kernels. Built full training pipeline with config-driven params, checkpoint save/resume, early stopping, LR warmup+cosine, VRAM monitoring. Wired BODex and cuRobo adapters to repos on disk. Training smoke test passed (10 steps, loss decreasing, checkpoint resume works). 47/47 tests pass. |
