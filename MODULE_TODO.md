# MANIP-ULTRADEX — Design & Implementation Checklist

## Paper: UltraDexGrasp: Learning Universal Dexterous Grasping for Bimanual Robots with Synthetic Data
## ArXiv: 2603.05312
## Repo: https://github.com/InternRobotics/UltraDexGrasp

---

## Phase 1: Scaffold + Verification
- [x] Project structure created
- [x] Correct paper PDF downloaded to papers/
- [x] Paper read and matched against repo
- [x] Reference repo vendored locally
- [ ] Reference demo runs successfully
- [ ] Datasets identified and accessibility confirmed
- [x] PRD.md filled with architecture and plan

## Phase 2: Reproduce
- [ ] Core model implemented in src/anima_manip_ultradex/
- [x] Training pipeline scaffolded in scripts/train.py
- [ ] Evaluation pipeline (scripts/eval.py)
- [ ] Metrics match paper (within ±5%)
- [ ] Dual-compute verified (MLX + CUDA)

## Phase 3: Adapt to Hardware
- [ ] ZED 2i data pipeline (if applicable)
- [ ] Unitree L2 LiDAR pipeline (if applicable)
- [ ] xArm 6 integration (if manipulation module)
- [ ] Real sensor inference test
- [ ] MLX inference port validated

## Phase 4: ANIMA Integration
- [ ] ROS2 bridge node
- [ ] Docker container builds and runs
- [ ] API endpoints defined
- [ ] Integration test with stack: PROMETHEUS

## Shenzhen Demo Readiness
- [ ] Demo script works end-to-end
- [ ] Demo data prepared
- [ ] Demo runs in < 30 seconds
- [ ] Demo visuals are compelling
