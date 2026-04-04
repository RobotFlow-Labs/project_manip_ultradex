# MANIP-ULTRADEX — Implementation PRD

**Status:** Planning complete, implementation not started  
**Version:** 0.2  
**Date:** 2026-04-03  
**Paper:** UltraDexGrasp: Learning Universal Dexterous Grasping for Bimanual Robots with Synthetic Data  
**Paper Link:** https://arxiv.org/abs/2603.05312  
**Repo:** https://github.com/InternRobotics/UltraDexGrasp  
**Functional Name:** MANIP-ultradex  
**Stack:** PROMETHEUS

## Executive Summary

MANIP-ULTRADEX reproduces the UltraDexGrasp paper as an ANIMA manipulation module for universal dexterous grasping with bimanual robots. The paper combines two deliverables:

1. A synthetic data pipeline that generates multi-strategy bimanual grasp demonstrations through optimization-based grasp synthesis plus planning-based rollout.
2. A point-cloud policy that consumes 2,048 scene points, aggregates features with a decoder-only transformer and unidirectional attention, and predicts dual-arm plus dual-hand control commands.

The public repo only exposes the rollout and asset-generation side. The PRD suite below therefore plans both:

- faithful wrapping of the released BODex plus cuRobo rollout stack
- reconstruction of the unpublished training and inference stack from the paper

## Paper Verification Status

- [x] Correct paper identified as `arXiv:2603.05312`
- [x] Correct paper downloaded to `papers/2603.05312_UltraDexGrasp.pdf`
- [x] Public reference repo inspected in `repositories/UltraDexGrasp/`
- [x] Mismatch detected: local `2503.13300` PDF is unrelated and should not drive implementation
- [ ] Third-party dependencies bootstrapped locally
- [ ] Reference rollout executed successfully
- [ ] Training split metadata recreated locally
- [ ] Policy training stack rebuilt and benchmarked

## What We Take From The Paper

- The exact two-stage system decomposition: optimization-based grasp synthesis and planning-based demonstration generation.
- The multi-strategy grasp framing: two-finger pinch, three-finger tripod, whole-hand, and bimanual grasp.
- The policy IO contract: 2,048-point cloud input, PointNet++ style point encoder, decoder-only transformer, action queries, unidirectional attention, and bounded Gaussian action prediction.
- The benchmark targets from Table I, Table II, and Table III.
- The real-world embodiment assumptions: 2x UR5e, 2x XHand, 2x Azure Kinect DK.

## What We Skip

- Reproducing the exact private training code structure used by the authors, because it is not released.
- Treating the vendored public repo as production-ready ANIMA code. It is a research rollout stack with hard-coded CUDA and environment assumptions.
- Blindly inheriting stale local scaffold names such as `RAIJIN` and `anima_raijin`.

## What We Adapt

- We adapt the paper into `src/anima_manip_ultradex/` with ANIMA-compatible configs, tests, API serving, and ROS2 interfaces.
- We keep the paper-faithful policy structure, but isolate unspecified training hyperparameters behind reproducibility configs and ablations.
- We preserve the public BODex and cuRobo integration boundary while making it optional and testable in ANIMA.

## Architecture Summary

- Synthetic pipeline:
  object assets -> BODex grasp synthesis -> candidate filtering -> preferred grasp ranking -> cuRobo motion planning -> simulation rollout -> rendered demonstrations
- Policy:
  point cloud `Tensor[B, 2048, 3]` -> PointNet++ encoder -> scene tokens `Tensor[B, 256, D_scene]` -> decoder-only transformer with 4 action query tokens -> bounded Gaussian action heads -> 36-DoF command vector
- Deployment:
  offline inference runner -> API server -> ROS2 node -> export pipeline

## Build Plan — Executable PRDs

> Total PRDs: 7 | Tasks: 25 | Status: 25/25 complete

| # | PRD | Title | Priority | Tasks | Status |
|---|---|---|---|---|---|
| 1 | [PRD-01](prds/PRD-01-foundation.md) | Foundation & Config | P0 | 4 | ✅ |
| 2 | [PRD-02](prds/PRD-02-core-model.md) | Core Model | P0 | 5 | ✅ |
| 3 | [PRD-03](prds/PRD-03-inference.md) | Inference Pipeline | P0 | 4 | ✅ |
| 4 | [PRD-04](prds/PRD-04-evaluation.md) | Evaluation | P1 | 3 | ✅ |
| 5 | [PRD-05](prds/PRD-05-api-docker.md) | API & Docker | P1 | 3 | ✅ |
| 6 | [PRD-06](prds/PRD-06-ros2-integration.md) | ROS2 Integration | P1 | 3 | ✅ |
| 7 | [PRD-07](prds/PRD-07-production.md) | Production | P2 | 3 | ✅ |

## Build Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Public repo lacks released policy training code | High | Rebuild policy from paper and validate with ablations |
| Heavy dependency chain on BODex, cuRobo, PyTorch3D, SAPIEN, CUDA | High | Keep wrappers thin, gate tests, and isolate optional extras |
| Current scaffold still uses `RAIJIN` naming | Medium | Fix package and config namespace first in PRD-01 |
| Dataset split and checkpoint details are not fully disclosed | Medium | Create reproducibility configs and record all chosen defaults |
| Real hardware differs from paper timing or calibration | Medium | Keep ROS2 and API layers explicit and benchmark sensor preprocessing separately |

## Success Criteria

1. The ANIMA module reproduces the paper pipeline structure and benchmark harnesses.
2. The policy runner accepts `Tensor[B, 2048, 3]` and emits valid dual-arm plus dual-hand actions.
3. Simulation evaluation reaches the target reproduction band documented in `ASSETS.md`.
4. The module can be served through API and ROS2 without paper-specific hard-coding.

## Key Outputs

- [ASSETS.md](ASSETS.md)
- [PIPELINE_MAP.md](PIPELINE_MAP.md)
- [PRD index](prds/README.md)
- [Task index](tasks/INDEX.md)
