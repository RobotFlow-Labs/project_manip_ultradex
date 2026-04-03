# MANIP-ULTRADEX — PRD Suite

This directory contains the executable PRD plan for reproducing UltraDexGrasp inside ANIMA.

## PRDs

1. [PRD-01-foundation.md](PRD-01-foundation.md) — package rename, config, adapters, and testable foundations
2. [PRD-02-core-model.md](PRD-02-core-model.md) — paper-faithful data pipeline and policy reconstruction
3. [PRD-03-inference.md](PRD-03-inference.md) — preprocessing, checkpoint loading, CLI, and export
4. [PRD-04-evaluation.md](PRD-04-evaluation.md) — simulation and real benchmark harnesses
5. [PRD-05-api-docker.md](PRD-05-api-docker.md) — FastAPI serving and containerization
6. [PRD-06-ros2-integration.md](PRD-06-ros2-integration.md) — ROS2 bridge for ANIMA stack composition
7. [PRD-07-production.md](PRD-07-production.md) — release validation, export, reliability, and packaging

## Notes

- The correct paper for this module is `arXiv:2603.05312`, not the stale `2503.13300` reference left in the scaffold.
- The public repo covers rollout and data generation but does not include the released policy training stack.
- All implementation should target `src/anima_manip_ultradex/` even though the current scaffold still contains `anima_raijin`.
