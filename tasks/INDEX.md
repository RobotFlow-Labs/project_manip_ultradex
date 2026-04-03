# MANIP-ULTRADEX — Task Index

## Build Order

### PRD-01 Foundation & Config
- [PRD-0101](PRD-0101.md) Rename scaffold to `anima_manip_ultradex`
- [PRD-0102](PRD-0102.md) Add typed config and asset registry
- [PRD-0103](PRD-0103.md) Wrap BODex and cuRobo reference stack
- [PRD-0104](PRD-0104.md) Add foundation smoke tests

### PRD-02 Core Model
- [PRD-0201](PRD-0201.md) Define replay schema and dataset manifests
- [PRD-0202](PRD-0202.md) Implement grasp selection and demo generation
- [PRD-0203](PRD-0203.md) Implement point-cloud encoder
- [PRD-0204](PRD-0204.md) Implement transformer plus action queries
- [PRD-0205](PRD-0205.md) Implement action head and policy assembly

### PRD-03 Inference Pipeline
- [PRD-0301](PRD-0301.md) Build preprocessing path
- [PRD-0302](PRD-0302.md) Implement checkpoint runner
- [PRD-0303](PRD-0303.md) Add CLI and export pipeline
- [PRD-0304](PRD-0304.md) Add inference integration tests

### PRD-04 Evaluation
- [PRD-0401](PRD-0401.md) Build simulation benchmark harness
- [PRD-0402](PRD-0402.md) Build reporting and ablation tables
- [PRD-0403](PRD-0403.md) Build real-world benchmark harness

### PRD-05 API & Docker
- [PRD-0501](PRD-0501.md) Implement FastAPI schemas and app
- [PRD-0502](PRD-0502.md) Add Docker assets
- [PRD-0503](PRD-0503.md) Add API contract tests and health checks

### PRD-06 ROS2 Integration
- [PRD-0601](PRD-0601.md) Implement ROS2 policy node
- [PRD-0602](PRD-0602.md) Add launch and parameter wiring
- [PRD-0603](PRD-0603.md) Add ROS2 bridge tests

### PRD-07 Production
- [PRD-0701](PRD-0701.md) Implement export manifest and artifact checks
- [PRD-0702](PRD-0702.md) Implement degradation and health rules
- [PRD-0703](PRD-0703.md) Package release bundle and validation tests

## Dependency Notes

- Finish PRD-01 before any model or inference work.
- PRD-02 and PRD-03 can partially overlap once config and wrappers are stable.
- PRD-04 depends on a runnable inference path.
- PRD-05 and PRD-06 both depend on PRD-03.
- PRD-07 depends on evaluation evidence plus serving and ROS2 outputs.
