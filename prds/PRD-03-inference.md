# PRD-03: Inference Pipeline

> Module: MANIP-ULTRADEX | Priority: P0  
> Depends on: PRD-01, PRD-02  
> Status: ✅ Scaffold implemented and validated locally

## Objective

Turn the reconstructed UltraDexGrasp policy into a deterministic and testable offline inference pipeline with checkpoint loading, point-cloud preprocessing, CLI entrypoints, and export hooks.

## Context (from paper)

The paper policy is evaluated from point-cloud observations and is deployed zero-shot in the real world. For ANIMA, this requires an explicit preprocessing path, stable checkpoint contracts, and exportable inference surfaces that the public rollout repo does not provide.

Paper references:
- §V.B: point-cloud preprocessing and tokenization
- §V.C: bounded Gaussian action prediction
- §VI.B: real-world point-cloud cleanup and sim-to-real handling

## Acceptance Criteria

- [x] Inference preprocessing reproduces the paper’s point-cloud contract with FPS downsampling to 2,048 points.
- [x] The runner can load a checkpoint and emit `Tensor[B, 36]` action vectors split into arms and hands.
- [x] Real-world preprocessing includes optional SOR filtering and robot imaged point-cloud fusion.
- [x] Export entrypoints exist for Torch checkpoint, ONNX, and MLX-oriented inference conversion.
- [x] Test: `uv run pytest tests/test_inference_runner.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_manip_ultradex/inference/preprocess.py` | crop, merge, SOR, FPS, normalization | §V.B, §VI.B | ~140 |
| `src/anima_manip_ultradex/inference/runner.py` | checkpoint-backed inference entrypoint | §V.C | ~140 |
| `src/anima_manip_ultradex/inference/postprocess.py` | split action vector and apply bounds | §V.C | ~80 |
| `scripts/run_inference.py` | local CLI for offline inference | — | ~80 |
| `scripts/export_policy.py` | Torch/ONNX/MLX export | — | ~120 |
| `tests/test_inference_runner.py` | inference contract tests | — | ~120 |

## Architecture Detail (from paper)

### Inputs

- `raw_pc: Tensor[N_raw, 3]`
- `robot_pc: Tensor[N_robot, 3]`
- `checkpoint_path: Path`

### Outputs

- `model_input: Tensor[1, 2048, 3]`
- `action_vector: Tensor[1, 36]`
- `arm_actions: Tensor[1, 2, 6]`
- `hand_actions: Tensor[1, 2, 12]`

### Algorithm

```python
# Paper Sections V and VI — offline inference contract

class UltraDexRunner:
    def __init__(self, cfg, checkpoint_path):
        self.policy = UltraDexPolicy(cfg)
        self.policy.load_state_dict(load_checkpoint(checkpoint_path))
        self.policy.eval()

    @torch.no_grad()
    def predict(self, raw_pc, robot_pc=None):
        scene_pc = build_scene_input(raw_pc, robot_pc)   # [1, 2048, 3]
        dist = self.policy(scene_pc)
        return split_actions(dist.mode())
```

## Dependencies

```toml
torch = ">=2.4"
onnx = ">=1.16"
onnxruntime = ">=1.18"
numpy = ">=1.25"
```

## Data Requirements

| Asset | Size | Path | Download |
|---|---|---|---|
| Policy checkpoint | TBD | `/mnt/forge-data/models/manip-ultradex/policy/latest.ckpt` | produced locally |
| Real benchmark point clouds | 25-object eval | `/mnt/forge-data/datasets/manip-ultradex/benchmarks/real_25/pc/` | captured locally |

## Test Plan

```bash
uv run pytest tests/test_inference_runner.py -v
uv run python scripts/run_inference.py --help
```

## References

- Paper: §V.B, §V.C, §VI.B
- Depends on: PRD-01, PRD-02
- Feeds into: PRD-04, PRD-05, PRD-06, PRD-07
