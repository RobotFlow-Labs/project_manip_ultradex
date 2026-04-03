# PRD-02: Core Model

> Module: MANIP-ULTRADEX | Priority: P0  
> Depends on: PRD-01  
> Status: ✅ Scaffold implemented and validated locally

## Objective

Implement the paper-faithful UltraDexGrasp core: the synthetic demonstration pipeline plus the universal dexterous grasp policy that maps point clouds to dual-arm and dual-hand control commands.

## Context (from paper)

UltraDexGrasp couples two components: a data-generation pipeline and a control policy. The data pipeline integrates optimization-based grasp synthesis with planning-based demonstration generation, and the policy consumes point clouds, aggregates scene features using unidirectional attention, and predicts bounded stochastic actions.

Paper references:
- §IV.A: grasp synthesis objective and preferred-grasp selection
- §IV.B: pregrasp, grasp, squeeze, and lift demonstration stages
- §V.A–§V.C and Fig. 4: point-cloud policy architecture

## Acceptance Criteria

- [x] The grasp-synthesis wrapper exposes candidate grasps and preferred-grasp selection for all four paper strategies.
- [x] The demonstration generator reproduces the four rollout stages and success criteria from the paper.
- [x] The point encoder accepts `Tensor[B, 2048, 3]` and emits scene tokens `Tensor[B, 256, D_scene]`.
- [x] The transformer backbone uses learnable action queries with unidirectional attention to scene tokens.
- [x] The action head predicts a bounded Gaussian distribution and computes negative log-likelihood loss.
- [x] Test: `uv run pytest tests/test_grasp_pipeline.py tests/test_policy_network.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_manip_ultradex/grasp/object_model.py` | object asset and grasp-strategy helpers | §IV.A | ~100 |
| `src/anima_manip_ultradex/grasp/selection.py` | candidate filtering and SE(3) ranking | §IV.A | ~120 |
| `src/anima_manip_ultradex/data/demo_generator.py` | four-stage demonstration generator | §IV.B | ~180 |
| `src/anima_manip_ultradex/data/replay_buffer.py` | dataset writer and reader for 20M-frame schema | §IV.B | ~140 |
| `src/anima_manip_ultradex/policy/point_encoder.py` | PointNet++ style encoder | §V.B | ~160 |
| `src/anima_manip_ultradex/policy/action_queries.py` | learnable dual-arm and dual-hand action queries | §V.C | ~80 |
| `src/anima_manip_ultradex/policy/transformer.py` | decoder-only transformer with unidirectional attention | §V.A–§V.C | ~160 |
| `src/anima_manip_ultradex/policy/action_head.py` | bounded Gaussian / truncated-normal action head | §V.C | ~120 |
| `src/anima_manip_ultradex/policy/network.py` | end-to-end policy assembly | Fig. 4 | ~140 |
| `tests/test_grasp_pipeline.py` | trajectory and candidate-shape tests | §IV | ~100 |
| `tests/test_policy_network.py` | forward-pass and loss tests | §V | ~120 |

## Architecture Detail (from paper)

### Inputs

- `scene_pc: Tensor[B, 2048, 3]`
- `grasp_strategy: Tensor[B]` with values in `{pinch, tripod, whole_hand, bimanual}`
- `candidate_grasps: Tensor[G, H, 3, 19]`

### Outputs

- `scene_tokens: Tensor[B, 256, D_scene]`
- `action_queries: Tensor[B, 4, D_model]`
- `arm_actions: Tensor[B, 2, 6]`
- `hand_actions: Tensor[B, 2, 12]`
- `policy_loss: Tensor[]`

### Algorithm

```python
# Paper Sections IV and V — core pipeline reconstruction

class UltraDexPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = PointEncoder(cfg)
        self.queries = ActionQueryBank(cfg)
        self.backbone = DecoderOnlyTransformer(cfg)
        self.head = BoundedGaussianActionHead(cfg)

    def forward(self, scene_pc, action_targets=None):
        scene_tokens = self.encoder(scene_pc)            # [B, 256, D_scene]
        query_tokens = self.queries(scene_pc.shape[0])   # [B, 4, D_model]
        fused_tokens = self.backbone(query_tokens, scene_tokens)
        return self.head(fused_tokens, action_targets)
```

## Dependencies

```toml
torch = ">=2.4"
numpy = ">=1.25"
scipy = ">=1.14"
trimesh = ">=4.0"
h5py = ">=3.14"
```

## Data Requirements

| Asset | Size | Path | Download |
|---|---|---|---|
| UltraDexGrasp-20M replay shards | 20M frames | `/mnt/forge-data/datasets/manip-ultradex/ultradexgrasp_20m/` | Generate with local pipeline |
| Bowl smoke-test object | 1 object | `repositories/UltraDexGrasp/asset/object_mesh/bowl/` | DONE |
| Strategy metadata | 4 strategies | `configs/strategies.toml` | local file |

## Test Plan

```bash
uv run pytest tests/test_grasp_pipeline.py tests/test_policy_network.py -v
```

## References

- Paper: §IV.A, §IV.B, §V.A, §V.B, §V.C, Fig. 2, Fig. 4
- Reference impl: `repositories/UltraDexGrasp/rollout.py`
- Reference impl: `repositories/UltraDexGrasp/util/bodex_util.py`
- Depends on: PRD-01
- Feeds into: PRD-03, PRD-04
