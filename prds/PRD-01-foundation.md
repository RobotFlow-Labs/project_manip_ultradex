# PRD-01: Foundation & Config

> Module: MANIP-ULTRADEX | Priority: P0  
> Depends on: None  
> Status: ✅ Scaffold implemented and validated locally

## Objective

Establish a clean MANIP-ULTRADEX foundation with correct package naming, typed configuration, reproducibility metadata, and thin wrappers around the released UltraDexGrasp research code.

## Context (from paper)

The paper’s system depends on a concrete robot embodiment, synthetic scene construction, optimization-based grasp synthesis, and planning-based demonstration generation. Those surfaces must be made explicit before any faithful model or inference work is possible.

Paper references:
- §III: bimanual grasp pose and grasp modeling
- §IV and Fig. 2: scene initialization plus rollout pipeline
- §VI: embodiment assumptions for UR5e, XHand, and Azure Kinect DK

## Acceptance Criteria

- [x] The repo namespace is corrected from `anima_raijin` / `RAIJIN` to `anima_manip_ultradex` / `MANIP-ULTRADEX`.
- [x] A typed config layer captures paper constants, asset paths, backend toggles, and optional third-party dependency switches.
- [x] Wrapper interfaces exist for BODex grasp synthesis, cuRobo planning, and SAPIEN observations without directly hard-coding upstream paths in business logic.
- [x] A fixture loader can resolve the vendored bowl asset and correct paper PDF from local paths.
- [x] Test: `uv run pytest tests/test_foundation_config.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_manip_ultradex/__init__.py` | package root | — | ~10 |
| `src/anima_manip_ultradex/version.py` | version and module metadata | — | ~10 |
| `src/anima_manip_ultradex/config.py` | Pydantic settings and path registry | §IV, §VI | ~120 |
| `src/anima_manip_ultradex/grasp/types.py` | typed grasp, pose, and action schemas | §III | ~100 |
| `src/anima_manip_ultradex/grasp/bodex_adapter.py` | wrapper for public BODex-based synthesizer | §IV.A | ~140 |
| `src/anima_manip_ultradex/planning/curobo_adapter.py` | wrapper for IK and motion planning | §IV.B | ~140 |
| `src/anima_manip_ultradex/sim/scene_env.py` | ANIMA-facing environment adapter | Fig. 2, §IV | ~160 |
| `configs/default.toml` | corrected project config | — | ~40 |
| `tests/test_foundation_config.py` | config and path tests | — | ~80 |
| `tests/test_reference_wrappers.py` | wrapper contract tests | — | ~80 |

## Architecture Detail (from paper)

### Inputs

- `module_root: Path`
- `paper_pdf: Path`
- `asset_root: Path`
- `robot_spec: Dict[str, Any]` for two UR5e arms and two 12-DoF XHands

### Outputs

- `ModuleConfig`
- `GraspCandidateSet`: `Tensor[G, H, 3, 19]`, where `H in {1, 2}`
- `DualArmHandActionSpec`: `Tensor[B, 36]` split into `2x6 arm + 2x12 hand`

### Algorithm

```python
# Paper Sections III and IV — embodiment and rollout foundation

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RobotEmbodiment:
    arm_dof: int = 6
    hand_dof: int = 12
    num_arms: int = 2


class ReferenceStack:
    def __init__(self, cfg: "ModuleConfig") -> None:
        self.cfg = cfg

    def make_grasp_adapter(self) -> "BodexAdapter":
        return BodexAdapter(self.cfg)

    def make_planner(self) -> "CuroboAdapter":
        return CuroboAdapter(self.cfg)
```

## Dependencies

```toml
pydantic = ">=2.7"
pyyaml = ">=6.0"
numpy = ">=1.25"
torch = ">=2.4"
```

## Data Requirements

| Asset | Size | Path | Download |
|---|---|---|---|
| Correct UltraDexGrasp paper PDF | 11.2 MB | `papers/2603.05312_UltraDexGrasp.pdf` | DONE |
| Public rollout repo | vendored | `repositories/UltraDexGrasp/` | DONE |
| BODex configs and solver | repo-sized | `/mnt/forge-data/deps/BODex_api/` | `git clone https://github.com/yangsizhe/BODex_api.git` |
| cuRobo planner | repo-sized | `/mnt/forge-data/deps/curobo/` | `git clone https://github.com/NVlabs/curobo.git` |

## Test Plan

```bash
uv run pytest tests/test_foundation_config.py tests/test_reference_wrappers.py -v
```

## References

- Paper: §III, §IV, §VI
- Reference impl: `repositories/UltraDexGrasp/util/bodex_util.py`
- Reference impl: `repositories/UltraDexGrasp/util/curobo_util.py`
- Reference impl: `repositories/UltraDexGrasp/env/base_env.py`
- Feeds into: PRD-02, PRD-03, PRD-04
