# PRD-07: Production

> Module: MANIP-ULTRADEX | Priority: P2  
> Depends on: PRD-04, PRD-05, PRD-06  
> Status: ⬜ Not started

## Objective

Package MANIP-ULTRADEX for reliable production use with export artifacts, release validation, graceful degradation, and ANIMA-ready operational documentation.

## Context (from paper)

The paper ends at research validation. ANIMA needs production-grade export artifacts, failure handling, and release validation to operate the module inside larger robot stacks.

## Acceptance Criteria

- [ ] Export artifacts exist for checkpoint, ONNX, and deployment metadata.
- [ ] Reliability rules cover missing CUDA, missing third-party solvers, and invalid point-cloud inputs.
- [ ] A release checklist links benchmark evidence to exported artifacts.
- [ ] Test: `uv run pytest tests/test_production_release.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_manip_ultradex/release/artifacts.py` | export manifest and artifact checks | — | ~100 |
| `src/anima_manip_ultradex/release/health.py` | readiness and degradation rules | — | ~100 |
| `scripts/package_release.py` | release bundling | — | ~80 |
| `tests/test_production_release.py` | release tests | — | ~80 |

## Architecture Detail (from paper)

### Inputs

- benchmark reports from PRD-04
- serving and ROS2 assets from PRD-05 and PRD-06
- exported model artifacts from PRD-03

### Outputs

- release manifest
- deployment bundle
- operational health rules

### Algorithm

```python
def build_release_bundle(report, artifacts):
    assert report.meets_minimum_targets()
    return write_release_manifest(report, artifacts)
```

## Dependencies

```toml
jsonschema = ">=4.23"
```

## Data Requirements

| Asset | Size | Path | Download |
|---|---|---|---|
| benchmark summary | report-sized | `/mnt/forge-data/models/manip-ultradex/reports/latest.json` | produced locally |
| export bundle | model-sized | `/mnt/forge-data/models/manip-ultradex/releases/latest/` | produced locally |

## Test Plan

```bash
uv run pytest tests/test_production_release.py -v
```

## References

- Depends on: PRD-04, PRD-05, PRD-06
- Feeds into: deployment
