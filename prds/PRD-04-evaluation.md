# PRD-04: Evaluation

> Module: MANIP-ULTRADEX | Priority: P1  
> Depends on: PRD-02, PRD-03  
> Status: ✅ Scaffold implemented and validated locally

## Objective

Recreate the paper’s simulation and real-world evaluation flows so MANIP-ULTRADEX can measure reproduction quality against the published success-rate tables.

## Context (from paper)

The paper reports success rates on a 600-object simulation benchmark and a 25-object real-world benchmark, plus policy ablations. Those numbers are the only quantitative targets available because the public repo does not ship model checkpoints.

Paper references:
- §VI.A and Table I: simulation benchmark
- Table II: ablations
- §VI.B and Table III: real-world benchmark

## Acceptance Criteria

- [x] A simulation benchmark harness reproduces seen/unseen and small/medium/large reporting groups.
- [x] A real-world benchmark harness supports 25-object logging with per-trial outcomes.
- [x] Report generation can emit markdown and JSON summaries that compare local metrics to paper targets.
- [x] Test: `uv run pytest tests/test_evaluation_metrics.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_manip_ultradex/eval/metrics.py` | success-rate aggregation and subgroup reporting | Table I–III | ~100 |
| `src/anima_manip_ultradex/eval/benchmark.py` | benchmark driver abstractions | §VI | ~120 |
| `scripts/evaluate_sim.py` | 600-object simulation evaluation | §VI.A | ~120 |
| `scripts/evaluate_real.py` | 25-object real-world evaluation | §VI.B | ~120 |
| `tests/test_evaluation_metrics.py` | metrics contract tests | — | ~80 |

## Architecture Detail (from paper)

### Inputs

- `eval_manifest.json`
- `predicted_actions` or rollout traces
- `object_metadata` with seen/unseen and size group tags

### Outputs

- `SimulationReport`
- `RealWorldReport`
- `AblationTable`

### Algorithm

```python
def compute_success_rate(records):
    grouped = group_by(records, keys=["split", "size_group"])
    return {key: mean(item["success"] for item in rows) for key, rows in grouped.items()}
```

## Dependencies

```toml
numpy = ">=1.25"
pandas = ">=2.2"
```

## Data Requirements

| Asset | Size | Path | Download |
|---|---|---|---|
| Simulation benchmark manifest | 600 objects | `/mnt/forge-data/datasets/manip-ultradex/benchmarks/sim_600/manifest.json` | curated locally |
| Real benchmark manifest | 25 objects | `/mnt/forge-data/datasets/manip-ultradex/benchmarks/real_25/manifest.json` | curated locally |

## Test Plan

```bash
uv run pytest tests/test_evaluation_metrics.py -v
uv run python scripts/evaluate_sim.py --help
uv run python scripts/evaluate_real.py --help
```

## References

- Paper: §VI.A, §VI.B, Table I, Table II, Table III
- Depends on: PRD-02, PRD-03
- Feeds into: PRD-07
