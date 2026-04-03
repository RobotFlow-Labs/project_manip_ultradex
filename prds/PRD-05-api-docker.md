# PRD-05: API & Docker

> Module: MANIP-ULTRADEX | Priority: P1  
> Depends on: PRD-03  
> Status: ⬜ Not started

## Objective

Expose UltraDexGrasp inference as a containerized API suitable for ANIMA composition and remote execution.

## Context (from paper)

The paper validates real-world inference but does not specify a serving layer. ANIMA needs a stable API boundary for point-cloud input, inference output, health checks, and deployment packaging.

## Acceptance Criteria

- [ ] A FastAPI app accepts point-cloud inputs and returns dual-arm plus dual-hand actions.
- [ ] GPU-aware Docker assets exist for local and server deployment.
- [ ] Health and readiness endpoints validate checkpoint loading and dependency presence.
- [ ] Test: `uv run pytest tests/test_api_contract.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_manip_ultradex/api/schemas.py` | request and response models | — | ~80 |
| `src/anima_manip_ultradex/api/app.py` | FastAPI service | — | ~120 |
| `docker/Dockerfile.api` | serving container | — | ~60 |
| `docker/docker-compose.api.yml` | local stack orchestration | — | ~40 |
| `tests/test_api_contract.py` | API tests | — | ~80 |

## Architecture Detail (from paper)

### Inputs

- `ScenePointCloudRequest` with `points: list[list[float]]`
- optional `robot_points`
- optional `grasp_strategy_hint`

### Outputs

- `DualArmHandActionResponse`
- `model_metadata`

### Algorithm

```python
@app.post("/predict")
def predict(req: ScenePointCloudRequest) -> DualArmHandActionResponse:
    action = runner.predict(req.points, req.robot_points)
    return encode_response(action)
```

## Dependencies

```toml
fastapi = ">=0.115"
uvicorn = ">=0.30"
```

## Data Requirements

| Asset | Size | Path | Download |
|---|---|---|---|
| Serving checkpoint | TBD | `/mnt/forge-data/models/manip-ultradex/policy/latest.ckpt` | produced locally |

## Test Plan

```bash
uv run pytest tests/test_api_contract.py -v
uv run uvicorn anima_manip_ultradex.api.app:app --host 127.0.0.1 --port 8000
```

## References

- Depends on: PRD-03
- Feeds into: PRD-06, PRD-07
