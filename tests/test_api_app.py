import importlib.util

import numpy as np
from fastapi.testclient import TestClient

from anima_manip_ultradex.api.app import app


def test_api_health_and_ready_contracts() -> None:
    client = TestClient(app)

    health = client.get("/healthz")
    ready = client.get("/readyz")

    assert health.status_code == 200
    assert health.json()["ok"] is True
    assert ready.status_code == 200
    assert "ready" in ready.json()


def test_predict_endpoint_contract() -> None:
    client = TestClient(app)
    payload = {
        "raw_points": np.random.default_rng(13).normal(size=(2500, 3)).astype(np.float32).tolist(),
        "robot_points": np.random.default_rng(23).normal(size=(128, 3)).astype(np.float32).tolist(),
        "apply_sor": True,
    }

    response = client.post("/predict", json=payload)

    if importlib.util.find_spec("torch") is None:
        assert response.status_code == 503
        return

    assert response.status_code == 200
    body = response.json()
    assert len(body["action_vector"]) == 1
    assert len(body["action_vector"][0]) == 36
    assert len(body["arm_actions"][0]) == 2
    assert len(body["hand_actions"][0]) == 2
