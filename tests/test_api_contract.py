"""PRD-05 tests: API contract verification."""

import importlib.util

import numpy as np
from fastapi.testclient import TestClient

from anima_manip_ultradex.api.app import app

client = TestClient(app)


def test_health_endpoint() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["module"] == "MANIP-ULTRADEX"
    assert "backend" in body
    assert "cuda_available" in body


def test_healthz_alias() -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_ready_endpoint() -> None:
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert "ready" in body
    assert "checkpoint_loaded" in body


def test_readyz_alias() -> None:
    resp = client.get("/readyz")
    assert resp.status_code == 200


def test_info_endpoint() -> None:
    resp = client.get("/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["module"] == "MANIP-ULTRADEX"
    assert body["action_dims"] == 36
    assert body["input_points"] == 2048
    assert "arXiv" in body["paper"]


def test_predict_returns_36dof_actions() -> None:
    if importlib.util.find_spec("torch") is None:
        return
    payload = {
        "raw_points": np.random.default_rng(42).normal(size=(2500, 3)).astype(np.float32).tolist(),
        "apply_sor": False,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["action_vector"][0]) == 36
    assert len(body["arm_actions"][0]) == 2
    assert len(body["hand_actions"][0]) == 2


def test_predict_with_robot_points() -> None:
    if importlib.util.find_spec("torch") is None:
        return
    rng = np.random.default_rng(7)
    payload = {
        "raw_points": rng.normal(size=(2000, 3)).astype(np.float32).tolist(),
        "robot_points": (rng.normal(size=(200, 3)) * 0.1).astype(np.float32).tolist(),
        "apply_sor": True,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
