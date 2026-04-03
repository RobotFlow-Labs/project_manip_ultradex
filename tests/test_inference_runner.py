import importlib
import importlib.util

import numpy as np
import pytest

from anima_manip_ultradex.config import load_module_config
from anima_manip_ultradex.inference.preprocess import build_scene_input

torch = importlib.import_module("torch") if importlib.util.find_spec("torch") is not None else None
pytestmark = pytest.mark.skipif(torch is None, reason="torch extra is not installed")
UltraDexRunner = None
UltraDexPolicy = None

if torch is not None:
    UltraDexRunner = importlib.import_module("anima_manip_ultradex.inference.runner").UltraDexRunner
    UltraDexPolicy = importlib.import_module("anima_manip_ultradex.policy.network").UltraDexPolicy


def test_preprocess_builds_2048_point_input() -> None:
    raw_pc = np.random.default_rng(3).normal(size=(3000, 3)).astype(np.float32)
    robot_pc = np.random.default_rng(11).normal(size=(200, 3)).astype(np.float32) * 0.2

    scene_input = build_scene_input(raw_pc, robot_pc=robot_pc, apply_sor=True, target_points=2048)

    assert scene_input.shape == (1, 2048, 3)


def test_runner_loads_checkpoint_and_splits_actions(tmp_path) -> None:
    cfg = load_module_config()
    policy = UltraDexPolicy(cfg, d_model=64, num_heads=4, num_layers=2)
    checkpoint_path = tmp_path / "policy.pt"
    torch.save({"state_dict": policy.state_dict()}, checkpoint_path)

    runner = UltraDexRunner(
        cfg,
        checkpoint_path=checkpoint_path,
        d_model=64,
        num_heads=4,
        num_layers=2,
    )
    raw_pc = np.random.default_rng(5).normal(size=(2500, 3)).astype(np.float32)
    result = runner.predict(raw_pc)

    assert result.checkpoint_loaded is True
    assert result.action_vector.shape == (1, 36)
    assert result.arm_actions.shape == (1, 2, 6)
    assert result.hand_actions.shape == (1, 2, 12)
    assert float(result.action_vector.abs().max()) <= 1.0
