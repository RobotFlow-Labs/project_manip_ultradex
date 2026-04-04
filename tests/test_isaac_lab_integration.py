"""Isaac Lab integration tests — verify adapter, env config, and backend detection."""

import numpy as np

from anima_manip_ultradex.config import load_module_config
from anima_manip_ultradex.sim.scene_env import (
    ISAAC_LAB_ROOT,
    ISAAC_LAB_SOURCE,
    IsaacLabGraspEnv,
    SceneEnvAdapter,
    _has_isaac_lab,
    _isaac_lab_version,
)


def test_isaac_lab_on_disk() -> None:
    assert ISAAC_LAB_ROOT.exists(), "Isaac Lab not found at shared path"
    assert (ISAAC_LAB_SOURCE / "isaaclab" / "isaaclab" / "__init__.py").exists()


def test_isaac_lab_version() -> None:
    version = _isaac_lab_version()
    assert version != "unknown"
    assert "2." in version  # v2.x


def test_isaac_lab_detected() -> None:
    assert _has_isaac_lab() is True


def test_isaac_lab_env_creates_and_resets() -> None:
    cfg = load_module_config()
    env = IsaacLabGraspEnv(cfg, num_envs=4)
    obs = env.reset()
    assert obs["scene_pc"].shape == (4, 2048, 3)
    assert obs["robot_state"].shape == (4, 36)


def test_isaac_lab_env_steps() -> None:
    cfg = load_module_config()
    env = IsaacLabGraspEnv(cfg, num_envs=2)
    env.reset()
    actions = np.zeros((2, 36), dtype=np.float32)
    obs, reward, done, info = env.step(actions)
    assert obs["scene_pc"].shape == (2, 2048, 3)
    assert "backend" in info


def test_isaac_lab_env_config() -> None:
    cfg = load_module_config()
    env = IsaacLabGraspEnv(cfg, num_envs=8)
    env_cfg = env.get_env_config()
    assert env_cfg["env_name"] == "UltraDexGrasp-v0"
    assert env_cfg["num_envs"] == 8
    assert env_cfg["sim_backend"] == "isaac_lab"
    assert "2." in env_cfg["isaac_lab_version"]
    assert env_cfg["robot"]["arms"] == "2x UR5e"
    assert env_cfg["action_space"] == [8, 36]


def test_adapter_prefers_isaac_lab() -> None:
    cfg = load_module_config()
    adapter = SceneEnvAdapter(cfg, sim_backend="auto")
    # Auto-select should prefer isaac_lab since it's on disk
    selected = adapter.select_backend()
    assert selected == "isaac_lab"


def test_adapter_availability_shows_isaac_lab() -> None:
    cfg = load_module_config()
    adapter = SceneEnvAdapter(cfg)
    avail = adapter.availability()
    assert avail["isaac_lab_on_disk"] is True
    assert avail["isaac_lab_version"] != "unknown"


def test_adapter_creates_isaac_lab_env() -> None:
    cfg = load_module_config()
    adapter = SceneEnvAdapter(cfg, sim_backend="isaac_lab")
    env = adapter.create_env(num_envs=2)
    assert isinstance(env, IsaacLabGraspEnv)
    obs = env.reset()
    assert obs["scene_pc"].shape == (2, 2048, 3)


def test_mock_still_works() -> None:
    cfg = load_module_config()
    adapter = SceneEnvAdapter(cfg, sim_backend="mock")
    env = adapter.create_env(num_envs=1)
    obs = env.reset()
    assert obs["scene_pc"].shape == (1, 2048, 3)
    _, _, _, info = env.step(np.zeros((1, 36)))
    assert info["backend"] == "mock"
