"""CUDA integration tests — run on GPU, verify CUDA kernels + policy."""

import importlib.util

import numpy as np
import pytest

torch = None
if importlib.util.find_spec("torch"):
    import torch as _torch

    torch = _torch

pytestmark = pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(),
    reason="CUDA not available",
)


def test_fps_cuda_kernel() -> None:
    from point_cloud_ops import farthest_point_sample

    pts = torch.randn(4, 4096, 3, device="cuda")
    idx = farthest_point_sample(pts, 2048)
    assert idx.shape == (4, 2048)
    assert idx.device.type == "cuda"
    # Indices should be unique per batch
    for b in range(4):
        assert len(set(idx[b].tolist())) == 2048


def test_se3_transform_cuda_kernel() -> None:
    from se3_transform import se3_transform

    pts = torch.randn(2, 1024, 3, device="cuda")
    T = torch.eye(4, device="cuda").unsqueeze(0).expand(2, -1, -1).contiguous()
    out = se3_transform(pts, T)
    assert out.shape == pts.shape
    assert (out - pts).abs().max().item() < 1e-5


def test_point_encoder_uses_fps_on_gpu() -> None:
    from anima_manip_ultradex.config import load_module_config
    from anima_manip_ultradex.policy.point_encoder import PointEncoder

    cfg = load_module_config()
    encoder = PointEncoder(cfg, d_scene=64).cuda()
    pts = torch.randn(2, 2048, 3, device="cuda")
    tokens = encoder(pts)
    assert tokens.shape == (2, 256, 64)
    assert tokens.device.type == "cuda"


def test_full_policy_forward_on_gpu() -> None:
    from anima_manip_ultradex.config import load_module_config
    from anima_manip_ultradex.policy.network import UltraDexPolicy

    cfg = load_module_config()
    policy = UltraDexPolicy(cfg, d_model=64, num_heads=4, num_layers=2).cuda()
    pts = torch.randn(4, 2048, 3, device="cuda")
    targets = torch.randn(4, 36, device="cuda")
    out = policy(pts, action_targets=targets)
    assert out.arm_actions.shape == (4, 2, 6)
    assert out.hand_actions.shape == (4, 2, 12)
    assert out.nll_loss is not None
    assert out.nll_loss.device.type == "cuda"


def test_preprocess_with_cuda_fps() -> None:
    from anima_manip_ultradex.inference.preprocess import build_scene_input

    raw = np.random.default_rng(1).normal(size=(5000, 3)).astype(np.float32)
    scene = build_scene_input(raw, target_points=2048)
    assert scene.shape == (1, 2048, 3)


def test_mock_sim_env() -> None:
    from anima_manip_ultradex.config import load_module_config
    from anima_manip_ultradex.sim.scene_env import SceneEnvAdapter

    cfg = load_module_config()
    adapter = SceneEnvAdapter(cfg, sim_backend="mock")
    env = adapter.create_env(num_envs=2)
    obs = env.reset()
    assert obs["scene_pc"].shape == (2, 2048, 3)
    obs2, reward, done, info = env.step(np.zeros((2, 36)))
    assert info["backend"] == "mock"
