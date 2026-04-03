import importlib
import importlib.util

import pytest

from anima_manip_ultradex.config import load_module_config

torch = importlib.import_module("torch") if importlib.util.find_spec("torch") is not None else None
pytestmark = pytest.mark.skipif(torch is None, reason="torch extra is not installed")
ActionQueryBank = None
UltraDexPolicy = None
PointEncoder = None
DecoderOnlyTransformer = None

if torch is not None:
    ActionQueryBank = importlib.import_module(
        "anima_manip_ultradex.policy.action_queries"
    ).ActionQueryBank
    UltraDexPolicy = importlib.import_module("anima_manip_ultradex.policy.network").UltraDexPolicy
    PointEncoder = importlib.import_module("anima_manip_ultradex.policy.point_encoder").PointEncoder
    DecoderOnlyTransformer = importlib.import_module(
        "anima_manip_ultradex.policy.transformer"
    ).DecoderOnlyTransformer


def test_point_encoder_shape_contract() -> None:
    cfg = load_module_config()
    encoder = PointEncoder(cfg, d_scene=64)
    points = torch.randn(2, cfg.paper.policy_input_points, 3)

    tokens = encoder(points)

    assert tokens.shape == (2, cfg.paper.abstraction_output_points, 64)


def test_action_queries_and_transformer_contract() -> None:
    cfg = load_module_config()
    query_bank = ActionQueryBank(cfg, d_model=64)
    transformer = DecoderOnlyTransformer(cfg, d_model=64, num_heads=4, num_layers=2)
    scene_tokens = torch.randn(2, cfg.paper.abstraction_output_points, 64)

    query_tokens = query_bank(batch_size=2)
    fused = transformer(query_tokens, scene_tokens)

    assert query_tokens.shape == (2, cfg.paper.action_query_tokens, 64)
    assert fused.shape == (2, cfg.paper.action_query_tokens, 64)


def test_policy_forward_and_nll() -> None:
    cfg = load_module_config()
    policy = UltraDexPolicy(cfg, d_model=64, num_heads=4, num_layers=2)
    points = torch.randn(2, cfg.paper.policy_input_points, 3)
    targets = torch.zeros(2, 36)

    output = policy(points, action_targets=targets)

    assert output.arm_actions.shape == (2, 2, 6)
    assert output.hand_actions.shape == (2, 2, 12)
    assert output.action_mean.shape == (2, 36)
    assert output.action_log_std.shape == (2, 36)
    assert output.nll_loss is not None
    assert output.nll_loss.ndim == 0
