"""Unit tests for training pipeline components: CheckpointManager, scheduler, early stopping."""

from pathlib import Path

import torch

# Import training components from scripts/train.py
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train import CheckpointManager, EarlyStopping, WarmupCosineScheduler


class TestCheckpointManager:
    def test_saves_checkpoint_file(self, tmp_path) -> None:
        mgr = CheckpointManager(str(tmp_path), keep_top_k=2)
        state = {"model": torch.zeros(10)}
        path = mgr.save(state, metric_value=1.0, step=100)
        assert path.exists()
        assert (tmp_path / "best.pth").exists()

    def test_keeps_only_top_k(self, tmp_path) -> None:
        mgr = CheckpointManager(str(tmp_path), keep_top_k=2, mode="min")
        for i, val in enumerate([1.0, 0.8, 0.6, 0.9]):
            mgr.save({"step": i}, metric_value=val, step=i * 100)
        # Only 2 best (lowest) should remain
        ckpts = list(tmp_path.glob("checkpoint_step*.pth"))
        assert len(ckpts) == 2

    def test_best_is_lowest_for_min_mode(self, tmp_path) -> None:
        mgr = CheckpointManager(str(tmp_path), keep_top_k=2, mode="min")
        mgr.save({"val": 1.0}, metric_value=1.0, step=100)
        mgr.save({"val": 0.5}, metric_value=0.5, step=200)
        mgr.save({"val": 0.8}, metric_value=0.8, step=300)
        best = torch.load(tmp_path / "best.pth", weights_only=True)
        assert best["val"] == 0.5

    def test_best_is_highest_for_max_mode(self, tmp_path) -> None:
        mgr = CheckpointManager(str(tmp_path), keep_top_k=2, mode="max")
        mgr.save({"val": 0.5}, metric_value=0.5, step=100)
        mgr.save({"val": 0.9}, metric_value=0.9, step=200)
        mgr.save({"val": 0.7}, metric_value=0.7, step=300)
        best = torch.load(tmp_path / "best.pth", weights_only=True)
        assert best["val"] == 0.9


class TestWarmupCosineScheduler:
    def _make_scheduler(self, warmup=10, total=100, lr=0.001):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        return WarmupCosineScheduler(optimizer, warmup, total), optimizer

    def test_warmup_ramps_linearly(self) -> None:
        sched, opt = self._make_scheduler(warmup=10, total=100, lr=0.001)
        lrs = []
        for _ in range(10):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        # LR should increase during warmup
        assert lrs[-1] > lrs[0]
        assert abs(lrs[-1] - 0.001) < 1e-6  # Should reach peak at end of warmup

    def test_cosine_decays_after_warmup(self) -> None:
        sched, opt = self._make_scheduler(warmup=5, total=50, lr=0.001)
        for _ in range(5):
            sched.step()  # warmup
        lr_at_peak = opt.param_groups[0]["lr"]
        for _ in range(45):
            sched.step()  # cosine
        lr_at_end = opt.param_groups[0]["lr"]
        assert lr_at_end < lr_at_peak

    def test_state_dict_roundtrip(self) -> None:
        sched, opt = self._make_scheduler(warmup=10, total=100)
        for _ in range(25):
            sched.step()
        state = sched.state_dict()
        assert state["current_step"] == 25

        sched2, _ = self._make_scheduler(warmup=10, total=100)
        sched2.load_state_dict(state)
        assert sched2.current_step == 25


class TestEarlyStopping:
    def test_no_stop_when_improving(self) -> None:
        es = EarlyStopping(patience=5, min_delta=0.001)
        for val in [1.0, 0.9, 0.8, 0.7, 0.6]:
            assert es.step(val) is False

    def test_stops_after_patience(self) -> None:
        es = EarlyStopping(patience=3, min_delta=0.001)
        es.step(0.5)  # new best
        assert es.step(0.6) is False  # worse, counter=1
        assert es.step(0.6) is False  # worse, counter=2
        assert es.step(0.6) is True  # worse, counter=3 >= patience

    def test_resets_on_improvement(self) -> None:
        es = EarlyStopping(patience=3, min_delta=0.001)
        es.step(0.5)
        es.step(0.6)  # counter=1
        es.step(0.6)  # counter=2
        es.step(0.3)  # improvement, reset
        assert es.counter == 0
        assert es.step(0.4) is False  # counter=1

    def test_max_mode(self) -> None:
        es = EarlyStopping(patience=2, min_delta=0.01, mode="max")
        es.step(0.8)
        assert es.step(0.7) is False  # counter=1
        assert es.step(0.7) is True  # counter=2


class TestConfigParsing:
    def test_paper_toml_loads_without_error(self) -> None:
        from anima_manip_ultradex.config import load_module_config

        cfg = load_module_config("configs/paper.toml")
        assert cfg.project.codename == "MANIP-ULTRADEX"
        assert cfg.compute.backend == "cuda"

    def test_debug_toml_loads_without_error(self) -> None:
        from anima_manip_ultradex.config import load_module_config

        cfg = load_module_config("configs/debug.toml")
        assert cfg.project.codename == "MANIP-ULTRADEX"
        assert cfg.compute.backend == "auto"

    def test_default_toml_loads_without_error(self) -> None:
        from anima_manip_ultradex.config import load_module_config

        cfg = load_module_config()
        assert cfg.paper.policy_input_points == 2048
