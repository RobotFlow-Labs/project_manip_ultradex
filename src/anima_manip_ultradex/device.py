"""Backend selection with macOS-first and CUDA-ready behavior."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@dataclass(frozen=True)
class BackendInfo:
    backend: str
    device: str
    torch_available: bool
    mlx_available: bool
    cuda_available: bool


def detect_backend(preferred: str | None = None) -> BackendInfo:
    backend = preferred or os.environ.get("ANIMA_BACKEND", "auto")
    mlx_available = _has_module("mlx.core")
    torch_available = _has_module("torch")
    cuda_available = False

    if torch_available:
        import torch

        cuda_available = torch.cuda.is_available()

    if backend == "mlx":
        return BackendInfo("mlx", "mlx", torch_available, mlx_available, cuda_available)
    if backend == "cuda":
        return BackendInfo("cuda", "cuda", torch_available, mlx_available, cuda_available)
    if backend == "cpu":
        return BackendInfo("cpu", "cpu", torch_available, mlx_available, cuda_available)

    if mlx_available:
        return BackendInfo("mlx", "mlx", torch_available, mlx_available, cuda_available)
    if cuda_available:
        return BackendInfo("cuda", "cuda", torch_available, mlx_available, cuda_available)
    return BackendInfo("cpu", "cpu", torch_available, mlx_available, cuda_available)
