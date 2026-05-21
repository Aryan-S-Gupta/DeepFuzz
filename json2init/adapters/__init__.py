from __future__ import annotations

from .base import GenericAdapter, SeedAdapter
from .jax_adapter import JaxAdapter
from .tensorflow_adapter import TensorFlowAdapter
from .torch_adapter import TorchAdapter


def get_adapter(backend: str) -> SeedAdapter:
    name = (backend or "python").lower()
    if name == "torch":
        return TorchAdapter()
    if name == "tensorflow":
        return TensorFlowAdapter()
    if name == "jax":
        return JaxAdapter()
    return GenericAdapter()


__all__ = ["SeedAdapter", "GenericAdapter", "JaxAdapter", "TorchAdapter", "TensorFlowAdapter", "get_adapter"]
