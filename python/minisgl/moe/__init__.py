from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry, init_logger

from .base import BaseMoeBackend

logger = init_logger(__name__)

if TYPE_CHECKING:
    from minisgl.engine import EngineConfig


class MoeBackendCreator(Protocol):
    def __call__(self, config: EngineConfig) -> BaseMoeBackend: ...


SUPPORTED_MOE_BACKENDS = Registry[MoeBackendCreator]("MoE Backend")


@SUPPORTED_MOE_BACKENDS.register("fused")
def create_fused_moe_backend(config: EngineConfig):
    from .fused import FusedMoe

    return FusedMoe()


@SUPPORTED_MOE_BACKENDS.register("ktransformers")
def create_kt_moe_backend(config: EngineConfig):
    from .ktransformers import KTransformersMoe

    return KTransformersMoe(config)


def create_moe_backend(backend: str, config: EngineConfig) -> BaseMoeBackend:
    return SUPPORTED_MOE_BACKENDS[backend](config)


__all__ = [
    "BaseMoeBackend",
    "create_moe_backend",
    "SUPPORTED_MOE_BACKENDS",
]
