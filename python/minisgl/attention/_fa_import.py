from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Callable


def load_flash_attn_with_kvcache(version: int) -> Callable:
    """Load FA3 without importing the optional FA4 CuTe DSL implementation."""
    if version == 4:
        module = importlib.import_module("sgl_kernel.flash_attn")
        return module.flash_attn_with_kvcache

    fa4_module_name = "sgl_kernel._fa4_interface"
    existing_fa4_module = sys.modules.get(fa4_module_name)
    if existing_fa4_module is not None:
        module = importlib.import_module("sgl_kernel.flash_attn")
        return module.flash_attn_with_kvcache

    # sgl-kernel eagerly imports its optional FA4 module even when ver=3 is
    # requested. On Ampere this can fail because newer CUTLASS DSL releases
    # no longer expose APIs used by that unused FA4 implementation.
    fa4_stub = ModuleType(fa4_module_name)
    fa4_stub.flash_attn_varlen_func = None  # type: ignore[attr-defined]
    sys.modules[fa4_module_name] = fa4_stub
    try:
        module = importlib.import_module("sgl_kernel.flash_attn")
    finally:
        if sys.modules.get(fa4_module_name) is fa4_stub:
            del sys.modules[fa4_module_name]
    return module.flash_attn_with_kvcache
