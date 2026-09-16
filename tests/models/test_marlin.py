"""Binary compatibility with KT archive plus real CUDA GEMM/graph checks."""

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from minisgl.layers.marlin import MarlinLinear, pack_marlin


def quantized_reference(weight):
    """Dense reference for group-64 W4A16, before Marlin's layout permutation."""
    n, k = weight.shape
    grouped = weight.reshape(n, k // 64, 64)
    scales = grouped.abs().amax(-1, keepdim=True).mul_(2 / 15)
    scales.masked_fill_(scales == 0, 1)
    codes = torch.round(grouped / scales).int().add_(8).clamp_(0, 15)
    return ((codes - 8).to(weight.dtype) * scales).reshape(n, k)


@pytest.fixture(scope="module")
def archive_quantize():
    # Import only upstream's pure packing functions. Importing its whole module
    # initializes CUDA and pulls in the old KT runtime even in CPU-only tests.
    root = Path(__file__).resolve().parents[2]
    utils = (
        root
        / "third_party/ktransformers/archive/ktransformers/ktransformers_ext/operators/custom_marlin/quantize/utils"
    )

    def read_module(name):
        spec = importlib.util.spec_from_file_location(name, utils / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    perms, quant = read_module("marlin_perms"), read_module("quant_utils")
    names = {"marlin_permute_weights", "marlin_weights", "marlin_permute_scales", "marlin_quantize"}
    source = ast.parse((utils / "marlin_utils.py").read_text(encoding="utf-8"))
    source.body = [
        node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = dict(
        vars(perms),
        **{
            "numpy": np,
            "torch": torch,
            "MARLIN_TILE": 16,
            "get_pack_factor": quant.get_pack_factor,
            "quantize_weights": quant.quantize_weights,
        },
    )
    exec(compile(source, str(utils / "marlin_utils.py"), "exec"), namespace)
    return namespace["marlin_quantize"]


@pytest.mark.parametrize("n,k", [(64, 128), (192, 256), (1088, 256)])
def test_packing_matches_archive(archive_quantize, n, k):
    weight = torch.randn(n, k, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    expected_weight, expected_scales, *_ = archive_quantize(weight.T, 4, 64, False)
    packed, scales = pack_marlin(weight)
    torch.testing.assert_close(packed, expected_weight, rtol=0, atol=0)
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
    assert (
        packed.numel() * packed.element_size() + scales.numel() * scales.element_size()
        < weight.numel()
    )


def test_zero_groups_and_output_padding():
    packed, scales = pack_marlin(torch.zeros(67, 256, dtype=torch.bfloat16))
    assert packed.shape == (16, 256) and scales.shape == (4, 128)
    assert (packed == -2004318072).all()  # eight signed-zero codes, 0x88888888
    assert (scales == 1).all()


@pytest.mark.skipif(
    sys.platform != "linux" or not torch.cuda.is_available(), reason="requires Linux CUDA"
)
@pytest.mark.parametrize(
    "n,k", [(67, 256), (5120, 2048), (2048, 4096), (9216, 4096), (4096, 8192), (151936, 128)]
)
def test_cuda_gemm_and_graph(n, k):
    weight = (torch.randn(n, k) * 0.03).to(torch.bfloat16)
    packed, scales = pack_marlin(weight)
    with torch.device("meta"):
        layer = MarlinLinear(k, n)
    layer.load_state_dict({"weight": packed.cuda(), "scales": scales.cuda()})
    reference = quantized_reference(weight).cuda()
    for tokens in (1, 17, 65, 257):
        x = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(layer.forward(x), F.linear(x, reference), atol=0.03, rtol=0.03)
    stream = torch.cuda.Stream()
    x = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer.forward(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            y = layer.forward(x)
        for _ in range(3):
            x.normal_()
            graph.replay()
            torch.testing.assert_close(y, F.linear(x, reference), atol=0.03, rtol=0.03)
    torch.cuda.current_stream().wait_stream(stream)
