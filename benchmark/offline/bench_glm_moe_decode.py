"""GLM-4.5-Air：CUDA Graph 下，只测 CPU routed experts 的计算时间。"""

import ctypes
import json
import os
import runpy
import statistics
import sys
import time
from pathlib import Path

# 服务器上的实验参数。激活 minisgl-kt 环境后可直接运行本文件。
MODEL = "/data1/models/GLM-4.5-Air-GGUF"
WEIGHT_PATH = "/data1/models/GLM-4.5-Air-GGUF/IQ4_XS"
PHYSICAL_GPU = "2"
LAYER_INDEX = 7  # 从 0 开始，即 GGUF 的 blk.7
BATCH_SIZES = range(1, 129)
REPEATS = 20
WARMUP = 20
CPU_THREADS = 64
NUMA_POOLS = 2  # KT 自动绑定到 NUMA 0/1，各 32 个工作线程
DEVICE = "cuda:0"
SEED = 42
OUTPUT = Path("results/glm_moe_decode.json")
AUTO_ANALYZE = True


# 在回调内部计时，避免把 router、传输、graph launch 或回调排队算进去。
# 这里仅调用同步 CPU 接口，不能调用任何 CUDA API。
@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def timed_experts(_):
    global elapsed_ms, callback_error
    try:
        start = time.perf_counter_ns()
        experts.forward(*forward_args)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    except BaseException as error:
        callback_error = error  # ctypes 回调不能直接向主线程抛异常


if __name__ == "__main__":
    # 必须在 import torch 之前设置。脚本中的 cuda:0 对应物理 GPU 2。
    os.environ["CUDA_VISIBLE_DEVICES"] = PHYSICAL_GPU
    if sys.platform != "linux":
        raise RuntimeError("需要 Linux、CUDA 和本仓库的 KT LLAMAFILE 后端")
    if ctypes.CDLL("libnuma.so.1").numa_available() < 0:
        raise RuntimeError("NUMA 不可用")
    for node in range(NUMA_POOLS):
        if not Path(f"/sys/devices/system/node/node{node}").exists():
            raise RuntimeError(f"NUMA 节点 {node} 不存在")

    import torch
    from kt_kernel import KTMoEWrapper
    from kt_kernel.utils.llamafile import LlamafileMoEWrapper
    from minisgl.models.config import ModelConfig
    from minisgl.models.gguf import GGUFWeights
    from minisgl.models.glm4_moe import Glm4MoeRouter
    from transformers import AutoConfig

    torch.cuda.set_device(DEVICE)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)  # 64 个计算线程由 KT 管理
    config = ModelConfig.from_hf(AutoConfig.from_pretrained(MODEL))
    if (
        not config.is_glm4_moe
        or not config.first_k_dense_replace <= LAYER_INDEX < config.num_layers
    ):
        raise ValueError("需要 GLM-4.5-Air 配置和有效的 MoE 层索引")

    # mmap 检查点，只加载这一层的 router 和 routed experts，不加载 shared experts。
    checkpoint = GGUFWeights(WEIGHT_PATH, config)
    with torch.device("meta"):
        router = Glm4MoeRouter(config)
    router.weight = checkpoint._dequantize(
        f"blk.{LAYER_INDEX}.ffn_gate_inp.weight", torch.device(DEVICE), torch.float32, 16 << 20
    )
    router.e_score_correction_bias = checkpoint._dequantize(
        f"blk.{LAYER_INDEX}.exp_probs_b.bias", torch.device(DEVICE), torch.float32, 16 << 20
    )
    LlamafileMoEWrapper._gguf_loaders_by_path[os.path.realpath(WEIGHT_PATH)] = checkpoint
    wrapper = KTMoEWrapper(
        layer_idx=LAYER_INDEX,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        moe_intermediate_size=config.moe_intermediate_size,
        gpu_experts_mask=None,
        cpuinfer_threads=CPU_THREADS,
        threadpool_count=NUMA_POOLS,
        weight_path=WEIGHT_PATH,
        chunked_prefill_size=max(BATCH_SIZES),
        method="LLAMAFILE",
        max_deferred_experts_per_token=0,
    )
    wrapper.load_weights(torch.arange(config.num_experts, dtype=torch.int32))
    experts = wrapper.moe
    # KT 的 submit_with_cuda_stream 接受 (函数地址, 参数地址)。它会先将
    # CPUInfer 指针写入参数首字段，因此预留一个指针槽；回调不使用它。
    callback_context = ctypes.c_void_p()
    callback_task = (
        ctypes.cast(timed_experts, ctypes.c_void_p).value,
        ctypes.addressof(callback_context),
    )
    elapsed_ms, callback_error = float("nan"), None
    stream = torch.cuda.Stream(device=DEVICE)
    stream.wait_stream(torch.cuda.current_stream())
    data = {
        "complete": False,
        "metadata": {
            "model": MODEL,
            "weight_path": WEIGHT_PATH,
            "layer_index": LAYER_INDEX,
            "cuda_graph": True,
            "kt_cpuinfer": CPU_THREADS,
            "kt_threadpool_count": NUMA_POOLS,
            "repeats": REPEATS,
            "warmup": WARMUP,
            "seed": SEED,
            "top_k": config.num_experts_per_tok,
            "input_source": "independent normal hidden states",
            "timing": "synchronous CPU routed-expert forward inside CUDA Graph host callback",
            "excluded": [
                "router",
                "D2H",
                "H2D",
                "shared experts",
                "graph launch",
                "callback queue",
            ],
            "gpu": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "expert_quantization": {
                p: checkpoint.tensors[f"blk.{LAYER_INDEX}.ffn_{p}_exps.weight"].tensor_type.name
                for p in ("gate", "up", "down")
            },
        },
        "results": [],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode(), torch.cuda.stream(stream):
        for batch_size in BATCH_SIZES:
            x = torch.randn(batch_size, config.hidden_size, device=DEVICE, dtype=torch.bfloat16)
            x_cpu = torch.empty_like(x, device="cpu", pin_memory=True)
            ids_cpu = torch.empty(
                batch_size, config.num_experts_per_tok, dtype=torch.int64, pin_memory=True
            )
            weights_cpu = torch.empty_like(ids_cpu, dtype=torch.float32, pin_memory=True)
            output_cpu = torch.empty_like(x_cpu, pin_memory=True)
            length_cpu = torch.tensor([batch_size], dtype=torch.int32)
            forward_args = (
                length_cpu.data_ptr(),
                config.num_experts_per_tok,
                ids_cpu.data_ptr(),
                weights_cpu.data_ptr(),
                x_cpu.data_ptr(),
                output_cpu.data_ptr(),
                False,
            )

            # 初始化 CUDA 算子；每行 token 都有独立的专家 IDs 和权重。
            router.forward(x)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                ids, weights = router.forward(x)
                x_cpu.copy_(x, non_blocking=True)
                ids_cpu.copy_(ids, non_blocking=True)
                weights_cpu.copy_(weights, non_blocking=True)
                wrapper.cpu_infer.submit_with_cuda_stream(stream.cuda_stream, callback_task)

            # 新输入 + 清空输出，检查 graph 回调确实重新计算。
            x.normal_()
            expected_ids, expected_weights = router.forward(x)
            x_cpu.copy_(x)
            ids_cpu.copy_(expected_ids)
            weights_cpu.copy_(expected_weights)
            stream.synchronize()
            experts.forward(*forward_args)
            expected = output_cpu.clone()
            output_cpu.fill_(float("nan"))
            graph.replay()
            stream.synchronize()
            if callback_error is not None:
                raise callback_error
            torch.testing.assert_close(output_cpu, expected, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(ids_cpu, expected_ids.cpu())

            times = []
            expert_counts = torch.zeros(config.num_experts, dtype=torch.int64)
            for repeat in range(WARMUP + REPEATS):
                x.normal_()  # 每次、每个 token 使用独立输入，不复制同一组专家路由
                elapsed_ms = float("nan")
                graph.replay()
                stream.synchronize()
                if callback_error is not None:
                    raise callback_error
                if not elapsed_ms > 0:
                    raise RuntimeError("CUDA Graph 没有执行计时回调")
                if repeat >= WARMUP:
                    times.append(elapsed_ms)
                    expert_counts += torch.bincount(ids_cpu.flatten(), minlength=config.num_experts)

            result = {
                "batch_size": batch_size,
                "avg_ms": statistics.mean(times),
                "num_measurements": len(times),
                "samples_ms": times,
                "expert_selection_counts": expert_counts.tolist(),
            }
            data["results"].append(result)
            data["complete"] = len(data["results"]) == len(BATCH_SIZES)
            OUTPUT.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(
                f"batch_size={batch_size:3d}, routed experts={result['avg_ms']:.4f} ms", flush=True
            )
            graph.reset()  # 释放 graph 后，下一组才替换 CPU 缓冲区和 forward_args

    print(f"\n测量完成：{OUTPUT}", flush=True)
    if AUTO_ANALYZE:
        print("开始画图和线性回归……", flush=True)
        analysis = Path(__file__).with_name("analyze_glm_moe_decode.py")
        runpy.run_path(str(analysis), run_name="__main__")
