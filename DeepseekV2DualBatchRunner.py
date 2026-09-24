# SPDX-License-Identifier: Apache-2.0
"""单 GPU、Eager 模式下，使用两个固定 microbatch 执行 DeepSeek-Coder-V2-Instruct decode。

TODO: 代码等待优化

复用 deepseek_v2.py 中已加载的 DeepseekModel 主干和 KT 权重，
GPU 阶段包含上一层 routed/shared 结果累加、Norm、Attention、router 和 shared expert。
MoE 阶段包含 CPU routed experts 计算及结果回传。
沿用调用方的一条 CUDA 流，通过 KT 原生接口提交、等待和回传 MoE 结果。
每个时间槽先提交 CPU MoE，再执行 GPU 工作，最后收回结果并同步。
输入传输和结果回传位于 CPU 计算前后，不与下一份 CPU MoE 重叠。

调用方提供两个未填充的 MiniSGL Batch，准备 input_ids、positions、out_loc 和 KV 映射。
本模块准备 Attention metadata，返回两份最终归一化后的 hidden states。
Batch 切分、KV 分配、LM head 和采样由调用方负责。

调用方负责启用推理模式。始终使用构造时的 CUDA 流，固定两个 microbatch 的大小，
同一时间只允许一个调用方使用模型及 KT CPUInfer 队列，并保持 runner 存活直到流执行完毕。
在模型加载后创建一次，随后在 engine.stream 上复用：

    runner = DeepseekV2DualBatchRunner(engine.model.model)
    h0, h1 = runner.forward(m0, m1)
"""

from __future__ import annotations

import torch
from minisgl.attention.fi_mla import FlashInferMLABackend
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.models.deepseek_v2 import DeepseekModel, DeepseekMoE


class DeepseekV2DualBatchRunner:
    def __init__(self, model: DeepseekModel):
        self.model = model
        self.layers = model.layers.op_list
        self.layer_num = len(self.layers)
        self.config = self.layers[0].self_attn._config
        self.device = model.embed_tokens.weight.device
        if self.config.model_type != "deepseek_v2" or get_tp_info().size != 1:
            raise ValueError("This runner requires a single-rank DeepSeek V2 model")
        if self.device.type != "cuda" or model.embed_tokens.weight.dtype != torch.bfloat16:
            raise ValueError("Load the model on CUDA with BF16 activations first")
        self.cpu_moe = [
            layer.mlp._wrapper if isinstance(layer.mlp, DeepseekMoE) else None
            for layer in self.layers
        ]
        wrappers = [
            self.cpu_moe[i]
            for i, layer in enumerate(self.layers)
            if isinstance(layer.mlp, DeepseekMoE)
        ]
        if not wrappers or any(w is None or w.moe is None for w in wrappers):
            raise ValueError("Load the KT routed experts before creating the runner")
        if any(
            w.cpu_infer is not wrappers[0].cpu_infer
            or w.method != "LLAMAFILE"
            or w.num_gpu_experts != 0
            for w in wrappers
        ):
            raise ValueError("KT must share one CPUInfer, with LLAMAFILE CPU experts")
        self.max_tokens = min(w.chunked_prefill_size for w in wrappers)
        self.stream = torch.cuda.current_stream(self.device)
        # 两份 Attention 计划各自持有工作区，避免后一次规划覆盖前一次。
        self.attention = (FlashInferMLABackend(self.config), FlashInferMLABackend(self.config))
        self.batch_sizes = None

    def _validate_inputs(self, batches: tuple[Batch, Batch]):
        if torch.cuda.current_stream(self.device) != self.stream:
            raise ValueError("Use the same CUDA stream as the runner's construction")
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("Dual-microbatch decode uses eager execution")
        for batch in batches:
            if not batch.is_decode or not 0 < batch.size == batch.padded_size <= self.max_tokens:
                raise ValueError("Supply two nonempty, unpadded decode batches within KT capacity")
            if any(req.extend_len != 1 for req in batch.reqs):
                raise ValueError("Decode requires one token per request")
            for tensor in (batch.input_ids, batch.positions, batch.out_loc):
                if tensor.device != self.device or tensor.shape != (batch.size,):
                    raise ValueError(
                        "Each batch needs one CUDA input_id, position and out_loc per request"
                    )
        rows = [req.table_idx for batch in batches for req in batch.reqs]
        if len(set(rows)) != len(rows):
            raise ValueError("The two microbatches must contain distinct request rows")
        if self.batch_sizes is not None and tuple(b.size for b in batches) != self.batch_sizes:
            raise ValueError("Microbatch sizes are fixed for the lifetime of this runner")

    def forward(self, m0: Batch, m1: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """执行一个 decode step，按输入顺序返回两份最终归一化的结果。"""
        self.batches = (m0, m1)
        return self._run_pipeline()

    def _run_pipeline(self) -> tuple[torch.Tensor, torch.Tensor]:
        """每轮对应一个时间槽，依次完成启动、并行执行和收尾。"""
        self._validate_inputs(self.batches)
        ctx = get_global_ctx()
        if ctx._batch is not None:
            raise RuntimeError("Call the runner outside Context.forward_batch")
        self.batch_sizes = tuple(b.size for b in self.batches)
        # 每步只规划一次 Attention，放在提交 CPU MoE 回调之前。
        for batch, attention in zip(self.batches, self.attention):
            attention.prepare_metadata(batch)
            attention._initialize(batch.attn_metadata)

        self.hidden_states = [self.model.embed_tokens.forward(b.input_ids) for b in self.batches]
        self.residual = [None, None]
        self.shared = [None, None]
        self.topk_ids = [None, None]
        self.topk_weights = [None, None]
        previous_backend = ctx.attn_backend
        overlap_steps = 2 * self.layer_num - 1
        for i in range(overlap_steps + 2):
            if i == 0:
                # 启动：GPU 执行第 0 层 m0 的 Attention，CPU 空闲。
                self.submit_attn(0, 0)
            elif i == overlap_steps + 1:
                # 收尾：提交并收回最后一层 m1 的 MoE。
                self.submit_moe(1, self.layer_num - 1)
                self.sync_moe(1, self.layer_num - 1)
            else:
                attn_batch = i % 2
                moe_batch = 1 - attn_batch
                attn_layer = i // 2
                moe_layer = (i - 1) // 2

                self.submit_moe(moe_batch, moe_layer)
                self.submit_attn(attn_batch, attn_layer)
                self.sync_moe(moe_batch, moe_layer)

            # 本槽完成后再复用 KT 暂存区，包括 CPU 结果回传和独立副本的保存。
            self.stream.synchronize()
        ctx.attn_backend = previous_backend
        # 最后一层没有下一次 Attention，在这里完成 routed/shared 累加。
        for mb in (0, 1):
            if self.shared[mb] is not None:
                self.hidden_states[mb] = self.hidden_states[mb] + self.shared[mb]
                self.shared[mb] = None
        return tuple(
            self.model.norm.forward(self.hidden_states[mb], self.residual[mb])[0] for mb in (0, 1)
        )

    def submit_attn(self, mb: int, layer_idx: int):
        layer = self.layers[layer_idx]
        # 上一层的 routed 结果已回传，与 shared 结果累加后进入本层。
        if self.shared[mb] is not None:
            self.hidden_states[mb] = self.hidden_states[mb] + self.shared[mb]
            self.shared[mb] = None
        ctx = get_global_ctx()
        ctx.attn_backend = self.attention[mb]
        with ctx.forward_batch(self.batches[mb]):
            x, self.residual[mb] = layer.input_layernorm.forward(
                self.hidden_states[mb], self.residual[mb]
            )
            x = layer.self_attn.forward(x)
            x, self.residual[mb] = layer.post_attention_layernorm.forward(x, self.residual[mb])
        self.hidden_states[mb] = x
        if isinstance(layer.mlp, DeepseekMoE):
            self.topk_ids[mb], self.topk_weights[mb] = layer.mlp.gate.forward(x)
            self.shared[mb] = layer.mlp.shared_experts.forward(x)

    def submit_moe(self, mb: int, layer_idx: int):
        mlp = self.layers[layer_idx].mlp
        x = self.hidden_states[mb]
        if not isinstance(mlp, DeepseekMoE):
            self.hidden_states[mb] = mlp.forward(x)
            return
        self.cpu_moe[layer_idx].submit_forward(
            x, self.topk_ids[mb], self.topk_weights[mb], self.stream.cuda_stream
        )

    def sync_moe(self, mb: int, layer_idx: int):
        """排入 KT 等待和结果复制操作，由循环末尾同步确保完成。"""
        if self.cpu_moe[layer_idx] is None:
            return
        output = self.cpu_moe[layer_idx].sync_forward(
            self.hidden_states[mb], self.stream.cuda_stream
        )
        # KT 返回共享暂存区，复制后才能复用；与 shared 的累加留给下一轮 GPU 阶段。
        self.hidden_states[mb] = output.clone()
        self.topk_ids[mb] = self.topk_weights[mb] = None
