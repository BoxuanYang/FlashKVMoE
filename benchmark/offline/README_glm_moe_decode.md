# GLM-4.5-Air 单层 MoE decode batch size 实验

在已安装本仓库与 pinned `kt-kernel` 的 Linux CUDA 环境中，从仓库根目录运行。
环境安装参考 [KT 安装说明](../../docs/ktransformers.md)。权重沿用仓库的 GGUF +
LLAMAFILE 路径；配置文件来自 `--model`，权重全部来自 `--kt-weight-path`。
请记录并保持 GGUF 量化版本一致，量化类型会写入结果 JSON。

```bash
python benchmark/offline/bench_glm_moe_decode.py \
  --model /path/to/GLM-4.5-Air \
  --kt-weight-path /path/to/GLM-4.5-Air-GGUF \
  --layer-index 7 \
  --kt-cpuinfer 64 \
  --kt-threadpool-count 2 \
  --output results/glm_moe_decode.json

python benchmark/offline/analyze_glm_moe_decode.py results/glm_moe_decode.json
```

分析脚本依赖 `numpy` 和 `matplotlib`，可在没有 CUDA 的机器上运行。
默认生成 `results/glm_moe_decode.png` 与 `results/glm_moe_decode_fit.json`。
也可用 `--output`、`--fit-output` 指定路径，图片支持 PNG/PDF/SVG。

## 实验定义

- batch size 固定默认为 `[1, 2, 4, 8, 16, 24, 32, 64, 128]`。
  一个 batch 有 B 个请求，每个请求贡献一个 decode token，输入形状为 `[B, hidden_size]`。
- 层索引按代码从 0 开始，默认加载 `blk.7` / `model.layers.7.mlp`。
  若“第 7 层”按自然数计数，使用 `--layer-index 6`。
- 只实例化并加载所选层的 MoE：router、routed experts、shared experts。
  GGUF 文件会被 mmap 并校验整个检查点的元数据，但不会加载其他层到 CPU 专家后端或 GPU，
  不会构造完整模型、attention、KV cache、embedding 或 LM head。
- KT 总工作线程数为 64，默认分配到 NUMA 0/1 各 32 个线程。
  KT 自己绑定线程并分配 NUMA 本地内存，无需额外 `numactl --interleave`。
  脚本检查 NUMA、可用物理核和 cpuset；配置不满足时直接报错。
  可用 `--kt-threadpool-count` 指定从节点 0 开始的线程池数，如四节点机器使用 4。
  避免外部 `taskset` 或容器只开放部分核心；同时检查 KT 启动日志是否有绑定/内存策略错误。
- 每个 batch size 单独捕获精确大小的 CUDA Graph，没有 padding。
  默认 5 轮，每轮每个大小先预热 20 次，再测量 100 次，共 500 个有效样本。
  每轮随机打乱 batch size 顺序，降低固定测量顺序造成的漂移。
- `avg_ms` 是**整个 batch 的单次 MoE 子层延迟**，单位 ms，包括 GPU router、CPU routed
  experts、GPU shared experts、输出相加、D2H/H2D 和 KT 同步。
  使用 `perf_counter_ns()` 包围 `graph.replay()` 和 `stream.synchronize()`，因此也包含
  graph launch 与主机等待开销。输入准备、编译、权重加载、graph capture、预热、结果统计均不计时。
  这不是每 token 延迟，也不是完整模型的每步 decode 延迟。
- 使用生产 LLAMAFILE 分派逻辑，未强制单 token kernel。
  当前后端会按 token 数切换内部计算路径，因此测量曲线可能存在拐点。

## 输入与逐 token 路由

默认每次 replay 前为 **每一个 token 独立生成新的 BF16 正态 hidden state**。
真实 checkpoint 的 GLM router 在 CUDA Graph 内执行，逐 token 计算 top-k IDs 和权重；
没有广播同一组专家、复制同一行 token，或将 capture 时的路由固化下来。
不同 token 可以自然地共享专家，但不假定它们的专家集合相同，也不强制它们互不重叠。

随机 hidden state 是合成微基准，并不能代表真实文本的专家冷热分布。若需要真实负载分布，
从目标层 **MoE 入口（post-attention layernorm 后）** 收集不同请求、不同 decode 步的
hidden states，保存成浮点 `.npy`，形状 `[N, hidden_size]`，其中 N 至少为最大 batch size。
文件中的行应来自不同 token，避免用重复行填充。运行时增加：

```bash
  --hidden-states /path/to/layer7_decode_hidden_states.npy
```

脚本每次从该池中无放回抽取 B 行作为一批；不同 replay 可再次抽到同一行。
抽样和输入传输均在计时区间之外。这个选项仍然只加载一层 MoE。

捕获后，每个大小都会用两组新输入对比生产 eager forward，并先将 KT 的 CPU/GPU
输出缓冲区写成 NaN，再验证 replay 结果，避免遗漏 host callback 却返回旧输出。
计时结束后读取 graph 实际产生的专家 IDs，累计：

- `expert_selection_counts`：每个专家的选中次数；
- `avg_unique_experts_per_batch`：每批覆盖的专家数均值；
- `avg_unique_expert_sets_per_batch`：每批不同 top-k 专家集合数均值，忽略集合内部顺序。

这些统计可帮助解释 batch size 增大时的专家覆盖与执行时间变化。批间的输入准备、同步和
诊断会产生间隔，因此结果表示预热后、逐次同步的隔离层延迟，不代表持续排队服务吞吐量。

## 输出和拟合

JSON 的 `results` 每一项包含 `batch_size`、`avg_ms`、`std_ms`、最小/最大值、
测量次数、每轮均值、所有原始 `samples_ms` 和路由统计；`metadata` 记录实验条件。
每完成一个大小的一轮测量就写入中间结果，全部完成后 `complete` 才为 `true`。
分析脚本拒绝拟合不完整数据。

折线图使用数值线性 batch size 横轴，显示实测均值、样本标准差与 OLS 拟合：
`latency_ms = slope_ms_per_token * batch_size + intercept_ms`。
回归给每个 batch size 的均值相同权重，输出斜率、截距、R²、RMSE、预测和残差。
R² 较低或残差有明显结构时，不宜用一条直线解释全部范围；不应直接外推至测量范围之外。

CPU 逻辑检查（不需要 CUDA/模型权重）：

```bash
python -m pytest tests/benchmark/test_glm_moe_decode.py -o addopts= -p no:cacheprovider
```
