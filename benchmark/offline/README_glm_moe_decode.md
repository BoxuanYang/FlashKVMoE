# GLM-4.5-Air routed experts 耗时实验

脚本已经配置好本机模型目录、IQ4_XS 权重目录和物理 GPU 2。
需要 Linux、CUDA、本仓库及 pinned KT LLAMAFILE 环境。

```bash
cd ~/FlashKVMoE
conda activate minisgl-kt
python benchmark/offline/bench_glm_moe_decode.py
```

这一条 Python 命令会完成测量、保存 JSON、画图和线性回归，无需传参数。

所有参数都在文件开头。默认 batch size 为 **1–128 的全部整数**，共 128 组。
每组预热 20 次，再正式测量 **20 次**，JSON 中的 avg_ms 是这 20 次的算术平均值。
正式测量总共 2560 次。输出到 results/glm_moe_decode.json，分析脚本生成 PNG 和回归 JSON。

只加载层索引 7（blk.7）的 router 与 routed experts；自然计数的第七层对应索引 6。
不加载 shared experts、attention 或其他层权重。GGUF 检查点仍会整体 mmap 和校验元数据。
CPU_THREADS=64，NUMA_POOLS=2，KT 将工作线程分配到 NUMA 0/1，各 32 个线程。
运行前确认每个节点有足够物理核且进程可访问这些核心和内存节点，检查 KT 的绑定日志。

CUDA Graph 捕获并重放以下顺序：

```text
GPU router → 输入/IDs/权重 D2H → CPU 回调内的同步 routed-expert forward
                                  [计时开始 ---------------- 计时结束]
```

计时器放在回调内部，**只包围 experts.forward(...)**。不包含 router、D2H、H2D、
shared experts、graph launch 或回调排队时间。结果留在 CPU，根本不执行输出 H2D。
时间包含专家输入量化、专家矩阵计算、激活、加权聚合、NUMA 工作线程调度与结果合并，
以及一次 Python→C++ 绑定调用的少量开销；它不是只测 GEMM 指令的时间。

回调直接调用 KT 的同步 CPU forward，使用同一个 NUMA 工作线程池；不经过生产路径的
CPUInfer 异步任务队列。这样可以隔离专家计算时间，不应将其解释为完整 MoE 延迟或服务吞吐量。
每个大小先用新输入对比普通同步 forward 和 graph replay，并把输出清为 NaN，检查是否真的重算。
每次 graph 执行完成后才读取时间和更换输入，回调内没有 CUDA API 调用。

每次测量都给 **每一个 token 独立生成 BF16 正态 hidden state**，由真实 router 分别计算
专家 IDs 和权重，不广播同一组专家，也不假设各专家收到相同数量的 token。
JSON 的 expert_selection_counts 记录 20 次正式测量中每个专家实际被选中的次数。

GLM 当前路由器采用固定 top-k：每个 token 的选中专家**数量**来自模型配置
num_experts_per_tok；选中的专家**集合**及各专家的 token 负载会随输入改变。
人为让每个 token 使用不同的 k，会改变模型的路由算法，本脚本不这样做。
随机 hidden states 是合成微基准，不代表真实文本的专家冷热分布。

分析脚本同样在顶部配置文件路径，按实际 batch size 数值做等权 OLS：
latency_ms = slope_ms_per_token * batch_size + intercept_ms，输出 R² 和 RMSE。
若曲线有拐点或 R² 较低，线性拟合只用于概括趋势，不宜直接外推。
