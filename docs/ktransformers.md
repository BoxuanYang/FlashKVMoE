# Qwen3 MoE / DeepSeek Coder V2：Mini-SGLang + KTransformers

DeepSeek-Coder-V2-Instruct 的 GPU router/shared experts、KT routed experts 和三段 IQ4_XS GGUF 启动命令见 [README](../README.md#quick-start-deepseek-coder-v2-instruct)。以下保留 Qwen3 说明与两者共用的安装流程。

支持 Qwen3-30B-A3B、Qwen3-235B-A22B，全部模型权重从同一份完整 GGUF 加载。
HF 目录或仓库只提供 config 和 tokenizer 等小文件，无需下载 safetensors。
使用 KT standalone Python API，不依赖 SGLang 或 sglang-kt。

| 部分 | 计算位置 | 权重来源 |
| --- | --- | --- |
| Attention Q/K/V/O、LM head | GPU | GGUF 在 CPU 转为 Marlin INT4，group size 64，GEMM 内融合解量化 |
| Embedding、Q/K norm、decoder norm、最终 norm、router | GPU | GGUF，加载时反量化为 BF16 |
| RoPE | GPU | HF config |
| 所有层、所有 MoE 专家的 gate/up/down 投影与激活 | CPU | KT 加载 GGUF |

这里的 CPU MoE 指专家计算；router 是 GPU 上的小矩阵。GGUF 加载器只反量化非专家张量，
不会把专家权重反量化或复制到 GPU。模型初始化使用 meta tensor，加载前
将整层 `mlp` 替换为 `KTransformersMoE`，其内部直接完成 GPU router/top-k 并调用 KT
wrapper，不再经过 `MoEMLP.experts`。原 router 保留为 `mlp.gate`，权重键不变；
GPU state dict 中没有专家参数。MiniSGL 从 GGUF 加载 GPU 权重，
KT 从同一路径获取 `blk.N.ffn_{gate,up,down}_exps.weight`，保持专家量化格式。
两个读取器使用 mmap，不需要第二份权重文件。

模型权重仅接受完整 GGUF，`--kt-weight-path` 指定单文件或同一量化版本的全部分片目录；
不递归查找。启动时检查模型架构、必需张量、形状、关键配置及分片完整性。
`--model` 只提供 config 和 tokenizer，不从该路径加载模型权重。

Q/K/V 按顺序合并，QKV/O/LM head 在 CPU 解量化后重新量化为 INT4，只把压缩权重和
BF16 scale 传到 GPU。CPU 量化临时空间限制为每次 1024 个输出行，CPU 上仍短暂保留当前
投影的完整 BF16 张量；GPU 不保留这些投影的 BF16 副本。prefill 和 decode 均使用 Marlin，
继续使用现有 Q/K norm、RoPE、FlashInfer 和 BF16 KV cache。
Qwen3 MoE 的 Q/K 权重无需 Llama 式排列还原，参见
[llama.cpp 的 Qwen3 MoE 转换实现](https://github.com/ggml-org/llama.cpp/blob/master/conversion/qwen.py)。
GGUF 的 Q4_K 等布局不能直接交给 Marlin；此次重新量化会增加误差，不能保证与原 BF16
投影数值完全相同。GGUF 解码复用 `gguf` 库，没有新增量化格式解码器。
CPU 测试覆盖 F32/F16/BF16、Q8_0、Q4_0、Q4_K、Q6_K GPU 张量；其他类型取决于
`gguf` 库的反量化实现，不支持的类型会报告具体张量和量化类型。专家类型仍受 KT 限制。

范围限定为单 GPU、BF16 激活、LLAMAFILE、全部专家在 CPU、无 deferred experts。
Qwen3 的 `qwen3_moe.py` 直接使用 `MarlinLinear`。它只保留 INT4/group-64 的打包和
MiniSGL BaseOP 适配；CUDA 源文件直接来自固定 KT 子模块的
`archive/csrc/ktransformers_ext/cuda/gptq_marlin`，首次启动由 PyTorch JIT 编译并缓存。
无需安装 archive 的 Python 包或旧版 `cpuinfer_ext`，也不复制 CUDA 内核。
必须保留源码 checkout，按下方命令使用 editable 安装，且启动时能找到 nvcc、C++ 编译器和 ninja。
KT 模式支持 decode CUDA graph，prefill 仍走 eager。其他模型/精度/TP 配置会报错。
`--cuda-graph-max-bs N` 指定捕获的最大 batch size（包含 N 本身），`0` 显式关闭。
捕获前会注册 KT 的固定 CPU/GPU 缓冲区，CPU 专家通过 CUDA host callback 在每次 replay
时重新执行；内部专家缓冲区包含 graph padding 所需容量。开启 graph 时不能设置
`KT_FORCE_SYNC_SUBMIT=1`，否则启动报错，避免 CPU 计算只在 capture 时执行。

## 1. 克隆与安装

在实验室 Linux x86-64 机器执行，全部用户态依赖通过 conda/pip 安装，不需要 sudo 或 apt。
需要 AVX2 CPU、支持 BF16 的 NVIDIA GPU，以及机器已有的兼容 NVIDIA 驱动。
示例使用 Python 3.12、PyTorch 2.9.1 CUDA 12.8；CUDA Toolkit 和 C/C++ 编译器也装入
conda 环境。Conda 不安装内核驱动，先确认机器上的 `nvidia-smi` 能正常运行。

```bash
conda create -n minisgl-kt --override-channels -c conda-forge -y \
  python=3.12 pip git 'cmake<4' ninja make pkg-config libhwloc libnuma \
  gcc_linux-64=13 gxx_linux-64=13
conda activate minisgl-kt
conda install --override-channels -c nvidia/label/cuda-12.8.1 -c conda-forge -y \
  cuda-toolkit=12.8.1

git clone --branch main https://github.com/BoxuanYang/FlashKVMoE.git
cd FlashKVMoE

# 使用主仓库固定的 KT 提交，只拉取 standalone kernel 必需的子模块。
git submodule update --init third_party/ktransformers
git -C third_party/ktransformers submodule update --init --recursive \
  third_party/llama.cpp third_party/pybind11
git -C third_party/ktransformers rev-parse HEAD
# 4882505c9a66a6784b3360a1b6ba9b23d53d0291

# 编译和后续启动服务时均使用当前 conda 环境的工具链。
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CXX"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
nvcc --version
"$CXX" --version
pkg-config --modversion hwloc
nvidia-smi

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
# 禁用构建隔离，让 KT 使用上面固定的 conda CMake 和编译环境。
python -m pip install pybind11
CMAKE_ARGS="-DUSE_CONDA_TOOLCHAIN=ON" \
  CPUINFER_USE_CUDA=1 CPUINFER_CPU_INSTRUCT=NATIVE CPUINFER_PARALLEL=8 \
  python -m pip install --no-build-isolation -v -e third_party/ktransformers/kt-kernel
python -m pip install pytest
python -m pip check

CUDA_VISIBLE_DEVICES=2 python - <<'PY'
import torch
import kt_kernel
from kt_kernel import kt_kernel_ext
assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
assert hasattr(kt_kernel_ext.CPUInfer, "submit_with_cuda_stream")
from kt_kernel_ext.moe import MOE
print(torch.__version__, torch.cuda.get_device_name(0))
print("KT:", kt_kernel.__version__, kt_kernel.__cpu_variant__)
print("KT source:", kt_kernel.__file__)
PY
```

KT 以 editable 方式安装，修改 `third_party/ktransformers/kt-kernel/python` 后重启服务即可。
修改 C++ 则重新执行上面的 KT 安装命令，增加 `CPUINFER_FORCE_REBUILD=1`。
`USE_CONDA_TOOLCHAIN=ON` 是固定 KT 源码已有的开关：选择 conda 编译器，并将 conda 的
头文件、库、pkg-config 和运行时 RPATH 加入搜索路径。仅安装 conda GCC 而不启用该开关，
KT 仍可能强制选择 `/usr/bin/gcc`。GCC 固定为 13，处于 CUDA 12.8 支持范围内。

新开终端后，先 `conda activate minisgl-kt`，再执行上面的环境变量 export 段，然后运行
后面的服务启动命令，确保 Mini-SGLang 的 JIT 编译能找到 conda CUDA 和 C++ 编译器。
这套 conda 安装命令已按上游构建脚本核对，但尚未在实验室 Linux 机器实际编译验证。
依赖来源：[NVIDIA conda 安装说明](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html#conda-installation)、
[libhwloc](https://anaconda.org/conda-forge/libhwloc)、[libnuma](https://anaconda.org/conda-forge/libnuma)。

## 2. 权重

已有的 `/data2/models/Qwen3-30B-A3B` 和 `/data2/models/Qwen3-30B-A3B-GGUF`
可以直接使用。HF 目录只需含 config 和 tokenizer；GGUF 必须与 HF
目录是同一模型版本。不要把不同量化版本的 GGUF 混放在一个目录。

没有权重时，可执行以下命令，只下载一份 GGUF 权重以及 HF 小文件：

```bash
hf download Qwen/Qwen3-30B-A3B \
  --include 'config.json' 'generation_config.json' 'tokenizer*' 'vocab.json' 'merges.txt' 'special_tokens_map.json' 'added_tokens.json' 'chat_template*' \
  --local-dir /data2/models/Qwen3-30B-A3B
hf download Qwen/Qwen3-30B-A3B-GGUF \
  --include 'Qwen3-30B-A3B-Q4_K_M.gguf' \
  --local-dir /data2/models/Qwen3-30B-A3B-GGUF

hf download Qwen/Qwen3-235B-A22B \
  --include 'config.json' 'generation_config.json' 'tokenizer*' 'vocab.json' 'merges.txt' 'special_tokens_map.json' 'added_tokens.json' 'chat_template*' \
  --local-dir /data2/models/Qwen3-235B-A22B
hf download Qwen/Qwen3-235B-A22B-GGUF \
  --include 'Q4_K_M/*.gguf' \
  --local-dir /data2/models/Qwen3-235B-A22B-GGUF
```

官方 235B GGUF 使用 `Q4_K_M/` 子目录。KT 不递归查找 GGUF，因此启动时指向这个子目录，
并保留全部分片。如果你的 GGUF 直接放在上一级目录，修改 `--kt-weight-path` 即可。
不要仅传入第一个分片文件；应传整个分片目录。也可以直接用 `--model Qwen/Qwen3-30B-A3B`
或对应 235B HF 仓库 ID，程序只按需获取 config/tokenizer，不调用 HF 权重下载器。
文件名与目录来自 [30B GGUF](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/tree/main)
和 [235B GGUF](https://huggingface.co/Qwen/Qwen3-235B-A22B-GGUF/tree/main)。

## 3. 验证

在仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=2 python -m pytest tests/models/test_marlin.py tests/models/test_gguf.py tests/moe -o addopts= -v
```

CPU 测试检查 GGUF 专家不进入 GPU state dict、两种模型的全部层结构、路由、stream
和配置约束。`test_gguf.py` 写入真实混合量化 GGUF，验证完整非专家 state dict、
QKV 数值和顺序、分片、缺失/重复张量、配置不匹配、绑定 embedding，以及不下载 HF 权重。
`test_marlin.py` 与 archive 原函数逐位比较 INT4 打包和 scale 排列，并检查全零分组和 padding。
其 CUDA 测试覆盖两种 Qwen3 的投影形状、LM head 输出维度、不同 batch size 和改变输入后的 graph replay。
其 CUDA 测试运行真实 QKV/norm/RoPE/FlashInfer/O 投影，比较 PyTorch SDPA 参考，覆盖
prefill、连续 decode、有 KV cache 的 extend prefill 与随后 decode。
`test_ktransformers_cuda.py` 在 Linux CUDA 上创建两层小型 Q8_0 GGUF，运行
真实 KT LLAMAFILE，比较 PyTorch 反量化参考计算，并测试非默认 stream 和 decode/prefill/
decode 的 batch 大小切换。GraphRunner 测试还覆盖多种捕获大小、padding、改变输入后的
重复 replay，以及 eager prefill 后的 graph replay，与 eager 输出比较。
该测试不需要下载完整模型。在 Linux CUDA 机器上 KT 导入失败
会导致测试失败；没有 Linux CUDA 时该项会跳过。

真实 CUDA/KT 测试需要在 Linux GPU 环境运行；Windows CPU 环境会跳过。
30B/235B 整模型生成尚未实机验证，不能把上述测试当作完整模型的运行或性能保证。

## 4. 启动 30B

```bash
conda activate minisgl-kt
CUDA_VISIBLE_DEVICES=2 python -m minisgl \
  --host 0.0.0.0 \
  --port 30000 \
  --model /data2/models/Qwen3-30B-A3B \
  --dtype bfloat16 \
  --tp-size 1 \
  --moe-backend kt \
  --kt-weight-path /data2/models/Qwen3-30B-A3B-GGUF \
  --kt-cpuinfer 128 \
  --kt-threadpool-count 2 \
  --kt-method LLAMAFILE \
  --attention-backend fi \
  --cuda-graph-max-bs 4 \
  --page-size 1 \
  --num-pages 2048 \
  --max-seq-len-override 2048 \
  --max-prefill-length 2048 \
  --max-running-requests 4
```

## 5. 启动 235B

先停止 30B 服务，再执行：

```bash
conda activate minisgl-kt
CUDA_VISIBLE_DEVICES=2 python -m minisgl \
  --host 0.0.0.0 \
  --port 30000 \
  --model /data2/models/Qwen3-235B-A22B \
  --dtype bfloat16 \
  --tp-size 1 \
  --moe-backend kt \
  --kt-weight-path /data2/models/Qwen3-235B-A22B-GGUF/Q4_K_M \
  --kt-cpuinfer 128 \
  --kt-threadpool-count 2 \
  --kt-method LLAMAFILE \
  --attention-backend fi \
  --cuda-graph-max-bs 4 \
  --page-size 1 \
  --num-pages 2048 \
  --max-seq-len-override 2048 \
  --max-prefill-length 2048 \
  --max-running-requests 4
```

128 线程、2 个 NUMA pool 沿用原命令，需匹配机器可用的物理核和 NUMA 拓扑（`lscpu`）。
若使用 `--max-running-requests 100`，可相应设置 `--cuda-graph-max-bs 100`；
捕获大小为 `1, 2, 4, 8, 12, 16, 24, ..., 96, 100`。上限为 24 时列表为
`[1, 2, 4, 8, 12, 16, 24]`；非标准上限也会加入列表，例如上限 20 会以 `16, 20` 结尾。
decode 按实际 batch size 选择能容纳它的最小 graph 并 padding，例如 9 个请求重放
bs=12，17 个请求重放 bs=24；超过捕获上限或处于 prefill 时走 eager。
首次启动会进行 warmup 和 graph capture，
需要额外的启动时间与显存；KV cache 仍由 `--num-pages` 和 `--page-size` 决定。
按模型形状估算，GPU 常驻权重从原 BF16 的约 2.9/14.9 GiB 降至 30B 约 1.21 GiB、
235B 约 4.88 GiB（含 INT4 权重、scale 以及 BF16 embedding/norm/router）。这不是运行峰值；
还需额外留出 KV cache、CUDA/FlashInfer 工作区、激活和 CUDA graph 空间。CPU RAM 需容纳
GGUF 专家及 KT NUMA 权重布局和工作区，不能仅按 GPU 显存判断 235B 能否启动。

与原 SGLang 命令的对应关系：

| 原参数 | Mini-SGLang |
| --- | --- |
| `python -m sglang.launch_server` | `python -m minisgl` |
| `--attention-backend flashinfer` | `--attention-backend fi` |
| `--disable-cuda-graph` | `--cuda-graph-max-bs 0`；KT 模式同样遵循此参数 |
| `--max-total-tokens 2048` | `--page-size 1 --num-pages 2048`，另有 1 个内部 dummy page |
| `--mem-fraction-static 0.5` | `--memory-ratio 0.5`；但显式 `--num-pages` 会覆盖自动预算，因此上面省略 |
| `--kt-num-gpu-experts 0` | 固定为 0，无需参数 |
| `--kt-max-deferred-experts-per-token 0` | 固定为 0，无需参数 |
| `--watchdog-timeout`、`--skip-server-warmup` | 没有对应参数；启用 graph 时会执行 graph warmup |

## 6. 请求验证

另开终端，选择与运行服务对应的模型路径：

```bash
curl --fail-with-body http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/data2/models/Qwen3-30B-A3B",
    "messages": [{"role": "user", "content": "请只回答：1+1等于几？ /no_think"}],
    "temperature": 0,
    "max_tokens": 32,
    "stream": false
  }'
```

235B 请求将 `model` 改为 `/data2/models/Qwen3-235B-A22B`。启动日志应出现
`KT: all experts on CPU; attention, norms, RoPE and router on GPU`、2048-token KV cache
和 `Start capturing CUDA graphs with sizes: [1, 2, 4]`（上述示例配置）。
只有显式关闭 graph 时才应出现 `CUDA graph is disabled.`。
成功请求才完成整模型 smoke test；测试跳过不算验收通过。

实现入口：`python/minisgl/moe/ktransformers.py`，权重过滤在
`python/minisgl/models/gguf.py`，接线与约束在 `python/minisgl/engine/engine.py`。
上游接口参考：[KT Direct Python API](https://github.com/kvcache-ai/ktransformers/blob/4882505c9a66a6784b3360a1b6ba9b23d53d0291/kt-kernel/README.md#direct-python-api-usage)。
固定提交的源码用 `gpu_experts_mask=None` 表示所有专家在 CPU；README 中的
`num_gpu_experts` 示例与该源码签名有差异，本实现按源码调用。
