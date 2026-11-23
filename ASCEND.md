# SPVNAS / SPVCNN on Ascend NPU

本分支（`ascend`）提供 **Ascend NPU 上的稀疏卷积后端 `torchsparse`（纯 PyTorch 实现）**，
让上游 SPVNAS/SPVCNN 代码可以在 **不编译任何 CUDA 扩展**、**不安装任何私有算子库** 的前提下跑起来。

- **零依赖**：默认只用 `torch` + `torch_npu`，`import torchsparse` 即可运行（本文件下面的数字都是这么测的）。
- **可选加速**：如果本机装了我们开源的 AscendC 算子库 `unum_ops`，设置一个环境变量就能自动接管热点算子，
  实测 eager 前向 **快 3.9×**。

---

## 1. 实测环境

| 项 | 值 |
|---|---|
| 设备 | Ascend 910B4（单卡） |
| torch | 2.14.0+cpu |
| torch_npu | 2.14.0 |
| CANN | 9.2.0（toolkit 路径按本机为准） |
| Python | 3.12 |

> torch 2.10 + CANN 9.2.1 也能跑纯 torch 路径；图模式（见 §5）建议 2.14 + 9.2.0。

## 2. 快速开始（零依赖）

```bash
# 1) 把本仓库的 torchsparse/ 与 torch_scatter/ 放到能 import 到的地方
#    （仓库根目录即可；或 pip install -e 一个带这两个包的目录）
export PYTHONPATH=$PWD:$PYTHONPATH

# 2) 需要 torch_npu
python -c "import torch, torch_npu; print(torch.npu.is_available())"

# 3) 跑一帧示例数据（仓库自带 assets/000000.bin）
python demo.py --cr 0.25
#   有官方权重可加上：  --load-model demo_spvcnn.pt
```

`demo.py` 会构造 SPVCNN、跑一帧、打印耗时；`--iters N` 可量多次迭代。

## 3. 精度等价性（零依赖 vs 加载算子库）

同一权重（`demo_spvcnn.pt`，SemanticKITTI 单帧 20,000 点，cr=0.5）：

| 配置 | logits 求和 | argmax 一致率 | `max｜d｜` |
|---|---|---:|---:|
| 纯 torch（默认，零依赖） | -596121.25 | 100% | — |
| 加载 unum_ops | -596121.25 | 100% | 1.29e-05 |

`1.29e-05` 是 fp32 累加顺序差异（logits 幅值约 14.6，相对误差 ~9e-7），
**argmax 100% 一致**，判别结果完全相同。

## 4. AscendC 算子（自带 4 个，可选）

本仓库自带 `csrc/ascend/`，里面是**本次 GE 动态图适配中真正被执行的 4 个算子**
（依据：GE 动态 profile 里的 `Spconv*` kernel 计数）：

| 算子 | kernel | 每迭代调用 |
|---|---|---:|
| `spconv_sortdedup_v3` | `SpconvSortDedupV3` | 15 |
| `spconv_keyprep_v2` | `SpconvKeyprepV2(+Prefix)` | 45 + 45 |
| `spconv_query` | `SpconvQuery` | 45 |
| `spconv_gather_gemm` | `SpconvGatherGemm` | 114 |

构建（CANN 已 source、torch ≥ 2.14 需 C++20，CMakeLists 已设好）：

```bash
bash csrc/ascend/spconv_gather_gemm/build.sh          # 每个算子独立构建
export UNUM_OPS_ROOT=$PWD                              # 指向本仓库根即可被自动发现
python demo.py --cr 0.5
```

**不构建也完全可用**（自动回退纯 torch）。

实测（同权重同帧，20,000 点，cr=0.5，10 次中位数）：

| 配置 | eager 前向 | 说明 |
|---|---:|---|
| 纯 torch（默认） | 288.6 ms | 零依赖 |
| **仅自带这 4 个算子** | 283.8 ms | **几乎不提速**（见下） |
| 完整 unum_ops 库 | 73.7 ms | 3.9×，大头是 `spconv_devox_v4` 等 |

> 只装这 4 个时 eager 几乎不提速：默认启发式会把大部分层排除在 `spconv_gather_gemm` 之外，
> 而 eager 的时间大头在 devox / 邻域聚合的 torch 实现。这 4 个算子的价值在**图模式**——
> 它们让 GE 动态图里 114 次 conv 全部落在同一融合核里，省掉每处约 5.7 ms 的 host 同步。
>
> 图模式下需放开阈值（否则大部分层仍回退）：
> `export TS_SGG_MAX_K=100000 TS_SGG_MAX_COUT=512 TS_GE_COMPAT=1 UNUM_SD_NOSYNC=1`
>
> 想拿 eager 的 3.9× 需要完整算子库（`spconv_devox_v4`、`spconv_gather` 等），
> 见 `unum_ops` 主库；换 torch 版本时 `.so` 需按对应 torch 重编（ABI 不匹配会回退，结果仍正确）。

## 4b. 两个免费的性能开关（实测）

```bash
# 一级流水任务队列：20k 档 -32%，94k 档 -8%
export TASK_QUEUE_ENABLE=2
# 内存池扩展段（长跑/大帧更稳）
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

> `TASK_QUEUE_ENABLE=2` 与"入图"互斥（入图时不要开）。更多调优结论见 [TUNING.md](TUNING.md)。

## 5. 图模式（可选，需 torch ≥ 2.14）

本后端同时支持 `torch.compile` 的三种图模式（aclgraph / GE 静态 / GE 动态）。要点：

```bash
export TS_GE_COMPAT=1          # GE/TorchAir 兼容改写（amin/amax、>> 等不在其 ATen 白名单）
export UNUM_SD_NOSYNC=1        # 去重算子跳过 device→host 读回（入图必需）
# aclgraph
GM_MODE=npe python demo.py --cr 0.5
# GE 动态
GM_MODE=ge GM_DYN=1 python demo.py --cr 0.5
```

实测（20k 点、cr=0.5，单位 ms）：

| 模式 | 耗时 | 精度 |
|---|---:|---|
| eager（+unum_ops） | 73.7 | — |
| aclgraph（npugraph_ex） | 27.5 | `max｜d｜`=1.5e-08 |
| GE 静态（分段） | **15.2** | 逐段 0 |
| GE 动态（每形状冻结一次） | 36.8 | 5.96e-08 |

图模式细节与坑位见我们另一份文档（`unum_ops` 仓库的 `ge.md` / `docs/spvnas/GRAPH_MODE_MATRIX.md`）。
纯 torch 路径**不需要**这些环境变量。

最小可用写法（图模式**必须先 import `ascend_meta`**，否则 `torch.compile` 会报
`Operator does not support running with fake tensors`）：

```python
import ascend_meta                      # 注册 4 个算子的 Meta（幂等）
import torchair
m = torch.compile(m, backend=torchair.get_npu_backend(), dynamic=True)   # GE 动态
# m = torch.compile(m, backend=torchair.get_npu_backend(), dynamic=False) # GE 静态
# aclgraph 用 torchair 的 npugraph_ex backend
```

## 6. 目录说明

| 路径 | 说明 |
|---|---|
| `torchsparse/` | Ascend 稀疏卷积后端（纯 torch；可选接 unum_ops 算子） |
| `torch_scatter/` | 极简 `scatter_mean` 实现，替代需要编译的 `torch_scatter` |
| `core/`、`configs/`、`train.py`、`evaluate.py` | 上游 SPVNAS 代码（未改动，除 `core/models/utils.py` 的哈希查询改写） |
| `demo.py` / `bench.py` | 免数据集的一帧演示与计时 |

## 7. 已知限制

1. **只支持推理/Eval 用途的语义**：后端按 TorchSparse++（torchsparse 2.1）语义实现
   （coords 为当前层的 `(batch, x, y, z)`、下采样取 `floor(coord/stride)`、转置卷积复用正向 conv 的 kernel map）。
2. **无 CUDA 路径**：本后端是纯 torch + NPU 实现，不上 GPU；CPU 也能跑（很慢）。
3. **大帧性能**：纯 torch 路径是「op 数量受限」的（一帧约 800 个小 op），
   这也是为什么加载 unum_ops 能到 3.9×；不加载时请预期 3–4 倍差距。
4. `demo_spvcnn.pt`（22 MB 权重）未随仓库分发，跑精度复现时自行准备。

> ⚠️ **「任意形状」的准确口径**：上表 6 档（20k/94k × cr 0.25/0.5/1.0）都实测跑通且结果正确，
> 但**每个形状需要一次 `TS_FG=record/replay` 冻结**（模型里有几处数据相关常量要读回 host，
> 占位后图才编得出来），所以实际能力是「**每形状冻结一次**」，
> **不是**「一次编译、任意 N、不重录」。
> 真正的零重录需要把 conv 链切成容量化算子（输出定长 CAP + device 侧 `n`），
> 该链路目前卡在 `spconv_keyprep_v3`（`start`/`cnt` 输出全 0），见 `csrc/ascend/README.md §7`。

## 容量化 + GE 动态（Stage D：一张图吃任意点数）

`TS_CAP=<CAP>`（coords 固定 CAP 行、真实行数放在 device 标量 `n`）配合
`torch.compile(backend=torchair.get_npu_backend(), dynamic=True)`，910B4 实测：

```bash
TS_GE_COMPAT=1 TS_CAP=20480 UNUM_OPS_ROOT=... \
  python3 stageD_test.py
```

| 帧点数 | max\|d\| vs eager | argmax 一致 |
|---|---|---|
| 20000 | 4.768e-06 | 100.0000% |
| 12000 | 5.007e-06 | 100.0000% |

- 20k 首跑 103.0 s（含编译），**12k 首跑 0.1 s → 没有重编**：一张图吃两种点数；
- `TORCH_LOGS=graph_breaks` 下**零 graph break**（整模进一张 GE 图）；
- 不再需要 `TS_FG` 按形状冻结。

### 四条硬约束（都是实测踩出来的）

1. **容量 .so 必须在入图之前加载**。在被 trace 的区域里调用
   `torch.ops.load_library` 会让 GE 报 `ERR03005 GRAPH internal error`。
   `torchsparse/nn/functional/__init__.py` 在 `TS_CAP>0` 时于导入期预加载全部算子。
2. **`keyprep_v3` 必须显式传固定 nbk**（≥ 4·CAP 的 2 的幂）。kernel 默认按**真实** n
   自动算，而 Meta 只能按 CAP 算；两者不一致会让 `cumsum(cnt)` 在未初始化的桶上累加。
3. **容量模式下禁止 host 侧建图**（`_compact_from_nb` 里的 `nonzero()/tolist()`）。
   被入图后会烘死，换一帧静默算错（实测 argmax 掉到 97.9%）。转置卷积因此强制走
   device-only 快路径，且**不允许静默回退**（回退直接报错）。
4. **`n` 必须有兜底**：`caches.nmaps[stride]`。某层 `n=None` 会让 `point_to_voxel`
   退回 `scatter_mean`，行数被 `max(index)+1` 截断（实测 17900 ≠ CAP 20480），
   进而让转置卷积的快路径条件失效。

### 与旧结论的关系

之前"GE 动态"是**每形状冻结一次**（`TS_FG=record/replay`）；容量化把 Python 侧那几处
host 读回（去重行数、包围盒乘积、数据相关 nonzero）全部搬到 device 之后，才真正做到
**一张图 + 任意 N ≤ CAP**。

### 性能（同协议：warmup 3 + 10 次中位数；910B4；同一帧）

单位 ms/帧。**只列已验证精度的数据**。

| 帧点数 | 固定行数 eager (TS_CAP=0) | 容量 eager | 容量 GE 动态（单档 20480） | 容量 GE 动态（分档） |
|---|---|---|---|---|
| 20000 | 63.20 | 70.52 | 52.39 | 52.22（档 20480） |
| 12000 | 59.71 | 69.20 | 45.26 | **38.80**（档 16384） |
| 6000  | 58.41 | 69.15 | 40.17 | **31.66**（档 8192） |

- 分档相对单档：1.00× / 1.17× / **1.27×**；
- 分档相对固定行数 eager：1.21× / 1.54× / **1.85×**；
- 精度（vs 同档 eager）：4.768e-06 / 5.007e-06 / 3.815e-06，argmax 100%；
- 编译一次性成本：每档 42–102 s（3 档合计约 190 s），之后任意 N ≤ 档位直接跑。

**容量化固定成本**：容量 eager 比固定行数 eager 慢 7–11 ms，且**与 N 基本无关**
（6k 也要 69 ms）——因为每层都按 CAP 行计算，补齐行是白算。这正是需要分档的原因。

用法：

```python
import torchsparse.nn.functional as F
F.set_cap(F.cap_for(n_pts))      # 选最小能装下的档位（默认 8192/16384/20480）
# TS_CAP_BUCKETS="8192,16384,20480" 可覆盖档位表
```

⚠️ **已知缺陷**：`dynamic=False`（GE 静态）在容量路径上**结果错**（`max|d| ≈ 10`），
不要使用它的计时数据（早期文档里 GE 静态的 49.48 / 42.36 / 35.97 / 22.42 ms 均属此类，
已作废）；分档必须搭配 `dynamic=True`。
