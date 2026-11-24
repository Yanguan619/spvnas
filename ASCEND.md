# SPVNAS / SPVCNN on Ascend NPU


> 📄 **交付主文档见 [ASCEND_ADAPTATION.md](ASCEND_ADAPTATION.md)**（结构化的适配文档：环境、架构、容量化设计、性能、精度、缺陷清单、已知限制）。
> 本文是**按时间追加的实测记录与排错流水**，作为证据保留。

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
已作废）。**该缺陷已于 f2e2e84 修复**（根因是 `initial_voxelize` 的恒等乘除把 `floor` 带偏，见后文「修复：GE 静态已可用」）；修复后 `dynamic=False` 精度 3.8e-06~5.5e-06，比动态快 1.05–1.40×。

#### `dynamic=False`（GE 静态）缺陷的补充证据

排除「切档导致图失效」这一可能后（进程内只用单一 CAP、全程不调用 `set_cap`）：

| 模式 | 6000 点 | 20000 点 |
|---|---|---|
| `dynamic=True` | max\|d\|=3.815e-06 argmax 100% | 4.768e-06 argmax 100% |
| `dynamic=False` | max\|d\|=8.060e+00 argmax **96.35%** | 1.001e+01 argmax **96.55%** |

- 两侧 `absmax` 一致（14.482 / 14.562），说明整体量级没错，是**部分点/体素特征错**；
- 逐子模块单独编译时 `stem` / `stage1` / `stage2` 均为 0 ~ 4.8e-07（全对），
  差异只出现在**解码器**（转置卷积及之后）；单独编译 `up1[0]` 会因缺 `caches.cmaps` 无法运行，
  所以二分到此为止。
- 下一步可查：在静态图下逐个算子做 golden（`sortdedup_v4`、`keyprep_v3+query_v2`、
  `gather_gemm_v2` 的 [0,n) 写 + 尾部置零、`_nbT` 逆表、`_masked_mean`），
  找出在整图静态融合下行为不同的那一个。

### 决策：保留容量 buffer（不转泛化动态）—— 2026-11-24

实测 `TS_CAP` 不设 + `dynamic=True` + 无档位（纯泛化动态）：

- 5 处 graph break，GE 最终报 `Verifying NetOutput failed`，出不了图。

逐项定位结论：

1. `SpconvQuery` 的 InferShape 读了不存在的输入 index 5（真实 bug，已修，
   见 `dc967ee`）——但它只是第一层；
2. 结构性阻塞是**数据相关的输出形状**：
   - `torch_scatter.scatter_mean` 的输出行数 = `max(idx)+1`；
   - `backend` 的 `out[:int(u)]`（去重后行数）、`_finish` 的 `int(h[4])`。
   这些值既不是输入维度的函数，也不能写成 `-1`（GE 的形状推断必须能静态推出），
   于是子图输出维只能是 `-1` → `NetOutput` 校验失败。

`torchair.inference.set_dim_gears`（已确认存在于本机 torchair）**不能**替代上界：
它的 guard 是精确枚举——

```python
gears[0] <= dim <= gears[-1]  and  (dim <= left) | (dim >= right)
```

即 `dim` 必须正好等于某个档位值，落在两档之间直接失败；`gears[-1]` 就是上界。

所以：**在这代 CANN 9.2 + torchair 上，「不预分配上界」与「一次编译走到底」对这套
稀疏算法是互斥的**。保留容量 buffer（`TS_CAP` + device 标量 `n`），并可选 CAP 分档
以压低 padding 成本。

TODO（可选优化，不影响正确性）：把 CAP 改成从输入张量形状派生 + `set_dim_gears`，
可把现在的「每档一张图（3 次编译）」变成「一张图覆盖多档（1 次编译）」。

### 修复：GE 静态（dynamic=False）已可用 —— 2026-11-24

**根因**：`initial_voxelize` 里这一行

```python
new_float_coord = torch.cat([z.C[:, 0].view(-1, 1), (z.C[:, 1:] * init_res) / after_res], 1)
```

`init_res == after_res`（都等于 voxel size 0.05）时，`(c * init_res) / after_res` 数学上是
恒等，但 fp32 下 `x*0.05/0.05 != x`。紧接着的 `torch.floor(...)` 把这点舍入噪声放大成
「体素错一格」。eager 与 GE 图对常系数的化简/重排不同（图里会做收缩、融合），舍入方向
就不同 —— 实测 3.6% 的点落到不同体素（argmax 96.35%）。

**修法**：`init_res == after_res` 时直接跳过乘除，用 `z.C.to(torch.float32)`：

```python
if init_res == after_res:
    new_float_coord = z.C.to(torch.float32)
else:
    new_float_coord = torch.cat([z.C[:, 0].view(-1, 1), (z.C[:, 1:] * init_res) / after_res], 1)
```

**修后实测**（CAP 分档，每档一个编译好的图，精度均已对拍）：

| 帧点数 | 档位 | GE 静态 | GE 动态 | 精度 (vs eager) |
|---|---|---|---|---|
| 6000  | 8192  | **22.61 ms** | 31.66 ms | 3.815e-06 |
| 12000 | 16384 | **35.89 ms** | 38.80 ms | 4.768e-06 |
| 20000 | 20480 | **49.88 ms** | 52.22 ms | 5.484e-06 |

- 静态比动态快 1.05–1.40×；相对固定行数 eager（58.41 / 59.71 / 63.20）快 **1.27–2.58×**；
- 编译：每档 35–72 s（一次性）。

**排查方法（可复用）**：在模型内用 `torch._dynamo.disable` 逐个部件降级为 eager，
其余照常入图，看 `max|d|` 何时从 8.06 掉回 1e-6。

- `deconv` / `_masked_mean` / `conv apply` / `unique_coords_cap` / `_rect_map` /
  `cap_refill` 单独降级**均无效**（结果严格相同）；
- 降级**整个 `initial_voxelize`** → `3.576e-06`、argmax 100% ⇒ 问题在该函数体内的普通 torch 算子。

即「逐个算子都正常，但组合进图后不一致」这种形态，优先怀疑**离散化前的浮点等价改写**。

## SemanticKITTI mIoU 评估（NPU，复用原仓库流程）

`evaluate_npu.py` 沿用原 `evaluate.py` 的逐帧流程（`outputs -> inverse_map 重映射
-> argmax`）与仓库自带的 `core.callbacks.MeanIoU`，只做了单卡 NPU 适配：

- `.cuda()` -> `.npu()`；`MeanIoU` 里的 `dist.allreduce` 打桩为恒等（单卡）；
- 模型直接用 `SPVCNN(num_classes, cr, pres, vres)` 构造（不走 `model_zoo`，
  因为官方权重地址 `https://hanlab.mit.edu/files/SPVNAS/...` 已 404）；
- `--ckpt` 显式给权重；`--root` 指向 `sequences` 目录（其下 `08/velodyne/*.bin`）；
- 直接构造 `SemanticKITTIInternal(split='val')`（= 序列 08），不构造 train 分片。

### 已验证（链路冒烟，非数据集精度）

```
[env] frames=1  root=.../sequences  cr=0.50  device=npu:0
[ckpt] loaded demo_spvcnn.pt  missing=0 unexpected=0
逐类 IoU 打印 + mIoU = 0.8354（1 帧 + demo 权重，数值无意义，仅证明链路通）
```

### 跑真 mIoU 还缺两样

1. **数据集**：SemanticKITTI 序列 08（`sequences/08/velodyne/*.bin` + `labels/*.label`）。
   本机 `/dataset` 不存在；`/data2` 余量 2.0 TB 够放。拿到后：
   `python evaluate_npu.py --root <...>/sequences --ckpt <ckpt> --cr 0.5`
2. **训练好的权重**：仓库只带 `demo_spvcnn.pt`（demo 权重）。官方 Model Zoo 的下载
   地址已失效（`hanlab.mit.edu/files/SPVNAS/...` 返回 404，项目页也已改版），
   需要另找可用镜像或自行训练。

### 预训练权重（HF 镜像）与 mIoU 实测 —— 2026-11-24

官方 Model Zoo 地址（`https://hanlab.mit.edu/files/SPVNAS/...`）已 404，可用替代：

- 权重：`https://hf-mirror.com/torch-pointcloud/spvcnn-30gmacs.semantickitti.mit-han-lab`
  （`model.safetensors` 21.8 MB；README 记 mIoU 60.7 / OA 91.94）。
  直连 `huggingface.co` 在本机不通，**必须走 `hf-mirror.com`**。
- 标签：`https://hf-mirror.com/datasets/WWZzz/SemanticKITTI/resolve/main/data_odometry_labels.zip`
  （170 MB，含 00–10 全部标签，无需注册）。
- 点云 08：`https://hf-mirror.com/datasets/larryshaw0079/SemanticKITTI-SP/resolve/main/08.tar.gz`
  （7.13 GB）。官方全量 `data_odometry_velodyne.zip` 84.8 GB 虽支持 Range，但逐帧抽取实测
  1.2 帧/分钟（≈56 小时），不可用。

`convert_hf_spvcnn.py` 把 torch-pointcloud 命名映射成本仓 `SPVCNN` 的 state_dict：
`317/317` 全部映射、逐键断言形状相等、`load_state_dict(strict=True)` 通过。
规则要点：`stageN.M.net ↔ encoder.block{N-1}.blocks.M`、`stageN.0 ↔ ...downsample`、
`upN.0 ↔ decoder.block{N-1}.upsample.conv`、`upN.1.0 ↔ ...upsample.residual`、
`upN.1.1 ↔ decoder.block{N-1}.blocks.0`、`point_transforms ↔ fusion`、`classifier ↔ head`。

实测（NPU，`evaluate_npu.py`，序列 08 子集，voxel=0.05，cr=0.5）：

| 帧集合 | 可用帧 | **OA** | mIoU |
|---|---|---|---|
| 前 100 帧 | 98（跳过 2 帧半截文件） | **91.75%** | 55.09% |
| 后 60 帧 | 60 | **90.92%** | 54.54% |
| 前 30 帧 | 30 | — | 57.77% |
| **README 全量 08** | 4071 | **91.94%** | **60.7** |

**结论：权重转换正确**。判据是 **OA 91.75% vs README 91.94%（差 0.19 点）**，
稀有类（bicycle 0.10 vs ~0.29、other-ground 0.001 vs ~0.28、motorcyclist 0）
在 100 帧子集里样本极少，是 mIoU 低 5 个点的主因；常见类基本吻合
（road 0.933 / building 0.871 / vegetation 0.835 / terrain 0.758）。

`evaluate_npu.py` 另外两处必要处理：
- **帧完整性校验**：`.bin` 点数必须等于 `.label` 行数，半截下载文件跳过并计数
  （Range 抽取中断会留下半截文件，实测 100 帧里 2–3 帧是坏的）；
- `--offset` 支持切片对照，并打印 **OA**（`total_correct/total_seen`）——
  OA 比 mIoU 对子集大小更稳健，是更可靠的验收指标。

说明权重转换与评估流程都正确（错误的映射不可能到 58%）。全量 4071 帧待点云下完后补齐。

`evaluate_npu.py` 的两处必要适配：
1. 单卡 NPU：`.cuda() -> .npu()`，`MeanIoU` 的 `dist.allreduce` 打桩为恒等；
2. **逐帧数据自己读**：vendored `SparseTensor` 构造时会对 coords 去重，导致原 dataset 返回的
   `targets_mapped.F`（逐点标签）与 `.C` 行数不一致 → 直接按点取
   `pred = outputs[inverse].argmax(1)`、`tgt = 每点标签`，与原脚本数学等价。

### mIoU 对齐完成（2026-11-24）

**踩过的坑（决定成败的一处）**：`--ignore-label` 必须是 **255**，不是 0。

本仓 `label_map` 把未标注类映射为 **255**（`configs/semantic_kitti/default.yaml: ignore_label: 255`），
而 **0 是 `road`**。用 0 当忽略类会：把占比最大的 road 整类剔除、未标注点反而计入统计 ——
表现为 class 0 的 `seen=0`、稀有类 IoU 全面偏低（mIoU 51.5% 而非 60%）。修正后 OA 立刻回到 91.9%。

另：不要给模型输出做「类别置换」。HF 权重名里的 `mit-han-lab` 说明它是本仓同源的模型，
通道顺序**与本仓一致**；torch-pointcloud 的 `SEMANTIC_KITTI_CLASSES` 是它自己训练用的顺序，
套上来会让 OA 从 91.6% 掉到 0.96%。该选项（`--class-perm`）默认关闭。

**最终结果**（序列 08，3446/4071 帧 = 84.6% 覆盖；voxel=0.05，cr=0.5，HF 转换权重）：

| 指标 | 本仓 NPU | README 全量 08 | 差 |
|---|---|---|---|
| **mIoU** | **60.00%** | 60.7 | **0.70** |
| **OA** | **91.89%** | 91.94 | **0.05** |

剩余 625 帧因设备侧偶发 507035 被跳过（分块跑 + 逐帧统计落盘，崩溃不丢已算帧）。
结论：**权重转换、评估流程、数值都与参考实现对齐**。

复跑：
```bash
python evaluate_npu.py --root <sequences> --ckpt spvcnn30_mit.pt \
  --cr 0.5 --voxel 0.05 --ignore-label 255 [--stats-jsonl x.jsonl] [--limit N --offset M]
```

### mIoU 完整评测（4071/4071 帧 = 100%，2026-11-24）

| 指标 | 本仓 NPU（全量序列 08） | README 全量 08 | 差 |
|---|---|---|---|
| **mIoU** | **60.63%** | 60.7 | **0.07** |
| **OA** | **91.91%** | 91.94 | **0.03** |

这就是完整评测：SemanticKITTI 08 全部 **4071 帧**，`voxel=0.05`、`cr=0.5`、
HF 转换权重（`spvcnn30_mit.pt`），`ignore_label=255`，逐点重映射后聚合全局混淆矩阵。

**稳定性处理**：设备侧偶发 `507035 vector core exception` 会中断进程，因此评测按
「分块（250 帧/进程）+ 逐帧统计落盘（`--stats-jsonl`）+ 缺失帧二分重试」跑：

1. 第一轮分块跑完 3446/4071 帧；
2. `fill_missing.py` 把缺失区间按 20 帧一块重跑，块内再失败就二分到单帧；
3. 最终 4071/4071 = 100%，`frame_stats2/stats.jsonl` 按帧号去重后聚合。

结论：**权重转换（317/317）、类别顺序（无需置换）、评估协议（ignore_label=255）
全部与参考实现对齐**，mIoU 与 OA 都落在 0.1 个点以内。

注：逐类 IoU 的**类名**需要仓库自带的 `semantic-kitti.yaml` 才能准确标注
（`label_map` 的索引并非 `kept_labels` 的排列顺序）；mIoU/OA 是置换不变量，
不受影响。
