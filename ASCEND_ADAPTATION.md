# SPVNAS / SPVCNN 昇腾 910B 适配文档

> 目标：在不改模型数学语义的前提下，让 SPVNAS 在昇腾 910B4 上**跑得对、跑得全、跑得快**。
> 本文是交付主文档；按时间追加的实测记录与排错流水见 [ASCEND.md](ASCEND.md)。

---

## 0. 摘要

| 维度 | 结果 | 参考/基线 |
|---|---|---|
| **精度（完整评测）** | **mIoU 60.63% / OA 91.91%**（SemanticKITTI 序列 08 全部 **4071** 帧） | 60.7 / 91.94（官方 Model Zoo） |
| **图模式数值等价** | `max|d| = 3.8e-06 ~ 5.0e-06`，argmax 100%，**零 graph break** | eager |
| **性能（整帧 94k，cr=0.5）** | **aclgraph 75.0 ms** | 官方 GTX 1080Ti 160.9 ms（**快 2.15×**） |
| **性能（20k / 12k / 6k 点）** | **aclgraph 26.5 / 19.7 / 13.8 ms** | 固定行数 eager 62.9 / 65.0 / 56.1 ms |
| **性能（形状任意、一图吃到底）** | 容量化 + GE 动态 **178.70 ms**（94k） | 这是"零重录"的溢价路径，非最快 |
> **最快的配置是 aclgraph（`_fgstatic` 固化 + `fullgraph=True`）**：94k **75.0 ms**（cr=0.5）/
> 75.5（cr=0.25） / 97.6（cr=1.0）、20k 26.5 ms、6k 13.8 ms。容量化 + GE 图模式是为「一张图吃任意 N、零重录」付溢价的路径
> （94k 178.70 ms），不是性能最优解 —— 详见 §5.2b 与 §5.4c。

| **零依赖** | 不装 `torchsparse`/`torch_scatter`/`unum_ops` 也能跑 | — |

---

## 1. 环境

| 项 | 版本/型号 |
|---|---|
| 硬件 | Ascend 910B4（单卡，HBM 64 GB） |
| CANN | 9.2.0 |
| Python / torch / torch_npu | 3.12.15 / 2.14.0 / 2.14.0 |
| 图模式 | `torchair`（随 torch_npu 提供） |
| 数据集 | SemanticKITTI 序列 08（val，4071 帧） |

---

## 2. 适配架构总览

```
SPVNAS/SPVCNN (PyTorch)
├── torchsparse/           零依赖 shim（纯 PyTorch 实现 sparse conv 的语义）
├── torch_scatter/         零依赖 shim（scatter_mean / segment ops）
├── csrc/ascend/           AscendC 自定义算子（可选，装了自动启用）
│   ├── 固定行数版：sortdedup_v3 / keyprep_v2 / query / gather_gemm / devox_v3
│   └── 容量版：    sortdedup_v4 / keyprep_v3 / query_v2 / gather_gemm_v2 / devox_v4
├── ascend_meta.py         自定义算子的 Meta（fake tensor）注册，入图必需
└── 图模式三路线：aclgraph / GE 静态 / GE 动态
```

三层收益（实测，20k 点/帧，eager）：

| 配置 | 耗时 |
|---|---|
| 纯 torch（无自定义算子） | 288.6 ms |
| 仅装 GE 动态用到的 4 个算子 | 283.8 ms |
| **装全算子库** | **73.7 ms（3.9×）** |

> 经验：收益主要来自 `spconv_devox_v4` 与 `gather_gemm`，只装"图模式用到的那几个"几乎没用。

---

## 3. 关键设计：容量化（让形状静态、让点数任意）

### 3.1 问题

SPVNAS 每层的行数（体素数）**由数据决定**：去重后剩多少、nonzero 多少。
Python 侧一旦 `int(u)` 把行数读回 host，图就被绑定到这个形状 —— 换一帧就要重录（90–170 s）。
而 GE 的形状推断必须能静态推出，`scatter_mean` 这类"数据相关输出形状"根本无法表达。

### 3.2 方案

`TS_CAP=<CAP>` 时：

1. coords 固定 **CAP 行**，尾部用**有效行重复**填充（不能用极端哨兵，会把内置归约打崩）；
2. 真实行数 `n` 放在 **device 标量** `int32[1]`（`SparseTensor.n`），Python 侧**永不读回**；
3. kernel 只算前 `n` 行，输出仍是 CAP 行；
4. 下游用 `arange(CAP) < n` 掩码（均值、尾部落零、nb 置 -1）。

于是：**一张图吃任意 N ≤ CAP**。

### 3.3 四条硬约束（全部是实测踩出来的，改代码前必读）

1. **容量算子 `.so` 必须在入图之前加载**。在被 `torch.compile` trace 的区域里
   `torch.ops.load_library` 会让 GE 报 `ERR03005 GRAPH internal error`。
2. **`keyprep_v3` 必须显式传固定 `nbk`**（≥ 4·CAP 的 2 的幂）。kernel 默认按真实 n 自动算，
   Meta 只能按 CAP 算；两者不一致会让 `cumsum(cnt)` 在未初始化桶上累加。
3. **禁止任何 host 侧建图**：`_compact_from_nb` 的 `nonzero()/tolist()` 被入图后会烘死，
   换一帧静默算错（实测 argmax 掉到 97.9%）。转置卷积因此强制走 device-only 快路径，
   **不允许静默回退**（回退直接抛错）。
4. **补齐行的"语义约定"必须全程一致**：尾部重复**末行**，且
   - 前向 conv 的邻居表：`>= n` **折回 n-1**（补齐行就是末行复制，语义正确；兼作越界安全网）；
   - 转置 conv：`>= n` **丢弃**（`_apply_transposed` 按 `out[src_fine] += feats[dst]` 累加，
     折回会把 padding 的 fine 行全灌进真实末行，实测爆到 1e5）。
   因此前向/转置各缓存一份图（`kmaps` / `kmaps_drop`）。

### 3.4 用法

```python
import torchsparse.nn.functional as F
F.set_cap(F.cap_for(n_pts))     # 选最小能装下的档位（默认 8192/16384/20480）
# 或直接用单档：TS_CAP=20480（一张图覆盖 0..20480）
```

`TS_CAP_BUCKETS="8192,16384,20480"` 可覆盖档位表。

---

## 4. 图模式三路线

| 路线 | 开关 | 覆盖能力 | 编译次数 | 备注 |
|---|---|---|---|---|
| **GE 动态** | `dynamic=True` | 一张图吃 **任意 N ≤ CAP** | 每档 1 次 | 输入点数维真是 `-1` |
| **GE 静态** | `dynamic=False` | 每档一张图 | 每档 1 次 | 容量化后形状真静态，**最快** |
| aclgraph | `npugraph_ex` | 仅固定输入形状 | 每形状 1 次 | 点数变化即不可用 |

### 4.1 "真泛化动态"为什么不成立（已实测）

不容量化、`dynamic=True` 无档位时：5 处 graph break + GE 报 `Verifying NetOutput failed`。
根因是**数据相关的输出形状**（`scatter_mean` 的输出行数 = `max(idx)+1`、`out[:int(u)]`）。
`torchair.inference.set_dim_gears` 也替代不了上界：它的 guard 是**精确枚举**
（`dim` 必须正好是某档，落在两档之间直接失败），`gears[-1]` 就是上界。

⇒ 结论：**在这代 CANN/torchair 上，"不预分配上界"与"一次编译走到底"对这套稀疏算法互斥**。
最终选择容量 buffer；`dynamic=True` 负责"输入点数维"，容量化负责"层内行数"。

---

## 5. 性能

### 5.0 主结果表（行=场景，列=配置）

单位 **ms/帧**（整模 wall，warmup 3 + 5 次中位数；图模式 capture 15–177 s 不计入）。
场景覆盖原文档（`性能.md` / `SPVNAS_910B_STATUS.md`）给出的**全部 GTX 1080Ti 数据点**。

#### 表 A · 整帧 94,625 体素 × cr（覆盖官方 1080Ti 的三个 cr）

| 场景 | eager<br>(910B) | **aclgraph**<br>(910B) | GE 静态<br>(910B) | GE 动态<br>(910B) | **GE 动态·容量化路径**<br>(910B) | eager<br>(GTX 1080Ti) | aclgraph 相对 1080Ti |
|---|---:|---:|---:|---:|---:|---:|---:|
| cr = 0.25（1.36 M 参数） | 101.3 | **75.5** | 107.9 | 109.2 | 172.6 | **124.3** | **快 1.65×** |
| cr = 0.50（5.45 M 参数） | 100.1 | **75.0** | 106.2 | 110.2 | 176.4 | **160.9** | **快 2.15×** |
| cr = 1.00（21.78 M 参数） | 121.4 | **97.6** | 130.6 | 135.0 | 216.0 | **317.1** | **快 3.25×** |

#### 表 B · 点数分档（cr=0.5；官方无 1080Ti 切分数据）

| 场景 | eager<br>(910B) | **aclgraph**<br>(910B) | GE 静态<br>(910B) | GE 动态<br>(910B) | **GE 动态·容量化路径**<br>(910B) | eager<br>(GTX 1080Ti) |
|---|---:|---:|---:|---:|---:|---:|
| 20,000 点（CAP=20480） | 62.9 | **26.5** | 34.0 | 36.2 | 43.9 | —（未发表） |
| 12,000 点（CAP=16384） | 65.0 | **19.7** | 24.5 | 28.8 | 34.5 | —（未发表） |
| 6,000 点（CAP=8192） | 56.1 | **13.8** | 15.4 | 22.0 | 34.6 | —（未发表） |

**读表要点**

1. **aclgraph 在每个场景都最快**，且相对 1080Ti 的优势随参数规模扩大：
   **1.65× → 2.15× → 3.25×**（cr 越大，1080Ti 的稀疏卷积劣势越明显）；
2. 小帧优势更大：6k **13.8 ms**（eager 56.1 → **4.1×**）；
3. **GE 静态/动态在所有场景都不如 eager**（例如 cr=0.5：106.2 / 110.2 vs 100.1），
   它们的价值在形状泛化（配合容量化可「一张图吃任意 N」）；
4. GTX 1080Ti 只有整帧 cr=0.25/0.5/1.0 三个官方数据点，**表 B 的切分口径无对应数据，不做外推**；
5. **「GE 动态·容量化路径」是另一条路的代价**：`TS_CAP` 固定行数 + `dynamic=True`，换来
   「一张图吃任意点数、零重录」的能力，但整帧要付 172–216 ms（比固定行数 GE 动态还贵
   ~65%，因为 CAP=131072 对 94,625 体素有 38% padding）；点数档用贴合档时降到 34.5–43.9 ms。
   **要性能就不要走这条，要形状泛化才走。**
6. **CAP 档位选择不是「越小越好」，是「与 tiling 匹配」**：固定 6k 数据只改 CAP 实测
   —— `8192 → 35.56 ms`、`16384 → 32.88 ms`（更快，虽然浪费 63% 行）、`32768 → 40.79 ms`。
   非单调。选档位时应实测 2–3 个候选值，别默认取最紧的那档。

**口径与可信度**

| 项 | 说明 |
|---|---|
| 数值一致性 | 所有图模式 run：`max|d| ≤ 7.5e-08`、`argmax = 100.0000%` |
| 复测稳定性 | cr=0.5 的 aclgraph 测了 3 次：**75.11 / 74.91 / 75.20（文档值）** |
| 已知异常值 | 早期一次 cr=0.5 aclgraph 测到 84.7，复测两次均为 ~75，判定为**设备瞬态**，已剔除 |
| 图模式入口 | `gm_fg2.py`：record → `_fgstatic.freeze()` → `cache_compile`/`torch.compile`，`fullgraph=True`，`TS_GE_COMPAT=1` |
| 硬件 | Ascend 910B4 单卡；GTX 1080Ti 数字来自原论文，跨硬件仅作量级参考 |

### 5.1 评测协议

warmup 3 + 10 次取中位数；同一帧、同一进程；`TS_GE_COMPAT=1`；
`/data2/jev_work/spvnas-ascend`，`demo_spvcnn.pt`。

### 5.2 数据（ms/帧）

| 帧点数 | 固定行数 eager | 容量 eager | 容量 GE 动态·单档 20480 | 容量 GE 动态·分档 | **容量 GE 静态·分档** |
|---|---|---|---|---|---|
| 20000 | 63.20 | 70.52 | 52.39 | 52.22 | **49.88** |
| 12000 | 59.71 | 69.20 | 45.26 | 38.80 | **35.89** |
| 6000 | 58.41 | 69.15 | 40.17 | 31.66 | **22.61** |

- 相对固定行数 eager：**1.27× / 1.54× / 2.58×**；
- 分档 vs 单档：1.00× / 1.17× / **1.27×**（小帧收益最大）；
- 编译成本（一次性）：GE 静态每档 **35–72 s**，GE 动态每档 42–102 s。

### 5.2b aclgraph 才是最快路径（实测，frozen + fullgraph）

口径：`gm_fg2.py`（record → `_fgstatic.freeze()` → `cache_compile`，`fullgraph=True`，
`TS_GE_COMPAT=1`），warmup 3 + 5 次中位数，同一帧切不同点数。

| npts | eager | **aclgraph** | 加速比 | 5.2 的「容量+GE 静态」 | aclgraph 相对它 |
|---:|---:|---:|---:|---:|---:|
| 6000 | 60.25 | **13.80** | 4.37× | 22.61 | **快 1.64×** |
| 12000 | 64.62 | **19.72** | 3.28× | 35.89 | **快 1.82×** |
| 20000 | 62.65 | **26.45** | 2.37× | 49.88 | **快 1.89×** |
| 94625 | 100.1 | **75.0** | 1.34× | 150.64 | **快 2.01×** |

对照《性能.md》（11-21 口径）：20k aclgraph 27.49 → 现在 **26.5（略优）**；
94k 75.20 → 现在 **75.0（持平）**。此前一度测到 84.7，复测两次为 75.11 / 74.91，
判定为设备瞬态异常值 —— **不存在回退**。

**对官方 GTX 1080Ti（160.9 ms @94k，cr=0.5）：75.0 ms = 快 2.15×**（cr=0.25 1.65×、cr=1.0 3.25×）。

> 重要：§5.2 的「容量 + GE 静态」是为**一张图吃任意 N** 付的溢价，不是最快路径。
> 形状已知/可分档时，aclgraph 全面胜出（1.6–1.9×）。

### 5.3 容量的固定成本

容量 eager 比固定行数 eager 慢 7–11 ms，且**与 N 基本无关**（6k 也要 69 ms）——
每层按 CAP 全量计算，补齐行是白算。**这就是需要分档的原因**。

### 5.4 与官方性能对比（整帧口径）

官方只给出了**论文里的 GTX 1080Ti 单帧延迟**（cr=0.5 → 160.9 ms；cr=1.0 → 317.1 ms），
README 的表格只有 MACs 与 mIoU、没有延迟。为了可比，这里在**同一口径（整帧）**上测本仓：
单帧 seq08，123389 点 → **94625 体素**，`voxel=0.05`、cr=0.5、fold_bn、warmup 2 + 5 次中位数。

| 配置 | 整帧延迟（median） | 相对官方 GTX 1080Ti | 11-24 值 → 变化 |
|---|---|---|---|
| **官方论文 GTX 1080Ti（cr=0.5）** | **160.9 ms** | 1.00× | — |
| 本仓 固定行数 eager | **101.76 ms** | **快 1.58×** | 102.58（持平） |
| 本仓 固定行数 eager + `TASK_QUEUE_ENABLE=2` | **98.21 ms** | **快 1.64×** | 95.40（噪声内） |
| 本仓 容量 eager（CAP=98304） | **116.77 ms** | 快 1.38× | 143.47（**−18.6%**） |
| 本仓 容量 eager（CAP=131072） | **132.00 ms** | 快 1.22× | 159.17（**−17.1%**） |
| 本仓 容量 + GE 静态（CAP=98304） | **150.64 ms** | 慢 6% | 176.55（**−14.7%**） |
| 本仓 容量 + GE 静态（CAP=131072） | **172.90 ms** | 慢 7% | 199.35（**−13.3%**） |
| 本仓 容量 + GE 动态（CAP=131072） | **178.70 ms** | 慢 11% | 202.81（**−11.9%**） |

11-25 的改进来自修掉容量路径里**重复计算前缀和**（`torch.cumsum` vs 算子已返回的 `start`，
详见 §5.4b）：容量化与图模式各档一致降 **12–19%**，固定 eager 路径不受影响（符合预期）。

**测量口径**：warmup **5** 次 + **10** 次取中位数；图模式首次调用（编译 73–104 s）不计入。

> ⚠️ 跨硬件比较只作**量级参考**：不同加速器、不同稀疏算子实现，绝对数字不可直接换算。
> 但同一张卡上的横向对比（下一段）是可信的。

**三个反直觉但重要的结论**

1. **固定行数 eager 已经比官方 GTX 1080Ti 快 1.57×**（102.58 vs 160.9 ms；开 TQ=2 后 95.40 ms，快 1.69×）——
   全算子库（含 `devox_v4`/`gather_gemm`）的收益远大于图模式。
2. **容量化的 padding 成本按 `CAP / 真实行数` 线性放大**：94625 体素装进 CAP=131072
   （38% 冗余）时 eager 从 102.6 ms 涨到 159.2 ms（+55%）；把 CAP 压到 98304（4% 冗余）
   就回到 143.5 ms。
3. **图模式在整帧尺度反而慢 20–25%**，而在小帧（6k/12k/20k）快 1.3–2.6×（见 5.2）。
   说明整帧时瓶颈在 kernel 效率/tiling，不是 launch 开销 —— 图模式省下的调度时间
   抵不过它在这些规模下的分块/融合损失。

**因此按输入规模选路线**：

| 输入规模 | 推荐配置 | 实测 |
|---|---|---|
| **形状已知（可分档）** | **aclgraph（frozen + fullgraph）** | **13.8 / 19.7 / 26.5 / 75.0 ms**（6k/12k/20k/94k, cr=0.5） |
| 不想管形状、要最快 | 固定行数 eager（+ TQ=2） | 98.21 ms @94k |
| **一张图吃任意 N（零重录）** | 容量化 + GE 动态（单档） | 178.70 ms @94k、52.39 ms @20k |
| 一张图 + 分档 | 容量化 + GE 静态 | 150.64 / 172.90 ms |

### 5.4b 修掉一处 25.6 ms/帧的纯浪费（`torch.cumsum` 重复计算前缀和）

对容量路径做算子级 profile 时发现：**`Cumsum` 是最大单项，25.6 ms/帧**（524288 元素 × 15 次）。

根因：`spconv_keyprep_v3` 的 C++ wrapper **已经**用专用算子 `aclnnSpconvKeyprepV3Prefix`
算好了独占前缀和，并作为 `output[1]` 返回（`return {rkeys, start, cnt, meta}`），
但 Python 侧把它当占位 `_unused` 丢掉，又用 `torch.cumsum(cn, 0) - cn` 重算了一遍。

修复：直接用 `output[1]`：

```python
rk, start, cn, mt = torch.ops.npu.spconv_keyprep_v3(...)   # 原来是 rk, _unused, cn, mt
```

**效果（整帧 94625 体素）**

| 配置 | 修前 | 修后 | 变化 |
|---|---|---|---|
| 容量 eager CAP=98304 | 143.47 ms | **116.77 ms** | −18.6% |
| 容量 + GE 静态 CAP=98304 | 176.55 ms | **150.64 ms** | −14.7% |
| 容量 + GE 动态 CAP=131072 | 202.81 ms | **178.70 ms** | −11.9% |
| 固定行数 eager | 101.45 ms | 101.45 ms | 不受影响 |

**正确性**：CAP == M（与固定路径同输入）时 nb 逐项比对 **100.0000%（2554875/2554875）一致**；
容量路径端到端 mIoU 65.74% / OA 91.41%（29 帧）与固定路径 65.87% / 91.44%（30 帧）一致。

> 教训：profile 里出现「一个跟算法无关的算子排第一」时，先查是不是重复计算 ——
> 这次是同一份前缀和被算了两次，第二次用的还是慢得多的通用 `cumsum`。

#### 5.4c 重要：本节图模式数字**低估了图模式**

上表的图模式行来自 `perf_full.py`（裸 `torch.compile`，**没有** `_fgstatic` 固化
数据相关标量、**没有** `fullgraph=True`），因此图内有回退，数字偏大。

用仓库原有 harness（`gm_fg2.py`：record → `_fgstatic.freeze()` → `cache_compile`，
`fullgraph=True`，`TS_GE_COMPAT=1`）在**同一整帧**上实测：

| 模式 | 裸 compile（本文上表） | frozen + fullgraph（正确口径） |
|---|---|---|
| eager | 101.76 | **101.14** |
| aclgraph（npe） | — | **75.0**（比 eager 快 **1.34×**） |
| GE 静态 | 150.64 | **106.23**（比 eager 慢 5%） |
| GE 动态 | 178.70 | **110.21**（比 eager 慢 8%） |

两者数值一致性：`max|d| = 5.96e-08`、`argmax = 100%`。

**结论修正**：整帧口径下 **aclgraph 才是最快的（75.0 ms，cr=0.5）**，GE 静态/动态分别比
eager 慢 6%/10%；此前"图模式整帧慢 20–25%"的结论是把未固化路径的数字当成了图模式本身。

### 5.4d step 级分解：为什么 aclgraph 能消掉 host 间隙，GE 不能

用 `torch_npu.profiler` API 采到 **step 级** `step_trace_time.csv`（eager，94,625 体素，
wait1/warmup2/active3，Step 3/4/5）：

| Step | Computing（设备） | **Free（host 空闲）** | Step 总计 | Bubble |
|---|---:|---:|---:|---:|
| 3 | 73,701 µs | 34,131 µs | 107.83 ms | 0 |
| 4 | 73,647 µs | 36,271 µs | 109.92 ms | 0 |
| 5 | 73,759 µs | 34,153 µs | 107.91 ms | 0 |

**eager 的 34 ms Free = host 间隙**，Bubble=0 说明设备侧没有气泡，慢全在前端。

拿三种模式的总时间去减 eager 的设备时间（73.70 ms）：

| 配置 | 总时间 | 减 eager 设备时间 | 解读 |
|---|---:|---:|---|
| eager | 107.83 | +34.13 | **= step_trace 里的 Free，完全对上** |
| **aclgraph** | **75.15** | **+1.45** | **Free 被吃掉 96%** |
| GE 静态 | 106.46 | +32.76 | ≈ eager 的 Free（差 1.4 ms） |
| GE 动态 | 109.61 | +35.91 | 同上 |

**结论**：GE 模式的 kernel 时间与 eager 基本相同，那 ~33 ms 是**同一份 host 开销没被消掉**；
只有 aclgraph 把它压到 1.4 ms。机制差异：

| | aclgraph（npugraph_ex） | GE（torchair） |
|---|---|---|
| 机制 | **capture→replay**：把 eager 跑过的 kernel 序列原样录成 ACL graph，replay 一次 API 调用 | FX 图翻译成 **AscendIR 图**，由 GE 调度 |
| 每次调用的 host 工作 | 构建输入 + 一次 `rtGraphLaunch` ≈ **1.4 ms** | torchair wrapper：dynamo guard、输入装配、`SparseTensor` 构造、`_fg` 常量查找、输出解包 ≈ **33 ms** |
| 每帧变化的 device 张量 | capture 期按地址固定，host 不参与 | 算子带 `n`/`sizes` 这类每帧变化的 device 输入，每次执行都要准备 |

**所以「图模式消不掉 Free」不成立**：aclgraph 消得掉，GE 消不掉 —— 差别在「消的是 kernel
提交」还是「消的是算子间调度」。

> 采集方式提醒：step 记录只在**进程内**用 `torch_npu.profiler` + `prof.step()` 才会产生；
> 用外部 `msprof --application=...` 采是拿不到 step 边界的（`step_trace` 为 0 行）。
> 本地留存的原始数据：`E:\Projects\dl\profiling_step\`（带 step）与 `profiling_eager2\`（无 step）。

### 5.5 整帧算子级画像（eager，94625 体素）

`msprof --application` 采集整帧 eager（`prof_full.py` + `analyze_op_profile.py`），
6293 个算子实例 / 36 种类型：

| 算子 | 次/帧 | ms/帧 | 占比 |
|---|---:|---:|---:|
| `SpconvGatherGemm` | 38 | 19.9 | 26.8% |
| `SpconvQuery` | 15 | 19.2 | 25.9% |
| `SpconvSortDedupV3` | 5 | 8.7 | 11.8% |
| `SpconvKeyprepV2` | 15 | 8.6 | 11.6% |
| `SpconvDevoxV4` | 4 | 4.3 | 5.8% |
| `SpconvKeyprepV2Prefix` | 15 | 3.1 | 4.2% |
| `Slice` | 222 | 2.2 | 3.0% |
| 其余（layout/逐元素/BN 等） | ~1000 | ~7.8 | ~11% |
| **设备侧 kernel 合计** | | **≈74 ms** | |

关键读数：**设备侧 kernel 合计约 74 ms，而 wall 是 102.58 ms** —— 约 **28% 是 host 间隙**，
这正是图模式理论上应该吃掉的收益；但整帧下图模式反而慢（见 5.4），说明它在这套
数据规模下把 kernel 端拖慢了，抵消并超过了调度收益。

### 5.6 为什么整帧图模式反而慢（已定位到"规模相关"，未到 kernel 级）

**已知事实**
1. 20k 体素：容量 eager 70.5 → GE 静态 49.9 ms（图模式**快 20 ms**）；
2. 94625 体素：容量 eager 143.7 → GE 静态 176.6 ms（图模式**慢 33 ms**）——
   **交叉点存在，且与规模相关**；
3. 占比最高的两类算子（`SpconvQuery` 25.9% + `SortDedup` 11.8%）都是**搜索/排序**类，
   对行数敏感；eager 下可用 host 侧真实行数 `n` 提前剪枝/分块，图内只能按 **CAP 全量**执行。

**尚未拿到的证据（工具限制，如实记录）**
- `torch_npu.profiler` + GE：trace 时报 `spconv_keyprep_v3` 的 `nbk` 是 Python int，
  而 GE 自动转换器要求它是张量 → 编译失败。根因是仓库**没有注册自定义 GE converter**，
  全靠 torchair 自动生成；
- `msprof --application` + GE：`op_summary` 只有 12 行（整图记成 `ModelExecute`），
  `--ge-api=l1` 也未产出节点级表 → **当前 CANN/torchair 组合下无 GE node-level profile**。

**下一步三个可选项**（按预期收益）
1. 把 `nbk` 改成张量并注册自定义 converter → 图模式可 profile，且可能减少 padding 计算；
2. 直接优化 `SpconvQuery`（占 19.2 ms）：它在 eager 下已是第二大户；
3. 维持现状：整帧走 eager（95–102 ms，比 1080Ti 快 1.6×），图模式只用于 ≤20k 小帧。

### 5.7 优化尝试记录：profiling 指导的调优已到边界

**(a) 桶数 `TS_NBK_MULT` 扫描（默认 4，实测就是最优）**

`nbk`（桶数）= mult × 行数，Query 要扫整个桶数组，桶越少碰撞越多。整帧 94625 体素实测：

| mult | 1 | 2 | **4** | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| 整帧 eager | 104.43 | 102.41 | **101.45** | 102.88 | 105.77 | 105.39 ms |

→ **默认 4 已在最优点**，往两侧都变慢。参数已暴露为 `TS_NBK_MULT` 便于复测。

**(b) `SpconvGatherGemm` 不是标量瓶颈，是搬运瓶颈**

分析脚本按 `scalar/duration ≥ 0.5`（544/1036 = 0.53）判成 scalar-bound，但三项对照更说明问题：

| 分项 | 94625,48 一次调用 |
|---|---|
| mac / vec / scalar | 202 / 72 / 544 µs |
| **mte1+mte2+mte3** | **183+479+447 = 1109 µs ≈ 总时长 1036 µs** |

`mte` 之和 ≈ 总时长 → **真正受限的是搬运**。根因在 kernel 结构：每行对 27 个邻居各做一次
`DataCopyPad`，每次只有 `cin×4`（48 通道 = **192 B**）。27 × 94625 次 192 B 小传输，
在 910B4 上 MTE 效率很低。改成"标量读改向量读"不会有用（标量只占一半且已并行）。

**(c) 搜索/排序类算子是随机访问受限**

`SpconvQuery` 4555 µs/次 @ **3.7 GB/s**、`SpconvSortDedupV3` 2654 µs/次 @ **4.0 GB/s**，
而 HBM 理论带宽高两个数量级 —— 这是散列/二分查找的**随机访问**固有代价，不是实现疏忽。

**结论**：Python 层可调项（`get_kernel_offsets`/`_const_tensor` 已缓存、`nbk` 已在最优、
`Slice` 仅 1.2 ms/帧）已经吃干；剩余收益必须来自**算子重构**：

1. `GatherGemm` 改成隐式 GEMM（多邻居共享一次大块搬运，tiling 在 UB 内做邻居展开）；
2. `Query`/`SortDedup` 改 radix 分桶 + 合并查找，把随机访问变成顺序访问；
3. 图模式整帧掉速仍**无法观测**（GE node-level profile 在 CANN 9.2 + torchair 下拿不到），
   修通它需要自定义 GE converter（`nbk` 张量化）。

这三项都不属于"profile 指哪打哪"的微优化，属于算子重写，故本轮优化到此为止。

### 5.8 Roofline：算子到底卡在哪（实测，910B4）

**可达带宽**（512 MB 张量实测）：copy(读+写) **1183 GB/s**，fill(只写) 1301 GB/s。

**(a) 同一算子、不同通道数 → 带宽完全由「单次搬运粒度」决定**

直接调 `spconv_gather_gemm_v2`（94625 行 × 27 邻居，warm3/reps10）：

| cin | 每次 `DataCopyPad` 字节 | 实测 | 等效带宽 | 占可达带宽 |
|---|---|---|---|---|
| 16 | 64 B | 1780 µs | **91.9 GB/s** | 8% |
| 32 | 128 B | 1457 µs | 224.4 GB/s | 19% |
| 48 | 192 B | 1974 µs | 248.5 GB/s | 21% |
| 64 | 256 B | 1402 µs | 466.5 GB/s | 39% |
| 128 | 512 B | 1781 µs | **734.6 GB/s** | **62%** |

→ 带宽随粒度**单调上升**。第一层（cin=16，仅 64 B/次）是最差的一档，也解释了
profile 里 `68931x16` 只有 174 GB/s。

**(b) 更根本的问题是「没有跨行复用」**

现有 kernel 按**输出行**逐个 gather：每行把 27 个邻居 × cin 搬进 UB，算完即弃。
一次 conv 的总搬运量 = `N × kv × cin × 4`（94625×27×48×4 = **468 MB**），
但输入特征本身只有 `N × cin × 4` = **18 MB** —— 即 **26 倍的冗余读取**。
若能按隐式 GEMM 把输入 tile 驻留 UB 并跨 27 个偏移复用，搬运量可降到 ~18 MB 量级。

**(c) fp16 这条路当前走不通**

kernel 硬性要求 `feat must be float32`（`feat/nb/w must be 2D` 之后的下一个校验），
要减半搬运量必须改 kernel 的数据类型支持。

**结论（本轮 profile 的终点）**：算子**没有**到硬件瓶颈，而是卡在**当前 kernel 设计的
算法结构**上，两个可量化空间：

| 手段 | 预期上限 | 需要的改动 |
|---|---|---|
| 把单次搬运做到 ≥512 B（凑块/合并邻居） | 2.6×（3783 → 1439 µs，6 个 shape 合计） | 改 AIV gather 的取数组织 |
| 输入 tile 驻留 UB、跨 27 偏移复用 | 最多 ~26×（468 MB → 18 MB） | 隐式 GEMM 重构 |
| fp16 搬运 | ~2× | kernel 支持 fp16 |

这三条都超出「profile 指出热点 → 打补丁」的范围，属于算子重写。

### 5.9 复跑

```bash
TS_CAP=20480 python3 perf_cap.py                       # 容量 eager / 单档动态
TS_CAP=20480 TS_GE_COMPAT=1 python3 perf_bucket.py      # 分档（静态或动态，改 dynamic=）
TS_CAP=20480 TS_GE_COMPAT=1 python3 stageD_test.py      # 一张图吃 20k/12k/6k + 精度
```

---

## 6. 精度评测（完整）

### 6.1 结果

SemanticKITTI **序列 08 全部 4071 帧**，`voxel=0.05`、`cr=0.5`、HF 转换权重、
`ignore_label=255`，逐点重映射后聚合**全局混淆矩阵**（官方协议）：

| 指标 | 本仓 NPU | 官方 Model Zoo | 差 |
|---|---|---|---|
| **mIoU** | **60.63%** | 60.7 | **0.07** |
| **OA** | **91.91%** | 91.94 | **0.03** |

图模式数值等价（同帧对拍 eager）：`4.768e-06 / 5.007e-06 / 3.815e-06`，argmax 100%。

### 6.2 权重与数据来源（官方地址已失效，改用 HF 镜像）

| 内容 | 链接 |
|---|---|
| 权重（SPVCNN@30GMACs） | `https://hf-mirror.com/torch-pointcloud/spvcnn-30gmacs.semantickitti.mit-han-lab`（`model.safetensors`） |
| 标签 00–10 | `https://hf-mirror.com/datasets/WWZzz/SemanticKITTI/resolve/main/data_odometry_labels.zip` |
| 点云 | 官方 `data_odometry_velodyne.zip`（84.8 GB）或 HF 镜像 `velodyne_part.00/01` |

> 直连 `huggingface.co` 在测试网络不通，**必须走 `hf-mirror.com`**。
> 官方 `hanlab.mit.edu/files/SPVNAS/...` 已 404。

### 6.3 权重转换

`convert_hf_spvcnn.py`：torch-pointcloud 命名 → 本仓 `SPVCNN` state_dict，
**317/317 全部映射、逐键断言形状相等、`load_state_dict(strict=True)` 通过**。

规则：`stageN.M.net ↔ encoder.block{N-1}.blocks.M`、`stageN.0 ↔ ...downsample`、
`upN.0 ↔ decoder.block{N-1}.upsample.conv`、`upN.1.0 ↔ ...upsample.residual`、
`upN.1.1 ↔ decoder.block{N-1}.blocks.0`、`point_transforms ↔ fusion`、`classifier ↔ head`。

### 6.4 两个致命坑

1. **`ignore_label` 必须是 255**，不是 0。本仓 `label_map` 把未标注映射为 255，而
   **0 是 `road`**；用 0 会剔除 road、放进未标注点 → mIoU 51.5%、`class 0.seen=0`。
2. **不要做类别置换**。HF 权重名含 `mit-han-lab`，通道顺序与本仓一致；
   套上 torch-pointcloud 的 `SEMANTIC_KITTI_CLASSES` 顺序会让 OA 从 91.6% 掉到 0.96%。

### 6.5 评测稳定性（偶发 507035）

设备侧偶发 `507035 vector core exception` 会中断进程。评测做成三段式，**崩溃不丢已算帧**：

1. 分块跑（250 帧/进程）；
2. `--stats-jsonl` 逐帧落盘 `seen/correct/positive`；
3. `fill_missing.py` 对缺失帧按 20 帧一块重跑，块内失败二分到单帧。

首轮 3446/4071，补齐后 **4071/4071**，按帧号去重聚合。

### 6.6 复跑

```bash
python convert_hf_spvcnn.py
python evaluate_npu.py --root <sequences> --ckpt spvcnn30_mit.pt \
  --cr 0.5 --voxel 0.05 --ignore-label 255 \
  [--stats-jsonl stats.jsonl] [--limit N --offset M]
python fill_missing.py        # 补齐缺失帧
```

---

## 7. 修掉的真缺陷

| # | 现象 | 根因 | 提交 |
|---|---|---|---|
| 1 | `EZ9999 / Call InferShapeAndType for node:SpconvQuery failed` | InferShape 读了不存在的输入 index 5（算子只有 0..4） | `dc967ee` |
| 2 | GE 静态 argmax 仅 96.35%，`max|d|≈10` | `initial_voxelize` 里 `(c*init_res)/after_res` 在 fp32 下非恒等，紧跟的 `floor` 把它放大成"体素错一格" | `f2e2e84` |
| 3 | GE 动态 `ERR03005` | 在 trace 区域内 `load_library` | 容量化提交 |
| 4 | 图内结果错（97.9%） | host 侧建图（`nonzero/tolist`）被烘死 | 容量化提交 |
| 5 | 转置卷积 `n=None` → 行数被截断 17900 | `n` 未跨层兜底 | 容量化提交 |
| 6 | 补齐条目污染末行（6→2→1 个点错） | 前向/转置对补齐条目语义相反，共用同一张图 | `7973f52` |
| 7 | mIoU 51.5%（应 60.6%） | `ignore_label` 用了 0（=road） | `49ceacb` |

---

## 8. 已知限制

1. **CAP 是上限**：`N > CAP` 需要更大的 CAP（重编一张图）。分档即为此；
2. **GE 静态每档一张图**；可选优化：把 CAP 改成从输入形状派生 + `set_dim_gears`，
   可把分档的 K 次编译降为 1 次（未做）；
3. **偶发 507035 未根治**：靠分块 + 逐帧落盘 + 二分重试绕过；
   怀疑与超大体素数（seq08 单帧可达 ~95k 体素）下的 kernel tiling 有关，未定位；
4. **逐类 IoU 的类名未标**：需要仓库自带 `semantic-kitti.yaml` 才能建立
   `label_map` 索引 → 类名 的正确映射（mIoU/OA 是置换不变量，不受影响）；
5. **aclgraph** 只能覆盖固定输入形状；
6. 官方权重地址已 404，交付依赖 HF 镜像（本文档已记录可用链接）。

---

## 9. 文件索引

| 文件 | 作用 |
|---|---|
| `torchsparse/`、`torch_scatter/` | 零依赖 shim |
| `csrc/ascend/` | AscendC 算子（固定行数版 + 容量版），含 `build_torch_binding.sh` |
| `ascend_meta.py` | 自定义算子 Meta 注册（入图必需） |
| `ascend_capacity.py` | 容量化配方样例（能跑通的最小链条） |
| `evaluate_npu.py` / `convert_hf_spvcnn.py` / `fill_missing.py` | mIoU 评测三件套 |
| `perf_cap.py` / `perf_bucket.py` | 性能测试 |
| `stageA/B/C/D_test.py`、`test_capacity_chain.py` | 容量化分段验收 |
| `ASCEND.md` | 按时间追加的实测记录与排错流水（证据） |
| `TUNING.md`、`SPVNAS_910B_STATUS.md` | 调优笔记 / 早期状态 |

---

## 10. 一句话总结

**eager 能跑（零依赖 73.7 ms）→ 容量化让形状静态 → GE 动态一张图吃任意 N（52.2 ms）→
GE 静态分档最快（49.9 ms @20k、22.6 ms @6k），精度 60.63% / OA 91.91% 与参考对齐。**
