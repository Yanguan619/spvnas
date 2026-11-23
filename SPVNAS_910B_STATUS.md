# NPU 适配与调优状态（固化版）

> ⚠️ **时效声明**：本文是 2026-11 早期（torch 2.10 / CANN 9.2.1）的记录，其中的绝对耗时为**历史值**，与当前环境不可直接比较。
> 当前环境的用法与实测见 [ASCEND.md](ASCEND.md)；三模式定版数据见 `unum_ops` 仓库的 `docs/spvnas/GRAPH_MODE_MATRIX.md`。


> 目标仓库：`Yanguan619/spvnas`（`ascend` 分支，与上游 `mit-han-lab/spvnas` master 同一 commit）
> 环境：Atlas 910B4-1 ×8 / CANN 9.2.1 / torch 2.10.0 + torch_npu 2.10.0.post7 / triton 3.6.0
> 远端工作副本：`/data2/jev_work/spvnas-ascend`（容器 `jev-vllm32`，`/data2` 已挂载）

## 1. 一句话现状

纯 PyTorch 补齐了 `torchsparse`（原仓库唯一 CUDA 依赖），再逐步把热点换成 AscendC 算子。
**94625 体素全帧：cr=0.5 从 796 ms → 209.5 ms（3.8×），cr=1.0 从 806 ms → 217.5 ms，已快过论文的 GTX 1080Ti（317.1 ms）。**

## 2. 默认行为（不需要任何环境变量）

开箱即用的最优配置：

| 开关 | 默认 | 作用 |
|---|---|---|
| `TS_GATHER` | **on** | conv 的 gather 走 `SpconvGather` 算子 |
| `TS_NB` | **on** | 邻域查找走 `SpconvNb`（替代 int64 argsort + searchsorted，那两者被 CANN 丢在 AiCPU） |
| `TS_QUERY` | **on** | 邻域 key 计算 + 查表走 `SpconvQuery`（不再在 host 拼 `[M,27,4]`） |
| `TS_DEVOX` | **on** | 三线性反池化走 `SpconvDevox` |
| `TS_FUSED` | **off** | 全融合 join+gather 核（数值精确、单层 15×，但端到端慢 10%，见文档 §15） |
| `TS_GRAPH` | **off** | TorchAir 入图（torch 路径受益、算子路径受损；开它会自动关 `TS_GATHER`） |
| `TS_PROF` | off | 打点（`_tick`） |
| `TS_DYNAMIC` / `TS_MODE` / `TS_MEM` / `TS_DENSE` / `TS_UNIQ` / `TS_FULLGRAPH` | 见源码 | 细分调参 |

关掉某一个：`TS_DEVOX=0` 等；全部关掉即回到纯 torch 实现（数值等价，只是慢）。

## 3. 四个算子

| 算子 | 目录（`unum_ops/csrc/ascend/`） | 替代了什么 | 单算子收益 |
|---|---|---|---|
| `SpconvGather` | `spconv_gather/` | conv 的 27 邻域 gather（原 `index_select`） | ~15× |
| `SpconvNb` | `spconv_nb/` | `argsort + searchsorted`（AiCPU 16.7 ms/帧） | 6–39× |
| `SpconvQuery` | `spconv_query/` | host 拼 `[M,27,4]` + `pack_keys` + 二分 | 9–11× |
| `SpconvDevox` | `spconv_devox/` | `index_select` + 8 路乘加 + `index_add_` | 2.4–3.7× |

另有 `spconv_fused/`（全融合，默认关）。每个算子一个独立 vendor（Nnopbase 要求）。

## 4. 对源仓库的改动

只有 **1 个文件、2 行**（其余全是新增文件）：

```diff
--- a/core/models/utils.py
+++ b/core/models/utils.py
+from torchsparse.backend import unique_coords
-    sparse_coord = torch.unique(new_int_coord, dim=0)
+    sparse_coord = unique_coords(new_int_coord)
```

`torchsparse/` 与 `torch_scatter/` 是新增目录（纯 PyTorch 实现 + 算子接线）。

## 5. 怎么跑

```bash
cd /data2/jev_work/spvnas-ascend
PYTHONPATH=. python3 check_shim.py     # 算子级自检（对拍朴素实现）
PYTHONPATH=. python3 test_npu.py       # CPU↔NPU 精度 + 训练冒烟
PYTHONPATH=. python3 demo.py           # 单帧推理/过拟合 demo（出 BEV 图）
PYTHONPATH=. python3 ab.py             # 端到端计时（含 VOX/CR 环境变量）
PYTHONPATH=. python3 /data2/jev_work/bench4.py    # 四种模式对比
```

算子重新编译：
```bash
source /usr/local/Ascend/cann-9.2.1/set_env.sh
cd /data2/jev_work/unum_ops
bash csrc/ascend/build_torch_binding.sh csrc/ascend/<op>      # 编译
cd csrc/ascend/<op>/build_out && ./custom_opp_ubuntu_aarch64.run   # 装 vendor
```
卸载单个 vendor：`rm -rf $ASCEND_OP_HOME/opp/vendors/<op>`。

## 6. 精度基线（必须保持）

| 项 | 值 |
|---|---|
| CPU ↔ NPU 输出 | **max abs diff 7.451e-08**（logit 尺度 0.229） |
| argmax 一致率 | 100.0000% |
| `check_shim.py` | 全过（conv k3s1/k2s2/转置/devoxelize/hash/autograd） |
| 训练一步 | loss 1.0539，grad 路径自动回落 torch 实现 |

## 7. 性能（94625 体素全帧，中位数）

| cr | 参数量 | 现在 | 纯 torch 路径 | GTX 1080Ti（论文） |
|---|---|---|---|---|
| 0.25 | 1.36 M | **200.3 ms** | 434.7 ms | 124.3 ms（慢 1.61×） |
| 0.50 | 5.45 M | **209.5 ms** | 306.5 ms | 160.9 ms（慢 1.30×） |
| 1.00 | 21.78 M | **217.5 ms** | 462.0 ms | 317.1 ms（**快 1.46×**） |

20k 体素子集：**121 ms**（纯 torch 路径 265–320 ms）。

## 8. 已否决的路线（省得重试）

| 尝试 | 结果 |
|---|---|
| Triton 融合 conv | 0.18–0.27×（aclnn 的 DMA 描述符赢过逐元素索引 load） |
| fp16 推理 | 0.94×（算子数为王，减半搬运救不了） |
| K 维折叠 gather（9 offset 拼一次） | 孤立测快 3.6×，整模慢 2×（局部性反转） |
| 稠密表 > 128M 条目 | 无改善（level-0 盒子 6.09e9 装不下 24 GB） |
| int32 / float32 key 拆分排序 | `searchsorted` 无 dtype 红利；`Sort` 只占 0.5 ms/帧 |
| 池化复用去重 bool 表 | 164 → 4092 ms（`index_fill_` 带大索引是毒药） |
| `scatter_` / `index_select` 替高级索引 | 更慢或无变化 |
| 全融合 join+gather 算子 | 端到端慢 10%（见文档 §15） |
| TorchAir 整模入图 / `max-autotune` | 前者撞 H2D 限制，后者 `tbe` 缺失 |

## 9. 910B 上踩过的坑（写代码前先看）

1. **`GlobalTensor::SetValue` 逐元素写 GM 不落地**，`DataCopyPad` 才落地（踩了两次）。
2. **UB 上 `GetValue` 恒为 0**，标量只能从 GM 读。
3. **`aclDataType` 常量**：float32=0、int32=3、**int64=9**（8 是 uint32）。
4. **每个 op 一个独立 vendor 目录**，同一 vendor 加第二个 op 会 161002。
5. **host 的 `cumsum` 把 int32 提升成 int64**，绑定时按 int32 读会得到交错视图（典型症状：下标变成 18 亿这种"像字符串"的值）。
6. **核里没有 `floorf`**，用截断 + 负数修正。
7. **`SparseTensor(f, C, s, caches)` 位置参数错位**：第 4 个是 `spatial_range`，用关键字参数 `caches=`。
8. **int64 `sort`/`searchsorted` 被 CANN 丢到 AiCPU**；`float32` 才是 AIV 快路径，但 `searchsorted` 无此红利。
9. **分段探针必须两端 sync**，否则会把上游算子链的等待吸收进该段——我因此一度把 `kmap_build` 高估到 58 ms。

## 10. 文档索引

| 内容 | 位置 |
|---|---|
| 完整优化历程与全部实测数据 | `references/aicpu-and-idle-analysis.md`（§1–§17） |
| 910B4 逐算子 profile | `references/profile-910b4.md` |
| 原始 A/B 记录 | `references/measurements.md` |
| 可直接抄的代码片段 | `references/patterns.md` |
| 方法论与判据 | `SKILL.md` |
