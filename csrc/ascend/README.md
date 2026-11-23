# AscendC 算子（图模式用）

这里只放**本次 GE 动态图适配中真正被执行的算子**。判据是 GE 动态 profile 里出现的
`Spconv*` kernel（20k 点、cr=0.5，2388 个 kernel 的实测）：

| 算子目录 | kernel 名 | 每次迭代调用 | 作用 |
|---|---|---:|---|
| `spconv_sortdedup_v3/` | `SpconvSortDedupV3` | 15 | 坐标去重 + 排序，产出 `unique_coords` |
| `spconv_keyprep_v2/` | `SpconvKeyprepV2` / `SpconvKeyprepV2Prefix` | 45 + 45 | 邻域查询准备（偏移/前缀和） |
| `spconv_query/` | `SpconvQuery` | 45 | 邻域索引查询（替代 hash 表） |
| `spconv_gather_gemm/` | `SpconvGatherGemm` | 114 | 稀疏卷积主体：gather 邻域特征 + GEMM |

> 其它算子（`spconv_devox*`、`spconv_gather`、`spconv_fused`、`spconv_nb`、`spconv_dedup_v2`…）
> 在 **eager** 路径上更快，但**不在** GE 动态图的执行链里（图里 devox 等走 torch 侧），
> 因此没有随本仓库分发；需要时见 unum_ops 主库。

## 构建

前置：CANN 已 source（`ASCEND_HOME_PATH` 可用）、`python3 -c "import torch"` 可用（torch ≥ 2.14 需 C++20，
本目录的 `op_extension/CMakeLists.txt` 已设为 20）。

```bash
# 任选一个算子单独构建（互不依赖）
bash csrc/ascend/spconv_gather_gemm/build.sh

# 产物
#   csrc/ascend/<op>/op_extension/build/lib<op>_ops.so   ← PyTorch 直调绑定
#   csrc/ascend/<op>/build_out/custom_opp_*.run         ← vendor 内核包
```

## 使用（PyTorch 直调）

把 `UNUM_OPS_ROOT` 指向**本仓库根目录**即可，`torchsparse` 会按
`$UNUM_OPS_ROOT/csrc/ascend/<op>/op_extension/build/lib<op>_ops.so` 自动查找：

```bash
export UNUM_OPS_ROOT=$PWD        # 仓库根
python demo.py --cr 0.5
```

没构建（或构建失败/ABI 不匹配）会自动回退纯 torch，结果仍正确。

## 图模式下必须放开融合算子尺寸阈值

`torchsparse` 默认只对「小层」用 `spconv_gather_gemm`（其余回退 `gather+MatMul`）。
GE 动态图里那个回退会引入 device→host 同步，所以图模式要放开：

```bash
export TS_SGG_MAX_K=100000
export TS_SGG_MAX_COUT=512
export TS_GE_COMPAT=1
export UNUM_SD_NOSYNC=1
```

（这些只在图模式需要；eager 保持默认即可。）

## 实测（910B4 / torch 2.14 / torch_npu 2.14 / CANN 9.2）

| 场景 | 耗时 |
|---|---:|
| eager，纯 torch（无算子） | 288.6 ms |
| eager，**仅这 4 个算子** | 283.8 ms |
| eager，完整 unum_ops 库 | 73.7 ms |
| aclgraph（+完整库） | 27.5 ms |
| GE 静态分段（+完整库） | 15.2 ms |
| GE 动态（+完整库，任意形状） | 36.8 ms |

> 只装这 4 个算子时 **eager 几乎不提速**（283.8 vs 288.6）：默认启发式会把大层排除在
> `spconv_gather_gemm` 之外，eager 的大头是 devox/邻域聚合的 torch 实现。
> 想拿 eager 的 3.9× 需要完整算子库（`spconv_devox_v4` 等）。
> 这 4 个算子的价值在**图模式**：它们让 GE 动态图里所有 conv（114 次/迭代）都落在同一个
> 融合核里，省掉每次边界约 5.7 ms 的 host 同步。

## 已知约束

- `spconv_gather_gemm` 要求 `cin % 8 == 0`（否则走 torch 回退；GE 路径会用
  `_pad_cin_to_8` 把 stem 的 4 通道补齐到 8）。
- 动态形状靠「容量 + device 侧 n 标量」，kernel 的签名带 `n`；host tiling 读 device 值，
  不把行数读回 host（否则图里就是一次同步）。
