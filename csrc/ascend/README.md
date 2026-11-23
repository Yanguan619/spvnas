# AscendC 算子

本目录有两组算子：**固定行数版**（GE 动态图实际执行的那一套）与**容量化版**
（让「任意 N、一张图」成立的前提）。两者的差别只有一个：**输出行数怎么来**。

| | 固定行数版 | 容量化版 |
|---|---|---|
| 输出行数 | 由 kernel 在 device 上算出 `U`，Python 侧 `out[:int(u)]` | **固定为容量 CAP**，真实行数放 device 标量 `n` |
| Python 侧读回 | 有（`int(u)`，入图必须冻结） | **没有**（用 `arange(CAP) < n` 做掩码） |
| 换形状 | 需要重录/重编（图绑定到形状） | 一张图吃任意 `N ≤ CAP` |
| 现状 | ✅ 已接入 shim，GE 动态跑的就是它 | ✅ 源码+Meta 就位、**构建通过**；shim 尚未接线 |

## 1. 固定行数版（GE 动态实际执行）

依据：GE 动态 profile 里的 `Spconv*` kernel 计数（20k 点、2388 个 kernel）。

| 算子 | kernel | 每迭代调用 | 作用 |
|---|---|---:|---|
| `spconv_sortdedup_v3/` | `SpconvSortDedupV3` | 15 | 坐标去重排序 |
| `spconv_keyprep_v2/` | `SpconvKeyprepV2(+Prefix)` | 45+45 | 邻域查询准备 |
| `spconv_query/` | `SpconvQuery` | 45 | 邻域索引查询 |
| `spconv_gather_gemm/` | `SpconvGatherGemm` | 114 | 稀疏卷积 gather+GEMM |

## 2. 容量化版（任意 N 一张图的前提）

| 算子 | 签名（比固定版多什么） | 对应固定版 |
|---|---|---|
| `spconv_sortdedup_v4/` | `(coords, n)` → CAP 行 + device `n` | `sortdedup_v3` |
| `spconv_keyprep_v3/` | `(in, out, stride, offs, nbk, sizes)` | `keyprep_v2` |
| `spconv_query_v2/` | `(q, rkeys, start, cnt, meta, n)` | `query` |
| `spconv_gather_gemm_v2/` | `(feat, nb, w, n)` | `gather_gemm` |
| `spconv_devox_v4/` | `(feat, idx, w, coords, n)` | `devox*` |

Meta 已在仓库根的 `ascend_meta.py` 里注册（`_meta_*_v4/_v3/_v2`）。

### ⚠️ 接线现状（重要）

这 5 个算子目前**只是源码 + Meta 就位、构建通过**，`torchsparse` 里仍走固定行数版，
所以：

- 图模式下换点数仍需要 `TS_FG=record/replay` 重新冻结（≈ 重录一次）；
- 要做到「一次编译、任意 N、不重录」，需要把下面 5 处接上容量版：

| 位置 | 现状 | 要改成 |
|---|---|---|
| `backend._sortdedup_kernel` | `sortdedup_v3(coords)` | `sortdedup_v4(coords, n_in)`，保留 `n` 为 device 标量 |
| `backend.unique_coords` 末行 | `out[: int(u)]`（host 读回） | 返回 CAP 全量 + `n`，不再切片 |
| `backend._query_args` | `keyprep_v2` / `query` | `keyprep_v3(..., sizes)` / `query_v2(..., n)` |
| `functional._apply_forward_eager`（SGG 分支） | `gather_gemm(feat, nb, w)` | 判 CAP：`gather_gemm_v2(feat, nb, w, n)` |
| `functional` 的 devox 调用 | `devox_v3/v4(feat, idx, w, coords)` | `devox_v4(feat, idx, w, coords, n)` |

配套：所有 `_fg.const(...)` 的 host 读回（`counts` / `sel` / `total`）要换成 device 侧的
`arange(CAP) < n` 掩码；补齐行必须从语义里排除（否则污染 scatter_mean / 桶表）。

## 3. 构建

前置：CANN 已 source（`ASCEND_HOME_PATH`）、`python3 -c "import torch"` 可用。
torch ≥ 2.14 要求 C++20 —— 本目录的 `op_extension/CMakeLists.txt` **全局与 target 级**都已是 20。

```bash
bash csrc/ascend/spconv_gather_gemm/build.sh
# 产物：csrc/ascend/<op>/op_extension/build/lib<op>_ops.so
#       csrc/ascend/<op>/build_out/custom_opp_*.run（vendor 内核包）
```

## 4. 使用

```bash
export UNUM_OPS_ROOT=$PWD          # 指向本仓库根，torchsparse 自动发现
python demo.py --cr 0.5
```
未构建或 ABI 不匹配时自动回退纯 torch，结果仍正确。

## 5. 实测（910B4 / torch 2.14 / torch_npu 2.14 / CANN 9.2，20k 点 cr=0.5）

| 场景 | 耗时 |
|---|---:|
| eager，纯 torch（无算子） | 288.6 ms |
| eager，仅固定行数版 4 个算子 | 283.8 ms |
| eager，完整 unum_ops 库 | 73.7 ms |
| aclgraph（+完整库） | 27.5 ms |
| GE 静态分段（+完整库） | 15.2 ms |
| GE 动态（+完整库，任意形状） | 36.8 ms |

> 只装固定行数版时 eager 几乎不提速：默认启发式把大层排除在 `spconv_gather_gemm` 之外，
> eager 的时间大头是 devox / 邻域聚合的 torch 实现。真正的收益在图模式
> （114 次 conv 全落在同一融合核，省掉每处约 5.7 ms 的 host 同步）。

## 6. 已知约束

- `spconv_gather_gemm*` 要求 `cin % 8 == 0`（GE 路径用 `_pad_cin_to_8` 把 stem 的 4 通道补齐）。
- 容量化版本的输出是**定长 CAP**：CAP 由调用方给定，超过 CAP 的帧需要更大 CAP（或分档）。
