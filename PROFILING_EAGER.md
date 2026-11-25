# Eager 模式算子级 Profiling（910B4）

> 采集时间：2026-11-25　|　对象：SPVCNN@30GMACs（cr=0.5），SemanticKITTI 序列 08 首帧整帧

## 0. 采集条件

| 项 | 值 |
|---|---|
| 硬件 / CANN | Ascend 910B4 / CANN 9.2.0 |
| 输入 | 123,389 点 → **94,625 体素**（voxel=0.05，整帧） |
| 模式 | **eager**（`TS_CAP=0`，固定行数路径，无图） |
| 采集 | `msprof --application="python3 prof_full.py eager2" --ai-core=on` |
| 迭代 | **6 次前向**（帧1 冷启、帧2–5 稳态、帧6 被缓冲截断，详见 §0.1） |
| 分析 | `analyze_op_profile.py`（本报告已把「6 帧合计」折算为**每帧**） |

### 0.1 这份数据是几个 step（重要）

采集到 **6 个 step（6 次前向）**，但只有 **4 个是完整稳态**：

| 帧 | 算子数 | 设备 ms | 帧周期 ms | 说明 |
|---|---:|---:|---:|---|
| 1 | 1051 | 73.99 | **226.8** | **冷启**：首次调用含 tiling 编译，周期被拉长，**分析时应剔除** |
| 2 | 1048 | 73.99 | 104.9 | 稳态 |
| 3 | 1048 | 73.92 | 103.5 | 稳态 |
| 4 | 1048 | 73.97 | 103.1 | 稳态 |
| 5 | 1048 | 73.90 | 102.5 | 稳态 |
| 6 | **847** | 57.55 | — | **被 msprof 缓冲截断**，不满一帧 |

- 稳态（帧 2–5）：**1048 个算子/帧、设备 73.9–74.0 ms/帧、周期 102.5–104.9 ms**；
- 全 6 帧合计 443.8 ms → 平均 73.97 ms/帧，与稳态值一致（截断帧的缺失被帧间算子在两侧摊平，误差 <0.2%）；
- `step_trace_*.csv/json` 为空（0 事件），所以**没有 step 标记**，必须按上面的时间戳分组才能分辨帧；
- 若你想要**每帧都完整**的数据，建议把前向次数降到 3 或调大 msprof 缓冲后重采。

按 `SpconvDevoxV4`（每帧固定 4 次）的时间戳分组即可复现上面的切分：

```python
d = df[df["OP Type"] == "SpconvDevoxV4"]["Task Start Time(us)"].sort_values().values
starts = [d[4*i] for i in range(6)]        # 每 4 个 Devox = 一帧
```

复跑：

```bash
cd /data2/jev_work/spvnas-ascend
NO_TORCH_PROFILER=1 TS_CAP=0 PERF_MODE=eager msprof \
  --application="python3 -u prof_full.py eager2" \
  --output=/data2/jev_work/msp_eager2 --ai-core=on
msprof --export=on --output=/data2/jev_work/msp_eager2
python3 analyze_op_profile.py --prof /data2/jev_work/msp_eager2 --out rep.md
```

## 1. 总览

| 指标 | 值 |
|---|---|
| wall（实测） | **101.76 ms/帧** |
| 设备侧 kernel 合计 | **73.97 ms/帧** |
| **host 间隙** | **27.8 ms/帧（27%）** |
| 算子实例 | 1049 个/帧（36 种类型） |

> 结论先行：**eager 有 27% 的时间不在设备上跑 kernel**（Python 调度 + 同步 + 分派）。
> 这正是图模式该吃的收益，也是 aclgraph 能到 75.0 ms 的空间来源。

## 2. 模块分解（ms/帧）

| 模块 | 算子数/帧 | ms/帧 | mac | vec | scalar | mte1 | mte2 | mte3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **自定义 AscendC 算子** | 568 | **70.25** | 4.71 | 3.37 | 9.38 | 4.20 | 7.67 | 21.07 |
| 通用算子（layout/elementwise） | 290 | 2.39 | 0 | 0.18 | 0 | 0 | 0 | 0.53 |
| 布局类（Slice/Concat/Cast…） | 191 | 1.32 | 0 | 0.17 | 0 | 0 | 0 | 0.14 |

**自定义算子占设备时间 95%** —— 优化重心应在这 5 个算子上。

## 3. 算子类型 Top（ms/帧）

| 算子 | 次/帧 | ms/帧 | 占比 | 说明 |
|---|---:|---:|---:|---|
| **SpconvGatherGemm**（卷积主体） | 38 | **19.81** | 26.8% | cube 利用率 97–99%，但标量 542 µs/次 |
| **SpconvQuery**（邻域查找） | 15 | **19.18** | 25.9% | 等效 **3.7 GB/s** |
| **SpconvSortDedupV3**（坐标去重） | 5 | **8.73** | 11.8% | 等效 **4.0 GB/s** |
| SpconvKeyprepV2（直方图） | 15 | 8.64 | 11.7% | |
| SpconvDevoxV4（反体素化） | 4 | 4.31 | 5.8% | 40 GB/s，正常 |
| SpconvKeyprepV2Prefix | 15 | 3.06 | 4.1% | 12.3 GB/s |
| Slice | 222 | 2.16 | 2.9% | 94625×4 取列，239 GB/s |
| InplaceScatterAdd | 40 | 1.25 | 1.7% | |
| Add / GatherV3 / Cast / MatMulV2 / ConcatD | 80/40/124/43/53 | 1.01/0.76/0.61/0.55/0.53 | 4.7% | 零散 |
| Relu/FloorDiv/ViewCopy/Sub/ZerosLike… | ~350 | ~3.0 | 4.1% | 零散 |

## 4. Top 算子 × shape（按每帧总时间）

| 算子 | shape | µs/次 | 次/帧 | ms/帧 |
|---|---|---:|---:|---:|
| SpconvSortDedupV3 | 94625,4 | 2658 | 2 | **5.32** |
| SpconvQuery | 94625,4;94625;524288;524288;93 | 4566 | 1 | **4.57** |
| SpconvQuery | 68931,4;68931;524288;524288;93 | 3276 | 1 | 3.28 |
| SpconvGatherGemm | 94625,48;94625,27;1296,48 | 1034 | 3 | 3.10 |
| SpconvGatherGemm | 68931,16;68931,27;432,16 | 680 | 4 | 2.72 |
| SpconvGatherGemm | 68931,48;68931,27;1296,48 | 799 | 3 | 2.40 |
| SpconvSortDedupV3 | 68931,4 | 1775 | 1 | 1.78 |
| SpconvQuery | 42166,4;42166;262144;262144;93 | 1749 | 1 | 1.75 |
| SpconvQuery | 94625,4;42166;262144;262144;36 | 1580 | 1 | 1.58 |
| SpconvQuery | 94625,4;94625;524288;524288;36 | 1537 | 1 | 1.54 |
| SpconvDevoxV4 | 9522,128;8,94625;8,94625;1,1;1 | 1482 | 1 | 1.48 |
| SpconvGatherGemm | 42166,64;42166,27;1728,64 | 438 | 3 | 1.31 |
| SpconvGatherGemm | 42166,32;42166,27;864,32 | 430 | 3 | 1.29 |
| SpconvQuery | 94625,4;9522;65536;65536;36 | 1212 | 1 | 1.21 |
| SpconvQuery | 68931,4;94625;524288;524288;36 | 1210 | 1 | 1.21 |
| SpconvGatherGemm | 21525,128;21525,27;3456,128 | 395 | 3 | 1.19 |
| SpconvSortDedupV3 | 42166,4 | 1130 | 1 | 1.13 |
| SpconvDevoxV4 | 42166,64;8,94625;8,94625;1,1;1 | 1126 | 1 | 1.13 |

## 5. 瓶颈判定

### 5.1 SpconvGatherGemm —— 实现层已到带宽上限，算法层有 8.9× 冗余

| 分项（94625,48 一次调用） | 值 |
|---|---|
| mac / vec / scalar | 202 / 72 / 542 µs |
| **mte1 + mte2 + mte3** | 183 + 479 + 447 = **1109 µs ≈ 总时长 1036 µs** |
| cube 利用率 | 97–98%（cube 本身健康） |

分析脚本按 `scalar/duration = 0.52 ≥ 0.5` 判成 **scalar-bound**，但三项之和说明真实约束是**搬运**。

更关键的是：把有效 tap 数从 27 降到 1，耗时只从 1969 降到 1059 µs ——
说明存在 **~1.06 ms/次 的固定地板**，来自每行无条件 `Duplicate(row,0,K)`（5 KB 清零）
+ 把 89% 为零的 27 宽矩阵写进 staging（490 MB/次）。详见 `REFACTOR_GATHER_GEMM.md`。

### 5.2 SpconvQuery / SpconvSortDedupV3 —— 随机访问受限

| 算子 | µs/次 | 等效带宽 | 判定 |
|---|---:|---:|---|
| SpconvQuery（94625 行） | 4566 | **3.7 GB/s** | 二分/散列查找，随机访问 |
| SpconvSortDedupV3（94625 行） | 2658 | **4.0 GB/s** | 排序去重，随机访问 |

可达带宽实测 1183 GB/s（512 MB copy）→ 这两类只跑到 0.3%。这是**访存模式**的代价，不是实现疏忽。

### 5.3 Slice —— 次数多但便宜

222 次/帧、2.16 ms、等效 239 GB/s。虽然次数最多（94625×4 取单列），但不是瓶颈，不建议动。

## 6. 结论与优化方向

| 优先级 | 对象 | 状态 |
|---|---|---|
| — | host 间隙 27.8 ms（27%） | **已可用图模式吃掉**：aclgraph 75.0 ms（−26.8 ms） |
| 1 | GatherGemm 的 490 MB 稠密零矩阵 | 需重构（`REFACTOR_GATHER_GEMM.md`，预期 4–6×） |
| 2 | Query / SortDedup 随机访问 | 需算法改写（radix/合并查找），非调参可解 |
| 3 | 零散 elementwise（~4 ms） | 收益小，可后期融合 |

**一句话**：eager 的设备时间里 95% 集中在 5 个自定义算子上，其中两个（Query、SortDedup）
是访存模式受限、两个（GatherGemm、Keyprep）是数据冗余受限、一个（Devox）已接近带宽；
另外 27% 的 wall 时间根本不在 kernel 上（host 间隙），那部分已经用 aclgraph 拿回来了。
