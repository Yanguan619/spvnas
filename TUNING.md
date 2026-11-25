# NPU 推理调优结论（SPVCNN @ 910B4）

## 推荐运行配置

```bash
cd /data2/jev_work/spvnas-ascend

# A. 免编译（推荐默认）：一级流水队列 + 内存池扩展段
TASK_QUEUE_ENABLE=2 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=. python3 demo.py

# B. 入图（首次约 150s 编译，之后更快）——与 TASK_QUEUE_ENABLE=2 互斥
TASK_QUEUE_ENABLE=1 PYTHONPATH=. TS_GRAPH=1 python3 demo.py
```

`TASK_QUEUE_ENABLE=2` 与 ACL graph 捕获不兼容（`RuntimeError: Do not support
TASK_QUEUE_ENABLE = 2 during NPU graph capture`），二者只能选一个。

## 实测（94625 体素全帧 / 20k 子集，同一帧 assets/000000.bin）

| 配置 | 20k cr=0.5 | 94k cr=0.5 | 94k cr=1.0 |
|---|---|---|---|
| 纯 torch shim 起点 | 467 ms | 796 ms | 806 ms |
| eager + TQ=1 | 318 ms | 581 ms | 584 ms |
| eager + TQ=2 | 240 ms | 515 ms | 528 ms |
| eager + TQ=2 + expandable_segments | **224 ms** | **493 ms** | **516 ms** |
| TQ=1 + TorchAir 入图 | **205 ms** | **476 ms** | **526 ms** |

对照论文 GTX 1080Ti（cr=0.5 → 160.9 ms，cr=1.0 → 317.1 ms）：
cr=0.5 由慢 4.95× 收窄到 **2.96×**，cr=1.0 由 2.54× 收窄到 **1.66×**；
batch=8 时 141 ms/帧，已优于 1080Ti 单帧。

## 已应用的环境变量（来自 ModelZoo 性能调优指南）

| 变量 | 作用 | 实测 |
|---|---|---|
| `TASK_QUEUE_ENABLE=2` | 一级流水优化，把下发迁移到二级流水 | **-27%**（318→240） |
| `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` | 虚拟地址/物理地址映射，降低动态 shape 碎片 | -6% |
| `ACLNN_CACHE_LIMIT=100000` | 加大 host 侧算子信息缓存 | ~0（本模型 tiling 已命中） |
| `CPU_AFFINITY_CONF=2` | 细粒度绑核 | ~0（单卡空载） |

注意 `expandable_segments` 不能与 `garbage_collection_threshold` /
`max_split_size_mb` 同时使用。

## 代码级（torch 亲和）

- `unique_coords`：坐标去重用「稠密 bool 表 + nonzero」替代 `torch.unique`
  （int64 排序在 910B4 上掉到 AiCPU，占 38% 设备时间）。
- 稠密哈希表替代 `argsort` + `searchsorted` 的邻居查找（上限 128M 条目）。
- conv 内层：按 offset 分组的紧凑 gather + 一次大 GEMM + 原地累加。
- devoxelize：8 次 gather/copy 拍平成 1 次 gather + 原地累加。
- `calc_ti_weights`：`w[mask] = 0` 改为掩码乘法（IndexPut 在 NPU 是随机写）。
- 推理期 BN 折叠进 conv（`torchsparse.nn.fold_bn`）。
- conv/devoxelize 可入 TorchAir 图（`TS_GRAPH=1`），`dynamic=True` 一个函数一张图，
  `torch_npu.npu.set_compile_mode(jit_compile=False)` 禁止二次编译。

## 还没做的（都需要自定义算子或换算法）

- GatherV3 占设备时间 24%（160 次/迭代，mte2 137µs/次）——需要把
  gather+GEMM+scatter 融成一个内核。
- 6 次 AiCPU 的 Sort/SearchSorted 仍占 22 ms——需要两级哈希或 float32 排序。
- MatMul 只占设备时间 5%：算力不是瓶颈，算子数与随机访存才是。
