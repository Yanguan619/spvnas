"""重构可行性前置验证：nb[m,k] 的局部性。

隐式 GEMM 复用输入 tile 的前提是：排序坐标下，输出行 m 的 27 个邻居在 uniq 里的
位置离 m 不远（于是可以维护一个 UB 滑动窗口，而不必每次回 GM 取）。
这里在真实帧上量化：|nb - m| 的分布、以及按 tile 处理时窗口要开多大。
"""
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401

REPO = "/data2/jev_work/spvnas-ascend"
sys.path.insert(0, REPO)
import ascend_meta  # noqa: F401,E402
import torchsparse.nn.functional as F  # noqa: E402
from torchsparse.utils.quantize import sparse_quantize  # noqa: E402
import pathlib  # noqa: E402

dev = torch.device("npu:0")
for _op in ("keyprep_v2", "keyprep_v1", "query", "nb", "sortdedup_v3"):
    F._load_cap_so(_op)

ROOT = pathlib.Path("/data2/jev_work/datasets/semantic-kitti/dataset/sequences/08/velodyne")
f0 = sorted(ROOT.glob("*.bin"))[0]
pts = np.fromfile(f0, dtype=np.float32).reshape(-1, 4)
pc_ = np.round(pts[:, :3] / 0.05).astype(np.int32)
pc_ -= pc_.min(0, keepdims=1)
uniq, inds, inv = sparse_quantize(pc_, return_index=True, return_inverse=True)
print("[data] points=%d voxels=%d" % (len(pts), len(uniq)), flush=True)

b = np.zeros((len(uniq), 1), dtype=np.int32)
coords = torch.from_numpy(np.concatenate([b, uniq], 1)).to(dev)

print("\n=== 计算 stride=1 / 3x3x3 的 nb（与 conv3d 同一条路径）===", flush=True)
nb = F._rect_map(coords, coords, (3, 3, 3), (1, 1, 1), 1,
                 torch.tensor(len(uniq), dtype=torch.int32, device=dev),
                 torch.tensor(len(uniq), dtype=torch.int32, device=dev))
nb = nb.cpu().numpy().astype(np.int64)
M, KV = nb.shape
print("  nb shape=%s  valid=%d/%d" % (nb.shape, (nb >= 0).sum(), nb.size), flush=True)

d = np.abs(nb - np.arange(M)[:, None])
valid = nb >= 0
dv = d[valid]
print("\n=== |nb - m| 分布（仅有效项）===")
for q in (50, 90, 99, 99.9, 100):
    print("  p%-5s %8d" % (q, int(np.percentile(dv, q))), flush=True)

print("\n=== 按输出 tile 处理需要的窗口（行）===")
for TB in (128, 256, 512, 1024):
    win = []
    for s in range(0, M, TB):
        e = min(s + TB, M)
        sub = nb[s:e]
        v = sub[sub >= 0]
        win.append(int(v.max() - v.min() + 1) if v.size else 0)
    win = np.array(win)
    need = int(np.percentile(win, 99))
    ub_kb = (need * 48 * 4) / 1024.0          # cin=48, float32
    print("  TB=%-5d 窗口中位=%5d  p99=%5d  -> 窗口需 %.0f KB UB (cin=48)"
          % (TB, int(np.median(win)), need, ub_kb), flush=True)

print("\n=== 若只按 |nb-m| <= W 截断，能覆盖多少 lookup ===")
for W in (256, 1024, 4096, 16384):
    cov = (dv <= W).mean()
    print("  W=%-6d 覆盖 %.3f%%" % (W, 100 * cov), flush=True)
print("### LOCALITY_DONE")
