"""验证 run-merge 前提：对固定 tap k，连续输出行 m,m+1,... 的源是否也连续。

若平均 run 长度≈1，则"合并拷贝"无效；若 ≥2，则 [tap][row] 布局 + blockCount 合并可行。
"""
import sys
import pathlib

import numpy as np
import torch
import torch_npu  # noqa: F401

REPO = "/data2/jev_work/spvnas-ascend"
sys.path.insert(0, REPO)
import ascend_meta  # noqa: F401,E402
import torchsparse.nn.functional as F  # noqa: E402
from torchsparse.utils.quantize import sparse_quantize  # noqa: E402

dev = torch.device("npu:0")
for _op in ("keyprep_v2", "keyprep_v1", "query", "nb"):
    F._load_cap_so(_op)

ROOT = pathlib.Path("/data2/jev_work/datasets/semantic-kitti/dataset/sequences/08/velodyne")
stats = []
for fi, f0 in enumerate(sorted(ROOT.glob("*.bin"))[:3]):
    pts = np.fromfile(f0, dtype=np.float32).reshape(-1, 4)
    pc_ = np.round(pts[:, :3] / 0.05).astype(np.int32)
    pc_ -= pc_.min(0, keepdims=1)
    uniq, inds, inv = sparse_quantize(pc_, return_index=True, return_inverse=True)
    b = np.zeros((len(uniq), 1), dtype=np.int32)
    coords = torch.from_numpy(np.concatenate([b, uniq], 1)).to(dev)
    M = len(uniq)
    nb = F._rect_map(coords, coords, (3, 3, 3), (1, 1, 1), 1,
                     torch.tensor(M, dtype=torch.int32, device=dev),
                     torch.tensor(M, dtype=torch.int32, device=dev)).cpu().numpy().astype(np.int64)
    KV = nb.shape[1]

    runs_all, valid_all = [], 0
    for k in range(KV):
        v = nb[:, k]
        ok = v >= 0
        valid_all += int(ok.sum())
        # 在连续有效且 src 连续的位置上累计 run 长度
        m = 0
        while m < M:
            if not ok[m]:
                m += 1
                continue
            run = 1
            while m + run < M and ok[m + run] and v[m + run] == v[m] + run:
                run += 1
            runs_all.append(run)
            m += run
    runs = np.array(runs_all)
    print("帧%d  voxels=%d  有效tap=%d  拷贝 run 数=%d  平均run=%.2f  run>=2 占比=%.1f%%  run>=4 占比=%.1f%%"
          % (fi, M, valid_all, len(runs), runs.mean(), 100 * (runs >= 2).mean(),
             100 * (runs >= 4).mean()), flush=True)
    stats.append((valid_all, len(runs), runs.mean()))

v, r, a = zip(*stats)
print("\n=== 汇总 ===")
print("  平均 run 长度 = %.2f  -> 拷贝指令数可降为原来的 1/%.2f" % (np.mean(a), np.mean(a)))
print("  若只合并 run>=2 的部分，指令数 %.0f%%" % (100 * np.mean([1.0 / x for x in a])))
print("### RUNMERGE_DONE")
