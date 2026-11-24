"""整帧性能：在真实 seq08 整帧（~94k 体素）上测 eager / 容量 eager / GE 静态 / GE 动态。

与官方论文的 GTX 1080Ti 单帧延迟（cr=0.5: 160.9 ms）同口径（都是整帧）。
"""
import os
import statistics
import sys
import time

import numpy as np
import torch
import torch_npu  # noqa: F401
import torchair  # noqa: F401
from torchair import get_npu_backend  # noqa: E402

REPO = "/data2/jev_work/spvnas-ascend"
sys.path.insert(0, REPO)

import ascend_meta  # noqa: F401,E402
from torchsparse import SparseTensor                            # noqa: E402
from torchsparse.nn import fold_bn                              # noqa: E402
from torchsparse.utils.quantize import sparse_quantize          # noqa: E402
from core.models.semantic_kitti.spvcnn import SPVCNN            # noqa: E402
from core.datasets.semantic_kitti import SemanticKITTIInternal  # noqa: E402
import torchsparse.nn.functional as F                           # noqa: E402

CAP = int(os.environ.get("TS_CAP", "0") or 0)
MODE = os.environ.get("PERF_MODE", "static")
WARM, REPS = 2, 5
ROOT = "/data2/jev_work/datasets/semantic-kitti/dataset/sequences"
dev = torch.device("npu:0")
print("[env] TS_CAP=%d mode=%s dev=%s" % (CAP, MODE, dev), flush=True)

for _op in ("keyprep_v3", "query_v2", "gather_gemm_v2", "sortdedup_v4", "sortdedup_v3",
            "keyprep_v2", "keyprep_v1", "gather_gemm", "gather", "gather_v2", "query",
            "devox", "devox_v3", "nb"):
    F._load_cap_so(_op)

ds = SemanticKITTIInternal(ROOT, 0.05, 180000, split='val')
lm = ds.label_map
f0 = sorted((__import__("pathlib").Path(ROOT) / "08" / "velodyne").glob("*.bin"))[0]
pts = np.fromfile(f0, dtype=np.float32).reshape(-1, 4)
pc_ = np.round(pts[:, :3] / 0.05).astype(np.int32)
pc_ -= pc_.min(0, keepdims=1)
uniq, inds, inv = sparse_quantize(pc_, return_index=True, return_inverse=True)
print("[data] points=%d  voxels=%d" % (len(pts), len(uniq)), flush=True)

_b = np.zeros((len(uniq), 1), dtype=np.int32)
x = SparseTensor(torch.from_numpy(pts[inds].astype(np.float32)).to(dev),
                 torch.from_numpy(np.concatenate([_b, uniq], 1)).to(dev), 1)

m = SPVCNN(num_classes=19, cr=0.5, pres=0.05, vres=0.05).eval()
sd = torch.load("/data2/jev_work/_ckpt/spvcnn30/spvcnn30_mit.pt", map_location="cpu")
m.load_state_dict(sd, strict=True)
fold_bn(m)
m = m.to(dev)

if CAP:
    F.set_cap(CAP)


def bench(model, tag):
    ts = []
    with torch.no_grad():
        for _ in range(WARM):
            model(x)
        torch.npu.synchronize()
        for _ in range(REPS):
            t0 = time.perf_counter()
            model(x)
            torch.npu.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    print("  %-18s median %8.2f ms  (min %.2f)" % (tag, statistics.median(ts), min(ts)),
          flush=True)
    return statistics.median(ts)


res = {}
res["eager"] = bench(m, "eager")
if CAP:
    t0 = time.perf_counter()
    mc = torch.compile(m, backend=get_npu_backend(), dynamic=(MODE == "dynamic"))
    with torch.no_grad():
        mc(x)
    torch.npu.synchronize()
    print("  [compile] %.1f s" % (time.perf_counter() - t0), flush=True)
    res["ge_" + MODE] = bench(mc, "ge_" + MODE)

print("\n[结果] TS_CAP=%d mode=%s  voxels=%d" % (CAP, MODE, len(uniq)))
for k, v in res.items():
    print("  %-12s %8.2f ms" % (k, v))
print("### PERFFULL_DONE")
