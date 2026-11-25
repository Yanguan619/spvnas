"""性能：容量路径 vs 固定行数路径（eager / GE 动态），同协议。

用法：
  TS_CAP=0     python3 perf_cap.py     # 固定行数：只测 eager
  TS_CAP=20480 python3 perf_cap.py     # 容量：eager + GE 动态（一次编译复用）
"""
import os
import statistics
import sys
import time

import torch
import torch_npu  # noqa: F401
import torchair  # noqa: F401
from torchair import get_npu_backend  # noqa: E402

REPO = "/data2/jev_work/spvnas-ascend"
sys.path.insert(0, REPO)

import ascend_meta  # noqa: F401,E402
from torchsparse import SparseTensor                            # noqa: E402
from torchsparse.nn import fold_bn                              # noqa: E402
from core.models.semantic_kitti.spvcnn import SPVCNN            # noqa: E402
import bench, demo                                              # noqa: E402
import torchsparse.nn.functional as F                           # noqa: E402

CAP = int(os.environ.get("TS_CAP", "0") or 0)
SHAPES = (20000, 12000, 6000)
WARM, REPS = 3, 10

dev = torch.device("npu:0")
print("[env] TS_CAP=%d  cap_enabled=%s  warm=%d reps=%d  (median)"
      % (CAP, F.cap_enabled(), WARM, REPS), flush=True)

torch.manual_seed(0)
frame = demo.load_frame("assets/000000.bin", "assets/000000.label")

m = SPVCNN(num_classes=19, cr=0.5, pres=demo.VOXEL, vres=demo.VOXEL).eval()
sd = torch.load("demo_spvcnn.pt", map_location="cpu")
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
m.load_state_dict(sd, strict=False)
fold_bn(m)
m = m.to(dev)

tensors = {}
for n in SHAPES:
    lidar = bench.subset(frame, n)
    tensors[n] = SparseTensor(lidar.F.to(dev), lidar.C.to(dev), lidar.s)


def bench_model(model, tag):
    out = {}
    for n in SHAPES:
        x = tensors[n]
        with torch.no_grad():
            for _ in range(WARM):
                model(x)
            torch.npu.synchronize()
            ts = []
            for _ in range(REPS):
                t0 = time.perf_counter()
                model(x)
                torch.npu.synchronize()
                ts.append((time.perf_counter() - t0) * 1000.0)
        out[n] = statistics.median(ts)
        print("  %-10s %-6d  %8.2f ms   (min %.2f / max %.2f)"
              % (tag, n, out[n], min(ts), max(ts)), flush=True)
    return out


res = {}
res["eager"] = bench_model(m, "eager")

if F.cap_enabled():
    mc = torch.compile(m, backend=get_npu_backend(), dynamic=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        mc(tensors[SHAPES[0]])
    torch.npu.synchronize()
    print("  [compile] 首跑 %.1f s（含编译，不计入下表）" % (time.perf_counter() - t0), flush=True)
    res["ge_dyn"] = bench_model(mc, "ge_dynamic")

print("\n[汇总] TS_CAP=%d" % CAP)
print("  %-8s %s" % ("shape", "  ".join("%9s" % k for k in res)))
for n in SHAPES:
    print("  %-8d %s" % (n, "  ".join("%9.2f" % res[k][n] for k in res)))
if "ge_dyn" in res:
    print("  %-8s %s" % ("speedup", "  ".join(
        "%9.2fx" % (res["eager"][n] / res["ge_dyn"][n]) for n in SHAPES)))
print("### PERF_DONE")
