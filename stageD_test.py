"""Stage D：固定 CAP 下一张图吃不同点数（GE 动态 + torch.compile）。

用法：TS_CAP=20480 python3 stageD_test.py
产出：两种点数各自的 eager/编译耗时、编译次数、精度。
"""
import os
import sys
import time

import numpy as np
import torch
import torch_npu  # noqa: F401
import torchair  # noqa: F401

REPO = "/data2/jev_work/spvnas-ascend"
sys.path.insert(0, REPO)

import ascend_meta  # noqa: F401,E402  (注册容量算子的 Meta)
from torchsparse import SparseTensor                            # noqa: E402
from torchsparse.nn import fold_bn                              # noqa: E402
from core.models.semantic_kitti.spvcnn import SPVCNN            # noqa: E402
import bench, demo                                              # noqa: E402
from torchair import get_npu_backend                            # noqa: E402

dev = torch.device("npu:0")
CAP = int(os.environ.get("TS_CAP", "0"))
print("[env] TS_CAP=%d  backend=torchair(dynamic=True)" % CAP, flush=True)
if CAP <= 0:
    print("FAIL: 需要 TS_CAP=<CAP> 运行")
    sys.exit(1)

torch.manual_seed(0)
frame = demo.load_frame("assets/000000.bin", "assets/000000.label")

m = SPVCNN(num_classes=19, cr=0.5, pres=demo.VOXEL, vres=demo.VOXEL).eval()
sd = torch.load("demo_spvcnn.pt", map_location="cpu")
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
m.load_state_dict(sd, strict=False)
fold_bn(m)
m = m.to(dev)

mc = torch.compile(m, backend=get_npu_backend(), dynamic=True)


def run(model, n_pts, reps=1):
    lidar = bench.subset(frame, n_pts)
    x = SparseTensor(lidar.F.to(dev), lidar.C.to(dev), lidar.s)
    with torch.no_grad():
        y = model(x)
    torch.npu.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.time()
        with torch.no_grad():
            y = model(x)
        torch.npu.synchronize()
        ts.append((time.time() - t0) * 1000)
    return y.float().cpu().numpy(), ts[0]


# 参考（eager）
ref = {}
for np_ in (20000, 12000):
    ref[np_] = run(m, np_)[0]
    print("[ref] %d 点 eager 完成" % np_, flush=True)

# 编译：第一次（20k）触发编译；第二次（12k）应复用同一张图
t0 = time.time()
y20, t20 = run(mc, 20000)
compile_s = time.time() - t0
print("[compile] 20k 首跑耗时 %.1f s（含编译）" % compile_s, flush=True)

t0 = time.time()
y12, t12 = run(mc, 12000)
second_s = time.time() - t0
print("[run] 12k 首跑耗时 %.1f s（若≈编译耗时说明重编了）" % second_s, flush=True)

ok = True
for np_, y in ((20000, y20), (12000, y12)):
    d = float(np.abs(ref[np_] - y).max())
    ag = float((ref[np_].argmax(1) == y.argmax(1)).mean()) * 100
    good = d < 1e-4 and ag > 99.99
    ok &= good
    print("  %-6d 点: max|d|=%.3e argmax=%.4f%%  %s" % (np_, d, ag, "PASS" if good else "FAIL"),
          flush=True)

print("  ['一张图'判据] 12k 首跑 %.1f s vs 20k 首跑 %.1f s → %s"
      % (second_s, compile_s,
         "PASS（未重编）" if second_s < 0.5 * compile_s else "FAIL（疑似重编）"), flush=True)
ok &= second_s < 0.5 * compile_s
print("\n总体:", "全部 PASS" if ok else "有 FAIL")
sys.exit(0 if ok else 1)
