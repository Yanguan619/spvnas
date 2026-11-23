"""分档 v2：每档一个独立的 compiled 包装，逐档对拍 + 计时。"""
import os, statistics, sys, time
import torch, torch_npu, torchair  # noqa
from torchair import get_npu_backend

REPO = "/data2/jev_work/spvnas-ascend"
sys.path.insert(0, REPO)
import ascend_meta  # noqa
from torchsparse import SparseTensor
from torchsparse.nn import fold_bn
from core.models.semantic_kitti.spvcnn import SPVCNN
import bench, demo
import torchsparse.nn.functional as F

WARM, REPS = 3, 10
SHAPES = (6000, 12000, 20000)
dev = torch.device("npu:0")
print("[env] buckets=%s" % (F._CAP_BUCKETS,), flush=True)

torch.manual_seed(0)
frame = demo.load_frame("assets/000000.bin", "assets/000000.label")
m = SPVCNN(num_classes=19, cr=0.5, pres=demo.VOXEL, vres=demo.VOXEL).eval()
sd = torch.load("demo_spvcnn.pt", map_location="cpu")
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
m.load_state_dict(sd, strict=False); fold_bn(m); m = m.to(dev)

base = {6000: 35.97, 12000: 42.36, 20000: 49.48}
rows = []
for n in SHAPES:
    cap = F.cap_for(n)
    F.set_cap(cap)
    d = bench.subset(frame, n)
    x = SparseTensor(d.F.to(dev), d.C.to(dev), d.s)
    with torch.no_grad():
        ref = m(x)
    torch.npu.synchronize()
    mc = torch.compile(m, backend=get_npu_backend(), dynamic=True)   # 每档独立包装
    t0 = time.perf_counter()
    with torch.no_grad():
        y = mc(x)
    torch.npu.synchronize()
    compile_s = time.perf_counter() - t0
    dd = float((ref.float() - y.float()).abs().max())
    ts = []
    with torch.no_grad():
        for _ in range(WARM):
            mc(x)
        torch.npu.synchronize()
        for _ in range(REPS):
            t0 = time.perf_counter(); mc(x); torch.npu.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    med = statistics.median(ts)
    rows.append((n, cap, compile_s, med, min(ts), dd))
    print("  %-6d 档=%-6d 首跑 %6.1f s  %7.2f ms (min %.2f)  max|d| vs eager=%.3e"
          % (n, cap, compile_s, med, min(ts), dd), flush=True)

print("\n[对比] 单档 20480 → 分档")
for n, cap, _c, med, _mn, _dd in rows:
    print("  %-6d 单档 %6.2f ms → 档%-6d %6.2f ms  (%.2fx)" % (n, base[n], cap, med, base[n] / med))
print("### BUCKET2_DONE")
