"""910B4 可达 HBM 带宽 + 稀疏卷积搬运量 roofline 对比。"""
import time

import torch
import torch_npu  # noqa: F401

dev = torch.device("npu:0")
N = 512 * 1024 * 1024 // 4          # 512 MB of float32


def bench(fn, tag, bytes_moved):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    med = ts[len(ts) // 2]
    print("  %-22s %8.3f ms   %7.1f GB/s" % (tag, med * 1e3, bytes_moved / med / 1e9))
    return bytes_moved / med / 1e9


a = torch.empty(N, dtype=torch.float32, device=dev)
b = torch.empty(N, dtype=torch.float32, device=dev)
print("[roofline] 910B4 实测带宽（512MB 张量）", flush=True)
bw_copy = bench(lambda: b.copy_(a), "copy (读+写)", N * 4 * 2)
bw_fill = bench(lambda: a.fill_(1.0), "fill (只写)", N * 4)

print("\n[roofline] 各卷积「gather 搬运量」下界 vs 实测")
print("  说明：一次稀疏 conv 需要把 N×kv×cin 个 float 从 featGm 搬进 UB")
ASSUM = bw_copy
rows_kv_cin = [
    ("94625x48 kv27", 94625, 27, 48, 1036),
    ("68931x48 kv27", 68931, 27, 48, 797),
    ("68931x16 kv27", 68931, 27, 16, 683),
    ("42166x64 kv27", 42166, 27, 64, 439),
    ("42166x32 kv27", 42166, 27, 32, 432),
    ("21525x128 kv27", 21525, 27, 128, 396),
]
tot_us = 0.0
for name, n, kv, cin, us in rows_kv_cin:
    byts = n * kv * cin * 4
    ideal_ms = byts / (ASSUM * 1e9) * 1e3
    eff = byts / (us * 1e-6) / 1e9
    frac = eff / ASSUM
    tot_us += us
    print("  %-16s 搬运 %.0f MB | 实测 %6.0f us | 等效 %5.1f GB/s | 可达带宽的 %4.0f%% | 理想 %5.0f us"
          % (name, byts / 2**20, us, eff, 100 * frac, ideal_ms * 1e3))
print("\n  这 6 个 shape 实测合计 %.0f us；按搬运下界合计 %.0f us"
      % (tot_us, sum(n * kv * cin * 4 / (ASSUM * 1e9) * 1e6 for _, n, kv, cin, _ in rows_kv_cin)))
print("### ROOFLINE_DONE")
