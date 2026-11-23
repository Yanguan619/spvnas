"""Stage A 验收：容量模式下 spdownsample（含 dedup + 哨兵重填）与固定模式逐位一致。

用法：UNUM_OPS_ROOT=... TS_CAP=1280 python3 stageA_test.py
"""
import os
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401

REPO = "/data2/jev_work/spvnas-ascend"
sys.path.insert(0, REPO)

import torchsparse.nn.functional as F      # noqa: E402
from torchsparse import SparseTensor       # noqa: E402

dev = torch.device("npu:0")
CAP = F.cap_size()
print("[env] TS_CAP=%d  cap_enabled=%s" % (CAP, F.cap_enabled()), flush=True)
if not F.cap_enabled():
    print("FAIL: 需要用 TS_CAP=<CAP> 运行")
    sys.exit(1)

ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print("  %-42s %s %s" % (name, "PASS" if cond else "FAIL", extra))


def uniq_sorted(n, seed, box=24):
    torch.manual_seed(seed)
    allc = [(0, x, y, z) for x in range(box) for y in range(box) for z in range(box)]
    perm = torch.randperm(len(allc))[:n].tolist()
    return torch.tensor(sorted(allc[i] for i in perm), dtype=torch.int32, device=dev).contiguous()


N = 900
coords = uniq_sorted(N, 3)
coords_cap, n_dev = F.cap_pad_coords(coords)
print("[输入] 真实 %d 行 → 容量 %d 行，n=%s" % (N, coords_cap.shape[0], n_dev.tolist()), flush=True)

# 参考：纯 torch 实现（TS_CAP 开启时不能再调 spdownsample 的固定模式）
def ref_downsample(c, st=2):
    out = c.clone()
    for kk in range(3):
        out[:, 1 + kk] = torch.div(out[:, 1 + kk], st, rounding_mode="floor")
    return torch.unique(out, dim=0).contiguous()


ref = ref_downsample(coords)
print("[ref] 固定模式输出 %s" % (tuple(ref.shape),), flush=True)

# 容量模式
out_cap, u = F.spdownsample(coords_cap, 2, 2, 1, n=n_dev)
u_val = int(u.reshape(-1)[0].item())
print("[cap] 输出 %s 真实去重数 u=%d" % (tuple(out_cap.shape), u_val), flush=True)

check("输出行数恒为 CAP", out_cap.shape[0] == CAP, "(CAP=%d)" % CAP)
check("真实去重数一致", u_val == ref.shape[0], "(ref=%d cap=%d)" % (ref.shape[0], u_val))

a = ref.to(torch.int64).cpu().numpy()
b = out_cap[:u_val].to(torch.int64).cpu().numpy()
a = a[np.lexsort(a.T[::-1])]
b = b[np.lexsort(b.T[::-1])]
check("真实区域坐标集合逐位一致", a.shape == b.shape and np.array_equal(a, b))

sent_tail = out_cap[u_val:]
check("尾部全部是哨兵键", bool((sent_tail[:, 1:] == F._SENTINEL).all()),
      "(尾部行数=%d)" % sent_tail.shape[0])

# 再走一级（验证 n 能继续传下去）
out_cap2, u2 = F.spdownsample(out_cap, 2, 2, 1, n=u)
ref2 = ref_downsample(ref)
u2_val = int(u2.reshape(-1)[0].item())
check("第二级：行数恒定", out_cap2.shape[0] == CAP)
check("第二级：去重数一致", u2_val == ref2.shape[0], "(ref=%d cap=%d)" % (ref2.shape[0], u2_val))
a2 = ref2.to(torch.int64).cpu().numpy()
b2 = out_cap2[:u2_val].to(torch.int64).cpu().numpy()
a2 = a2[np.lexsort(a2.T[::-1])]
b2 = b2[np.lexsort(b2.T[::-1])]
check("第二级：坐标集合逐位一致", a2.shape == b2.shape and np.array_equal(a2, b2))

# SparseTensor 的 n 能透传
st = SparseTensor(torch.zeros(N, 4, device=dev), coords, 1, n=n_dev)
check("SparseTensor.n 透传（clone/detach）",
      st.clone().n is n_dev and st.detach().n is n_dev)

print("\n总体:", "全部 PASS" if ok else "有 FAIL")
sys.exit(0 if ok else 1)
