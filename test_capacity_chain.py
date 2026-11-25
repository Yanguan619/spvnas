"""容量化算子链对拍：capacity + device-n 路径必须与固定行数路径逐位一致。

⚠️ 数据构造要求（这条踩过坑）：coords 必须**唯一**且**按 pack_keys 单调序排序**，
这正是生产里 `unique_coords` 的输出形态；随机/未排序/有重复键会让 query 返回全 -1。

用法：
    UNUM_OPS_ROOT=<有容量版 .so 的树> python3 test_capacity_chain.py
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import torchsparse.nn.functional as F                 # noqa: E402
import torchsparse.backend as B                       # noqa: E402
import ascend_capacity as cap                         # noqa: E402
from torchsparse.nn.utils import get_kernel_offsets   # noqa: E402

dev = torch.device("npu:0")
BOX = 24
print("[ops] available =", cap.available(), flush=True)
if not cap.available():
    print("FAIL: 容量化算子未加载，检查 UNUM_OPS_ROOT")
    sys.exit(1)

ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print("  %-40s %s %s" % (name, "PASS" if cond else "FAIL", extra))


def uniq_sorted(n, seed):
    """batch=0、按 (x,y,z) 字典序去重排序 —— 与 unique_coords 输出形态一致。"""
    torch.manual_seed(seed)
    allc = [(0, x, y, z) for x in range(BOX) for y in range(BOX) for z in range(BOX)]
    perm = torch.randperm(len(allc))[:n].tolist()
    return torch.tensor(sorted(allc[i] for i in perm), dtype=torch.int32, device=dev).contiguous()


def pad_to(t, cap_rows):
    return torch.cat([t, t[-1:].repeat(cap_rows - t.shape[0], 1)]).contiguous()


# ------------------------------------------------------------------ 1) 去重
print("\n[1] 容量版去重 sortdedup_v4 vs 固定版 unique_coords")
N_REAL, CAP = 900, 1280
coords = uniq_sorted(N_REAL, 3)
pad = torch.randint(0, BOX, (CAP - N_REAL, 4), dtype=torch.int32, device=dev)
pad[:, 0] = 0
coords_cap = torch.cat([coords, pad], 0).contiguous()

uc_ref = B.unique_coords(coords.contiguous())
out_cap, u = cap.unique_coords_cap(coords_cap, N_REAL, dev)
u_val = int(u.reshape(-1)[0].item())
check("行数一致 (ref U vs cap U)", uc_ref.shape[0] == u_val, "(ref=%d cap=%d)" % (uc_ref.shape[0], u_val))
check("容量输出行数 == CAP", out_cap.shape[0] == CAP, "(rows=%d)" % out_cap.shape[0])
a = uc_ref.to(torch.int64).cpu().numpy()
b = out_cap[:u_val].to(torch.int64).cpu().numpy()
import numpy as np
a = a[np.lexsort(a.T[::-1])]
b = b[np.lexsort(b.T[::-1])]
check("去重坐标集合逐位一致", a.shape == b.shape and np.array_equal(a, b))

# ------------------------------------------------------------------ 2) 邻域表 + 卷积
print("\n[2] 容量版 kernel_map + conv vs 固定版 _lookup_offsets + gather_gemm")
N, M = 900, 700
CAP_IN, CAP_OUT = 1280, 1024
ic = uniq_sorted(N, 7)
oc = uniq_sorted(M, 11)
ic_cap, oc_cap = pad_to(ic, CAP_IN), pad_to(oc, CAP_OUT)

offs = get_kernel_offsets((3, 3, 3), 1, device=dev)
nb_ref = F._lookup_offsets(ic, oc, (1, 1, 1), offs).to(torch.int32)
nb_cap = cap.kernel_map(ic_cap, oc_cap, (N, M), ks=3, stride=1, dev=dev)
check("nb 形状 [CAP_OUT, kv]", tuple(nb_cap.shape) == (CAP_OUT, 27), str(tuple(nb_cap.shape)))
check("nb 前 M 行与固定版逐位一致",
      torch.equal(nb_cap[:M].to(torch.int64), nb_ref.to(torch.int64)),
      "(非-1 ref=%.4f cap=%.4f)" % (float((nb_ref >= 0).float().mean()),
                                    float((nb_cap[:M] >= 0).float().mean())))

cin = cout = 64
kv = 27
feats = torch.randn(N, cin, device=dev) * 0.1
w = torch.randn(kv * cin, cout, device=dev) * 0.02
feats_cap = torch.cat([feats, feats[-1:].repeat(CAP_IN - N, 1)]).contiguous()

_v = (nb_ref >= 0)
_g = feats[nb_ref.clamp(min=0).long()] * _v.unsqueeze(-1).to(feats.dtype)
conv_ref = _g.reshape(M, kv * cin) @ w
out = cap.conv(feats_cap, nb_cap, w, M, dev=dev)
d = float((out[:M].float() - conv_ref.float()).abs().max())
check("conv 前 M 行 vs 手算", d < 1e-5, "max|d|=%.3e" % d)

# ------------------------------------------------------------------ 3) devox
print("\n[3] 容量版 devox_v4 vs torch 参考")
K, NR, NCAP, C = 8, 500, 640, 32
feat = torch.randn(300, C, device=dev)
idx_cap = torch.zeros(K, NCAP, dtype=torch.int32, device=dev)
idx_cap[:, :NR] = torch.randint(0, 300, (K, NR), dtype=torch.int32, device=dev)
w_cap = torch.zeros(K, NCAP, device=dev)
w_cap[:, :NR] = torch.rand(K, NR, device=dev)
out_dv = cap.devox(feat, idx_cap, w_cap, NR, dev=dev)
ref_dv = (feat[idx_cap[:K, :NR].long()] * w_cap[:K, :NR].unsqueeze(-1)).sum(0)
d_dv = float((out_dv[:NR].float() - ref_dv.float()).abs().max())
check("devox 前 NR 行 vs torch", d_dv < 1e-5, "max|d|=%.3e" % d_dv)

print("\n总体:", "全部 PASS" if ok else "有 FAIL")
sys.exit(0 if ok else 1)
