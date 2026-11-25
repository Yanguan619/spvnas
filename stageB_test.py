"""Stage B 验收：容量化 conv（_rect_map_cap + gather_gemm_v2 带 device n）与固定路径逐位一致。

用法：UNUM_OPS_ROOT=... TS_CAP=1280 python3 stageB_test.py
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
from torchsparse.nn.utils import get_kernel_offsets  # noqa: E402

dev = torch.device("npu:0")
CAP = F.cap_size()
print("[env] TS_CAP=%d" % CAP, flush=True)

ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print("  %-46s %s %s" % (name, "PASS" if cond else "FAIL", extra))


def uniq_sorted(n, seed, box=24):
    torch.manual_seed(seed)
    allc = [(0, x, y, z) for x in range(box) for y in range(box) for z in range(box)]
    perm = torch.randperm(len(allc))[:n].tolist()
    return torch.tensor(sorted(allc[i] for i in perm), dtype=torch.int32, device=dev).contiguous()


N, CIN, COUT = 900, 64, 64
KS, ST = 3, 2
coords = uniq_sorted(N, 5)
coords_cap, n_in = F.cap_pad_coords(coords)

torch.manual_seed(0)
feats = torch.randn(N, CIN, device=dev) * 0.1
feats_cap = torch.cat([feats, feats.new_zeros(CAP - N, CIN)], 0).contiguous()
w = torch.randn(KS ** 3, CIN, COUT, device=dev) * 0.02

# ---------------- 参考（纯 torch，固定路径）
_floored = torch.div(coords[:, 1:], ST, rounding_mode="floor")
out_coords_ref = torch.unique(torch.cat([coords[:, :1], _floored], 1), dim=0).contiguous()
offs = get_kernel_offsets((KS, KS, KS), 1, device=dev)
nb_ref = F._lookup_offsets(coords, out_coords_ref, (ST, ST, ST), offs).to(torch.int32)
M = out_coords_ref.shape[0]
kv = KS ** 3
_v = (nb_ref >= 0)
_g = feats[nb_ref.clamp(min=0).long()] * _v.unsqueeze(-1).to(feats.dtype)
conv_ref = _g.reshape(M, kv * CIN) @ w.reshape(kv * CIN, COUT)      # [M, COUT]
print("[ref] out_coords=%s conv=%s" % (tuple(out_coords_ref.shape), tuple(conv_ref.shape)), flush=True)

# ---------------- 容量路径：先下采样，再建表，再卷积
out_coords_cap, n_out = F.spdownsample(coords_cap, ST, KS, 1, n=n_in)
u_out = int(n_out.reshape(-1)[0].item())
print("[cap] out_coords=%s u=%d (ref M=%d)" % (tuple(out_coords_cap.shape), u_out, M), flush=True)
check("下采样输出行数 = CAP", out_coords_cap.shape[0] == CAP)
check("真实输出行数一致", u_out == M, "(ref=%d cap=%d)" % (M, u_out))

nb_cap = F._rect_map(coords_cap, out_coords_cap, KS, (ST, ST, ST), 1,
                     n_in=n_in, n_out=n_out)
print("[cap] nb=%s 非-1=%.4f (ref 非-1=%.4f)"
      % (tuple(nb_cap.shape), float((nb_cap[:M] >= 0).float().mean()),
         float((nb_ref >= 0).float().mean())), flush=True)
check("nb 形状 = [CAP, kv]", tuple(nb_cap.shape) == (CAP, kv), str(tuple(nb_cap.shape)))
_a = nb_ref.to(torch.int64).cpu()
_b = nb_cap[:M].to(torch.int64).cpu()
_eq = torch.equal(_a, _b)
_ex = ((_a >= 0) != (_b >= 0))
_bo = (_a >= 0) & (_b >= 0)
print("[diag] 不同元素=%d/%d  存在性不一致=%d  两者非-1时相同=%s"
      % (int((_a != _b).sum()), _a.numel(), int(_ex.sum()),
         bool(torch.equal(_a[_bo], _b[_bo])) if _bo.any() else "n/a"), flush=True)
if not _eq:
    for r, c in (_a != _b).nonzero()[:6].tolist():
        print("   [%d,%d] ref=%d cap=%d" % (r, c, int(_a[r, c]), int(_b[r, c])), flush=True)
    _d = (_b - _a)
    print("[diag] 差值的取值分布: min=%d max=%d" % (int(_d[_bo].min()), int(_d[_bo].max())), flush=True)
check("nb 前 M 行与固定版逐位一致", _eq)

with torch.no_grad():
    out = F._apply_forward(feats_cap, w, None, CAP, nb_cap, n_real=n_out)
d = float((out[:M].float() - conv_ref.float()).abs().max())
check("conv 前 M 行 vs 参考", d < 1e-5, "max|d|=%.3e" % d)

# ---------------- 集成：直接走 conv3d（SparseTensor 带 n）
x = SparseTensor(feats_cap, coords_cap, 1, n=n_in)
with torch.no_grad():
    y = F.conv3d(x, w, kernel_size=KS, stride=ST)
check("conv3d 输出携带 n", getattr(y, "n", None) is not None and
      int(y.n.reshape(-1)[0].item()) == M, "(n=%s)" % (int(y.n.reshape(-1)[0].item()),))
d2 = float((y.F[:M].float() - conv_ref.float()).abs().max())
check("conv3d 前 M 行 vs 参考", d2 < 1e-5, "max|d|=%.3e" % d2)
check("conv3d 输出行数 = CAP", y.F.shape[0] == CAP)

print("\n总体:", "全部 PASS" if ok else "有 FAIL")
sys.exit(0 if ok else 1)
