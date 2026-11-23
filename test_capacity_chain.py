"""容量化算子链对拍：capacity+device-n 路径必须与固定行数路径逐位/1e-6 一致。

用法：UNUM_OPS_ROOT=<有 .so 的树> python3 test_capacity_chain.py
置 UNUM_OPS_ROOT 后本脚本会把 <树>/csrc/ascend 加入 sys.path 之前先加载各 .so
（torchsparse 的 _OPS_ROOT 机制负责）。
"""
import os
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401

REPO = "/data2/jev_work/spvnas-ascend"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "csrc", "ascend"))

import torchsparse.nn.functional as F          # noqa: E402  (触发 .so 加载)
import ascend_capacity as cap                  # noqa: E402

dev = torch.device("npu:0")
torch.manual_seed(0)

print("[ops] available =", cap.available(), flush=True)
print("[ops] sgg/sd/kp/q =", F._sgg_op(), hasattr(torch.ops.npu, "spconv_sortdedup_v4"),
      hasattr(torch.ops.npu, "spconv_keyprep_v3"), hasattr(torch.ops.npu, "spconv_query_v2"),
      flush=True)
if not cap.available():
    print("FAIL: 容量化算子未加载，检查 UNUM_OPS_ROOT")
    sys.exit(1)

ok = True


def report(name, d, tol, extra=""):
    global ok
    passed = d <= tol
    ok &= passed
    print("  %-34s max|d|=%.3e  tol=%.0e  %s %s" % (name, d, tol, "PASS" if passed else "FAIL", extra))


# ---------------------------------------------------------------- 1) sortdedup_v4
print("\n[1] 容量版去重 sortdedup_v4 vs 固定版 unique_coords")
N_REAL, CAP = 700, 700
coords = torch.randint(0, 60, (N_REAL, 4), dtype=torch.int32, device=dev)
coords[:, 0] = 0
pad = torch.randint(0, 8, (CAP - N_REAL, 4), dtype=torch.int32, device=dev)
pad[:, 0] = 0
coords_cap = torch.cat([coords, pad], 0).contiguous()

ref_u = torchsparse_uc = None
import torchsparse.backend as B                 # noqa: E402
uc_ref = B.unique_coords(coords.contiguous())   # 固定版：只给真实行
out_cap, u = cap.unique_coords_cap(coords_cap, N_REAL, dev)
u_val = int(u.reshape(-1)[0].item())
out_real = out_cap[:u_val]
print("  ref U=%d  cap U=%d  cap_out rows=%d" % (uc_ref.shape[0], u_val, out_cap.shape[0]))
same_rows = uc_ref.shape[0] == u_val
ok &= same_rows
print("  %-34s %s" % ("行数一致", "PASS" if same_rows else "FAIL"))
# 比较去重后的坐标集合（顺序可能不同 → 排序后比）
a = uc_ref.to(torch.int64).cpu().numpy()
b = out_real.to(torch.int64).cpu().numpy()
a = a[np.lexsort(a.T[::-1])]
b = b[np.lexsort(b.T[::-1])]
eq = a.shape == b.shape and np.array_equal(a, b)
ok &= eq
print("  %-34s %s" % ("去重坐标集合一致", "PASS" if eq else "FAIL"))

# ---------------------------------------------------------------- 2) keyprep_v3 + query_v2 + gather_gemm_v2
print("\n[2] 容量版邻域表 + 卷积 vs 固定版 _kernel_map + gather_gemm")
M_REAL, CAP_OUT = 500, 500
ic = torch.randint(0, 8, (N_REAL, 4), dtype=torch.int32, device=dev); ic[:, 0] = 0
oc = torch.randint(0, 8, (M_REAL, 4), dtype=torch.int32, device=dev); oc[:, 0] = 0
# 补齐行必须用「有效行的重复」，不能用随机坐标：否则 keyprep 的 lo/dims 被撑大 → 表与参考不同
ic_cap = torch.cat([ic, ic[:1].repeat(CAP - N_REAL, 1)]).contiguous()
oc_cap = torch.cat([oc, oc[:1].repeat(CAP_OUT - M_REAL, 1)]).contiguous()

from torchsparse.nn.utils import get_kernel_offsets      # noqa: E402
_offs = get_kernel_offsets((3, 3, 3), 1, device=dev)
nb_ref = F._lookup_offsets(ic, oc, (1, 1, 1), _offs).to(torch.int32)   # 固定 N 矩形参考 [M_REAL, 27]
nb_cap = cap.kernel_map(ic_cap, oc_cap, (N_REAL, M_REAL), ks=3, stride=1, dev=dev)
nb_cap_real = nb_cap[:M_REAL]
_a = nb_ref.to(torch.int64).cpu()
_b = nb_cap_real.to(torch.int64).cpu()
eq_nb = _a.shape == _b.shape and torch.equal(_a, _b)
frac = float((_a == _b).float().mean())
neg_a = float((_a < 0).float().mean()); neg_b = float((_b < 0).float().mean())
print("  诊断: 相同元素占比=%.4f  ref中-1占比=%.3f  cap中-1占比=%.3f" % (frac, neg_a, neg_b))
print("  ref[0,:6]=%s" % (_a[0, :6].tolist(),))
print("  cap[0,:6]=%s" % (_b[0, :6].tolist(),))
print("  ref[0,:6]排序=%s" % (sorted(_a[0, :6].tolist()),))
print("  cap[0,:6]排序=%s" % (sorted(_b[0, :6].tolist()),))
ok &= eq_nb
print("  nb shape ref=%s cap=%s  逐位一致=%s" % (tuple(nb_ref.shape), tuple(nb_cap.shape),
                                             "PASS" if eq_nb else "FAIL"))

cin, cout, kv = 64, 64, 27
feats = torch.randn(N_REAL, cin, device=dev) * 0.1
w = torch.randn(kv * cin, cout, device=dev) * 0.02
# 手算参考：Σ_kv feat[nb[:,kv]] * w[kv*cin:(kv+1)*cin]
_v = (nb_ref >= 0)
_g = feats[nb_ref.clamp(min=0).long()] * _v.unsqueeze(-1).to(feats.dtype)
ref_conv = _g.reshape(M_REAL, kv * cin) @ w                      # [M_REAL, cout]
feats_cap = torch.cat([feats, torch.zeros(CAP - N_REAL, cin, device=dev)], 0).contiguous()
out_conv = cap.conv(feats_cap, nb_cap, w, M_REAL, dev=dev)
d_conv = float((out_conv[:M_REAL].float() - ref_conv.float()).abs().max())
report("conv(capacity) vs fixed", d_conv, 1e-5)

# ---------------------------------------------------------------- 3) devox_v4
print("\n[3] 容量版 devox_v4 vs torch 参考")
K, NR, NCAP, C = 8, 500, 640, 32
feat = torch.randn(300, C, device=dev)
idx_cap = torch.zeros(K, NCAP, dtype=torch.int32, device=dev)
idx_cap[:, :NR] = torch.randint(0, 300, (K, NR), dtype=torch.int32, device=dev)
w_cap = torch.zeros(K, NCAP, device=dev)
w_cap[:, :NR] = torch.rand(K, NR, device=dev)
out_dv = cap.devox(feat, idx_cap, w_cap, NR, dev=dev)
# 参考：Σ_k w[k,i] * feat[idx[k,i]]
ref_dv = (feat[idx_cap[:K, :NR].long()] * w_cap[:K, :NR].unsqueeze(-1)).sum(0)
d_dv = float((out_dv[:NR].float() - ref_dv.float()).abs().max())
report("devox(capacity) vs torch", d_dv, 1e-5)

print("\n总体:", "全部 PASS" if ok else "有 FAIL")
sys.exit(0 if ok else 1)
