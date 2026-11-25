"""纯 torch 稀疏卷积路径的精度复现用例（已知缺陷，见 ASCEND_ADAPTATION.md §8）。

结论（实测）：装载 AscendC 算子的路径与「手写稠密真值」吻合到 3e-7；
不装算子（纯 torch 路径）与 4 种 kernel 索引约定**全都不符**（最小误差 2.40），
说明纯 torch 的 conv3d 实现本身算错，而不是索引约定差异。

用法：
    # 装载算子（应 PASS）
    UNUM_OPS_ROOT=<ops> SPVNAS_OPS_ROOT=<ops> python test_conv3d_repro.py ops
    # 纯 torch（应 FAIL）
    python test_conv3d_repro.py noops

真值约定（x-major，实测与算子实现一致）：
    k = ((x + 1) * 3 + (y + 1)) * 3 + (z + 1)
"""
import itertools
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

import npu_patches  # noqa: E402
npu_patches.install(verbose=False, require_ops=False)

from torchsparse import SparseTensor                      # noqa: E402
import torchsparse.nn as spnn                             # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
N, CIN, COUT, G = 1200, 8, 8, 20

pick = np.random.RandomState(1234).choice(G ** 3, N, replace=False)
xyz = np.stack([pick // (G * G), (pick // G) % G, pick % G], 1).astype(np.int32)
order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
xyz = xyz[order]
coords = np.concatenate([np.zeros((N, 1), np.int32), xyz], 1)

torch.manual_seed(0)
conv = spnn.Conv3d(CIN, COUT, 3, bias=False)
with torch.no_grad():
    conv.kernel.normal_(0, 0.2)
W = conv.kernel.detach().float().clone()          # (27, cin, cout)
feats = torch.randn(N, CIN) * 0.3
key2row = {tuple(c[1:]): i for i, c in enumerate(coords)}

# 收集每个 (row, k) 的贡献来源，便于复用
per_k = {}
for dz, dy, dx in itertools.product((-1, 0, 1), repeat=3):
    rows, nbr = [], []
    for r, c in enumerate(coords):
        t = (int(c[1]) + dz, int(c[2]) + dy, int(c[3]) + dx)
        q = key2row.get(t)
        if q is not None:
            rows.append(r)
            nbr.append(q)
    per_k[(dz, dy, dx)] = (torch.tensor(rows).long(), torch.tensor(nbr).long())


def gt_for(kformula):
    gt = torch.zeros(N, COUT)
    for (dz, dy, dx), (rows, nbr) in per_k.items():
        if len(rows) == 0:
            continue
        gt.index_add_(0, rows, feats[nbr] @ W[kformula(dz, dy, dx)])
    return gt


FORMS = {
    "z-major":  lambda z, y, x: ((z + 1) * 3 + (y + 1)) * 3 + (x + 1),
    "x-major":  lambda z, y, x: ((x + 1) * 3 + (y + 1)) * 3 + (z + 1),
    "z-flip":   lambda z, y, x: ((1 - z) * 3 + (1 - y)) * 3 + (1 - x),
    "x-flip":   lambda z, y, x: ((1 - x) * 3 + (1 - y)) * 3 + (1 - z),
}

dev = torch.device("npu:0")
conv = conv.to(dev)
x = SparseTensor(feats.to(dev), torch.from_numpy(coords).to(dev), 1)
with torch.no_grad():
    o = conv(x)
    y = o.F if hasattr(o, "F") else o
torch.npu.synchronize()
impl = y.float().cpu()

out = {"tag": TAG, "impl_absmax": round(float(impl.abs().max()), 6)}
best = None
for name, f in FORMS.items():
    g = gt_for(f)
    d = float((impl - g).abs().max())
    nbad = int(((impl - g).abs().max(1).values > 1e-4).sum())
    out[name] = {"max_abs_diff": round(d, 8), "rows_bad": nbad,
                 "gt_absmax": round(float(g.abs().max()), 6)}
    if best is None or d < best[1]:
        best = (name, d)
out["best"] = best[0]
out["best_d"] = round(best[1], 8)
np.save("/data2/jev_work/_mz/ab_impl_%s.npy" % TAG, impl.numpy())
with open("/data2/jev_work/_mz/abk_%s.json" % TAG, "w") as fh:
    json.dump(out, fh, indent=1)
print("[%s] %s" % (TAG, json.dumps(out)))
print("### ABK_DONE")
