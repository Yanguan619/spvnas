"""算法验证：按 k 分组做「紧凑 gather + GEMM + scatter-add」，与现有 fused kernel 对比。

背景：真实数据里 27 个邻居只有 11% 有效 → 现有 kernel 把 (M × 27*cin) 的矩阵
89% 填零再送给 cube。按 k 分组后每个 k 只处理真正有邻居的行（~10-15k），
cube 工作量与 staging 都降一个数量级。
"""
import statistics
import sys
import time

import numpy as np
import torch
import torch_npu  # noqa: F401
import pathlib

REPO = "/data2/jev_work/spvnas-ascend"
sys.path.insert(0, REPO)
import ascend_meta  # noqa: F401,E402
import torchsparse.nn.functional as F  # noqa: E402
from torchsparse.utils.quantize import sparse_quantize  # noqa: E402

dev = torch.device("npu:0")
for _op in ("keyprep_v2", "keyprep_v1", "query", "nb", "sortdedup_v3",
            "gather_gemm_v2", "gather_gemm"):
    F._load_cap_so(_op)

ROOT = pathlib.Path("/data2/jev_work/datasets/semantic-kitti/dataset/sequences/08/velodyne")
f0 = sorted(ROOT.glob("*.bin"))[0]
pts = np.fromfile(f0, dtype=np.float32).reshape(-1, 4)
pc_ = np.round(pts[:, :3] / 0.05).astype(np.int32)
pc_ -= pc_.min(0, keepdims=1)
uniq, inds, inv = sparse_quantize(pc_, return_index=True, return_inverse=True)
b = np.zeros((len(uniq), 1), dtype=np.int32)
coords = torch.from_numpy(np.concatenate([b, uniq], 1)).to(dev)
M = len(uniq)

for CIN, COUT in ((48, 48), (128, 128), (16, 16)):
    print("\n=== cin=cout=%d  rows=%d ===" % (CIN, M), flush=True)
    nb = F._rect_map(coords, coords, (3, 3, 3), (1, 1, 1), 1,
                     torch.tensor(M, dtype=torch.int32, device=dev),
                     torch.tensor(M, dtype=torch.int32, device=dev))
    KV = nb.shape[1]
    valid = (nb >= 0)
    print("  有效邻居 %d / %d (%.1f%%)" % (int(valid.sum()), nb.numel(),
                                           100.0 * int(valid.sum()) / nb.numel()), flush=True)

    feat = torch.randn(M, CIN, device=dev)
    w2d = (torch.randn(KV * CIN, COUT, device=dev) * 0.05).contiguous()
    w3d = w2d.view(KV, CIN, COUT).contiguous()
    nT = torch.tensor([M], dtype=torch.int32, device=dev)
    out_ref = torch.ops.npu.spconv_gather_gemm_v2(feat, nb, w2d, nT)

    # ---- 按 k 分组：紧凑 gather + GEMM + scatter-add ----
    rows_k, src_k = [], []
    for k in range(KV):
        v = nb[:, k]
        idx = torch.nonzero(v >= 0).flatten()
        rows_k.append(idx)
        src_k.append(v[idx].long())

    def grouped():
        out = torch.zeros(M, COUT, device=dev)
        for k in range(KV):
            g = feat.index_select(0, src_k[k])          # [c_k, CIN]
            o = g @ w3d[k]                              # [c_k, COUT]
            out.index_add_(0, rows_k[k], o)
        return out

    def fused():
        return torch.ops.npu.spconv_gather_gemm_v2(feat, nb, w2d, nT)

    def bench(fn, tag):
        for _ in range(3):
            fn()
        torch.npu.synchronize()
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            r = fn()
            torch.npu.synchronize()
            ts.append(time.perf_counter() - t0)
        med = statistics.median(ts)
        print("  %-14s %8.0f us" % (tag, med * 1e6), flush=True)
        return med, r

    t_f, r_f = bench(fused, "fused kernel")
    t_g, r_g = bench(grouped, "grouped(k)")
    err = (r_f - r_g).abs().max().item()
    print("  最大绝对差 %.3e   加速比 %.2fx" % (err, t_f / t_g), flush=True)
print("### GROUPED_DONE")
