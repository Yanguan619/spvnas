"""验证「耗时 ∝ 拷贝条数」：固定数据量结构，只改有效 tap 数。"""
import statistics, sys, time
import torch, torch_npu  # noqa
REPO = "/data2/jev_work/spvnas-ascend"
sys.path.insert(0, REPO)
import ascend_meta  # noqa
import torchsparse.nn.functional as F

dev = torch.device("npu:0")
for op in ("gather_gemm_v2","gather_gemm"):
    F._load_cap_so(op)
M, KV, CIN = 94625, 27, 48
feat = torch.randn(M, CIN, device=dev)
w = (torch.randn(KV*CIN, CIN, device=dev)*0.05).contiguous()
nT = torch.tensor([M], dtype=torch.int32, device=dev)
op = torch.ops.npu.spconv_gather_gemm_v2

print("  有效tap数  耗时us   每条ns   数据MB", flush=True)
for valid_k in (27, 14, 7, 4, 1):
    nb = torch.full((M, KV), -1, dtype=torch.int32, device=dev)
    base = torch.arange(M, device=dev, dtype=torch.int32).view(M,1)
    for k in range(valid_k):
        nb[:, k] = (base + (k - valid_k//2)).clamp_(0, M-1).flatten()
    nb = nb.contiguous()
    for _ in range(3):
        op(feat, nb, w, nT)
    torch.npu.synchronize()
    ts = []
    for _ in range(10):
        t0 = time.perf_counter(); op(feat, nb, w, nT); torch.npu.synchronize()
        ts.append(time.perf_counter()-t0)
    med = statistics.median(ts)
    ncopies = M*valid_k
    print("  %-10d %8.0f %8.2f %8.0f" % (valid_k, med*1e6, med*1e9/ncopies,
                                         M*valid_k*CIN*4/2**20), flush=True)
print("### CSCALE_DONE")
