"""分解 grouped 路径的成本：gather / GEMM / scatter-add 各自多少。"""
import statistics, sys, time, pathlib
import numpy as np
import torch, torch_npu  # noqa
REPO = "/data2/jev_work/spvnas-ascend"
sys.path.insert(0, REPO)
import ascend_meta  # noqa
import torchsparse.nn.functional as F
from torchsparse.utils.quantize import sparse_quantize

dev = torch.device("npu:0")
for op in ("keyprep_v2","keyprep_v1","query","nb"):
    F._load_cap_so(op)
ROOT = pathlib.Path("/data2/jev_work/datasets/semantic-kitti/dataset/sequences/08/velodyne")
pts = np.fromfile(sorted(ROOT.glob("*.bin"))[0], dtype=np.float32).reshape(-1,4)
pc_ = np.round(pts[:,:3]/0.05).astype(np.int32); pc_ -= pc_.min(0, keepdims=1)
uniq, inds, inv = sparse_quantize(pc_, return_index=True, return_inverse=True)
b = np.zeros((len(uniq),1), dtype=np.int32)
coords = torch.from_numpy(np.concatenate([b,uniq],1)).to(dev)
M = len(uniq)
nb = F._rect_map(coords, coords, (3,3,3), (1,1,1), 1,
                 torch.tensor(M,dtype=torch.int32,device=dev),
                 torch.tensor(M,dtype=torch.int32,device=dev))
CIN = COUT = 48; KV = nb.shape[1]
feat = torch.randn(M, CIN, device=dev)
w3 = (torch.randn(KV, CIN, COUT, device=dev)*0.05).contiguous()
rows_k, src_k = [], []
for k in range(KV):
    v = nb[:, k]; idx = torch.nonzero(v>=0).flatten()
    rows_k.append(idx); src_k.append(v[idx].long())

def T(fn, n=10):
    for _ in range(3): fn()
    torch.npu.synchronize()
    ts=[]
    for _ in range(n):
        t0=time.perf_counter(); fn(); torch.npu.synchronize(); ts.append(time.perf_counter()-t0)
    return statistics.median(ts)*1e6

# gather 只有
def only_gather():
    for k in range(KV): feat.index_select(0, src_k[k])
# gather + gemm
gs=[]
def gather_gemm():
    gs.clear()
    for k in range(KV):
        g = feat.index_select(0, src_k[k]); gs.append(g @ w3[k])
# scatter 只有
outs=[]
for k in range(KV): outs.append(torch.zeros(len(rows_k[k]), COUT, device=dev))
def only_scatter():
    out = torch.zeros(M, COUT, device=dev)
    for k in range(KV): out.index_add_(0, rows_k[k], outs[k])
def full():
    out = torch.zeros(M, COUT, device=dev)
    for k in range(KV):
        g = feat.index_select(0, src_k[k]); out.index_add_(0, rows_k[k], g @ w3[k])

print("  M=%d 有效条目=%d" % (M, sum(len(r) for r in rows_k)), flush=True)
print("  gather only      %8.0f us" % T(only_gather), flush=True)
print("  gather+gemm      %8.0f us" % T(gather_gemm), flush=True)
print("  scatter only     %8.0f us" % T(only_scatter), flush=True)
print("  full grouped     %8.0f us" % T(full), flush=True)
print("### COMP_DONE")
