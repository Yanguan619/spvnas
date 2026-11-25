"""行内相邻 tap 的源是否连续（零布局改动变体的前提）。"""
import sys, pathlib
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
for fi, f0 in enumerate(sorted(ROOT.glob("*.bin"))[:2]):
    pts = np.fromfile(f0, dtype=np.float32).reshape(-1,4)
    pc_ = np.round(pts[:,:3]/0.05).astype(np.int32); pc_ -= pc_.min(0, keepdims=1)
    uniq, inds, inv = sparse_quantize(pc_, return_index=True, return_inverse=True)
    b = np.zeros((len(uniq),1), dtype=np.int32)
    coords = torch.from_numpy(np.concatenate([b,uniq],1)).to(dev)
    M = len(uniq)
    nb = F._rect_map(coords, coords, (3,3,3), (1,1,1), 1,
                     torch.tensor(M,dtype=torch.int32,device=dev),
                     torch.tensor(M,dtype=torch.int32,device=dev)).cpu().numpy().astype(np.int64)
    KV = nb.shape[1]
    runs, valid = [], 0
    for m in range(M):
        v = nb[m]
        k = 0
        while k < KV:
            if v[k] < 0:
                k += 1; continue
            valid += 1
            r = 1
            while k + r < KV and v[k+r] >= 0 and v[k+r] == v[k] + r:
                r += 1
            runs.append(r); k += r
    runs = np.array(runs)
    print("帧%d M=%d 有效tap=%d run数=%d 平均run=%.2f run>=2=%.1f%%"
          % (fi, M, valid, len(runs), runs.mean(), 100*(runs>=2).mean()), flush=True)
print("### RM2_DONE")
