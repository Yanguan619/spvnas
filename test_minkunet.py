"""Same end-to-end check for MinkUNet (no point-voxel branch, pure sparse)."""
import time

import numpy as np
import torch
import torch_npu  # noqa: F401

from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from core.models.semantic_kitti.minkunet import MinkUNet


def make_batch(n=20000, voxel=0.05, seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform([0, -25, -2], [50, 25, 2], size=(n, 3))
    intensity = rng.random((n, 1))
    pc = np.round(xyz / voxel).astype(np.int32)
    pc -= pc.min(0)
    inds = sparse_quantize(pc, return_index=True)[1]
    pc = pc[inds]
    feat = np.concatenate([xyz, intensity], 1)[inds].astype(np.float32)
    return sparse_collate_fn([{"lidar": SparseTensor(feat, pc)}])["lidar"]


def run(model, lidar, dev, tag):
    t0 = time.time()
    with torch.no_grad():
        out = model(SparseTensor(lidar.F.to(dev), lidar.C.to(dev), lidar.s))
    if dev.type == "npu":
        torch.npu.synchronize()
    print(f"{tag}: out {tuple(out.shape)}  {time.time()-t0:.2f}s")
    return out


torch.manual_seed(0)
model = MinkUNet(num_classes=19, cr=1.0).eval()
print(f"MinkUNet params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
lidar = make_batch()
print("voxels", lidar.C.shape[0])

a = run(model, lidar, torch.device("cpu"), "cpu")
dev = torch.device("npu:0")
model = model.to(dev)
b = run(model, lidar, dev, "npu")
d = (a - b.cpu()).abs().max().item()
print(f"max |cpu-npu| = {d:.3e}  (logit scale {a.abs().max().item():.3f})")
assert d < 1e-3
print("SUCCESS")
