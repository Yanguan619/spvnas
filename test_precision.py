"""Is the CPU/NPU gap numerical, or a shim bug?

Compares the same SPVCNN in float64-CPU, float32-CPU and float32-NPU.
"""
import copy
import time

import numpy as np
import torch

from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from core.models.semantic_kitti.spvcnn import SPVCNN


def make_batch(n=4000, voxel=0.05, seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform([0, -25, -2], [50, 25, 2], size=(n, 3))
    intensity = rng.random((n, 1))
    pc = np.round(xyz / voxel).astype(np.int32)
    pc -= pc.min(0)
    inds = sparse_quantize(pc, return_index=True)[1]
    pc = pc[inds]
    feat = np.concatenate([xyz, intensity], 1)[inds].astype(np.float32)
    return sparse_collate_fn([{"lidar": SparseTensor(feat, pc)}])["lidar"]


torch.manual_seed(0)
model = SPVCNN(num_classes=19, cr=1.0, pres=0.05, vres=0.05).eval()
lidar = make_batch()
print("voxels", lidar.C.shape[0])

with torch.no_grad():
    out32 = model(SparseTensor(lidar.F.clone(), lidar.C.clone(), lidar.s))
    out32b = model(SparseTensor(lidar.F.clone(), lidar.C.clone(), lidar.s))
print("cpu float32 rerun identical:", torch.equal(out32, out32b))

model64 = copy.deepcopy(model).double()
with torch.no_grad():
    out64 = model64(SparseTensor(lidar.F.double(), lidar.C.clone(), lidar.s))
scale = out64.abs().max().item()
d = (out32.double() - out64).abs().max().item()
print(f"logit scale {scale:.3f}  cpu fp32 vs fp64 max diff {d:.3e} ({d/scale:.2%})")

import torch_npu  # noqa: F401
dev = "npu:0"
model_npu = model.to(dev)
with torch.no_grad():
    t0 = time.time()
    out_npu = model_npu(SparseTensor(lidar.F.to(dev), lidar.C.to(dev), lidar.s))
    torch.npu.synchronize()
    print(f"npu run {time.time()-t0:.2f}s")
dn = (out_npu.double().cpu() - out64).abs().max().item()
print(f"npu fp32 vs fp64 max diff {dn:.3e} ({dn/scale:.2%})")
