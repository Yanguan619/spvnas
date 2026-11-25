"""Run SPVCNN end-to-end with the pure-torch torchsparse shim, CPU vs NPU."""
import os as _os
_os.environ.setdefault("TASK_QUEUE_ENABLE", "2")  # deep task queue: -8% @94k, -32% @20k
import time

import numpy as np
import torch

import torch_scatter
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
import core.models.semantic_kitti.spvcnn as spvcnn_mod
from core.models.semantic_kitti.spvcnn import SPVCNN
import core.models.utils as U


def exact_initial_voxelize(z, init_res, after_res):
    """SPVCNN's initial_voxelize computes coord*pres/vres, which is not exactly
    coord in fp32 and therefore yields slightly different voxel sets per device.
    For a cross-device comparison we keep the integer coords untouched."""
    new_int_coord = z.C.long()
    sparse_coord = torch.unique(new_int_coord, dim=0)
    idx_query = U.sphashquery(new_int_coord, sparse_coord).reshape(-1)
    sparse_feat = torch_scatter.scatter_mean(z.F, idx_query.long(), dim=0)
    out = SparseTensor(sparse_feat, sparse_coord, 1)
    z._caches.idx_query[z.s] = idx_query
    z.C = new_int_coord.float()
    return out


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
    x = SparseTensor(lidar.F.to(dev), lidar.C.to(dev), lidar.s)
    t0 = time.time()
    with torch.no_grad():
        out = model(x)
    if dev.type == "npu":
        torch.npu.synchronize()
    print(f"{tag}: out {tuple(out.shape)}  {time.time() - t0:.2f}s")
    return out


def main():
    torch.manual_seed(0)
    model = SPVCNN(num_classes=19, cr=1.0, pres=0.05, vres=0.05).eval()
    print(f"SPVCNN params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    lidar = make_batch()
    print(f"voxels: {lidar.C.shape[0]}  feats: {tuple(lidar.F.shape)}")

    out_cpu = run(model, lidar, torch.device("cpu"), "cpu (stock voxelize)")

    import torch_npu  # noqa: F401
    dev = torch.device("npu:0")
    print("npu:", torch.npu.get_device_name(0))
    model_npu = model.to(dev)
    out_npu = run(model_npu, lidar, dev, "npu (stock voxelize)")
    print(f"  stock max |cpu-npu| = {(out_cpu-out_npu.cpu()).abs().max().item():.3e}"
          "   (voxel sets differ by the model's own fp32 coord*pres/vres)")

    # apples-to-apples: identical coords on both devices
    orig = spvcnn_mod.initial_voxelize
    spvcnn_mod.initial_voxelize = exact_initial_voxelize
    try:
        model = model_npu.cpu()
        a = run(model, lidar, torch.device("cpu"), "cpu (exact voxelize)")
        model = model.to(dev)
        b = run(model, lidar, dev, "npu (exact voxelize)")
    finally:
        spvcnn_mod.initial_voxelize = orig
    diff = (a - b.cpu()).abs().max().item()
    print(f"  exact max |cpu-npu| = {diff:.3e}  (logit scale {a.abs().max().item():.3f})")
    assert diff < 1e-3, "cpu/npu mismatch"

    # training smoke test: autograd + BatchNorm update on NPU
    model.train()
    x = SparseTensor(lidar.F.to(dev), lidar.C.to(dev), lidar.s)
    t0 = time.time()
    out = model(x)
    loss = out.float().pow(2).mean()
    loss.backward()
    torch.npu.synchronize()
    print(f"train step ok: loss={loss.item():.4f}  {time.time()-t0:.2f}s")
    print("SUCCESS")


if __name__ == "__main__":
    main()
