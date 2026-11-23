"""逐段 dump：容量 vs 参考，在每段的真实行上比较，定位第一个发散点。

用法：python3 stageC_dump.py <tag>     tag=ref | cap
"""
import os
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401

REPO = "/data2/jev_work/spvnas-ascend"
sys.path.insert(0, REPO)

TAG = sys.argv[1] if len(sys.argv) > 1 else "ref"

from torchsparse import SparseTensor                     # noqa: E402
from torchsparse.nn import fold_bn                       # noqa: E402
from core.models.semantic_kitti.spvcnn import SPVCNN     # noqa: E402
import core.models.utils as U                             # noqa: E402
import torchsparse.nn.functional as F                     # noqa: E402
import bench, demo                                        # noqa: E402

dev = torch.device("npu:0")
torch.manual_seed(0)
frame = demo.load_frame("assets/000000.bin", "assets/000000.label")
lidar = bench.subset(frame, 20000)
m = SPVCNN(num_classes=19, cr=0.5, pres=demo.VOXEL, vres=demo.VOXEL).eval()
sd = torch.load("demo_spvcnn.pt", map_location="cpu")
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
m.load_state_dict(sd, strict=False)
fold_bn(m)
m = m.to(dev)

out = {}
orig_iv = U.initial_voxelize


def spy_iv(z, a, b):
    r = orig_iv(z, a, b)
    out["init_vox"] = (r.F.detach().float().cpu().numpy(),
                       r.C.detach().int().cpu().numpy(),
                       -1 if r.n is None else int(r.n.reshape(-1)[0]))
    return r


U.initial_voxelize = spy_iv
import core.models.semantic_kitti.spvcnn as SM            # noqa: E402
SM.initial_voxelize = spy_iv

for nm, mod in [("stem", m.stem), ("stage1", m.stage1), ("stage2", m.stage2),
                ("stage3", m.stage3), ("stage4", m.stage4),
                ("up1a", m.up1[0]), ("up1b", m.up1[1]),
                ("up2a", m.up2[0]), ("up2b", m.up2[1]),
                ("up3a", m.up3[0]), ("up3b", m.up3[1]),
                ("up4a", m.up4[0]), ("up4b", m.up4[1])]:
    def hk(mod, inp, res, _n=nm):
        if hasattr(res, "F"):
            out[_n] = (res.F.detach().float().cpu().numpy(),
                       res.C.detach().int().cpu().numpy(),
                       -1 if getattr(res, "n", None) is None
                       else int(res.n.reshape(-1)[0]))
    mod.register_forward_hook(hk)

z = U.PointTensor(lidar.F.to(dev), lidar.C.to(dev), lidar.s)
print("[env] tag=%s TS_CAP=%s" % (TAG, os.environ.get("TS_CAP")), flush=True)
with torch.no_grad():
    y = m(z)      # 走模型真实 forward（含 v2p / 上采样 / 分类头），hook 才能全部触发
torch.npu.synchronize()
np.savez("/data2/jev_work/_fixer/scdump_%s.npz" % TAG,
         **{k + "_F": v[0] for k, v in out.items()},
         **{k + "_C": v[1] for k, v in out.items()},
         **{k + "_n": np.array(v[2]) for k, v in out.items()})
print("[dump]", TAG, {k: (v[0].shape, v[2]) for k, v in out.items()}, flush=True)
