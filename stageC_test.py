"""Stage C 整模对拍：TS_CAP 开/关两种模式跑同一帧，比较 logits。

用法：python3 stageC_test.py <tag>   # tag=ref | cap
"""
import os
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401

REPO = "/data2/jev_work/spvnas-ascend"
sys.path.insert(0, REPO)

TAG = sys.argv[1] if len(sys.argv) > 1 else "ref"

from torchsparse import SparseTensor                            # noqa: E402
from torchsparse.nn import fold_bn                              # noqa: E402
from core.models.semantic_kitti.spvcnn import SPVCNN            # noqa: E402
import bench, demo                                              # noqa: E402

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

x = SparseTensor(lidar.F.to(dev), lidar.C.to(dev), lidar.s)
import torchsparse.nn.functional as F
print("[env] tag=%s TS_CAP=%s cap_enabled=%s" % (TAG, os.environ.get("TS_CAP"), F.cap_enabled()),
      flush=True)

with torch.no_grad():
    y = m(x)
torch.npu.synchronize()
arr = y.float().cpu().numpy()
np.save("/data2/jev_work/_fixer/stageC_%s.npy" % TAG, arr)
print("[out] %s shape=%s sum=%.6f absmax=%.4f argmax头8=%s"
      % (TAG, arr.shape, float(arr.sum()), float(np.abs(arr).max()),
         arr.argmax(1)[:8].tolist()), flush=True)
print("[done]", TAG, flush=True)
