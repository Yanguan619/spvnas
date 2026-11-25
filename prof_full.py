"""整帧 profiling：eager vs GE 静态，导出 MindStudio 可分析的数据。

用法：PERF_MODE=static TS_CAP=98304 python3 prof_full.py <out_tag>
"""
import os
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401
import torchair  # noqa: F401
from torchair import get_npu_backend  # noqa: E402

REPO = "/data2/jev_work/spvnas-ascend"
sys.path.insert(0, REPO)

import ascend_meta  # noqa: F401,E402
from torchsparse import SparseTensor                            # noqa: E402
from torchsparse.nn import fold_bn                              # noqa: E402
from torchsparse.utils.quantize import sparse_quantize          # noqa: E402
from core.models.semantic_kitti.spvcnn import SPVCNN            # noqa: E402
import torchsparse.nn.functional as F                           # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "eager"
CAP = int(os.environ.get("TS_CAP", "0") or 0)
MODE = os.environ.get("PERF_MODE", "eager")     # eager | static | dynamic
OUT = "/data2/jev_work/prof_%s" % TAG
ROOT = "/data2/jev_work/datasets/semantic-kitti/dataset/sequences"
dev = torch.device("npu:0")
print("[env] tag=%s CAP=%d mode=%s" % (TAG, CAP, MODE), flush=True)

for _op in ("keyprep_v3", "query_v2", "gather_gemm_v2", "sortdedup_v4", "sortdedup_v3",
            "keyprep_v2", "keyprep_v1", "gather_gemm", "gather", "gather_v2", "query",
            "devox", "devox_v3", "nb"):
    F._load_cap_so(_op)

f0 = sorted((__import__("pathlib").Path(ROOT) / "08" / "velodyne").glob("*.bin"))[0]
pts = np.fromfile(f0, dtype=np.float32).reshape(-1, 4)
pc_ = np.round(pts[:, :3] / 0.05).astype(np.int32)
pc_ -= pc_.min(0, keepdims=1)
uniq, inds, inv = sparse_quantize(pc_, return_index=True, return_inverse=True)
_b = np.zeros((len(uniq), 1), dtype=np.int32)
x = SparseTensor(torch.from_numpy(pts[inds].astype(np.float32)).to(dev),
                 torch.from_numpy(np.concatenate([_b, uniq], 1)).to(dev), 1)
print("[data] voxels=%d" % len(uniq), flush=True)

m = SPVCNN(num_classes=19, cr=0.5, pres=0.05, vres=0.05).eval()
m.load_state_dict(torch.load("/data2/jev_work/_ckpt/spvcnn30/spvcnn30_mit.pt",
                             map_location="cpu"), strict=True)
fold_bn(m)
m = m.to(dev)
if CAP:
    F.set_cap(CAP)

model = m
if MODE != "eager":
    model = torch.compile(m, backend=get_npu_backend(), dynamic=(MODE == "dynamic"))
    with torch.no_grad():
        model(x)                    # 触发编译，不计入 profile
    torch.npu.synchronize()
    print("[compile] done", flush=True)

exp = torch_npu.profiler._ExperimentalConfig(
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    export_type=[torch_npu.profiler.ExportType.Db],
)
WAIT, WARM, ACTIVE = 1, 2, 3
with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=WAIT, warmup=WARM, active=ACTIVE, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(OUT),
        experimental_config=exp) as prof:
    with torch.no_grad():
        for _ in range(WAIT + WARM + ACTIVE):
            model(x)
            prof.step()
torch.npu.synchronize()
print("[prof] 数据目录:", OUT, flush=True)
print("### PROF_RUN_DONE")
