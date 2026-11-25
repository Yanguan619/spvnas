"""把 HF(torch-pointcloud) 的 SPVCNN 权重映射成本仓 SPVCNN 的 state_dict。"""
import json, struct, sys
import numpy as np
import torch

sys.path.insert(0, "/data2/jev_work/spvnas-ascend")
from core.models.semantic_kitti.spvcnn import SPVCNN

SAFE = "/data2/jev_work/_ckpt/spvcnn30/model.safetensors"
OUT = "/data2/jev_work/_ckpt/spvcnn30/spvcnn30_mit.pt"
DT = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32,
      "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        raw = f.read()
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        off, end = v["data_offsets"]      # safetensors: [start, end]，不是 [start, len]
        buf = raw[off:end]
        if v["dtype"] == "BF16":
            u16 = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
            arr = u16.view(np.float32)
        else:
            arr = np.frombuffer(buf, dtype=DT[v["dtype"]])
        arr = np.asarray(arr).reshape(v["shape"] if v["shape"] else ())
        out[k] = torch.from_numpy(arr.copy())
    return out


m = SPVCNN(num_classes=19, cr=0.5, pres=0.05, vres=0.05)
ours = list(m.state_dict().items())          # 定义顺序
hf = load_safetensors(SAFE)
hf_keys = list(hf.keys())                    # safetensors 内已是按名字排序
# 按“形状 + 名字结构”做单调对齐：遍历 ours，在 hf 中从当前游标向后找同形状的键
mapping, cur, misses = {}, 0, []
for k, v in ours:
    sh = tuple(v.shape)
    found = None
    for j in range(cur, len(hf_keys)):
        hk = hf_keys[j]
        if tuple(hf[hk].shape) == sh:
            found, cur = hk, j
            break
    if found is None:
        misses.append(k)
    else:
        mapping[k] = found
print("  ours=%d  mapped=%d  misses=%d" % (len(ours), len(mapping), len(misses)))
if misses:
    print("  未匹配:", misses[:10])

sd = {}
for k, v in ours:
    hk = mapping.get(k)
    if hk is None:
        continue
    t = hf[hk]
    if t.shape != v.shape:
        print("  !! shape 不符:", k, tuple(v.shape), hk, tuple(t.shape)); continue
    sd[k] = t.to(v.dtype)

print("  mapped tensors=%d / %d" % (len(sd), len(ours)))
missing, unexpected = m.load_state_dict(sd, strict=False)
print("  load_state_dict: missing=%d unexpected=%d" % (len(missing), len(unexpected)))
if missing[:5]:
    print("   missing 示例:", missing[:5])
torch.save(sd, OUT)
print("  saved:", OUT)

# 合理性检查：同一帧，转换权重 vs demo 权重
from torchsparse import SparseTensor
import bench, demo
dev = torch.device("npu:0")
m = m.eval().to(dev)
frame = demo.load_frame("assets/000000.bin", "assets/000000.label")
lid = bench.subset(frame, 20000)
x = SparseTensor(lid.F.to(dev), lid.C.to(dev), lid.s)
with torch.no_grad():
    y = m(x)
torch.npu.synchronize()
pred = y.argmax(1).cpu().numpy()
import numpy as np
print("  logits: absmax=%.3f  sum=%.1f" % (float(y.abs().max()), float(y.sum())))
print("  argmax 分布:", np.bincount(pred, minlength=19).tolist())
print("### CONVERT_DONE")
