"""CPU vs NPU benchmark for SPVCNN on the pure-torch torchsparse shim.

OMP_NUM_THREADS=1 in this image, so CPU is timed at 1, 16 and 96 threads.
"""
import argparse
import os
import resource
import statistics
import time

import numpy as np
import torch

from torchsparse import SparseTensor
from core.models.semantic_kitti.spvcnn import SPVCNN
import demo

VOXEL = demo.VOXEL


def subset(frame, n_points, seed=1):
    rng = np.random.default_rng(seed)
    total = frame["coords"].shape[0]
    k = min(n_points or total, total)
    keep = np.sort(rng.choice(total, k, replace=False))
    coords = np.concatenate([np.zeros((k, 1), np.int32),
                             frame["coords"][keep, 1:]], 1)
    return SparseTensor(torch.from_numpy(frame["feat"][keep]),
                        torch.from_numpy(coords), 1)


def median_time(fn, reps, dev):
    ts = []
    for _ in range(reps):
        t0 = time.time()
        fn()
        if dev.type == "npu":
            torch.npu.synchronize()
        ts.append(time.time() - t0)
    return statistics.median(ts)


def infer(model, lidar, dev, reps):
    x = SparseTensor(lidar.F.to(dev), lidar.C.to(dev), lidar.s)
    with torch.no_grad():
        model(SparseTensor(x.F, x.C, x.s))            # cold: builds kernel maps
    with torch.no_grad():
        return median_time(lambda: model(SparseTensor(x.F, x.C, x.s)), reps, dev)


def train_step(model, lidar, dev, steps):
    x = SparseTensor(lidar.F.to(dev), lidar.C.to(dev), lidar.s)
    tgt = torch.zeros(lidar.C.shape[0], dtype=torch.long, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = torch.nn.CrossEntropyLoss()
    model.train()
    t0 = time.time()
    for _ in range(steps):
        out = model(SparseTensor(x.F, x.C, x.s))
        loss = crit(out.float(), tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if dev.type == "npu":
        torch.npu.synchronize()
    model.eval()
    return (time.time() - t0) / steps


def stage_profile(model, lidar, dev, reps=3):
    acc, marks = {}, {}

    def hook_pre(name):
        def fn(mod, inp):
            if dev.type == "npu":
                torch.npu.synchronize()
            marks[name] = time.time()
        return fn

    def hook_post(name):
        def fn(mod, inp, out):
            if dev.type == "npu":
                torch.npu.synchronize()
            acc.setdefault(name, []).append(time.time() - marks[name])
        return fn

    handles = []
    for name, child in model.named_children():
        handles.append(child.register_forward_pre_hook(hook_pre(name)))
        handles.append(child.register_forward_post_hook(hook_post(name)))

    x = SparseTensor(lidar.F.to(dev), lidar.C.to(dev), lidar.s)
    with torch.no_grad():
        model(SparseTensor(x.F, x.C, x.s))
    acc.clear()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(reps):
            model(SparseTensor(x.F, x.C, x.s))
    if dev.type == "npu":
        torch.npu.synchronize()
    total = (time.time() - t0) / reps
    for h in handles:
        h.remove()
    rows = []
    for name in acc:
        ms = sum(acc[name]) / len(acc[name]) * 1e3
        rows.append((name, ms, ms / (total * 1e3)))
    return total * 1e3, sorted(rows, key=lambda r: -r[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", default="1,16,96")
    args = ap.parse_args()
    threads = [int(t) for t in args.threads.split(",")]

    frame = demo.load_frame("assets/000000.bin", "assets/000000.label")
    print(f"frame: {len(frame['xyz'])} points -> {frame['coords'].shape[0]} voxels")
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, "
          f"affinity={len(os.sched_getaffinity(0))} cpus, "
          f"torch_default_threads={torch.get_num_threads()}")

    import torch_npu  # noqa: F401
    dev_npu = torch.device("npu:0")
    print("npu:", torch.npu.get_device_name(0), flush=True)

    print("\n=== inference (median of %d) ===" % args.reps)
    header = "model/voxels".ljust(22) + "NPU".rjust(12) + \
        "".join(f"CPU{t}t".rjust(12) for t in threads) + "speedup(96t)".rjust(14)
    print(header)
    for n_pts, cr in [(20000, 0.25), (20000, 1.0), (None, 0.25), (None, 0.5), (None, 1.0)]:
        lidar = subset(frame, n_pts)
        nvox = lidar.C.shape[0]
        model_npu = SPVCNN(num_classes=19, cr=cr, pres=VOXEL, vres=VOXEL).eval().to(dev_npu)
        t_npu = infer(model_npu, lidar, dev_npu, args.reps)

        cpus = []
        for nt in threads:
            torch.set_num_threads(nt)
            model = SPVCNN(num_classes=19, cr=cr, pres=VOXEL, vres=VOXEL).eval()
            cpus.append(infer(model, lidar, torch.device("cpu"), args.reps))

        row = f"cr={cr} {nvox//1000}k voxels".ljust(22)
        row += f"{t_npu*1e3:9.1f} ms"
        row += "".join(f"{t*1e3:9.1f} ms" for t in cpus)
        row += f"{cpus[-1]/t_npu:11.2f}x"
        print(row, flush=True)

    print("\n=== training step (fwd + bwd + AdamW, full frame) ===")
    lidar = subset(frame, None)
    for cr in (0.25, 1.0):
        model_npu = SPVCNN(num_classes=19, cr=cr, pres=VOXEL, vres=VOXEL).to(dev_npu)
        t_npu = train_step(model_npu, lidar, dev_npu, 3)
        torch.set_num_threads(threads[-1])
        model = SPVCNN(num_classes=19, cr=cr, pres=VOXEL, vres=VOXEL)
        t_cpu = train_step(model, lidar, torch.device("cpu"), 2)
        print(f"cr={cr}: npu {t_npu*1e3:8.1f} ms/step   cpu({threads[-1]}t) "
              f"{t_cpu*1e3:8.1f} ms/step   speedup {t_cpu/t_npu:5.2f}x", flush=True)

    print("\n=== memory ===")
    print(f"cpu maxrss        {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB")
    print(f"npu peak allocated {torch.npu.max_memory_allocated()/2**30:.2f} GiB")

    print("\n=== stage profile, cr=1.0 full frame ===")
    torch.set_num_threads(threads[-1])
    for tag, dev in (("npu", dev_npu), ("cpu", torch.device("cpu"))):
        model = SPVCNN(num_classes=19, cr=1.0, pres=VOXEL, vres=VOXEL).eval().to(dev)
        total, rows = stage_profile(model, lidar, dev)
        print(f"{tag}: total {total:.1f} ms")
        for name, ms, frac in rows[:6]:
            print(f"  {name:16s} {ms:8.1f} ms  {frac:5.1%}")
        if tag == "npu":
            model = model.cpu()
    print("done")


if __name__ == "__main__":
    main()
