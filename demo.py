"""SPVCNN demo on Ascend NPU with the pure-torch torchsparse shim.

Uses the real SemanticKITTI frame bundled in assets/000000.bin.

    python3 demo.py                 # inference only (random init)
    python3 demo.py --iters 150     # overfit this frame first, save curves
"""
import argparse
import time

import cv2
import numpy as np
import torch

from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from core.datasets.semantic_kitti import kept_labels, label_name_mapping
from core.models.semantic_kitti.spvcnn import SPVCNN

VOXEL = 0.05
# 19 distinct colours, one per class index produced by label_map()
PALETTE = np.array([[245, 150, 100], [245, 230, 100], [180, 30, 80], [30, 30, 255],
                    [200, 40, 255], [255, 0, 255], [255, 150, 255], [75, 0, 75],
                    [175, 60, 40], [150, 240, 80], [230, 240, 40], [0, 200, 255],
                    [0, 150, 100], [0, 175, 0], [128, 60, 128], [150, 100, 30],
                    [120, 180, 60], [0, 60, 135], [80, 240, 150]], dtype=np.uint8)


def class_names():
    """Names in label_map index order (car first, not road)."""
    names = []
    for lid in label_name_mapping:
        name = label_name_mapping[lid]
        if lid > 250 or lid == 0:
            continue
        if name in kept_labels:
            names.append(name)
    return names


def label_map():
    m = np.zeros(260, dtype=np.int64)
    rev = {}
    cnt = 0
    for lid in label_name_mapping:
        name = label_name_mapping[lid]
        if lid > 250:
            m[lid] = rev.get(name.replace("moving-", ""), 255)
        elif lid == 0:
            m[lid] = 255
        elif name in kept_labels:
            m[lid] = cnt
            rev[name] = cnt
            cnt += 1
        else:
            m[lid] = 255
    return m


def load_frame(bin_path, label_path):
    raw = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    pc = np.round(raw[:, :3] / VOXEL).astype(np.int32)
    pc -= pc.min(0)
    _, inds, inv = sparse_quantize(pc, return_index=True, return_inverse=True)
    labels = np.fromfile(label_path, dtype=np.int32).reshape(-1) & 0xFFFF
    labels = label_map()[labels]
    coords4 = np.concatenate([np.zeros((len(inds), 1), np.int32), pc[inds]], 1)
    return {
        "xyz": raw[:, :3].astype(np.float32),       # all points, for rendering
        "labels": labels,                            # all points
        "feat": np.concatenate([raw[inds, :3], raw[inds, 3:4]], 1).astype(np.float32),
        "coords": coords4,
        "voxel_labels": labels[inds],
        "inv": inv,
    }


def render_bev(xyz, colors, path, size=900, span=None):
    span = span or (np.abs(xyz[:, :2]).max() * 1.05)
    img = np.full((size, size, 3), 255, np.uint8)
    u = ((xyz[:, 0] / span) * 0.5 + 0.5) * (size - 1)
    v = ((-xyz[:, 1] / span) * 0.5 + 0.5) * (size - 1)
    img[v.astype(int), u.astype(int)] = colors
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--cr", type=float, default=0.25)
    ap.add_argument("--out", default=".")
    ap.add_argument("--save-model", default=None)
    ap.add_argument("--load-model", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    frame = load_frame("assets/000000.bin", "assets/000000.label")
    coords, feat = frame["coords"], frame["feat"]
    print(f"frame: {len(frame['xyz'])} points -> {len(coords)} voxels")

    import torch_npu  # noqa: F401
    dev = torch.device("npu:0")
    print("device:", torch.npu.get_device_name(0))

    model = SPVCNN(num_classes=19, cr=args.cr, pres=VOXEL, vres=VOXEL).to(dev)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, map_location=dev))
        print(f"loaded {args.load_model}")
    print(f"model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params (cr={args.cr})")

    lidar = SparseTensor(torch.from_numpy(feat).to(dev),
                         torch.from_numpy(coords).to(dev), 1)
    tgt = torch.from_numpy(frame["voxel_labels"]).to(dev)

    if args.iters:
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
        crit = torch.nn.CrossEntropyLoss(ignore_index=255)
        t0 = time.time()
        for it in range(args.iters):
            out = model(SparseTensor(lidar.F, lidar.C, 1))
            loss = crit(out.float(), tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % 10 == 0 or it == args.iters - 1:
                print(f"  iter {it:3d}  loss {loss.item():.4f}  "
                      f"{(time.time()-t0)/(it+1):.2f}s/it", flush=True)
        torch.npu.synchronize()
        print(f"overfit {args.iters} iters in {time.time()-t0:.1f}s")
        if args.save_model:
            torch.save(model.state_dict(), args.save_model)
            print(f"saved weights to {args.save_model}")

    model.eval()
    with torch.no_grad():
        t0 = time.time()
        out = model(SparseTensor(lidar.F, lidar.C, 1))
        torch.npu.synchronize()
        dt = time.time() - t0
    print(f"inference: {dt*1000:.0f} ms  ({len(coords)/dt/1e3:.0f}k voxels/s)")

    logits = out.float().cpu().numpy()
    point_pred = logits[frame["inv"]].argmax(1)
    cnt = np.bincount(point_pred, minlength=19)
    top = np.argsort(cnt)[::-1][:5]
    names = class_names()
    print("top predicted classes:", ", ".join(f"{names[i]} {cnt[i]}" for i in top))

    valid = frame["labels"] != 255
    acc = (point_pred[valid] == frame["labels"][valid]).mean()
    print(f"point accuracy on this frame: {acc:.2%}")

    render_bev(frame["xyz"], PALETTE[np.where(frame["labels"] == 255, 0, frame["labels"])],
               f"{args.out}/demo_gt.png")
    render_bev(frame["xyz"], PALETTE[point_pred], f"{args.out}/demo_pred.png")
    print(f"saved {args.out}/demo_gt.png, demo_pred.png")


if __name__ == "__main__":
    main()
