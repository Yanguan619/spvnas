"""SemanticKITTI mIoU 评估（NPU，复用原 evaluate.py 的逐帧流程与 core.callbacks.MeanIoU）。

与原 evaluate.py 的差异（只为能在单卡 NPU 上跑通）：
  1. 设备：.cuda() -> .npu()；MeanIoU 里的 dist.allreduce 打桩为恒等（单卡）；
  2. 模型：直接 SPVCNN(num_classes, cr, pres, vres)，不走 model_zoo（官方权重地址已 404）；
  3. 逐帧数据自己读：原 dataset 把 inverse_map / 逐点标签包进 SparseTensor，而 vendored
     SparseTensor 会对 coords 去重，导致 .F 与 .C 行数不一致；这里直接按点取，语义与
     原脚本一致（pred = outputs[voxel_of_point][inverse].argmax(1)，tgt = 每点标签）。

用法：
  python evaluate_npu.py --root <sequences> --ckpt <ckpt> --cr 0.5 --voxel 0.05 [--limit N]
"""
import argparse
import pathlib
import sys

import numpy as np
import torch

from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from core.datasets.semantic_kitti import SemanticKITTIInternal
from core.models.semantic_kitti.spvcnn import SPVCNN
import core.callbacks as callbacks_mod
from core.callbacks import MeanIoU


def iter_frames(root, seq='08', voxel=0.05, limit=0, label_map=None, offset=0):
    vdir = pathlib.Path(root) / seq / 'velodyne'
    ldir = pathlib.Path(root) / seq / 'labels'
    files = sorted(vdir.glob('*.bin'))
    if offset:
        files = files[offset:]
    if limit:
        files = files[:limit]
    for f in files:
        pts = np.fromfile(f, dtype=np.float32)
        if pts.size % 4:
            continue
        pts = pts.reshape(-1, 4)
        lf = ldir / (f.stem + '.label')
        if lf.exists():
            raw = np.fromfile(lf, dtype=np.int32).reshape(-1)
        else:
            raw = np.zeros(pts.shape[0], dtype=np.int32)
        if raw.size != pts.shape[0]:
            # 半截下载：.bin 与 .label 点数不一致，跳过（并计入统计）
            yield dict(bad=True)
            continue
        labels_pt = label_map[raw & 0xFFFF].astype(np.int64)

        pc_ = np.round(pts[:, :3] / voxel).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)
        uniq, inds, inv = sparse_quantize(pc_, return_index=True, return_inverse=True)
        yield dict(feat=pts[inds].astype(np.float32),
                   pc=uniq.astype(np.int32),
                   labels_pt=labels_pt,
                   inverse=inv.astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='sequences 目录（其下 08/velodyne/*.bin）')
    ap.add_argument('--seq', default='08')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--cr', type=float, default=0.5)
    ap.add_argument('--voxel', type=float, default=0.05)
    ap.add_argument('--num-points', type=int, default=180000)
    ap.add_argument('--num-classes', type=int, default=19)
    ap.add_argument('--ignore-label', type=int, default=255,
                    help='忽略类；本仓 label_map 把未标注映射为 255（见 configs）')
    ap.add_argument('--limit', type=int, default=0, help='只评估前 N 帧（0=全部）')
    ap.add_argument('--offset', type=int, default=0, help='从第 offset 帧开始')
    ap.add_argument('--device', default='npu:0')
    ap.add_argument('--fold-bn', action='store_true')
    ap.add_argument('--class-perm', default=None,
                    help='把模型输出通道映射到本仓类别顺序，如 5,7,8,...')
    ap.add_argument('--stats-jsonl', default=None, help='逐帧混淆统计落盘（可崩溃后聚合）')
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(0)

    # 单卡：把 MeanIoU 里的 dist.allreduce 打成恒等（原脚本依赖 torchpack 分布式）
    callbacks_mod.dist = type('D', (), {'allreduce': staticmethod(lambda v, reduction=None: v)})()

    # 复用数据集类只为拿 label_map（19 类映射表）
    ds = SemanticKITTIInternal(args.root, args.voxel, args.num_points, split='val')
    lm = ds.label_map

    model = SPVCNN(num_classes=args.num_classes, cr=args.cr,
                   pres=args.voxel, vres=args.voxel).eval()
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        sd = {k.replace('module.', '').replace('model.', ''): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print('[ckpt] %s  missing=%d unexpected=%d' % (args.ckpt, len(missing), len(unexpected)),
              flush=True)
    if args.fold_bn:
        from torchsparse.nn import fold_bn
        fold_bn(model)
    model = model.to(dev)

    perm = None
    if args.class_perm:
        perm = torch.tensor([int(x) for x in args.class_perm.split(',')], dtype=torch.long,
                            device=dev)

    iou = MeanIoU(args.num_classes, args.ignore_label)
    iou._before_epoch()

    n = 0
    with torch.no_grad():
        skipped = 0
        for fr in iter_frames(args.root, args.seq, args.voxel, args.limit, lm, args.offset):
            if fr.get('bad'):
                skipped += 1
                continue
            feat = torch.from_numpy(fr['feat']).to(dev)
            pc = torch.from_numpy(fr['pc']).to(dev)
            batch = torch.zeros(pc.shape[0], 1, dtype=pc.dtype, device=dev)
            x = SparseTensor(feat, torch.cat([batch, pc], 1), 1)
            outputs = model(x)

            inv = torch.from_numpy(fr['inverse']).long().to(dev)
            pred = outputs[inv].argmax(1)
            if perm is not None:
                pred = perm[pred]
            pred = pred.cpu()
            tgt = torch.from_numpy(fr['labels_pt']).long()
            iou._after_step({'outputs': pred, 'targets': tgt})
            if args.stats_jsonl:
                import json as _json
                m = (tgt != args.ignore_label)
                tt = tgt[m].numpy(); pp = pred[m].numpy()
                C = args.num_classes
                seen = np.bincount(tt, minlength=C)
                corr = np.bincount(tt[pp == tt], minlength=C)
                pos = np.bincount(pp, minlength=C)
                with open(args.stats_jsonl, 'a') as fh:
                    fh.write(_json.dumps({'i': args.offset + n,
                                          'seen': seen.tolist(),
                                          'correct': corr.tolist(),
                                          'positive': pos.tolist()}) + '\n')
            n += 1
            if n % 50 == 0:
                print('  ... %d 帧' % n, flush=True)

    print('[eval] 可用帧数=%d  跳过(半截/损坏)=%d' % (n, skipped), flush=True)
    iou._after_epoch()
    seen = float(iou.total_seen.sum())
    corr = float(iou.total_correct.sum())
    if seen > 0:
        print('[OA] %.4f%%   (total_seen=%.0f)' % (100.0 * corr / seen, seen), flush=True)
    print('### EVAL_NPU_DONE')


if __name__ == '__main__':
    sys.exit(main())
