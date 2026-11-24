"""SemanticKITTI mIoU 评估（NPU 版，复用原仓库 evaluate.py 的流程与 core.callbacks.MeanIoU）。

与原 evaluate.py 的差异（只为能在单卡 NPU 上跑）：
  1. 设备：.cuda() -> .npu()；不做 DistributedDataParallel，dist.allreduce 打桩为恒等；
  2. 模型：直接用 builder.make_model() 的构造参数（SPVCNN(cr, pres, vres)），
     不经 model_zoo（官方下载地址 https://hanlab.mit.edu/files/SPVNAS/... 已 404）；
  3. checkpoint：--ckpt 显式给；
  4. 数据集：--root 指向 sequences 目录（其下 08/velodyne/*.bin），split 'test' 即 val(08)。

逐帧流程与原脚本一致：outputs -> inverse_map 映射 -> argmax -> MeanIoU。
"""
import argparse
import sys

import numpy as np
import torch
import torch.utils.data

from torchsparse import SparseTensor
from core.datasets.semantic_kitti import SemanticKITTIInternal
from core.models.semantic_kitti.spvcnn import SPVCNN
import core.callbacks as callbacks_mod
from core.callbacks import MeanIoU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='sequences 目录（其下 08/velodyne/*.bin）')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--cr', type=float, default=0.5)
    ap.add_argument('--voxel', type=float, default=0.05)
    ap.add_argument('--num-points', type=int, default=180000)
    ap.add_argument('--num-classes', type=int, default=19)
    ap.add_argument('--ignore-label', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='只评估前 N 帧（0=全部）')
    ap.add_argument('--device', default='npu:0')
    ap.add_argument('--fold-bn', action='store_true')
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(0)

    # 单卡：把 MeanIoU 里的 dist.allreduce 打成恒等（原脚本依赖 torchpack 分布式）
    callbacks_mod.dist = type('D', (), {'allreduce': staticmethod(lambda v, reduction=None: v)})()

    # 直接构造 val 分片（= 序列 08）。原仓库的 SemanticKITTI 包装类会同时构造 train
    # 分片（需要 00-07/09/10 全部序列），评估时不需要。
    split = SemanticKITTIInternal(args.root, args.voxel, args.num_points, split='val')
    files = split.files
    if args.limit:
        split.files = files[:args.limit]
    print('[env] frames=%d  root=%s  cr=%.2f  voxel=%.2f  device=%s'
          % (len(split.files), args.root, args.cr, args.voxel, dev), flush=True)

    loader = torch.utils.data.DataLoader(split, batch_size=1, num_workers=0,
                                         collate_fn=split.collate_fn)

    model = SPVCNN(num_classes=args.num_classes, cr=args.cr,
                   pres=args.voxel, vres=args.voxel).eval()
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        sd = {k.replace('module.', '').replace('model.', ''): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print('[ckpt] loaded %s  missing=%d unexpected=%d'
              % (args.ckpt, len(missing), len(unexpected)), flush=True)
    if args.fold_bn:
        from torchsparse.nn import fold_bn
        fold_bn(model)
    model = model.to(dev)

    iou = MeanIoU(args.num_classes, args.ignore_label)
    iou._before_epoch()

    with torch.no_grad():
        for i, feed_dict in enumerate(loader):
            inputs = feed_dict['lidar'].to(dev)
            outputs = model(inputs)
            invs = feed_dict['inverse_map']
            all_labels = feed_dict['targets_mapped']

            _outs, _tgts = [], []
            for idx in range(int(invs.C[:, 0].max()) + 1):
                cur_scene_pts = (inputs.C[:, 0] == idx).cpu().numpy()
                cur_inv = invs.F[invs.C[:, 0] == idx].cpu().numpy()
                cur_label = (all_labels.C[:, 0] == idx).cpu().numpy()
                outputs_mapped = outputs[cur_scene_pts][cur_inv].argmax(1)
                targets_mapped = all_labels.F[cur_label]
                _outs.append(outputs_mapped)
                _tgts.append(targets_mapped)
            out = torch.cat(_outs, 0).cpu()
            tgt = torch.cat(_tgts, 0).cpu()
            iou._after_step({'outputs': out, 'targets': tgt})
            if (i + 1) % 50 == 0:
                print('  ... %d 帧' % (i + 1), flush=True)

    iou._after_epoch()
    print('### EVAL_NPU_DONE')


if __name__ == '__main__':
    sys.exit(main())
