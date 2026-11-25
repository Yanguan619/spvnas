"""统计 mIoU 评测数据（SemanticKITTI seq 08）的真实构成：每帧点数 + 类别分布。"""
import pathlib
import sys

import numpy as np

REPO = "/data2/jev_work/spvnas-ascend"
sys.path.insert(0, REPO)
from core.datasets.semantic_kitti import SemanticKITTIInternal  # noqa: E402

ROOT = pathlib.Path("/data2/jev_work/datasets/semantic-kitti/dataset/sequences/08")
ds = SemanticKITTIInternal(str(ROOT.parent), 0.05, 180000, split='val')
lm = ds.label_map

# 原始 id -> 名称（learning map 用到的那些）
# SemanticKITTI 原始 id -> 类别名（label 文件里存的就是这些 id）
RAW = {10: "car", 11: "bicycle", 13: "bus", 15: "motorcycle", 16: "on-rails",
       18: "truck", 20: "other-vehicle", 30: "person", 31: "bicyclist",
       32: "motorcyclist", 40: "road", 44: "parking", 48: "sidewalk",
       49: "other-ground", 50: "building", 51: "fence", 52: "vegetation",
       60: "trunk", 70: "terrain", 71: "pole", 72: "traffic-sign"}

bins = np.zeros(512, dtype=np.int64)
npts = []
vbins = []
for f in sorted((ROOT / "velodyne").glob("*.bin")):
    sz = f.stat().st_size
    n = sz // 16
    npts.append(n)
    lf = ROOT / "labels" / (f.stem + ".label")
    if lf.exists():
        raw = np.fromfile(lf, dtype=np.int32).reshape(-1)
        bins += np.bincount(raw & 0xFFFF, minlength=512)

npts = np.array(npts)
print("[seq08] 帧数=%d" % len(npts))
print("  每帧点数: 中位 %.0f  最小 %d  最大 %d  合计 %.1f M" %
      (np.median(npts), npts.min(), npts.max(), npts.sum() / 1e6))

tot = bins.sum()
print("\n  原始语义点合计: %.1f M" % (tot / 1e6))
valid = sum(bins[r] for r in RAW)
print("  标注率: 有效类 %.1f%%  未标注(0) %.1f%%  其他id %.1f%%"
      % (100 * valid / tot, 100 * bins[0] / tot, 100 * (tot - valid - bins[0]) / tot))
print("\n  %-14s %10s %7s" % ("类别", "点数(M)", "占比"))
rows = []
for rid, name in RAW.items():
    c = bins[rid]
    rows.append((c, name))
for c, name in sorted(rows, reverse=True):
    print("  %-14s %10.2f %6.2f%%" % (name, c / 1e6, 100 * c / tot))
print("### SCENE_STATS_DONE")
