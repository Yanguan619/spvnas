"""从 mIoU 评测的逐帧统计里取 seq08 的真实类别分布（模型 19 类空间）。"""
import json
import pathlib

import numpy as np

P = pathlib.Path("/data2/jev_work/_fixer/frame_stats2/stats.jsonl")
rows, seen_i = [], set()
for line in P.read_text().splitlines():
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r["i"] in seen_i:
        continue
    seen_i.add(r["i"])
    rows.append(r)

C = 19
seen = np.zeros(C, dtype=np.int64)
for r in rows:
    seen += np.array(r["seen"][:C], dtype=np.int64)

tot = seen.sum()
print("[seq08 评测集] 帧数=%d  参与统计的标注点=%.1f M" % (len(rows), tot / 1e6))
print("\n  按点数排序（模型 19 类，索引为训练时的类下标）")
order = np.argsort(-seen)
# 官方 seq08 常用分布（用于对齐名字，占比量级可比）
EXPECT = [
    ("road", 0.194), ("vegetation", 0.162), ("building", 0.135), ("sidewalk", 0.126),
    ("terrain", 0.092), ("car", 0.069), ("trunk", 0.043), ("fence", 0.026),
    ("parking", 0.015), ("pole", 0.013), ("other-ground", 0.005),
    ("other-vehicle", 0.004), ("truck", 0.002), ("bicycle", 0.001),
    ("motorcycle", 0.0005), ("person", 0.0003), ("bicyclist", 0.0001),
    ("traffic-sign", 0.0002), ("motorcyclist", 0.00005),
]
print("  %-4s %-16s %10s %8s" % ("idx", "推断类别", "点数(M)", "占比"))
for rank, i in enumerate(order):
    name = EXPECT[rank][0] if rank < len(EXPECT) else "?"
    print("  %-4d %-16s %10.2f %7.2f%%" % (i, name, seen[i] / 1e6, 100 * seen[i] / tot))
print("### IDXSTATS_DONE")
