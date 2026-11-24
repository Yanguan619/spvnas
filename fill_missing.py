"""补齐缺失帧：小分块跑，块内失败则二分到单帧重试。

统计以 frame_stats2/stats.jsonl 为准（每行一帧，含 i）。
"""
import json
import pathlib
import subprocess
import sys

REPO = "/data2/jev_work/spvnas-ascend"
STATS = pathlib.Path("/data2/jev_work/_fixer/frame_stats2/stats.jsonl")
ROOT = "/data2/jev_work/datasets/semantic-kitti/dataset/sequences"
CK = "/data2/jev_work/_ckpt/spvcnn30/spvcnn30_mit.pt"
TOTAL = len(list((pathlib.Path(ROOT) / "08" / "velodyne").glob("*.bin")))
BLOCK = 20
RETRY = 2


def have_set():
    s = set()
    if STATS.exists():
        for line in STATS.read_text().splitlines():
            try:
                s.add(json.loads(line)["i"])
            except Exception:
                pass
    return s


def run(off, limit):
    before = len(have_set())
    cmd = [sys.executable, "-u", "evaluate_npu.py",
           "--root", ROOT, "--ckpt", CK, "--cr", "0.5", "--voxel", "0.05",
           "--ignore-label", "255", "--limit", str(limit), "--offset", str(off),
           "--stats-jsonl", str(STATS)]
    try:
        subprocess.run(cmd, cwd=REPO, timeout=1800,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        pass
    return len(have_set()) - before


def fill(off, limit, depth=0):
    """返回本段补到的帧数。"""
    got = run(off, limit)
    if got >= limit:
        return got
    if limit == 1:
        return got
    # 二分
    half = limit // 2
    a = fill(off, half, depth + 1)
    b = fill(off + half, limit - half, depth + 1)
    return a + b


def main():
    have = have_set()
    missing = [i for i in range(TOTAL) if i not in have]
    print("[fill] total=%d have=%d missing=%d" % (TOTAL, len(have), len(missing)), flush=True)
    if not missing:
        print("### FILL_DONE"); return

    # 把连续缺失压成区间
    runs, start, prev = [], missing[0], missing[0]
    for i in missing[1:]:
        if i == prev + 1:
            prev = i
        else:
            runs.append((start, prev - start + 1)); start = prev = i
    runs.append((start, prev - start + 1))
    print("[fill] 区间数=%d" % len(runs), flush=True)

    filled = 0
    for (off, ln) in runs:
        cur = off
        while cur < off + ln:
            n = min(BLOCK, off + ln - cur)
            got = fill(cur, n)
            filled += got
            print("   off=%-5d n=%-3d 补回=%d  累计=%d" % (cur, n, got, filled), flush=True)
            cur += n
    have = have_set()
    print("[done] 现有帧数=%d / %d  仍缺=%d" % (len(have), TOTAL, TOTAL - len(have)), flush=True)
    print("### FILL_DONE")


if __name__ == "__main__":
    main()
