"""整模入图（TS_FG）的「数据相关值静态化」层 —— 每 key 一条队列。

aclgraph 要求整张图的形状在 capture 时全部确定，而这条管线里有一堆值只有 device
上算完才知道：unique 后的行数 U、每层每 kernel offset 的 counts、打包用的 dims/total。
eager 下它们靠 `.item()`/`.tolist()` 回 host（那 45 ms host 空档的一部分），
在 capture 下更是 graph break。

用法：`_fg.const(lambda: <真值>, ("u", tuple(coords.shape)))`

key = (调用点, 形状)。同一 key 可能被多个层级撞上（比如两层 conv 的 in_coords
形状相同、total 却不同），所以**每个 key 记一条队列、按调用顺序取值**：record 与
replay 的数据流一致，队列位置就一致。

代价：**一张图只对同一份数据成立**，形状或数据一变就要重捕获（代价见 fg2.py）。
"""

import os

MODE = os.environ.get("TS_FG", "")      # ""（关） | record | replay
_q = {}      # key -> [value, ...]
_pos = {}    # key -> 下一个该取的下标


def const(fn, key=None):
    if MODE != "replay":
        v = fn()
        if MODE == "record" and key is not None:
            _q.setdefault(key, []).append(v)
        return v
    if key is None or key not in _q:
        raise RuntimeError("TS_FG replay 缺 key %r" % (key,))
    q = _q[key]
    i = _pos.get(key, 0)
    _pos[key] = i + 1
    return q[min(i, len(q) - 1)]


def get(key):
    """只在 replay 用：按 key 取记录值，完全不接触 device 张量。"""
    if MODE != "replay":
        raise RuntimeError("_fg.get() 只能在 replay 模式下用")
    if key not in _q:
        raise RuntimeError("TS_FG replay 缺 key %r" % (key,))
    q = _q[key]
    i = _pos.get(key, 0)
    _pos[key] = i + 1
    return q[min(i, len(q) - 1)]


def freeze():
    return {"keys": len(_q), "calls": sum(len(v) for v in _q.values())}


def reset():
    """只归零 replay 游标（record 下来的值保留）。"""
    _pos.clear()


def clear():
    _q.clear()
    _pos.clear()


def stats():
    return {"mode": MODE, "keys": len(_q), "calls": sum(len(v) for v in _q.values())}
