"""AscendC 算子的 Meta（fake tensor）推导 —— torch.compile / TorchAir 入图必需。

用法（图模式前 import 一次即可）：

    import ascend_meta          # 注册幂等，且 hook 了 torch.ops.load_library

没有它，`torch.compile` 会报：

    torch._dynamo.exc.Unsupported: Operator does not support running with fake tensors

每条推导都照抄 `csrc/ascend/<op>/op_extension/<op>_torch.cpp` 里 `at::empty(...)` 的
实参与 options()，所以 Meta 与真实 kernel 的 shape/dtype 同源。

注意：`spconv_sortdedup_v3` 的 U 是 kernel 在 device 上算出来的（数据相关），
Meta 只能给上界 N；需要「任意形状」时请用容量化变体（unum_ops 主库的 *_v4 / _v3 / _v2）。
"""
import os

import torch

OPS = ()


def _nbk_for(n):
    """keyprep 在 nbk<=0 时自己算 nbk 的公式（spconv_keyprep_v1_torch.cpp:156）。"""
    mul = 4
    try:
        v = int(os.environ.get("TS_NBK_MUL") or 4)
        mul = v if v > 0 else 4          # C++ 侧 <=0 一律回落到 4
    except ValueError:
        pass
    nbk = 1
    while nbk < mul * n:
        nbk <<= 1
    return min(nbk, 1 << 20)


def _meta_query(qcoords, rkeys, start, cnt, meta):
    return qcoords.new_empty((qcoords.size(0), (meta.numel() - 12) // 3),
                             dtype=torch.int32)


def _meta_keyprep(in_coords, out_coords, stride, offs, nbk):
    n = in_coords.size(0) if in_coords.dim() > 0 else 0
    kv = offs.size(0) if offs.dim() > 0 else 0
    if nbk <= 0:
        nbk = _nbk_for(n)
    return [in_coords.new_empty((n,), dtype=torch.int64),
            in_coords.new_empty((nbk,), dtype=torch.int32),
            in_coords.new_empty((nbk,), dtype=torch.int32),
            in_coords.new_empty((12 + 3 * kv,), dtype=torch.int32)]


def _meta_gather_gemm(feat, nb, w):
    return feat.new_empty((nb.size(0), w.size(1)))


def _meta_devox(feat, idx, w, coords):
    # coords 只是占位（host tiling 用 coords.dim(1) 判是否核内算权重），不参与形状
    return feat.new_empty((idx.size(1), feat.size(1)))


def _meta_nb(qkeys, rkeys, start, cnt, meta):
    return qkeys.new_empty(qkeys.shape, dtype=torch.int32)


def _meta_sortdedup(coords):
    n = coords.size(0) if coords.dim() > 0 else 0
    if n <= 0:
        return [coords.new_empty((0, 4), dtype=torch.int32),
                coords.new_empty((1,), dtype=torch.int64)]
    # 真实输出是 [U,4]，U 由 kernel 在 device 上算出来（数据相关），Meta 只能给上界 n。
    # 这一点不致命：Python 侧 out[:int(u)] 里那个 .item() 本身就是 D2H 同步，
    # 有这个同步就不可能被 capture 进图，所以 [n,4] 走不到 replay。
    # 若将来去掉 kernel 里的 .item()，这里要换成 unbacked SymInt（否则 tail 行是脏的）。
    return [coords.new_empty((n, 4), dtype=torch.int32),
            coords.new_empty((1,), dtype=torch.int64)]


def _meta_gather(feat, nb):
    """spconv_gather(feat[M,cin], nb[M,KV]) -> [M, KV*cin]（见 *_torch.cpp:110）。"""
    return feat.new_empty((nb.size(0), nb.size(1) * feat.size(1)))


def _meta_gather3(feat, nb, n):
    """spconv_gather_v2(feat, nb, n)：n 只决定真实行数，形状按容量 nb.size(0)。"""
    return feat.new_empty((nb.size(0), nb.size(1) * feat.size(1)))


def _meta_keyprep_v2(in_coords, out_coords, stride, offs, nbk):
    """spconv_keyprep_v2 -> (rkeys[N], start[nbk], cnt[nbk], meta[12+3kv], u[1])。

    见 spconv_keyprep_v2_torch.cpp:187-190；u 是 32B 状态（屏障标记/U），形状固定 [1]。
    """
    n = in_coords.size(0) if in_coords.dim() > 0 else 0
    kv = offs.size(0) if offs.dim() > 0 else 0
    nb = _nbk_for(int(n)) if int(nbk) <= 0 else int(nbk)
    return [
        in_coords.new_empty((n,), dtype=torch.int64),
        in_coords.new_empty((nb,), dtype=torch.int32),
        in_coords.new_empty((nb,), dtype=torch.int32),
        in_coords.new_empty((12 + 3 * kv,), dtype=torch.int32),
        in_coords.new_empty((1,), dtype=torch.int64),
    ]


SPECS = (
    ("spconv_query", _meta_query),
    ("spconv_keyprep_v1", _meta_keyprep),
    ("spconv_keyprep_v2", _meta_keyprep_v2),
    ("spconv_gather_gemm", _meta_gather_gemm),
    ("spconv_gather", _meta_gather),
    ("spconv_gather_v2", _meta_gather3),
    ("spconv_devox", _meta_devox),
    ("spconv_devox_v3", _meta_devox),
    ("spconv_nb", _meta_nb),
    ("spconv_sortdedup_v3", _meta_sortdedup),
)


# ---------------------------------------------------------------- 容量版（device-n）算子
def _meta_sortdedup_v4(coords, n):
    """容量版去重：输出行数 = 输入容量（真实 U 在 u[0]）。"""
    return [coords.new_empty((coords.size(0), 4), dtype=torch.int32),
            coords.new_empty((1,), dtype=torch.int64)]


def _meta_keyprep_v3(in_coords, out_coords, stride, offs, nbk, sizes):
    n = in_coords.size(0)
    kv = offs.size(0) if offs.dim() > 0 else 0
    nb = _nbk_for(int(n)) if int(nbk) <= 0 else int(nbk)
    return [in_coords.new_empty((n,), dtype=torch.int64),
            in_coords.new_empty((nb,), dtype=torch.int32),
            in_coords.new_empty((nb,), dtype=torch.int32),
            in_coords.new_empty((12 + 3 * kv,), dtype=torch.int32)]


def _meta_query_v2(q, rkeys, start, cnt, meta, n):
    kv = (meta.numel() - 12) // 3
    return q.new_empty((q.size(0), kv), dtype=torch.int32)


def _meta_devox_v4(feat, idx, w, coords, n):
    return feat.new_empty((idx.size(1), feat.size(1)))


def _meta_gather_gemm_v2(feat, nb, w, n):
    return feat.new_empty((nb.size(0), w.size(1)))


_DYN_SPECS = (
    ("spconv_sortdedup_v4", _meta_sortdedup_v4),
    ("spconv_keyprep_v3", _meta_keyprep_v3),
    ("spconv_query_v2", _meta_query_v2),
    ("spconv_devox_v4", _meta_devox_v4),
    ("spconv_gather_gemm_v2", _meta_gather_gemm_v2),
)


def register():
    """注册所有已定义的 op 的 Meta；重复调用幂等。返回本次注册成功的名字。"""
    global OPS
    done = list(OPS)
    for name, fn in tuple(SPECS) + tuple(globals().get("_DYN_SPECS", ())):
        if name in done or not hasattr(torch.ops.npu, name):
            continue
        torch.library.impl("npu::" + name, "Meta", fn)
        done.append(name)
    OPS = tuple(done)
    return done


def _hook():
    """.so 是懒加载的，import 顺序不能保证；每次 load_library 后补一遍。"""
    orig = torch.ops.load_library
    if getattr(orig, "_npu_meta", False):
        return

    def load_library(path):
        r = orig(path)
        register()
        return r

    load_library._npu_meta = True
    torch.ops.load_library = load_library


_hook()
register()
