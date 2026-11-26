import os


# 可选加速：AscendC 算子（unum_ops）的 .so 根目录，默认不启用（纯 torch 路径）。
import os as _os
# AscendC 算子（unum_ops）是**可选加速**：默认不加载，纯 torch 路径即可正确运行。
# 想启用加速时显式设置：export UNUM_OPS_ROOT=/path/to/unum_ops
# 换 torch 版本时指向另一份构建（例如 unum_ops_t214）。
_OPS_ROOT = _os.environ.get("UNUM_OPS_ROOT", "").rstrip("/")
"""Hashing / lookup replacing torchsparse's CUDA backends.

Coordinates are packed into one int64 per voxel, so a lookup is a sort +
searchsorted instead of a cuckoo hash table. Pure torch ops, hence CPU/NPU
agnostic and autograd friendly.
"""
import torch

__all__ = ["GPUHashTable", "pack_keys", "lookup", "dense_lookup",
           "unique_coords", "unique_coords_fast", "DENSE_LIMIT"]

DENSE_LIMIT = int(os.environ.get("TS_DENSE_M", "256")) << 20
_NBK_MUL = int(os.environ.get("TS_NBK_MUL", "4"))
_TABLES = {}
_N_TABLES = 4



# 容量化去重（TS_CAP 路径用）：输出固定 CAP 行，真实去重数在 device 的 u 里
_SD4_SO = _OPS_ROOT + "/csrc/ascend/spconv_sortdedup_v4/op_extension/build/libspconv_sortdedup_v4_ops.so"
_SD4_OK = None


def _sd4_op():
    global _SD4_OK
    if _SD4_OK is None:
        try:
            torch.ops.load_library(_SD4_SO)
            _SD4_OK = hasattr(torch.ops.npu, "spconv_sortdedup_v4")
        except Exception:
            _SD4_OK = False
    return _SD4_OK


def unique_coords_cap(coords_cap, n):
    """容量化去重：coords_cap 已有 CAP 行（尾部是哨兵键），n 是真实行数(device int32[1])。

    返回 (out_cap, u)：形如 (CAP,4) 与 int64[1]；**u 不要读回 host**，下游用
    `arange(CAP) < u` 做掩码。注意 kernel 输出的尾部行是未定义的，调用方必须
    在 dedup 之后重新把尾部填成哨兵（见 functional.spdownsample）。
    """
    if not _sd4_op():
        raise RuntimeError("spconv_sortdedup_v4 not available (检查 UNUM_OPS_ROOT)")
    if not torch.is_tensor(n):
        n = torch.tensor([int(n)], dtype=torch.int32, device=coords_cap.device)
    else:
        n = n.to(dtype=torch.int32).reshape(1)
    out, u = torch.ops.npu.spconv_sortdedup_v4(
        coords_cap.to(torch.int32).contiguous(), n)
    return out, u


_DEDUP_SO = _OPS_ROOT + "/csrc/ascend/spconv_dedup_v2/op_extension/build/libspconv_dedup_v2_ops.so"
_DEDUP_OK = None


def _dedup_op():
    global _DEDUP_OK
    if _DEDUP_OK is None:
        try:
            torch.ops.load_library(_DEDUP_SO)
            _DEDUP_OK = hasattr(torch.ops.npu, "spconv_dedup_v2")
        except Exception:
            _DEDUP_OK = False
    return _DEDUP_OK


def _row_sort_key(coords):
    """Lexicographic row order without an int64 sort.

    int64/int32 argsort lands on AiCPU (16.7 ms per 123k rows); float32 argsort
    stays on the AI core (0.27 ms). Two stable passes over two <2^24 keys give
    the same full ordering: hi=(b*Dx+x), lo=(y*Dz+z).
    """
    b = coords[:, 0].long()
    x = coords[:, 1].long()
    y = coords[:, 2].long()
    z = coords[:, 3].long()
    dx = _fg.const(lambda: int(x.max()) + 1, ("dx", tuple(coords.shape)))
    dz = _fg.const(lambda: int(z.max()) + 1, ("dz", tuple(coords.shape)))
    hi = b * dx + x
    lo = y * dz + z
    if _fg.const(lambda: int(hi.max()) >= (1 << 24)
                 or int(lo.max()) >= (1 << 24),
                 ("key24", tuple(coords.shape))):
        return None
    order = torch.argsort(lo.float(), stable=True)
    return order[torch.argsort(hi[order].float(), stable=True)]


_SD_SO = _OPS_ROOT + "/csrc/ascend/spconv_sortdedup_v3/op_extension/build/libspconv_sortdedup_v3_ops.so"
# v2: the v1 global barrier needed all 40 AIV resident and spun 1e8 times
# (~70 s) when one was busy. v2 bounds the spin, reports failure through
# u[1] and the extension raises -> the caller falls back to torch.
# v3 adds SetScheduleMode(1) (batch scheduling) so all 40 AIV start
# together and the barrier is not starved in the first place.
from .. import _fgstatic as _fg          # 整模入图：数据相关值静态化（见 _fgstatic.py）


_CONST_TENSORS = {}


def _const_tensor(vals, dtype, device):
    """常量张量缓存（见 nn/functional 同名函数）：避免每次前向 H2D，
    aclgraph capture 阶段同步 H2D 会直接报 aclrtMemcpy 107030。"""
    key = (tuple(vals), str(dtype), str(device))
    t = _CONST_TENSORS.get(key)
    if t is None:
        t = torch.tensor(key[0], dtype=dtype, device=device)
        _CONST_TENSORS[key] = t
    return t


def _as_ints(x):
    """kernel_size / stride 允许 python int|tuple|device 张量。

    张量分支会 .tolist() 同步一次；新调用方一律传 int/tuple，
    因为入图时那次同步既是 graph break 也是白送的 host 空档。"""
    if torch.is_tensor(x):
        return tuple(int(v) for v in x.reshape(-1).tolist())
    if isinstance(x, (tuple, list)):
        return tuple(int(v) for v in x)
    return (int(x), int(x), int(x))


_SD = os.environ.get("TS_SD", "1") != "0"
_SD_OK = None


def _sd_op():
    global _SD_OK
    if _SD_OK is None:
        try:
            torch.ops.load_library(_SD_SO)
            _SD_OK = hasattr(torch.ops.npu, "spconv_sortdedup_v3")
        except Exception:
            _SD_OK = False
    return _SD_OK


_SIGNED_OK = None


def _nonneg_ok(coords):
    """Kernel has no signed support. Verify once per process, not per call.

    shortcut: the check runs once per process, so a dataset whose coords go
    negative later would silently use the kernel; the torch path only comes
    back on a fresh process. Upgrade if a signed-input dataset shows up.
    the per-call amin() costs ~0.1 ms and a device sync in the hot path."""
    global _SIGNED_OK
    if _SIGNED_OK is None:
        _SIGNED_OK = bool(int(coords.amin()) >= 0)
        if not _SIGNED_OK:
            print("[torchsparse] negative coords: SpconvSortDedupV1 disabled",
                  file=os.sys.stderr)
    return _SIGNED_OK


def _sortdedup_kernel(coords):
    """Sorted-unique rows in one launch (4-pass LSD radix + run-length).

    Bit-identical to torch.unique(coords, dim=0) row-for-row and in order;
    1.06x-5.94x per level, and 38 -> 2 kernels per call (launch-bound elsewhere).
    """
    out, u = torch.ops.npu.spconv_sortdedup_v3(coords.to(torch.int32).contiguous())
    if _fg.MODE == "replay":
        # 图模式：U 在 record 阶段已固化；这里绝不能碰 u —— tracing 时 u 会被
        # dynamo 提成 CPU 常量再 H2D，而 aclgraph 捕获期禁止同步 H2D（107030）。
        return out[:_fg.get(("u", tuple(coords.shape)))]
    return out[:_fg.const(lambda: int(u), ("u", tuple(coords.shape)))]


def unique_coords_fast(coords):
    """Sorted unique rows: 2-pass float32 stable sort + run-length dedup.

    Bit-identical to torch.unique(coords, dim=0) and ~5x faster on 123k rows.
    """
    n = coords.shape[0]
    if n < 2:
        return coords.clone()
    if _SD and _sd_op() and coords.device.type == "npu" and _nonneg_ok(coords):
        try:
            return _sortdedup_kernel(coords)
        except Exception as e:
            if os.environ.get("TS_FG_DEBUG"):
                print("[uniq-fast] sortdedup 回退:", type(e).__name__, e, flush=True)
    order = _row_sort_key(coords)
    if order is None:
        return torch.unique(coords, dim=0)
    srt = coords[order].contiguous()
    if _dedup_op():
        res = torch.ops.npu.spconv_dedup_v2(srt.to(torch.int32))
        # 内容自检：静默 no-op 会把未初始化的显存当结果返回
        if res.numel() and bool((res[0].cpu() == srt[0].cpu()).all()):
            return res
    keep = torch.ones(n, dtype=torch.bool, device=coords.device)
    keep[1:] = (srt[1:] != srt[:-1]).any(1)
    return srt[keep]


def unique_coords(coords):
    """Sorted unique coord rows without the sort torch.unique needs.

    int64 sort/searchsorted fall back to AiCPU on this device; a dense presence
    table + nonzero stays on AiCore. Only worth it when the box is not much
    larger than the row count."""
    n = coords.shape[0]
    if coords.device.type != "npu" or n < 2:
        return torch.unique(coords, dim=0)
    # float32 两级稳定排序 + run-length 去重；TS_UNIQ=0 回退 torch.unique
    if coords.shape[1] == 4 and os.environ.get("TS_UNIQ", "1") != "0":
        try:
            return unique_coords_fast(coords)
        except Exception as e:
            if os.environ.get("TS_FG_DEBUG"):
                print("[uniq] fast 回退 dense:", type(e).__name__, e, flush=True)
    mins, dims = _bounds([coords])
    total = _fg.const(lambda: int(torch.prod(dims.long())),
                       ("total_uniq", tuple(coords.shape)))
    if total > DENSE_LIMIT or total < 4 * n:
        return torch.unique(coords, dim=0)

    tab = torch.zeros(total, dtype=torch.bool, device=coords.device)
    tab[pack_keys(coords, mins, dims)] = True
    keys = tab.nonzero().squeeze(1)
    z = keys % dims[3]
    r = keys // dims[3]
    y = r % dims[2]
    r = r // dims[2]
    x = r % dims[1]
    b = r // dims[1]
    return (torch.stack([b, x, y, z], 1) + mins).to(coords.dtype)


def dense_lookup(total, in_coords, mins, dims, qkeys):
    """table[key] = row index, one gather instead of an argsort. The table is
    reused per size; only the slots the previous user wrote are cleared."""
    entry = _TABLES.get(total)
    if entry is None or entry[0].device != in_coords.device:
        if len(_TABLES) >= _N_TABLES:
            _TABLES.pop(next(iter(_TABLES)))
        entry = [torch.full((total,), -1, dtype=torch.int32, device=in_coords.device), None]
        _TABLES[total] = entry
    table, prev = entry
    if prev is not None:
        table[prev] = -1
    keys = pack_keys(in_coords, mins, dims)
    # 重复键时希望存**最小**（真实）行号（padding 行是尾部重复，不能抢走真实键），
    # 做法是「按下标降序写入、值写自己的行号」：最后一次写的是最小下标。
    # 注意必须同时翻转索引侧，否则唯一键会得到 n-1-row —— 整表反序（实测 100% 错位）。
    _rows = torch.arange(in_coords.shape[0] - 1, -1, -1, dtype=torch.int32,
                         device=in_coords.device)
    table[keys.flip(0)] = _rows
    _TABLES[total] = [table, keys]
    return table[qkeys].long()


def _bounds(arrays):
    dev = arrays[0].device
    dt = arrays[0].dtype if arrays[0].dtype == torch.int32 else torch.long
    mins = torch.full((4,), 2**31 - 1, dtype=dt, device=dev)
    maxs = torch.full((4,), -(2**31), dtype=dt, device=dev)
    for a in arrays:
        a = a.to(dt)
        if a.numel() == 0:
            continue
        mins = torch.minimum(mins, a.amin(0))
        maxs = torch.maximum(maxs, a.amax(0))
    dims = maxs.long() - mins.long() + 1
    # shortcut: keys must stay below 2**62; a grid that big would need a real
    # hash table instead of dense-integer packing.
    if _fg.const(lambda: bool(int(torch.prod(dims.long())) > 2**62),
                 ("bounds_ok", tuple(arrays[0].shape))):
        raise RuntimeError(f"coordinate range too large to pack: {dims.tolist()}")
    return mins, dims


def pack_keys(coords, mins, dims):
    """Lexicographic (b, x, y, z) -> one integer. int32 when the whole box
    fits, which keeps the later sort/searchsorted off AiCPU."""
    d = _fg.const(lambda: [int(v) for v in dims.tolist()],
                   ("dims", tuple(coords.shape)))
    total = d[0]
    for v in d[1:4]:
        total *= v
    int32 = total < 2 ** 31 - 1
    c = coords - mins
    b, x, y, z = c.unbind(-1)
    if int32:
        return ((b * d[1] + x) * d[2] + y) * d[3] + z
    return (((b.long() * d[1] + x.long()) * d[2] + y.long()) * d[3] + z.long())


_NB_SO = _OPS_ROOT + "/csrc/ascend/spconv_nb/op_extension/build/libspconv_nb_ops.so"
_NB_OK = None
_NB_META = {}


def _nb_op():
    global _NB_OK
    if _NB_OK is None:
        try:
            torch.ops.load_library(_NB_SO)
            _NB_OK = hasattr(torch.ops.npu, "spconv_nb")
        except Exception:
            _NB_OK = False
    return _NB_OK


def _nb_meta(shift, device):
    key = (device, shift)
    t = _NB_META.get(key)
    if t is None:
        if len(_NB_META) > 8:
            _NB_META.clear()
        t = torch.tensor([shift], dtype=torch.int32, device=device)
        _NB_META[key] = t
    return t


def bucket_lookup(qkeys, rkeys, total=None):
    """Index of rkeys matching each qkeys (-1 when absent). None -> caller
    falls back. Requires rkeys sorted ascending (guaranteed: coordinate keys
    come out of a dense-presence nonzero, which is ordered)."""
    # On by default: the SpconvNb kernel replaces torch's int64
    # argsort + searchsorted, which CANN runs on AiCPU (16.7 ms/frame at 20k
    # voxels). Verified bit-identical against the torch path at four sizes
    # (probe 8/8, 9/9, 12/12) and at model level (CPU<->NPU 7.451e-08).
    # TS_NB=0 falls back. Graph mode gains nothing (see the analysis doc).
    # compare against "0": `if not getenv(...)` treats the string "0" as truthy
    if os.environ.get("TS_NB", "1") == "0":
        return None
    if not _nb_op() or qkeys.numel() == 0 or rkeys.numel() == 0:
        return None
    n = rkeys.numel()
    nbk = 1
    while nbk < _NBK_MUL * n:
        nbk <<= 1
    nbk = min(nbk, 1 << 20)
    if total is not None:
        shift = max(0, (int(total) - 1).bit_length() - (nbk.bit_length() - 1))
    else:
        shift = max(0, int(rkeys[-1]) .bit_length() - (nbk.bit_length() - 1))
    dev = rkeys.device
    ones = torch.ones(n, dtype=torch.int32, device=dev)
    hist = torch.zeros(nbk, dtype=torch.int32, device=dev)
    hist.index_add_(0, _shr(rkeys, shift).to(torch.int64)
                    if rkeys.dtype != torch.int64 else _shr(rkeys, shift), ones)
    start = (torch.cumsum(hist, 0) - hist).to(torch.int32)
    out = torch.ops.npu.spconv_nb(qkeys.to(torch.int64), rkeys.to(torch.int64),
                                  start, hist, _nb_meta(shift, dev))
    return out.long()


# ---------------------------------------------------------------- GE 兼容层
# TorchAir 的 ATen 支持清单里没有 aten::bitwise_right_shift（也有没有 amin/amax）。
# 打开 TS_GE_COMPAT=1 时把这些算子换成清单内等价写法；默认关闭，生产路径不变。
_GE_COMPAT = __import__("os").environ.get("TS_GE_COMPAT", "0") == "1"


def _shr(x, k):
    """右移：GE 兼容模式下用 div(2**k, floor) 代替 >>（打包键非负，数值等价）。"""
    if k <= 0:
        return x
    if _GE_COMPAT:
        return torch.div(x, 1 << k, rounding_mode="floor")
    return x >> k


_QUERY_SO = _OPS_ROOT + "/csrc/ascend/spconv_query/op_extension/build/libspconv_query_ops.so"
_QUERY_OK = None


def _query_op():
    global _QUERY_OK
    if _QUERY_OK is None:
        try:
            torch.ops.load_library(_QUERY_SO)
            _QUERY_OK = hasattr(torch.ops.npu, "spconv_query")
        except Exception:
            _QUERY_OK = False
    return _QUERY_OK


_KPREP_SO = _OPS_ROOT + "/csrc/ascend/spconv_keyprep_v2/op_extension/build/libspconv_keyprep_v2_ops.so"
_KPREP = os.environ.get("TS_KPREP", "1") != "0"
_KPREP_OK = None


def _kprep_op():
    global _KPREP_OK
    if _KPREP_OK is None:
        try:
            torch.ops.load_library(_KPREP_SO)
            _KPREP_OK = hasattr(torch.ops.npu, "spconv_keyprep_v2")
        except Exception:
            _KPREP_OK = False
    return _KPREP_OK


def _query_args(in_coords, out_coords, stride, offsets):
    """(rkeys, start, cnt, meta, total) for SpconvQuery. Bounds mirror
    _lookup_offsets so the two paths agree bit for bit."""
    st = _const_tensor(tuple(int(v) for v in stride), in_coords.dtype,
                       in_coords.device)
    # One kernel does box + rkeys + histogram (and a second the prefix sum),
    # replacing ~40 torch ops and the AiCPU cumsum: bit-identical, 2.3-2.4x.
    # nbk == 1<<20 is excluded -- that path is untested upstream (see report).
    if (_KPREP and _kprep_op() and in_coords.device.type == "npu"
            and in_coords.shape[0] > 0 and offsets.shape[0] <= 32):
        n_ = int(in_coords.shape[0])
        nbk_ = 1
        while nbk_ < _NBK_MUL * n_:
            nbk_ <<= 1
        nbk_ = min(nbk_, 1 << 20)
        if nbk_ < (1 << 20):
            try:
                rk, srt, cn, mt, u = torch.ops.npu.spconv_keyprep_v2(
                    in_coords.to(torch.int32).contiguous(),
                    out_coords.to(torch.int32).contiguous(),
                    st.to(torch.int32).contiguous(),
                    offsets.to(torch.int32).contiguous(), nbk_)

                def _finish():
                    h = torch.cat([mt[8:12], u[:1]]).to("cpu")
                    if int(h[4]) < 0:
                        # cross-core barrier timed out: discard, caller retries in torch
                        raise RuntimeError("SpconvKeyprepV2: barrier timeout")
                    return int(h[0]) * int(h[1]) * int(h[2]) * int(h[3])

                total_ = _fg.const(_finish, ("total_kp", tuple(in_coords.shape)))
                return rk, srt, cn, mt.to(torch.int32), total_
            except Exception:
                pass
    off_lo, off_hi = offsets.amin(0).long(), offsets.amax(0).long()
    in_lo, in_hi = in_coords.amin(0), in_coords.amax(0)
    out_lo, out_hi = out_coords.amin(0), out_coords.amax(0)
    lo = torch.minimum(in_lo, torch.cat([out_lo[:1], out_lo[1:] * st + off_lo]))
    hi = torch.maximum(in_hi, torch.cat([out_hi[:1], out_hi[1:] * st + off_hi]))
    dims = hi.long() - lo.long() + 1
    total = _fg.const(lambda: int(torch.prod(dims.long())),
                       ("total_b", tuple(out_coords.shape)))

    n = in_coords.shape[0]
    nbk = 1
    while nbk < _NBK_MUL * n:
        nbk <<= 1
    nbk = min(nbk, 1 << 20)
    shift = max(0, (total - 1).bit_length() - (nbk.bit_length() - 1))
    rkeys = pack_keys(in_coords, lo, dims)
    hist = torch.zeros(nbk, dtype=torch.float32, device=in_coords.device)
    hist.index_add_(0, rkeys >> shift,
                    torch.ones(n, dtype=torch.float32, device=in_coords.device))
    # float32 cumsum: same values (counts << 2^24) but 1.1-1.6x faster than int32
    start = (torch.cumsum(hist, 0) - hist).to(torch.int32)
    meta = torch.cat([
        torch.tensor([shift], dtype=torch.int32, device=in_coords.device),
        torch.tensor([int(v) for v in stride], dtype=torch.int32,
                     device=in_coords.device),
        lo.to(torch.int32), dims.to(torch.int32),
        offsets.to(torch.int32).reshape(-1),
    ]).contiguous()
    return rkeys, start, hist.to(torch.int32), meta, total


def lookup(qkeys, rkeys, total=None):
    """Index of each query key in rkeys, -1 when absent."""
    if rkeys.numel() == 0:
        return torch.full_like(qkeys, -1)
    if qkeys.device.type == "npu":
        fast = bucket_lookup(qkeys, rkeys, total)
        if fast is not None:
            return fast
    order = torch.argsort(rkeys)
    sk = rkeys[order]
    pos = torch.searchsorted(sk, qkeys)
    inside = pos < sk.numel()
    pos_c = pos.clamp(max=sk.numel() - 1)
    hit = inside & (sk[pos_c] == qkeys)
    return torch.where(hit, order[pos_c], torch.full_like(pos_c, -1))


class GPUHashTable:
    """Same call pattern as torchsparse.backend.GPUHashTable. Coordinates are
    (x, y, z, batch); the returned tensor is [kernel_volume, N] of
    (index + 1), 0 when absent -- the layout core/models/utils.py expects."""

    def __init__(self, keys, values):
        self.keys = keys
        self.values = values
        self._coords = None

    def insert_coords(self, coords):
        if coords.dim() != 2 or coords.shape[1] != 4:
            raise ValueError(f"coords must be [N, 4] (batch, x, y, z), got {tuple(coords.shape)}")
        _c = coords.long()
        # slice+cat 而不是 t[:, [3,0,1,2]]：列表索引会被编译成索引张量（CPU 常量 + H2D）
        self._coords = torch.cat([_c[:, 3:4], _c[:, 0:3]], dim=1).contiguous()

    def lookup_coords(self, query, kernel_size, stride, kernel_volume):
        from ..nn.utils import get_kernel_offsets

        if self._coords is None:
            raise RuntimeError("insert_coords() must be called first")
        ks = _as_ints(kernel_size)
        st = _as_ints(stride)

        if query.dim() != 2 or query.shape[1] != 4:
            raise ValueError(f"query must be [N, 4] (x, y, z, batch), got {tuple(query.shape)}")
        _q = query.long()
        q = torch.cat([_q[:, 3:4], _q[:, 0:3]], dim=1).contiguous()   # [N, 4] (b, x, y, z)
        offs = get_kernel_offsets(ks, 1, device=q.device)    # [KV, 3]
        kv = offs.shape[0]

        # One kernel does the coord transform, bounds check and bucket search
        # (torchsparse's hash_query_cuda does the same on CUDA); no [N,KV,4]
        # intermediate and no host pack_keys over N*KV query keys.
        if _query_op() and q.device.type == "npu" and q.shape[0] > 0 and kv <= 32:
            try:
                rkeys, start, cnt, meta, _ = _query_args(
                    self._coords, q, st, offs)
                idx = torch.ops.npu.spconv_query(
                    q.to(torch.int32), rkeys, start, cnt, meta)
                return (idx + 1).transpose(0, 1).contiguous()
            except Exception:
                pass
        base = q[:, None, 1:4] * torch.tensor(st, dtype=torch.long, device=q.device)
        pos = base + offs.long()[None]                       # [N, KV, 3]
        batch = q[:, None, 0:1].expand(-1, kv, 1)
        full = torch.cat([batch, pos], dim=2).reshape(-1, 4)

        mins, dims = _bounds([self._coords, full])
        total = _fg.const(lambda: int(torch.prod(dims.long())),
                           ("total_lc", tuple(q.shape)))
        qk = pack_keys(full, mins, dims)
        if q.device.type == "npu" and total <= DENSE_LIMIT:
            # dense table: no argsort (CANN runs int64 sort on AiCpu)
            idx = dense_lookup(total, self._coords, mins, dims, qk)
        else:
            idx = lookup(qk, pack_keys(self._coords, mins, dims))
        return (idx.view(q.shape[0], kv) + 1).transpose(0, 1).contiguous()
