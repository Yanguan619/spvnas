"""Sparse conv3d / devoxelize (pure torch, inference-optimised).

Reproduces TorchSparse++ (torchsparse 2.1) semantics, which is what SPVNAS
targets:
  * coords are (batch, x, y, z) in *level* units (indices at the current
    stride), not level-0 units,
  * a strided conv downsamples coords by `floor(coord / stride)` and reads
    neighbours at `out_coord * stride + kernel_offset` (centred offsets),
  * transposed conv reuses the kernel map of the matching forward conv
    (torchsparse keys them on the parent tensor stride as well).

This device is launch-bound, not FLOP-bound: one 3x3 layer is ~80 tiny ops and
the whole frame runs at <10% of the NPU's compute. So the loops below are
written for op count: one selection builds all neighbour pairs, one gather +
one GEMM covers a group of kernel offsets, and accumulation is in-place
(out-of-place `index_add` copies the full output once per offset).
"""
import os

import os as _os
# AscendC 算子（unum_ops）是**可选加速**：默认不加载，纯 torch 路径即可正确运行。
# 想启用加速时显式设置：export UNUM_OPS_ROOT=/path/to/unum_ops
# 换 torch 版本时指向另一份构建（例如 unum_ops_t214）。
_OPS_ROOT = _os.environ.get("UNUM_OPS_ROOT", "").rstrip("/")
import time

import torch

from ...backend import lookup, pack_keys, unique_coords
from ...tensor import SparseTensor
from ...utils import make_ntuple
from ..utils import get_kernel_offsets

__all__ = [
    "set_conv_mode", "get_conv_mode", "conv3d", "spdevoxelize", "calc_ti_weights",
    "spdownsample", "get_kernel_offsets", "make_ntuple", "prof_report",
]

_CONV_MODE = 0
_APPLY_GROUP = 9                       # kernel offsets per matmul (27 -> 3 GEMMs)
_DENSE_LIMIT = 256 * 1024 * 1024   # entries; level-0 KITTI box is ~179 M
_MEM_BUDGET = int(os.environ.get("TS_MEM", "512")) << 20
_TABLES = {}                           # size -> [table, keys written last time]
_N_TABLES = 4

_PROF = {} if os.environ.get("TS_PROF") else None
_GATHER_SO = _OPS_ROOT + "/csrc/ascend/spconv_gather/op_extension/build/libspconv_gather_ops.so"
from ... import _fgstatic as _fg         # 整模入图：数据相关值静态化

_GRAPH = bool(os.environ.get("TS_GRAPH"))
_DYNAMIC = os.environ.get("TS_DYNAMIC", "1") != "0"
# The 910B gather op is the fastest path (177-188 ms vs 290-320 ms at 20k
# voxels), so it is on by default; TS_GATHER=0 forces the torch path.
_GATHER = os.environ.get("TS_GATHER", "1") != "0"
# Off by default: the fused join+gather kernel is numerically exact and 15x
# faster per layer than "rebuild map + SpconvGather", but end-to-end it is
# SLOWER (163 vs 147 ms at 20k voxels). The map is built once per level
# (10x/frame) as a block over all 27 taps, which vectorises; the fused kernel
# repeats the search inline per tap -- 27 scalar reads per row. TS_FUSED=1 on.
_FUSED = os.environ.get("TS_FUSED") == "1"
# 转置卷积：取逆映射 + 一次 gather_gemm（替代逐 offset index_select+index_add）
_TS_DECONV_INV = os.environ.get("TS_DECONV_INV", "0") != "0"
_FUSED_SO = _OPS_ROOT + "/csrc/ascend/spconv_fused/op_extension/build/libspconv_fused_ops.so"
_FUSED_OK = None


def _fused_op():
    global _FUSED_OK
    if _FUSED_OK is None:
        try:
            torch.ops.load_library(_FUSED_SO)
            _FUSED_OK = hasattr(torch.ops.npu, "spconv_fused")
        except Exception:
            _FUSED_OK = False
    return _FUSED_OK
if _GRAPH and _DYNAMIC:
    # ACL graph fixes tensor addresses, but this op's output shape (M, K*cin)
    # differs per layer, so one dynamic graph cannot cover it (verified:
    # aclnnInplaceCopy shape mismatch). Fall back to the torch path instead of
    # crashing; TS_DYNAMIC=0 gives per-shape graphs, where the op does work.
    _GATHER = False
_GATHER_OK = None


_SGG_SO = _OPS_ROOT + "/csrc/ascend/spconv_gather_gemm/op_extension/build/libspconv_gather_gemm_ops.so"
_SGG = os.environ.get("TS_SGG", "1") != "0"
_SGG_OK = None


def _sgg_op():
    global _SGG_OK
    if _SGG_OK is None:
        try:
            torch.ops.load_library(_SGG_SO)
            _SGG_OK = hasattr(torch.ops.npu, "spconv_gather_gemm")
        except Exception:
            _SGG_OK = False
    return _SGG_OK


def _sgg_fits(m, cin, cout, kv):
    """是否用融合的 gather+GEMM（spconv_gather_gemm）。

    两套策略（实测，见 docs/spvnas/GRAPH_MODE_MATRIX.md §35）：
    * 静态图（aclgraph / GE 静态）：只在小盒子里用融合算子——盒子外 torch matmul 更快
      （20 层实测 37.7 -> 22.7 ms）；把盒子放开反而让 aclgraph 94k cr0.5 慢 14%。
    * GE 动态形状：必须全部走融合算子——gather+MatMul 两段式会在每个边界触发
      ~5.7 ms 的 host 同步（每迭代 8 处 = 45 ms），20k cr1.0 136.8 -> 76.4 ms。
      run_model_debug.sh 在 GM_MODE=ge 时注入 TS_SGG_MAX_K=100000 / TS_SGG_MAX_COUT=512。
    """
    _kmax = int(os.environ.get("TS_SGG_MAX_K", "5184"))
    _cmax = int(os.environ.get("TS_SGG_MAX_COUT", "128"))
    return (_SGG and m >= 512 and cout <= _cmax and cin * kv <= _kmax)


def _gather_op():
    """(M,K) int32 nb -> dense [M, K*cin] float32, or None if unavailable."""
    global _GATHER_OK
    if _GATHER_OK is None:
        try:
            torch.ops.load_library(_GATHER_SO)
            _GATHER_OK = hasattr(torch.ops.npu, "spconv_gather")
        except Exception:
            _GATHER_OK = False
    return _GATHER_OK


_DENSE = bool(os.environ.get("TS_DENSE"))
_UNIQ_MODE = os.environ.get("TS_UNIQ", "dense")
_GRAPH_FNS = {}


def _compiled(fn):
    """TorchAir graph replay for a function with static *op* structure (opt-in,
    TS_GRAPH=1).

    dynamic=True keeps one graph per function instead of one per shape (the
    model has ~30 conv shapes); only functions without host round-trips may go
    through here, since graph capture forbids H2D/D2H copies."""
    import time

    import torch_npu
    from torchair import CompilerConfig, get_npu_backend

    torch._dynamo.config.cache_size_limit = 64
    torch_npu.npu.set_compile_mode(jit_compile=False)   # no second-stage JIT
    cfg = CompilerConfig()
    cfg.mode = os.environ.get("TS_MODE", "reduce-overhead")
    dyn = _DYNAMIC   # single source of truth (env read once, at import)
    t0 = time.time()
    cfn = torch.compile(fn, backend=get_npu_backend(compiler_config=cfg),
                        dynamic=dyn, fullgraph=os.environ.get("TS_FULLGRAPH", "0") == "1")
    cfn._ts_compile_seconds = time.time() - t0
    print(f"[graph] compiled <{fn.__name__}> dynamic={dyn} "
          f"in {cfn._ts_compile_seconds:.1f}s", flush=True)
    return cfn


def _graph_call(tag, fn, *args):
    if not _GRAPH:
        return fn(*args)
    ent = _GRAPH_FNS.get(tag)
    if ent is None:
        ent = _compiled(fn)
        _GRAPH_FNS[tag] = ent
    return ent(*args)


def _now():
    """prof 计时；编译/入图时返回 0，避免 time.time() 变成 graph break。"""
    if _PROF is None or torch.compiler.is_compiling():
        return 0.0
    return time.time()


def _tick(name, t0):
    if _PROF is not None:
        _PROF[name] = _PROF.get(name, 0.0) + (time.time() - t0)


def prof_report():
    return {} if _PROF is None else dict(_PROF)


def set_conv_mode(mode):
    global _CONV_MODE
    _CONV_MODE = mode


def get_conv_mode():
    return _CONV_MODE


def spdownsample(coords, stride=2, kernel_size=2, dilation=1):
    st = make_ntuple(stride, ndim=3)
    if all(s == 1 for s in st):
        return coords
    out = coords.clone()
    for k in range(3):
        if st[k] > 1:
            out[:, 1 + k] = torch.div(out[:, 1 + k], st[k], rounding_mode="floor")
    if _UNIQ_MODE == "torch":
        from torch import unique as _u
        return _u(out, dim=0).contiguous()
    return unique_coords(out).contiguous()


def _table_fill(total, in_coords, mins, dims, device):
    """table[key] = row index. Cached per size so consecutive layers do not
    re-initialise tens of MB; only the entries the previous user wrote are
    cleared."""
    entry = _TABLES.get(total)
    if entry is None or entry[0].device != device:
        if len(_TABLES) >= _N_TABLES:
            _TABLES.pop(next(iter(_TABLES)))
        entry = [torch.full((total,), -1, dtype=torch.int32, device=device), None]
        _TABLES[total] = entry
    table, prev = entry
    if prev is not None:
        table[prev] = -1
    keys = pack_keys(in_coords, mins, dims)
    table[keys] = torch.arange(in_coords.shape[0], dtype=torch.int32, device=device)
    _TABLES[total] = [table, keys]
    return table


_CONST_TENSORS = {}


def _const_tensor(vals, dtype, device):
    """常量张量缓存：避免每次前向 H2D（capture 阶段同步 H2D 直接报错）。"""
    key = (tuple(vals), str(dtype), str(device))
    t = _CONST_TENSORS.get(key)
    if t is None:
        t = torch.tensor(key[0], dtype=dtype, device=device)
        _CONST_TENSORS[key] = t
    return t


def _ge_compat():
    """TS_GE_COMPAT=1 时启用 GE 兼容写法（amin/amax、>> 等不在 TorchAir ATen 清单里）。"""
    import os
    return os.environ.get("TS_GE_COMPAT", "0") == "1"


_LOHI_CACHE = {}


def _lo_hi(t, cache=False):
    """amin/amax 的 GE 友好替代（int32 也适用）。

    TorchAir 的 ATen 支持清单没有 torch.amin/amax；torch.max 又不支持 int32。
    先 _to_copy 到 int64（全 dtype 支持），上界 torch.max，下界用 0-(max(0-t))
    绕开不在清单里的 aten::neg，最后转回原 dtype。
    常量张量（如 offsets）按 data_ptr 缓存，避免每层重复算。
    """
    if not _ge_compat():
        # 生产路径：amin/amax 最快（aclgraph 支持）；GE 才走下面的替代实现
        return t.amin(0), t.amax(0)
    if t.numel() == 0:
        z = t.new_zeros(t.shape[1:])
        return z, z
    key = None
    if cache:
        key = (t.data_ptr(), tuple(t.shape), str(t.dtype))
        hit = _LOHI_CACHE.get(key)
        if hit is not None:
            return hit
    t64 = t.to(torch.int64)
    zero = torch.zeros_like(t64)
    hi = torch.max(t64, dim=0).values
    lo = zero[0] - torch.max(zero - t64, dim=0).values
    if t.dtype != torch.int64:
        lo, hi = lo.to(t.dtype), hi.to(t.dtype)
    if key is not None:
        _LOHI_CACHE[key] = (lo, hi)
    return lo, hi


def _pad_cin_to_8(feats, w):
    """把输入通道零填充到 8 的倍数，让 _GATHER 快路径（spconv_gather_gemm）生效。

    stem 第一层是 Conv3d(4, ...) → cin=4，不满足 `% 8 == 0`，只有这一层会掉进
    `_kernel_map` 回退（sort/index_add_/cat…）。补零通道后结果完全等价：
    多出来的输入通道全 0、对应权重也全 0。
    """
    cin = feats.shape[1]
    if cin % 8 == 0:
        return feats, w
    pad = 8 - cin % 8
    f2 = torch.cat([feats, feats.new_zeros(feats.shape[0], pad)], dim=1)
    # w: [kv, cin, cout]
    w2 = torch.cat([w, w.new_zeros(w.shape[0], pad, w.shape[2])], dim=1)
    return f2, w2


def _lookup_offsets(in_coords, out_coords, stride, offsets):
    """[n_out, kernel_volume] row of `out_coord * stride + offset` inside
    in_coords, -1 when that neighbour voxel does not exist."""
    n_out, kv = out_coords.shape[0], offsets.shape[0]
    cdt = out_coords.dtype                      # 保持 int32，避免 strided int64 cast
    st = _const_tensor(tuple(int(v) for v in stride), cdt, out_coords.device)
    base = out_coords[:, None, 1:] * st
    queries = torch.cat([
        out_coords[:, None, 0:1].expand(-1, kv, 1),
        base + offsets[None].to(cdt),
    ], dim=2)
    q_flat = queries.reshape(-1, 4)

    # Bounds come from the two coord sets' own extremes. Reducing the
    # [n_out, kv, 3] query tensor instead cost 0.70 ms/layer (~28 ms/forward):
    # it materialises 20000x27x3 int64 only to take a min over it.
    lo_off, hi_off = _lo_hi(offsets, cache=True)
    in_lo, in_hi = _lo_hi(in_coords)
    out_lo, out_hi = _lo_hi(out_coords)
    st3 = _const_tensor(tuple(int(v) for v in stride), out_coords.dtype,
                        out_coords.device)
    lo = torch.minimum(in_lo, torch.cat([out_lo[:1], out_lo[1:] * st3 + lo_off]))
    hi = torch.maximum(in_hi, torch.cat([out_hi[:1], out_hi[1:] * st3 + hi_off]))
    mins, dims = lo, hi.long() - lo.long() + 1
    total = _fg.const(lambda: int(torch.prod(dims.long())),
                       ("total_f", tuple(out_coords.shape)))
    qk = pack_keys(q_flat, mins, dims)
    if in_coords.device.type == "npu" and total <= _DENSE_LIMIT:
        idx = _table_fill(total, in_coords, mins, dims, in_coords.device)[qk].long()
    else:
        idx = lookup(qk, pack_keys(in_coords, mins, dims), total)
    return idx.view(n_out, kv)


def _fused_args(in_coords, out_coords, ks, stride, dilation):
    """(rkeys, start, cnt, meta) for SpconvFused -- everything the kernel needs
    to compute the 27 neighbour lookups itself. Bounds mirror _lookup_offsets."""
    import torchsparse.backend as _B

    offsets = get_kernel_offsets(ks, 1, dilation, device=in_coords.device)
    if offsets.shape[0] > 32:
        return None
    st3 = torch.tensor(stride, dtype=in_coords.dtype, device=in_coords.device)
    lo_off, hi_off = offsets.amin(0).long(), offsets.amax(0).long()
    in_lo, in_hi = in_coords.amin(0), in_coords.amax(0)
    out_lo, out_hi = out_coords.amin(0), out_coords.amax(0)
    lo = torch.minimum(in_lo, torch.cat([out_lo[:1], out_lo[1:] * st3 + lo_off]))
    hi = torch.maximum(in_hi, torch.cat([out_hi[:1], out_hi[1:] * st3 + hi_off]))
    dims = hi.long() - lo.long() + 1
    total = _fg.const(lambda: int(torch.prod(dims.long())),
                       ("total_fa", tuple(in_coords.shape)))

    n = in_coords.shape[0]
    nbk = 1
    while nbk < 4 * n:
        nbk <<= 1
    nbk = min(nbk, 1 << 20)
    shift = max(0, (total - 1).bit_length() - (nbk.bit_length() - 1))

    rkeys = _B.pack_keys(in_coords, lo, dims)
    hist = torch.zeros(nbk, dtype=torch.int32, device=in_coords.device)
    hist.index_add_(0, rkeys >> shift,
                    torch.ones(n, dtype=torch.int32, device=in_coords.device))
    start = (torch.cumsum(hist, 0) - hist).to(torch.int32)
    meta = torch.cat([
        torch.tensor([shift], dtype=torch.int32, device=in_coords.device),
        torch.tensor([int(v) for v in stride], dtype=torch.int32,
                     device=in_coords.device),
        lo.to(torch.int32), dims.to(torch.int32),
        offsets.to(torch.int32).reshape(-1),
    ]).contiguous()
    return rkeys, start, hist, meta


_QUERY = os.environ.get("TS_QUERY", "1") != "0"
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


def _rect_map(in_coords, out_coords, ks, stride, dilation):
    """[n_out, kernel_volume] neighbour indices (-1 missing) -- what the
    gather op consumes. No compaction, no host sync.

    The SpconvQuery kernel does coord transform + bounds check + bucket search
    itself (torchsparse's subm_hashmap_kmap_stage2 does the same on CUDA), so
    no [n_out, KV, 4] query tensor is materialised on the host."""
    import torchsparse.backend as _B

    offsets = get_kernel_offsets(ks, 1, dilation, device=in_coords.device)
    if (_QUERY and in_coords.device.type == "npu" and out_coords.shape[0]
            and offsets.shape[0] <= 32 and _query_op()):
        rkeys, start, cnt, meta, _ = _B._query_args(
            in_coords, out_coords, stride, offsets)
        if rkeys is not None:
            return torch.ops.npu.spconv_query(
                out_coords.to(torch.int32), rkeys, start, cnt, meta)
    return _lookup_offsets(in_coords, out_coords, stride, offsets).to(torch.int32)


def _compact_from_nb(nb):
    """(src, dst, counts) from a rectangular [n_out, kv] index map."""
    n_out, kv = nb.shape
    flat = nb.T.reshape(-1).long()
    valid = flat >= 0
    dst = torch.arange(n_out, device=nb.device).repeat(kv)
    sel = _fg.const(lambda: valid.nonzero().squeeze(1),
                    ("sel_nb", tuple(nb.shape)))      # 数据相关形状，图里不放 nonzero
    return (flat.index_select(0, sel), dst.index_select(0, sel),
            _fg.const(lambda: valid.view(kv, n_out).sum(1).tolist(),
                      ("counts_nb", tuple(nb.shape))))


def _kernel_map(in_coords, out_coords, ks, stride, dilation):
    """Compacted neighbour pairs, grouped by kernel offset.

    One selection over the whole [kernel_volume, n_out] matrix (offset-major)
    instead of compare + nonzero + two gathers per offset."""
    offsets = get_kernel_offsets(ks, 1, dilation, device=in_coords.device)
    idx = _lookup_offsets(in_coords, out_coords, stride, offsets)   # [n_out, kv]
    n_out, kv = idx.shape
    flat = idx.T.reshape(-1)                                       # offset-major
    valid = flat >= 0
    dst = torch.arange(n_out, device=idx.device).repeat(kv)
    sel = _fg.const(lambda: valid.nonzero().squeeze(1),
                    ("sel_km", tuple(idx.shape)))     # 同上：nonzero 不能进图
    src = flat.index_select(0, sel)
    dst = dst.index_select(0, sel)
    counts = _fg.const(lambda: valid.view(kv, n_out).sum(1).tolist(),
                       ("counts_km", tuple(idx.shape)))   # 一次 host 同步
    return src, dst, counts


def _apply_forward(feats, weight, kmap, n_out, nb=None):
    return _graph_call("conv_fwd", _apply_forward_eager, feats, weight, kmap, n_out, nb)


def _apply_forward_eager(feats, weight, kmap, n_out, nb=None):
    """out[dst] += feats[src] @ W[k]: one gather + one GEMM per group of kernel
    offsets, in-place scatter per offset."""
    cout = weight.shape[-1]
    if (_GATHER and nb is not None and feats.device.type == "npu"
            and not torch.is_grad_enabled() and _gather_op()
            and feats.shape[1] % 8 == 0):
        ktot = weight.shape[0] * weight.shape[1]
        cin = feats.shape[1]
        if _sgg_fits(nb.shape[0], cin, cout, weight.shape[0]) and _sgg_op():
            try:
                return torch.ops.npu.spconv_gather_gemm(
                    feats, nb, weight.reshape(ktot, cout).contiguous())
            except Exception:
                pass
        dense = torch.ops.npu.spconv_gather(feats, nb)
        return dense @ weight.reshape(ktot, cout)

    kmap = kmap[:3]
    src, dst, counts = kmap
    out = feats.new_zeros((n_out, cout))
    if not any(counts):
        return out

    src_parts, dst_parts = src.split(counts), dst.split(counts)
    if feats.device.type != "npu":
        # CPU: BLAS likes the small per-offset GEMMs; the densified matrix pays
        # for a much larger write.
        for k, n in enumerate(counts):
            if n:
                out.index_add_(0, dst_parts[k],
                               feats.index_select(0, src_parts[k]) @ weight[k])
        return out

    # NPU: densify into one [n_out, K*cin] buffer, then a single GEMM. Each
    # offset writes only its own column block, so total traffic is the valid
    # entries (not 27x the output), and there is no scatter-add read/modify.
    # Measured 1.8x over the per-offset form on 910B4 (20k voxels):
    # 32->32 5.21->2.86 ms, 64->64 5.15->2.81, 128->128 5.10->2.85.
    if _DENSE:
        # One GEMM over a densified [n_out, K*cin] buffer: measured faster in
        # eager (5.2 -> 2.9 ms per layer), but slower under graph capture
        # (ACL graph serialises the 27 large index_put_ writes).
        ktot = weight.shape[0] * weight.shape[1]
        dense = feats.new_zeros((n_out, ktot))
        for k, n in enumerate(counts):
            if n:
                dense[dst_parts[k], k * weight.shape[1]:(k + 1) * weight.shape[1]] = \
                    feats.index_select(0, src_parts[k])
        return dense @ weight.reshape(ktot, cout)

    for g0 in range(0, len(counts), _APPLY_GROUP):
        gk = [k for k in range(g0, min(g0 + _APPLY_GROUP, len(counts))) if counts[k]]
        if not gk:
            continue
        gathered = feats.index_select(0, torch.cat([src_parts[k] for k in gk]))
        mixed = gathered @ torch.cat([weight[k] for k in gk], dim=1)
        off = 0
        for i, k in enumerate(gk):
            n = counts[k]
            out.index_add_(0, dst_parts[k], mixed[off:off + n, i * cout:(i + 1) * cout])
            off += n
    return out


def _apply_transposed(feats, weight, kmap, n_out):
    """Transpose of the forward map: out[src_fine] += feats[dst_coarse] @ W[k]."""
    src, dst, counts = kmap[:3]
    out = feats.new_zeros((n_out, weight.shape[-1]))
    src_parts, dst_parts = src.split(counts), dst.split(counts)
    for k, n in enumerate(counts):
        if n:
            out.index_add_(0, src_parts[k],
                           feats.index_select(0, dst_parts[k]) @ weight[k])
    return out


def conv3d(input, weight, kernel_size, bias=None, stride=1, dilation=1,
           transposed=False):
    ks = make_ntuple(kernel_size, ndim=3)
    st = make_ntuple(stride, ndim=3)
    dl = make_ntuple(dilation, ndim=3)
    w = weight if weight.dim() == 3 else weight.unsqueeze(0)

    feats, coords = input.F, input.C
    caches = input._caches
    # cin 不是 8 的倍数（stem 第一层 Conv3d(4, ...)）时补零到 8 的倍数，
    # 让 _GATHER 快路径（spconv_gather_gemm）生效；多余通道输入/权重都是 0，结果等价。
    # _pad_cin_to_8 只在 GE 兼容模式下启用：它会把 stem 从 `_kernel_map` 切到 SGG 快路径，
    # 而 SGG 那条路上 keyprep 有一次普通 D2H（未被 _fg 冻结），aclgraph capture 期会失败。
    if (_ge_compat() and _GATHER and coords.device.type == "npu"
            and feats.shape[1] % 8 != 0 and _gather_op()):
        feats, w = _pad_cin_to_8(feats, w)

    if ks == (1, 1, 1) and st == (1, 1, 1) and dl == (1, 1, 1):
        out = feats @ w[0]
        out_coords, out_stride = coords, input.stride
    elif not transposed:
        _t0 = _now()
        out_coords = spdownsample(coords, st, ks, dl) \
            if any(s > 1 for s in st) else coords
        _tick("1a_spdownsample", _t0)
        key = (input.stride, ks, st, dl)
        if (_FUSED and coords.device.type == "npu" and coords.shape[0]
                and not torch.is_grad_enabled() and feats.shape[1] % 8 == 0
                and _fused_op()):
            ent = caches.fusedmaps.get(key)
            if ent is None:
                _t0 = _now()
                ent = _fused_args(coords, out_coords, ks, st, dl)
                _tick("1b_kmap_build", _t0)
                caches.fusedmaps[key] = ent
            if ent is not None:
                rkeys, start, cnt, meta = ent
                _t0 = _now()
                dense = torch.ops.npu.spconv_fused(
                    feats, rkeys, start, cnt, out_coords.to(torch.int32), meta)
                cout = w.shape[-1]
                out = dense @ w.reshape(-1, cout)
                _tick("2_conv_apply", _t0)
                out_stride = tuple(input.stride[k] * st[k] for k in range(3))
                if bias is not None:
                    out = out + bias
                caches.cmaps.setdefault(out_stride, out_coords)
                return SparseTensor(out, out_coords, out_stride, caches=caches)
        op_path = (_GATHER and coords.device.type == "npu"
                   and not torch.is_grad_enabled() and _gather_op()
                   and feats.shape[1] % 8 == 0)
        nb_c = None
        if op_path:
            nb_c = caches.nbmaps.get(key)
            if nb_c is None:
                _t0 = _now()
                nb_c = _rect_map(coords, out_coords, ks, st, dl)
                _tick("1b_kmap_build", _t0)
                caches.nbmaps[key] = nb_c
        kmap = caches.kmaps.get(key)
        if kmap is None:
            _t0 = _now()
            kmap = None if op_path else _kernel_map(coords, out_coords, ks, st, dl)
            _tick("1b_kmap_build", _t0)
            caches.kmaps[key] = kmap
        _t0 = _now()
        out = _apply_forward(feats, w, kmap, out_coords.shape[0], nb_c)
        _tick("2_conv_apply", _t0)
        out_stride = tuple(input.stride[k] * st[k] for k in range(3))
    else:
        out_stride = tuple(input.stride[k] // st[k] for k in range(3))
        tkey = (out_stride, ks, st, dl)
        kmap = caches.kmaps.get(tkey)
        out_coords = caches.cmaps.get(out_stride)
        if kmap is None:
            nb_t = caches.nbmaps.get(tkey)
            if nb_t is None and _FUSED and _fused_op() and out_coords is not None:
                # the fused forward path never builds nb; rebuild it from the
                # parent coords it published (rows = child = this input)
                _t0 = _now()
                nb_t = _rect_map(out_coords, coords, ks, st, dl)
                _tick("1b_kmap_build", _t0)
                caches.nbmaps[tkey] = nb_t
            if nb_t is not None:
                kmap = _compact_from_nb(nb_t)
                caches.kmaps[tkey] = kmap
        # ---- 快路径：nb_t 取逆 + 一次 gather_gemm ----
        if (_TS_DECONV_INV and nb_t is not None and out_coords is not None
                and coords.device.type == "npu" and not torch.is_grad_enabled()
                and _GATHER and _sgg_op() and feats.shape[1] % 8 == 0
                and nb_t.shape[0] == feats.shape[0]
                and nb_t.shape[1] == w.shape[0]):
            try:
                _kv = int(w.shape[0])
                _cin = int(feats.shape[1])
                _cout = int(w.shape[-1])
                _nf = int(out_coords.shape[0])
                _nb = nb_t.long()
                _row = torch.where(_nb >= 0, _nb, torch.full_like(_nb, _nf))
                _coarse = torch.arange(_nb.shape[0], device=_nb.device,
                                       dtype=torch.int32).view(-1, 1).expand_as(_nb)
                _buf = torch.full((_nf + 1, _kv), -1, dtype=torch.int32, device=_nb.device)
                _buf.scatter_(0, _row, _coarse)
                _nbT = _buf[:_nf]
                _out = torch.ops.npu.spconv_gather_gemm(
                    feats, _nbT, w.reshape(_kv * _cin, _cout).contiguous())
                if bias is not None:
                    _out = _out + bias
                _tick("2_conv_apply_t", _now())
                caches.cmaps.setdefault(out_stride, out_coords)
                return SparseTensor(_out, out_coords, out_stride, caches=caches)
            except Exception:
                pass
        # ---- 原路径 ----

        if kmap is None:
            raise RuntimeError(
                "transposed conv needs the kernel map of the matching forward "
                f"conv (parent stride {out_stride}, kernel {ks}, stride {st})")
        if out_coords is None:
            raise RuntimeError(f"no cached coord map for stride {out_stride}")
        _t0 = _now()
        out = _apply_transposed(feats, w, kmap, out_coords.shape[0])
        _tick("2_conv_apply_t", _t0)

    if bias is not None:
        out = out + bias
    caches.cmaps.setdefault(out_stride, out_coords)
    return SparseTensor(out, out_coords, out_stride, caches=caches)


def calc_ti_weights(coords, idx_query, scale=1):
    """idx_query is [kernel_volume, N] (GPUHashTable layout); the weights come
    out as [kernel_volume, N] to match spdevoxelize."""
    with torch.no_grad():
        p = coords
        pf = torch.floor(coords / scale) * scale if scale != 1 else torch.floor(coords)
        pc = pf + scale

        x, y, z = (p[:, k].view(-1, 1) for k in range(3))
        xf, yf, zf = (pf[:, k].view(-1, 1).to(p.dtype) for k in range(3))
        xc, yc, zc = (pc[:, k].view(-1, 1).to(p.dtype) for k in range(3))

        w = torch.cat([
            (xc - x) * (yc - y) * (zc - z),
            (xc - x) * (yc - y) * (z - zf),
            (xc - x) * (y - yf) * (zc - z),
            (xc - x) * (y - yf) * (z - zf),
            (x - xf) * (yc - y) * (zc - z),
            (x - xf) * (yc - y) * (z - zf),
            (x - xf) * (y - yf) * (zc - z),
            (x - xf) * (y - yf) * (z - zf),
        ], dim=1).transpose(1, 0).contiguous()
        if scale != 1:
            w = w / scale**3
        # mask-multiply instead of index-put: IndexPut is a random-access op on
        # NPU (see ModelZoo performance_tuning/npu_affinity_opt)
        w = w * (idx_query != -1).to(w.dtype)
        w = w / (w.sum(dim=0) + 1e-8)
    return w


def spdevoxelize(feats, coords, weights):
    return _graph_call("devox", _spdevoxelize_eager, feats, coords, weights)


_DEVOX = os.environ.get("TS_DEVOX", "1") != "0"
# v3 wins 1.4-1.6x over v1 on real layer shapes (per-slot double buffering);
# TS_DEVOX_V picks the build for A/B, default v3.
# 4 = spconv_devox_v4（t214 已编、ge214 已装 vendor）。v3 在 t214 未编，
# 原来默认 3 导致 _devox_op() 为 False → 全模式回退到 torch 的 GatherV2+IndexAdd，
# 实测 aclgraph 20k cr0.5 从 47.3 降到 27.2 ms。缺 .so 时会自动回退，安全。
_DEVOX_V = os.environ.get("TS_DEVOX_V", "4")
_DEVOX_SO = {
    "1": (_OPS_ROOT + "/csrc/ascend/spconv_devox/op_extension/build/libspconv_devox_ops.so"),
    "2": (_OPS_ROOT + "/csrc/ascend/spconv_devox_v2/op_extension/build/libspconv_devox_v2_ops.so"),
    "3": (_OPS_ROOT + "/csrc/ascend/spconv_devox_v3/op_extension/build/libspconv_devox_v3_ops.so"),
    "4": (_OPS_ROOT + "/csrc/ascend/spconv_devox_v4/op_extension/build/libspconv_devox_v4_ops.so"),
}.get(_DEVOX_V, None)
_DEVOX_OP = {"1": "spconv_devox", "2": "spconv_devox_v2",
             "3": "spconv_devox_v3", "4": "spconv_devox_v4"}.get(_DEVOX_V, "spconv_devox_v3")
_DEVOX_OK = None


def _devox_op():
    global _DEVOX_OK
    if _DEVOX_OK is None:
        try:
            torch.ops.load_library(_DEVOX_SO)
            _DEVOX_OK = hasattr(torch.ops.npu, _DEVOX_OP)
        except Exception:
            _DEVOX_OK = False
    return _DEVOX_OK


def _spdevoxelize_eager(feats, coords, weights):
    """Trilinear gather: out[n] = sum_k weights[k, n] * feats[coords[k, n]].

    Flattened into one gather + one in-place accumulate (8 separate gathers and
    8 full-sized copies were ~15x slower here)."""
    idx, w = coords, weights
    if (_DEVOX and feats.device.type == "npu" and _devox_op()
            and idx.dim() == 2 and tuple(idx.shape) == tuple(w.shape)
            and idx.shape[0] <= 8 and feats.dtype == torch.float32
            and w.dtype == torch.float32):
        try:
            # coords is only a sink here: the host infers "compute weights in
            # kernel" from coords.dim(1) == 3, so pass a dummy that can't match
            if _DEVOX_V == "4":
                # v4 多一个 device-n（真实点数），与容量化路径同一签名
                _n = torch.tensor([int(idx.shape[1])], dtype=torch.int32,
                                  device=feats.device)
                return torch.ops.npu.spconv_devox_v4(
                    feats.contiguous(), idx.to(torch.int32), w,
                    w.new_empty((1, 1)), _n)
            return getattr(torch.ops.npu, _DEVOX_OP)(
                feats.contiguous(), idx.to(torch.int32), w,
                w.new_empty((1, 1)))
        except Exception:
            pass
    if idx.shape != w.shape:  # accept [N, K] indices as well
        if tuple(idx.T.shape) == tuple(w.shape):
            idx = idx.T.contiguous()
        else:
            raise ValueError(f"index/weight shape mismatch: {idx.shape} vs {w.shape}")

    K, N = idx.shape
    C = feats.shape[-1]
    idx = idx.long()
    keep = idx >= 0
    rows = torch.arange(N, device=idx.device).repeat(K)
    wflat = w.reshape(-1) * keep.reshape(-1).to(w.dtype)

    out = feats.new_zeros((N, C))
    step = max(1, int(_MEM_BUDGET // max(1, K * C * feats.element_size())))
    wflat = wflat.to(feats.dtype)
    for s in range(0, N, step):
        e = min(s + step, N)
        k0, k1 = s * K, e * K
        gathered = feats.index_select(0, idx.reshape(-1)[k0:k1].clamp(min=0))
        out.index_add_(0, rows[k0:k1], gathered * wflat[k0:k1].unsqueeze(-1))
    return out
