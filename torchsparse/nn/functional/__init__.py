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
        _FUSED_OK = _load_cap_so("fused")
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


def _op_so_candidates(name):
    """算子 .so 的候选路径，按优先级：

    1. pip 安装后产物随包分发在 ``core/ops/``（`pip install -e .` 自动编译并放入）；
    2. 源码树内 ``<repo>/csrc/ascend/<name>/op_extension/build/``（直接在仓库里跑）；
    3. ``UNUM_OPS_ROOT`` / ``SPVNAS_OPS_ROOT`` 指向的外部构建树。
    """
    import pathlib as _p

    out = []
    try:
        import core.ops as _pkg_ops                       # pip 安装布局
        out.append(str(_p.Path(_pkg_ops.OPS_DIR) / ("lib%s_ops.so" % name)))
    except Exception:
        pass
    try:                                                  # 源码树布局
        repo = _p.Path(__file__).resolve().parents[3]
        out.append(str(repo / "csrc" / "ascend" / name /
                       "op_extension" / "build" / ("lib%s_ops.so" % name)))
    except Exception:
        pass
    for env in ("SPVNAS_OPS_ROOT", "UNUM_OPS_ROOT"):
        root = (os.environ.get(env) or "").rstrip("/")
        if root:
            out.append(root + "/csrc/ascend/%s/op_extension/build/lib%s_ops.so" % (name, name))
            out.append(root + "/%s/op_extension/build/lib%s_ops.so" % (name, name))
    return out


def _load_cap_so(op):
    """按需加载算子 .so（幂等）。返回该算子是否可用。

    依次尝试 :func:`_op_so_candidates` 给出的路径 —— pip 安装（产物在 core/ops）、
    源码树、外部构建树都覆盖，因此**不需要**用户设置任何环境变量。
    """
    name = "spconv_" + op
    if hasattr(torch.ops.npu, name):
        return True
    for so in _op_so_candidates(name):
        if not os.path.exists(so):
            continue
        try:
            torch.ops.load_library(so)
        except Exception:
            continue
        if hasattr(torch.ops.npu, name):
            return True
    return False


_SGG2_SO = _OPS_ROOT + "/csrc/ascend/spconv_gather_gemm_v2/op_extension/build/libspconv_gather_gemm_v2_ops.so"
_SGG2_OK = None


def _sgg_v2_op():
    global _SGG2_OK
    if _SGG2_OK is None:
        _SGG2_OK = _load_cap_so("gather_gemm_v2")
    return _SGG2_OK


def _sgg_op():
    global _SGG_OK
    if _SGG_OK is None:
        _SGG_OK = _load_cap_so("gather_gemm")
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
        _GATHER_OK = _load_cap_so("gather")
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


def spdownsample(coords, stride=2, kernel_size=2, dilation=1, n=None):
    """下采样 + 去重。

    容量化模式（TS_CAP>0）：输入/输出都是 CAP 行，真实行数随返回值用 device 标量带出，
    返回 `(coords_cap, n_new)`；否则保持原行为，只返回 coords。
    """
    st = make_ntuple(stride, ndim=3)
    if cap_enabled():
        from ...backend import unique_coords_cap
        if coords.shape[0] != _CAP:
            raise RuntimeError("容量模式要求输入 coords 已是 %d 行（TS_CAP），实际 %d 行"
                               % (_CAP, coords.shape[0]))
        out = coords.clone()
        for kk in range(3):
            if st[kk] > 1:
                out[:, 1 + kk] = torch.div(out[:, 1 + kk], st[kk], rounding_mode="floor")
        if n is None:
            n = torch.tensor([coords.shape[0]], dtype=torch.int32, device=coords.device)
        out_cap, u = unique_coords_cap(out, n)
        return cap_refill_sentinel(out_cap, u), u
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
    # 必须是**升序**行号：table[key] = 该行的原始行号。
    # 原实现写成降序（arange(n-1,-1,-1)），导致 table[key] = n-1-row，
    # 于是 _lookup_offsets 生成的邻域表整体反序（实测 100% 错位），
    # 而 lookup() 分支返回 order[pos] 是原始行号 —— 两支不一致。
    # table[key] 必须是该行的原始行号。重复键时想存**最小**行号（padding 尾部重复
    # 不能抢走真实键），所以按下标降序写入、值写自己的行号；索引侧必须同步翻转，
    # 否则唯一键会得到 n-1-row，整表反序（实测 100% 错位）。
    _rows = torch.arange(in_coords.shape[0] - 1, -1, -1, dtype=torch.int32,
                         device=device)
    table[keys.flip(0)] = _rows
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

    if cap_enabled() and n_in is not None and n_out is not None \
            and _load_cap_so("keyprep_v3"):
        return _rect_map_cap(in_coords, out_coords, ks, stride, dilation, n_in, n_out)

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
    while nbk < _NBK_MULT * n:
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


def _rect_map_cap(in_coords, out_coords, ks, stride, dilation, n_in, n_out):
    """容量版邻域表：容量 buffer + device 真实行数 → nb_cap[CAP_out, kv]。

    `start` 必须是 `cnt` 的**独占前缀和**（keyprep_v3 的第 2 个输出是未初始化槽位）。
    """
    if not _load_cap_so("keyprep_v3"):
        raise RuntimeError("spconv_keyprep_v3 未加载（检查 UNUM_OPS_ROOT）")
    _load_cap_so("query_v2")
    if not _load_cap_so("keyprep_v3"):
        raise RuntimeError("spconv_keyprep_v3 未加载（检查 UNUM_OPS_ROOT）")
    _load_cap_so("query_v2")
    offs = make_ntuple(ks, ndim=3)
    offs = get_kernel_offsets(offs, 1, make_ntuple(dilation, ndim=3),
                              device=in_coords.device).to(torch.int32)
    st = _const_tensor(tuple(int(v) for v in stride), torch.int32, in_coords.device)
    sz = torch.stack([n_in.reshape(()).to(torch.int32), n_out.reshape(()).to(torch.int32)])
    rk, start, cn, mt = torch.ops.npu.spconv_keyprep_v3(
        in_coords.to(torch.int32).contiguous(), out_coords.to(torch.int32).contiguous(),
        st, offs, _CAP_NBK, sz)
    start = start.to(torch.int32)
    q = out_coords.to(torch.int32).contiguous()
    if hasattr(torch.ops.npu, "spconv_query_v2"):
        nb = torch.ops.npu.spconv_query_v2(q, rk, start, cn, mt,
                                           sz[1:2].contiguous())
    else:
        nb = torch.ops.npu.spconv_query(q, rk, start, cn, mt)
    # nb 的值是**输入特征的行号**：补齐行不存在 → 丢弃为 -1（不能折回末行）
    lim = n_in.reshape(()).to(torch.int32)
    nb = torch.where(nb >= lim, torch.full_like(nb, -1), nb)
    # kernel 只写前 n_out 行，其余是 new_empty 的脏数据；必须显式置 -1，
    # 否则下游 gather 会拿随机索引去读特征（越界 → 507035，且 eager/图不一致）。
    _rows = torch.arange(nb.shape[0], device=nb.device)
    _keep = (_rows < n_out.reshape(())).unsqueeze(1)
    return torch.where(_keep, nb, torch.full_like(nb, -1))


def _rect_map(in_coords, out_coords, ks, stride, dilation,
              n_in=None, n_out=None):
    """[n_out, kernel_volume] neighbour indices (-1 missing) -- what the
    gather op consumes. No compaction, no host sync.

    The SpconvQuery kernel does coord transform + bounds check + bucket search
    itself (torchsparse's subm_hashmap_kmap_stage2 does the same on CUDA), so
    no [n_out, KV, 4] query tensor is materialised on the host."""
    import torchsparse.backend as _B

    # 容量化路径：容量 buffer + device 真实行数 → 固定 CAP 行的 nb
    if (cap_enabled() and n_in is not None and n_out is not None
            and _load_cap_so("keyprep_v3")):
        return _rect_map_cap(in_coords, out_coords, ks, stride, dilation, n_in, n_out)

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


def _kernel_map(in_coords, out_coords, ks, stride, dilation, n_in=None, pad="redirect"):
    """Compacted neighbour pairs, grouped by kernel offset.

    One selection over the whole [kernel_volume, n_out] matrix (offset-major)
    instead of compare + nonzero + two gathers per offset."""
    offsets = get_kernel_offsets(ks, 1, dilation, device=in_coords.device)
    idx = _lookup_offsets(in_coords, out_coords, stride, offsets)   # [n_out, kv]
    if n_in is not None and cap_enabled():
        # 补齐行复用了末行坐标，查表（lookup 对重复键返回**最大**行号）会落在补齐区。
        # 两种用法需要不同处理：
        #   pad="redirect"（前向卷积 gather 特征）：折回 n-1 —— 补齐行就是末行的复制，
        #       语义正确；同时兼作越界安全网。
        #   pad="drop"（转置卷积）：必须丢弃，因为 `_apply_transposed` 按
        #       out[src_fine] += feats[dst_coarse] 累加，折回会把 padding 的 fine 行
        #       全灌进真实末行（实测爆到 1e5）。
        lim = n_in.reshape(()).to(idx.dtype)
        if pad == "drop":
            idx = torch.where(idx >= lim, torch.full_like(idx, -1), idx)
        else:
            idx = torch.where(idx >= lim, lim - 1, idx)
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


def _apply_forward(feats, weight, kmap, n_out, nb=None, n_real=None):
    return _graph_call("conv_fwd", _apply_forward_eager, feats, weight, kmap, n_out, nb,
                       n_real)


def _apply_forward_eager(feats, weight, kmap, n_out, nb=None, n_real=None):
    """out[dst] += feats[src] @ W[k]: one gather + one GEMM per group of kernel
    offsets, in-place scatter per offset."""
    cout = weight.shape[-1]
    if (_GATHER and nb is not None and feats.device.type == "npu"
            and not torch.is_grad_enabled() and _gather_op()
            and feats.shape[1] % 8 == 0):
        ktot = weight.shape[0] * weight.shape[1]
        cin = feats.shape[1]
        if n_real is not None and _sgg_v2_op():
            # 容量模式：真实行数在 device 上，kernel 自己夹紧循环上界。
            # kernel 只写前 n 行，其余是 **未初始化内存** —— 必须置零，否则下游 BN/conv
            # 会读到随机位并放大成 1e20 量级的垃圾（实测过）。
            res = torch.ops.npu.spconv_gather_gemm_v2(
                feats, nb, weight.reshape(ktot, cout).contiguous(),
                n_real.reshape(1).to(torch.int32))
            keep = (torch.arange(res.shape[0], device=res.device)
                    < n_real.reshape(())).unsqueeze(1)
            return torch.where(keep, res,
                               torch.zeros((), dtype=res.dtype, device=res.device))
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
    # 容量模式：真实行数随 SparseTensor 传入；所有分支都需要它（尾部 return 用）
    n_in = getattr(input, "n", None)
    if n_in is None and cap_enabled():
        # 兜底：任何一层只要有 scope 内的 caches，就能按 stride 取回真实行数
        n_in = caches.nmaps.get(tuple(input.stride))
    if n_in is not None:
        caches.nmaps.setdefault(tuple(input.stride), n_in)
    n_out = n_in
    # 全局不变量：补齐行的特征一律视为 0。映射层怎么绕都不会把 padding 值放大进真实行。
    if n_in is not None and cap_enabled():
        _keep = (torch.arange(feats.shape[0], device=feats.device)
                 < n_in.reshape(())).unsqueeze(1)
        feats = torch.where(_keep, feats,
                            torch.zeros((), dtype=feats.dtype, device=feats.device))
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
        if any(s > 1 for s in st):
            ds = spdownsample(coords, st, ks, dl, n=n_in)
            out_coords, n_out = ds if cap_enabled() else (ds, None)
        else:
            out_coords = coords
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
                return SparseTensor(out, out_coords, out_stride, caches=caches, n=n_out)
        op_path = (_GATHER and coords.device.type == "npu"
                   and not torch.is_grad_enabled() and _gather_op()
                   and feats.shape[1] % 8 == 0)
        nb_c = None
        if op_path:
            nb_c = caches.nbmaps.get(key)
            if nb_c is None:
                _t0 = _now()
                nb_c = _rect_map(coords, out_coords, ks, st, dl,
                                 n_in=n_in, n_out=n_out)
                _tick("1b_kmap_build", _t0)
                caches.nbmaps[key] = nb_c
        kmap = caches.kmaps.get(key)
        if kmap is None:
            _t0 = _now()
            kmap = (None if op_path
                    else _kernel_map(coords, out_coords, ks, st, dl, n_in=n_in))
            if (not op_path and cap_enabled() and n_in is not None
                    and out_coords is not None):
                # 转置卷积要一份「补齐条目丢弃」的图（见 _kernel_map 的 pad 说明）
                caches.kmaps_drop[key] = _kernel_map(
                    coords, out_coords, ks, st, dl, n_in=n_in, pad="drop")
            _tick("1b_kmap_build", _t0)
            caches.kmaps[key] = kmap
        _t0 = _now()
        out = _apply_forward(feats, w, kmap, out_coords.shape[0], nb_c, n_real=n_out)
        _tick("2_conv_apply", _t0)
        out_stride = tuple(input.stride[k] * st[k] for k in range(3))
        if n_out is not None:
            caches.nmaps.setdefault(out_stride, n_out)   # 容量模式：供转置卷积取真实行数
    else:
        out_stride = tuple(input.stride[k] // st[k] for k in range(3))
        tkey = (out_stride, ks, st, dl)
        kmap = (caches.kmaps_drop.get(tkey) if cap_enabled()
                else None) or caches.kmaps.get(tkey)
        out_coords = caches.cmaps.get(out_stride)
        if n_out is not None:
            n_out = caches.nmaps.get(out_stride, None)   # 本层真实行数（由对应前向卷积记录）
        if kmap is None:
            nb_t = caches.nbmaps.get(tkey)
            if nb_t is None and _FUSED and _fused_op() and out_coords is not None:
                # the fused forward path never builds nb; rebuild it from the
                # parent coords it published (rows = child = this input)
                _t0 = _now()
                # 容量模式必须带 n，否则会掉回非容量路径（_query_args 里有 host 读回，
                # 入图就是 graph break）。这里 in=细层、out=粗层，与 forward 的 nb 同构。
                nb_t = _rect_map(out_coords, coords, ks, st, dl,
                                 n_in=caches.nmaps.get(out_stride),
                                 n_out=getattr(input, "n", None))
                _tick("1b_kmap_build", _t0)
                caches.nbmaps[tkey] = nb_t
            if nb_t is not None and not cap_enabled():
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
        if kmap is None and nb_t is not None:
            if cap_enabled():
                # 不能退化：host 侧建图（nonzero/tolist）入图后会烘死，静默算错。
                raise RuntimeError(
                    "容量模式下转置卷积的 device-only 快路径未命中"
                    "（检查 feats.shape[1]%%8、nb_t 形状、_TS_DECONV_INV）；"
                    "拒绝回退到 _compact_from_nb，避免入图后静默算错")
            kmap = _compact_from_nb(nb_t)
            caches.kmaps[tkey] = kmap

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
    return SparseTensor(out, out_coords, out_stride, caches=caches, n=n_out)


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

# ---------------------------------------------------------------- 容量化路径（Stage A）
# TS_CAP=<CAP>：coords 固定 CAP 行、真实行数放 device 标量 n（Python 侧不读回），
# 图模式因此无需按形状冻结。默认 0 = 关闭（走原来的固定行数路径）。
_CAP = int(os.environ.get("TS_CAP", "0") or 0)
# 桶数 = mult × 行数（2 的幂）。桶越多碰撞越少，但 Query 要扫整个桶数组，
# 所以这是「内存扫描 vs 桶内二分」的折中；TS_NBK_MULT 可调（默认 4）。
_NBK_MULT = max(1, int(os.environ.get("TS_NBK_MULT", "4") or 4))
_SENTINEL = -1048576          # 哨兵坐标值：远离任何真实体素，且不会与真实键冲突


# 显式固定 nbk（桶数）：kernel 默认按**真实** n 自动算，而 Meta 只能按 CAP 算，
# 两者不一致会让 cumsum(cn) 在未初始化的桶上累加 → 图模式结果错。
# 取 power-of-two ≥ 4*CAP，对任意 n ≤ CAP 都够用，且图形状静态。
_CAP_NBK = 1
while _CAP_NBK < _NBK_MULT * max(_CAP, 1):
    _CAP_NBK <<= 1

# 容量档位（分档用）：默认 3 档。TS_CAP_BUCKETS="8192,16384,20480" 可覆盖。
_CAP_BUCKETS = tuple(
    int(x) for x in (os.environ.get("TS_CAP_BUCKETS") or "8192,16384,20480").split(",")
    if x.strip())


def cap_enabled():
    return _CAP > 0


def cap_size():
    return _CAP


def set_cap(cap):
    """切换容量档位（分档用）。每个 CAP 值对应一张编译好的图：
    torch.compile/torchair 会对模块全局 `_CAP` 加 guard，改值即触发重编，
    所以「按点数选桶」= 每个桶只编译一次。"""
    global _CAP, _CAP_NBK
    cap = int(cap)
    if cap == _CAP:
        return _CAP
    if cap <= 0:
        raise ValueError("set_cap 需要正数")
    _CAP = cap
    nbk = 1
    while nbk < _NBK_MULT * max(cap, 1):
        nbk <<= 1
    _CAP_NBK = nbk
    return _CAP


def cap_for(n, buckets=None):
    """返回能装下 n 个点的最小档位；超出最大档则报错（提示调大档位）。"""
    bs = tuple(int(b) for b in (buckets or _CAP_BUCKETS))
    for b in sorted(bs):
        if n <= b:
            return b
    raise RuntimeError("点数 %d 超过最大容量档 %d，请加入更大的档位" % (n, max(bs)))


# 容量模式下强制 device-only 的转置路径：`_compact_from_nb` 用 nonzero/tolist 在
# host 侧建图，被入图后会烘死在图里（不同帧复用 → 结果错到 97.9%）。实测开这条
# 路径后 编译结果与 eager 完全一致（4.768e-06 / argmax 100%）。
if _CAP > 0:
    _TS_DECONV_INV = True


def cap_size():
    return _CAP


def cap_pad_coords(coords):
    """把 coords 补齐到 CAP 行（尾部哨兵），返回 (coords_cap, n_dev)。

    真实行数取 coords 现有行数 —— 只在**入口**调用一次（此时行数就是真实行数）。"""
    dev = coords.device
    k = coords.shape[0]
    if k > _CAP:
        raise RuntimeError("TS_CAP=%d < 实际行数 %d" % (_CAP, k))
    n = torch.tensor([k], dtype=torch.int32, device=dev)
    if k == _CAP:
        return coords.contiguous(), n
    # 补齐行 = **重复末行有效坐标**（不能用极端哨兵值：那会把 keyprep 的包围盒/
    # 内置归约打崩，实测触发 507035 vector core exception；重复有效行既不扩包围盒，
    # 也只会产生"已存在的键"，对查询无副作用）
    tail = coords[-1:].repeat(_CAP - k, 1)
    return torch.cat([coords, tail], 0).contiguous(), n


def cap_refill_sentinel(coords_cap, u):
    """dedup 之后：kernel 输出的尾部行是未定义的，必须重填（device 侧，无同步）。

    重填值用 **行 0（必为真实体素）的重复**，不用极端哨兵：极端值会打崩内置归约。
    尾部是"已存在的键"的重复，所以不会给后续查询引入新键。
    """
    dev = coords_cap.device
    keep = (torch.arange(_CAP, device=dev) < u.reshape(())).unsqueeze(1)
    # 统一约定：补齐/重填行 = **最后一行的重复**（与 cap_pad_coords 一致），
    # 这样各处 clamp「>= n → n-1」都能正确指回被重复的真实行。
    j = (u.reshape(()).to(torch.long) - 1).clamp(min=0)
    tail = torch.gather(coords_cap, 0, j.expand(1, coords_cap.shape[1]))
    return torch.where(keep, coords_cap, tail)
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


# 容量化 .so 必须在入图**之前**加载：在被 torch.compile trace 的区域里调用
# torch.ops.load_library 会让 GE 报 "GRAPH internal error"。
if _CAP > 0:
    # 固定行数版也要预加载：转置卷积的 device-only 快路径依赖 _sgg_op()，
    # 而它在首次调用里做 load_library —— 那已经发生在被 trace 的区域了。
    for _op in ("keyprep_v3", "query_v2", "gather_gemm_v2", "sortdedup_v4",
                "sortdedup_v3", "keyprep_v1", "keyprep_v2", "query",
                "gather", "gather_v2", "gather_gemm", "devox", "devox_v3",
                "nb"):
        _load_cap_so(_op)
