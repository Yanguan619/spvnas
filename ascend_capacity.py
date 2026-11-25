"""容量化（capacity + device-n）算子链的参考实现 —— 「任意 N 一张图」的接线配方。

为什么需要这一层
----------------
固定行数版算子的输出行数由 kernel 在 device 上算出 `U`，Python 侧必须
`out[: int(u)]` 把它读回 host —— 在图上这就是一次同步，`torch.compile` 只能把
这批值**冻结**（`_fgstatic` 的 record/replay），于是「一张图」被绑定到录制时的形状。

容量化版把这件事反过来：**输出固定为容量 CAP 行**，真实行数放在 device 标量 `n` 里，
Python 侧永远不读回。于是图里所有张量形状都是静态的（= CAP），换帧无需重录/重编；
补齐行由调用方用 `arange(CAP) < n` 掩掉（不能进入任何 reduction）。

⚠️ 前置条件（很关键，踩过坑）
----------------------------
`query`/`query_v2` 在桶内是按 **key 有序** 来定位的，所以传入的 coords 必须
**唯一、且按 `pack_keys` 的单调序排序** —— 生产里 `unique_coords` 的输出正是这种形态。
随机/未排序/有重复键的 coords 会让 `query` 返回全 `-1`（我校验时正是踩了这条）。

算子契约（与 csrc/ascend/*/op_extension/register.cpp 一致）
----------------------------------------------------------
  n      = torch.tensor([n], dtype=torch.int32, device=...)      # 真实行数（设备标量）
  sizes  = torch.tensor([n_in, n_out], dtype=torch.int32, ...)   # keyprep 用

  spconv_sortdedup_v4(coords_cap, n)          -> [out_cap, u]      u: int64[1] 真实去重数
  spconv_keyprep_v3(in_cap, out_cap, stride, offs, nbk, sizes)
                                              -> [rkeys, **未使用**, cnt, meta]
  spconv_query_v2(out_cap, rkeys, start, cnt, meta, n_out) -> nb_cap [CAP_out, kv]
  spconv_gather_gemm_v2(feat, nb_cap, w, n_out)            -> [CAP_out, cout]
  spconv_devox_v4(feat, idx_cap, w_cap, coords, n)          -> [CAP, cin]

  **`start` 不是 keyprep 的第 2 个输出**（v3 那个位置是未初始化的 `start` 槽位），
  而是 `cnt` 的 **独占前缀和**：`start = cumsum(cnt) - cnt`
  （与 `spconv_keyprep_v2` 里独立的 `SpconvKeyprepV2Prefix` kernel 等价）。

用法
----
    import ascend_capacity as cap
    n = cap.n_scalar(real_rows, dev)
    nb = cap.kernel_map(ic_cap, oc_cap, sizes=[n_in, n_out], ks=3, stride=2, dev=dev)
    feats = cap.conv(feats, nb, weight, n_out)

本模块**不改** `torchsparse` 的默认路径（默认仍是固定行数版）。
"""

import os

import torch

__all__ = ["load", "available", "n_scalar", "valid_mask", "unique_coords_cap",
           "kernel_map", "conv", "devox"]

_REQUIRED = ("spconv_sortdedup_v4", "spconv_keyprep_v3", "spconv_query_v2",
             "spconv_gather_gemm_v2", "spconv_devox_v4")

# 调试开关：强制用固定版 spconv_query（无 n 参数，跑整个容量 buffer）
_FORCE_Q1 = os.environ.get("TS_CAP_QUERY_V1", "") == "1"


def load(ops_root=None, quiet=True):
    """按 UNUM_OPS_ROOT 加载容量化 .so（幂等）。torchsparse 默认不会加载它们。"""
    root = ops_root or os.environ.get("UNUM_OPS_ROOT", "")
    if not root:
        return False
    for op in _REQUIRED + ("spconv_query",):
        if hasattr(torch.ops.npu, op):
            continue
        so = os.path.join(root, "csrc", "ascend", op, "op_extension", "build",
                          "lib%s_ops.so" % op)
        if not os.path.exists(so) or not hasattr(torch.ops, "load_library"):
            continue
        try:
            torch.ops.load_library(so)
        except Exception:
            if not quiet:
                raise
    return available()


def available():
    """容量化算子是否都已加载（未加载时应回退固定行数版）。"""
    return all(hasattr(torch.ops.npu, n) for n in _REQUIRED)


def n_scalar(n, device):
    """真实行数 → device 侧 int32[1] 标量（不触发任何 host 同步）。"""
    if torch.is_tensor(n):
        return n.to(device=device, dtype=torch.int32).reshape(1)
    return torch.tensor([int(n)], dtype=torch.int32, device=device)


def valid_mask(cap_rows, n, device):
    """补齐行掩码：前 n 行为 True。用于把补齐行从 reduction / scatter 里排除。"""
    return torch.arange(cap_rows, device=device) < n_scalar(n, device).reshape(1)


def unique_coords_cap(coords_cap, n, device=None):
    """容量版去重。返回 (coords_cap_out, u) —— u 是 device 上的 int64[1]，**不要**读回 host。"""
    device = device or coords_cap.device
    out, u = torch.ops.npu.spconv_sortdedup_v4(
        coords_cap.to(torch.int32).contiguous(), n_scalar(n, device))
    return out, u


def kernel_map(ic_cap, oc_cap, sizes, ks=3, stride=1, dilation=1, nbk=0, dev=None):
    """容量版邻域表：容量 buffer + 真实 sizes → nb_cap[CAP_out, kv]（补齐行是 -1）。

    前置：`ic_cap` / `oc_cap` 必须**唯一且按 pack_keys 单调序排序**（`unique_coords` 的输出即如此）。
    """
    from torchsparse.utils import make_ntuple
    from torchsparse.nn.utils import get_kernel_offsets

    dev = dev or ic_cap.device
    ks = make_ntuple(ks, 3)
    stride = make_ntuple(stride, 3)
    offs = get_kernel_offsets(ks, 1, dilation, device=dev).to(torch.int32)
    s_st = torch.tensor(list(stride), dtype=torch.int32, device=dev)
    s_sz = torch.tensor([int(sizes[0]), int(sizes[1])], dtype=torch.int32, device=dev)
    ic = ic_cap.to(torch.int32).contiguous()
    oc = oc_cap.to(torch.int32).contiguous()

    rk, _unused_start, cn, mt = torch.ops.npu.spconv_keyprep_v3(ic, oc, s_st, offs, int(nbk), s_sz)
    # query 的 `start` 参数 = 独占前缀和（== keyprep_v2 里 Prefix kernel 的产物）
    start = (torch.cumsum(cn, 0) - cn).to(torch.int32)

    if hasattr(torch.ops.npu, "spconv_query_v2") and not _FORCE_Q1:
        return torch.ops.npu.spconv_query_v2(oc, rk, start, cn, mt, s_sz[1:2].contiguous())
    # 兜底：固定版 query 没有 n 参数，会把整个容量 buffer 都算一遍（结果由调用方掩掉）
    if hasattr(torch.ops.npu, "spconv_query"):
        return torch.ops.npu.spconv_query(oc, rk, start, cn, mt)
    raise RuntimeError("neither spconv_query_v2 nor spconv_query works")


def conv(feats, nb_cap, weight, n_out, dev=None):
    """容量版稀疏卷积（gather+GEMM）。nb_cap 来自 `kernel_map`，返回 [CAP_out, cout]。"""
    return torch.ops.npu.spconv_gather_gemm_v2(
        feats, nb_cap.to(torch.int32).contiguous(), weight,
        n_scalar(n_out, dev or feats.device))


def devox(feats, idx_cap, w_cap, n, coords=None, dev=None):
    """容量版 devoxelize（feat[idx] 加权求和）。coords 占位传 (1,1) 表示用给定的 w。"""
    dev = dev or feats.device
    if coords is None:
        coords = torch.zeros(1, 1, device=dev)
    return torch.ops.npu.spconv_devox_v4(
        feats, idx_cap.to(torch.int32).contiguous(), w_cap, coords, n_scalar(n, dev))


load()
