"""容量化（capacity + device-n）算子链的参考实现 —— 「任意 N 一张图」的接线配方。

为什么需要这一层
----------------
固定行数版算子的输出行数由 kernel 在 device 上算出 `U`，Python 侧必须
`out[: int(u)]` 把它读回 host —— 在图上这就是一次同步，`torch.compile` 只能把
这批值**冻结**（`_fgstatic` 的 record/replay），于是「一张图」就被绑定到录制时的形状。

容量化版把这件事反过来：**输出固定为容量 CAP 行**，真实行数放在 device 标量 `n` 里，
Python 侧永远不读回。于是图里所有张量形状都是静态的（= CAP），换帧无需重录/重编；
补齐行由调用方用 `arange(CAP) < n` 掩掉（不能进入任何 reduction）。

算子契约（与 csrc/ascend/*/op_extension/register.cpp 一致）
----------------------------------------------------------
  n      = torch.tensor([n], dtype=torch.int32, device=...)      # 真实行数（设备标量）
  sizes  = torch.tensor([n_in, n_out], dtype=torch.int32, ...)   # keyprep 用

  spconv_sortdedup_v4(coords_cap, n)          -> [out_cap, u]      u: int64[1] 真实去重数
  spconv_keyprep_v3(in_cap, out_cap, stride, offs, nbk, sizes)
                                              -> [rkeys, start, cnt, meta]
  spconv_query_v2(out_cap, rkeys, start, cnt, meta, n_out) -> nb_cap [CAP_out, kv]
  spconv_gather_gemm_v2(feat, nb_cap, w, n_out)            -> [CAP_out, cout]
  spconv_devox_v4(feat, idx_cap, w_cap, coords, n)          -> [CAP, cin]

用法
----
    import ascend_capacity as cap
    n = cap.n_scalar(real_rows, dev)
    nb = cap.kernel_map(ic_cap, oc_cap, sizes=[n_in, n_out], ks=3, stride=2, dev=dev)
    feats = cap.conv(feats, nb, weight, n_out)

本模块**不改** `torchsparse` 的默认路径（默认仍是固定行数版）。模型侧接线只需把对应
调用换成这里的函数，并用 `mask = cap.valid_mask(CAP, n)` 处理补齐行。
"""

import os

import torch

__all__ = ["load", "available", "n_scalar", "valid_mask", "unique_coords_cap",
           "kernel_map", "conv", "devox"]

_REQUIRED = ("spconv_sortdedup_v4", "spconv_keyprep_v3", "spconv_query_v2",
             "spconv_gather_gemm_v2", "spconv_devox_v4")


def load(ops_root=None, quiet=True):
    """按 UNUM_OPS_ROOT 加载 5 个容量化 .so（幂等）。torchsparse 默认不会加载它们。"""
    root = ops_root or os.environ.get("UNUM_OPS_ROOT", "")
    if not root:
        return False
    for op in _REQUIRED:
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
    """5 个容量化算子是否都已加载（未加载时应回退固定行数版）。"""
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
    """容量版邻域表：输入输出的**容量** buffer + 真实 sizes，返回 nb_cap[CAP_out, kv]。

    `ks` 支持 int 或 3 元组；`stride`/`dilation` 同 conv 语义。
    补齐行不参与（sizes 里给真实值），返回的补齐行是 -1，与固定行数版语义一致。
    """
    from torchsparse.utils import make_ntuple
    from torchsparse.nn.utils import get_kernel_offsets

    dev = dev or ic_cap.device
    ks = make_ntuple(ks, 3)
    stride = make_ntuple(stride, 3)
    offs = get_kernel_offsets(ks, dilation, device=dev).to(torch.int32)
    s_st = torch.tensor(list(stride), dtype=torch.int32, device=dev)
    s_sz = torch.tensor([int(sizes[0]), int(sizes[1])], dtype=torch.int32, device=dev)

    rk, st, cn, mt = torch.ops.npu.spconv_keyprep_v3(
        ic_cap.to(torch.int32).contiguous(), oc_cap.to(torch.int32).contiguous(),
        s_st, offs, int(nbk), s_sz)
    nb = torch.ops.npu.spconv_query_v2(
        oc_cap.to(torch.int32).contiguous(), rk, st, cn, mt, s_sz[1:2].contiguous())
    return nb


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
        feats, idx_cap.to(torch.int32).contiguous(), w_cap, coords,
        n_scalar(n, dev))


load()
