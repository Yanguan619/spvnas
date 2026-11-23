"""Minimal stand-in for torch_scatter (only scatter_mean is used here)."""
import torch

__all__ = ["scatter_mean"]


def _const(fn, key=None):
    """数据相关标量：整模入图时由 torchsparse._fgstatic 固化成常量。

    eager 下等同于直接算（record 模式返回真实值）；编译期则不再把
    int(index.max()) 变成图里的标量算子（那样 FX 执行时会炸）。
    """
    try:
        from torchsparse import _fgstatic
    except Exception:
        return fn()
    return _fgstatic.const(fn, key)


def scatter_mean(src, index, dim=0, dim_size=None, out=None):
    if dim != 0:
        raise NotImplementedError("only dim=0 is supported")
    index = index.long().reshape(-1)
    if dim_size is None:
        dim_size = _const(lambda: int(index.max()) + 1,
                          ("dim", tuple(index.shape)))
    res = src.new_zeros((dim_size,) + tuple(src.shape[1:]))
    res = res.index_add(0, index, src)
    ones = torch.ones(index.shape[0], dtype=src.dtype, device=src.device)
    cnt = src.new_zeros(dim_size).index_add(0, index, ones)
    return res / cnt.clamp(min=1).view((-1,) + (1,) * (src.dim() - 1))
