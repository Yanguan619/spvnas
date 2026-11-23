from itertools import repeat

import numpy as np
import torch

from ..utils import make_ntuple

__all__ = ["get_kernel_offsets", "make_ntuple", "make_tensor"]
from ..utils import make_tensor  # noqa: E402


_OFFS_CACHE = {}


def get_kernel_offsets(size, stride=1, dilation=1, device="cpu"):
    """Same layout rule as torchsparse 1.4 / MinkowskiEngine: odd kernel
    volume orders z slowest (`[z][y][x]`), even volume orders x slowest."""
    # 纯 python int：原来走 np.prod / np.arange，在 torch.compile 里是
    # data-dependent branching（graph break），且每次多造一遍 numpy 数组。
    size = tuple(int(v) for v in make_ntuple(size, ndim=3))
    stride = tuple(int(v) for v in make_ntuple(stride, ndim=3))
    dilation = tuple(int(v) for v in make_ntuple(dilation, ndim=3))

    axis = [[v * stride[k] * dilation[k]
             for v in range(-size[k] // 2 + 1, size[k] // 2 + 1)]
            for k in range(3)]
    if size[0] * size[1] * size[2] % 2 == 1:
        offsets = [[x, y, z] for z in axis[2] for y in axis[1] for x in axis[0]]
    else:
        offsets = [[x, y, z] for x in axis[0] for y in axis[1] for z in axis[2]]
    # 按参数缓存：每帧十几次调用、每层一次 H2D；而且 aclgraph capture 阶段
    # 不允许同步 H2D（aclrtMemcpy 107030 / GLOBAL capture mode），所以常量
    # 必须在 capture 之前就建好。
    key = (size, stride, dilation, str(device))
    t = _OFFS_CACHE.get(key)
    if t is None:
        t = torch.tensor(offsets, dtype=torch.long, device=device)
        _OFFS_CACHE[key] = t
    return t
