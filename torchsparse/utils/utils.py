from itertools import repeat

import torch

__all__ = ["make_ntuple", "make_tensor"]


def make_ntuple(x, ndim):
    if isinstance(x, int):
        x = tuple(repeat(x, ndim))
    elif isinstance(x, list):
        x = tuple(x)
    elif isinstance(x, torch.Tensor):
        x = tuple(x.view(-1).cpu().numpy().tolist())
    assert isinstance(x, tuple) and len(x) == ndim, x
    return x


def make_tensor(x, device=None, dtype=None):
    if isinstance(x, int):
        x = (x, x, x)
    return torch.tensor(x, device=device, dtype=dtype)
