import math

import numpy as np
import torch
from torch import nn

from ..tensor import SparseTensor
from ..utils import make_ntuple
from . import functional

__all__ = ["Conv3d", "BatchNorm", "ReLU", "functional"]


def _wrap(feats, tensor):
    return SparseTensor(feats, tensor.C, tensor.s, caches=tensor._caches,
                        n=getattr(tensor, "n", None))


@torch.no_grad()
def fold_bn(model):
    """Fold BatchNorm1d into the preceding Conv3d / Linear -- inference only.

    Removes one full read+write of the feature tensor per BN layer (the model
    has ~38 of them); the result is the same up to fp32 rounding."""
    folded = 0
    for module in model.modules():
        if not isinstance(module, nn.Sequential):
            continue
        kids = list(module.named_children())
        for i in range(len(kids) - 1):
            name, conv = kids[i]
            nxt, bn = kids[i + 1]
            if not isinstance(bn, nn.BatchNorm1d) or bn.training:
                continue
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            shift = bn.bias - bn.running_mean * scale
            if isinstance(conv, nn.Linear):
                conv.weight.data *= scale.view(-1, 1)
            elif getattr(conv, "kernel", None) is not None:
                conv.kernel.data *= (scale.view(1, 1, -1) if conv.kernel.dim() == 3
                                     else scale.view(1, -1))
            else:
                continue
            if conv.bias is None:
                conv.bias = nn.Parameter(shift.clone())
            else:
                conv.bias.data = shift.clone()
            setattr(module, nxt, nn.Identity())
            folded += 1
    return folded


class Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1, bias=False, transposed=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_ntuple(kernel_size, ndim=3)
        self.stride = make_ntuple(stride, ndim=3)
        self.dilation = dilation
        self.transposed = transposed

        self.kernel_volume = int(np.prod(self.kernel_size))
        if self.kernel_volume > 1:
            self.kernel = nn.Parameter(
                torch.zeros(self.kernel_volume, in_channels, out_channels))
        else:
            self.kernel = nn.Parameter(torch.zeros(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1 / math.sqrt((self.out_channels if self.transposed
                             else self.in_channels) * self.kernel_volume)
        self.kernel.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}"
        if self.stride != (1, 1, 1):
            s += ", stride={stride}"
        if self.dilation != 1:
            s += ", dilation={dilation}"
        if self.bias is None:
            s += ", bias=False"
        if self.transposed:
            s += ", transposed=True"
        return s.format(**self.__dict__)

    def forward(self, input):
        return functional.conv3d(input, self.kernel, kernel_size=self.kernel_size,
                                 bias=self.bias, stride=self.stride,
                                 dilation=self.dilation, transposed=self.transposed)


class BatchNorm(nn.BatchNorm1d):

    def forward(self, input):
        if isinstance(input, SparseTensor):
            return _wrap(super().forward(input.F), input)
        return super().forward(input)


class ReLU(nn.Module):

    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        if isinstance(input, SparseTensor):
            return _wrap(nn.functional.relu(input.F, self.inplace), input)
        return nn.functional.relu(input, self.inplace)

    def extra_repr(self):
        return f"inplace={self.inplace}"
