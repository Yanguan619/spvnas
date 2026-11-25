"""Minimal pure-PyTorch stand-in for torchsparse (v1.4 API subset).

Only the operators used by SPVNAS/SPVCNN/MinkUNet are implemented, with the
exact coordinate / stride / kernel-offset conventions of torchsparse 1.4 so
that checkpoints trained with the CUDA build load and behave the same.
Runs on CPU and NPU (no CUDA extension, no compiler at import time).
"""
from .tensor import SparseTensor, PointTensor, cat
from . import backend, nn, utils

__all__ = ["SparseTensor", "PointTensor", "cat", "backend", "nn", "utils"]
