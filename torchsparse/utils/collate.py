from typing import Any, List

import numpy as np
import torch

from ..tensor import SparseTensor

__all__ = ["sparse_collate", "sparse_collate_fn"]


def sparse_collate(inputs: List[SparseTensor]) -> SparseTensor:
    coords, feats = [], []
    stride = inputs[0].stride

    for k, x in enumerate(inputs):
        if isinstance(x.coords, np.ndarray):
            x.coords = torch.tensor(x.coords)
        if isinstance(x.feats, np.ndarray):
            x.feats = torch.tensor(x.feats)

        assert x.stride == stride, (x.stride, stride)
        batch = torch.full((x.coords.shape[0], 1), k, dtype=torch.int32)
        coords.append(torch.cat((batch, x.coords.int()), dim=1))
        feats.append(x.feats)

    return SparseTensor(torch.cat(feats, dim=0), torch.cat(coords, dim=0), stride)


def sparse_collate_fn(inputs: List[Any]) -> Any:
    if isinstance(inputs[0], dict):
        output = {}
        for name in inputs[0].keys():
            if isinstance(inputs[0][name], dict):
                output[name] = sparse_collate_fn([x[name] for x in inputs])
            elif isinstance(inputs[0][name], np.ndarray):
                output[name] = torch.stack(
                    [torch.tensor(x[name]) for x in inputs], dim=0)
            elif isinstance(inputs[0][name], torch.Tensor):
                output[name] = torch.stack([x[name] for x in inputs], dim=0)
            elif isinstance(inputs[0][name], SparseTensor):
                output[name] = sparse_collate([x[name] for x in inputs])
            else:
                output[name] = [x[name] for x in inputs]
        return output
    return inputs
