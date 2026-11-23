import numpy as np
import torch
import torchsparse
import torchsparse.nn.functional as spf
from torchsparse import SparseTensor
from torchsparse.nn.functional.devoxelize import calc_ti_weights
from torchsparse.nn.utils import *
from torchsparse.backend import unique_coords, unique_coords_cap
from torchsparse.utils import *
from torchsparse.utils.tensor_cache import TensorCache
import torch_scatter
from typing import Union, Tuple

__all__ = ["initial_voxelize", "point_to_voxel", "voxel_to_point", "PointTensor"]


class PointTensor(SparseTensor):
    def __init__(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        stride: Union[int, Tuple[int, ...]] = 1,
    ):
        super().__init__(feats=feats, coords=coords, stride=stride)
        self._caches.idx_query = dict()
        self._caches.idx_query_devox = dict()
        self._caches.weights_devox = dict()


def sphashquery(query, target, kernel_size=1):
    hashmap_keys = torch.zeros(
        2 * target.shape[0], dtype=torch.int64, device=target.device
    )
    hashmap_vals = torch.zeros(
        2 * target.shape[0], dtype=torch.int32, device=target.device
    )
    hashmap = torchsparse.backend.GPUHashTable(hashmap_keys, hashmap_vals)
    hashmap.insert_coords(torch.cat([target[:, 1:4], target[:, 0:1]], dim=1))
    kernel_size = make_ntuple(kernel_size, 3)
    # 纯 Python 乘积：np.prod 在 torch.compile 下会被 trace 成 aten.prod，
    # 而 GE 没有它的转换器（NotImplementedError: ge_converter ... prod）
    kernel_volume = 1
    for _v in kernel_size:
        kernel_volume *= int(_v)
    # 传 python int/tuple：原来先 make_tensor 到 device、再在 lookup_coords 里
    # .tolist() 回来，等于每层白送一次 H2D+D2H（也是入图的一个 graph break）。
    results = (
        hashmap.lookup_coords(
            torch.cat([query[:, 1:4], query[:, 0:1]], dim=1),
            kernel_size, (1, 1, 1), kernel_volume
        )
        - 1
    )[: query.shape[0]]
    return results


# z: PointTensor
# return: SparseTensor
def _cap_fix_index(idx, n):
    """容量模式：把落在补齐区的索引映射回行 0（尾部是行 0 坐标的重复）。"""
    if n is None or not spf.cap_enabled():
        return idx
    lim = n.reshape(()).to(idx.dtype)
    return torch.where(idx >= lim, (lim - 1).to(idx.dtype), idx)


def _masked_mean(feats, idx, n_rows):
    """按 idx 做「加权和 / 有效计数」的均值，输出恒为 n_rows 行（补齐行得 0）。

    不用 scatter_mean：它按 max(idx)+1 推输出行数，容量模式下会短于 CAP。
    """
    dev, dt = feats.device, feats.dtype
    sums = torch.zeros((n_rows, feats.shape[1]), dtype=dt, device=dev)
    sums.index_add_(0, idx.long(), feats)
    ones = torch.ones(feats.shape[0], dtype=dt, device=dev)
    cnts = torch.zeros(n_rows, dtype=dt, device=dev)
    cnts.index_add_(0, idx.long(), ones)
    return sums / cnts.clamp(min=1).unsqueeze(1)


def initial_voxelize(z, init_res, after_res):
    new_float_coord = torch.cat(
        [z.C[:, 0].view(-1, 1), (z.C[:, 1:] * init_res) / after_res], 1
    )
    # optimization TBD: init_res = after_res
    new_int_coord = torch.floor(new_float_coord).int()
    if spf.cap_enabled():
        cap_coord, n_pts = spf.cap_pad_coords(new_int_coord)
        sparse_coord, u = unique_coords_cap(cap_coord, n_pts)
        sparse_coord = spf.cap_refill_sentinel(sparse_coord, u)
        # 查询索引**用紧凑的唯一坐标**算：容量 buffer 的尾部是重复行，直接查会污染映射
        # （实测返回全 0）。两者顺序一致（都由 sortdedup 产出、按 pack_keys 单调），
        # 所以紧凑索引可直接用作 CAP 空间的索引（真实体素都在前 u 行）。
        sparse_ref = unique_coords(new_int_coord)
        idx_query = sphashquery(new_int_coord, sparse_ref).reshape(-1)
        sparse_feat = _masked_mean(z.F, idx_query, sparse_coord.shape[0])
        new_tensor = SparseTensor(sparse_feat, sparse_coord, 1, n=u)
    else:
        sparse_coord = unique_coords(new_int_coord)
        idx_query = sphashquery(new_int_coord, sparse_coord).reshape(-1)
        sparse_feat = torch_scatter.scatter_mean(z.F, idx_query.long(), dim=0)
        new_tensor = SparseTensor(sparse_feat, sparse_coord, 1)
    z._caches.idx_query[z.s] = idx_query
    z.C = new_float_coord
    return new_tensor


# x: SparseTensor, z: PointTensor
# return: SparseTensor
def point_to_voxel(x, z):
    if z._caches.idx_query.get(x.s) is None:
        # Note: x.C has a smaller range after downsampling.
        new_int_coord = torch.cat(
            [
                z.C[:, 0].int().view(-1, 1),
                torch.floor(z.C[:, 1:] / x.s[0]).int(),
            ],
            1,
        )
        idx_query = sphashquery(new_int_coord, x.C)
        z._caches.idx_query[x.s] = idx_query
    else:
        idx_query = z._caches.idx_query[x.s]
    # Haotian: This impl. is not elegant
    idx_query = idx_query.clamp_(0)
    if spf.cap_enabled() and getattr(x, "n", None) is not None:
        idx_query = _cap_fix_index(idx_query, x.n)
        sparse_feat = _masked_mean(z.F, idx_query.reshape(-1), x.C.shape[0])
        new_tensor = SparseTensor(sparse_feat, x.C, x.s, n=x.n)
        new_tensor._caches = x._caches
        return new_tensor
    sparse_feat = torch_scatter.scatter_mean(z.F, idx_query.long(), dim=0)
    new_tensor = SparseTensor(sparse_feat, x.C, x.s)
    new_tensor._caches = x._caches

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False):
    if (
        z._caches.idx_query_devox.get(x.s) is None
        or z._caches.weights_devox.get(x.s) is None
    ):
        point_coords_float = torch.cat(
            [z.C[:, 0].int().view(-1, 1), z.C[:, 1:] / x.s[0]],
            1,
        )
        point_coords_int = torch.floor(point_coords_float).int()
        idx_query = sphashquery(point_coords_int, x.C, kernel_size=2)
        if spf.cap_enabled() and getattr(x, "n", None) is not None:
            idx_query = _cap_fix_index(idx_query, x.n)
        weights = calc_ti_weights(point_coords_float[:, 1:], idx_query, scale=1)

        if nearest:
            weights[:, 1:] = 0.0
            idx_query[:, 1:] = -1
        new_feat = spf.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat, z.C)
        new_tensor._caches = z._caches
        new_tensor._caches.idx_query_devox[x.s] = idx_query
        new_tensor._caches.weights_devox[x.s] = weights
        z._caches.idx_query_devox[x.s] = idx_query
        z._caches.weights_devox[x.s] = weights

    else:
        new_feat = spf.spdevoxelize(
            x.F, z._caches.idx_query_devox.get(x.s), z._caches.weights_devox.get(x.s)
        )
        new_tensor = PointTensor(new_feat, z.C)
        new_tensor._caches = z._caches

    return new_tensor
