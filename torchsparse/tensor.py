import torch

from .utils.tensor_cache import TensorCache


def _tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


class SparseTensor:
    """feats: [N, C]; coords: [N, 4] int, columns are (batch, x, y, z) in
    level-0 voxel units; stride: spatial stride of this level."""

    def __init__(self, feats, coords, stride=1, spatial_range=None, caches=None,
                 n=None):
        # n：容量化路径（TS_CAP>0）下「真实行数」的 device 标量 int32[1]；
        #    None 表示固定行数模式（默认），此时 coords 行数就是真实行数。
        self.n = n
        self.feats = _tensor(feats)
        self.coords = _tensor(coords)
        self.stride = (stride, stride, stride) if isinstance(stride, int) else tuple(stride)
        self.spatial_range = spatial_range
        self._caches = TensorCache() if caches is None else caches

    @property
    def F(self):
        return self.feats

    @F.setter
    def F(self, feats):
        self.feats = feats

    @property
    def C(self):
        return self.coords

    @C.setter
    def C(self, coords):
        self.coords = coords

    @property
    def s(self):
        return self.stride

    @s.setter
    def s(self, stride):
        self.stride = (stride, stride, stride) if isinstance(stride, int) else tuple(stride)

    @property
    def kmaps(self):
        return self._caches.kmaps

    @property
    def cmaps(self):
        return self._caches.cmaps

    def to(self, device, non_blocking=True):
        self.coords = self.coords.to(device)
        self.feats = self.feats.to(device)
        return self

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")

    def half(self):
        self.feats = self.feats.half()
        return self

    def float(self):
        self.feats = self.feats.float()
        return self

    def detach(self):
        return SparseTensor(self.feats.detach(), self.coords.detach(), self.stride,
                            self.spatial_range, self._caches, n=self.n)

    def clone(self):
        return SparseTensor(self.feats.clone(), self.coords.clone(), self.stride,
                            self.spatial_range, self._caches, n=self.n)

    def __len__(self):
        return self.coords.shape[0]

    def __add__(self, other):
        assert (torch.compiler.is_compiling()
                or torch.equal(self.coords, other.coords)), \
            "SparseTensor add needs identical coord sets"
        return SparseTensor(self.feats + other.feats, self.coords, self.stride,
                            self.spatial_range, self._caches, n=self.n)

    def __repr__(self):
        return (f"SparseTensor(feats={tuple(self.feats.shape)}, "
                f"coords={tuple(self.coords.shape)}, stride={self.stride})")


class PointTensor(SparseTensor):
    pass


def cat(tensors, dim=-1):
    first = tensors[0]
    for t in tensors[1:]:
        assert (t.stride == first.stride
                or torch.compiler.is_compiling()) \
            and (torch.compiler.is_compiling()
                 or torch.equal(t.coords, first.coords)), \
            "torchsparse.cat needs identical coords and stride"
    return SparseTensor(torch.cat([t.feats for t in tensors], dim=dim),
                        first.coords, first.stride, first.spatial_range,
                        first._caches, n=getattr(first, "n", None))
