#include <torch/extension.h>
#include "spconv_devox_v4_ops.h"

TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("spconv_devox_v4(Tensor feat, Tensor idx, Tensor w, Tensor coords, Tensor n) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("spconv_devox_v4", [](const at::Tensor& feat, const at::Tensor& idx,
                              const at::Tensor& w, const at::Tensor& coords,
                              const at::Tensor& n) {
        return ascend_kernel::spconv_devox(feat, idx, w, coords, n);
    });
}
