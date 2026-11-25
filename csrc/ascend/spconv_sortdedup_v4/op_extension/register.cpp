#include <torch/extension.h>
#include "spconv_sortdedup_v4_ops.h"
TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("spconv_sortdedup_v4(Tensor coords, Tensor n) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("spconv_sortdedup_v4", [](const at::Tensor& coords, const at::Tensor& n) {
        return ascend_kernel::spconv_sortdedup_v4(coords, n);
    });
}
