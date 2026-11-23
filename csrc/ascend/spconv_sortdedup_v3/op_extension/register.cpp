#include <torch/extension.h>
#include "spconv_sortdedup_v3_ops.h"
TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("spconv_sortdedup_v3(Tensor coords) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("spconv_sortdedup_v3", [](const at::Tensor& coords) {
        return ascend_kernel::spconv_sortdedup_v3(coords);
    });
}
