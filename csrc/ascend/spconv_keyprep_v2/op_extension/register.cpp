#include <torch/extension.h>
#include "spconv_keyprep_v2_ops.h"

TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("spconv_keyprep_v2(Tensor in_coords, Tensor out_coords, Tensor stride, Tensor offs, "
          "int nbk) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("spconv_keyprep_v2", [](const at::Tensor& in_coords, const at::Tensor& out_coords,
                                   const at::Tensor& stride, const at::Tensor& offs, int64_t nbk) {
        return ascend_kernel::spconv_keyprep_v2(in_coords, out_coords, stride, offs, nbk);
    });
}
