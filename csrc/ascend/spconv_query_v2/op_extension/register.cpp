#include <torch/extension.h>
#include "spconv_query_v2_ops.h"

TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("spconv_query_v2(Tensor qcoords, Tensor rkeys, Tensor start, Tensor cnt, "
          "Tensor meta, Tensor n) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("spconv_query_v2", [](const at::Tensor& qcoords, const at::Tensor& rkeys,
                              const at::Tensor& start, const at::Tensor& cnt,
                              const at::Tensor& meta, const at::Tensor& n) {
        return ascend_kernel::spconv_query_v2(qcoords, rkeys, start, cnt, meta, n);
    });
}
