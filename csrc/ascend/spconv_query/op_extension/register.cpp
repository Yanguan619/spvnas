#include <torch/extension.h>
#include "spconv_query_ops.h"

TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("spconv_query(Tensor qcoords, Tensor rkeys, Tensor start, Tensor cnt, "
          "Tensor meta) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("spconv_query", [](const at::Tensor& qcoords, const at::Tensor& rkeys,
                              const at::Tensor& start, const at::Tensor& cnt,
                              const at::Tensor& meta) {
        return ascend_kernel::spconv_query(qcoords, rkeys, start, cnt, meta);
    });
}
