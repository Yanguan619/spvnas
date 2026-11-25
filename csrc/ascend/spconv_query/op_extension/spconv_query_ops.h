#ifndef SPCONV_QUERY_OPS_H
#define SPCONV_QUERY_OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

at::Tensor spconv_query(const at::Tensor& qcoords, const at::Tensor& rkeys,
                        const at::Tensor& start, const at::Tensor& cnt,
                        const at::Tensor& meta);

}  // namespace ascend_kernel

#endif  // SPCONV_QUERY_OPS_H
