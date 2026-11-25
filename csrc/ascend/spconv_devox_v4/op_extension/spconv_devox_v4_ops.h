#ifndef SPCONV_DEVOX_OPS_H
#define SPCONV_DEVOX_OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

at::Tensor spconv_devox(const at::Tensor& feat, const at::Tensor& idx,
                        const at::Tensor& w, const at::Tensor& coords,
                        const at::Tensor& nrow);

}  // namespace ascend_kernel

#endif  // SPCONV_DEVOX_OPS_H
