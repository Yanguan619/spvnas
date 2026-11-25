#ifndef SPCONV_KEYPREP_V3_OPS_H
#define SPCONV_KEYPREP_V3_OPS_H
#include <torch/extension.h>
#include <vector>
namespace ascend_kernel {
std::vector<at::Tensor> spconv_keyprep_v3(const at::Tensor& in_coords, const at::Tensor& out_coords,
                                          const at::Tensor& stride, const at::Tensor& offs,
                                          int64_t nbk, const at::Tensor& sizes);
}  // namespace ascend_kernel
#endif  // SPCONV_KEYPREP_V3_OPS_H
