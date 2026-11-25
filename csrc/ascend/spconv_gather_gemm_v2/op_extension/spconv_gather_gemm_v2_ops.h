/*
 * Copyright (c) 2026 Yanguan02
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef SPCONV_GATHER_GEMM_V2_OPS_H
#define SPCONV_GATHER_GEMM_V2_OPS_H
#include <torch/extension.h>
namespace ascend_kernel {
at::Tensor spconv_gather_gemm_v2(const at::Tensor& feat, const at::Tensor& nb,
                                const at::Tensor& w, const at::Tensor& nrow);
}  // namespace ascend_kernel
#endif  // SPCONV_GATHER_GEMM_V2_OPS_H
