/*
 * Copyright (c) 2026 Yanguan02
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef SPCONV_GATHER_GEMM_OPS_H
#define SPCONV_GATHER_GEMM_OPS_H
#include <torch/extension.h>
namespace ascend_kernel {
at::Tensor spconv_gather_gemm(const at::Tensor& feat, const at::Tensor& nb, const at::Tensor& w);
}  // namespace ascend_kernel
#endif  // SPCONV_GATHER_GEMM_OPS_H
