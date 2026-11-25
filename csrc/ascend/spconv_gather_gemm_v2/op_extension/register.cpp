/*
 * Copyright (c) 2026 Yanguan02
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <torch/extension.h>
#include "spconv_gather_gemm_v2_ops.h"

TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("spconv_gather_gemm_v2(Tensor feat, Tensor nb, Tensor w, Tensor n) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("spconv_gather_gemm_v2", [](const at::Tensor& feat, const at::Tensor& nb,
                                    const at::Tensor& w, const at::Tensor& n) {
        return ascend_kernel::spconv_gather_gemm_v2(feat, nb, w, n);
    });
}
