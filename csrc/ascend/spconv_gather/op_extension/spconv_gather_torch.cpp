/*
 * Copyright (c) 2026 Yanguan02
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cstdlib>
#include <dlfcn.h>
#include <string>
#include <unistd.h>

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

// vendor aclnn 加载器：绝对路径 dlopen + RTLD_LOCAL + 符号来源校验（见头文件注释）
#include "unum_aclnn_loader.h"
#ifndef UNUM_VENDOR_DIR
#define UNUM_VENDOR_DIR "spconv_gather"
#endif

namespace {

struct aclTensor;
using aclOpExecutor = ::aclOpExecutor;
typedef void* aclrtStream;

constexpr int kAclFloat = 0;
constexpr int kAclInt32 = 3;
constexpr int kAclFormatNd = 2;

typedef int (*GetWorkspaceSizeFunc4)(const aclTensor*, const aclTensor*, const aclTensor*,
                                     const aclTensor*, uint64_t*, aclOpExecutor**);
typedef int (*RunFunc)(void*, uint64_t, aclOpExecutor*, aclrtStream);
typedef aclTensor* (*CreateTensorFunc)(const int64_t*, uint64_t, int,
                                       const int64_t*, int64_t, int,
                                       const int64_t*, uint64_t, void*);

struct Funcs {
    GetWorkspaceSizeFunc4 getWorkspaceSize = nullptr;
    RunFunc run = nullptr;
    CreateTensorFunc createTensor = nullptr;
    int (*destroyTensor)(const aclTensor*) = nullptr;
    std::string err;  // 最近一次解析失败原因（候选路径 / dlerror / 串味判定）
    bool loaded = false;
};

Funcs& GetFuncs()
{
    static Funcs f;
    if (!f.loaded) {
        const unum::VendorLib& lib =
            unum::OpenVendor(UNUM_VENDOR_DIR, reinterpret_cast<const void*>(&GetFuncs));
        f.getWorkspaceSize = unum::Resolve<GetWorkspaceSizeFunc4>(lib, "aclnnSpconvGatherGetWorkspaceSize", &f.err);
        f.run = unum::Resolve<RunFunc>(lib, "aclnnSpconvGather", &f.err);
        void* nnop = unum::OpenDep("libnnopbase.so");
        f.createTensor = unum::ResolveDep<CreateTensorFunc>(nnop, "aclCreateTensor", &f.err);
        f.destroyTensor = unum::ResolveDep<int (*)(const aclTensor*)>(nnop, "aclDestroyTensor", &f.err);
        f.loaded = (f.getWorkspaceSize != nullptr && f.run != nullptr &&
                     f.createTensor != nullptr && f.destroyTensor != nullptr);
    }
    return f;
}

aclTensor* WrapTensor(const at::Tensor& t, int acl_dtype)
{
    auto& f = GetFuncs();
    const auto& sizes = t.sizes();
    const auto& strides = t.strides();
    int64_t storageDim = t.storage().nbytes() / t.itemsize();
    return f.createTensor(sizes.data(), sizes.size(), acl_dtype,
                          strides.data(), t.storage_offset(), kAclFormatNd,
                          &storageDim, 1, const_cast<void*>(t.storage().data()));
}

}  // namespace

namespace ascend_kernel {

at::Tensor spconv_gather(const at::Tensor& feat, const at::Tensor& nb,
                         const at::Tensor& params)
{
    auto& f = GetFuncs();
    TORCH_CHECK(f.getWorkspaceSize && f.run && f.createTensor && f.destroyTensor,
                "aclnnSpconvGather symbols not found (vendor not installed / env)", f.err);

    TORCH_CHECK(feat.dim() == 2, "feat must be (N, cin)");
    TORCH_CHECK(nb.dim() == 2, "nb must be (M, K)");
    TORCH_CHECK(params.numel() >= 3, "params must be [M, cin, kcnt]");
    TORCH_CHECK(feat.scalar_type() == at::kFloat, "feat must be float32");
    TORCH_CHECK(nb.scalar_type() == at::kInt, "nb must be int32");
    TORCH_CHECK(params.scalar_type() == at::kInt, "params must be int32");
    TORCH_CHECK(feat.is_contiguous() && nb.is_contiguous() && params.is_contiguous(),
                "all inputs must be contiguous");

    const int64_t M = nb.size(0);
    const int64_t K = nb.size(1);
    const int64_t cin = feat.size(1);

    const int64_t mPad = (M + 7) / 8 * 8;   // 与 kernel/host 的 8 行对齐一致
    at::Tensor outBuf = at::empty({mPad, K * cin}, feat.options());

    aclTensor* featT = WrapTensor(feat, kAclFloat);
    aclTensor* nbT = WrapTensor(nb, kAclInt32);
    aclTensor* paramsT = WrapTensor(params, kAclInt32);
    aclTensor* outT = WrapTensor(outBuf, kAclFloat);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int st = f.getWorkspaceSize(featT, nbT, paramsT, outT, &workspaceSize, &executor);
    TORCH_CHECK(st == 0, "aclnnSpconvGatherGetWorkspaceSize failed: ", st);

    void* wsAddr = nullptr;
    at::Tensor wsTensor;
    if (workspaceSize > 0) {
        wsTensor = at::empty({static_cast<int64_t>(workspaceSize)},
                             feat.options().dtype(at::kByte));
        wsAddr = const_cast<void*>(wsTensor.storage().data());
    }

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnSpconvGather");
    cmd.SetCustomHandler([f, wsAddr, workspaceSize, executor, aclStream,
                          featT, nbT, paramsT, outT]() -> int {
        int ret = f.run(wsAddr, workspaceSize, executor, aclStream);
        TORCH_CHECK(ret == 0, "aclnnSpconvGather failed: ", ret);
        f.destroyTensor(featT);
        f.destroyTensor(nbT);
        f.destroyTensor(paramsT);
        f.destroyTensor(outT);
        return ret;
    });
    cmd.Run();
    return outBuf.narrow(0, 0, M);   // 前缀切片，保持连续
}

}  // namespace ascend_kernel


// ---- torch 算子注册（原 register.cpp，合并以减少文件数）----
/*
 * Copyright (c) 2026 Yanguan02
 * SPDX-License-Identifier: BSD-3-Clause
 */


TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("spconv_gather(Tensor feat, Tensor nb, Tensor params) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("spconv_gather", [](const at::Tensor& feat,
                               const at::Tensor& nb,
                               const at::Tensor& params) {
        return ascend_kernel::spconv_gather(feat, nb, params);
    });
}
