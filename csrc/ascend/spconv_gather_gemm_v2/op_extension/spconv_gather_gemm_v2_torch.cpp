/*
 * Copyright (c) 2026 Yanguan02
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdlib>
#include <dlfcn.h>
#include <string>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "spconv_gather_gemm_v2_ops.h"

#ifndef UNUM_VENDOR_DIR
#define UNUM_VENDOR_DIR "spconv_gather_gemm_v2"
#endif

namespace {
struct aclTensor;
using aclOpExecutor = ::aclOpExecutor;
typedef void* aclrtStream;
constexpr int kAclFloat = 0;
constexpr int kAclInt32 = 3;
constexpr int kAclFormatNd = 2;

typedef int (*GetWorkspaceSizeFunc4)(const aclTensor*, const aclTensor*, const aclTensor*,
                                     const aclTensor*, const aclTensor*,
                                     uint64_t*, aclOpExecutor**);
typedef int (*RunFunc)(void*, uint64_t, aclOpExecutor*, aclrtStream);
typedef aclTensor* (*CreateTensorFunc)(const int64_t*, uint64_t, int,
                                       const int64_t*, int64_t, int,
                                       const int64_t*, uint64_t, void*);

template <typename T>
T Dlsym(void* handle, const char* name)
{
    return reinterpret_cast<T>(dlsym(handle, name));
}

std::string VendorLibPath()
{
    const char* opp = std::getenv("ASCEND_OPP_PATH");
    if (opp == nullptr) {
        opp = "/usr/local/Ascend/cann-9.2.1/opp";
    }
    return std::string(opp) + "/vendors/" UNUM_VENDOR_DIR "/op_api/lib/libcust_opapi.so";
}

struct Funcs {
    GetWorkspaceSizeFunc4 getWorkspaceSize = nullptr;
    RunFunc run = nullptr;
    CreateTensorFunc createTensor = nullptr;
    int (*destroyTensor)(const aclTensor*) = nullptr;
    bool loaded = false;
};

Funcs& GetFuncs()
{
    static Funcs f;
    if (!f.loaded) {
        void* cust = dlopen(VendorLibPath().c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (cust != nullptr) {
            f.getWorkspaceSize = Dlsym<GetWorkspaceSizeFunc4>(
                cust, "aclnnSpconvGatherGemmV2GetWorkspaceSize");
            f.run = Dlsym<RunFunc>(cust, "aclnnSpconvGatherGemmV2");
        }
        void* nnop = dlopen("libnnopbase.so", RTLD_LAZY | RTLD_LOCAL);
        if (nnop != nullptr) {
            f.createTensor = Dlsym<CreateTensorFunc>(nnop, "aclCreateTensor");
            f.destroyTensor = Dlsym<int (*)(const aclTensor*)>(nnop, "aclDestroyTensor");
        }
        f.loaded = true;
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
at::Tensor spconv_gather_gemm_v2(const at::Tensor& feat, const at::Tensor& nb,
                                const at::Tensor& w, const at::Tensor& nrow)
{
    auto& f = GetFuncs();
    TORCH_CHECK(f.getWorkspaceSize && f.run && f.createTensor && f.destroyTensor,
                "aclnnSpconvGatherGemmV2 symbols not found (vendor not installed / env)");
    TORCH_CHECK(feat.dim() == 2 && nb.dim() == 2 && w.dim() == 2, "feat/nb/w must be 2D");
    TORCH_CHECK(feat.scalar_type() == at::kFloat, "feat must be float32");
    TORCH_CHECK(nb.scalar_type() == at::kInt, "nb must be int32");
    TORCH_CHECK(w.scalar_type() == at::kFloat, "w must be float32");
    TORCH_CHECK(feat.is_contiguous() && nb.is_contiguous() && w.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(nrow.scalar_type() == at::kInt && nrow.numel() == 1,
                "n must be a 1-element int32 tensor (真实行数)");
    const int64_t M = nb.size(0);   // 容量；真实行数由 n 决定
    const int64_t K = nb.size(1) * feat.size(1);
    TORCH_CHECK(w.size(0) == K, "w must be (kv*cin, cout)");
    const int64_t cout = w.size(1);

    at::Tensor out = at::empty({M, cout}, feat.options());
    aclTensor* featT = WrapTensor(feat, kAclFloat);
    aclTensor* nbT = WrapTensor(nb, kAclInt32);
    aclTensor* wT = WrapTensor(w, kAclFloat);
    aclTensor* nT = WrapTensor(nrow, kAclInt32);
    aclTensor* outT = WrapTensor(out, kAclFloat);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int st = f.getWorkspaceSize(featT, nbT, wT, nT, outT, &workspaceSize, &executor);
    TORCH_CHECK(st == 0, "aclnnSpconvGatherGemmV2GetWorkspaceSize failed: ", st);
    void* wsAddr = nullptr;
    at::Tensor wsTensor;
    if (workspaceSize > 0) {
        wsTensor = at::empty({static_cast<int64_t>(workspaceSize)},
                             feat.options().dtype(at::kByte));
        wsAddr = const_cast<void*>(wsTensor.storage().data());
    }
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnSpconvGatherGemmV2");
    cmd.SetCustomHandler([f, wsAddr, workspaceSize, executor, aclStream,
                          featT, nbT, wT, nT, outT]() -> int {
        int ret = f.run(wsAddr, workspaceSize, executor, aclStream);
        TORCH_CHECK(ret == 0, "aclnnSpconvGatherGemmV2 failed: ", ret);
        f.destroyTensor(featT);
        f.destroyTensor(nbT);
        f.destroyTensor(wT);
        f.destroyTensor(nT);
        f.destroyTensor(outT);
        return ret;
    });
    cmd.Run();
    return out;
}
}  // namespace ascend_kernel
