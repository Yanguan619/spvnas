/*
 * SpconvQuery 的 aclnn 直调层（与 spconv_nb / spconv_fused 同款 dlopen 结构）。
 * 本环境 int64 的 aclDataType 常量是 9。
 */

#include <cstdlib>
#include <dlfcn.h>
#include <string>

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#ifndef UNUM_VENDOR_DIR
#define UNUM_VENDOR_DIR "spconv_query"
#endif

namespace {

struct aclTensor;
using aclOpExecutor = ::aclOpExecutor;
typedef void* aclrtStream;

constexpr int kAclInt32 = 3;
constexpr int kAclInt64 = 9;
constexpr int kAclFormatNd = 2;

typedef int (*GetWorkspaceSizeFunc)(const aclTensor*, const aclTensor*, const aclTensor*,
                                    const aclTensor*, const aclTensor*, const aclTensor*,
                                    uint64_t*, aclOpExecutor**);
typedef int (*RunFunc)(void*, uint64_t, aclOpExecutor*, aclrtStream);
typedef aclTensor* (*CreateTensorFunc)(const int64_t*, uint64_t, int,
                                       const int64_t*, int64_t, int,
                                       const int64_t*, uint64_t, void*);

template <typename T>
T Dlsym(void* h, const char* n) { return reinterpret_cast<T>(dlsym(h, n)); }

std::string VendorLibPath()
{
    const char* opp = std::getenv("ASCEND_OPP_PATH");
    if (opp == nullptr) {
        opp = "/usr/local/Ascend/cann-9.2.1/opp";
    }
    return std::string(opp) + "/vendors/" UNUM_VENDOR_DIR "/op_api/lib/libcust_opapi.so";
}

struct Funcs {
    GetWorkspaceSizeFunc getWorkspaceSize = nullptr;
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
            f.getWorkspaceSize = Dlsym<GetWorkspaceSizeFunc>(cust,
                "aclnnSpconvQueryGetWorkspaceSize");
            f.run = Dlsym<RunFunc>(cust, "aclnnSpconvQuery");
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

aclTensor* Wrap(const at::Tensor& t, int dtype)
{
    auto& f = GetFuncs();
    const auto& sizes = t.sizes();
    const auto& strides = t.strides();
    int64_t storageDim = t.storage().nbytes() / t.itemsize();
    return f.createTensor(sizes.data(), sizes.size(), dtype, strides.data(),
                          t.storage_offset(), kAclFormatNd, &storageDim, 1,
                          const_cast<void*>(t.storage().data()));
}

}  // namespace

namespace ascend_kernel {

at::Tensor spconv_query(const at::Tensor& qcoords, const at::Tensor& rkeys,
                        const at::Tensor& start, const at::Tensor& cnt,
                        const at::Tensor& meta)
{
    auto& f = GetFuncs();
    TORCH_CHECK(f.getWorkspaceSize && f.run && f.createTensor && f.destroyTensor,
                "aclnnSpconvQuery symbols not found");

    at::Tensor qc = qcoords.to(at::kInt).contiguous();
    at::Tensor rk = rkeys.to(at::kLong).contiguous();
    at::Tensor sk = start.to(at::kInt).contiguous();
    at::Tensor ct = cnt.to(at::kInt).contiguous();
    at::Tensor mt = meta.to(at::kInt).contiguous();

    const int64_t M = qc.size(0);
    const int64_t kcnt = (mt.numel() - 12) / 3;
    const int64_t mPad = (M + 7) / 8 * 8;   // 与 kernel/host 的 8 行对齐一致
    at::Tensor outBuf = at::empty({mPad, kcnt}, qc.options());
    at::Tensor out = outBuf;

    aclTensor* qT = Wrap(qc, kAclInt32);
    aclTensor* rT = Wrap(rk, kAclInt64);
    aclTensor* sT = Wrap(sk, kAclInt32);
    aclTensor* cT = Wrap(ct, kAclInt32);
    aclTensor* mT = Wrap(mt, kAclInt32);
    aclTensor* oT = Wrap(out, kAclInt32);

    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    int st = f.getWorkspaceSize(qT, rT, sT, cT, mT, oT, &wsSize, &exec);
    TORCH_CHECK(st == 0, "aclnnSpconvQueryGetWorkspaceSize failed: ", st);

    void* wsAddr = nullptr;
    at::Tensor ws;
    if (wsSize > 0) {
        ws = at::empty({static_cast<int64_t>(wsSize)}, qc.options().dtype(at::kByte));
        wsAddr = const_cast<void*>(ws.storage().data());
    }
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnSpconvQuery");
    cmd.SetCustomHandler([f, wsAddr, wsSize, exec, stream, qT, rT, sT, cT, mT, oT]() -> int {
        int ret = f.run(wsAddr, wsSize, exec, stream);
        TORCH_CHECK(ret == 0, "aclnnSpconvQuery failed: ", ret);
        f.destroyTensor(qT); f.destroyTensor(rT); f.destroyTensor(sT);
        f.destroyTensor(cT); f.destroyTensor(mT); f.destroyTensor(oT);
        return ret;
    });
    cmd.Run();
    return outBuf.narrow(0, 0, M);   // 前缀切片，保持连续
}

}  // namespace ascend_kernel


// ---- torch 算子注册（原 register.cpp，合并以减少文件数）----
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
