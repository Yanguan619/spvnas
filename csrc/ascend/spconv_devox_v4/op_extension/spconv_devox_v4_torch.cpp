#include <cstdlib>
#include <dlfcn.h>
#include <string>

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#ifndef UNUM_VENDOR_DIR
#define UNUM_VENDOR_DIR "spconv_devox"
#endif

namespace {

struct aclTensor;
using aclOpExecutor = ::aclOpExecutor;
typedef void* aclrtStream;

constexpr int kAclFloat = 0;
constexpr int kAclInt32 = 3;
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
                "aclnnSpconvDevoxV4GetWorkspaceSize");
            f.run = Dlsym<RunFunc>(cust, "aclnnSpconvDevoxV4");
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

at::Tensor spconv_devox(const at::Tensor& feat, const at::Tensor& idx,
                        const at::Tensor& w, const at::Tensor& coords,
                        const at::Tensor& nrow)
{
    auto& f = GetFuncs();
    TORCH_CHECK(f.getWorkspaceSize && f.run && f.createTensor && f.destroyTensor,
                "aclnnSpconvDevoxV4 symbols not found");

    TORCH_CHECK(nrow.scalar_type() == at::kInt && nrow.numel() == 1,
                "n must be a 1-element int32 tensor (真实点数)");
    at::Tensor ft = feat.contiguous();
    at::Tensor nt = nrow.to(at::kInt).contiguous();
    at::Tensor ix = idx.to(at::kInt).contiguous();
    at::Tensor wt = w.contiguous();
    at::Tensor cd = coords.contiguous();

    const int64_t N = ix.size(1);
    const int64_t c = ft.size(1);
    at::Tensor out = at::empty({N, c}, ft.options());

    aclTensor* fT = Wrap(ft, kAclFloat);
    aclTensor* iT = Wrap(ix, kAclInt32);
    aclTensor* wT = Wrap(wt, kAclFloat);
    aclTensor* cT = Wrap(cd, kAclFloat);
    aclTensor* nT = Wrap(nt, kAclInt32);
    aclTensor* oT = Wrap(out, kAclFloat);

    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    int st = f.getWorkspaceSize(fT, iT, wT, cT, nT, oT, &wsSize, &exec);
    TORCH_CHECK(st == 0, "aclnnSpconvDevoxV4GetWorkspaceSize failed: ", st);

    void* wsAddr = nullptr;
    at::Tensor ws;
    if (wsSize > 0) {
        ws = at::empty({static_cast<int64_t>(wsSize)}, ft.options().dtype(at::kByte));
        wsAddr = const_cast<void*>(ws.storage().data());
    }
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnSpconvDevoxV4");
    cmd.SetCustomHandler([f, wsAddr, wsSize, exec, stream, fT, iT, wT, cT, nT, oT]() -> int {
        int ret = f.run(wsAddr, wsSize, exec, stream);
        TORCH_CHECK(ret == 0, "aclnnSpconvDevoxV4 failed: ", ret);
        f.destroyTensor(fT); f.destroyTensor(iT); f.destroyTensor(wT);
        f.destroyTensor(cT); f.destroyTensor(nT); f.destroyTensor(oT);
        return ret;
    });
    cmd.Run();
    return out;
}

}  // namespace ascend_kernel


// ---- torch 算子注册（原 register.cpp，合并以减少文件数）----
TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("spconv_devox_v4(Tensor feat, Tensor idx, Tensor w, Tensor coords, Tensor n) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("spconv_devox_v4", [](const at::Tensor& feat, const at::Tensor& idx,
                              const at::Tensor& w, const at::Tensor& coords,
                              const at::Tensor& n) {
        return ascend_kernel::spconv_devox(feat, idx, w, coords, n);
    });
}
