/* SpconvSortDedupV4 的 aclnn 直调层：一次 launch，host 只做三件事——
 * 分配 res / u / keys / scratch，挂 kernel，读回 u（唯一一次标量同步，32B）。
 *
 * keys / scratch 是 kernel 的中间缓冲，但在这条直调路径上 kernel 的 workspace
 * 参数不可写（keyprep 踩过），所以走「输出张量」由 host 分配、kernel 直接读写。
 *
 * u 的 4 个 int64 一起读回：u[0]=U，u[1]=失败标记（跨核屏障自旋超时），
 * u[2]=core0 观测到的最大自旋次数（诊断）。u[1] != 0 时抛错，调用方
 * （torchsparse.backend 的 try/except）回退到 torch 路径——绝不把没到齐的结果当结果。 */
#include <cstdlib>
#include <dlfcn.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "spconv_sortdedup_v4_ops.h"
#ifndef UNUM_VENDOR_DIR
#define UNUM_VENDOR_DIR "spconv_sortdedup_v4"
#endif
namespace {
struct aclTensor;
using aclOpExecutor = ::aclOpExecutor;
typedef void* aclrtStream;
constexpr int kAclInt32 = 3;
constexpr int kAclInt64 = 9;
constexpr int kAclFormatNd = 2;
/* scratch 元素数，必须与 op_kernel/spconv_sortdedup_v4_tiling.h 一致 */
constexpr int64_t kScratchInts = 27712;
typedef int (*GetWorkspaceSizeFunc)(const aclTensor*, const aclTensor*, const aclTensor*,
                                    const aclTensor*, const aclTensor*, const aclTensor*, uint64_t*,
                                    aclOpExecutor**);
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
                "aclnnSpconvSortDedupV4GetWorkspaceSize");
            f.run = Dlsym<RunFunc>(cust, "aclnnSpconvSortDedupV4");
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
/* 屏障超时是芯片上 AIV 被别的 kernel 占住的产物，同进程后续调用一样会踩到。
 * 失败一次就整进程停用（每次调用只付一次有界自旋，而不是每帧 5 次）。
 * UNUM_SD_STICKY=0 可以让它每次都重试。 */
bool Sticky()
{
    static const bool s = [] {
        const char* e = std::getenv("UNUM_SD_STICKY");
        return !(e != nullptr && std::atoi(e) == 0);
    }();
    return s;
}
bool& Disabled()
{
    static bool d = false;
    return d;
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
std::vector<at::Tensor> spconv_sortdedup_v4(const at::Tensor& coords,
                                            const at::Tensor& nrow)
{
    if (Disabled()) {
        TORCH_CHECK(false, "SpconvSortDedupV4: disabled in this process after a barrier timeout");
    }
    auto& f = GetFuncs();
    TORCH_CHECK(f.getWorkspaceSize && f.run && f.createTensor && f.destroyTensor,
                "aclnnSpconvSortDedupV4 symbols not found");
    TORCH_CHECK(nrow.scalar_type() == at::kInt && nrow.numel() == 1,
                "n must be a 1-element int32 tensor (真实行数)");
    at::Tensor nt = nrow.to(at::kInt).contiguous();
    at::Tensor cd = coords.to(at::kInt).contiguous();
    const int64_t N = (cd.dim() > 0) ? cd.size(0) : 0;
    if (N <= 0) {
        return {at::empty({0, 4}, cd.options()),
                at::zeros({1}, cd.options().dtype(at::kLong))};
    }
    at::Tensor res = at::empty({N, 8}, cd.options());
    at::Tensor u = at::empty({4}, cd.options().dtype(at::kLong));   // 32B 对齐的写回宽度
    at::Tensor rows = at::empty({2 * N * 8}, cd.options());
    at::Tensor scratch = at::empty({kScratchInts}, cd.options());
    aclTensor* cT = Wrap(cd, kAclInt32);
    aclTensor* rT = Wrap(res, kAclInt32);
    aclTensor* uT = Wrap(u, kAclInt64);
    aclTensor* kT = Wrap(rows, kAclInt32);
    aclTensor* sT = Wrap(scratch, kAclInt32);
    aclTensor* nT = Wrap(nt, kAclInt32);
    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    // aclnn 入参顺序 = OpDef 顺序：先输入（coords, n）再输出（res, u, rows, scratch）
    int st = f.getWorkspaceSize(cT, nT, rT, uT, kT, sT, &wsSize, &exec);
    TORCH_CHECK(st == 0, "aclnnSpconvSortDedupV4GetWorkspaceSize failed: ", st);
    void* wsAddr = nullptr;
    at::Tensor ws;
    if (wsSize > 0) {
        ws = at::empty({static_cast<int64_t>(wsSize)}, cd.options().dtype(at::kByte));
        wsAddr = const_cast<void*>(ws.storage().data());
    }
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnSpconvSortDedupV4");
    cmd.SetCustomHandler([f, wsAddr, wsSize, exec, stream, cT, rT, uT, kT, sT, nT]() -> int {
        int ret = f.run(wsAddr, wsSize, exec, stream);
        TORCH_CHECK(ret == 0, "aclnnSpconvSortDedupV4 failed: ", ret);
        f.destroyTensor(cT); f.destroyTensor(rT); f.destroyTensor(uT);
        f.destroyTensor(kT); f.destroyTensor(sT); f.destroyTensor(nT);
        return ret;
    });
    cmd.Run();
    // v4 默认**完全不回读 host**：u[0]=U、u[1]=屏障标记都留在 device，由下游算子读。
    // 这样 capture 期没有同步 D2H，也不需要 _fgstatic 把 U 冻成常量。
    // 需要 eager 诊断（屏障超时排查）时开 UNUM_SD_CHECK=1。
    if (std::getenv("UNUM_SD_CHECK") != nullptr) {
        at::Tensor uh = u.to(at::kCPU);
        const int64_t flag = uh[1].item<int64_t>();
        if (flag != 0) {
            if (Sticky()) {
                Disabled() = true;
            }
            TORCH_CHECK(false, "SpconvSortDedupV4: cross-core barrier timed out "
                        "(AIV contention), max_spin=", uh[2].item<int64_t>());
        }
        const int64_t nOut = uh[0].item<int64_t>();
        TORCH_CHECK(nOut >= 0 && nOut <= N, "SpconvSortDedupV4: bad U=", nOut, " N=", N);
        return {res.narrow(0, 0, nOut).narrow(1, 0, 4).contiguous(), u.narrow(0, 0, 1)};
    }
    return {res.narrow(1, 0, 4).contiguous(), u.narrow(0, 0, 1)};
}
}  // namespace ascend_kernel
