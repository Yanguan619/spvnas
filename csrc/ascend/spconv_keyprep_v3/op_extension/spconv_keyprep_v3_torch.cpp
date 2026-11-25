/*
 * SpconvKeyprepV3 的 aclnn 直调层（与 spconv_query / spconv_nb 同款 dlopen 结构）。
 *
 * 两段 launch：
 *   1) aclnnSpconvKeyprepV3        box + rkeys + 直方图 + 每核桶段和
 *   2) aclnnSpconvKeyprepV3Prefix  cnt -> start 独占前缀和
 * 拆两段是因为原子加出来的 cnt 在同一个 kernel 里隔屏障读回来会少读几笔。
 *
 * nbk 以 device 张量形式进 kernel（kernel 自己读，host 不需要它的值），输出长度
 * 由调用方给出的 nbk 决定（默认按 _query_args 同一条公式从 N 算）。nbk 小张量按值
 * 缓存，避免每次 H2D —— 整条路径上没有 .item() / stream 同步。
 * 本环境 int64 的 aclDataType 常量是 9。
 */
#include <cstdlib>
#include <dlfcn.h>
#include <map>
#include <string>
#include <vector>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "spconv_keyprep_v3_ops.h"

#ifndef UNUM_VENDOR_DIR
#define UNUM_VENDOR_DIR "spconv_keyprep_v3"
#endif

namespace {
struct aclTensor;
using aclOpExecutor = ::aclOpExecutor;
typedef void* aclrtStream;
constexpr int kAclInt32 = 3;
constexpr int kAclInt64 = 9;
constexpr int kAclFormatNd = 2;
/* box/rkeys/hist 段：5 输入 + 4 输出 */
typedef int (*GetWsMainFunc)(const aclTensor*, const aclTensor*, const aclTensor*,
                             const aclTensor*, const aclTensor*, const aclTensor*,
                             const aclTensor*, const aclTensor*, const aclTensor*,
                             const aclTensor*, uint64_t*, aclOpExecutor**);
/* prefix 段：2 输入 + 1 输出 */
typedef int (*GetWsPrefixFunc)(const aclTensor*, const aclTensor*, uint64_t*, aclOpExecutor**);
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
    GetWsMainFunc wsMain = nullptr;
    RunFunc runMain = nullptr;
    GetWsPrefixFunc wsPrefix = nullptr;
    RunFunc runPrefix = nullptr;
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
            f.wsMain = Dlsym<GetWsMainFunc>(cust, "aclnnSpconvKeyprepV3GetWorkspaceSize");
            f.runMain = Dlsym<RunFunc>(cust, "aclnnSpconvKeyprepV3");
            f.wsPrefix = Dlsym<GetWsPrefixFunc>(cust, "aclnnSpconvKeyprepV3PrefixGetWorkspaceSize");
            f.runPrefix = Dlsym<RunFunc>(cust, "aclnnSpconvKeyprepV3Prefix");
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
/* nbk 张量按值缓存：同一进程内每层一个，避免每次调用一次 H2D */
at::Tensor NbkT(int64_t nbk, const at::TensorOptions& opts)
{
    static std::map<int64_t, at::Tensor> cache;
    auto it = cache.find(nbk);
    if (it != cache.end()) {
        return it->second;
    }
    if (cache.size() > 64) {
        cache.clear();
    }
    at::Tensor t = at::empty({1}, opts.dtype(at::kLong));
    t.fill_(nbk);
    cache.emplace(nbk, t);
    return t;
}
/* 一次 aclnn 调用的收尾：workspace 从 GetWorkspaceSize 拿，用 OpCommand 挂到当前 stream */
struct LaunchArgs {
    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    std::vector<aclTensor*> tensors;
};
void Dispatch(const char* name, RunFunc run, LaunchArgs& la, const at::Tensor& opt)
{
    auto& f = GetFuncs();
    void* wsAddr = nullptr;
    at::Tensor ws;
    if (la.wsSize > 0) {
        ws = at::empty({static_cast<int64_t>(la.wsSize)}, opt.options().dtype(at::kByte));
        wsAddr = const_cast<void*>(ws.storage().data());
    }
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    void* wsKeep = ws.defined() ? wsAddr : nullptr;
    at_npu::native::OpCommand cmd;
    cmd.Name(name);
    std::vector<aclTensor*> ts = la.tensors;
    cmd.SetCustomHandler([f, run, wsKeep, la, stream, ts]() -> int {
        int ret = run(wsKeep, la.wsSize, la.exec, stream);
        for (auto* t : ts) {
            f.destroyTensor(t);
        }
        TORCH_CHECK(ret == 0, "aclnn ", ret);
        return ret;
    });
    cmd.Run();
}
}  // namespace

namespace ascend_kernel {
std::vector<at::Tensor> spconv_keyprep_v3(const at::Tensor& in_coords, const at::Tensor& out_coords,
                                          const at::Tensor& stride, const at::Tensor& offs, int64_t nbk,
                                          const at::Tensor& sizes)
{
    auto& f = GetFuncs();
    TORCH_CHECK(f.wsMain && f.runMain && f.wsPrefix && f.runPrefix &&
                f.createTensor && f.destroyTensor, "aclnnSpconvKeyprepV3 symbols not found");
    at::Tensor ic = in_coords.to(at::kInt).contiguous();
    at::Tensor oc = out_coords.to(at::kInt).contiguous();
    at::Tensor st = stride.to(at::kInt).contiguous();
    at::Tensor of = offs.to(at::kInt).contiguous();
    const int64_t N = ic.dim() > 0 ? ic.size(0) : 0;
    const int64_t kv = of.dim() > 0 ? of.size(0) : 0;
    if (nbk <= 0) {   /* 与 backend._query_args 同一条公式 */
        int64_t mul = 4;
        const char* m = std::getenv("TS_NBK_MUL");
        if (m != nullptr && std::atoi(m) > 0) {
            mul = std::atoi(m);
        }
        nbk = 1;
        while (nbk < mul * N) {
            nbk <<= 1;
        }
        if (nbk > (static_cast<int64_t>(1) << 20)) {
            nbk = static_cast<int64_t>(1) << 20;
        }
    }
    at::Tensor rkeys = at::empty({N}, ic.options().dtype(at::kLong));
    at::Tensor start = at::empty({nbk}, ic.options());
    at::Tensor cnt = at::empty({nbk}, ic.options());   /* kernel 自己清 */
    at::Tensor meta = at::empty({12 + 3 * kv}, ic.options());
    at::Tensor scratch = at::zeros({4096}, ic.options());  /* 跨核交换 + 屏障格子 */
    at::Tensor nk = NbkT(nbk, ic.options());
    TORCH_CHECK(sizes.scalar_type() == at::kInt && sizes.numel() == 2,
                "sizes must be int32[2] = [n, m] (真实 in/out 行数)");
    at::Tensor sz = sizes.to(at::kInt).contiguous();

    aclTensor* inT = Wrap(ic, kAclInt32);
    aclTensor* ocT = Wrap(oc, kAclInt32);
    aclTensor* stT = Wrap(st, kAclInt32);
    aclTensor* ofT = Wrap(of, kAclInt32);
    aclTensor* nkT = Wrap(nk, kAclInt64);
    aclTensor* szT = Wrap(sz, kAclInt32);
    aclTensor* rkT = Wrap(rkeys, kAclInt64);
    aclTensor* cnT = Wrap(cnt, kAclInt32);
    aclTensor* mtT = Wrap(meta, kAclInt32);
    aclTensor* scT = Wrap(scratch, kAclInt32);
    LaunchArgs la1;
    // 入参顺序 = OpDef：输入(in,out,stride,offs,nbk,sizes) 再输出(rkeys,cnt,meta,scratch)
    int rc = f.wsMain(inT, ocT, stT, ofT, nkT, szT, rkT, cnT, mtT, scT,
                      &la1.wsSize, &la1.exec);
    TORCH_CHECK(rc == 0, "aclnnSpconvKeyprepV3GetWorkspaceSize failed: ", rc);
    la1.tensors = {inT, ocT, stT, ofT, nkT, szT, rkT, cnT, mtT, scT};
    Dispatch("aclnnSpconvKeyprepV3", f.runMain, la1, ic);

    aclTensor* saT = Wrap(start, kAclInt32);
    aclTensor* cn2T = Wrap(cnt, kAclInt32);
    LaunchArgs la2;
    rc = f.wsPrefix(cn2T, saT, &la2.wsSize, &la2.exec);
    TORCH_CHECK(rc == 0, "aclnnSpconvKeyprepV3PrefixGetWorkspaceSize failed: ", rc);
    la2.tensors = {cn2T, saT};
    Dispatch("aclnnSpconvKeyprepV3Prefix", f.runPrefix, la2, ic);
    return {rkeys, start, cnt, meta};
}
}  // namespace ascend_kernel


