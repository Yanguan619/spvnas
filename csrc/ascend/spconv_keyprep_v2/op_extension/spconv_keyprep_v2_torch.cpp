/* SpconvKeyprepV2 的 aclnn 直调层（与 V1 同款 dlopen 结构）。
 *
 * 两段 launch：
 *   1) aclnnSpconvKeyprepV2        box + rkeys + 直方图 + 每核桶段和 + u（失败标记）
 *   2) aclnnSpconvKeyprepV2Prefix  cnt -> start 独占前缀和
 * 拆两段是因为原子加出来的 cnt 在同一个 kernel 里隔屏障读回来会少读几笔。
 *
 * 与 V1 的唯一接口差别：返回 5 个张量，最后一个是 u = int64[4]
 *   u[0] = 0 正常 / -1 跨核屏障自旋超时     u[1] = core0 观测到的最大自旋次数
 * 本层不做 device->host 同步（V1 的整条路径上没有 .item()，这里也不加），
 * 失败检查在 Python 侧做一次 `int(u[0])`，和 `total_ = int(torch.prod(...))`
 * 共用同一次 stream 落地，成本可忽略。
 *
 * nbk 以 device 张量形式进 kernel（kernel 自己读，host 不需要它的值），输出长度
 * 由调用方给出的 nbk 决定（默认按 _query_args 同一条公式从 N 算）。nbk 小张量按值
 * 缓存，避免每次 H2D。本环境 int64 的 aclDataType 常量是 9。
 */
#include <cstdlib>
#include <dlfcn.h>
#include <map>
#include <string>
#include <vector>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "spconv_keyprep_v2_ops.h"

#ifndef UNUM_VENDOR_DIR
#define UNUM_VENDOR_DIR "spconv_keyprep_v2"
#endif

namespace {
struct aclTensor;
using aclOpExecutor = ::aclOpExecutor;
typedef void* aclrtStream;
constexpr int kAclInt32 = 3;
constexpr int kAclInt64 = 9;
constexpr int kAclFormatNd = 2;
/* box/rkeys/hist 段：5 输入 + 5 输出 */
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
            f.wsMain = Dlsym<GetWsMainFunc>(cust, "aclnnSpconvKeyprepV2GetWorkspaceSize");
            f.runMain = Dlsym<RunFunc>(cust, "aclnnSpconvKeyprepV2");
            f.wsPrefix = Dlsym<GetWsPrefixFunc>(cust, "aclnnSpconvKeyprepV2PrefixGetWorkspaceSize");
            f.runPrefix = Dlsym<RunFunc>(cust, "aclnnSpconvKeyprepV2Prefix");
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
/* 32B 状态张量按 device 缓存：避免每次调用一次 at::empty（实测 0.015 ms/次，
 * 在 N=9528 那档调用上就是 4%） */
at::Tensor UT(const at::TensorOptions& opts)
{
    static std::map<int, at::Tensor> cache;
    const int d = static_cast<int>(opts.device().index());
    auto it = cache.find(d);
    if (it == cache.end()) {
        it = cache.emplace(d, at::empty({4}, opts.dtype(at::kLong))).first;
    }
    return it->second;
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
std::vector<at::Tensor> spconv_keyprep_v2(const at::Tensor& in_coords, const at::Tensor& out_coords,
                                          const at::Tensor& stride, const at::Tensor& offs,
                                          int64_t nbk)
{
    auto& f = GetFuncs();
    TORCH_CHECK(f.wsMain && f.runMain && f.wsPrefix && f.runPrefix &&
                f.createTensor && f.destroyTensor, "aclnnSpconvKeyprepV2 symbols not found");
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
    /* u: [失败标记, core0 最大自旋, 0, 0]。32B 状态张量，整进程复用同一个
     * （每次 at::empty 实测 +0.015 ms）。kernel 在正常/失败路径上都会写满 32B，
     * 只有 UNUM_KPREP_STAGE 调试提前 return 时不写 —— 那种跑法不看 u。
     * 注意 u 是复用的：只在本次调用返回后立刻读 u[0]，不要长期持有。 */
    at::Tensor u = UT(ic.options());
    at::Tensor nk = NbkT(nbk, ic.options());

    aclTensor* inT = Wrap(ic, kAclInt32);
    aclTensor* ocT = Wrap(oc, kAclInt32);
    aclTensor* stT = Wrap(st, kAclInt32);
    aclTensor* ofT = Wrap(of, kAclInt32);
    aclTensor* nkT = Wrap(nk, kAclInt64);
    aclTensor* rkT = Wrap(rkeys, kAclInt64);
    aclTensor* cnT = Wrap(cnt, kAclInt32);
    aclTensor* mtT = Wrap(meta, kAclInt32);
    aclTensor* scT = Wrap(scratch, kAclInt32);
    aclTensor* uT = Wrap(u, kAclInt64);
    LaunchArgs la1;
    int rc = f.wsMain(inT, ocT, stT, ofT, nkT, rkT, cnT, mtT, scT, uT, &la1.wsSize, &la1.exec);
    TORCH_CHECK(rc == 0, "aclnnSpconvKeyprepV2GetWorkspaceSize failed: ", rc);
    la1.tensors = {inT, ocT, stT, ofT, nkT, rkT, cnT, mtT, scT, uT};
    Dispatch("aclnnSpconvKeyprepV2", f.runMain, la1, ic);

    aclTensor* saT = Wrap(start, kAclInt32);
    aclTensor* cn2T = Wrap(cnt, kAclInt32);
    LaunchArgs la2;
    rc = f.wsPrefix(cn2T, saT, &la2.wsSize, &la2.exec);
    TORCH_CHECK(rc == 0, "aclnnSpconvKeyprepV2PrefixGetWorkspaceSize failed: ", rc);
    la2.tensors = {cn2T, saT};
    Dispatch("aclnnSpconvKeyprepV2Prefix", f.runPrefix, la2, ic);
    return {rkeys, start, cnt, meta, u};
}
}  // namespace ascend_kernel
