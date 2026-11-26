/*
 * Copyright (c) 2026 Yanguan02
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * unum_aclnn_loader.h —— 自建 OPP vendor 的 aclnn 加载器（路径隔离版）。
 *
 * 为什么需要这一层（2026-11-26 在 310P 容器实测，证据见
 * docs/notes/aclnn_vendor_loader.md）：
 *
 *  1. 每个 vendor 的 op_api 库，文件名与 SONAME **全都**是 libcust_opapi.so；
 *     版本号只体现在 vendor 目录名（bev_pool / bev_pool_v3）和 aclnn 符号名
 *     （aclnnBevPool / aclnnBEVPoolV3）上，库本身的交付名由 CANN 的
 *     npu_op_library 模板钉死，改不了。
 *  2. 所以「链接期 -lcust_opapi」是危险的：ELF 里只记下 SONAME，运行期由
 *     LD_LIBRARY_PATH 决定真正加载哪个 vendor 的同名库 —— 指错目录就是静默
 *     串味（实测：NEEDED=[libcust_opapi.so]，LD_LIBRARY_PATH 指到 spconv 就
 *     加载 spconv 的那份）。
 *  3. 反过来，**绝对路径 dlopen 不会**因 SONAME 相同而合并：实测同时 dlopen
 *     bev_pool 与 spconv 两份，得到两个不同 handle，各自符号正确解析。
 *     所以隔离手段就是「绝对路径 + RTLD_LAZY|RTLD_LOCAL + 每 vendor 独立 handle」。
 *  4. dlopen 失败与符号缺失必须分开报（带 dlerror、带试过的候选路径），
 *     否则现场只剩一句 "symbols not found"，排不动。
 *
 * 用法（每个 op_extension 的 *_torch.cpp 都按这个形状写）：
 *
 *   const unum::VendorLib& lib = unum::OpenVendor(UNUM_VENDOR_DIR, &SomeFuncInThisTU);
 *   std::string err;
 *   auto get_ws = unum::Resolve<GetWorkspaceSizeFunc>(lib, "aclnnBevPoolGetWorkspaceSize", &err);
 *   auto run    = unum::Resolve<RunFunc>(lib, "aclnnBevPool", &err);
 *   void* nnop  = unum::OpenDep("libnnopbase.so");   // 内部带 CANN 绝对路径兜底
 *   TORCH_CHECK(get_ws && run && nnop, err);
 *
 * 语义约定：
 *   - 失败**不缓存**：装了 OPP / 换了机器 / 换了 ASCEND_OPP_PATH 后，同进程重试
 *     仍能成功（旧实现里 loaded = true 会把一次瞬时失败变成永久失败）。
 *   - 成功才缓存 handle；缓存键是 vendor 名。
 *   - 不写死 CANN 版本号：路径从 ASCEND_OPP_PATH / ASCEND_HOME_PATH 推导，
 *     最后才扫 /usr/local/Ascend/<任意版本>/opp。
 */

#ifndef UNUM_ACLNN_LOADER_H
#define UNUM_ACLNN_LOADER_H

// dladdr / Dl_info 需要 _GNU_SOURCE；必须在任何系统头之前
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include <dlfcn.h>
#include <dirent.h>
#include <limits.h>
#include <unistd.h>

#include <cstdlib>
#include <map>
#include <string>
#include <vector>

// Dl_info / dladdr 只在 _GNU_SOURCE 生效（__USE_GNU）时可见。若上游头（Python.h、
// torch）已经先一步 include 过 <dlfcn.h> 且没开 _GNU_SOURCE，这里就拿不到 Dl_info，
// 于是退化成「只做绝对路径隔离、跳过符号来源校验」，而不是编译失败。
#if defined(__USE_GNU)
#define UNUM_HAS_DLADDR 1
#else
#define UNUM_HAS_DLADDR 0
#endif

namespace unum {

// ---------------------------------------------------------------------------
// 基础工具
// ---------------------------------------------------------------------------

inline const char* DlErr()
{
    const char* e = ::dlerror();
    return e != nullptr ? e : "(dlerror 无信息)";
}

inline std::string RealPath(const std::string& p)
{
    char buf[PATH_MAX];
    if (::realpath(p.c_str(), buf) != nullptr) {
        return std::string(buf);
    }
    return p;
}

inline bool PathExists(const std::string& p)
{
    return ::access(p.c_str(), F_OK) == 0;
}

namespace detail {

inline void PushUnique(std::vector<std::string>& v, const std::string& s)
{
    if (s.empty()) {
        return;
    }
    for (const auto& e : v) {
        if (e == s) {
            return;
        }
    }
    v.push_back(s);
}

}  // namespace detail

// ---------------------------------------------------------------------------
// opp / CANN 根目录
// ---------------------------------------------------------------------------

// ASCEND_OPP_PATH 允许 ':' 分隔多值（CANN 官方行为）；没有就由 ASCEND_HOME_PATH
// 推导；两者都没有时才去扫 /usr/local/Ascend/*/opp（扫不到就只剩一条候选）。
inline std::vector<std::string> OppRoots()
{
    std::vector<std::string> roots;

    if (const char* raw = std::getenv("ASCEND_OPP_PATH")) {
        std::string cur;
        for (const char* p = raw; *p != '\0'; ++p) {
            if (*p == ':') {
                detail::PushUnique(roots, cur);
                cur.clear();
            } else {
                cur.push_back(*p);
            }
        }
        detail::PushUnique(roots, cur);
    }

    if (const char* home = std::getenv("ASCEND_HOME_PATH")) {
        detail::PushUnique(roots, std::string(home) + "/opp");
    }

    // 最后兜底：不写死版本号，扫一遍 /usr/local/Ascend/*/opp
    if (DIR* d = ::opendir("/usr/local/Ascend")) {
        while (dirent* ent = ::readdir(d)) {
            const std::string name(ent->d_name);
            if (name == "." || name == "..") {
                continue;
            }
            detail::PushUnique(roots, "/usr/local/Ascend/" + name + "/opp");
        }
        ::closedir(d);
    }

    std::vector<std::string> existing;
    for (const auto& r : roots) {
        if (PathExists(r)) {
            existing.push_back(r);
        }
    }
    return existing;
}

// opp 根上一级就是 CANN 根（$ASCEND_HOME_PATH）
inline std::vector<std::string> CannRoots()
{
    std::vector<std::string> roots;
    for (const auto& opp : OppRoots()) {
        const std::string suffix = "/opp";
        if (opp.size() > suffix.size() &&
            opp.compare(opp.size() - suffix.size(), suffix.size(), suffix) == 0) {
            detail::PushUnique(roots, opp.substr(0, opp.size() - suffix.size()));
        }
    }
    if (const char* home = std::getenv("ASCEND_HOME_PATH")) {
        detail::PushUnique(roots, home);
    }
    return roots;
}

// ---------------------------------------------------------------------------
// vendor 的 libcust_opapi.so
// ---------------------------------------------------------------------------

struct VendorLib {
    void* handle = nullptr;
    std::string path;  // 实际加载路径（realpath 之后）
    std::string err;   // 失败原因，含试过的候选与 dlerror

    explicit operator bool() const { return handle != nullptr; }
};

namespace detail {

// wheel 安装场景：本 .so 位于 <pkg>/<op>/_libs/，vendor 打包在同级 _opp/<vendor>/
inline std::string PackagedVendorLib(const void* sentinel, const std::string& vendor)
{
#if UNUM_HAS_DLADDR
    Dl_info info{};
    if (sentinel == nullptr || ::dladdr(sentinel, &info) == 0 || info.dli_fname == nullptr) {
        return std::string();
    }
    const std::string self(info.dli_fname);
    std::string::size_type slash = self.find_last_of('/');
    if (slash == std::string::npos) {
        return std::string();
    }
    const std::string libs_dir = self.substr(0, slash);
    slash = libs_dir.find_last_of('/');
    if (slash == std::string::npos) {
        return std::string();
    }
    const std::string op_dir = libs_dir.substr(0, slash);
    return op_dir + "/_opp/" + vendor + "/op_api/lib/libcust_opapi.so";
#else
    (void)sentinel;
    (void)vendor;
    return std::string();
#endif
}

}  // namespace detail

inline std::vector<std::string> VendorCandidates(const std::string& vendor, const void* sentinel)
{
    std::vector<std::string> cands;

    // 显式覆盖，排障时最省事：UNUM_VENDOR_LIB=/path/to/libcust_opapi.so
    if (const char* explicit_lib = std::getenv("UNUM_VENDOR_LIB")) {
        detail::PushUnique(cands, explicit_lib);
    }
    for (const auto& opp : OppRoots()) {
        detail::PushUnique(cands, opp + "/vendors/" + vendor + "/op_api/lib/libcust_opapi.so");
    }
    detail::PushUnique(cands, detail::PackagedVendorLib(sentinel, vendor));
    return cands;
}

// 打开某 vendor 的 aclnn 库：绝对路径 dlopen，每 vendor 独立 handle。
// 成功缓存；失败不缓存（下次调用会重新尝试，并把新的错误原因带回）。
inline const VendorLib& OpenVendor(const std::string& vendor, const void* sentinel)
{
    static std::map<std::string, VendorLib> cache;

    auto it = cache.find(vendor);
    if (it != cache.end() && it->second.handle != nullptr) {
        return it->second;
    }

    VendorLib lib;
    std::string tried;
    for (const auto& cand : VendorCandidates(vendor, sentinel)) {
        if (!PathExists(cand)) {
            tried += "\n  [不存在]   " + cand;
            continue;
        }
        void* h = ::dlopen(cand.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (h == nullptr) {
            tried += "\n  [dlopen失败] " + cand + ": " + DlErr();
            continue;
        }
        lib.handle = h;
        lib.path = RealPath(cand);
        break;
    }

    if (lib.handle == nullptr) {
        lib.err = "无法加载 vendor '" + vendor +
                  "' 的 libcust_opapi.so（确认 OPP vendor 已安装 / ASCEND_OPP_PATH 正确）。候选：" + tried;
    }

    cache[vendor] = lib;
    return cache[vendor];
}

// dlsym + 来源校验：符号必须真的来自 lib.path。
// 这一步兜住「预加载了别的 vendor / LD_PRELOAD 顶替」这类静默串味——
// 正常绝对路径 dlopen 不会串（实测），但静默用错实现比报错难查得多。
template <typename Fn>
Fn Resolve(const VendorLib& lib, const char* symbol, std::string* err = nullptr)
{
    if (lib.handle == nullptr) {
        if (err != nullptr) {
            *err = lib.err;
        }
        return nullptr;
    }

    ::dlerror();
    void* p = ::dlsym(lib.handle, symbol);
    if (p == nullptr) {
        if (err != nullptr) {
            *err = std::string("dlsym(") + symbol + ") 失败: " + DlErr() + "；库: " + lib.path;
        }
        return nullptr;
    }

#if UNUM_HAS_DLADDR
    Dl_info info{};
    if (::dladdr(p, &info) != 0 && info.dli_fname != nullptr &&
        RealPath(info.dli_fname) != lib.path) {
        if (err != nullptr) {
            *err = std::string("符号 ") + symbol + " 来自 " + info.dli_fname + "，期望来自 " +
                   lib.path + "（vendor 串味）";
        }
        return nullptr;
    }
#endif

    return reinterpret_cast<Fn>(p);
}

// ---------------------------------------------------------------------------
// 依赖库（libnnopbase / libascendcl / libopapi）：先按 soname 让 ld.so 找，
// 再按 CANN 绝对路径兜底。
//
// OpenDep      → RTLD_LOCAL，取 createTensor 这类自己直接调用的符号
// OpenDepGlobal→ RTLD_GLOBAL，用于「让 vendor 库自己内部的未定义符号能解析」
//                （如 l0op::Contiguous / aclrt*，见 voxelization 的踩坑注释）
// ---------------------------------------------------------------------------

namespace detail {

inline void* OpenDepWithFlags(const char* soname, int flags)
{
    static std::map<std::string, void*> cache;
    const std::string key = std::string(soname) + ((flags & RTLD_GLOBAL) != 0 ? "|GLOBAL" : "|LOCAL");

    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    void* h = ::dlopen(soname, flags);

    if (h == nullptr) {
        static const char* kSubDirs[] = {
            "/aarch64-linux/devlib/",
            "/aarch64-linux/devlib/linux/aarch64/",
            "/aarch64-linux/lib64/",
            "/lib64/",
            "/runtime/lib64/",
        };
        const std::string name(soname);
        for (const auto& root : CannRoots()) {
            for (const char* sub : kSubDirs) {
                const std::string cand = root + sub + name;
                if (!PathExists(cand)) {
                    continue;
                }
                h = ::dlopen(cand.c_str(), flags);
                if (h != nullptr) {
                    break;
                }
            }
            if (h != nullptr) {
                break;
            }
        }
    }

    if (h != nullptr) {
        cache[key] = h;
    }
    return h;
}

}  // namespace detail

inline void* OpenDep(const char* soname)
{
    return detail::OpenDepWithFlags(soname, RTLD_LAZY | RTLD_LOCAL);
}

// 以 RTLD_GLOBAL 提升依赖库，供 vendor 库内部符号解析（voxelization 需要）
inline void* OpenDepGlobal(const char* soname)
{
    return detail::OpenDepWithFlags(soname, RTLD_LAZY | RTLD_GLOBAL);
}

// 同 Resolve，但只校验 dlsym 成功与否（依赖库不参与 vendor 串味判定）
template <typename Fn>
Fn ResolveDep(void* handle, const char* symbol, std::string* err = nullptr)
{
    if (handle == nullptr) {
        if (err != nullptr) {
            *err = std::string("依赖库未加载，无法取 ") + symbol;
        }
        return nullptr;
    }
    ::dlerror();
    void* p = ::dlsym(handle, symbol);
    if (p == nullptr && err != nullptr) {
        *err = std::string("dlsym(") + symbol + ") 失败: " + DlErr();
    }
    return reinterpret_cast<Fn>(p);
}

}  // namespace unum

#endif  // UNUM_ACLNN_LOADER_H
