#!/usr/bin/env python3
"""构建 csrc/ascend 下的 AscendC 算子，并把产物复制进 core/ops/。

被 `pip install -e .` / `pip install .` 的 build hook 自动调用（见 setup_hooks.py），
也可单独运行：

    python3 build_ascendc.py                  # 全量构建
    python3 build_ascendc.py --list           # 只列算子与产物名
    python3 build_ascendc.py --only spconv_query
    SPVNAS_SKIP_ASCENDC_BUILD=1 pip install -e .   # 跳过构建

环境要求：已 source CANN 的 set_env.sh（需要 ascendc / cmake / python3-dev）。
任一算子构建失败不中断整体流程（除非传 --strict）。
"""
import argparse
import glob
import os
import pathlib
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "csrc" / "ascend"
PKG_OPS = HERE / "core" / "ops"


def op_dirs():
    return sorted(d for d in SRC.iterdir()
                  if d.is_dir() and (d / "build.sh").exists())


def so_path(op_dir):
    """构建产物：op_extension/build/lib*_ops.so（工程名以生成物为准）。"""
    hits = sorted((op_dir / "op_extension" / "build").glob("lib*_ops.so"))
    return hits[0] if hits else None


def _lf(path):
    """把脚本的 CRLF 规范成 LF（Windows 检出/传输会破坏 bash 解释）。"""
    try:
        b = path.read_bytes()
        if b"\r\n" in b:
            path.write_bytes(b.replace(b"\r\n", b"\n"))
    except Exception:
        pass


def _install_vendor_run(op_dir, verbose=True):
    """把 build_out/custom_opp_*.run 安装进 CANN 的 opp/vendors（aclnn 库由此检索）。

    torch 绑定运行期是 dlopen ``<opp>/vendors/<vendor>/op_api/lib/libcust_opapi.so``
    再取 aclnn 符号；只编不装，干净机器上首次调用会报 symbols not found。
    """
    if os.environ.get("SPVNAS_SKIP_OPP_INSTALL") == "1":
        return True
    runs = sorted((op_dir / "build_out").glob("custom_opp_*.run"))
    if not runs:
        if verbose:
            print("    [skip] 未找到 build_out/custom_opp_*.run，跳过 vendor 安装")
        return False
    run = runs[0]
    opp = os.environ.get("OPP_ROOT") or (os.environ.get("ASCEND_HOME_PATH", "").rstrip("/") + "/opp")
    if not os.path.isdir(opp):
        if verbose:
            print("    [skip] opp 目录不存在：%s" % opp)
        return False
    for cmd in (["bash", str(run), "--install-path=%s" % opp],
                ["bash", str(run), "--quiet", "--install-path=%s" % opp],
                ["bash", str(run), "--quiet"]):
        try:
            rc = subprocess.call(cmd)
        except Exception:
            rc = 1
        if rc == 0:
            if verbose:
                print("    [installed] %s -> %s/vendors/" % (run.name, opp))
            return True
    if verbose:
        print("    [WARN] vendor 包装载失败：%s" % run.name)
    return False


def build(dirs, keep_going=True):
    # 共享脚本与各算子 build.sh 都必须是 LF
    _lf(SRC / "build_torch_binding.sh")
    for d in dirs:
        _lf(d / "build.sh")
    ok, failed = 0, []
    for d in dirs:
        print("==> 构建 %s" % d.name, flush=True)
        t0 = time.time()
        rc = subprocess.call(["bash", str(d / "build.sh")], cwd=str(d))
        dt = time.time() - t0
        if rc != 0:
            print("    [FAIL] rc=%d %.1fs" % (rc, dt), flush=True)
            failed.append(d.name)
            if not keep_going:
                break
        else:
            so = so_path(d)
            print("    [OK] %.1fs  %s" % (dt, so.name if so else "（未找到产物）"), flush=True)
            _install_vendor_run(d)
            ok += 1
    return ok, failed


def copy_into_pkg(dirs):
    PKG_OPS.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in dirs:
        so = so_path(d)
        if so:
            shutil.copy2(so, PKG_OPS / so.name)
            n += 1
    if n:
        print("  已复制 %d 个 .so -> %s" % (n, PKG_OPS), flush=True)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=None, help="只构建指定算子目录")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--strict", action="store_true", help="任一失败即退出非零")
    ap.add_argument("--no-copy", action="store_true", help="只构建，不复制进 core/ops")
    args = ap.parse_args(argv)

    if not SRC.exists():
        print("[skip] 未找到 %s（可能不是源码树）" % SRC)
        return 0

    dirs = op_dirs()
    if args.only:
        want = set(args.only)
        dirs = [d for d in dirs if d.name in want]
        if not dirs:
            print("[skip] 未匹配到算子：%s" % ", ".join(sorted(want)))
            return 0

    if args.list:
        for d in dirs:
            so = so_path(d)
            print("  %-24s -> %s" % (d.name, so.name if so else "（尚未构建）"))
        return 0

    ok, failed = build(dirs, keep_going=not args.strict)
    if not args.no_copy:
        copy_into_pkg(dirs)
    print("\n构建成功 %d 个%s" % (ok, "，失败：%s" % ", ".join(failed) if failed else ""))
    return 1 if (failed and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
