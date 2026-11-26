"""setuptools 构建钩子：`pip install -e .` / `pip install .` 时自动编译 AscendC 算子。

两种安装方式走不同命令，两个都要挂钩子：
- `pip install .`        → build_py（打包前编译，产物进 wheel）
- `pip install -e .`     → editable_wheel（可编辑安装不经过 build_py）

行为：
- 编译 csrc/ascend 下各算子，产物复制进 core/ops/（随包分发）；
- 任一步失败只告警、不中断安装（无 CANN 的机器仍可纯 torch 运行）；
- `SPVNAS_SKIP_ASCENDC_BUILD=1` 完全跳过；`SPVNAS_ASCENDC_ONLY=a,b` 只编指定算子。
"""
import os

from setuptools.command.build_py import build_py as _build_py

try:                                             # setuptools >= 64
    from setuptools.command.editable_wheel import editable_wheel as _editable_wheel
except Exception:                                # 老版本没有该命令
    _editable_wheel = None


def _build_ops():
    if os.environ.get("SPVNAS_SKIP_ASCENDC_BUILD") == "1":
        print("[spvnas] 跳过 AscendC 构建（SPVNAS_SKIP_ASCENDC_BUILD=1）")
        return
    try:
        import build_ascendc

        argv = []
        only = os.environ.get("SPVNAS_ASCENDC_ONLY")
        if only:
            for name in only.split(","):
                if name.strip():
                    argv += ["--only", name.strip()]
        print("[spvnas] 构建 AscendC 算子 %s" % (argv or "（全量）"), flush=True)
        build_ascendc.main(argv)
    except Exception as exc:                     # 没有 CANN / 无 cmake 时不阻断安装
        print("[spvnas] AscendC 构建跳过：%s" % exc)


class build_py(_build_py):
    def run(self):
        _build_ops()
        super().run()


if _editable_wheel is not None:

    class editable_wheel(_editable_wheel):
        def run(self):
            _build_ops()
            super().run()
else:                                            # pragma: no cover
    editable_wheel = None
