#!/bin/bash
# 共享构建脚本（模板 A：CMakePresets 内核包 + op_extension torch 绑定双阶段）。
#
# 使用方：bev_pool / bev_pool_v3 / spconv / swin_ffn（各自目录下的 build.sh
# 为本脚本的薄包装，传自身目录进来）；也可直接调用：
#
#   bash csrc/ascend/build_torch_binding.sh csrc/ascend/<operator>
#
# 阶段 1：preset_parse.py 解析 CMakePresets.json → cmake --target binary package
#         产出 build_out/custom_opp_*.run（vendor 内核包）
# 阶段 2：cmake op_extension/ → 产出 op_extension/build/*.so（torch 直调绑定）
set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <operator_project_dir>"
    echo "  e.g. $0 csrc/ascend/bev_pool"
    exit 1
fi
script_path=$(realpath "$1")
if [ ! -f "$script_path/CMakePresets.json" ]; then
    echo "ERROR: $script_path/CMakePresets.json not found."
    echo "  本脚本仅适用于模板 A 工程（CMakePresets + op_extension）。"
    exit 1
fi

if [ -z "$BASE_LIBS_PATH" ]; then
    if [ -z "$ASCEND_HOME_PATH" ]; then
        if [ -z "$ASCEND_AICPU_PATH" ]; then
            echo "please set env."
            exit 1
        else
            export ASCEND_HOME_PATH=$ASCEND_AICPU_PATH
        fi
    else
        export ASCEND_HOME_PATH=$ASCEND_HOME_PATH
    fi
else
    export ASCEND_HOME_PATH=$BASE_LIBS_PATH
fi
echo "using ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
cd "$script_path"

BUILD_DIR="build_out"
mkdir -p build_out
rm -rf build_out/*
opts=$(python3 $ASCEND_HOME_PATH/tools/tikcpp/ascendc_kernel_cmake/fwk_modules/util/preset_parse.py $script_path/CMakePresets.json)
cmake_version=$(cmake --version | grep "cmake version" | awk '{print $3}')
min_version="3.19.0"

if [ "$(printf '%s\n' "$min_version" "$cmake_version" | sort -V | head -n1)" != "$min_version" ] ; then
    cmake -S . -B "$BUILD_DIR" $opts
else
    cmake -S . -B "$BUILD_DIR" --preset=default
fi
cmake --build "$BUILD_DIR" --target binary package -j$(nproc)

echo "--- building op_extension ---"
EXT_DIR="$script_path/op_extension"
EXT_BUILD="$EXT_DIR/build"
mkdir -p "$EXT_BUILD"
TORCH_CMAKE=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null)
if [ -z "$TORCH_CMAKE" ] || [ ! -d "$TORCH_CMAKE" ]; then
    echo "ERROR: torch not found (python3 -c 'import torch' failed)."
    echo "Please install torch (and torch_npu) in the python3 environment first."
    exit 1
fi
cd "$EXT_BUILD"
# torch >= 2.14 要求 C++20（否则 "#error C++20 or later compatible compiler is required"）
cmake "$EXT_DIR" -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
      -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_STANDARD_REQUIRED=ON
cmake --build "$EXT_BUILD" -j$(nproc)
echo "--- op_extension built: $EXT_BUILD/*.so ---"
