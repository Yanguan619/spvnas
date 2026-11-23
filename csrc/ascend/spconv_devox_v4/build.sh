#!/bin/bash
# 构建入口 — 复用共享模板脚本（CMakePresets 内核包 + op_extension torch 绑定）
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$DIR/../build_torch_binding.sh" "$DIR" "$@"
