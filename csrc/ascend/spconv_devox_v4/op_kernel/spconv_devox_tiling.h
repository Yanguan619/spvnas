#ifndef SPCONV_DEVOX_TILING_H
#define SPCONV_DEVOX_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct SpconvDevoxTilingData {
    uint32_t blockNum;    // 使用核数（= host SetBlockDim）
    uint32_t N;           // 输出行数（点数）
    uint32_t c;           // 特征列数
    uint32_t kcnt;        // 每行的角数（8）
    uint32_t calcWeights; // 1 = 核内算三线性权重；0 = 读 weights 张量
    uint32_t dynamic;     // 1 = 行数由 device 上的 n 决定（容量捕获）
    float scale;          // 三线性网格尺度（通常 1）
};

#endif  // SPCONV_DEVOX_TILING_H
