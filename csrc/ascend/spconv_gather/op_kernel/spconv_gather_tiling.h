#ifndef SPCONV_GATHER_TILING_H
#define SPCONV_GATHER_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct SpconvGatherTilingData {
    uint32_t blockNum;   // 实际使用核数（= host SetBlockDim）
};

#endif  // SPCONV_GATHER_TILING_H
