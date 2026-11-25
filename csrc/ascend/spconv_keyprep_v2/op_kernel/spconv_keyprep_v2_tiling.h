#ifndef SPCONV_KEYPREP_V2_TILING_H
#define SPCONV_KEYPREP_V2_TILING_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
struct SpconvKeyprepV2TilingData {
    uint32_t blockNum;  // AIV 核数（= host SetBlockDim）
    uint32_t N;         // in_coords 行数
    uint32_t M;         // out_coords 行数
    uint32_t kv;        // kernel offsets 行数
    uint32_t nbk;       // 桶个数；只有 prefix 那段从 cnt 的 shape 得到
    uint32_t stage;     // 调试：跑到哪个阶段（默认 5 = 全跑）
    uint32_t spinLimit; // 软屏障自旋上限（默认 4000 ≈ 数十 ms；0 = kernel 默认值）
};
#endif  // SPCONV_KEYPREP_V2_TILING_H
