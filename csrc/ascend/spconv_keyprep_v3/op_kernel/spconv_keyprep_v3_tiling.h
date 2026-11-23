#ifndef SPCONV_KEYPREP_V3_TILING_H
#define SPCONV_KEYPREP_V3_TILING_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
struct SpconvKeyprepV3TilingData {
    uint32_t blockNum;  // AIV 核数（= host SetBlockDim）
    uint32_t N;         // in_coords 行数
    uint32_t M;         // out_coords 行数
    uint32_t kv;        // kernel offsets 个数
    uint32_t nbk;       // 桶数（只有 prefix 那段从 cnt 的 shape 拿得到）
    uint32_t dynamic;   // 1 = N/M 由 device 上的 sizes=[n,m] 决定
    uint32_t stage;     // 调试：跑到第几个阶段（默认 5 = 全跑）
};
#endif  // SPCONV_KEYPREP_V3_TILING_H
