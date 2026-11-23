#ifndef SPCONV_QUERY_V2_TILING_H
#define SPCONV_QUERY_V2_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct SpconvQueryV2TilingData {
    uint32_t blockNum;   // 使用核数（= host SetBlockDim）
    uint32_t M;          // 查询点个数
    uint32_t kcnt;       // 每点的 kernel 偏移个数（1 / 8 / 27）
    uint32_t buckets;
    uint32_t dynamic;    // 1 = 行数由 device 上的 n 决定（容量捕获）    // 桶表长度（= start/cnt 元素数）
};

#endif  // SPCONV_QUERY_V2_TILING_H
