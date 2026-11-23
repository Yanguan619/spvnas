#ifndef SPCONV_QUERY_TILING_H
#define SPCONV_QUERY_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct SpconvQueryTilingData {
    uint32_t blockNum;   // 使用核数（= host SetBlockDim）
    uint32_t M;          // 查询点个数
    uint32_t kcnt;       // 每点的 kernel 偏移个数（1 / 8 / 27）
    uint32_t buckets;    // 桶表长度（= start/cnt 元素数）
};

#endif  // SPCONV_QUERY_TILING_H
