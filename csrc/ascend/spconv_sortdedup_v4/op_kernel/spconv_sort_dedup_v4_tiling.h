#ifndef SPCONV_SORTDEDUP_V4_TILING_H
#define SPCONV_SORTDEDUP_V4_TILING_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
struct SpconvSortDedupV4TilingData {
    uint32_t blockNum;   // AIV 核数（= host SetBlockDim）
    uint32_t N;          // 输入行数
    uint32_t cols;       // 输入列数（4）
    uint32_t outCols;    // 输出行宽（8，32B 对齐）
    uint32_t stage;      // 调试：跑到哪个阶段（默认 0xFFFF = 全跑）
    uint32_t mode;       // 调试：散射写回通路（0=pad-to-pos 1=pad-to-idx 2=copy-to-pos 3=copy-to-idx）
    uint32_t dynamic;    // 1 = 真实行数在 kernel 内从 n 读（容量捕获）
    uint32_t spinLimit;  // 软屏障自旋上限（默认 20000 ≈ 数十 ms；0 = 用 kernel 默认值）
};
/* scratch 元素数（int32）：屏障格子 + box + 直方图 + keeps。host 按它分配。 */
static const int64_t kSDScratchInts = 27712;
#endif  // SPCONV_SORTDEDUP_V4_TILING_H
