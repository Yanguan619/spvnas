#ifndef SPCONV_KEYPREP_V3_COMMON_H
#define SPCONV_KEYPREP_V3_COMMON_H
#include <cstdint>
/* 两个 kernel 共用的 scratch 布局与 tile 尺寸（scratch 是 4096 个 int32 的输出张量，
 * 因为 aclnn 直调路径上 kernel 的 workspace 参数不可写）。 */
namespace kprep {
constexpr int64_t kSlot = 8;            // 32B / sizeof(int32)
constexpr int64_t kMaxBlocks = 64;
constexpr int64_t kTileRows = 2048;     // 2048 行 * 16B = 32KB
constexpr int64_t kTileInts = 8192;     // 8192 * 4B = 32KB
constexpr int64_t kBoxPart = 0;         // [64][8] 每核局部 box
constexpr int64_t kPart = 1024;         // [64][8] 每核桶段和，值在 slot0
constexpr int64_t kBar = 1600;          // [1][64][8] 屏障，每核一格
constexpr int64_t kScratchInts = 4096;

__aicore__ inline int32_t BitLen(int64_t v)
{
    int32_t n = 0;
    while (v > 0) {
        v >>= 1;
        ++n;
    }
    return n;
}
}  // namespace kprep
#endif  // SPCONV_KEYPREP_V3_COMMON_H
