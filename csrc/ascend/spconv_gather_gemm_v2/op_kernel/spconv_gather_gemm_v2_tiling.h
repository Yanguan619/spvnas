/* SpconvGatherGemmV2 tiling data: host/kernel 共享。
 *
 * out[m, o] = sum_{k} (nb[m,k] >= 0 ? feat[nb[m,k], c] : 0) * w[k*cin + c, o]
 */
#ifndef SPCONV_GATHER_GEMM_V2_TILING_H
#define SPCONV_GATHER_GEMM_V2_TILING_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
struct SpconvGatherGemmV2TilingData {
    AscendC::tiling::TCubeTiling mm;  // 单 tile 形状 (TM, cout, K) 的 Matmul tiling
    uint32_t blockNum;  // AIC 核数（= host SetBlockDim）；
    uint32_t M;         // 输出行数（nb 行数）
    uint32_t Nr;        // feat 行数
    uint32_t cin;       // feat 列数
    uint32_t kv;        // nb 列数
    uint32_t cout;      // 输出列数
    uint32_t TM;
    uint32_t dynamic;    // 1 = 行数由 device 上的 n 决定（容量捕获）        // 每 tile 行数（A 暂存缓冲按此分配）
};
#endif  // SPCONV_GATHER_GEMM_V2_TILING_H
