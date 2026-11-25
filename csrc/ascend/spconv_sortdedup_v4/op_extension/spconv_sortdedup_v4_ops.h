#ifndef SPCONV_SORTDEDUP_V4_OPS_H
#define SPCONV_SORTDEDUP_V4_OPS_H
#include <torch/extension.h>
#include <vector>
namespace ascend_kernel {
/* coords int32[N,4] -> {out int32[U,4]（contiguous，按字典序去重）, u int64[1]}
 * 一次 launch；u 的 32B 回读是这条路径上唯一的标量同步。
 * kernel 侧 u 是 int64[4]：u[0]=U，u[1]=失败标记（跨核屏障自旋超时，非 0 即失败），
 * u[2]=core0 观测到的最大自旋次数（诊断）。本函数在 u[1]!=0 时抛错，不返回结果。 */
std::vector<at::Tensor> spconv_sortdedup_v4(const at::Tensor& coords,
                                            const at::Tensor& nrow);
}  // namespace ascend_kernel
#endif  // SPCONV_SORTDEDUP_V4_OPS_H
