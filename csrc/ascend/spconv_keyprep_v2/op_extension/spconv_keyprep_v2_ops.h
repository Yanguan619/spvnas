#ifndef SPCONV_KEYPREP_V2_OPS_H
#define SPCONV_KEYPREP_V2_OPS_H
#include <torch/extension.h>
#include <vector>
namespace ascend_kernel {
/* -> {rkeys int64[N], start int32[nbk], cnt int32[nbk], meta int32[12+3kv],
 *     u int64[4]}
 * u[0] = 0 正常 / -1 跨核屏障自旋超时（此时 rkeys/cnt/meta 不可用，必须回退）；
 * u[1] = core0 观测到的最大自旋次数（诊断）。
 * 本层不做 device->host 同步；失败检查在 Python 侧 `if int(u[0]) < 0: raise`。 */
std::vector<at::Tensor> spconv_keyprep_v2(const at::Tensor& in_coords, const at::Tensor& out_coords,
                                          const at::Tensor& stride, const at::Tensor& offs,
                                          int64_t nbk);
}  // namespace ascend_kernel
#endif  // SPCONV_KEYPREP_V2_OPS_H
