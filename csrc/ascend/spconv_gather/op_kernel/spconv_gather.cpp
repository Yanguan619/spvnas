/* -------------------------------------------------------------------------
 * SpconvGather：稀疏卷积的 gather（densify）阶段，为 dav-c220（910B）重写。
 *
 *   dense[m, k*cin + c] = (nb[m,k] >= 0) ? feat[nb[m,k], c] : 0
 *
 * 为什么只做 gather：实测 910B 上 torch fp16/fp32 GEMM 只要 0.09-0.10 ms，
 * 而 gather 占单层 ~75%（3.6 ms/层）。GEMM 交给 torch，核只干搬运。
 *
 * 与 310P 版（SpconvGemmNB）的区别：
 *   - 纯 AIV 核（KERNEL_TYPE_AIV_ONLY），不碰 Cube/L0/L1，不写 Mmad
 *   - 不用 desc/tbeg 描述符：直接按 nb 逐行 DataCopyPad，索引走 GM 标量读
 *     （910B 上 GM 标量读可靠；310P 那套 GM→UB 后再 GetValue 在 910B 不生效）
 *   - 用 TQue 分配的 UB，不用「TBuf + 绝对偏移 arena」
 *   - 输出 fp32，保持与原 torch 路径一致的精度（可选 fp16 由 cin/all 决定）
 * ------------------------------------------------------------------------- */

#include "kernel_operator.h"
#include "spconv_gather_tiling.h"

using namespace AscendC;

namespace {

constexpr int64_t kRowsPerBlk = 8;   // 8 行 × ktot_ × 4B ≡ 0 (mod 32)

class KernelSpconvGather {
public:
    __aicore__ inline KernelSpconvGather(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR feat, GM_ADDR nb, GM_ADDR params,
                                GM_ADDR out, GM_ADDR tiling)
    {
        __gm__ int32_t* p = reinterpret_cast<__gm__ int32_t*>(params);
        M_ = static_cast<int64_t>(p[0]);
        cin_ = static_cast<int64_t>(p[1]);
        kcnt_ = static_cast<int64_t>(p[2]);
        ktot_ = cin_ * kcnt_;
        featGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(feat));
        nbGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(nb));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));
        pipe_->InitBuffer(qRow_, 1, static_cast<uint32_t>(ktot_ * sizeof(float)));
        evtV_ = pipe_->AllocEventID<HardEvent::V_MTE2>();
        evtM_ = pipe_->AllocEventID<HardEvent::MTE2_MTE3>();
        evtO_ = pipe_->AllocEventID<HardEvent::MTE3_V>();
    }

    __aicore__ inline void Process()
    {
        if (M_ <= 0 || ktot_ <= 0) {
            return;
        }
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        const int64_t perRaw = (M_ + blockNum - 1) / blockNum;
        const int64_t per = (perRaw + kRowsPerBlk - 1) / kRowsPerBlk * kRowsPerBlk;
        const int64_t g0 = static_cast<int64_t>(GetBlockIdx()) * per;
        int64_t g1 = g0 + per;
        if (g1 > M_) {
            g1 = M_;
        }
        DataCopyPadExtParams<float> pp;
        pp.isPad = false;
        DataCopyExtParams piece;
        piece.blockCount = 1;
        piece.blockLen = static_cast<uint32_t>(cin_ * sizeof(float));
        piece.srcStride = 0;
        piece.dstStride = 0;
        DataCopyExtParams whole;
        whole.blockCount = 1;
        whole.blockLen = static_cast<uint32_t>(ktot_ * sizeof(float));
        whole.srcStride = 0;
        whole.dstStride = 0;

        for (int64_t base = g0; base < g1; base += kRowsPerBlk) {
            int64_t rows = g1 - base;
            if (rows > kRowsPerBlk) {
                rows = kRowsPerBlk;
            }
            LocalTensor<float> buf = qRow_.AllocTensor<float>();
            Duplicate(buf, 0.0f, static_cast<int32_t>(kRowsPerBlk * ktot_));
            SetFlag<HardEvent::V_MTE2>(evtV_);
            WaitFlag<HardEvent::V_MTE2>(evtV_);
            for (int64_t r = 0; r < rows; r++) {
                const int64_t m = base + r;
                const int64_t nbRow = m * kcnt_;
                for (int64_t k = 0; k < kcnt_; k++) {
                    const int32_t src = nbGm_.GetValue(nbRow + k);
                    if (src < 0) {
                        continue;
                    }
                    DataCopyPad(buf[r * ktot_ + k * cin_],
                                featGm_[static_cast<int64_t>(src) * cin_], piece, pp);
                }
            }
            SetFlag<HardEvent::MTE2_MTE3>(evtM_);
            WaitFlag<HardEvent::MTE2_MTE3>(evtM_);
            whole.blockLen = static_cast<uint32_t>(kRowsPerBlk * ktot_ * sizeof(float));
            DataCopyPad(outGm_[base * ktot_], buf, whole);
            SetFlag<HardEvent::MTE3_V>(evtO_);
            WaitFlag<HardEvent::MTE3_V>(evtO_);
            qRow_.FreeTensor(buf);
        }
    }

    __aicore__ inline void Release()
    {
        pipe_->ReleaseEventID<HardEvent::V_MTE2>(evtV_);
        pipe_->ReleaseEventID<HardEvent::MTE2_MTE3>(evtM_);
        pipe_->ReleaseEventID<HardEvent::MTE3_V>(evtO_);
    }

private:
    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> qRow_;
    TEventID evtV_;
    TEventID evtM_;
    TEventID evtO_;
    GlobalTensor<float> featGm_;
    GlobalTensor<int32_t> nbGm_;
    GlobalTensor<float> outGm_;
    int64_t M_ = 0, cin_ = 0, kcnt_ = 0, ktot_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void spconv_gather(GM_ADDR feat, GM_ADDR nb,
                                                    GM_ADDR params, GM_ADDR out,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SpconvGatherTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KernelSpconvGather op(&pipe);
    op.Init(feat, nb, params, out, tiling);
    op.Process();
    op.Release();
}
