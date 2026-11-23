/* -------------------------------------------------------------------------
 * SpconvGatherGemmV2：MIX 核（1 AIC + 2 AIV）融合 gather 与 GEMM。
 *
 *   out[m, o] = sum_k (nb[m,k] >= 0 ? feat[nb[m,k], c] : 0) * w[k*cin+c, o]
 *
 * 动机：v1（纯 AIV gather）把 [M, kv*cin] 稠密脏 buffer 写进 GM，GEMM 再整块读回，
 * 一写一读 1.1 GB 全打 HBM。本核把该 buffer 换成「按 TM 行复用的小暂存区」：
 * AIV 把 tap 搬进 UB 再写暂存区（L2 常驻），AIC 直接从暂存区做 Cube。
 *
 * CrossCore 标志语义（实测，910B4 + CANN 9.2.1）：
 *   V→C 的一个 flag id 是「两个 AIV 都到齐才算一个事件」，且一个 wait 只消费一次；
 *   AIC 等两次会死锁（两个 set 合并成一个事件）。所以 V→C 每个 tile 只等一次、
 *   两个 AIV 各写 tile 的一半（FLAG_I2A）；C→V 反向各 AIV 用独立 id（FLAG_A2I+sub）。
 *   只让一个 AIV set 也不行（事件永远不到齐，AIC 挂死）。
 *
 * A tile 只能停在 GM 暂存区（AIV 无法写 AIC 的 L1），所以这里省下的不是
 * 「A 不落 GM」，而是「A 只落 L2 常驻的小 buffer，不落 HBM」。
 * ------------------------------------------------------------------------- */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "spconv_gather_gemm_v2_tiling.h"
using namespace AscendC;

namespace {
constexpr uint16_t FLAG_I2A = 0;    // 两 AIV 都 set，AIC 等一次
constexpr uint16_t FLAG_A2I0 = 2;   // AIC set，AIV sub0 等
constexpr uint16_t FLAG_A2I1 = 3;   // AIC set，AIV sub1 等
constexpr auto MM_CFG = GetMDLConfig(false, false, 0, false);

class KernelFused {
public:
    using AType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    using CType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;

    __aicore__ inline KernelFused(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR feat, GM_ADDR nb, GM_ADDR w, GM_ADDR nrow,
                                GM_ADDR out,
                                GM_ADDR workspace, const SpconvGatherGemmV2TilingData* t)
    {
        M_ = static_cast<int64_t>(t->M);
        dyn_ = (t->dynamic != 0);
        nGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(nrow));
        Nr_ = static_cast<int64_t>(t->Nr);
        cin_ = static_cast<int64_t>(t->cin);
        kv_ = static_cast<int64_t>(t->kv);
        cout_ = static_cast<int64_t>(t->cout);
        TM_ = static_cast<int64_t>(t->TM);
        blockNum_ = static_cast<int64_t>(t->blockNum);
        K_ = cin_ * kv_;
        featGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(feat), Nr_ * cin_);
        nbGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(nb), M_ * kv_);
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(w), K_ * cout_);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out), M_ * cout_);
        stageGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspace)),
                                 2 * blockNum_ * TM_ * K_);
        if ASCEND_IS_AIV {
            pipe_->InitBuffer(qRow_, 1, static_cast<uint32_t>(K_ * sizeof(float)));
            evtV_ = pipe_->AllocEventID<HardEvent::V_MTE2>();
            evtM_ = pipe_->AllocEventID<HardEvent::MTE2_MTE3>();
            evtO_ = pipe_->AllocEventID<HardEvent::MTE3_V>();
        }
        if ASCEND_IS_AIC {
            mm_.SetSubBlockIdx(0);
            mm_.Init(&t->mm, pipe_);
            mm_.SetOrgShape(static_cast<int>(TM_), static_cast<int>(cout_), static_cast<int>(K_),
                            static_cast<int>(K_));
        }
    }

    __aicore__ inline void Process()
    {
        if (dyn_) {
            // 容量捕获：M_ 是容量，真实行数从 device 读；AIV gather 与 Cube 尾块都按它收
            const int32_t nReal = nGm_.GetValue(0);
            if (nReal <= 0) {
                return;
            }
            if (static_cast<int64_t>(nReal) < M_) {
                M_ = static_cast<int64_t>(nReal);
            }
        }
        if (M_ <= 0 || K_ <= 0 || cout_ <= 0) {
            return;
        }
        if ASCEND_IS_AIV {
            GatherLoop();
        }
        if ASCEND_IS_AIC {
            CubeLoop();
        }
    }

private:
    __aicore__ inline void GatherRow(int64_t m, int64_t stageRow)
    {
        LocalTensor<float> row = qRow_.AllocTensor<float>();
        Duplicate(row, 0.0f, static_cast<int32_t>(K_));
        SetFlag<HardEvent::V_MTE2>(evtV_);
        WaitFlag<HardEvent::V_MTE2>(evtV_);
        const int64_t nbRow = m * kv_;
        for (int64_t k = 0; k < kv_; ++k) {
            const int32_t src = nbGm_.GetValue(nbRow + k);
            if (src < 0) {
                continue;
            }
            DataCopyPad(row[k * cin_], featGm_[static_cast<int64_t>(src) * cin_], piece_, pad_);
        }
        SetFlag<HardEvent::MTE2_MTE3>(evtM_);
        WaitFlag<HardEvent::MTE2_MTE3>(evtM_);
        DataCopyPad(stageGm_[stageRow * K_], row, whole_);
        // 行缓冲下一行要复用：等写回读完 UB 再覆写
        SetFlag<HardEvent::MTE3_V>(evtO_);
        WaitFlag<HardEvent::MTE3_V>(evtO_);
        qRow_.FreeTensor(row);
    }

    __aicore__ inline void GatherLoop()
    {
        const int64_t blk = static_cast<int64_t>(GetBlockIdx());
        const int64_t aic = blk >> 1;
        const int32_t sub = static_cast<int32_t>(blk & 1);
        const int64_t half = TM_ >> 1;
        const uint16_t myRelease = (sub == 0) ? FLAG_A2I0 : FLAG_A2I1;
        piece_.blockCount = 1;
        piece_.blockLen = static_cast<uint32_t>(cin_ * sizeof(float));
        piece_.srcStride = 0;
        piece_.dstStride = 0;
        whole_.blockCount = 1;
        whole_.blockLen = static_cast<uint32_t>(K_ * sizeof(float));
        whole_.srcStride = 0;
        whole_.dstStride = 0;
        int64_t ti = 0;
        for (int64_t t = aic; t * TM_ < M_; t += blockNum_, ++ti) {
            const int32_t ping = static_cast<int32_t>(ti & 1);
            const int64_t tileStart = t * TM_;
            const int64_t rows = (M_ - tileStart < TM_) ? (M_ - tileStart) : TM_;
            const int64_t stageBase = (aic * 2 + ping) * TM_;
            if (ti >= 2) {
                CrossCoreWaitFlag<0x2>(myRelease);
            }
            int64_t r0 = sub * half;
            int64_t r1 = r0 + half;
            if (r1 > rows) {
                r1 = rows;
            }
            for (int64_t r = r0; r < r1; ++r) {
                GatherRow(tileStart + r, stageBase + r);
            }
            CrossCoreSetFlag<0x2, PIPE_MTE3>(FLAG_I2A);
        }
    }

    __aicore__ inline void CubeLoop()
    {
        const int64_t aic = static_cast<int64_t>(GetBlockIdx());
        int64_t ti = 0;
        mm_.SetTensorB(wGm_[0]);  // B 指针与 tile 无关，循环外设一次
        for (int64_t t = aic; t * TM_ < M_; t += blockNum_, ++ti) {
            const int32_t ping = static_cast<int32_t>(ti & 1);
            const int64_t tileStart = t * TM_;
            const int64_t rows = (M_ - tileStart < TM_) ? (M_ - tileStart) : TM_;
            CrossCoreWaitFlag<0x2>(FLAG_I2A);  // 两个 AIV 到齐算一次事件，只等一次
            mm_.SetTensorA(stageGm_[(aic * 2 + ping) * TM_ * K_]);
            mm_.SetSingleShape(static_cast<int>(rows), static_cast<int>(cout_), static_cast<int>(K_));
            mm_.IterateAll(outGm_[tileStart * cout_], 0);
            CrossCoreSetFlag<0x2, PIPE_FIX>(FLAG_A2I0);
            CrossCoreSetFlag<0x2, PIPE_FIX>(FLAG_A2I1);
        }
        if (ti > 0) {
            mm_.End();
        }
    }

    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> qRow_;
    TEventID evtV_;
    TEventID evtM_;
    TEventID evtO_;
    DataCopyExtParams piece_;
    DataCopyExtParams whole_;
    DataCopyPadExtParams<float> pad_;
    GlobalTensor<float> featGm_;
    GlobalTensor<int32_t> nbGm_;
    GlobalTensor<float> wGm_;
    GlobalTensor<float> outGm_;
    GlobalTensor<float> stageGm_;
    matmul::MatmulImpl<AType, BType, CType, CType, MM_CFG> mm_;
    GlobalTensor<int32_t> nGm_;
    bool dyn_ = false;
    int64_t M_ = 0;
    int64_t Nr_ = 0;
    int64_t cin_ = 0;
    int64_t kv_ = 0;
    int64_t cout_ = 0;
    int64_t TM_ = 0;
    int64_t blockNum_ = 0;
    int64_t K_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void spconv_gather_gemm_v2(GM_ADDR feat, GM_ADDR nb, GM_ADDR w,
                                                               GM_ADDR nrow, GM_ADDR out,
                                                               GM_ADDR workspace,
                                                               GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(SpconvGatherGemmV2TilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KernelFused op(&pipe);
    op.Init(feat, nb, w, nrow, out, workspace, &tilingData);
    op.Process();
}
