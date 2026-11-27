/* -------------------------------------------------------------------------
 * SpconvDevox v3：v2 的真题——让双缓冲**真的重叠**
 *
 * v2 的问题（实测只快 5–8%）：prefetch 发出后立刻 SetFlag/WaitFlag(MTE2_V)，
 * 且两个 slot 共用一个 event id，MTE2 与 Vector 仍然串行。
 *
 * v3 的做法：
 *   1. 两个固定 UB buffer（TBuf 各 depth=1），slot 身份由下标显式管理，
 *      不依赖 AllocTensor 返回哪个 slot；
 *   2. **每个 slot 各自一对 event**（MTE2→V、V→MTE2），互不干扰；
 *   3. 软件流水：先给下一个有效角发 DMA，再等/算当前角 → 搬运与计算重叠；
 *   4. 用 pendingV[] 精确记账每个 slot 是否还有未消费的 V→MTE2 标志，
 *      避免"等下一条永远不会来的 flag"（v1/v2 都栽在这类配对错误上）；
 *   5. 全无效行用哨兵 first == k_ 短路，不去索引 feat[-c]（v2 死锁点）。
 *
 * 语义与 v1/v2 一致：out[n,c] = Σ_k w[k,n]·feat[idx[k,n],c]，idx<0 跳过。
 * ------------------------------------------------------------------------- */

#include "kernel_operator.h"
#include "spconv_devox_tiling.h"

using namespace AscendC;

namespace {

constexpr int32_t kMaxK = 8;

class KernelSpconvDevox {
public:
    __aicore__ inline KernelSpconvDevox(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR feat, GM_ADDR idx, GM_ADDR w,
                                GM_ADDR coords, GM_ADDR nrow, GM_ADDR out,
                                const SpconvDevoxTilingData* tiling)
    {
        N_ = static_cast<int64_t>(tiling->N);
        c_ = static_cast<int64_t>(tiling->c);
        k_ = static_cast<int64_t>(tiling->kcnt);
        calcW_ = tiling->calcWeights != 0;
        dyn_ = (tiling->dynamic != 0);
        if (dyn_) {
            nGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(nrow));
        }
        scale_ = tiling->scale;
        featGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(feat));
        idxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idx));
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(w));
        coordGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(coords));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));

        const uint32_t bytes = static_cast<uint32_t>(c_ * sizeof(float));
        if (N_ > 0 && c_ > 0) {
            pipe_->InitBuffer(accBuf_, bytes);
            pipe_->InitBuffer(slotBuf_[0], bytes);
            pipe_->InitBuffer(slotBuf_[1], bytes);
        }
        evtAccV_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
        evtAccO_ = pipe_->AllocEventID<HardEvent::MTE3_V>();
        for (int32_t s = 0; s < 2; s++) {
            evtM2V_[s] = pipe_->AllocEventID<HardEvent::MTE2_V>();
            evtV2M_[s] = pipe_->AllocEventID<HardEvent::V_MTE2>();
            pendingV_[s] = false;
        }
    }

    __aicore__ inline float Fl(float v) const
    {
        int32_t i = static_cast<int32_t>(v);
        if (static_cast<float>(i) > v) {
            i -= 1;
        }
        return static_cast<float>(i);
    }

    __aicore__ inline void FillWeights(int64_t n, float* wv, const int32_t* iv)
    {
        const int64_t b = n * 3;
        const float fx = coordGm_.GetValue(b);
        const float fy = coordGm_.GetValue(b + 1);
        const float fz = coordGm_.GetValue(b + 2);
        const float xf = Fl(fx / scale_) * scale_;
        const float yf = Fl(fy / scale_) * scale_;
        const float zf = Fl(fz / scale_) * scale_;
        const float xc = xf + scale_;
        const float yc = yf + scale_;
        const float zc = zf + scale_;
        const float sx = xc - fx, sy = yc - fy, sz = zc - fz;
        const float tx = fx - xf, ty = fy - yf, tz = fz - zf;
        wv[0] = sx * sy * sz;
        wv[1] = sx * sy * tz;
        wv[2] = sx * ty * sz;
        wv[3] = sx * ty * tz;
        wv[4] = tx * sy * sz;
        wv[5] = tx * sy * tz;
        wv[6] = tx * ty * sz;
        wv[7] = tx * ty * tz;
        float s = 0.0f;
        for (int32_t k = 0; k < k_; k++) {
            if (iv[k] < 0) {
                wv[k] = 0.0f;
            }
            s += wv[k];
        }
        const float inv = (scale_ != 1.0f)
                              ? (1.0f / (s + 1e-8f)) / (scale_ * scale_ * scale_)
                              : 1.0f / (s + 1e-8f);
        for (int32_t k = 0; k < k_; k++) {
            wv[k] *= inv;
        }
    }

    /* 给 slot 发一次 DMA。若该 slot 还有未消费的 V→MTE2 标志，先等掉。 */
    __aicore__ inline void Prefetch(int32_t slot, int32_t k, const int32_t* iv,
                                    const DataCopyExtParams& piece,
                                    const DataCopyPadExtParams<float>& pp)
    {
        if (pendingV_[slot]) {
            WaitFlag<HardEvent::V_MTE2>(evtV2M_[slot]);
            pendingV_[slot] = false;
        }
        LocalTensor<float> buf = slotBuf_[slot].Get<float>();
        DataCopyPad(buf, featGm_[static_cast<int64_t>(iv[k]) * c_], piece, pp);
        SetFlag<HardEvent::MTE2_V>(evtM2V_[slot]);
    }

    __aicore__ inline void Process()
    {
        if (N_ <= 0 || c_ <= 0 || k_ <= 0) {
            return;
        }
        // N_ 是 idx/w 的行距（容量），Nreal_ 才是本轮要算的点数
        int64_t Nreal_ = N_;
        if (dyn_) {
            const int32_t nReal = nGm_.GetValue(0);
            if (nReal <= 0) {
                return;
            }
            if (static_cast<int64_t>(nReal) < Nreal_) {
                Nreal_ = static_cast<int64_t>(nReal);
            }
        }
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        const int64_t per = (Nreal_ + blockNum - 1) / blockNum;
        const int64_t g0 = static_cast<int64_t>(GetBlockIdx()) * per;
        int64_t g1 = g0 + per;
        if (g1 > Nreal_) {
            g1 = Nreal_;
        }

        DataCopyPadExtParams<float> pp;
        pp.isPad = false;
        DataCopyExtParams piece;
        piece.blockCount = 1;
        piece.blockLen = static_cast<uint32_t>(c_ * sizeof(float));
        piece.srcStride = 0;
        piece.dstStride = 0;
        const DataCopyExtParams whole = piece;

        LocalTensor<float> acc = accBuf_.Get<float>();

        for (int64_t n = g0; n < g1; n++) {
            int32_t iv[kMaxK];
            float wv[kMaxK];
            for (int32_t k = 0; k < k_; k++) {
                iv[k] = idxGm_.GetValue(static_cast<int64_t>(k) * N_ + n);
            }
            if (calcW_) {
                FillWeights(n, wv, iv);
            } else {
                for (int32_t k = 0; k < k_; k++) {
                    wv[k] = wGm_.GetValue(static_cast<int64_t>(k) * N_ + n);
                }
            }

            int32_t first = k_;
            for (int32_t k = 0; k < k_; k++) {
                if (iv[k] >= 0 && wv[k] != 0.0f) {
                    first = k;
                    break;
                }
            }

            Duplicate(acc, 0.0f, static_cast<int32_t>(c_));
            SetFlag<HardEvent::V_MTE2>(evtV2M_[0]);
            WaitFlag<HardEvent::V_MTE2>(evtV2M_[0]);
            SetFlag<HardEvent::V_MTE2>(evtV2M_[1]);
            WaitFlag<HardEvent::V_MTE2>(evtV2M_[1]);
            pendingV_[0] = false;
            pendingV_[1] = false;

            if (first < k_) {
                Prefetch(0, first, iv, piece, pp);
                int32_t cur = 0;
                for (int32_t k = first; k < k_; k++) {
                    if (iv[k] < 0 || wv[k] == 0.0f) {
                        continue;
                    }
                    int32_t nxt = k + 1;
                    while (nxt < k_ && (iv[nxt] < 0 || wv[nxt] == 0.0f)) {
                        nxt++;
                    }
                    if (nxt < k_) {
                        Prefetch(cur ^ 1, nxt, iv, piece, pp);
                    }

                    WaitFlag<HardEvent::MTE2_V>(evtM2V_[cur]);
                    LocalTensor<float> buf = slotBuf_[cur].Get<float>();
                    Muls(buf, buf, wv[k], static_cast<int32_t>(c_));
                    Add(acc, acc, buf, static_cast<int32_t>(c_));
                    SetFlag<HardEvent::V_MTE2>(evtV2M_[cur]);
                    pendingV_[cur] = true;
                    cur ^= 1;
                }
                // 收尾：把两个 slot 的 V→MTE2 标志都消费掉
                for (int32_t s = 0; s < 2; s++) {
                    if (pendingV_[s]) {
                        WaitFlag<HardEvent::V_MTE2>(evtV2M_[s]);
                        pendingV_[s] = false;
                    }
                }
            }

            SetFlag<HardEvent::V_MTE3>(evtAccV_);
            WaitFlag<HardEvent::V_MTE3>(evtAccV_);
            DataCopyPad(outGm_[n * c_], acc, whole);
            SetFlag<HardEvent::MTE3_V>(evtAccO_);
            WaitFlag<HardEvent::MTE3_V>(evtAccO_);
        }
    }

    __aicore__ inline void Release()
    {
        pipe_->ReleaseEventID<HardEvent::V_MTE3>(evtAccV_);
        pipe_->ReleaseEventID<HardEvent::MTE3_V>(evtAccO_);
        for (int32_t s = 0; s < 2; s++) {
            pipe_->ReleaseEventID<HardEvent::MTE2_V>(evtM2V_[s]);
            pipe_->ReleaseEventID<HardEvent::V_MTE2>(evtV2M_[s]);
        }
    }

private:
    TPipe* pipe_;
    TBuf<QuePosition::VECIN> accBuf_;
    TBuf<QuePosition::VECIN> slotBuf_[2];
    TEventID evtAccV_;
    TEventID evtAccO_;
    TEventID evtM2V_[2];
    TEventID evtV2M_[2];
    bool pendingV_[2] = {false, false};
    GlobalTensor<float> featGm_;
    GlobalTensor<int32_t> idxGm_;
    GlobalTensor<float> wGm_;
    GlobalTensor<float> coordGm_;
    GlobalTensor<float> outGm_;
    GlobalTensor<int32_t> nGm_;
    bool dyn_ = false;
    int64_t N_ = 0;
    int64_t c_ = 0;
    int64_t k_ = 0;
    bool calcW_ = false;
    float scale_ = 1.0f;
};

}  // namespace

extern "C" __global__ __aicore__ void spconv_devox_v4(GM_ADDR feat, GM_ADDR idx,
                                                     GM_ADDR w, GM_ADDR coords,
                                                     GM_ADDR nrow, GM_ADDR out,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SpconvDevoxTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KernelSpconvDevox op(&pipe);
    op.Init(feat, idx, w, coords, nrow, out, &tilingData);
    op.Process();
    op.Release();
}
