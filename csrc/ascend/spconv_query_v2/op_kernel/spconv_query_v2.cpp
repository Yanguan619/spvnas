/* -------------------------------------------------------------------------
 * SpconvQueryV2：把「拼 [M,KV,4] 查询张量 → pack_keys → 二分」三步合成一个 AIV 核。
 *
 *   idx[m*KV + k] = rkeys 中等于 key(qcoords[m], stride, offset[k]) 的下标，缺失 -1
 *
 * 对应 torchsparse CUDA 的 subm_hashmap_kmap_stage2：coord 变换、越界判断、
 * 查表全部在核内，一个线程一个 (点, kernel) 对，不产生任何主机侧索引张量。
 * 主机侧只剩一张桶表（index_add_ + cumsum，与 CUDA 的 insert 阶段对应）。
 *
 * 只用两条已验证可用的通路：GM 标量读、UB + DataCopyPad 写回。
 * ------------------------------------------------------------------------- */

#include "kernel_operator.h"
#include "spconv_query_v2_tiling.h"

using namespace AscendC;

namespace {

constexpr int32_t kMaxK = 32;

class KernelSpconvQueryV2 {
public:
    __aicore__ inline KernelSpconvQueryV2(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR qcoords, GM_ADDR rkeys, GM_ADDR start,
                                GM_ADDR cnt, GM_ADDR meta, GM_ADDR nrow, GM_ADDR out,
                                const SpconvQueryV2TilingData* tiling)
    {
        M_ = static_cast<int64_t>(tiling->M);
        dyn_ = (tiling->dynamic != 0);
        nGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(nrow));
        K_ = static_cast<int64_t>(tiling->kcnt);
        buckets_ = static_cast<int64_t>(tiling->buckets);
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(qcoords));
        rGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(rkeys));
        startGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(start));
        cntGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cnt));
        metaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(meta));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(out));

        shift_ = static_cast<int64_t>(metaGm_.GetValue(0));
        for (int32_t i = 0; i < 3; i++) {
            stride_[i] = static_cast<int64_t>(metaGm_.GetValue(1 + i));
        }
        for (int32_t i = 0; i < 4; i++) {
            mins_[i] = static_cast<int64_t>(metaGm_.GetValue(4 + i));
            dims_[i] = static_cast<int64_t>(metaGm_.GetValue(8 + i));
        }
        for (int32_t k = 0; k < K_ && k < kMaxK; k++) {
            off_[k][0] = static_cast<int64_t>(metaGm_.GetValue(12 + 3 * k));
            off_[k][1] = static_cast<int64_t>(metaGm_.GetValue(13 + 3 * k));
            off_[k][2] = static_cast<int64_t>(metaGm_.GetValue(14 + 3 * k));
        }
        if (M_ > 0 && K_ > 0) {
            pipe_->InitBuffer(qRow_, 1, static_cast<uint32_t>(K_ * sizeof(int32_t)));
        }
        evtV_ = pipe_->AllocEventID<HardEvent::V_MTE2>();
        evtM_ = pipe_->AllocEventID<HardEvent::MTE2_MTE3>();
        evtO_ = pipe_->AllocEventID<HardEvent::MTE3_V>();
    }

    __aicore__ inline int64_t Key(int64_t b, int64_t x, int64_t y, int64_t z) const
    {
        return (((b - mins_[0]) * dims_[1] + (x - mins_[1])) * dims_[2] + (y - mins_[2]))
                   * dims_[3] + (z - mins_[3]);
    }

    __aicore__ inline int32_t Find(int64_t key)
    {
        const int64_t bucket = key >> shift_;
        if (bucket < 0 || bucket >= buckets_) {
            return -1;
        }
        int64_t lo = static_cast<int64_t>(startGm_.GetValue(bucket));
        int64_t hi = lo + static_cast<int64_t>(cntGm_.GetValue(bucket)) - 1;
        while (lo <= hi) {
            const int64_t mid = (lo + hi) >> 1;
            const int64_t v = rGm_.GetValue(mid);
            if (v == key) {
                return static_cast<int32_t>(mid);
            }
            if (v < key) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return -1;
    }

    __aicore__ inline void Process()
    {
        if (M_ <= 0 || K_ <= 0) {
            return;
        }
        if (dyn_) {
            const int32_t nReal = nGm_.GetValue(0);
            if (nReal <= 0) {
                return;
            }
            if (static_cast<int64_t>(nReal) < M_) {
                M_ = static_cast<int64_t>(nReal);
            }
        }
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        const int64_t per = (M_ + blockNum - 1) / blockNum;
        const int64_t g0 = static_cast<int64_t>(GetBlockIdx()) * per;
        int64_t g1 = g0 + per;
        if (g1 > M_) {
            g1 = M_;
        }

        DataCopyPadExtParams<int32_t> pp;
        pp.isPad = false;
        DataCopyExtParams whole;
        whole.blockCount = 1;
        whole.blockLen = static_cast<uint32_t>(K_ * sizeof(int32_t));
        whole.srcStride = 0;
        whole.dstStride = 0;

        for (int64_t m = g0; m < g1; m++) {
            const int64_t c4 = m * 4;
            const int64_t b = static_cast<int64_t>(qGm_.GetValue(c4));
            const int64_t xs = static_cast<int64_t>(qGm_.GetValue(c4 + 1)) * stride_[0];
            const int64_t ys = static_cast<int64_t>(qGm_.GetValue(c4 + 2)) * stride_[1];
            const int64_t zs = static_cast<int64_t>(qGm_.GetValue(c4 + 3)) * stride_[2];

            LocalTensor<int32_t> row = qRow_.AllocTensor<int32_t>();
            for (int64_t k = 0; k < K_; k++) {
                const int64_t key = Key(b, xs + off_[k][0], ys + off_[k][1],
                                        zs + off_[k][2]);
                row.SetValue(static_cast<int32_t>(k), Find(key));
            }
            PipeBarrier<PIPE_ALL>();   // 标量写 UB 之后再交给 MTE3
            DataCopyPad(outGm_[m * K_], row, whole);
            SetFlag<HardEvent::MTE3_V>(evtO_);
            WaitFlag<HardEvent::MTE3_V>(evtO_);
            qRow_.FreeTensor(row);
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
    GlobalTensor<int32_t> qGm_;
    GlobalTensor<int64_t> rGm_;
    GlobalTensor<int32_t> startGm_;
    GlobalTensor<int32_t> cntGm_;
    GlobalTensor<int32_t> metaGm_;
    GlobalTensor<int32_t> outGm_;
    GlobalTensor<int32_t> nGm_;
    bool dyn_ = false;
    int64_t M_ = 0;
    int64_t K_ = 0;
    int64_t buckets_ = 0;
    int64_t shift_ = 0;
    int64_t stride_[3] = {1, 1, 1};
    int64_t mins_[4] = {0, 0, 0, 0};
    int64_t dims_[4] = {1, 1, 1, 1};
    int64_t off_[kMaxK][3] = {{0, 0, 0}};
};

}  // namespace

extern "C" __global__ __aicore__ void spconv_query_v2(GM_ADDR qcoords, GM_ADDR rkeys,
                                                   GM_ADDR start, GM_ADDR cnt,
                                                   GM_ADDR meta, GM_ADDR nrow,
                                                   GM_ADDR out, GM_ADDR workspace,
                                                   GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SpconvQueryV2TilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KernelSpconvQueryV2 op(&pipe);
    op.Init(qcoords, rkeys, start, cnt, meta, nrow, out, &tilingData);
    op.Process();
    op.Release();
}
