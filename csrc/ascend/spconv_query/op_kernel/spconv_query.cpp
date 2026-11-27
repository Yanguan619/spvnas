/* -------------------------------------------------------------------------
 * SpconvQuery：把「拼 [M,KV,4] 查询张量 → pack_keys → 二分」三步合成一个 AIV 核。
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
#include "spconv_query_tiling.h"

using namespace AscendC;

namespace {

constexpr int32_t kMaxK = 32;
constexpr int64_t kRowsPerBlk = 8;   // DataCopyPad 对齐所需：8 行 × K_ × 4B ≡ 0 (mod 32)

class KernelSpconvQuery {
public:
    __aicore__ inline KernelSpconvQuery(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR qcoords, GM_ADDR rkeys, GM_ADDR start,
                                GM_ADDR cnt, GM_ADDR meta, GM_ADDR out,
                                const SpconvQueryTilingData* tiling)
    {
        M_ = static_cast<int64_t>(tiling->M);
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
            // 8 行一组写回：8*K_*4 字节必为 32B 的倍数，满足 DataCopyPad 目的地址对齐
            pipe_->InitBuffer(qRow_, 1, static_cast<uint32_t>(kRowsPerBlk * K_ * sizeof(int32_t)));
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
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        // 每核起点必须是 8 的倍数，否则 DataCopyPad 的 GM 目的地址仍会错位
        // （per 按 kRowsPerBlk 向上取整，保证 GetBlockIdx()*per % 8 == 0）。
        const int64_t perRaw = (M_ + blockNum - 1) / blockNum;
        const int64_t per = (perRaw + kRowsPerBlk - 1) / kRowsPerBlk * kRowsPerBlk;
        const int64_t g0 = static_cast<int64_t>(GetBlockIdx()) * per;
        int64_t g1 = g0 + per;
        if (g1 > M_) {
            g1 = M_;
        }

        // DataCopyPad 的 GM 目的地址必须 32B 对齐：单行 K_=27 时行距仅 108 B，逐行写
        // 会构成未对齐 GM 写（910B 上偶发 507035 aivec / MTE 非法地址，实测 10~1500
        // 帧不等必现）。这里改为「8 行一组」整块写：8*K_*4 字节恒为 32B 倍数，
        // 目的地址 m0*K_*4 也随 m0 按 8 递增而对齐。输出行数在 host/binding 侧补齐到
        // 8 的倍数，torch 侧再用 narrow 切回 M 行（前缀切片，仍连续）。
        DataCopyPadExtParams<int32_t> pp;
        pp.isPad = false;
        DataCopyExtParams whole;
        whole.blockCount = 1;
        whole.srcStride = 0;
        whole.dstStride = 0;

        for (int64_t base = g0; base < g1; base += kRowsPerBlk) {
            int64_t rows = g1 - base;
            if (rows > kRowsPerBlk) {
                rows = kRowsPerBlk;
            }
            LocalTensor<int32_t> buf = qRow_.AllocTensor<int32_t>();
            for (int64_t r = 0; r < kRowsPerBlk; r++) {
                const int64_t m = base + r;
                for (int64_t k = 0; k < K_; k++) {
                    int32_t v = -1;
                    if (r < rows) {
                        const int64_t c4 = m * 4;
                        const int64_t b = static_cast<int64_t>(qGm_.GetValue(c4));
                        const int64_t xs = static_cast<int64_t>(qGm_.GetValue(c4 + 1)) * stride_[0];
                        const int64_t ys = static_cast<int64_t>(qGm_.GetValue(c4 + 2)) * stride_[1];
                        const int64_t zs = static_cast<int64_t>(qGm_.GetValue(c4 + 3)) * stride_[2];
                        v = Find(Key(b, xs + off_[k][0], ys + off_[k][1], zs + off_[k][2]));
                    }
                    buf.SetValue(static_cast<int32_t>(r * K_ + k), v);
                }
            }
            PipeBarrier<PIPE_ALL>();   // 标量写 UB 之后再交给 MTE3
            whole.blockLen = static_cast<uint32_t>(kRowsPerBlk * K_ * sizeof(int32_t));
            DataCopyPad(outGm_[base * K_], buf, whole);
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
    GlobalTensor<int32_t> qGm_;
    GlobalTensor<int64_t> rGm_;
    GlobalTensor<int32_t> startGm_;
    GlobalTensor<int32_t> cntGm_;
    GlobalTensor<int32_t> metaGm_;
    GlobalTensor<int32_t> outGm_;
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

extern "C" __global__ __aicore__ void spconv_query(GM_ADDR qcoords, GM_ADDR rkeys,
                                                   GM_ADDR start, GM_ADDR cnt,
                                                   GM_ADDR meta, GM_ADDR out,
                                                   GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SpconvQueryTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KernelSpconvQuery op(&pipe);
    op.Init(qcoords, rkeys, start, cnt, meta, out, &tilingData);
    op.Process();
    op.Release();
}
