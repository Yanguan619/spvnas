/* -------------------------------------------------------------------------
 * SpconvKeyprepV3Prefix：SpconvKeyprepV3 的第二段，独立 launch。
 *
 * 拆两段的原因：cnt 是 SetAtomicAdd 累加出来的，同一个 kernel 里原子加的完成
 * 没有被 MTE3 事件排住，隔屏障读回来会少读几笔；kernel 边界天然给出可见性。
 *
 * 输入 cnt[nbk]（上一段的直方图），输出 start[nbk] = cnt 的独占前缀和：
 *   每核先把自己桶段之前的 cnt 用向量 reduce 加出来当起点（最多读 2MB，但
 *   Cast+ReduceSum 很快），再顺序独占扫描自己的桶段写 start。
 * ------------------------------------------------------------------------- */
#include "kernel_operator.h"
#include "spconv_keyprep_v3_tiling.h"

using namespace AscendC;

namespace {
constexpr int64_t kTileInts = 2048;   // 2048 * 4B = 8KB

class KernelPrefix {
public:
    __aicore__ inline KernelPrefix(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR cnt, GM_ADDR start, const SpconvKeyprepV3TilingData* t)
    {
        nbk_ = static_cast<int64_t>(t->nbk);
        blockNum_ = static_cast<int64_t>(t->blockNum);
        if (blockNum_ < 1) {
            blockNum_ = 1;
        }
        blockIdx_ = static_cast<int64_t>(GetBlockIdx());
        cntGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cnt));
        startGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(start));
        pipe_->InitBuffer(tileBuf_, kTileInts * static_cast<int64_t>(sizeof(int32_t)));
        pipe_->InitBuffer(fBuf_, kTileInts * static_cast<int64_t>(sizeof(float)));
        pipe_->InitBuffer(redBuf_, kTileInts * static_cast<int64_t>(sizeof(float)));
        evMte2S_ = pipe_->AllocEventID<HardEvent::MTE2_S>();
        evSMte2_ = pipe_->AllocEventID<HardEvent::S_MTE2>();
        evSMte3_ = pipe_->AllocEventID<HardEvent::S_MTE3>();
        evMte3S_ = pipe_->AllocEventID<HardEvent::MTE3_S>();
    }

    __aicore__ inline void Process()
    {
        const int64_t per = ((nbk_ + blockNum_ - 1) / blockNum_ + 7) & ~static_cast<int64_t>(7);
        const int64_t b0 = blockIdx_ * per;
        if (b0 >= nbk_) {
            return;
        }
        int64_t b1 = b0 + per;
        if (b1 > nbk_) {
            b1 = nbk_;
        }
        LocalTensor<int32_t> tile = tileBuf_.Get<int32_t>();
        /* 起点 = cnt[0, b0) 之和（向量 reduce，最多的核读 2MB） */
        int32_t run = 0;
        for (int64_t b = 0; b < b0; b += kTileInts) {
            int64_t n = b0 - b;
            if (n > kTileInts) {
                n = kTileInts;
            }
            const int32_t nv = static_cast<int32_t>(((n + 7) / 8) * 8);
            CopyIn(tile, b, n);
            for (int32_t i = static_cast<int32_t>(n); i < nv; ++i) {
                tile.SetValue(i, static_cast<int32_t>(0));
            }
            PipeBarrier<PIPE_ALL>();
            Cast(fBuf_.Get<float>(), tile, RoundMode::CAST_NONE, nv);
            LocalTensor<float> rb = redBuf_.Get<float>();
            ReduceSum<float>(rb, fBuf_.Get<float>(), rb[8], nv);
            PipeBarrier<PIPE_ALL>();
            run += static_cast<int32_t>(rb.GetValue(0));
            SetFlag<HardEvent::S_MTE2>(evSMte2_);
            WaitFlag<HardEvent::S_MTE2>(evSMte2_);
        }
        /* 顺序独占扫描自己的桶段：原地把 cnt 改成 start */
        for (int64_t b = b0; b < b1; b += kTileInts) {
            int64_t n = b1 - b;
            if (n > kTileInts) {
                n = kTileInts;
            }
            CopyIn(tile, b, n);
            for (int64_t i = 0; i < n; ++i) {
                const int32_t v = tile.GetValue(static_cast<int32_t>(i));
                tile.SetValue(static_cast<int32_t>(i), run);
                run += v;
            }
            SetFlag<HardEvent::S_MTE2>(evSMte2_);
            WaitFlag<HardEvent::S_MTE2>(evSMte2_);
            SetFlag<HardEvent::S_MTE3>(evSMte3_);
            WaitFlag<HardEvent::S_MTE3>(evSMte3_);
            DataCopyExtParams ext;
            ext.blockCount = 1;
            ext.blockLen = static_cast<uint32_t>(n * static_cast<int64_t>(sizeof(int32_t)));
            ext.srcStride = 0;
            ext.dstStride = 0;
            DataCopyPad(startGm_[b], tile, ext);
            SetFlag<HardEvent::MTE3_S>(evMte3S_);
            WaitFlag<HardEvent::MTE3_S>(evMte3S_);
        }
    }

private:
    __aicore__ inline void CopyIn(const LocalTensor<int32_t>& tile, int64_t off, int64_t n)
    {
        DataCopyExtParams ext;
        ext.blockCount = 1;
        ext.blockLen = static_cast<uint32_t>(n * static_cast<int64_t>(sizeof(int32_t)));
        ext.srcStride = 0;
        ext.dstStride = 0;
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(tile, cntGm_[off], ext, pad);
        SetFlag<HardEvent::MTE2_S>(evMte2S_);
        WaitFlag<HardEvent::MTE2_S>(evMte2S_);
    }

private:
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> tileBuf_;
    TBuf<TPosition::VECCALC> fBuf_;
    TBuf<TPosition::VECCALC> redBuf_;
    GlobalTensor<int32_t> cntGm_;
    GlobalTensor<int32_t> startGm_;
    TEventID evMte2S_ = 0;
    TEventID evSMte2_ = 0;
    TEventID evSMte3_ = 0;
    TEventID evMte3S_ = 0;
    int64_t nbk_ = 0;
    int64_t blockNum_ = 1;
    int64_t blockIdx_ = 0;
};
}  // namespace

/* 参数顺序：输入(cnt) -> 输出(start) -> workspace, tiling */
extern "C" __global__ __aicore__ void spconv_keyprep_v3_prefix(
    GM_ADDR cnt, GM_ADDR start, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SpconvKeyprepV3TilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KernelPrefix op(&pipe);
    op.Init(cnt, start, &tilingData);
    op.Process();
}
