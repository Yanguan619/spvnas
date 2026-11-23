/* -------------------------------------------------------------------------
 * SpconvKeyprepV3：把 torchsparse.backend._query_args 每帧 ~40 个 torch 小算子
 * （amin/amax、pack_keys 的 Slice/Cast/Mul/Add、hist 的 ZerosLike/ScatterAdd、
 *  AiCPU cumsum、cat meta）收进 1 个 kernel。
 *
 * 单个 kernel + 3 次软屏障（GM 每核一格 + 轮询，机制同 SDK 的 IBSet/IBWait）：
 *   P0  每核清零自己那段 cnt，并算 in/out coords 的局部 box（min/max）
 *   B0  屏障
 *   P1  每核各自归约 blockNum 组局部 box -> lo/dims/shift（全核一致）；core0 写 meta
 *   P2  每核算 rkeys[int64]（写回）并对 cnt 做 GM 原子加（SetAtomicAdd+DataCopyPad）
 *   B1  屏障
 *   P3  每核求 cnt 分段和 -> part[core]
 *   B2  屏障
 *   P4  每核把 part[0,core) 加总当起点，顺序独占扫描自己的桶段写 start
 *
 * 语义与 _query_args 逐位一致：
 *   lo = min(in_lo, cat([out_lo[:1], out_lo[1:]*stride + off_lo]))
 *   hi = max(in_hi, cat([out_hi[:1], out_hi[1:]*stride + off_hi]))
 *   dims = hi - lo + 1; shift = max(0, bitlen(total-1) - bitlen(nbk-1))
 *   rkeys = ((b*d1 + x)*d2 + y)*d3 + z   （coords - lo 之后）
 *   cnt[b] = #{i : rkeys[i] >> shift == b}; start = exclusive_cumsum(cnt)
 *   meta = [shift, stride(3), lo(4), dims(4), offs(KV*3)]
 *
 * 不用 SyncAll：本环境（aclnn 直调 + dav_c220）上 FFTS 全核 barrier 不可用
 * （不声明 batch mode 会挂，声明了又报 MTE 错）。另外 kernel 的 workspace 参数
 * 在本路径上不可写，跨核交换一律走 scratch 输出张量。
 * ------------------------------------------------------------------------- */
#include "kernel_operator.h"
#include "spconv_keyprep_v3_tiling.h"

using namespace AscendC;

namespace {
constexpr int64_t kSlot = 8;            // 32B / sizeof(int32)
constexpr int64_t kMaxBlocks = 64;
constexpr int64_t kTileRows = 2048;
constexpr int64_t kTileInts = 8192;
constexpr int64_t kBoxPart = 0;         // [64][8] lo(4)+hi(4)
constexpr int64_t kPart = 1024;         // [64][8] 每核分段和，值在 slot0
constexpr int64_t kBar = 1600;          // [3][64][8] 每个屏障阶段每核一格
constexpr int64_t kBarStride = kMaxBlocks * kSlot;
__aicore__ inline int32_t BitLen(int64_t v)
{
    int32_t n = 0;
    while (v > 0) {
        v >>= 1;
        ++n;
    }
    return n;
}

class KernelKeyprep {
public:
    __aicore__ inline KernelKeyprep(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR in_coords, GM_ADDR out_coords, GM_ADDR stride, GM_ADDR offs,
                                GM_ADDR nbk, GM_ADDR sizes, GM_ADDR rkeys, GM_ADDR cnt, GM_ADDR meta,
                                GM_ADDR scratch, GM_ADDR workspace, const SpconvKeyprepV3TilingData* t)
    {
        N_ = static_cast<int64_t>(t->N);
        dyn_ = (t->dynamic != 0);
        if (dyn_) {
            szGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sizes));
        }
        M_ = static_cast<int64_t>(t->M);
        kv_ = static_cast<int64_t>(t->kv);
        blockNum_ = static_cast<int64_t>(t->blockNum);
        if (blockNum_ < 1) {
            blockNum_ = 1;
        }
        blockIdx_ = static_cast<int64_t>(GetBlockIdx());
        stage_ = static_cast<int64_t>(t->stage);
        inGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(in_coords));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(out_coords));
        strideGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(stride));
        offsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(offs));
        nbkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(nbk));
        rkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(rkeys));

        cntGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cnt));
        metaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(meta));
        wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(scratch));
        nbk_ = nbkGm_.GetValue(0);
        pipe_->InitBuffer(addQ_, 8, 32);
        pipe_->InitBuffer(tileBuf_, kTileInts * static_cast<int64_t>(sizeof(int32_t)));

        pipe_->InitBuffer(rkBuf_, kTileRows * static_cast<int64_t>(sizeof(int64_t)));
        pipe_->InitBuffer(scBuf_, 512 * static_cast<int64_t>(sizeof(int32_t)));
        evMte2S_ = pipe_->AllocEventID<HardEvent::MTE2_S>();
        evSMte2_ = pipe_->AllocEventID<HardEvent::S_MTE2>();
        evSMte3_ = pipe_->AllocEventID<HardEvent::S_MTE3>();
        evMte3S_ = pipe_->AllocEventID<HardEvent::MTE3_S>();
        evVMte3_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
        evMte3V_ = pipe_->AllocEventID<HardEvent::MTE3_V>();
    }

    __aicore__ inline void Process()
    {
        if (dyn_) {
            // 容量捕获：真实 n / m 从 device 读，核内所有行循环按它们截断
            const int32_t nReal = szGm_.GetValue(0);
            const int32_t mReal = szGm_.GetValue(1);
            if (nReal < 0 || mReal < 0) {
                return;
            }
            if (static_cast<int64_t>(nReal) < N_) {
                N_ = static_cast<int64_t>(nReal);
            }
            if (static_cast<int64_t>(mReal) < M_) {
                M_ = static_cast<int64_t>(mReal);
            }
        }
        /* 行分段按 max(N, M) 切，两张表各按自己的长度截断 */
        const int64_t rowMax = (N_ > M_) ? N_ : M_;
        rowPer_ = ((rowMax + blockNum_ - 1) / blockNum_ + 7) & ~static_cast<int64_t>(7);
        const int64_t r0 = blockIdx_ * rowPer_;
        int64_t r1 = r0 + rowPer_;
        if (r1 > rowMax) {
            r1 = rowMax;
        }
        bktPer_ = ((nbk_ + blockNum_ - 1) / blockNum_ + 7) & ~static_cast<int64_t>(7);
        b0_ = blockIdx_ * bktPer_;
        b1_ = b0_ + bktPer_;
        if (b1_ > nbk_) {
            b1_ = nbk_;
        }

        ZeroCnt();                 /* P0：清自己那段 cnt */
        BoxPart(r0, r1);
        if (stage_ == 0) {
            return;
        }
        Barrier(0);                /* B0：cnt 清零 + box 局部量就绪 */
        ReduceBox();               /* P1：每核各自归约，结果全核一致 */
        if (stage_ == 1) {
            return;
        }
        RkeysHist(r0, r1);         /* P2：rkeys + 直方图 */
        if (stage_ == 2) {
            return;
        }
        StoreSlot(kBar + blockIdx_ * kSlot, 0);   /* 清屏障格子，下一轮从 0 开始 */
    }

private:
    /* ---------- GM 可见性：写侧落地 / 读侧失效（HCCL 同款序列） ---------- */
    __aicore__ inline void FlushWrite()
    {
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::ALL>();
    }
    __aicore__ inline void FlushRead(const GlobalTensor<int32_t>& gt)
    {
        PipeBarrier<PIPE_ALL>();
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(gt);
        DataSyncBarrier<MemDsbT::ALL>();
    }
    /* ---------- 软屏障：每核在自己的格子里写 core+1，轮询到所有核都到 ----------
     * 用 core+1 而不是 1，避免上一轮/垃圾数据碰巧满足条件；kernel 末尾各核把
     * 自己的格子清 0（此时已没人再读这段），下一轮从 0 开始。 */
    __aicore__ inline void Barrier(int32_t phase)
    {
        const int64_t base = kBar + phase * kBarStride;
        LocalTensor<int32_t> sb = scBuf_.Get<int32_t>();
        sb.SetValue(0, static_cast<int32_t>(blockIdx_ + 1));
        CopyOutScalar(wsGm_, base + blockIdx_ * kSlot, sb, kSlot);
        FlushWrite();
        for (int64_t spin = 0; spin < 100000000; ++spin) {
            FlushRead(wsGm_);
            CopyInInts(sb, wsGm_, base, blockNum_ * kSlot);
            int32_t arrived = 0;
            for (int64_t c = 0; c < blockNum_; ++c) {
                arrived += (sb.GetValue(c * kSlot) == static_cast<int32_t>(c + 1)) ? 1 : 0;
            }
            SetFlag<HardEvent::S_MTE2>(evSMte2_);
            WaitFlag<HardEvent::S_MTE2>(evSMte2_);
            if (arrived == blockNum_) {
                return;
            }
        }
    }
    /* ---------- 小工具 ---------- */
    __aicore__ inline void CopyInInts(const LocalTensor<int32_t>& tile, const GlobalTensor<int32_t>& gm,
                                      int64_t off, int64_t n)
    {
        DataCopyExtParams ext;
        ext.blockCount = 1;
        ext.blockLen = static_cast<uint32_t>(n * static_cast<int64_t>(sizeof(int32_t)));
        ext.srcStride = 0;
        ext.dstStride = 0;
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(tile, gm[off], ext, pad);
        SetFlag<HardEvent::MTE2_S>(evMte2S_);
        WaitFlag<HardEvent::MTE2_S>(evMte2S_);
    }
    /* 标量写出的 UB 数据 -> GM（S_MTE3 保证标量写落地）*/
    __aicore__ inline void CopyOutScalar(const GlobalTensor<int32_t>& gm, int64_t off,
                                         const LocalTensor<int32_t>& tile, int64_t n)
    {
        SetFlag<HardEvent::S_MTE3>(evSMte3_);
        WaitFlag<HardEvent::S_MTE3>(evSMte3_);
        DataCopyExtParams ext;
        ext.blockCount = 1;
        ext.blockLen = static_cast<uint32_t>(n * static_cast<int64_t>(sizeof(int32_t)));
        ext.srcStride = 0;
        ext.dstStride = 0;
        DataCopyPad(gm[off], tile, ext);
        SetFlag<HardEvent::MTE3_S>(evMte3S_);
        WaitFlag<HardEvent::MTE3_S>(evMte3S_);
    }
    __aicore__ inline void StoreSlot(int64_t gmSlot, int32_t val)
    {
        LocalTensor<int32_t> sb = scBuf_.Get<int32_t>();
        sb.SetValue(0, val);
        CopyOutScalar(wsGm_, gmSlot, sb, 1);
    }

    /* ---------- P0 ---------- */
    __aicore__ inline void ZeroCnt()
    {
        LocalTensor<int32_t> tile = tileBuf_.Get<int32_t>();
        Duplicate(tile, static_cast<int32_t>(0), static_cast<int32_t>(kTileInts));
        SetFlag<HardEvent::V_MTE3>(evVMte3_);
        WaitFlag<HardEvent::V_MTE3>(evVMte3_);
        DataCopyExtParams ext;
        ext.blockCount = 1;
        ext.srcStride = 0;
        ext.dstStride = 0;
        for (int64_t b = b0_; b < b1_; b += kTileInts) {
            int64_t n = b1_ - b;
            if (n > kTileInts) {
                n = kTileInts;
            }
            ext.blockLen = static_cast<uint32_t>(n * static_cast<int64_t>(sizeof(int32_t)));
            DataCopyPad(cntGm_[b], tile, ext);
            SetFlag<HardEvent::MTE3_V>(evMte3V_);
            WaitFlag<HardEvent::MTE3_V>(evMte3V_);   /* 每笔都等，tile 马上会被别的阶段覆盖 */
        }
    }

    __aicore__ inline void ScanRows(const GlobalTensor<int32_t>& gm, int64_t r0, int64_t r1,
                                    int32_t* mn, int32_t* mx)
    {
        LocalTensor<int32_t> tile = tileBuf_.Get<int32_t>();
        for (int64_t r = r0; r < r1; r += kTileRows) {
            int64_t n = r1 - r;
            if (n > kTileRows) {
                n = kTileRows;
            }
            CopyInInts(tile, gm, r * 4, n * 4);
            for (int64_t i = 0; i < n; ++i) {
                for (int32_t j = 0; j < 4; ++j) {
                    const int32_t v = tile.GetValue(i * 4 + j);
                    if (v < mn[j]) {
                        mn[j] = v;
                    }
                    if (v > mx[j]) {
                        mx[j] = v;
                    }
                }
            }
            SetFlag<HardEvent::S_MTE2>(evSMte2_);
            WaitFlag<HardEvent::S_MTE2>(evSMte2_);
        }
    }

    __aicore__ inline void BoxPart(int64_t r0, int64_t r1)
    {
        int32_t imn[4] = {2147483647, 2147483647, 2147483647, 2147483647};
        int32_t imx[4] = {-2147483648, -2147483648, -2147483648, -2147483648};
        int32_t omn[4] = {2147483647, 2147483647, 2147483647, 2147483647};
        int32_t omx[4] = {-2147483648, -2147483648, -2147483648, -2147483648};
        ScanRows(inGm_, r0, (r1 > N_) ? N_ : r1, imn, imx);
        ScanRows(outGm_, r0, (r1 > M_) ? M_ : r1, omn, omx);
        /* offsets 的 min/max（KV*3 个，先搬进 UB 再读标量）*/
        LocalTensor<int32_t> tile = tileBuf_.Get<int32_t>();
        int32_t ofl[3] = {2147483647, 2147483647, 2147483647};
        int32_t ofh[3] = {-2147483648, -2147483648, -2147483648};
        if (kv_ > 0) {
            CopyInInts(tile, offsGm_, 0, kv_ * 3);
            for (int64_t k = 0; k < kv_ * 3; ++k) {
                const int32_t v = tile.GetValue(k);
                const int32_t d = static_cast<int32_t>(k % 3);
                if (v < ofl[d]) {
                    ofl[d] = v;
                }
                if (v > ofh[d]) {
                    ofh[d] = v;
                }
            }
            SetFlag<HardEvent::S_MTE2>(evSMte2_);
            WaitFlag<HardEvent::S_MTE2>(evSMte2_);
        }
        int32_t box[8];
        box[0] = (imn[0] < omn[0]) ? imn[0] : omn[0];
        box[4] = (imx[0] > omx[0]) ? imx[0] : omx[0];
        for (int32_t d = 0; d < 3; ++d) {
            const int64_t st = strideGm_.GetValue(d);
            const int64_t ol = static_cast<int64_t>(omn[d + 1]) * st + ofl[d];
            const int64_t oh = static_cast<int64_t>(omx[d + 1]) * st + ofh[d];
            box[d + 1] = (static_cast<int64_t>(imn[d + 1]) < ol) ? imn[d + 1]
                                                               : static_cast<int32_t>(ol);
            box[d + 5] = (static_cast<int64_t>(imx[d + 1]) > oh) ? imx[d + 1]
                                                               : static_cast<int32_t>(oh);
        }
        LocalTensor<int32_t> sb = scBuf_.Get<int32_t>();
        for (int32_t i = 0; i < 8; ++i) {
            sb.SetValue(i, box[i]);
        }
        CopyOutScalar(wsGm_, kBoxPart + blockIdx_ * kSlot, sb, kSlot);
    }

    /* ---------- P1 ---------- */
    __aicore__ inline void ReduceBox()
    {
        LocalTensor<int32_t> tile = tileBuf_.Get<int32_t>();
        CopyInInts(tile, wsGm_, kBoxPart, blockNum_ * kSlot);
        int32_t lo[4] = {2147483647, 2147483647, 2147483647, 2147483647};
        int32_t hi[4] = {-2147483648, -2147483648, -2147483648, -2147483648};
        for (int64_t c = 0; c < blockNum_; ++c) {
            for (int32_t j = 0; j < 4; ++j) {
                const int32_t l = tile.GetValue(c * kSlot + j);
                const int32_t h = tile.GetValue(c * kSlot + 4 + j);
                if (l < lo[j]) {
                    lo[j] = l;
                }
                if (h > hi[j]) {
                    hi[j] = h;
                }
            }
        }
        SetFlag<HardEvent::S_MTE2>(evSMte2_);
        WaitFlag<HardEvent::S_MTE2>(evSMte2_);
        int32_t dims[4];
        int64_t total = 1;
        for (int32_t j = 0; j < 4; ++j) {
            dims[j] = hi[j] - lo[j] + 1;
            total *= static_cast<int64_t>(dims[j]);
        }
        for (int32_t j = 0; j < 4; ++j) {
            lo_[j] = lo[j];
            dims_[j] = dims[j];
        }
        const int32_t sh = BitLen(total - 1) - BitLen(nbk_ - 1);
        shift_ = (sh > 0) ? sh : 0;
        if (blockIdx_ != 0) {
            return;   /* 归约结果每核都算，meta 只由 core0 写一次 */
        }
        /* meta = [shift, stride(3), lo(4), dims(4), offs(KV*3)] */
        LocalTensor<int32_t> sb = scBuf_.Get<int32_t>();
        if (kv_ > 0) {
            CopyInInts(tile, offsGm_, 0, kv_ * 3);
            for (int64_t k = 0; k < kv_ * 3; ++k) {
                sb.SetValue(static_cast<int32_t>(12 + k), tile.GetValue(k));
            }
            SetFlag<HardEvent::S_MTE2>(evSMte2_);
            WaitFlag<HardEvent::S_MTE2>(evSMte2_);
        }
        sb.SetValue(0, shift_);
        for (int32_t d = 0; d < 3; ++d) {
            sb.SetValue(1 + d, strideGm_.GetValue(d));
        }
        for (int32_t j = 0; j < 4; ++j) {
            sb.SetValue(4 + j, lo[j]);
            sb.SetValue(8 + j, dims[j]);
        }
        CopyOutScalar(metaGm_, 0, sb, 12 + kv_ * 3);
    }

    /* ---------- P2 ---------- */
    __aicore__ inline void RkeysHist(int64_t r0, int64_t r1)
    {
        if (r1 > N_) {
            r1 = N_;
        }
        if (r1 <= r0) {
            return;
        }
        LocalTensor<int32_t> tile = tileBuf_.Get<int32_t>();
        LocalTensor<int64_t> rk = rkBuf_.Get<int64_t>();
        const int64_t lo0 = lo_[0];
        const int64_t lo1 = lo_[1];
        const int64_t lo2 = lo_[2];
        const int64_t lo3 = lo_[3];
        const int64_t d1 = dims_[1];
        const int64_t d2 = dims_[2];
        const int64_t d3 = dims_[3];
        const int64_t sh = shift_;
        DataCopyExtParams extA;      /* 原子 payload 专用，绝不和 rkeys 那笔共用 */
        extA.blockCount = 1;
        extA.blockLen = 32;
        extA.srcStride = 0;
        extA.dstStride = 0;
        for (int64_t r = r0; r < r1; r += kTileRows) {
            int64_t n = r1 - r;
            if (n > kTileRows) {
                n = kTileRows;
            }
            CopyInInts(tile, inGm_, r * 4, n * 4);
            SetAtomicAdd<int32_t>();
            for (int64_t i = 0; i < n; ++i) {
                const int64_t b = static_cast<int64_t>(tile.GetValue(i * 4 + 0)) - lo0;
                const int64_t x = static_cast<int64_t>(tile.GetValue(i * 4 + 1)) - lo1;
                const int64_t y = static_cast<int64_t>(tile.GetValue(i * 4 + 2)) - lo2;
                const int64_t z = static_cast<int64_t>(tile.GetValue(i * 4 + 3)) - lo3;
                const int64_t k = ((b * d1 + x) * d2 + y) * d3 + z;
                rk.SetValue(i, k);
                const int64_t bucket = k >> sh;
                const int64_t base = bucket & ~static_cast<int64_t>(7);
                const int32_t slot = static_cast<int32_t>(bucket & 7);
                /* 8 深队列把原子加的往返流水起来（每一笔仍然 SetAtomic 开关） */
                LocalTensor<int32_t> pay = addQ_.AllocTensor<int32_t>();
                Duplicate(pay, static_cast<int32_t>(0), 8);
                pay.SetValue(slot, static_cast<int32_t>(1));
                SetFlag<HardEvent::S_MTE3>(evSMte3_);
                WaitFlag<HardEvent::S_MTE3>(evSMte3_);
                addQ_.EnQue(pay);
                LocalTensor<int32_t> cur = addQ_.DeQue<int32_t>();
                DataCopyPad(cntGm_[base], cur, extA);
                addQ_.FreeTensor(cur);
            }
            SetAtomicNone();
            SetFlag<HardEvent::MTE3_S>(evMte3S_);
            WaitFlag<HardEvent::MTE3_S>(evMte3S_);
            SetFlag<HardEvent::S_MTE2>(evSMte2_);
            WaitFlag<HardEvent::S_MTE2>(evSMte2_);
            SetFlag<HardEvent::S_MTE3>(evSMte3_);
            WaitFlag<HardEvent::S_MTE3>(evSMte3_);
            DataCopyExtParams extRk;    /* 独立参数，避免动到原子 payload 的长度 */
            extRk.blockCount = 1;
            extRk.blockLen = static_cast<uint32_t>(n * static_cast<int64_t>(sizeof(int64_t)));
            extRk.srcStride = 0;
            extRk.dstStride = 0;
            DataCopyPad(rkGm_[r], rk, extRk);
            SetFlag<HardEvent::MTE3_S>(evMte3S_);
            WaitFlag<HardEvent::MTE3_S>(evMte3S_);
        }
        FlushWrite();   /* 原子加必须真正落地，下一个 kernel 才读得到 */
    }


private:
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> tileBuf_;

    TBuf<TPosition::VECCALC> rkBuf_;
    TBuf<TPosition::VECCALC> scBuf_;
    TQue<QuePosition::VECOUT, 8> addQ_;
    GlobalTensor<int32_t> inGm_;
    GlobalTensor<int32_t> outGm_;
    GlobalTensor<int32_t> strideGm_;
    GlobalTensor<int32_t> offsGm_;
    GlobalTensor<int64_t> nbkGm_;
    GlobalTensor<int64_t> rkGm_;

    GlobalTensor<int32_t> cntGm_;
    GlobalTensor<int32_t> metaGm_;
    GlobalTensor<int32_t> wsGm_;
    TEventID evMte2S_ = 0;
    TEventID evSMte2_ = 0;
    TEventID evSMte3_ = 0;
    TEventID evMte3S_ = 0;
    TEventID evVMte3_ = 0;
    TEventID evMte3V_ = 0;
    GlobalTensor<int32_t> szGm_;
    bool dyn_ = false;
    int64_t N_ = 0;
    int64_t M_ = 0;
    int64_t kv_ = 0;
    int64_t blockNum_ = 1;
    int64_t blockIdx_ = 0;
    int64_t nbk_ = 0;
    int64_t rowPer_ = 0;
    int64_t bktPer_ = 0;
    int64_t b0_ = 0;
    int64_t b1_ = 0;
    int64_t lo_[4] = {0, 0, 0, 0};
    int64_t dims_[4] = {1, 1, 1, 1};
    int64_t shift_ = 0;
    int64_t stage_ = 5;
};
}  // namespace

/* 参数顺序 = OpDef：输入(in,out,stride,offs,nbk,sizes) 在前，输出(rkeys,cnt,meta,scratch) 在后 */
extern "C" __global__ __aicore__ void spconv_keyprep_v3(
    GM_ADDR in_coords, GM_ADDR out_coords, GM_ADDR stride, GM_ADDR offs, GM_ADDR nbk,
    GM_ADDR sizes, GM_ADDR rkeys, GM_ADDR cnt, GM_ADDR meta, GM_ADDR scratch,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SpconvKeyprepV3TilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KernelKeyprep op(&pipe);
    op.Init(in_coords, out_coords, stride, offs, nbk, sizes, rkeys, cnt, meta, scratch, workspace, &tilingData);
    op.Process();
}





