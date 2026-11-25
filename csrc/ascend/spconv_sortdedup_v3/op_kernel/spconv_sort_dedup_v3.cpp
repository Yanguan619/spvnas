/* -------------------------------------------------------------------------
 * SpconvSortDedupV3：坐标行「字典序排序 + 去重」整条链路收进 1 个 kernel
 *
 *   in : coords[N, 4] int32（任意顺序，非负）
 *   out: res[N, 8] int32（前 u 行有效，行宽补到 8 个 int32 = 32B 对齐）
 *        u[4] int64     （u[0] = 去重后的行数；u[1] = 失败标记（0 成功 / -1 失败）；
 *                        u[2] = core0 观测到的最大屏障自旋次数（诊断）；u[3] = 0）
 *
 * 语义 = torch.unique(coords, dim=0)：按 (b,x,y,z) 字典序升序的、去重的行，
 * 行内容与行顺序逐位一致。重复行内容相同，所以同一个 key 内部谁先谁后不影响输出。
 *
 * 与 V1 的唯一区别：**跨核软屏障有界**。
 *
 *   V1 的软屏障 spin 上限是 1e8 次且超限后「静默继续」。而 radix 的每一趟都要
 *   全部 block 到齐：屏障要求 blockNum 个 block 同时驻留。blockNum 取的是整片
 *   AIV 数（910B4 = 40），一个余量都没有 —— 只要有 1 个 AIV 被别的 kernel 占着
 *   （同进程别的 stream、同芯片别的进程），到齐的 block 就一直自旋，实测每次
 *   自旋迭代 ~2 µs，1e8 次 = 200 s 以上；实测症状是一帧从 137 ms 变 71540 ms
 *   （见 benchmark/reports/spconv_sortdedup_v1_bughunt.md）。
 *
 *   V2 把它改成：自旋到 spinLimit 次仍不齐 -> 写 u[1] = -1 并立刻返回，
 *   一行结果都不写。host 侧看到 u[1] 直接抛错，调用方已有的 try/except 会回退到
 *   torch 路径（~3.4 ms/次），代价有界且可观测。正常路径零成本（u[1]/u[2] 和
 *   u[0] 用同一次 32B D2H 读回）。
 *
 * 结构（单 launch，AIV-only，软屏障跨核同步）：
 *   P0  每核清自己的屏障格子 + 自己那段的 4 列 max -> scratch box[core]
 *   B0
 *   P1  每核归约 blockNum 组 box -> dims，把 (b,x,y,z) 压成紧致位域 key：
 *         widths[c] = bitlen(dims[c]-1)，key = b<<s0 | x<<s1 | y<<s2 | z
 *       位域按字典序排，所以「key 数值升序」= 行字典序升序；解码只需移位/掩码。
 *   P2  每核算自己那段的 key 写 keys[0]（每元素占 32B，只用前 8B）
 *   LSD 基数排序，每趟 8 bit（256 桶），npass = ceil(bits/8)：
 *      H  每核对自己那段做直方图 -> scratch hist[core][256]
 *      B
 *      Bs 每核算出 base[256] = radix(全局 digit 前缀) + hist[0..core) 的向量和
 *      S  每核按输入顺序把自己那段的 key 散射到 keys[dst]，落点 =
 *         base[digit] + 段内同 digit 的序号（稳定，故 LSD 正确）
 *      B
 *   最后：每核数自己段的 keep 数 -> B -> 前缀和 -> 每核把去重行攒 512 行
 *   （16 KB）一次 DataCopyPad 写到 res[off + local]
 *
 * 三个硬约束（本平台实测）：
 *   1) UB -> GM 的 DataCopyPad 目的地址必须 32B 对齐，所以中间数组按「每元素
 *      32B」布局，散射写回是每个元素一笔 32B 的 DMA（8B 写会 MTE 报非法地址）。
 *   2) 标量 SetValue 写 GM 不落地，跨核交换一律 UB + DataCopyPad。
 *   3) UB 标量访问约 15 ns/次，所以每个元素每趟只碰 1 个 key（2 个字），
 *      不要把整行 4 列带进排序主循环。
 * ------------------------------------------------------------------------- */
#include "kernel_operator.h"
#include "spconv_sort_dedup_v3_tiling.h"
using namespace AscendC;
namespace {
constexpr int64_t kSlot = 8;             // scratch 每核一格 = 8 个 int32 = 32B
constexpr int64_t kMaxBlocks = 64;
constexpr int64_t kBins = 256;           // 8 bit 一趟
constexpr int64_t kBinBits = 8;
constexpr int64_t kPhases = 20;          // 屏障阶段数上限
constexpr int64_t kBarStride = kMaxBlocks * kSlot;
constexpr int64_t OFF_BAR = 0;                              // [kPhases][64][8]
constexpr int64_t OFF_BOX = kPhases * kBarStride;           // [64][8]
constexpr int64_t OFF_HIST = OFF_BOX + kMaxBlocks * kSlot;  // [64][256]
constexpr int64_t OFF_KEEPS = OFF_HIST + kMaxBlocks * kBins;  // [64][8]
constexpr int64_t kRowInts = 8;          // 每元素 8 个 int32 = 32B（只用前 2 个）
constexpr int64_t kTileRows = 1024;      // 每 tile 32 KB
constexpr int64_t kOutRows = 512;        // 512 行 × 32B = 16 KB 一次 DMA
constexpr int64_t kDefaultSpinLimit = 4000;   // 正常屏障只需几十次；这是「有 block 分不到 AIV」的兜底

__aicore__ inline int64_t BitLen(int64_t v)
{
    int64_t n = 0;
    while (v > 0) {
        v >>= 1;
        ++n;
    }
    return n;
}

class KernelSpconvSortDedupV3 {
public:
    __aicore__ inline KernelSpconvSortDedupV3(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR coords, GM_ADDR res, GM_ADDR u, GM_ADDR keys,
                                GM_ADDR scratch, const SpconvSortDedupV3TilingData* t)
    {
        N_ = static_cast<int64_t>(t->N);
        blockNum_ = static_cast<int64_t>(t->blockNum);
        if (blockNum_ < 1) {
            blockNum_ = 1;
        }
        if (blockNum_ > kMaxBlocks) {
            blockNum_ = kMaxBlocks;
        }
        stage_ = static_cast<int64_t>(t->stage);
        mode_ = static_cast<int64_t>(t->mode);
        spinLimit_ = static_cast<int64_t>(t->spinLimit);
        if (spinLimit_ <= 0) {
            spinLimit_ = kDefaultSpinLimit;
        }
        blockIdx_ = static_cast<int64_t>(GetBlockIdx());
        coordGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(coords));
        resGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(res));
        uGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(u));
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(keys));
        wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(scratch));

        pipe_->InitBuffer(coordBuf_, kTileRows * 4 * sizeof(int32_t));
        pipe_->InitBuffer(keyBuf_, kTileRows * kRowInts * sizeof(int32_t));
        pipe_->InitBuffer(histBuf_, kBins * sizeof(int32_t));
        pipe_->InitBuffer(baseBuf_, kBins * sizeof(int32_t));
        pipe_->InitBuffer(histAllBuf_, kMaxBlocks * kBins * sizeof(int32_t));
        pipe_->InitBuffer(outBuf_, kOutRows * kRowInts * sizeof(int32_t));
        pipe_->InitBuffer(scBuf_, kMaxBlocks * kSlot * sizeof(int32_t));
        pipe_->InitBuffer(stgBuf_, kSlot * sizeof(int32_t));

        evMte2S_ = pipe_->AllocEventID<HardEvent::MTE2_S>();
        evMte2V_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
        evSMte2_ = pipe_->AllocEventID<HardEvent::S_MTE2>();
        evSMte3_ = pipe_->AllocEventID<HardEvent::S_MTE3>();
        evMte3S_ = pipe_->AllocEventID<HardEvent::MTE3_S>();
        evMte3V_ = pipe_->AllocEventID<HardEvent::MTE3_V>();
    }

    __aicore__ inline void Process()
    {
        rowPer_ = (N_ + blockNum_ - 1) / blockNum_;
        if (rowPer_ < 1) {
            rowPer_ = 1;
        }
        r0_ = blockIdx_ * rowPer_;
        r1_ = r0_ + rowPer_;
        if (r1_ > N_) {
            r1_ = N_;
        }
        if (r0_ > N_) {
            r0_ = N_;
        }
        ClearBarCells();
        BoxPart();
        if (!Barrier(0)) {
            Fail();
            return;
        }
        Meta();
        if (stage_ == 0) {
            return;
        }
        if (N_ > 0) {
            PackKeys(0);
        }
        if (npass_ == 0) {
            /* 无排序趟时 CountKeeps 仍要跨核读上一核的尾元素 */
            if (!Barrier(kPhases - 1)) {
                Fail();
                return;
            }
        }
        if (stage_ == 1) {
            return;
        }
        for (int64_t p = 0; p < npass_; ++p) {
            const int64_t src = (p & 1) ? 1 : 0;
            const int64_t dst = 1 - src;
            HistPass(p, src);
            if (stage_ == 101 + 10 * p) {
                return;
            }
            if (!Barrier(1 + 2 * p)) {
                Fail();
                return;
            }
            BasePrefix();
            if (stage_ == 102 + 10 * p) {
                return;
            }
            ScatterPass(p, src, dst);
            if (stage_ == 103 + 10 * p) {
                return;
            }
            if (!Barrier(2 + 2 * p)) {
                Fail();
                return;
            }
        }
        const int64_t fin = (npass_ & 1) ? 1 : 0;
        CountKeeps(fin);
        if (stage_ == 200) {
            return;
        }
        if (!Barrier(1 + 2 * npass_)) {
            Fail();
            return;
        }
        WriteOut(fin);
    }

    __aicore__ inline void Release()
    {
        pipe_->ReleaseEventID<HardEvent::MTE2_S>(evMte2S_);
        pipe_->ReleaseEventID<HardEvent::MTE2_V>(evMte2V_);
        pipe_->ReleaseEventID<HardEvent::S_MTE2>(evSMte2_);
        pipe_->ReleaseEventID<HardEvent::S_MTE3>(evSMte3_);
        pipe_->ReleaseEventID<HardEvent::MTE3_S>(evMte3S_);
        pipe_->ReleaseEventID<HardEvent::MTE3_V>(evMte3V_);
    }

private:
    /* ---------------- GM 可见性（HCCL 同款序列） ---------------- */
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
    /* 软屏障：每核在自己的格子里写 core+1，轮询到全部到齐（同 keyprep）。
     * 返回 false = 有 block 在 spinLimit 次内没到齐（AIV 被别的 kernel 占住），
     * 调用方必须立刻放弃本次调用，不能带着没到齐的直方图继续散射。 */
    __aicore__ inline bool Barrier(int32_t phase)
    {
        const int64_t base = OFF_BAR + phase * kBarStride;
        LocalTensor<int32_t> sb = scBuf_.Get<int32_t>();
        sb.SetValue(0, static_cast<int32_t>(blockIdx_ + 1));
        CopyOutInts(wsGm_, base + blockIdx_ * kSlot, sb, kSlot);
        FlushWrite();
        for (int64_t spin = 0; spin < spinLimit_; ++spin) {
            FlushRead(wsGm_);
            CopyInInts(sb, wsGm_, base, blockNum_ * kSlot);
            int32_t arrived = 0;
            for (int64_t c = 0; c < blockNum_; ++c) {
                arrived += (sb.GetValue(c * kSlot) == static_cast<int32_t>(c + 1)) ? 1 : 0;
            }
            SetFlag<HardEvent::S_MTE2>(evSMte2_);
            WaitFlag<HardEvent::S_MTE2>(evSMte2_);
            if (arrived == blockNum_) {
                if (spin > maxSpin_) {
                    maxSpin_ = spin;
                }
                return true;
            }
        }
        return false;
    }
    __aicore__ inline void ClearBarCells()
    {
        LocalTensor<int32_t> sb = scBuf_.Get<int32_t>();
        for (int32_t c = 0; c < static_cast<int32_t>(kSlot); ++c) {
            sb.SetValue(c, 0);
        }
        for (int32_t ph = 0; ph < static_cast<int32_t>(kPhases); ++ph) {
            CopyOutInts(wsGm_, OFF_BAR + ph * kBarStride + blockIdx_ * kSlot, sb, kSlot);
        }
    }
    /* 屏障超时：写 u[1] = -1 后整核退出（res/keys 一行都不写，host 侧会拒收） */
    __aicore__ inline void Fail()
    {
        LocalTensor<int64_t> ub = stgBuf_.Get<int64_t>();
        ub.SetValue(0, 0);
        ub.SetValue(1, -1);
        ub.SetValue(2, static_cast<int64_t>(spinLimit_));
        ub.SetValue(3, 0);
        SetFlag<HardEvent::S_MTE3>(evSMte3_);
        WaitFlag<HardEvent::S_MTE3>(evSMte3_);
        DataCopyExtParams ep;
        ep.blockCount = 1;
        ep.blockLen = 32;
        ep.srcStride = 0;
        ep.dstStride = 0;
        DataCopyPad(uGm_, ub, ep);
        SetFlag<HardEvent::MTE3_V>(evMte3V_);
        WaitFlag<HardEvent::MTE3_V>(evMte3V_);
    }
    /* ---------------- 小工具 ---------------- */
    __aicore__ inline void CopyInInts(const LocalTensor<int32_t>& tile,
                                      const GlobalTensor<int32_t>& gm, int64_t off, int64_t n)
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
    __aicore__ inline void CopyOutInts(const GlobalTensor<int32_t>& gm, int64_t off,
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
    /* 32B 的一个元素从 UB 的第 src 格搬到 GM 的第 dst 格 */
    __aicore__ inline void CopyElemOut(const LocalTensor<int32_t>& tile, int64_t src,
                                       int64_t dstBase, int64_t dst)
    {
        DataCopyExtParams ep;
        ep.blockCount = 1;
        ep.blockLen = static_cast<uint32_t>(kRowInts * sizeof(int32_t));
        ep.srcStride = 0;
        ep.dstStride = 0;
        if (mode_ >= 2) {
            DataCopy(keyGm_[dstBase + dst * kRowInts], tile[static_cast<int32_t>(src * kRowInts)],
                     static_cast<uint32_t>(kRowInts));
            return;
        }
        DataCopyPad(keyGm_[dstBase + dst * kRowInts], tile[static_cast<int32_t>(src * kRowInts)], ep);
    }
    __aicore__ inline void TileKeys(const LocalTensor<int32_t>& tile, int64_t i, int64_t n)
    {
        DataCopyExtParams ext;
        ext.blockCount = 1;
        ext.blockLen = static_cast<uint32_t>(n * kRowInts * sizeof(int32_t));
        ext.srcStride = 0;
        ext.dstStride = 0;
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(tile, keyGm_[i * kRowInts], ext, pad);
        SetFlag<HardEvent::MTE2_S>(evMte2S_);
        WaitFlag<HardEvent::MTE2_S>(evMte2S_);
    }
    __aicore__ inline int64_t LoadKey(const LocalTensor<int32_t>& tile, int32_t o)
    {
        int64_t lo = static_cast<int64_t>(static_cast<uint32_t>(tile.GetValue(o)));
        int64_t hi = static_cast<int64_t>(static_cast<uint32_t>(tile.GetValue(o + 1)));
        return (hi << 32) | lo;
    }
    __aicore__ inline int64_t LoadKeyGm(int64_t idx)
    {
        int64_t lo = static_cast<int64_t>(static_cast<uint32_t>(keyGm_.GetValue(idx)));
        int64_t hi = static_cast<int64_t>(static_cast<uint32_t>(keyGm_.GetValue(idx + 1)));
        return (hi << 32) | lo;
    }

    /* ---------------- P0：4 列 max ---------------- */
    __aicore__ inline void BoxPart()
    {
        int64_t mx[4] = {0, 0, 0, 0};
        LocalTensor<int32_t> ct = coordBuf_.Get<int32_t>();
        for (int64_t i = r0_; i < r1_; i += kTileRows) {
            int64_t n = r1_ - i;
            if (n > kTileRows) {
                n = kTileRows;
            }
            CopyInInts(ct, coordGm_, i * 4, n * 4);
            for (int64_t j = 0; j < n; ++j) {
                for (int32_t c = 0; c < 4; ++c) {
                    int64_t v = static_cast<int64_t>(ct.GetValue(static_cast<int32_t>(j * 4 + c)));
                    if (v > mx[c]) {
                        mx[c] = v;
                    }
                }
            }
        }
        LocalTensor<int32_t> sb = scBuf_.Get<int32_t>();
        for (int32_t c = 0; c < 4; ++c) {
            sb.SetValue(c, static_cast<int32_t>(mx[c]));
        }
        for (int32_t c = 4; c < static_cast<int32_t>(kSlot); ++c) {
            sb.SetValue(c, 0);
        }
        CopyOutInts(wsGm_, OFF_BOX + blockIdx_ * kSlot, sb, kSlot);
    }
    /* ---------------- P1：dims / 位宽 / 位移 ---------------- */
    __aicore__ inline void Meta()
    {
        int64_t dims[4] = {0, 0, 0, 0};
        for (int32_t c = 0; c < 4; ++c) {
            int64_t mx = 0;
            for (int64_t b = 0; b < blockNum_; ++b) {
                int32_t v = wsGm_.GetValue(OFF_BOX + b * kSlot + c);
                if (static_cast<int64_t>(v) > mx) {
                    mx = static_cast<int64_t>(v);
                }
            }
            dims[c] = mx + 1;
        }
        for (int32_t c = 0; c < 4; ++c) {
            w_[c] = BitLen(dims[c] - 1);
            m_[c] = (w_[c] <= 0) ? 0 : ((static_cast<int64_t>(1) << w_[c]) - 1);
        }
        sh_[3] = 0;
        sh_[2] = w_[3];
        sh_[1] = sh_[2] + w_[2];
        sh_[0] = sh_[1] + w_[1];
        bits_ = sh_[0] + w_[0];
        npass_ = (bits_ + kBinBits - 1) / kBinBits;
        if (npass_ > 8) {
            npass_ = 8;
        }
    }
    __aicore__ inline int64_t PackKey(int64_t b, int64_t x, int64_t y, int64_t z)
    {
        return (b << sh_[0]) | (x << sh_[1]) | (y << sh_[2]) | (z << sh_[3]);
    }
    /* ---------------- P2：key 写 keys[0] ---------------- */
    __aicore__ inline void PackKeys(int64_t buf)
    {
        LocalTensor<int32_t> ct = coordBuf_.Get<int32_t>();
        LocalTensor<int32_t> kt = keyBuf_.Get<int32_t>();
        const int64_t base = buf * N_ * kRowInts;
        for (int64_t i = r0_; i < r1_; i += kTileRows) {
            int64_t n = r1_ - i;
            if (n > kTileRows) {
                n = kTileRows;
            }
            CopyInInts(ct, coordGm_, i * 4, n * 4);
            for (int64_t j = 0; j < n; ++j) {
                const int32_t o = static_cast<int32_t>(j * kRowInts);
                int64_t k = PackKey(static_cast<int64_t>(ct.GetValue(static_cast<int32_t>(j * 4))),
                                    static_cast<int64_t>(ct.GetValue(static_cast<int32_t>(j * 4 + 1))),
                                    static_cast<int64_t>(ct.GetValue(static_cast<int32_t>(j * 4 + 2))),
                                    static_cast<int64_t>(ct.GetValue(static_cast<int32_t>(j * 4 + 3))));
                kt.SetValue(o, static_cast<int32_t>(static_cast<uint32_t>(k & 0xFFFFFFFFLL)));
                kt.SetValue(o + 1, static_cast<int32_t>(static_cast<uint32_t>(k >> 32)));
            }
            SetFlag<HardEvent::S_MTE3>(evSMte3_);
            WaitFlag<HardEvent::S_MTE3>(evSMte3_);
            DataCopyExtParams ext;
            ext.blockCount = 1;
            ext.blockLen = static_cast<uint32_t>(n * kRowInts * sizeof(int32_t));
            ext.srcStride = 0;
            ext.dstStride = 0;
            DataCopyPad(keyGm_[base + i * kRowInts], kt, ext);
            SetFlag<HardEvent::MTE3_V>(evMte3V_);
            WaitFlag<HardEvent::MTE3_V>(evMte3V_);
        }
    }
    /* ---------------- H：直方图 ---------------- */
    __aicore__ inline void HistPass(int64_t p, int64_t buf)
    {
        LocalTensor<int32_t> hb = histBuf_.Get<int32_t>();
        Duplicate(hb, static_cast<int32_t>(0), static_cast<int32_t>(kBins));
        PipeBarrier<PIPE_ALL>();
        const int64_t shift = p * kBinBits;
        LocalTensor<int32_t> kt = keyBuf_.Get<int32_t>();
        const int64_t base = buf * N_ * kRowInts;
        for (int64_t i = r0_; i < r1_; i += kTileRows) {
            int64_t n = r1_ - i;
            if (n > kTileRows) {
                n = kTileRows;
            }
            TileKeys(kt, i + base / kRowInts, n);
            for (int64_t j = 0; j < n; ++j) {
                int64_t k = LoadKey(kt, static_cast<int32_t>(j * kRowInts));
                int32_t d = static_cast<int32_t>((k >> shift) & (kBins - 1));
                hb.SetValue(d, hb.GetValue(d) + 1);
            }
        }
        CopyOutInts(wsGm_, OFF_HIST + blockIdx_ * kBins, hb, kBins);
    }
    /* ---------------- Bs：base = radix(全局 digit 前缀) + hist[0..core) ---------------- */
    __aicore__ inline void BasePrefix()
    {
        LocalTensor<int32_t> all = histAllBuf_.Get<int32_t>();
        LocalTensor<int32_t> base = baseBuf_.Get<int32_t>();
        LocalTensor<int32_t> tot = histBuf_.Get<int32_t>();
        SetFlag<HardEvent::S_MTE2>(evSMte2_);
        WaitFlag<HardEvent::S_MTE2>(evSMte2_);
        DataCopy(all, wsGm_[OFF_HIST], static_cast<int32_t>(blockNum_ * kBins));
        SetFlag<HardEvent::MTE2_V>(evMte2V_);
        WaitFlag<HardEvent::MTE2_V>(evMte2V_);
        Duplicate(tot, static_cast<int32_t>(0), static_cast<int32_t>(kBins));
        for (int64_t b = 0; b < blockNum_; ++b) {
            Add(tot, tot, all[b * kBins], static_cast<int32_t>(kBins));
        }
        PipeBarrier<PIPE_ALL>();
        int64_t acc = 0;
        for (int32_t d = 0; d < static_cast<int32_t>(kBins); ++d) {
            int32_t v = tot.GetValue(d);
            base.SetValue(d, static_cast<int32_t>(acc));
            acc += static_cast<int64_t>(v);
        }
        PipeBarrier<PIPE_ALL>();
        if (blockIdx_ > 0) {
            Duplicate(tot, static_cast<int32_t>(0), static_cast<int32_t>(kBins));
            for (int64_t b = 0; b < blockIdx_; ++b) {
                Add(tot, tot, all[b * kBins], static_cast<int32_t>(kBins));
            }
            Add(base, base, tot, static_cast<int32_t>(kBins));
        }
        PipeBarrier<PIPE_ALL>();
    }
    /* ---------------- S：散射 ---------------- */
    __aicore__ inline void ScatterPass(int64_t p, int64_t src, int64_t dst)
    {
        LocalTensor<int32_t> run = histBuf_.Get<int32_t>();   // 复用为运行计数
        LocalTensor<int32_t> base = baseBuf_.Get<int32_t>();
        Duplicate(run, static_cast<int32_t>(0), static_cast<int32_t>(kBins));
        PipeBarrier<PIPE_ALL>();
        LocalTensor<int32_t> kt = keyBuf_.Get<int32_t>();
        const int64_t shift = p * kBinBits;
        const int64_t srcBase = src * N_ * kRowInts;
        const int64_t dstBase = dst * N_ * kRowInts;
        for (int64_t i = r0_; i < r1_; i += kTileRows) {
            int64_t n = r1_ - i;
            if (n > kTileRows) {
                n = kTileRows;
            }
            TileKeys(kt, i + srcBase / kRowInts, n);
            for (int64_t j = 0; j < n; ++j) {
                int64_t k = LoadKey(kt, static_cast<int32_t>(j * kRowInts));
                int32_t d = static_cast<int32_t>((k >> shift) & (kBins - 1));
                int32_t r = run.GetValue(d);
                int32_t pos = base.GetValue(d) + r;
                run.SetValue(d, r + 1);
                const int64_t tgt = (mode_ & 1) ? (i + j) : static_cast<int64_t>(pos);
                CopyElemOut(kt, j, dstBase, tgt);
            }
            SetFlag<HardEvent::MTE3_V>(evMte3V_);
            WaitFlag<HardEvent::MTE3_V>(evMte3V_);
        }
    }
    /* ---------------- keep 计数 ---------------- */
    __aicore__ inline void CountKeeps(int64_t buf)
    {
        const int64_t base = buf * N_ * kRowInts;
        int64_t kept = 0;
        int64_t prev = 0;
        bool havePrev = false;
        if (r0_ > 0 && r0_ <= N_) {
            prev = LoadKeyGm(base + (r0_ - 1) * kRowInts);
            havePrev = true;
        }
        LocalTensor<int32_t> kt = keyBuf_.Get<int32_t>();
        for (int64_t i = r0_; i < r1_; i += kTileRows) {
            int64_t n = r1_ - i;
            if (n > kTileRows) {
                n = kTileRows;
            }
            TileKeys(kt, i + base / kRowInts, n);
            for (int64_t j = 0; j < n; ++j) {
                int64_t k = LoadKey(kt, static_cast<int32_t>(j * kRowInts));
                if (!havePrev || k != prev) {
                    kept++;
                }
                prev = k;
                havePrev = true;
            }
        }
        LocalTensor<int32_t> sb = scBuf_.Get<int32_t>();
        for (int32_t c = 0; c < static_cast<int32_t>(kSlot); ++c) {
            sb.SetValue(c, 0);
        }
        sb.SetValue(0, static_cast<int32_t>(kept));
        CopyOutInts(wsGm_, OFF_KEEPS + blockIdx_ * kSlot, sb, kSlot);
    }
    /* ---------------- 写出去重行 ---------------- */
    __aicore__ inline void WriteOut(int64_t buf)
    {
        const int64_t base = buf * N_ * kRowInts;
        int64_t off = 0;
        int64_t total = 0;
        for (int64_t b = 0; b < blockNum_; ++b) {
            int64_t v = static_cast<int64_t>(wsGm_.GetValue(OFF_KEEPS + b * kSlot));
            if (b < blockIdx_) {
                off += v;
            }
            total += v;
        }
        if (blockIdx_ == 0) {
            LocalTensor<int64_t> ub = stgBuf_.Get<int64_t>();
            ub.SetValue(0, total);
            ub.SetValue(1, 0);
            ub.SetValue(2, static_cast<int64_t>(maxSpin_));
            ub.SetValue(3, 0);
            SetFlag<HardEvent::S_MTE3>(evSMte3_);
            WaitFlag<HardEvent::S_MTE3>(evSMte3_);
            DataCopyExtParams ep;
            ep.blockCount = 1;
            ep.blockLen = 32;
            ep.srcStride = 0;
            ep.dstStride = 0;
            DataCopyPad(uGm_, ub, ep);
            SetFlag<HardEvent::MTE3_V>(evMte3V_);
            WaitFlag<HardEvent::MTE3_V>(evMte3V_);
        }
        int64_t dst = off;
        int64_t nOut = 0;
        int64_t prev = 0;
        bool havePrev = false;
        if (r0_ > 0 && r0_ <= N_) {
            prev = LoadKeyGm(base + (r0_ - 1) * kRowInts);
            havePrev = true;
        }
        LocalTensor<int32_t> ob = outBuf_.Get<int32_t>();
        Duplicate(ob, static_cast<int32_t>(0), static_cast<int32_t>(kOutRows * kRowInts));
        PipeBarrier<PIPE_ALL>();
        LocalTensor<int32_t> kt = keyBuf_.Get<int32_t>();
        for (int64_t i = r0_; i < r1_; i += kTileRows) {
            int64_t n = r1_ - i;
            if (n > kTileRows) {
                n = kTileRows;
            }
            TileKeys(kt, i + base / kRowInts, n);
            for (int64_t j = 0; j < n; ++j) {
                int64_t k = LoadKey(kt, static_cast<int32_t>(j * kRowInts));
                if (havePrev && k == prev) {
                    continue;
                }
                prev = k;
                havePrev = true;
                const int32_t od = static_cast<int32_t>(nOut * kRowInts);
                ob.SetValue(od + 0, static_cast<int32_t>((k >> sh_[0]) & m_[0]));
                ob.SetValue(od + 1, static_cast<int32_t>((k >> sh_[1]) & m_[1]));
                ob.SetValue(od + 2, static_cast<int32_t>((k >> sh_[2]) & m_[2]));
                ob.SetValue(od + 3, static_cast<int32_t>((k >> sh_[3]) & m_[3]));
                nOut++;
                if (nOut >= kOutRows) {
                    FlushRows(ob, nOut, dst);
                    dst += nOut;
                    nOut = 0;
                }
            }
        }
        if (nOut > 0) {
            FlushRows(ob, nOut, dst);
        }
    }
    __aicore__ inline void FlushRows(const LocalTensor<int32_t>& ob, int64_t n, int64_t dst)
    {
        SetFlag<HardEvent::S_MTE3>(evSMte3_);
        WaitFlag<HardEvent::S_MTE3>(evSMte3_);
        DataCopyExtParams ep;
        ep.blockCount = 1;
        ep.blockLen = static_cast<uint32_t>(n * kRowInts * static_cast<int64_t>(sizeof(int32_t)));
        ep.srcStride = 0;
        ep.dstStride = 0;
        DataCopyPad(resGm_[dst * kRowInts], ob, ep);
        SetFlag<HardEvent::MTE3_V>(evMte3V_);
        WaitFlag<HardEvent::MTE3_V>(evMte3V_);
    }

private:
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> coordBuf_;
    TBuf<TPosition::VECCALC> keyBuf_;
    TBuf<TPosition::VECCALC> histBuf_;
    TBuf<TPosition::VECCALC> baseBuf_;
    TBuf<TPosition::VECCALC> histAllBuf_;
    TBuf<TPosition::VECCALC> outBuf_;
    TBuf<TPosition::VECCALC> scBuf_;
    TBuf<TPosition::VECCALC> stgBuf_;
    TEventID evMte2S_ = 0;
    TEventID evMte2V_ = 0;
    TEventID evSMte2_ = 0;
    TEventID evSMte3_ = 0;
    TEventID evMte3S_ = 0;
    TEventID evMte3V_ = 0;
    GlobalTensor<int32_t> coordGm_;
    GlobalTensor<int32_t> resGm_;
    GlobalTensor<int64_t> uGm_;
    GlobalTensor<int32_t> keyGm_;
    GlobalTensor<int32_t> wsGm_;
    int64_t N_ = 0;
    int64_t blockNum_ = 1;
    int64_t blockIdx_ = 0;
    int64_t rowPer_ = 1;
    int64_t r0_ = 0;
    int64_t r1_ = 0;
    int64_t w_[4] = {0, 0, 0, 0};
    int64_t m_[4] = {0, 0, 0, 0};
    int64_t sh_[4] = {0, 0, 0, 0};
    int64_t bits_ = 0;
    int64_t npass_ = 0;
    int64_t stage_ = 0xFFFF;
    int64_t mode_ = 0;
    int64_t spinLimit_ = kDefaultSpinLimit;
    int64_t maxSpin_ = 0;
};
}  // namespace
extern "C" __global__ __aicore__ void spconv_sort_dedup_v3(
    GM_ADDR coords, GM_ADDR res, GM_ADDR u, GM_ADDR keys, GM_ADDR scratch,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SpconvSortDedupV3TilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KernelSpconvSortDedupV3 op(&pipe);
    op.Init(coords, res, u, keys, scratch, &tilingData);
    op.Process();
    op.Release();
}
