/* SpconvKeyprepV2 host：只看 shape 填 tiling；nbk 是 device 张量，kernel 自己读。
 * 所以这里不需要它的值，也没有 .item() 同步。
 * start/cnt 的长度 = nbk，由调用方（torch 绑定）分配，InferShape 无法算（推出长度
 * 会写成 -1），走 aclnn 直调路径的调用方自己分配。
 * 与 V1 的差别只有两处：多一个 u 输出（int64[4]：失败标记/最大自旋），tiling 多一个
 * 屏障自旋上限。 */
#include <cstdlib>
#include "../op_kernel/spconv_keyprep_v2_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* inS = context->GetInputShape(0);    // in_coords (N, 4)
    const auto* outS = context->GetInputShape(1);   // out_coords (M, 4)
    const auto* offS = context->GetInputShape(3);   // offs (KV, 3)
    if (inS == nullptr || outS == nullptr || offS == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& inShape = inS->GetStorageShape();
    const gert::Shape& outShape = outS->GetStorageShape();
    const gert::Shape& offShape = offS->GetStorageShape();
    if (inShape.GetDimNum() < 2 || outShape.GetDimNum() < 2 || offShape.GetDimNum() < 2) {
        return ge::GRAPH_FAILED;
    }
    const int64_t N = inShape.GetDim(0);
    const int64_t M = outShape.GetDim(0);
    const int64_t kv = offShape.GetDim(0);
    if (N < 0 || M < 0 || kv <= 0 || kv > 128 || offShape.GetDim(1) != 3 || inShape.GetDim(1) != 4) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = (platform != nullptr) ? platform->GetCoreNumAiv() : 1;
    if (coreNum < 1) {
        coreNum = 1;
    }
    const char* envBn = std::getenv("UNUM_KPREP_BLOCKNUM");
    if (envBn != nullptr && std::atoi(envBn) > 0 && static_cast<uint32_t>(std::atoi(envBn)) <= coreNum) {
        coreNum = static_cast<uint32_t>(std::atoi(envBn));
    }
    if (coreNum > 64) {
        coreNum = 64;   // kernel 的 workspace 按 64 核分布
    }
    SpconvKeyprepV2TilingData* tiling = context->GetTilingData<SpconvKeyprepV2TilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->blockNum = coreNum;
    tiling->N = static_cast<uint32_t>(N);
    tiling->M = static_cast<uint32_t>(M);
    tiling->kv = static_cast<uint32_t>(kv);
    tiling->nbk = 0;   // 先置零；kernel 直接从 nbk 输入读
    tiling->stage = 5;
    const char* envSt = std::getenv("UNUM_KPREP_STAGE");
    if (envSt != nullptr && std::atoi(envSt) >= 0) {
        tiling->stage = static_cast<uint32_t>(std::atoi(envSt));
    }
    /* 软屏障自旋上限：正常只需几十次，这里只是「有核分不到 AIV」时的兜底。
     * 不要放大到 1e8 那种量级（V1 就是这样静默慢几十秒的）。 */
    const char* envSp = std::getenv("UNUM_SD_SPIN");
    tiling->spinLimit = (envSp != nullptr) ? static_cast<uint32_t>(std::atoi(envSp)) : 4000u;
    context->SetBlockDim(coreNum);
    /* SyncAll 用的 FFTS 全核 barrier 要求全核 co-resident，所以声明 batch 模式；
     * 不加的话同流下游 kernel 的 SyncAll 会挂（保持与 V1 一致） */
    context->SetScheduleMode(1);
    size_t* ws = context->GetWorkspaceSizes(1);
    if (ws == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ws[0] = 0;   // 跨核交换走 scratch 输出张量（workspace 在 aclnn 直调路径上不可靠）
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling
namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inS = context->GetInputShape(0);
    const gert::Shape* offS = context->GetInputShape(3);
    if (inS == nullptr || offS == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t N = inS->GetDim(0);
    gert::Shape* rk = context->GetOutputShape(0);
    gert::Shape* cn = context->GetOutputShape(1);
    gert::Shape* mt = context->GetOutputShape(2);
    gert::Shape* sc = context->GetOutputShape(3);
    gert::Shape* uu = context->GetOutputShape(4);
    if (rk == nullptr || cn == nullptr || mt == nullptr || sc == nullptr || uu == nullptr) {
        return ge::GRAPH_FAILED;
    }
    rk->SetDimNum(1);
    rk->SetDim(0, N);
    cn->SetDimNum(1);
    cn->SetDim(0, -1);   // nbk 只有 device 知道
    mt->SetDimNum(1);
    mt->SetDim(0, 12 + 3 * offS->GetDim(0));
    sc->SetDimNum(1);
    sc->SetDim(0, 4096);   // 跨核交换 + 屏障格子，固定大小
    uu->SetDimNum(1);
    uu->SetDim(0, 4);     // [失败标记, core0 最大自旋次数, 0, 0]，32B 对齐
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_INT64);
    context->SetOutputDataType(1, ge::DT_INT32);
    context->SetOutputDataType(2, ge::DT_INT32);
    context->SetOutputDataType(3, ge::DT_INT32);
    context->SetOutputDataType(4, ge::DT_INT64);
    return GRAPH_SUCCESS;
}
}  // namespace ge
namespace ops {
class SpconvKeyprepV2 : public OpDef {
public:
    explicit SpconvKeyprepV2(const char* name) : OpDef(name)
    {
        this->Input("in_coords").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("out_coords").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("stride").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("offs").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("nbk").ParamType(REQUIRED).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("rkeys").ParamType(REQUIRED).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("cnt").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("meta").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("scratch").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("u").ParamType(REQUIRED).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(SpconvKeyprepV2);
}  // namespace ops
IMPL_OP_INFERSHAPE(SpconvKeyprepV2)
    .InferShape(ge::InferShape)
    .InferDataType(ge::InferDataType);
