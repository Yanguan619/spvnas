/* SpconvKeyprepV2Prefix host 侧：cnt -> start 的独占前缀和。
 * nbk 直接来自 cnt 的 shape，所以 tiling 里没有额外信息。 */
#include <cstdlib>
#include "../op_kernel/spconv_keyprep_v2_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* cs = context->GetInputShape(0);    // cnt (nbk)
    if (cs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& cShape = cs->GetStorageShape();
    if (cShape.GetDimNum() < 1) {
        return ge::GRAPH_FAILED;
    }
    const int64_t nbk = cShape.GetDim(0);
    if (nbk < 0) {
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
        coreNum = 64;
    }
    SpconvKeyprepV2TilingData* tiling = context->GetTilingData<SpconvKeyprepV2TilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->blockNum = coreNum;
    tiling->N = 0;
    tiling->M = 0;
    tiling->kv = 0;
    tiling->nbk = static_cast<uint32_t>(nbk);
    tiling->stage = 3;
    const char* envSt = std::getenv("UNUM_KPREP_PSTAGE");
    if (envSt != nullptr && std::atoi(envSt) >= 0) {
        tiling->stage = static_cast<uint32_t>(std::atoi(envSt));
    }
    context->SetBlockDim(coreNum);
    context->SetScheduleMode(1);
    size_t* ws = context->GetWorkspaceSizes(1);
    if (ws == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling
namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* cs = context->GetInputShape(0);
    if (cs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* out = context->GetOutputShape(0);
    if (out == nullptr) {
        return ge::GRAPH_FAILED;
    }
    out->SetDimNum(1);
    out->SetDim(0, cs->GetDim(0));
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    return GRAPH_SUCCESS;
}
}  // namespace ge
namespace ops {
class SpconvKeyprepV2Prefix : public OpDef {
public:
    explicit SpconvKeyprepV2Prefix(const char* name) : OpDef(name)
    {
        this->Input("cnt").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("start").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(SpconvKeyprepV2Prefix);
}  // namespace ops
IMPL_OP_INFERSHAPE(SpconvKeyprepV2Prefix)
    .InferShape(ge::InferShape)
    .InferDataType(ge::InferDataType);

