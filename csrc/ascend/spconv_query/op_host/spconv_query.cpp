/* SpconvQuery host 侧：形状从 tiling context 取。
 * kcnt 由 meta 张量长度反推：len = 12 + 3*kcnt。 */

#include <cstdlib>

#include "../op_kernel/spconv_query_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* qs = context->GetInputShape(0);    // qcoords (M, 4)
    const auto* ms = context->GetInputShape(4);    // meta
    const auto* ss = context->GetInputShape(2);    // start (buckets)
    if (qs == nullptr || ms == nullptr || ss == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& qShape = qs->GetStorageShape();
    const gert::Shape& mShape = ms->GetStorageShape();
    const gert::Shape& sShape = ss->GetStorageShape();
    if (qShape.GetDimNum() < 2 || mShape.GetDimNum() < 1 || sShape.GetDimNum() < 1) {
        return ge::GRAPH_FAILED;
    }
    const int64_t M = qShape.GetDim(0);
    const int64_t metaLen = mShape.GetDim(0);
    const int64_t kcnt = (metaLen - 12) / 3;
    if (M < 0 || kcnt <= 0 || kcnt > 32) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = (platform != nullptr) ? platform->GetCoreNumAiv() : 1;
    if (coreNum < 1) {
        coreNum = 1;
    }
    const char* envBn = std::getenv("UNUM_QUERY_BLOCKNUM");
    if (envBn != nullptr && std::atoi(envBn) > 0) {
        coreNum = static_cast<uint32_t>(std::atoi(envBn));
    }
    if (static_cast<uint32_t>(M) < coreNum) {
        coreNum = (M > 0) ? static_cast<uint32_t>(M) : 1u;
    }

    SpconvQueryTilingData* tiling = context->GetTilingData<SpconvQueryTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->blockNum = coreNum;
    tiling->M = static_cast<uint32_t>(M);
    tiling->kcnt = static_cast<uint32_t>(kcnt);
    tiling->buckets = static_cast<uint32_t>(sShape.GetDim(0));
    context->SetBlockDim(coreNum);

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
    const gert::Shape* qs = context->GetInputShape(0);
    const gert::Shape* ms = context->GetInputShape(5);
    if (qs == nullptr || ms == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t M = qs->GetDim(0);
    const int64_t kcnt = (ms->GetDim(0) - 12) / 3;
    gert::Shape* out = context->GetOutputShape(0);
    if (out == nullptr) {
        return ge::GRAPH_FAILED;
    }
    out->SetDimNum(2);
    out->SetDim(0, M);
    out->SetDim(1, kcnt);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class SpconvQuery : public OpDef {
public:
    explicit SpconvQuery(const char* name) : OpDef(name)
    {
        this->Input("qcoords").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("rkeys").ParamType(REQUIRED).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("start").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cnt").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("meta").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("idx").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SpconvQuery);

}  // namespace ops

IMPL_OP_INFERSHAPE(SpconvQuery)
    .InferShape(ge::InferShape)
    .InferDataType(ge::InferDataType);
