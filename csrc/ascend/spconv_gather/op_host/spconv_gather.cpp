/* SpconvGather host 侧：tiling（按 AIV 核数分 M 行）+ 形状推导 + 注册。 */

#include <cstdlib>
#include <cstring>

#include "../op_kernel/spconv_gather_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = (platform != nullptr) ? platform->GetCoreNumAiv() : 1;
    if (coreNum < 1) {
        coreNum = 1;
    }
    const char* envBn = std::getenv("UNUM_GATHER_BLOCKNUM");
    if (envBn != nullptr && std::atoi(envBn) > 0) {
        coreNum = static_cast<uint32_t>(std::atoi(envBn));
    }

    SpconvGatherTilingData* tiling = context->GetTilingData<SpconvGatherTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->blockNum = coreNum;
    context->SetBlockDim(coreNum);

    size_t* wsSizes = context->GetWorkspaceSizes(1);
    if (wsSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    wsSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* nbs = context->GetInputShape(1);      // nb: (M, K)
    const gert::Shape* ps = context->GetInputShape(2);       // params: [M, cin, kcnt]
    int64_t M = (nbs->GetDimNum() >= 1) ? nbs->GetDim(0) : 0;
    int64_t K = (nbs->GetDimNum() >= 2) ? nbs->GetDim(1) : 0;

    int64_t cin = 0;
    if (ps->GetDimNum() >= 1 && ps->GetDim(0) > 2) {
        // params 只有形状信息，取不到值；cin 由 feat 的列数给出
        (void)cin;
    }
    const gert::Shape* fs = context->GetInputShape(0);        // feat: (N, cin)
    cin = (fs->GetDimNum() >= 2) ? fs->GetDim(1) : 0;

    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outShape->SetDimNum(2);
    outShape->SetDim(0, (M + 7) / 8 * 8);   // 8 行对齐：配合内核整块 DataCopyPad
    outShape->SetDim(1, K * cin);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class SpconvGather : public OpDef {
public:
    explicit SpconvGather(const char* name) : OpDef(name)
    {
        this->Input("feat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("nb")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("params")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("dense")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SpconvGather);

}  // namespace ops

IMPL_OP_INFERSHAPE(SpconvGather)
    .InferShape(ge::InferShape)
    .InferDataType(ge::InferDataType);
