/* SpconvDevox host 侧。calcWeights / scale 由环境变量给（避免为推理路径
 * 再开一个张量入参）。 */

#include <cstdlib>

#include "../op_kernel/spconv_devox_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* nsz = context->GetInputShape(4);   // n: (1,) 设备上的真实点数
    if (nsz == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* fs = context->GetInputShape(0);   // feat (Nv, c)
    const auto* is = context->GetInputShape(1);   // idx  (K, N)
    if (fs == nullptr || is == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& fShape = fs->GetStorageShape();
    const gert::Shape& iShape = is->GetStorageShape();
    if (fShape.GetDimNum() < 2 || iShape.GetDimNum() < 2) {
        return ge::GRAPH_FAILED;
    }
    const int64_t N = iShape.GetDim(1);
    const int64_t kcnt = iShape.GetDim(0);
    const int64_t c = fShape.GetDim(1);
    if (N < 0 || c <= 0 || kcnt <= 0 || kcnt > 8) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = (platform != nullptr) ? platform->GetCoreNumAiv() : 1;
    if (coreNum < 1) {
        coreNum = 1;
    }
    const char* envBn = std::getenv("UNUM_DEVOX_BLOCKNUM");
    if (envBn != nullptr && std::atoi(envBn) > 0) {
        coreNum = static_cast<uint32_t>(std::atoi(envBn));
    }
    if (static_cast<uint32_t>(N) < coreNum) {
        coreNum = (N > 0) ? static_cast<uint32_t>(N) : 1u;
    }

    SpconvDevoxTilingData* tiling = context->GetTilingData<SpconvDevoxTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* cs = context->GetInputShape(3);   // coords (N, 3) 或占位
    const bool hasCoords = (cs != nullptr && cs->GetStorageShape().GetDimNum() == 2
                            && cs->GetStorageShape().GetDim(1) == 3);
    const char* sc = std::getenv("UNUM_DEVOX_SCALE");
    tiling->blockNum = coreNum;
    tiling->N = static_cast<uint32_t>(N);
    tiling->dynamic = 1;   // 真实点数在 kernel 内从 n 读
    tiling->c = static_cast<uint32_t>(c);
    tiling->kcnt = static_cast<uint32_t>(kcnt);
    tiling->calcWeights = hasCoords ? 1u : 0u;
    tiling->scale = (sc != nullptr) ? static_cast<float>(std::atof(sc)) : 1.0f;
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
    const gert::Shape* fs = context->GetInputShape(0);
    const gert::Shape* is = context->GetInputShape(1);
    if (fs == nullptr || is == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* out = context->GetOutputShape(0);
    if (out == nullptr) {
        return ge::GRAPH_FAILED;
    }
    out->SetDimNum(2);
    out->SetDim(0, is->GetDim(1));
    {
        const int64_t cc = fs->GetDim(1);
        out->SetDim(1, (cc + 7) / 8 * 8);   // 宽度补齐 8：配合内核的 32B 对齐行距
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class SpconvDevoxV4 : public OpDef {
public:
    explicit SpconvDevoxV4(const char* name) : OpDef(name)
    {
        this->Input("feat").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("idx").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("coords").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("n").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SpconvDevoxV4);

}  // namespace ops

IMPL_OP_INFERSHAPE(SpconvDevoxV4)
    .InferShape(ge::InferShape)
    .InferDataType(ge::InferDataType);
