/* SpconvKeyprepV3 host 侧：只用 shape 算 tiling（nbk 是 device 张量，kernel 自己读，
 * 所以这里不需要它的值，也就没有 .item() 同步）。
 * start/cnt 的长度 = nbk，由调用方（torch 绑定）分配，InferShape 无法从输入推出，
 * 记为 -1；本算子是 aclnn 直调路径，输出张量由绑定自己分配。 */
#include <cstdlib>
#include "../op_kernel/spconv_keyprep_v3_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* ssz = context->GetInputShape(5);   // sizes: [n, m]
    if (ssz == nullptr) {
        return ge::GRAPH_FAILED;
    }
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
        coreNum = 64;   // kernel 侧 workspace 按 64 核分槽
    }
    SpconvKeyprepV3TilingData* tiling = context->GetTilingData<SpconvKeyprepV3TilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->blockNum = coreNum;
    tiling->N = static_cast<uint32_t>(N);
    tiling->M = static_cast<uint32_t>(M);
    tiling->kv = static_cast<uint32_t>(kv);
    tiling->nbk = 0;   // 本段不用（kernel 直接从 nbk 张量读）
    tiling->dynamic = 1;   // N/M 在 kernel 内从 sizes 读
    tiling->stage = 5;
    const char* envSt = std::getenv("UNUM_KPREP_STAGE");
    if (envSt != nullptr && std::atoi(envSt) >= 0) {
        tiling->stage = static_cast<uint32_t>(std::atoi(envSt));
    }
    context->SetBlockDim(coreNum);
    /* SyncAll 用的 FFTS 全核 barrier 要求所有核 co-resident；不声明 batch 模式
     * 的话跨核同步不成立（kernel 侧 SyncAll 立刻返回 / 结果错乱）。 */
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
    if (rk == nullptr || cn == nullptr || mt == nullptr || sc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    rk->SetDimNum(1);
    rk->SetDim(0, N);
    cn->SetDimNum(1);
    cn->SetDim(0, -1);   // nbk 只有 device 端知道
    mt->SetDimNum(1);
    mt->SetDim(0, 12 + 3 * offS->GetDim(0));
    sc->SetDimNum(1);
    sc->SetDim(0, 4096);   // 跨核交换缓冲区（固定大小）
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_INT64);
    context->SetOutputDataType(1, ge::DT_INT32);
    context->SetOutputDataType(2, ge::DT_INT32);
    context->SetOutputDataType(3, ge::DT_INT32);
    return GRAPH_SUCCESS;
}
}  // namespace ge
namespace ops {
class SpconvKeyprepV3 : public OpDef {
public:
    explicit SpconvKeyprepV3(const char* name) : OpDef(name)
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
        this->Input("sizes").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("rkeys").ParamType(REQUIRED).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("cnt").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("meta").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("scratch").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(SpconvKeyprepV3);
}  // namespace ops
IMPL_OP_INFERSHAPE(SpconvKeyprepV3)
    .InferShape(ge::InferShape)
    .InferDataType(ge::InferDataType);
