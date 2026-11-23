/* SpconvSortDedupV4 host：单 launch，tiling 只给 blockNum / N / 行宽 / 屏障自旋上限。
 * dims、位宽、趟数都在 kernel 里由全局 max 推出来（host 读它们要一次 .item()
 * 同步，而整条路径的目标就是消掉这些同步）。 */
#include <cstdlib>
#include "../op_kernel/spconv_sort_dedup_v4_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* nsz = context->GetInputShape(1);   // n: (1,) 设备上的真实行数
    if (nsz == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* cs = context->GetInputShape(0);   // coords (N, 4)
    if (cs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& cShape = cs->GetStorageShape();
    if (cShape.GetDimNum() < 2) {
        return ge::GRAPH_FAILED;
    }
    const int64_t N = cShape.GetDim(0);
    const int64_t cols = cShape.GetDim(1);
    if (N < 0 || cols <= 0 || cols > 4) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = (platform != nullptr) ? platform->GetCoreNumAiv() : 1;
    if (coreNum < 1) {
        coreNum = 1;
    }
    const char* envBn = std::getenv("UNUM_SD_BLOCKNUM");
    if (envBn != nullptr && std::atoi(envBn) > 0) {
        coreNum = static_cast<uint32_t>(std::atoi(envBn));
    }
    if (static_cast<uint32_t>(N) < coreNum) {
        coreNum = (N > 0) ? static_cast<uint32_t>(N) : 1u;
    }
    if (coreNum > 64) {
        coreNum = 64;
    }
    SpconvSortDedupV4TilingData* tiling = context->GetTilingData<SpconvSortDedupV4TilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->blockNum = coreNum;
    tiling->N = static_cast<uint32_t>(N);
    tiling->dynamic = 1;   // 真实行数在 kernel 内读
    tiling->cols = static_cast<uint32_t>(cols);
    tiling->outCols = 8;   // 32B 对齐（DataCopyPad 行宽陷阱）
    const char* es = std::getenv("UNUM_SD_STAGE");
    tiling->stage = (es != nullptr) ? static_cast<uint32_t>(std::atoi(es)) : 0xFFFFu;
    const char* em = std::getenv("UNUM_SD_MODE");
    tiling->mode = (em != nullptr) ? static_cast<uint32_t>(std::atoi(em)) : 0u;
    /* 软屏障自旋上限。正常屏障只需几次迭代；上限只是「有 block 分不到 AIV」时的
     * 兜底，不要放大到 1e8 那种量级（V1 就是这样一帧慢 71 s 的）。 */
    const char* esp = std::getenv("UNUM_SD_SPIN");
    tiling->spinLimit = (esp != nullptr) ? static_cast<uint32_t>(std::atoi(esp)) : 4000u;
    context->SetBlockDim(coreNum);
    // batch 调度：40 个 AIV 凑齐才开跑，避免部分驻留把软屏障等成死循环
    // （spconv_keyprep_v1/v2 一直这么写，实测挤占下不会超时）
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
    const int64_t N = cs->GetDim(0);
    gert::Shape* res = context->GetOutputShape(0);
    gert::Shape* u = context->GetOutputShape(1);
    gert::Shape* keys = context->GetOutputShape(2);
    gert::Shape* scratch = context->GetOutputShape(3);
    if (res == nullptr || u == nullptr || keys == nullptr || scratch == nullptr) {
        return ge::GRAPH_FAILED;
    }
    res->SetDimNum(2);
    res->SetDim(0, N);
    res->SetDim(1, 8);
    u->SetDimNum(1);
    u->SetDim(0, 4);
    keys->SetDimNum(1);
    keys->SetDim(0, 2 * N * 8);
    scratch->SetDimNum(1);
    scratch->SetDim(0, kSDScratchInts);
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_INT64);
    context->SetOutputDataType(2, ge::DT_INT32);
    context->SetOutputDataType(3, ge::DT_INT32);
    return GRAPH_SUCCESS;
}
}  // namespace ge
namespace ops {
class SpconvSortDedupV4 : public OpDef {
public:
    explicit SpconvSortDedupV4(const char* name) : OpDef(name)
    {
        this->Input("coords").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("n").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("res").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("u").ParamType(REQUIRED).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("rows").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("scratch").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(SpconvSortDedupV4);
}  // namespace ops
IMPL_OP_INFERSHAPE(SpconvSortDedupV4)
    .InferShape(ge::InferShape)
    .InferDataType(ge::InferDataType);
