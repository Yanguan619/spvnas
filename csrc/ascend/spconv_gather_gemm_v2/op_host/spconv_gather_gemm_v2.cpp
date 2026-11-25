/* SpconvGatherGemmV2 host 侧：形状来自输入张量，TM 由暂存区预算反推，
 * Matmul tiling 按「单 tile 形状 (TM, cout, K)」生成（kernel 侧按 tile 循环，
 * 用 SetSingleShape 处理尾块）。 */
#include <cstdlib>
#include "../op_kernel/spconv_gather_gemm_v2_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "adv_api/matmul/matmul_tiling.h"
#include "adv_api/matmul/bmm_tiling.h"  // MultiCoreMatmulTiling

namespace optiling {
using namespace matmul_tiling;
static constexpr uint32_t kStageBudget = 48u * 1024u * 1024u;  // A 暂存区预算（AIV 写 + AIC 读都在 L2）

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* fs = context->GetInputShape(0);  // feat: (N, cin)
    const auto* ns = context->GetInputShape(1);  // nb:   (M, kv)
    const auto* ws = context->GetInputShape(2);  // w:    (kv*cin, cout)
    const auto* nsz = context->GetInputShape(3); // n: (1,) 设备上的真实行数
    if (nsz == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (fs == nullptr || ns == nullptr || ws == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& fShape = fs->GetStorageShape();
    const gert::Shape& nShape = ns->GetStorageShape();
    const gert::Shape& wShape = ws->GetStorageShape();
    if (fShape.GetDimNum() < 2 || nShape.GetDimNum() < 2 || wShape.GetDimNum() < 2) {
        return ge::GRAPH_FAILED;
    }
    const int64_t M = nShape.GetDim(0);
    const int64_t kv = nShape.GetDim(1);
    const int64_t Nr = fShape.GetDim(0);
    const int64_t cin = fShape.GetDim(1);
    const int64_t cout = wShape.GetDim(1);
    const int64_t K = kv * cin;
    if (M <= 0 || K <= 0 || cout <= 0 || wShape.GetDim(0) != K) {
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    if (aicNum < 1) {
        aicNum = 1;
    }
    const char* envAic = std::getenv("SGG_AIC");
    if (envAic != nullptr && std::atoi(envAic) > 0) {
        aicNum = static_cast<uint32_t>(std::atoi(envAic));
    }

    // TM：每 AIC 两份（ping-pong），每份 TM 行 × K 宽的 A 暂存区。默认 64——探针扫描
    // 最优（32 时 B 的重装次数翻倍，≥128 时暂存区过大反而慢）；暂存区预算 64MB。
    int64_t tm = 64;
    const int64_t stageBudget = 64LL * 1024 * 1024;
    const int64_t perRow = 2LL * aicNum * K * 4LL;  // 每个 TM 行占的暂存字节数
    if (perRow > 0 && perRow * tm > stageBudget) {
        tm = stageBudget / perRow;
    }
    const int64_t want = (M + aicNum - 1) / aicNum;  // 让每个 AIC 至少摊到一行，避免大 TM 饿核
    if (tm > want) {
        tm = want;
    }
    const char* envTm = std::getenv("SGG_TM");
    if (envTm != nullptr && std::atoi(envTm) > 0) {
        tm = std::atoi(envTm);
    }
    tm = (tm / 16) * 16;
    if (tm < 16) {
        tm = 16;
    }
    if (tm > 1024) {
        tm = 1024;
    }
    uint32_t blockNum = aicNum;
    const int64_t tiles = (M + tm - 1) / tm;
    if (tiles < static_cast<int64_t>(blockNum)) {
        blockNum = static_cast<uint32_t>(tiles);
    }

    SpconvGatherGemmV2TilingData* tiling = context->GetTilingData<SpconvGatherGemmV2TilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->blockNum = blockNum;
    tiling->M = static_cast<uint32_t>(M);
    tiling->Nr = static_cast<uint32_t>(Nr);
    tiling->cin = static_cast<uint32_t>(cin);
    tiling->kv = static_cast<uint32_t>(kv);
    tiling->cout = static_cast<uint32_t>(cout);
    tiling->TM = static_cast<uint32_t>(tm);
    tiling->dynamic = 1;   // 真实 M 在 kernel 内从 n 读

    // 单 tile 形状的 Matmul tiling（fp32 A/B/C，ND，无 bias）
    matmul_tiling::MultiCoreMatmulTiling mt(ascendcPlatform);
    mt.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mt.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mt.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mt.SetBias(false);
    mt.SetShape(static_cast<int32_t>(tm), static_cast<int32_t>(cout), static_cast<int32_t>(K));
    mt.SetOrgShape(static_cast<int32_t>(tm), static_cast<int32_t>(cout), static_cast<int32_t>(K));
    mt.SetDim(1);
    mt.SetSingleShape(static_cast<int32_t>(tm), static_cast<int32_t>(cout), -1);
    if (mt.GetTiling(tiling->mm) == -1) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(blockNum);
    size_t* wsSizes = context->GetWorkspaceSizes(1);
    if (wsSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    wsSizes[0] = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize()) +
                 static_cast<size_t>(2) * blockNum * static_cast<size_t>(tm) *
                     static_cast<size_t>(K) * sizeof(float);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* nbs = context->GetInputShape(1);  // nb: (M, kv)
    const gert::Shape* ws = context->GetInputShape(2);   // w: (K, cout)
    if (nbs == nullptr || ws == nullptr || nbs->GetDimNum() < 2 || ws->GetDimNum() < 2) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outShape->SetDimNum(2);
    outShape->SetDim(0, nbs->GetDim(0));
    outShape->SetDim(1, ws->GetDim(1));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class SpconvGatherGemmV2 : public OpDef {
public:
    explicit SpconvGatherGemmV2(const char* name) : OpDef(name)
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
        this->Input("w")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("n")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(SpconvGatherGemmV2);
}  // namespace ops
IMPL_OP_INFERSHAPE(SpconvGatherGemmV2)
    .InferShape(ge::InferShape)
    .InferDataType(ge::InferDataType);
