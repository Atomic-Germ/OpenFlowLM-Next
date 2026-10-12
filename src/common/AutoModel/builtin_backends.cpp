/// \file builtin_backends.cpp
/// \brief The backends this build ships, keyed by model family
/// \author OpenFlowLM Team
/// \note Ported from the FastFlowLM fork's builtin_backends.cpp (MIT); see
///       model_backend.hpp for the renaming notes. One place knows both the
///       family names from model_list.json and the engine types behind them.
///       Everything else goes through the registry.
///
///       NOT ported: the fork's "corelib_aie4_gguf" phi4 backend. It executes
///       through the proprietary ryzenai/corelib.h (exact version 0.3.0) and
///       its prebuilt AIE4 kernels, neither of which is in this repo. A
///       catalog entry asking for that backend resolves to a clear
///       "not implemented" error from resolve_backend_id (unknown id for the
///       family) instead of a missing-header build break. An open replacement
///       that loads the same GGUF directly is future work on the GGUF track.
///
///       Also not registered here: "granite" (open-kernels-only, no closed
///       engine to wrap -- see modeling_granite.cpp) and "qwen3.5-omni"
///       (its qwen3_5_omni class is not a causal_lm, so the shared template
///       does not apply). Both keep their existing frontend paths untouched.
#include "AutoModel/automodel.hpp"
#include "AutoModel/oflm_npu_backend.hpp"
#include "AutoModel/model_backend.hpp"

namespace oflm::backend {
namespace {

/// \brief register the OpenFlowLM NPU backend for one family
/// \tparam Engine the concrete engine type
/// \param registry the registry to populate
/// \param family the family name, as in details.family
/// \note kLegacyBackendId ("flm_npu") is registered alongside kDefaultBackendId
///       so catalogs shared with the FastFlowLM ecosystem keep resolving.
template <class Engine>
void RegisterOflmNpu(BackendRegistry& registry, const char* family) {
    registry.register_backend(family, kDefaultBackendId,
                              oflm_npu_factory<Engine>());
    registry.register_backend(family, kLegacyBackendId,
                              oflm_npu_factory<Engine>());
}

}  // namespace

void register_builtin_backends(BackendRegistry& registry) {
    RegisterOflmNpu<llama_npu>(registry, "llama3");
    RegisterOflmNpu<llama_npu>(registry, "deepseek-r1");
    RegisterOflmNpu<qwen3_npu>(registry, "deepseek-r1-0528");
    RegisterOflmNpu<qwen2_npu>(registry, "qwen2");
    RegisterOflmNpu<qwen2vl_npu>(registry, "qwen2vl");
    RegisterOflmNpu<qwen3_npu>(registry, "qwen3");
    RegisterOflmNpu<qwen3_npu>(registry, "qwen3-it");
    RegisterOflmNpu<qwen3_npu>(registry, "qwen3-tk");
    RegisterOflmNpu<qwen3vl_npu>(registry, "qwen3vl");
    RegisterOflmNpu<qwen3_5vl_npu>(registry, "qwen3.5");
    RegisterOflmNpu<qwen3_6_moe_npu>(registry, "qwen3.6-moe");
    RegisterOflmNpu<gemma_npu>(registry, "gemma3");
    RegisterOflmNpu<gemma_text_npu>(registry, "gemma3-text");
    RegisterOflmNpu<gemma4e_npu>(registry, "gemma4e");
    RegisterOflmNpu<gemma4_12b_npu>(registry, "gemma4-12b");
    RegisterOflmNpu<gpt_oss_npu>(registry, "gpt-oss");
    RegisterOflmNpu<lfm2_npu>(registry, "lfm2");
    RegisterOflmNpu<lfm2_npu>(registry, "lfm2.5-tk");
    RegisterOflmNpu<nanbeige_npu>(registry, "nanbeige");
    RegisterOflmNpu<phi4_npu>(registry, "phi4");
}

}  // namespace oflm::backend
