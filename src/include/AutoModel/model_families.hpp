/// \file model_families.hpp
/// \brief Which family a model tag belongs to, and whether that family is a chat model.
///
/// Split out of all_models.hpp because the question has to be answerable WITHOUT
/// constructing an engine. all_models.hpp includes every modeling header and so
/// needs XRT and a device; this needs the model list and nothing else. The server
/// asks `is_chat_model()` BEFORE it evicts what is on the NPU -- the factory can
/// only refuse by returning null, and by then the eviction has happened.
#pragma once

#include <map>
#include <string>

#include "model_list.hpp"

typedef enum {
    llama3,
    granite,
    deepseek_r1,
    deepseek_r1_0528,
    qwen2,
    qwen2vl,
    qwen3,
    qwen3_it,
    qwen3_tk,
    qwen3vl,
    qwen3_5,
    qwen3_5_omni,
    qwen3_6_moe,
    gemma3,
    gemma3_text,
    gemma4e,
    gemma4_12b,
    gpt_oss,
    lfm2,
    lfm2_5_tk,
    phi4,
    nanbeige,
    error_whiper,
    error_embedding
} SupportedModelFamily;

/// The family name -> engine-selector map. Two callers need it: get_auto_model()'s
/// switch, and is_chat_model() below.
inline const std::map<std::string, SupportedModelFamily>& model_family_map() {
    static const std::map<std::string, SupportedModelFamily> modelFamilyMap = {
        {"llama3", SupportedModelFamily::llama3},
        {"granite", SupportedModelFamily::granite},
        {"deepseek-r1", SupportedModelFamily::deepseek_r1},
        {"deepseek-r1-0528", SupportedModelFamily::deepseek_r1_0528},
        {"qwen2", SupportedModelFamily::qwen2},
        {"qwen3", SupportedModelFamily::qwen3},
        {"qwen3-it", SupportedModelFamily::qwen3_it},
        {"qwen3-tk", SupportedModelFamily::qwen3_tk},
        {"qwen3vl", SupportedModelFamily::qwen3vl},
        {"qwen3.5", SupportedModelFamily::qwen3_5},
        {"qwen3.5-omni", SupportedModelFamily::qwen3_5_omni},
        {"qwen3.6-moe", SupportedModelFamily::qwen3_6_moe},
        {"gemma3", SupportedModelFamily::gemma3},
        {"gemma3-text", SupportedModelFamily::gemma3_text},
        {"gemma4e", SupportedModelFamily::gemma4e},
        {"gemma4-12b", SupportedModelFamily::gemma4_12b},
        {"gpt-oss", SupportedModelFamily::gpt_oss},
        {"lfm2", SupportedModelFamily::lfm2},
        {"lfm2.5-tk", SupportedModelFamily::lfm2_5_tk},
        {"qwen2vl", SupportedModelFamily::qwen2vl},
        {"phi4", SupportedModelFamily::phi4},
        {"nanbeige", SupportedModelFamily::nanbeige},
        {"whisper-v3", SupportedModelFamily::error_whiper},
        {"embed-gemma", SupportedModelFamily::error_embedding}
    };
    return modelFamilyMap;
}

/// True when `model_tag` names something this build can serve AS A CHAT MODEL.
///
/// `model_list` answers a different question -- whether the tag exists -- and
/// `embed-gemma:300m` and `whisper-v3:turbo` exist. They are not chat models, and
/// asking only the model list is how a request for either was answered by a
/// Llama3 renamed to "llama3.2:1b".
inline bool is_chat_model(const std::string& model_tag, model_list& available_models) {
    if (!available_models.is_model_supported(model_tag)) return false;
    auto [resolved, model_info] = available_models.get_model_info(model_tag);
    (void)resolved;
    if (!model_info.contains("details") || !model_info["details"].contains("family")) return false;
    const auto& m = model_family_map();
    const auto it = m.find(model_info["details"]["family"].get<std::string>());
    if (it == m.end()) return false;            // a family this build has no engine for
    return it->second != SupportedModelFamily::error_whiper &&
           it->second != SupportedModelFamily::error_embedding;
}
