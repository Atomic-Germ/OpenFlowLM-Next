/// \file all_models.hpp
/// \brief all_models class
/// \author OpenFlowLM Team
/// \date 2025-09-10
/// \version 0.9.24
/// \note This is a header file for the all_models class
#pragma once

#include "modeling_llama3.hpp"
#include "modeling_granite.hpp"
#include "modeling_gemma3.hpp"
#include "modeling_gemma3_text.hpp"
#include "modeling_qwen3.hpp"
#include "modeling_gpt_oss.hpp"
#include "modeling_lfm2.hpp"
#include "modeling_phi4.hpp"
#include "modeling_qwen2.hpp"
#include "modeling_qwen3.hpp"
#include "modeling_qwen2vl.hpp"
#include "modeling_qwen3vl.hpp"
#include "modeling_qwen3_5vl.hpp"
#include "modeling_qwen3_5_omni.hpp"
#include "modeling_qwen3_6_moe.hpp"
#include "modeling_nanbeige.hpp"
#include "modeling_gemma4e.hpp"
#include "modeling_gemma4_12b.hpp"
#include "model_list.hpp"
#include "nlohmann/json.hpp"

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

/// The family name -> engine map. At namespace scope because two callers need it:
/// the factory below, and `is_chat_model()`, which the server asks BEFORE it takes
/// the loaded model off the NPU.
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
/// the caller has to learn that BEFORE it evicts what is loaded, because the
/// factory can only say so by returning null, and by then the NPU is already clear.
inline bool is_chat_model(const std::string& model_tag, model_list& available_models) {
    if (!available_models.is_model_supported(model_tag)) return false;
    auto [resolved, model_info] = available_models.get_model_info(model_tag);
    (void)resolved;
    const auto& m = model_family_map();
    const auto it = m.find(model_info["details"]["family"].get<std::string>());
    if (it == m.end()) return false;            // a family this build has no engine for
    return it->second != SupportedModelFamily::error_whiper &&
           it->second != SupportedModelFamily::error_embedding;
}

inline std::pair<std::string, std::unique_ptr<AutoModel>> get_auto_model(const std::string& model_tag, model_list& available_models, oflm_rt::device* npu_device_inst) {
    if (available_models.is_model_supported(model_tag) == false) {
        // An unsupported tag used to return a Llama3 engine under the name
        // "llama3.2:1b" -- so `oflm serve` answered a request for a model it does
        // not have with a DIFFERENT MODEL, HTTP 200, after taking the loaded one off
        // the NPU. The error below went to the server console and nowhere else.
        //
        // The engine is null now. Every caller checks it; none may dereference it.
        header_print_r("ERROR", "Model tag '" << model_tag << "' is not supported. Please check the model list.");
        return std::make_pair(model_tag, std::unique_ptr<AutoModel>(nullptr));
    }

    std::unique_ptr<AutoModel> auto_chat_engine = nullptr;
    auto [new_model_tag, model_info] = available_models.get_model_info(model_tag);

    const auto& modelFamilyMap = model_family_map();
    const auto family_it = modelFamilyMap.find(model_info["details"]["family"].get<std::string>());
    if (family_it == modelFamilyMap.end()) {
        // `.at()` here used to throw std::out_of_range for a family this build has no
        // entry for -- an exception from a factory whose contract is "null on failure".
        header_print_r("ERROR", "Model '" << model_tag << "' is family '"
                       << model_info["details"]["family"].get<std::string>()
                       << "', which this build has no engine for.");
        return std::make_pair(model_tag, std::unique_ptr<AutoModel>(nullptr));
    }
    switch(family_it->second) {
        case SupportedModelFamily::llama3:
            auto_chat_engine = std::make_unique<Llama3>(npu_device_inst);
            break;
        case SupportedModelFamily::granite:
            auto_chat_engine = std::make_unique<Granite>(npu_device_inst);
            break;
        case SupportedModelFamily::deepseek_r1:
            auto_chat_engine = std::make_unique<DeepSeek_r1_8b>(npu_device_inst);
            break;
        case SupportedModelFamily::deepseek_r1_0528:
            auto_chat_engine = std::make_unique<DeepSeek_r1_0528_8b>(npu_device_inst);
            break;
        case SupportedModelFamily::qwen2:
            auto_chat_engine = std::make_unique<Qwen2>(npu_device_inst);
            break;
        case SupportedModelFamily::qwen2vl:
            auto_chat_engine = std::make_unique<Qwen2VL>(npu_device_inst);
            break;
        case SupportedModelFamily::qwen3:
            auto_chat_engine = std::make_unique<Qwen3>(npu_device_inst);
            break;
        case SupportedModelFamily::qwen3_it:
            auto_chat_engine = std::make_unique<Qwen3_IT>(npu_device_inst);
            break;
        case SupportedModelFamily::qwen3_tk:
            auto_chat_engine = std::make_unique<Qwen3_TK>(npu_device_inst);
            break;
        case SupportedModelFamily::gemma3:
            auto_chat_engine = std::make_unique<Gemma3>(npu_device_inst);
            break;
        case SupportedModelFamily::gemma3_text:
            auto_chat_engine = std::make_unique<Gemma3_Text_Only>(npu_device_inst);
            break;
        case SupportedModelFamily::gemma4e:
            auto_chat_engine = std::make_unique<Gemma4e>(npu_device_inst);
            break;
        case SupportedModelFamily::gemma4_12b:
            auto_chat_engine = std::make_unique<Gemma4_12B>(npu_device_inst);
            break;
        case SupportedModelFamily::gpt_oss:
            auto_chat_engine = std::make_unique<GPT_OSS>(npu_device_inst);
            break;
        case SupportedModelFamily::qwen3vl:
            auto_chat_engine = std::make_unique<Qwen3VL>(npu_device_inst);
            break;
        case SupportedModelFamily::qwen3_5:
            auto_chat_engine = std::make_unique<Qwen3_5VL>(npu_device_inst);
            break;
        case SupportedModelFamily::qwen3_5_omni:
            auto_chat_engine = std::make_unique<Qwen3_5_Omni>(npu_device_inst);
            break;
        case SupportedModelFamily::qwen3_6_moe:
            auto_chat_engine = std::make_unique<Qwen3_6_MOE>(npu_device_inst);
            break;
        case SupportedModelFamily::lfm2:
            auto_chat_engine = std::make_unique<LFM2>(npu_device_inst);
            break;
        case SupportedModelFamily::lfm2_5_tk:
            auto_chat_engine = std::make_unique<LFM2_5_TK>(npu_device_inst);
            break;
        case SupportedModelFamily::nanbeige:
            auto_chat_engine = std::make_unique<Nanbeige>(npu_device_inst);
            break;
        case SupportedModelFamily::phi4:
            auto_chat_engine = std::make_unique<Phi4>(npu_device_inst);
            break;
        case SupportedModelFamily::error_whiper:
        case SupportedModelFamily::error_embedding:
        default:
            // The SECOND substitution path, and the one the tag guard above does not
            // cover: `embed-gemma:300m` and `whisper-v3:turbo` ARE in model_list.json,
            // so they pass is_model_supported() and arrive here -- where a chat engine
            // is what the caller wants and this family cannot provide one. It used to
            // build a Llama3 and rename the request to "llama3.2:1b", so asking for an
            // embedding model over /v1/chat/completions was answered, HTTP 200, by a
            // different model under a name the client never sent.
            //
            // Null, like the unsupported tag. The caller decides what to say.
            header_print_r("ERROR", "Model '" << model_tag << "' is family '"
                           << model_info["details"]["family"].get<std::string>()
                           << "', which is not a chat model this build can run.");
            return std::make_pair(model_tag, std::unique_ptr<AutoModel>(nullptr));
    }
  
    return std::make_pair(new_model_tag, std::move(auto_chat_engine));
} 