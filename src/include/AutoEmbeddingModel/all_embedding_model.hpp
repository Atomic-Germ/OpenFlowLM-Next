/// \file all_embedding_models.hpp
/// \brief get_auto_embedding_model func
/// \author FastFlowLM Team
/// \date 2026-09-02
/// \version 0.1.0
/// \note This is a header file for get_auto_embedding_model func
/// \note Always instantiates OpenGemma_Embedding (open replacement for closed libgemma_embedding.so).
#pragma once

#include <memory>
#include <string>
#include "AutoEmbeddingModel/open_gemma_embedding.hpp"


inline std::string complete_simple_embedding_tag(std::string model_tag) {
    if (model_tag == "embed-gemma:300m")
        return "embed-gemma:300m";
    else
        return model_tag;
}


inline std::pair<std::string, std::unique_ptr<AutoEmbeddingModel>> get_auto_embedding_model(const std::string& model_tag, flm_rt::device* npu_device_inst) {

    std::string new_model_tag = complete_simple_embedding_tag(model_tag);
    if (new_model_tag != "embed-gemma:300m") {
        new_model_tag = "embed-gemma:300m";
    }
    auto auto_embedding_engine = std::make_unique<OpenGemma_Embedding>(npu_device_inst);

    return std::make_pair(new_model_tag, std::move(auto_embedding_engine));
}
