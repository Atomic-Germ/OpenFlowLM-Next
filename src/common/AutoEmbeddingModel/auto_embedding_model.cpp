/// \file auto_embedding_model.cpp
/// \brief AutoEmbeddingModel class
/// \author FastFlowLM Team
/// \date 2025-10-23
/// \version 0.9.24
/// \note This is a source file for the AutoEmbeddingModel class

#include "AutoEmbeddingModel/auto_embedding_model.hpp"

std::unordered_set<std::string> embeddingModelTags = {
    "embed-gemma", "embed-gemma:300m"
};

AutoEmbeddingModel::AutoEmbeddingModel(flm_rt::device* npu_device_inst, std::string current_model) {
    this->npu_device_inst = npu_device_inst;
    this->current_model = current_model;
}


std::string AutoEmbeddingModel::get_current_model() {
    return this->current_model;
}
