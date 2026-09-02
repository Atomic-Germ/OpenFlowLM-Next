/// \file open_gemma_embedding.hpp
/// \brief OpenGemma_Embedding: AutoEmbeddingModel backed by the open engine.
/// \author FastFlowLM Team
/// \date 2026-09-02
/// \version 0.1.0
/// \note Fully open replacement for the closed gemma_embedding stack.
#pragma once

#include "auto_embedding_model.hpp"
#include "open_embedding/engine.hpp"

inline const char* open_task_prefix(embedding_task_type_t task_type) {
    switch (task_type) {
        case task_document:
            return "title: none | text: ";
        case task_clustering:
            return "task: clustering | query: ";
        case task_classification:
            return "task: classification | query: ";
        case task_code_retrieval:
            return "task: code retrieval | query: ";
        case task_multilabel_classification:
            return "task: classification | query: ";
        case task_sentence_similarity:
            return "task: sentence similarity | query: ";
        case task_summarization:
            return "task: summarization | query: ";
        case task_bitextmining:
        case task_search_result:
        case task_query:
        default:
            return "task: search result | query: ";
    }
}

class OpenGemma_Embedding : public AutoEmbeddingModel {
public:
    explicit OpenGemma_Embedding(flm_rt::device* npu_device_inst)
        : AutoEmbeddingModel(npu_device_inst, "embed-gemma:300m") {}

    ~OpenGemma_Embedding() override = default;

    void load_model(std::string model_path, json model_info, bool enable_preemption = false) override {
        (void)model_info;
        (void)enable_preemption;
        this->model_path = std::move(model_path);
        if (!engine_.load(this->model_path)) {
            throw std::runtime_error("OpenGemma_Embedding: failed to load model at " + this->model_path);
        }
        this->is_model_loaded = true;
    }

    std::vector<float> embed(std::string& text, embedding_task_type_t task_type) override {
        return engine_.embed_with_prefix(text, open_task_prefix(task_type));
    }

private:
    open_embedding::Engine engine_;
};
