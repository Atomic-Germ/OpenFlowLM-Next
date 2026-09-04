/// \file all_embedding_models.hpp
/// \brief get_auto_embedding_model func
/// \author FastFlowLM Team
/// \date 2026-09-03
/// \version 0.2.0
/// \note This is a header file for get_auto_embedding_model func
/// \note A REGISTRY, not a funnel: embed-gemma:300m keeps routing to
///       OpenGemma_Embedding, other known tags route to their own backend, and
///       an unknown tag is an explicit error.
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include "AutoEmbeddingModel/open_gemma_embedding.hpp"
#include "AutoEmbeddingModel/npue_embedding.hpp"


/// Which backend serves which tag.
///
/// THIS USED TO BE A FUNNEL. Every tag was rewritten to "embed-gemma:300m" and
/// then served by OpenGemma_Embedding, so asking for a model the tree did not
/// have returned embeddings for one it did -- silently, with a correctly
/// shaped, correctly normed, deterministic vector that nothing downstream
/// could tell was for the wrong model. AGENTS.md asks for an explicit error in
/// exactly this situation; a silent substitution is the one thing it forbids.
///
/// Adding a model is one line here plus an entry in model_list.json.
enum class EmbeddingBackend { OpenGemma, Npue };

inline const std::unordered_map<std::string, EmbeddingBackend>&
embedding_backend_registry() {
    // NpuEmbeddings serves seven encoders from four design sets, keyed by GEMM
    // GEOMETRY rather than by fine-tune name -- bge-base, gte-multilingual and
    // nomic share one set outright, which is this tree's own kernel policy
    // falling out for free. The tags below are the ones with a design set and
    // a validated container.
    static const std::unordered_map<std::string, EmbeddingBackend> reg = {
        {"embed-gemma:300m",        EmbeddingBackend::OpenGemma},
        {"bge-base:en-v1.5",        EmbeddingBackend::Npue},
        {"bge-small:en-v1.5",       EmbeddingBackend::Npue},
        {"bge-large:en-v1.5",       EmbeddingBackend::Npue},
        {"all-minilm:l6-v2",        EmbeddingBackend::Npue},
        {"nomic-embed-text:v1.5",   EmbeddingBackend::Npue},
        {"gte-multilingual:base",   EmbeddingBackend::Npue},
    };
    return reg;
}

inline std::string complete_simple_embedding_tag(std::string model_tag) {
    if (model_tag == "embed-gemma:300m")
        return "embed-gemma:300m";
    else
        return model_tag;
}


inline std::pair<std::string, std::unique_ptr<AutoEmbeddingModel>>
get_auto_embedding_model(const std::string& model_tag,
                         flm_rt::device* npu_device_inst) {

    const std::string tag = complete_simple_embedding_tag(model_tag);
    const auto& reg = embedding_backend_registry();
    const auto it = reg.find(tag);
    if (it == reg.end()) {
        std::string known;
        for (const auto& kv : reg)
            known += (known.empty() ? "" : ", ") + kv.first;
        throw std::runtime_error(
            "unknown embedding model '" + tag + "'. Known tags: " + known +
            ". Refusing to substitute one: an embedding for the wrong model is "
            "correctly shaped and correctly normed, so nothing downstream can "
            "tell that the answer is wrong.");
    }

    if (it->second == EmbeddingBackend::Npue) {
        return std::make_pair(
            tag, std::unique_ptr<AutoEmbeddingModel>(
                     new NpueEmbedding(npu_device_inst, tag)));
    }
    return std::make_pair(
        tag, std::unique_ptr<AutoEmbeddingModel>(
                 new OpenGemma_Embedding(npu_device_inst)));
}
