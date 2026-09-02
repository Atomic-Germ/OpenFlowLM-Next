/// \file engine.hpp
/// \brief Open CPU implementation of google/embeddinggemma-300m
/// \author FastFlowLM Team
/// \date 2026-08-30
/// \version 0.1.0
/// \note Fully open replacement for the closed libgemma_embedding.so stack.
///
/// Weights are loaded straight out of the model's safetensors files via the
/// weights_manifest.json the manifest tool emits (no Q4NX, no NPU, no closed
/// code). The forward pass mirrors transformers' Gemma3TextModel and was
/// validated against the official model-card similarity numbers:
///
///   "Which planet is known as the Red Planet?" vs 4 documents (document
///   prompt) -> [0.3011, 0.6359, 0.4930, 0.4889]
///
/// Reference oracle: src/open_embedding/tools/gemma3_reference.py
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"
#include "tokenizers_cpp.h"

namespace open_embedding {

class NpuMatmul;  // opaque NPU2 BF16 matmul backend (src/open_embedding/npu_matmul.cpp)

enum class task_type_t : uint8_t {
    task_query = 0,
    task_document = 1,
};

class Engine {
public:
    /// \brief Load config.json + weights_manifest.json from model_dir.
    bool load(const std::string& model_dir);

    /// \brief Embed text under the given task prefix (task_query default).
    /// \return 768-dim fp32 vector, L2-normalized after the contrastive head.
    std::vector<float> embed(const std::string& text, task_type_t task = task_type_t::task_query,
                             std::string* task_prefix_out = nullptr);

    /// \brief Embed text under an explicit task prefix.
    /// \return 768-dim fp32 vector, L2-normalized after the contrastive head.
    std::vector<float> embed_with_prefix(const std::string& text, const std::string& task_prefix);

    bool track_layers_ = false;
    std::vector<std::vector<float>> track_;
    const std::string& model_dir() const { return model_dir_; }
    std::vector<int32_t> debug_ids(const std::string& text, task_type_t task, std::string* pfx);
    std::vector<std::vector<float>> debug_stages(const std::string& text, task_type_t task);
    std::vector<std::vector<float>> debug_layers(const std::string& text, task_type_t task);
    std::map<std::string, std::vector<float>> debug_kp(const std::string& text, task_type_t task);

private:
    struct Tensor {
        std::string file;
        size_t offset = 0;
        std::vector<size_t> shape;
    };

    float embed_scale_ = 1.0f;
    float scaling_ = 1.0f;
    float eps_ = 1e-6f;
    size_t hidden_ = 0, intermediate_ = 0, head_dim_ = 0, num_layers_ = 0, vocab_ = 0;
    size_t n_heads_ = 0, n_kv_ = 0, sliding_window_ = 0, max_pos_ = 0, head_mid_ = 0;
    std::vector<std::string> layer_types_;

    std::string model_dir_;
    nlohmann::json cfg_;
    nlohmann::json manifest_;
    std::unordered_map<std::string, Tensor> tensors_;
    std::unordered_map<std::string, std::vector<float>> w_;
    std::unique_ptr<tokenizers::Tokenizer> tok_;

    /// Transposed [K,N] BF16 projection weights for the NPU backend.
    std::unordered_map<std::string, std::vector<uint16_t>> w_bf16_;
    /// NPU2 BF16 matmul backend (null when off or unavailable).
    std::shared_ptr<NpuMatmul> npu_;

    bool load_weights();
    bool load_npu();
    const std::vector<float>& weight(const std::string& name) const;
    std::vector<float> transformer(std::vector<int32_t> ids);

    static void rmsnorm(const float* x, const float* w, size_t rows, size_t dim, float eps,
                        std::vector<float>& out);
    static void gelu_tanh(std::vector<float>& x);
    // y[M,N] = sum_k x[M,K] * w[N,K]   (projection weights, applied transposed)
    static void matmul_t(const std::vector<float>& x, const std::vector<float>& w, size_t M, size_t K, size_t N,
                         std::vector<float>& y);
    // y[M,N] = sum_k x[M,K] * w[N,K]   (NPU BF16 offload when available)
    void matmul_t_npu(const std::string& name, const std::vector<float>& x, size_t M, size_t K, size_t N,
                      std::vector<float>& y);
    // y[M,N] = sum_k x[M,K] * w[K,N]   (row-major)
    static void matmul(const std::vector<float>& x, const std::vector<float>& w, size_t M, size_t K, size_t N,
                       std::vector<float>& y);
};

}  // namespace open_embedding