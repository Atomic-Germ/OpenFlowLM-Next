/// \file engine.hpp
/// \brief Open CPU implementation of Gemma3 text (causal LM) models.
/// \author FastFlowLM Team
/// \date 2026-09-02
/// \note Fully open replacement for the closed libgemma_text_npu.so stack.
///
/// Weights load straight from the model's bf16 safetensors via the manifest the
/// builder emits. The forward pass mirrors the validated NumPy reference
/// (utilities/q4nx-build/q4nx/reference.py), which is the Phase 1 acceptance
/// oracle:
///
///   * embeddings scaled by sqrt(hidden_size)
///   * RMSNorm applied as x * (1 + w)
///   * dual-base RoPE (rope_theta global, rope_local_base_freq sliding)
///   * hybrid attention: every sliding_window_pattern-th layer is global, the
///     rest use a sliding band
///   * GQA with a single KV head for Gemma3-1B
///   * gelu_pytorch_tanh MLP with pre/post feed-forward norms
///   * tied LM head (embed_tokens reused as lm_head)
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"

namespace open_gemma3 {

class Engine {
public:
    /// \brief Load config.json + weights_manifest.json from model_dir.
    bool load(const std::string& model_dir);

    /// \brief Full-sequence forward; returns logits for the last position.
    /// \return vector of length vocab_size (fp32).
    ///
    /// This recomputes the whole sequence and never touches the KV cache. It is
    /// the validated reference path: `step()` must agree with it exactly, which
    /// is how the cache is tested.
    std::vector<float> prefill(const std::vector<int32_t>& ids);

    /// \brief Allocate a KV cache able to hold max_len positions.
    /// Memory is num_layers * 2 * max_len * (n_kv * head_dim) floats (fp32 for
    /// now; bf16 would halve it and is a follow-up).
    bool enable_cache(size_t max_len);

    /// \brief Reset the KV cache and the current position.
    void clear_context();

    /// \brief Incremental step: consume ids at the current position, append
    /// their keys/values to the cache, and return logits for the last id.
    std::vector<float> step(const std::vector<int32_t>& ids);

    size_t context_length() const { return pos_; }
    bool cache_enabled() const { return cache_len_ > 0; }
    size_t cache_capacity() const { return cache_len_; }

    size_t vocab() const { return vocab_; }
    size_t hidden() const { return hidden_; }
    size_t num_layers() const { return num_layers_; }
    size_t max_position_embeddings() const { return max_pos_; }
    const std::string& model_dir() const { return model_dir_; }
    const std::vector<std::string>& layer_types() const { return layer_types_; }

private:
    struct Tensor {
        std::string file;
        size_t offset = 0;
        std::vector<size_t> shape;
        std::string dtype;
    };

    // Architecture
    size_t hidden_ = 0, intermediate_ = 0, head_dim_ = 0, num_layers_ = 0;
    size_t vocab_ = 0, n_heads_ = 0, n_kv_ = 0, max_pos_ = 0;
    size_t sliding_window_ = 0, sliding_pattern_ = 6;
    double rope_theta_ = 1e6, rope_local_ = 1e4;
    float eps_ = 1e-6f, q_scalar_ = 0.0f;

    float embed_scale_ = 1.0f;
    float attn_scale_ = 1.0f;
    size_t gqa_groups_ = 1;
    std::vector<std::string> layer_types_;

    std::string model_dir_;
    nlohmann::json cfg_;
    nlohmann::json manifest_;
    std::unordered_map<std::string, Tensor> tensors_;
    std::unordered_map<std::string, std::vector<float>> w_;
    /// Explicit tied-weight mapping (for example lm_head -> embed_tokens).
    std::unordered_map<std::string, std::string> tied_;

    // KV cache: [layer][pos][kv_dim], laid out as layer-major contiguous blocks.
    std::vector<float> k_cache_, v_cache_;
    size_t cache_len_ = 0;   // positions the cache can hold
    size_t kv_dim_ = 0;      // n_kv * head_dim
    size_t pos_ = 0;         // next write position / current context length

    bool ensure_manifest();
    bool load_weights();
    std::string resolve_path(const std::string& p) const;
    const std::vector<float>& weight(const std::string& name) const;

    /// Shared forward body. When use_cache is set, keys/values are appended to
    /// (and read from) the KV cache starting at start_pos; otherwise the batch
    /// computes against itself (the full-recompute reference path).
    std::vector<float> forward_impl(const std::vector<int32_t>& ids, size_t start_pos,
                                    bool use_cache);
    /// Fill cos/sin tables for absolute positions [p0, p0 + T).
    void rope_tables_range(size_t p0, size_t T, double theta, std::vector<float>& cos,
                           std::vector<float>& sin) const;

    static void rmsnorm(const float* x, const float* w, size_t rows, size_t dim, float eps,
                        std::vector<float>& out);
    static void matmul_t(const std::vector<float>& x, const std::vector<float>& w, size_t M,
                         size_t K, size_t N, std::vector<float>& y);
    static void gelu_tanh(std::vector<float>& x);
    void rope_tables(size_t T, double theta, std::vector<float>& cos,
                     std::vector<float>& sin) const;
    static void apply_rope(std::vector<float>& x, size_t T, size_t heads, size_t head_dim,
                           const std::vector<float>& cos, const std::vector<float>& sin);
};

}  // namespace open_gemma3
