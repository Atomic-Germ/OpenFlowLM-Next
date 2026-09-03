/// \file engine.cpp
/// \brief Open CPU Gemma3 text engine.
/// \author FastFlowLM Team
/// \date 2026-09-02
/// \note Mirrors the validated NumPy reference in
/// utilities/q4nx-build/q4nx/reference.py. Any behavioural change here must be
/// reflected there (and vice versa) so the two stay bit-comparable.

#include "open_gemma3/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

namespace open_gemma3 {

using json = nlohmann::json;

static constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// ------------------------------------------------------------------ helpers

static bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

/// Widen a bf16 value to fp32. bf16 is the high 16 bits of an IEEE fp32, so a
/// 16-bit left shift and a reinterpret is exact. This matches the NumPy oracle
/// exactly, so oracle and engine agree bit for bit.
static float bf16_to_f32(uint16_t h) {
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static size_t dtype_size(const std::string& dtype) {
    if (dtype == "F32") return 4;
    if (dtype == "BF16" || dtype == "F16") return 2;
    if (dtype == "F64" || dtype == "I64") return 8;
    if (dtype == "I32" || dtype == "U32") return 4;
    if (dtype == "I16" || dtype == "U16") return 2;
    if (dtype == "I8" || dtype == "U8") return 1;
    return 0;
}

std::string Engine::resolve_path(const std::string& p) const {
    if (p.empty()) return p;
    std::filesystem::path path(p);
    if (path.is_absolute()) return path.string();
    return (std::filesystem::path(model_dir_) / path).string();
}

const std::vector<float>& Engine::weight(const std::string& name) const {
    static const std::vector<float> kEmpty;
    auto it = w_.find(name);
    return it == w_.end() ? kEmpty : it->second;
}

// ------------------------------------------------------------------- loading

bool Engine::load(const std::string& model_dir) {
    model_dir_ = model_dir;
    std::string cfg_text;
    if (!read_file((std::filesystem::path(model_dir_) / "config.json").string(), cfg_text) ||
        !(cfg_ = json::parse(cfg_text, nullptr, false)).is_object()) {
        std::fprintf(stderr, "open_gemma3: cannot read %s/config.json\n", model_dir.c_str());
        return false;
    }

    hidden_ = cfg_.value("hidden_size", 0u);
    intermediate_ = cfg_.value("intermediate_size", 0u);
    num_layers_ = cfg_.value("num_hidden_layers", 0u);
    n_heads_ = cfg_.value("num_attention_heads", 0u);
    n_kv_ = cfg_.value("num_key_value_heads", 0u);
    head_dim_ = cfg_.value("head_dim", n_heads_ ? hidden_ / n_heads_ : 0u);
    vocab_ = cfg_.value("vocab_size", 0u);
    max_pos_ = cfg_.value("max_position_embeddings", 32768u);
    eps_ = cfg_.value("rms_norm_eps", 1e-6f);
    sliding_window_ = cfg_.value("sliding_window", 512u);
    sliding_pattern_ = cfg_.value("sliding_window_pattern", 6u);
    rope_theta_ = cfg_.value("rope_theta", 1e6);
    rope_local_ = cfg_.value("rope_local_base_freq", 1e4);
    q_scalar_ = cfg_.value("query_pre_attn_scalar",
                           static_cast<float>(head_dim_ ? head_dim_ : 1));

    if (!hidden_ || !num_layers_ || !n_heads_ || !head_dim_ || !vocab_ || !n_kv_) {
        std::fprintf(stderr, "open_gemma3: incomplete config (hidden=%zu layers=%zu heads=%zu "
                             "head_dim=%zu vocab=%zu kv=%zu)\n",
                     hidden_, num_layers_, n_heads_, head_dim_, vocab_, n_kv_);
        return false;
    }

    embed_scale_ = std::sqrt(static_cast<float>(hidden_));
    attn_scale_ = 1.0f / std::sqrt(q_scalar_);
    gqa_groups_ = n_heads_ / n_kv_;

    // Gemma3 ships no `layer_types` key; derive it. Every
    // sliding_window_pattern-th layer is global (full attention).
    layer_types_.clear();
    for (size_t i = 0; i < num_layers_; ++i) {
        layer_types_.push_back(((i + 1) % sliding_pattern_ == 0) ? "full_attention"
                                                                 : "sliding_attention");
    }

    if (!load_weights()) return false;

    std::fprintf(stderr,
                 "open_gemma3: loaded %zu tensors (hidden=%zu layers=%zu heads=%zu kv=%zu "
                 "head_dim=%zu vocab=%zu, global layers:",
                 w_.size(), hidden_, num_layers_, n_heads_, n_kv_, head_dim_, vocab_);
    for (size_t i = 0; i < layer_types_.size(); ++i) {
        if (layer_types_[i] == "full_attention") std::fprintf(stderr, " %zu", i);
    }
    std::fprintf(stderr, ")\n");
    return true;
}

bool Engine::ensure_manifest() {
    namespace fs = std::filesystem;
    const fs::path dir(model_dir_);
    const fs::path mpath = dir / "weights_manifest.json";

    std::string mtext;
    if (read_file(mpath.string(), mtext)) manifest_ = json::parse(mtext, nullptr, false);
    if (manifest_.is_object() && manifest_.contains("tensors")) return true;

    // No manifest: fall back to scanning the model dir's safetensors files.
    try {
        std::vector<fs::path> shards;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
                shards.push_back(entry.path());
            }
        }
        std::sort(shards.begin(), shards.end());
        if (shards.empty()) {
            std::fprintf(stderr, "open_gemma3: no manifest and no safetensors in %s\n",
                         model_dir_.c_str());
            return false;
        }
        json tensors = json::object();
        for (const auto& shard : shards) {
            std::ifstream f(shard, std::ios::binary);
            uint64_t header_len = 0;
            f.read(reinterpret_cast<char*>(&header_len), 8);
            std::string header(header_len, '\0');
            f.read(&header[0], static_cast<std::streamsize>(header_len));
            auto hdr = json::parse(header, nullptr, false);
            const size_t base = 8 + static_cast<size_t>(header_len);
            const std::string rel = shard.filename().string();
            for (auto it = hdr.begin(); it != hdr.end(); ++it) {
                if (it.key() == "__metadata__") continue;
                const auto& meta = it.value();
                tensors[it.key()] = {
                    {"file", rel},
                    {"offset", base + meta["data_offsets"][0].get<size_t>()},
                    {"shape", meta["shape"]},
                    {"dtype", meta["dtype"]},
                };
            }
        }
        manifest_ = json{{"format", "flm-open-causal-manifest-v1"},
                         {"config", "config.json"},
                         {"tokenizer", "tokenizer.json"},
                         {"tensors", tensors}};
        std::ofstream(mpath) << manifest_.dump(1) << "\n";
        std::fprintf(stderr, "open_gemma3: generated weights_manifest.json (%zu tensors)\n",
                     tensors.size());
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "open_gemma3: manifest generation failed: %s\n", e.what());
        return false;
    }
}

bool Engine::load_weights() {
    if (!ensure_manifest()) return false;

    for (auto it = manifest_.at("tensors").begin(); it != manifest_.at("tensors").end(); ++it) {
        const auto& meta = it.value();
        Tensor t;
        t.file = meta.value("file", "model.safetensors");
        t.offset = meta.at("offset").get<size_t>();
        t.shape = meta.at("shape").get<std::vector<size_t>>();
        t.dtype = meta.value("dtype", "F32");
        tensors_[it.key()] = t;
    }

    for (const auto& [name, t] : tensors_) {
        size_t n = 1;
        for (size_t s : t.shape) n *= s;
        const size_t dsz = dtype_size(t.dtype);
        if (dsz == 0) {
            std::fprintf(stderr, "open_gemma3: unsupported dtype %s for %s\n", t.dtype.c_str(),
                         name.c_str());
            return false;
        }
        std::ifstream f(resolve_path(t.file), std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "open_gemma3: cannot open %s\n", t.file.c_str());
            return false;
        }
        f.seekg(static_cast<std::streamoff>(t.offset));
        std::vector<float> out(n);
        if (t.dtype == "F32") {
            f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n * 4));
        } else if (t.dtype == "BF16") {
            std::vector<uint16_t> raw(n);
            f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(n * 2));
            for (size_t i = 0; i < n; ++i) out[i] = bf16_to_f32(raw[i]);
        } else {
            std::fprintf(stderr, "open_gemma3: dtype %s not yet supported for %s\n",
                         t.dtype.c_str(), name.c_str());
            return false;
        }
        if (!f) {
            std::fprintf(stderr, "open_gemma3: short read on %s\n", t.file.c_str());
            return false;
        }
        w_[name] = std::move(out);
    }

    // Record explicit tied weights so the forward pass does not guess.
    if (manifest_.contains("tied")) {
        for (auto it = manifest_.at("tied").begin(); it != manifest_.at("tied").end(); ++it) {
            tied_[it.key()] = it.value().get<std::string>();
        }
    }
    return true;
}

// ----------------------------------------------------------------- primitives

void Engine::rmsnorm(const float* x, const float* w, size_t rows, size_t dim, float eps,
                     std::vector<float>& out) {
    out.resize(rows * dim);
    for (size_t r = 0; r < rows; ++r) {
        const float* xr = x + r * dim;
        float sum = 0.0f;
        for (size_t i = 0; i < dim; ++i) sum += xr[i] * xr[i];
        float scale = 1.0f / std::sqrt(sum / static_cast<float>(dim) + eps);
        float* orow = out.data() + r * dim;
        for (size_t i = 0; i < dim; ++i) orow[i] = xr[i] * scale * (1.0f + w[i]);
    }
}

void Engine::matmul_t(const std::vector<float>& x, const std::vector<float>& w, size_t M,
                      size_t K, size_t N, std::vector<float>& y) {
    // y[M,N] = sum_k x[M,K] * w[N,K]  (weights stored out-major)
    y.assign(M * N, 0.0f);
    for (size_t m = 0; m < M; ++m) {
        const float* xr = x.data() + m * K;
        float* yr = y.data() + m * N;
        for (size_t n = 0; n < N; ++n) {
            const float* wr = w.data() + n * K;
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) acc += xr[k] * wr[k];
            yr[n] = acc;
        }
    }
}

void Engine::gelu_tanh(std::vector<float>& x) {
    constexpr float kSqrt2Pi = 0.7978845608028654f;
    for (float& v : x) {
        v = 0.5f * v * (1.0f + std::tanh(kSqrt2Pi * (v + 0.044715f * v * v * v)));
    }
}

void Engine::rope_tables(size_t T, double theta, std::vector<float>& cos,
                         std::vector<float>& sin) const {
    cos.assign(T * head_dim_, 0.0f);
    sin.assign(T * head_dim_, 0.0f);
    const size_t half = head_dim_ / 2;
    for (size_t i = 0; i < half; ++i) {
        const double inv = 1.0 / std::pow(theta, (2.0 * static_cast<double>(i)) /
                                                     static_cast<double>(head_dim_));
        for (size_t p = 0; p < T; ++p) {
            const double freq = static_cast<double>(p) * inv;
            const float c = static_cast<float>(std::cos(freq));
            const float s = static_cast<float>(std::sin(freq));
            cos[p * head_dim_ + i] = c;
            cos[p * head_dim_ + i + half] = c;
            sin[p * head_dim_ + i] = s;
            sin[p * head_dim_ + i + half] = s;
        }
    }
}

void Engine::apply_rope(std::vector<float>& x, size_t T, size_t heads, size_t head_dim,
                        const std::vector<float>& cos, const std::vector<float>& sin) {
    const size_t half = head_dim / 2;
    for (size_t t = 0; t < T; ++t) {
        const float* c = cos.data() + t * head_dim;
        const float* s = sin.data() + t * head_dim;
        for (size_t h = 0; h < heads; ++h) {
            float* r = x.data() + (t * heads + h) * head_dim;
            for (size_t d = 0; d < half; ++d) {
                const float v0 = r[d];
                const float v1 = r[d + half];
                r[d] = v0 * c[d] - v1 * s[d];
                r[d + half] = v1 * c[d] + v0 * s[d];
            }
        }
    }
}

// -------------------------------------------------------------------- forward

std::vector<float> Engine::prefill(const std::vector<int32_t>& ids) {
    const size_t T = ids.size();
    const size_t HD = hidden_;
    const size_t QD = n_heads_ * head_dim_;
    const size_t ND = n_kv_ * head_dim_;

    std::vector<float> h(T * HD);
    {
        const std::vector<float>& emb = weight("model.embed_tokens.weight");
        for (size_t t = 0; t < T; ++t) {
            const float* e = emb.data() + static_cast<size_t>(ids[t]) * HD;
            float* dst = h.data() + t * HD;
            for (size_t i = 0; i < HD; ++i) dst[i] = e[i] * embed_scale_;
        }
    }

    std::vector<float> cos_g, sin_g, cos_l, sin_l;
    rope_tables(T, rope_theta_, cos_g, sin_g);
    rope_tables(T, rope_local_, cos_l, sin_l);

    std::vector<float> x, q, k, v, o, scores, gate, up, mlp;
    for (size_t L = 0; L < num_layers_; ++L) {
        const std::string p = "model.layers." + std::to_string(L) + ".";
        const bool is_global = layer_types_[L] == "full_attention";
        const std::vector<float>& cos = is_global ? cos_g : cos_l;
        const std::vector<float>& sin = is_global ? sin_g : sin_l;

        rmsnorm(h.data(), weight(p + "input_layernorm.weight").data(), T, HD, eps_, x);

        matmul_t(x, weight(p + "self_attn.q_proj.weight"), T, HD, QD, q);
        matmul_t(x, weight(p + "self_attn.k_proj.weight"), T, HD, ND, k);
        matmul_t(x, weight(p + "self_attn.v_proj.weight"), T, HD, ND, v);

        // Per-head q/k norms over head_dim.
        {
            const std::vector<float>& qn = weight(p + "self_attn.q_norm.weight");
            const std::vector<float>& kn = weight(p + "self_attn.k_norm.weight");
            for (size_t t = 0; t < T; ++t) {
                for (size_t hh = 0; hh < n_heads_; ++hh) {
                    float* r = q.data() + (t * n_heads_ + hh) * head_dim_;
                    float sum = 0.0f;
                    for (size_t d = 0; d < head_dim_; ++d) sum += r[d] * r[d];
                    const float sc = 1.0f / std::sqrt(sum / static_cast<float>(head_dim_) + eps_);
                    for (size_t d = 0; d < head_dim_; ++d) r[d] = r[d] * sc * (1.0f + qn[d]);
                }
                for (size_t kv = 0; kv < n_kv_; ++kv) {
                    float* r = k.data() + (t * n_kv_ + kv) * head_dim_;
                    float sum = 0.0f;
                    for (size_t d = 0; d < head_dim_; ++d) sum += r[d] * r[d];
                    const float sc = 1.0f / std::sqrt(sum / static_cast<float>(head_dim_) + eps_);
                    for (size_t d = 0; d < head_dim_; ++d) r[d] = r[d] * sc * (1.0f + kn[d]);
                }
            }
        }

        apply_rope(q, T, n_heads_, head_dim_, cos, sin);
        apply_rope(k, T, n_kv_, head_dim_, cos, sin);

        // Attention over T positions, GQA: query head -> kv head h/groups.
        o.assign(T * QD, 0.0f);
        scores.assign(n_heads_ * T * T, 0.0f);
        for (size_t hh = 0; hh < n_heads_; ++hh) {
            const size_t kv_id = hh / gqa_groups_;
            float* srow_base = scores.data() + hh * T * T;
            for (size_t t = 0; t < T; ++t) {
                const float* qp = q.data() + (t * n_heads_ + hh) * head_dim_;
                float* srow = srow_base + t * T;
                for (size_t kp = 0; kp < T; ++kp) {
                    // Causal; sliding layers additionally restrict to the window.
                    if (kp > t) {
                        srow[kp] = kNegInf;
                        continue;
                    }
                    if (!is_global && (t - kp) >= sliding_window_) {
                        srow[kp] = kNegInf;
                        continue;
                    }
                    const float* kpv = k.data() + (kp * n_kv_ + kv_id) * head_dim_;
                    float acc = 0.0f;
                    for (size_t d = 0; d < head_dim_; ++d) acc += qp[d] * kpv[d];
                    srow[kp] = acc * attn_scale_;
                }
            }
            // Softmax per row.
            for (size_t t = 0; t < T; ++t) {
                float* srow = srow_base + t * T;
                float mx = -std::numeric_limits<float>::infinity();
                for (size_t kp = 0; kp <= t; ++kp) mx = std::max(mx, srow[kp]);
                float denom = 0.0f;
                for (size_t kp = 0; kp <= t; ++kp) {
                    srow[kp] = std::exp(srow[kp] - mx);
                    denom += srow[kp];
                }
                if (denom > 0.0f) {
                    for (size_t kp = 0; kp <= t; ++kp) srow[kp] /= denom;
                }
            }
            // o = att @ v
            for (size_t t = 0; t < T; ++t) {
                const float* srow = srow_base + t * T;
                float* orow = o.data() + (t * n_heads_ + hh) * head_dim_;
                for (size_t kp = 0; kp <= t; ++kp) {
                    const float a = srow[kp];
                    if (a == 0.0f) continue;
                    const float* vv = v.data() + (kp * n_kv_ + kv_id) * head_dim_;
                    for (size_t d = 0; d < head_dim_; ++d) orow[d] += a * vv[d];
                }
            }
        }

        std::vector<float> attn_out(T * HD);
        matmul_t(o, weight(p + "self_attn.o_proj.weight"), T, QD, HD, attn_out);
        rmsnorm(attn_out.data(), weight(p + "post_attention_layernorm.weight").data(), T, HD,
                eps_, x);
        for (size_t i = 0; i < h.size(); ++i) h[i] += x[i];

        rmsnorm(h.data(), weight(p + "pre_feedforward_layernorm.weight").data(), T, HD, eps_, x);
        matmul_t(x, weight(p + "mlp.gate_proj.weight"), T, HD, intermediate_, gate);
        matmul_t(x, weight(p + "mlp.up_proj.weight"), T, HD, intermediate_, up);
        gelu_tanh(gate);
        for (size_t i = 0; i < gate.size(); ++i) gate[i] *= up[i];
        matmul_t(gate, weight(p + "mlp.down_proj.weight"), T, intermediate_, HD, mlp);
        rmsnorm(mlp.data(), weight(p + "post_feedforward_layernorm.weight").data(), T, HD, eps_,
                x);
        for (size_t i = 0; i < h.size(); ++i) h[i] += x[i];
    }

    std::vector<float> final_hidden(HD);
    rmsnorm(h.data() + (T - 1) * HD, weight("model.norm.weight").data(), 1, HD, eps_,
            final_hidden);

    // Tied LM head: lm_head.weight reuses embed_tokens unless overridden.
    std::string lm_name = "model.embed_tokens.weight";
    if (w_.count("lm_head.weight")) {
        lm_name = "lm_head.weight";
    } else if (tied_.count("lm_head.weight")) {
        lm_name = tied_.at("lm_head.weight");
    }

    std::vector<float> logits(vocab_);
    matmul_t(final_hidden, weight(lm_name), 1, HD, vocab_, logits);
    return logits;
}

}  // namespace open_gemma3
