#include "open_embedding/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

#include "nlohmann/json.hpp"
#include "tokenizers_cpp.h"

#ifdef FLM_USE_OPEN_EMBEDDING_NPU
#include "npu_utils/npu_utils_matmul.hpp"
#endif

namespace open_embedding {

using json = nlohmann::json;

static constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// FP32 <-> BF16 helpers (round-to-nearest-even on the way down).
static uint16_t f32_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t lsb = (x >> 16) & 1u;
    x += 0x7fffu + lsb;
    return static_cast<uint16_t>(x >> 16);
}

static float bf16_to_f32(uint16_t h) {
    float f;
    uint32_t x = static_cast<uint32_t>(h) << 16;
    std::memcpy(&f, &x, 4);
    return f;
}

// Projection tensors that the NPU backend can serve (2-D, [N,K] fp32).
// Only the five large per-layer projections; k/v and contrastive head stay on CPU
// to preserve numerical fidelity for the final E8 threshold.
static bool is_npu_projection_weight(const std::string& name) {
    static const char* kSuffixes[] = {
        "q_proj.weight", "o_proj.weight",
        "gate_proj.weight", "up_proj.weight", "down_proj.weight",
    };
    for (const char* s : kSuffixes) {
        if (name.size() >= std::strlen(s) &&
            name.compare(name.size() - std::strlen(s), std::strlen(s), s) == 0)
            return true;
    }
    return false;
}

static const char* kPrompt(task_type_t t) {
    switch (t) {
        case task_type_t::task_document: return "title: none | text: ";
        default:                        return "task: search result | query: ";
    }
}

// ---------------------------------------------------------------- helpers

static bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

bool Engine::load(const std::string& model_dir) {
    model_dir_ = model_dir;
    std::string cfg_path = (std::filesystem::path(model_dir) / "config.json").string();
    std::string cfg_text;
    if (!read_file(cfg_path, cfg_text) || !(cfg_ = json::parse(cfg_text, nullptr, false)).is_object()) {
        std::fprintf(stderr, "open_embedding: cannot read %s\n", cfg_path.c_str());
        return false;
    }
    hidden_         = cfg_.value("hidden_size", 0u);
    intermediate_   = cfg_.value("intermediate_size", 0u);
    head_dim_       = cfg_.value("head_dim", 0u);
    n_heads_        = cfg_.value("num_attention_heads", 0u);
    n_kv_           = cfg_.value("num_key_value_heads", 0u);
    num_layers_     = cfg_.value("num_hidden_layers", 0u);
    vocab_          = cfg_.value("vocab_size", 0u);
    sliding_window_ = cfg_.value("sliding_window", 0u);
    max_pos_        = cfg_.value("max_position_embeddings", 2048u);
    eps_            = cfg_.value("rms_norm_eps", 1e-6f);
    embed_scale_    = std::sqrt(static_cast<float>(hidden_));
    scaling_        = 1.0f / std::sqrt(cfg_.value("query_pre_attn_scalar", 1.0f));
    layer_types_    = cfg_.value("layer_types", std::vector<std::string>{});
    if (layer_types_.size() != num_layers_) {
        std::fprintf(stderr, "open_embedding: layer_types %zu != num_hidden_layers %zu\n",
                     layer_types_.size(), num_layers_);
        return false;
    }
    if (!load_weights()) return false;
    load_npu();
    return true;
}

bool Engine::load_npu() {
#ifdef FLM_USE_OPEN_EMBEDDING_NPU
    if (std::getenv("FLM_NPU_DISABLE")) return false;
    const std::string asset_dir =
        (std::filesystem::path(model_dir_) / "npu_matmul_f32").string();
    const char* dev_id = std::getenv("FLM_NPU_DEVICE_ID");
    npu_ = std::make_shared<NpuMatmul>();
    if (!npu_->init(asset_dir, dev_id ? dev_id : "")) {
        npu_.reset();
        std::fprintf(stderr, "open_embedding: running CPU-only\n");
        return false;
    }
    // Precompute transposed [K,N] BF16 projection weights once.
    for (const auto& [name, wvec] : w_) {
        if (!is_npu_projection_weight(name)) continue;
        auto tit = tensors_.find(name);
        if (tit == tensors_.end() || tit->second.shape.size() != 2) continue;
        const size_t N = tit->second.shape[0];
        const size_t K = tit->second.shape[1];
        if (wvec.size() != N * K) continue;
        std::vector<uint16_t> bf((size_t)K * N);
        for (size_t n = 0; n < N; n++) {
            const float* wr = wvec.data() + n * K;
            uint16_t* br = bf.data() + n;  // [k,n] -> row k, column n
            for (size_t k = 0; k < K; k++) br[k * N] = f32_to_bf16(wr[k]);
        }
        w_bf16_[name] = std::move(bf);
    }
    std::fprintf(stderr, "open_embedding: NPU matmul ready (%zu bf16 weights)\n",
                 w_bf16_.size());
#else
    (void)model_dir_;
#endif
    return true;
}

bool Engine::load_weights() {
    std::string mpath = (std::filesystem::path(model_dir_) / "weights_manifest.json").string();
    std::string mtext;
    if (!read_file(mpath, mtext) || !(manifest_ = json::parse(mtext, nullptr, false)).is_object()) {
        std::fprintf(stderr, "open_embedding: missing weights_manifest.json (run "
                             "src/open_embedding/tools/make_manifest.py)\n");
        return false;
    }
    if ((std::filesystem::path(manifest_.value("config", "")) !=
         std::filesystem::path(model_dir_) / "config.json")) {
        std::fprintf(stderr, "open_embedding: manifest config path mismatch\n");
        return false;
    }
    std::string tok_path = manifest_.value("tokenizer", "");
    std::string tok_blob;
    if (tok_path.empty() || !read_file(tok_path, tok_blob)) {
        std::fprintf(stderr, "open_embedding: cannot read tokenizer.json\n");
        return false;
    }
    tok_ = tokenizers::Tokenizer::FromBlobJSON(tok_blob);
    if (!tok_) {
        std::fprintf(stderr, "open_embedding: tokenizer init failed\n");
        return false;
    }

    for (auto it = manifest_.at("tensors").begin(); it != manifest_.at("tensors").end(); ++it) {
        const auto& meta = it.value();
        Tensor t;
        t.file   = meta.at("file").get<std::string>();
        t.offset = meta.at("offset").get<size_t>();
        t.shape  = meta.at("shape").get<std::vector<size_t>>();
        tensors_[it.key()] = t;
    }
    head_mid_ = tensors_.at("2_Dense.linear.weight").shape.at(0);
    std::vector<float> buf;
    for (auto& [name, t] : tensors_) {
        size_t n = 1;
        for (size_t s : t.shape) n *= s;
        buf.resize(n);
        std::ifstream f(t.file, std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "open_embedding: cannot open %s\n", t.file.c_str());
            return false;
        }
        f.seekg(static_cast<std::streamoff>(t.offset));
        f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n * sizeof(float)));
        if (!f || n == 0) {
            std::fprintf(stderr, "open_embedding: short read on %s\n", t.file.c_str());
            return false;
        }
        w_[name] = buf;
    }
    return true;
}

const std::vector<float>& Engine::weight(const std::string& name) const {
    static const std::vector<float> kEmpty;
    auto it = w_.find(name);
    return it == w_.end() ? kEmpty : it->second;
}

// ------------------------------------------------------------ math kernels

void Engine::rmsnorm(const float* x, const float* w, size_t rows, size_t dim, float eps,
                     std::vector<float>& out) {
    out.resize(rows * dim);
    for (size_t r = 0; r < rows; r++) {
        const float* xr = x + r * dim;
        float sum = 0.0f;
        for (size_t i = 0; i < dim; i++) sum += xr[i] * xr[i];
        float scale = 1.0f / std::sqrt(sum / static_cast<float>(dim) + eps);
        float* orow = out.data() + r * dim;
        for (size_t i = 0; i < dim; i++) orow[i] = xr[i] * scale * (1.0f + w[i]);
    }
}

void Engine::gelu_tanh(std::vector<float>& x) {
    constexpr float kSqrt2Pi = 0.7978845608028654f;
    for (size_t i = 0; i < x.size(); i++) {
        const float v = x[i];
        x[i] = 0.5f * v * (1.0f + std::tanh(kSqrt2Pi * (v + 0.044715f * v * v * v)));
    }
}

void Engine::matmul_t(const std::vector<float>& x, const std::vector<float>& w, size_t M, size_t K, size_t N,
                      std::vector<float>& y) {
    // y[M,N] = sum_k x[M,K] * w[N,K]  (projection weights stored out-major)
    y.assign(M * N, 0.0f);
    for (size_t m = 0; m < M; m++) {
        const float* xr = x.data() + m * K;
        float* yr = y.data() + m * N;
        for (size_t n = 0; n < N; n++) {
            const float* wr = w.data() + n * K;
            float acc = 0.0f;
            for (size_t k = 0; k < K; k++) acc += xr[k] * wr[k];
            yr[n] = acc;
        }
    }
}

void Engine::matmul_t_npu(const std::string& name, const std::vector<float>& x, size_t M, size_t K, size_t N,
                          std::vector<float>& y) {
    const std::vector<float>& w = weight(name);
#ifdef FLM_USE_OPEN_EMBEDDING_NPU
    if (npu_ && M > 0 && M <= 2048) {
        const int m_pad = npu_->m_pad_for(static_cast<int>(K), static_cast<int>(N),
                                          static_cast<int>(M));
        auto it = w_bf16_.find(name);
        if (m_pad > 0 && it != w_bf16_.end() &&
            it->second.size() == static_cast<size_t>(K) * N) {
            std::vector<uint16_t> a_pad(static_cast<size_t>(m_pad) * K, 0);
            for (size_t m = 0; m < M; m++) {
                const float* xr = x.data() + m * K;
                uint16_t* ar = a_pad.data() + m * K;
                for (size_t k = 0; k < K; k++) ar[k] = f32_to_bf16(xr[k]);
            }
            std::vector<float> c_pad(static_cast<size_t>(m_pad) * N);
            if (npu_->matmul_bf16(m_pad, static_cast<int>(K), static_cast<int>(N),
                                  a_pad.data(), it->second.data(), c_pad.data())) {
                y.assign(M * N, 0.0f);
                for (size_t m = 0; m < M; m++) {
                    const float* cr = c_pad.data() + m * N;
                    float* yr = y.data() + m * N;
                    for (size_t n = 0; n < N; n++) yr[n] = cr[n];
                }
                return;
            }
        }
    }
#endif
    matmul_t(x, w, M, K, N, y);
}

void Engine::matmul(const std::vector<float>& x, const std::vector<float>& w, size_t M, size_t K, size_t N,
                    std::vector<float>& y) {
    // y[M,N] = sum_k x[M,K] * w[K,N]   (row-major)
    y.assign(M * N, 0.0f);
    for (size_t m = 0; m < M; m++) {
        const float* xr = x.data() + m * K;
        float* yr = y.data() + m * N;
        for (size_t n = 0; n < N; n++) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; k++) acc += xr[k] * w[k * N + n];
            yr[n] = acc;
        }
    }
}

// ---------------------------------------------------------------- forward

std::vector<float> Engine::transformer(std::vector<int32_t> ids) {
    const size_t T = ids.size();
    const size_t HD = hidden_, QD = n_heads_ * head_dim_, ND = n_kv_ * head_dim_;

    std::vector<float> h(T * HD);
    {
        const auto& em = weight("embed_tokens.weight");
        for (size_t t = 0; t < T; t++) {
            const float* e = em.data() + static_cast<size_t>(ids[t]) * HD;
            float* dst = h.data() + t * HD;
            for (size_t i = 0; i < HD; i++) dst[i] = e[i] * embed_scale_;
        }
    }

    // Precompute rotary cos/sin per position and layer type: cos[p*D+d], sin[p*D+d]
    if (track_layers_) track_.push_back(h);
    std::vector<float> cos_tbl[2], sin_tbl[2];  // 0 = sliding_attention, 1 = full_attention
    {
        const char* names[2] = {"sliding_attention", "full_attention"};
        for (int idx = 0; idx < 2; idx++) {
            const std::string lt = names[idx];
            const bool present =
                std::find(layer_types_.begin(), layer_types_.end(), lt) != layer_types_.end();
            if (!present) {
                cos_tbl[idx].assign(max_pos_ * head_dim_, 1.0f);
                sin_tbl[idx].assign(max_pos_ * head_dim_, 0.0f);
                continue;
            }
            const float base = (lt == "sliding_attention")
                                   ? cfg_.value("rope_local_base_freq", 10000.0f)
                                   : cfg_.value("rope_theta", 10000.0f);
            std::vector<float> inv(head_dim_ / 2);
            for (size_t i = 0; i < head_dim_ / 2; i++)
                inv[i] = 1.0f / std::pow(base, (2.0f * static_cast<float>(i)) / static_cast<float>(head_dim_));
            cos_tbl[idx].resize(max_pos_ * head_dim_);
            sin_tbl[idx].resize(max_pos_ * head_dim_);
            for (size_t p = 0; p < max_pos_; p++) {
                for (size_t i = 0; i < head_dim_ / 2; i++) {
                    const float f = static_cast<float>(p) * inv[i];
                    const float c = std::cos(f), s = std::sin(f);
                    cos_tbl[idx][p * head_dim_ + i]                   = c;
                    cos_tbl[idx][p * head_dim_ + i + head_dim_ / 2]   = c;
                    sin_tbl[idx][p * head_dim_ + i]                   = s;
                    sin_tbl[idx][p * head_dim_ + i + head_dim_ / 2]   = s;
                }
            }
        }
    }
    auto table_idx = [](const std::string& lt) -> int { return lt == "sliding_attention" ? 0 : 1; };

    std::vector<float> x, o, buf;
    for (size_t L = 0; L < num_layers_; L++) {
        const std::string lt = layer_types_[L];
        const std::string pfx = "layers." + std::to_string(L) + ".";
        const int ci = table_idx(lt);

        rmsnorm(h.data(), weight(pfx + "input_layernorm.weight").data(), T, HD, eps_, x);

        // --- projections
        std::vector<float> q, kt, v;
        matmul_t_npu(pfx + "self_attn.q_proj.weight", x, T, HD, QD, q);
        matmul_t_npu(pfx + "self_attn.k_proj.weight", x, T, HD, ND, kt);
        matmul_t_npu(pfx + "self_attn.v_proj.weight", x, T, HD, ND, v);

        // --- q/k norm per head (over head_dim) ---
        std::vector<float> qn(T * QD), kn(T * ND);
        for (size_t t = 0; t < T; t++) {
            for (size_t h_id = 0; h_id < n_heads_; h_id++) {
                const float* src = q.data() + (t * n_heads_ + h_id) * head_dim_;
                rmsnorm(src, weight(pfx + "self_attn.q_norm.weight").data(), 1, head_dim_, eps_, buf);
                std::memcpy(qn.data() + (t * n_heads_ + h_id) * head_dim_, buf.data(), head_dim_ * sizeof(float));
            }
            for (size_t kv_id = 0; kv_id < n_kv_; kv_id++) {
                const float* src = kt.data() + (t * n_kv_ + kv_id) * head_dim_;
                rmsnorm(src, weight(pfx + "self_attn.k_norm.weight").data(), 1, head_dim_, eps_, buf);
                std::memcpy(kn.data() + (t * n_kv_ + kv_id) * head_dim_, buf.data(), head_dim_ * sizeof(float));
            }
        }

        // --- rotary positions (rotate_half) on q and k ---
        {
            const float* cos_p = cos_tbl[ci].data();
            const float* sin_p = sin_tbl[ci].data();
            const size_t half = head_dim_ / 2;
            for (size_t t = 0; t < T; t++) {
                const float* c = cos_p + t * head_dim_;
                const float* s = sin_p + t * head_dim_;
                for (size_t h_id = 0; h_id < n_heads_; h_id++) {
                    float* r = qn.data() + (t * n_heads_ + h_id) * head_dim_;
                    for (size_t d = 0; d < half; d++) {
                        float v0 = r[d], v1 = r[d + half];
                        r[d]         = v0 * c[d] - v1 * s[d];
                        r[d + half]  = v1 * c[d] + v0 * s[d];
                    }
                }
                for (size_t kv_id = 0; kv_id < n_kv_; kv_id++) {
                    float* r = kn.data() + (t * n_kv_ + kv_id) * head_dim_;
                    for (size_t d = 0; d < half; d++) {
                        float v0 = r[d], v1 = r[d + half];
                        r[d]        = v0 * c[d] - v1 * s[d];
                        r[d + half] = v1 * c[d] + v0 * s[d];
                    }
                }
            }
        }

        // --- attention scores (H,T,T), masked ---
        std::vector<float> scores(n_heads_ * T * T, kNegInf);
        for (size_t h_id = 0; h_id < n_heads_; h_id++) {
            for (size_t t = 0; t < T; t++) {
                const float* qp = qn.data() + (t * n_heads_ + h_id) * head_dim_;
                float* srow = scores.data() + (h_id * T + t) * T;
                for (size_t kpos = 0; kpos < T; kpos++) {
                    const bool allowed = lt == "full_attention" ||
                        (t >= kpos ? t - kpos : kpos - t) < sliding_window_;
                    if (!allowed) continue;
                    const float* kp = kn.data() + (kpos * n_kv_) * head_dim_;
                    float acc = 0.0f;
                    for (size_t d = 0; d < head_dim_; d++) acc += qp[d] * kp[d];
                    srow[kpos] = acc * scaling_;
                }
            }
        }

        // --- softmax (fp32, max-subtracted) ---
        std::vector<float> att(n_heads_ * T * T);
        for (size_t r = 0; r < scores.size() / T; r++) {
            float* srow = scores.data() + r * T;
            float mx = srow[0];
            for (size_t kpos = 1; kpos < T; kpos++) mx = std::max(mx, srow[kpos]);
            float denom = 0.0f;
            for (size_t kpos = 0; kpos < T; kpos++) {
                if (std::isfinite(srow[kpos])) {
                    srow[kpos] = std::exp(srow[kpos] - mx);
                    denom += srow[kpos];
                } else {
                    srow[kpos] = 0.0f;
                }
            }
            if (denom > 0.0f) {
                for (size_t kpos = 0; kpos < T; kpos++) srow[kpos] /= denom;
            }
            std::memcpy(att.data() + r * T, srow, T * sizeof(float));
        }

        // --- o = att @ v  (GQA: query head h_id attends to kv head h_id / groups) ---
        std::vector<float> oatt(T * QD, 0.0f);
        const size_t num_query_groups = n_heads_ / n_kv_;
        for (size_t h_id = 0; h_id < n_heads_; h_id++) {
            const size_t kv_id = h_id / num_query_groups;
            for (size_t t = 0; t < T; t++) {
                const float* arow = att.data() + (h_id * T + t) * T;
                float* orow = oatt.data() + (t * n_heads_ + h_id) * head_dim_;
                for (size_t d = 0; d < head_dim_; d++) {
                    float acc = 0.0f;
                    for (size_t kpos = 0; kpos < T; kpos++)
                        acc += arow[kpos] * v[(kpos * n_kv_ + kv_id) * head_dim_ + d];
                    orow[d] = acc;
                }
            }
        }

        matmul_t_npu(pfx + "self_attn.o_proj.weight", oatt, T, QD, HD, o);
        rmsnorm(o.data(), weight(pfx + "post_attention_layernorm.weight").data(), T, HD, eps_, x);
        for (size_t i = 0; i < h.size(); i++) h[i] += x[i];

        // --- MLP (gate, up) → down ---
        std::vector<float> gate, up;
        rmsnorm(h.data(), weight(pfx + "pre_feedforward_layernorm.weight").data(), T, HD, eps_, x);
        matmul_t_npu(pfx + "mlp.gate_proj.weight", x, T, HD, intermediate_, gate);
        matmul_t_npu(pfx + "mlp.up_proj.weight", x, T, HD, intermediate_, up);
        gelu_tanh(gate);
        for (size_t i = 0; i < gate.size(); i++) gate[i] *= up[i];
        matmul_t_npu(pfx + "mlp.down_proj.weight", gate, T, intermediate_, HD, o);
        rmsnorm(o.data(), weight(pfx + "post_feedforward_layernorm.weight").data(), T, HD, eps_, x);
        for (size_t i = 0; i < h.size(); i++) h[i] += x[i];
        if (track_layers_) track_.push_back(h);
    }

    rmsnorm(h.data(), weight("norm.weight").data(), T, HD, eps_, x);
    if (track_layers_) track_.push_back(x);
    return x;
}

std::vector<int32_t> Engine::debug_ids(const std::string& text, task_type_t task, std::string* pfx) {
    if (!tok_) return {};
    *pfx = kPrompt(task);
    std::vector<int32_t> ids = tok_->Encode(std::string(*pfx) + text);
    ids.insert(ids.begin(), static_cast<int32_t>(cfg_.value("bos_token_id", 2)));
    ids.push_back(static_cast<int32_t>(cfg_.value("eos_token_id", 1)));
    return ids;
}

std::vector<std::vector<float>> Engine::debug_layers(const std::string& text, task_type_t task) {
    track_.clear();
    if (!tok_) return track_;
    const char* pfx = kPrompt(task);
    std::vector<int32_t> ids = tok_->Encode(std::string(pfx) + text);
    ids.insert(ids.begin(), static_cast<int32_t>(cfg_.value("bos_token_id", 2)));
    ids.push_back(static_cast<int32_t>(cfg_.value("eos_token_id", 1)));
    track_layers_ = true;
    transformer(ids);
    track_layers_ = false;
    return track_;
}

std::map<std::string, std::vector<float>> Engine::debug_kp(const std::string& text, task_type_t task) {
    std::map<std::string, std::vector<float>> out;
    if (!tok_) return out;
    const char* pfx = kPrompt(task);
    std::vector<int32_t> ids = tok_->Encode(std::string(pfx) + text);
    ids.insert(ids.begin(), static_cast<int32_t>(cfg_.value("bos_token_id", 2)));
    ids.push_back(static_cast<int32_t>(cfg_.value("eos_token_id", 1)));

    const size_t T = ids.size();
    const size_t HD = hidden_, QD = n_heads_ * head_dim_, ND = n_kv_ * head_dim_;

    std::vector<float> h(T * HD);
    {
        const auto& em = weight("embed_tokens.weight");
        for (size_t t = 0; t < T; t++) {
            const float* e = em.data() + static_cast<size_t>(ids[t]) * HD;
            float* dst = h.data() + t * HD;
            for (size_t i = 0; i < HD; i++) dst[i] = e[i] * embed_scale_;
        }
    }
    out["embed"] = h;

    std::vector<float> cos_tbl[2], sin_tbl[2];
    {
        const char* names[2] = {"sliding_attention", "full_attention"};
        for (int idx = 0; idx < 2; idx++) {
            const std::string lt = names[idx];
            const bool present =
                std::find(layer_types_.begin(), layer_types_.end(), lt) != layer_types_.end();
            if (!present) {
                cos_tbl[idx].assign(max_pos_ * head_dim_, 1.0f);
                sin_tbl[idx].assign(max_pos_ * head_dim_, 0.0f);
                continue;
            }
            const float base = (lt == "sliding_attention")
                                   ? cfg_.value("rope_local_base_freq", 10000.0f)
                                   : cfg_.value("rope_theta", 10000.0f);
            std::vector<float> inv(head_dim_ / 2);
            for (size_t i = 0; i < head_dim_ / 2; i++)
                inv[i] = std::pow(base, -2.0 * i / head_dim_);
            cos_tbl[idx].assign(max_pos_ * head_dim_, 1.0f);
            sin_tbl[idx].assign(max_pos_ * head_dim_, 0.0f);
            for (size_t p = 0; p < max_pos_; p++) {
                for (size_t i = 0; i < head_dim_ / 2; i++) {
                    const float f = static_cast<float>(p) * inv[i];
                    const float c = std::cos(f), s = std::sin(f);
                    cos_tbl[idx][p * head_dim_ + i]                  = c;
                    cos_tbl[idx][p * head_dim_ + i + head_dim_ / 2]  = c;
                    sin_tbl[idx][p * head_dim_ + i]                  = s;
                    sin_tbl[idx][p * head_dim_ + i + head_dim_ / 2]  = s;
                }
            }
        }
    }
    auto table_idx = [](const std::string& lt) -> int { return lt == "sliding_attention" ? 0 : 1; };

    const std::string lt = layer_types_[0];
    const std::string wpfx = "layers.0.";
    const int ci = table_idx(lt);
    std::vector<float> x, o, buf;
    rmsnorm(h.data(), weight(wpfx + "input_layernorm.weight").data(), T, HD, eps_, x);
    out["x_rms"] = x;

    std::vector<float> q, kt, v;
    matmul_t(x, weight(wpfx + "self_attn.q_proj.weight"), T, HD, QD, q);
    matmul_t(x, weight(wpfx + "self_attn.k_proj.weight"), T, HD, ND, kt);
    matmul_t(x, weight(wpfx + "self_attn.v_proj.weight"), T, HD, ND, v);
    out["q_proj"] = q;
    out["k_proj"] = kt;

    std::vector<float> qn(T * QD), kn(T * ND);
    for (size_t t = 0; t < T; t++) {
        for (size_t h_id = 0; h_id < n_heads_; h_id++) {
            const float* src = q.data() + (t * n_heads_ + h_id) * head_dim_;
            rmsnorm(src, weight(wpfx + "self_attn.q_norm.weight").data(), 1, head_dim_, eps_, buf);
            std::memcpy(qn.data() + (t * n_heads_ + h_id) * head_dim_, buf.data(), head_dim_ * sizeof(float));
        }
        for (size_t kv_id = 0; kv_id < n_kv_; kv_id++) {
            const float* src = kt.data() + (t * n_kv_ + kv_id) * head_dim_;
            rmsnorm(src, weight(wpfx + "self_attn.k_norm.weight").data(), 1, head_dim_, eps_, buf);
            std::memcpy(kn.data() + (t * n_kv_ + kv_id) * head_dim_, buf.data(), head_dim_ * sizeof(float));
        }
    }
    out["q_norm"] = qn;
    out["k_norm"] = kn;

    {
        const float* cos_p = cos_tbl[ci].data();
        const float* sin_p = sin_tbl[ci].data();
        const size_t half = head_dim_ / 2;
        for (size_t t = 0; t < T; t++) {
            const float* c = cos_p + t * head_dim_;
            const float* s = sin_p + t * head_dim_;
            for (size_t h_id = 0; h_id < n_heads_; h_id++) {
                float* r = qn.data() + (t * n_heads_ + h_id) * head_dim_;
                for (size_t d = 0; d < half; d++) {
                    float v0 = r[d], v1 = r[d + half];
                    r[d]         = v0 * c[d] - v1 * s[d];
                    r[d + half]  = v1 * c[d] + v0 * s[d];
                }
            }
            for (size_t kv_id = 0; kv_id < n_kv_; kv_id++) {
                float* r = kn.data() + (t * n_kv_ + kv_id) * head_dim_;
                for (size_t d = 0; d < half; d++) {
                    float v0 = r[d], v1 = r[d + half];
                    r[d]        = v0 * c[d] - v1 * s[d];
                    r[d + half] = v1 * c[d] + v0 * s[d];
                }
            }
        }
    }
    out["q_rot"] = qn;
    out["k_rot"] = kn;

    std::vector<float> scores(n_heads_ * T * T, kNegInf);
    for (size_t h_id = 0; h_id < n_heads_; h_id++) {
        for (size_t t = 0; t < T; t++) {
            const float* qp = qn.data() + (t * n_heads_ + h_id) * head_dim_;
            float* srow = scores.data() + (h_id * T + t) * T;
            for (size_t kpos = 0; kpos < T; kpos++) {
                const bool allowed = lt == "full_attention" ||
                    (t >= kpos ? t - kpos : kpos - t) < sliding_window_;
                if (!allowed) continue;
                const float* kp = kn.data() + (kpos * n_kv_) * head_dim_;
                float acc = 0.0f;
                for (size_t d = 0; d < head_dim_; d++) acc += qp[d] * kp[d];
                srow[kpos] = acc * scaling_;
            }
        }
    }
    out["scores"] = scores;

    std::vector<float> att(n_heads_ * T * T);
    for (size_t r = 0; r < scores.size() / T; r++) {
        float* srow = scores.data() + r * T;
        float mx = srow[0];
        for (size_t kpos = 1; kpos < T; kpos++) mx = std::max(mx, srow[kpos]);
        float denom = 0.0f;
        for (size_t kpos = 0; kpos < T; kpos++) {
            if (std::isfinite(srow[kpos])) {
                srow[kpos] = std::exp(srow[kpos] - mx);
                denom += srow[kpos];
            } else {
                srow[kpos] = 0.0f;
            }
        }
        if (denom > 0.0f) {
            for (size_t kpos = 0; kpos < T; kpos++) srow[kpos] /= denom;
        }
        std::memcpy(att.data() + r * T, srow, T * sizeof(float));
    }
    out["att"] = att;

    std::vector<float> oatt(T * QD, 0.0f);
    const size_t num_query_groups = n_heads_ / n_kv_;
    for (size_t h_id = 0; h_id < n_heads_; h_id++) {
        const size_t kv_id = h_id / num_query_groups;
        for (size_t t = 0; t < T; t++) {
            const float* arow = att.data() + (h_id * T + t) * T;
            float* orow = oatt.data() + (t * n_heads_ + h_id) * head_dim_;
            for (size_t d = 0; d < head_dim_; d++) {
                float acc = 0.0f;
                for (size_t kpos = 0; kpos < T; kpos++)
                    acc += arow[kpos] * v[(kpos * n_kv_ + kv_id) * head_dim_ + d];
                orow[d] = acc;
            }
        }
    }
    out["oatt_v"] = oatt;

    matmul_t(oatt, weight(wpfx + "self_attn.o_proj.weight"), T, QD, HD, o);
    out["o_proj"] = o;
    rmsnorm(o.data(), weight(wpfx + "post_attention_layernorm.weight").data(), T, HD, eps_, x);
    out["postattn_norm"] = x;
    for (size_t i = 0; i < h.size(); i++) h[i] += x[i];
    out["attn_resid"] = h;

    std::vector<float> preff, gate, up;
    rmsnorm(h.data(), weight(wpfx + "pre_feedforward_layernorm.weight").data(), T, HD, eps_, preff);
    out["preff_norm"] = preff;
    {
        const std::vector<float>& gw = weight(wpfx + "mlp.gate_proj.weight");
        std::vector<float> sample(gw.begin(), gw.begin() + 32);
        out["gate_w_head"] = sample;
        std::vector<float> prf(preff.begin(), preff.begin() + 32);
        out["preff_head"] = prf;
    }
    matmul_t(preff, weight(wpfx + "mlp.gate_proj.weight"), T, HD, intermediate_, gate);
    matmul_t(preff, weight(wpfx + "mlp.up_proj.weight"), T, HD, intermediate_, up);
    out["gate_raw"] = gate;
    out["up_raw"] = up;
    gelu_tanh(gate);
    for (size_t i = 0; i < gate.size(); i++) gate[i] *= up[i];
    out["gate_up_prod"] = gate;
    matmul_t(gate, weight(wpfx + "mlp.down_proj.weight"), T, intermediate_, HD, o);
    out["mlp_down_out"] = o;
    rmsnorm(o.data(), weight(wpfx + "post_feedforward_layernorm.weight").data(), T, HD, eps_, x);
    for (size_t i = 0; i < h.size(); i++) h[i] += x[i];
    out["mlp_resid"] = h;
    return out;
}

std::vector<std::vector<float>> Engine::debug_stages(const std::string& text, task_type_t task) {
    std::vector<std::vector<float>> out;
    if (!tok_) return out;
    const char* pfx = kPrompt(task);
    std::vector<int32_t> ids = tok_->Encode(std::string(pfx) + text);
    ids.insert(ids.begin(), static_cast<int32_t>(cfg_.value("bos_token_id", 2)));
    ids.push_back(static_cast<int32_t>(cfg_.value("eos_token_id", 1)));
    const size_t T = ids.size(), D = hidden_;
    std::vector<float> hid = transformer(ids);
    std::vector<float> pooled(D, 0.0f);
    for (size_t t = 0; t < T; t++) for (size_t i = 0; i < D; i++) pooled[i] += hid[t * D + i];
    for (size_t i = 0; i < D; i++) pooled[i] /= static_cast<float>(T);
    out.push_back(pooled);
    std::vector<float> d1, d2;
    matmul_t(pooled, weight("2_Dense.linear.weight"), 1, D, head_mid_, d1);
    out.push_back(d1);
    matmul_t(d1, weight("3_Dense.linear.weight"), 1, head_mid_, D, d2);
    out.push_back(d2);
    float norm = 0.0f;
    for (size_t i = 0; i < D; i++) norm += d2[i] * d2[i];
    norm = std::sqrt(norm);
    for (size_t i = 0; i < D; i++) d2[i] /= (norm > 0.0f ? norm : 1.0f);
    out.push_back(d2);
    return out;
}

std::vector<float> Engine::embed(const std::string& text, task_type_t task, std::string* task_prefix_out) {
    const char* pfx = kPrompt(task);
    if (task_prefix_out) *task_prefix_out = pfx;
    return embed_with_prefix(text, pfx);
}

std::vector<float> Engine::embed_with_prefix(const std::string& text, const std::string& task_prefix) {
    if (!tok_) return {};
    std::vector<int32_t> ids = tok_->Encode(task_prefix + text);
    ids.insert(ids.begin(), static_cast<int32_t>(cfg_.value("bos_token_id", 2)));
    ids.push_back(static_cast<int32_t>(cfg_.value("eos_token_id", 1)));
    const int32_t eos_id = static_cast<int32_t>(cfg_.value("eos_token_id", 1));
    if (ids.size() > max_pos_) {  // rope tables are sized to max_pos_
        ids.resize(max_pos_);
        ids[max_pos_ - 1] = eos_id;
    }
    std::vector<float> hid = transformer(ids);

    const size_t T = ids.size(), D = hidden_;
    std::vector<float> pooled(D, 0.0f);
    for (size_t t = 0; t < T; t++)
        for (size_t i = 0; i < D; i++) pooled[i] += hid[t * D + i];
    for (size_t i = 0; i < D; i++) pooled[i] /= static_cast<float>(T);

    std::vector<float> d1, d2;
    matmul_t_npu("2_Dense.linear.weight", pooled, 1, D, head_mid_, d1);
    matmul_t_npu("3_Dense.linear.weight", d1, 1, head_mid_, D, d2);
    float norm = 0.0f;
    for (size_t i = 0; i < D; i++) norm += d2[i] * d2[i];
    norm = std::sqrt(norm);
    std::vector<float> out(D);
    for (size_t i = 0; i < D; i++) out[i] = d2[i] / (norm > 0.0f ? norm : 1.0f);
    return out;
}

}  // namespace open_embedding