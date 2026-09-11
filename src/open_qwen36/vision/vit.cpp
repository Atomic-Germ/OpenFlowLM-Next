#include "open_qwen36/vision/vit.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "open_qwen36/q4nx_file.hpp"

namespace open_qwen36::vision {
namespace {

constexpr int kTileN = 64, kTileK = 256;

inline float bf16f(uint16_t u) {
    uint32_t v = static_cast<uint32_t>(u) << 16;
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}

std::vector<float> f32_of(const Q4nxFile& f, const std::string& name) { return f.bf16(name); }

/// [nt, kt, 64 * 256] bf16 (zero-padded tiles) -> bf16 [out, in], the padding dropped.
Linear untile(const Q4nxFile& f, const std::string& wname, const std::string& bname, int out, int in) {
    const TensorMeta& m = f.meta(wname);
    if (m.dtype != "BF16" || m.shape.size() != 3 || m.shape[2] != static_cast<size_t>(kTileN * kTileK))
        throw std::runtime_error("vit: " + wname + " is not a tiled bf16 [nt, kt, 16384] tensor");
    const size_t nt = m.shape[0], kt = m.shape[1];
    if (nt * kTileN < static_cast<size_t>(out) || kt * kTileK < static_cast<size_t>(in))
        throw std::runtime_error("vit: " + wname + " tiles do not cover [" + std::to_string(out) + ", " + std::to_string(in) + "]");
    size_t nbytes = 0;
    const uint16_t* t = reinterpret_cast<const uint16_t*>(f.raw(wname, &nbytes));
    Linear L;
    L.out = out;
    L.in = in;
    L.w.resize(static_cast<size_t>(out) * in);
    for (int o = 0; o < out; ++o) {
        const size_t tn = o / kTileN, rn = o % kTileN;
        uint16_t* dst = L.w.data() + static_cast<size_t>(o) * in;
        for (int k = 0; k < in; k += kTileK) {
            const size_t tk = k / kTileK;
            const int len = std::min(kTileK, in - k);
            std::memcpy(dst + k, t + ((tn * kt + tk) * kTileN + rn) * kTileK, static_cast<size_t>(len) * 2);
        }
    }
    L.b = f32_of(f, bname);
    if (L.b.size() != static_cast<size_t>(out)) throw std::runtime_error("vit: " + bname + " has the wrong length");
    return L;
}

/// y[n, out] = x[n, in] . W^T + b. Eight output columns at a time: their W rows are
/// widened to f32 once, then every x row streams past them. Parallel over column blocks.
void linear(const float* x, int n, const Linear& L, float* y) {
    const int in = L.in, out = L.out;
    const int nb = (out + 7) / 8;
#pragma omp parallel
    {
        std::vector<float> wf(static_cast<size_t>(8) * in);
#pragma omp for schedule(dynamic, 4)
        for (int blk = 0; blk < nb; ++blk) {
            const int o0 = blk * 8, oc = std::min(8, out - o0);
            for (int j = 0; j < oc; ++j) {
                const uint16_t* wr = L.w.data() + static_cast<size_t>(o0 + j) * in;
                float* wd = wf.data() + static_cast<size_t>(j) * in;
                for (int k = 0; k < in; ++k) wd[k] = bf16f(wr[k]);
            }
            for (int i = 0; i < n; ++i) {
                const float* xr = x + static_cast<size_t>(i) * in;
                float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                if (oc == 8) {
                    const float *w0 = wf.data(), *w1 = w0 + in, *w2 = w1 + in, *w3 = w2 + in;
                    const float *w4 = w3 + in, *w5 = w4 + in, *w6 = w5 + in, *w7 = w6 + in;
                    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
                    for (int k = 0; k < in; ++k) {
                        const float xv = xr[k];
                        a0 += xv * w0[k]; a1 += xv * w1[k]; a2 += xv * w2[k]; a3 += xv * w3[k];
                        a4 += xv * w4[k]; a5 += xv * w5[k]; a6 += xv * w6[k]; a7 += xv * w7[k];
                    }
                    acc[0] = a0; acc[1] = a1; acc[2] = a2; acc[3] = a3;
                    acc[4] = a4; acc[5] = a5; acc[6] = a6; acc[7] = a7;
                } else {
                    for (int j = 0; j < oc; ++j) {
                        const float* wd = wf.data() + static_cast<size_t>(j) * in;
                        float a = 0;
                        for (int k = 0; k < in; ++k) a += xr[k] * wd[k];
                        acc[j] = a;
                    }
                }
                float* yr = y + static_cast<size_t>(i) * out + o0;
                for (int j = 0; j < oc; ++j) yr[j] = acc[j] + L.b[o0 + j];
            }
        }
    }
}

void layer_norm(const float* x, int n, int d, const float* w, const float* b, float eps, float* y) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const float* xr = x + static_cast<size_t>(i) * d;
        float* yr = y + static_cast<size_t>(i) * d;
        double mu = 0;
        for (int k = 0; k < d; ++k) mu += xr[k];
        mu /= d;
        double var = 0;
        for (int k = 0; k < d; ++k) { const double t = xr[k] - mu; var += t * t; }
        var /= d;
        const float inv = static_cast<float>(1.0 / std::sqrt(var + eps));
        for (int k = 0; k < d; ++k) yr[k] = (xr[k] - static_cast<float>(mu)) * inv * w[k] + b[k];
    }
}

inline float gelu_tanh(float x) {
    const float c = 0.7978845608028654f;  // sqrt(2/pi)
    return 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * x * x * x)));
}
inline float gelu_erf(float x) { return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865476f)); }

/// (h, w) per patch in merge-block-major order (transformers' get_vision_position_ids).
void position_ids(int gh, int gw, int merge, std::vector<int>& ph, std::vector<int>& pw) {
    ph.clear();
    pw.clear();
    for (int bh = 0; bh < gh / merge; ++bh)
        for (int bw = 0; bw < gw / merge; ++bw)
            for (int ih = 0; ih < merge; ++ih)
                for (int iw = 0; iw < merge; ++iw) {
                    ph.push_back(bh * merge + ih);
                    pw.push_back(bw * merge + iw);
                }
}

/// The 48x48 learned table bilinearly interpolated to the grid, rows in the same order.
void add_pos_embed(const VitConfig& cfg, const VitWeights& w, int gh, int gw, const std::vector<int>& ph,
                   const std::vector<int>& pw, float* x) {
    const int side = static_cast<int>(std::lround(std::sqrt(static_cast<double>(cfg.npos)))), H = cfg.hidden;
    auto grid = [&](int n, int i) { return n == 1 ? 0.0f : static_cast<float>(i) * (side - 1) / static_cast<float>(n - 1); };
    const size_t n = ph.size();
#pragma omp parallel for schedule(static)
    for (int t = 0; t < static_cast<int>(n); ++t) {
        const float hg = grid(gh, ph[t]), wg = grid(gw, pw[t]);
        const int hf = static_cast<int>(hg), wf = static_cast<int>(wg);
        const int hc = std::min(hf + 1, side - 1), wc = std::min(wf + 1, side - 1);
        const float hr = hg - hf, wr = wg - wf;
        const float wt[4] = {(1 - hr) * (1 - wr), (1 - hr) * wr, hr * (1 - wr), hr * wr};
        const int idx[4] = {hf * side + wf, hf * side + wc, hc * side + wf, hc * side + wc};
        float* xr = x + static_cast<size_t>(t) * H;
        for (int c = 0; c < 4; ++c) {
            const float* pr = w.pos.data() + static_cast<size_t>(idx[c]) * H;
            for (int k = 0; k < H; ++k) xr[k] += wt[c] * pr[k];
        }
    }
}

/// cos / sin [n, head_dim]: (h, w) x 18 frequencies (theta 1e4 over dim head_dim/2), duplicated.
void rope_tables(const VitConfig& cfg, const std::vector<int>& ph, const std::vector<int>& pw,
                 std::vector<float>& cs, std::vector<float>& sn) {
    const int hd = cfg.head_dim, dim = hd / 2, nf = dim / 2;   // 72, 36, 18
    std::vector<float> inv(nf);
    for (int i = 0; i < nf; ++i) inv[i] = 1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / dim);
    const size_t n = ph.size();
    cs.assign(n * hd, 0.f);
    sn.assign(n * hd, 0.f);
    for (size_t t = 0; t < n; ++t) {
        float* c = cs.data() + t * hd;
        float* s = sn.data() + t * hd;
        for (int i = 0; i < nf; ++i) {
            const float ah = ph[t] * inv[i], aw = pw[t] * inv[i];
            c[i] = std::cos(ah); s[i] = std::sin(ah);                 // [h freqs | w freqs] ...
            c[nf + i] = std::cos(aw); s[nf + i] = std::sin(aw);
        }
        for (int i = 0; i < dim; ++i) { c[dim + i] = c[i]; s[dim + i] = s[i]; }   // ... duplicated
    }
}

/// Bidirectional attention over the whole image, one head per task, 32 query rows at a time.
void attention(const VitConfig& cfg, const float* qkv, int n, const std::vector<float>& cs, const std::vector<float>& sn,
               float* o) {
    const int NH = cfg.heads, HD = cfg.head_dim, H = cfg.hidden, half = HD / 2;
    const size_t row = static_cast<size_t>(3) * H;   // one token's [q | k | v]
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    // q and k with RoPE, per head contiguous: [NH][n][HD]
    std::vector<float> q(static_cast<size_t>(NH) * n * HD), k(q.size());
#pragma omp parallel for schedule(static)
    for (int t = 0; t < n; ++t) {
        const float* c = cs.data() + static_cast<size_t>(t) * HD;
        const float* s = sn.data() + static_cast<size_t>(t) * HD;
        for (int h = 0; h < NH; ++h) {
            const float* qs = qkv + t * row + h * HD;
            const float* ks = qkv + t * row + H + h * HD;
            float* qd = q.data() + (static_cast<size_t>(h) * n + t) * HD;
            float* kd = k.data() + (static_cast<size_t>(h) * n + t) * HD;
            for (int i = 0; i < HD; ++i) {
                const float rq = i < half ? -qs[i + half] : qs[i - half];
                const float rk = i < half ? -ks[i + half] : ks[i - half];
                qd[i] = qs[i] * c[i] + rq * s[i];
                kd[i] = ks[i] * c[i] + rk * s[i];
            }
        }
    }
    const int QB = 32, nqb = (n + QB - 1) / QB;
#pragma omp parallel
    {
        std::vector<float> sc(static_cast<size_t>(QB) * n);
#pragma omp for schedule(dynamic, 1) collapse(2)
        for (int h = 0; h < NH; ++h)
            for (int qb = 0; qb < nqb; ++qb) {
                const int q0 = qb * QB, qc = std::min(QB, n - q0);
                const float* kh = k.data() + static_cast<size_t>(h) * n * HD;
                for (int i = 0; i < qc; ++i) {
                    const float* qr = q.data() + (static_cast<size_t>(h) * n + q0 + i) * HD;
                    float* sr = sc.data() + static_cast<size_t>(i) * n;
                    float mx = -1e30f;
                    for (int t = 0; t < n; ++t) {
                        const float* kr = kh + static_cast<size_t>(t) * HD;
                        float a = 0;
                        for (int d = 0; d < HD; ++d) a += qr[d] * kr[d];
                        a *= scale;
                        sr[t] = a;
                        mx = std::max(mx, a);
                    }
                    float sum = 0;
                    for (int t = 0; t < n; ++t) { sr[t] = std::exp(sr[t] - mx); sum += sr[t]; }
                    const float inv = 1.0f / sum;
                    float* orow = o + static_cast<size_t>(q0 + i) * H + h * HD;
                    float acc[128] = {};
                    for (int t = 0; t < n; ++t) {
                        const float p = sr[t] * inv;
                        const float* vr = qkv + t * row + 2 * H + h * HD;
                        for (int d = 0; d < HD; ++d) acc[d] += p * vr[d];
                    }
                    for (int d = 0; d < HD; ++d) orow[d] = acc[d];
                }
            }
    }
}

}  // namespace

VitConfig VitConfig::from_model_dir(const std::string& model_dir) {
    std::ifstream f(model_dir + "/config.json");
    if (!f) throw std::runtime_error("vit: cannot open " + model_dir + "/config.json");
    nlohmann::json j;
    f >> j;
    const auto& v = j.at("vision_config");
    // FLM prefixes the keys per family: QWEN3_6_MOE_* on the 35B, QWEN3_5_* on Qwen3.5 (same tower)
    const char* prefix = v.contains("QWEN3_6_MOE_VISION_NUM_LAYERS") ? "QWEN3_6_MOE_"
                         : v.contains("QWEN3_5_VISION_NUM_LAYERS") ? "QWEN3_5_" : nullptr;
    if (!prefix) throw std::runtime_error("vit: config.json vision_config has neither QWEN3_6_MOE_* nor QWEN3_5_* keys");
    auto g = [&](const char* k) { return v.at(std::string(prefix) + k); };
    VitConfig c;
    c.depth = g("VISION_NUM_LAYERS");
    c.hidden = g("VISION_EMBED_DIM");
    c.heads = g("VISION_NUM_HEADS");
    c.head_dim = g("VISION_HEAD_DIM");
    c.inter = g("VISION_MLP_INTERMEDIATE_SIZE");
    c.out = g("VISION_OUT_HIDDEN_SIZE");
    c.patch = g("PATCH_SIZE");
    c.temporal = g("TEMPORAL_PATCH_SIZE");
    c.merge = g("SPATIAL_MERGE_SIZE");
    c.npos = g("VISION_NUM_POSITION_EMBEDDINGS");
    c.eps = g("VISION_LAYER_NORM_EPSILON");
    if (c.hidden != c.heads * c.head_dim) throw std::runtime_error("vit: heads x head_dim != hidden");
    return c;
}

VitWeights load_vit(const std::string& path, const VitConfig& cfg) {
    Q4nxFile f(path);
    const std::string p = "model.visual.";
    const int H = cfg.hidden, M = cfg.merge * cfg.merge;
    VitWeights w;
    {
        // patch_embed.proj is a plain [hidden, C, T, P, P] conv weight = a [hidden, C*T*P*P] linear
        const TensorMeta& m = f.meta(p + "patch_embed.proj.weight");
        size_t nb = 0;
        const uint16_t* t = reinterpret_cast<const uint16_t*>(f.raw(p + "patch_embed.proj.weight", &nb));
        w.patch.out = H;
        w.patch.in = cfg.patch_dim();
        if (nb != static_cast<size_t>(H) * w.patch.in * 2 || m.dtype != "BF16")
            throw std::runtime_error("vit: patch_embed.proj.weight is not bf16 [hidden, C*T*P*P]");
        w.patch.w.assign(t, t + static_cast<size_t>(H) * w.patch.in);
        w.patch.b = f32_of(f, p + "patch_embed.proj.bias");
    }
    w.pos = f32_of(f, p + "pos_embed.weight");
    if (w.pos.size() != static_cast<size_t>(cfg.npos) * H) throw std::runtime_error("vit: pos_embed.weight has the wrong shape");
    w.blocks.resize(cfg.depth);
    for (int i = 0; i < cfg.depth; ++i) {
        const std::string b = p + "blocks." + std::to_string(i) + ".";
        VitBlock& B = w.blocks[i];
        B.ln1_w = f32_of(f, b + "norm1.weight");
        B.ln1_b = f32_of(f, b + "norm1.bias");
        B.ln2_w = f32_of(f, b + "norm2.weight");
        B.ln2_b = f32_of(f, b + "norm2.bias");
        B.qkv = untile(f, b + "attn.qkv.weight", b + "attn.qkv.bias", 3 * H, H);
        B.proj = untile(f, b + "attn.proj.weight", b + "attn.proj.bias", H, H);
        B.fc1 = untile(f, b + "mlp.linear_fc1.weight", b + "mlp.linear_fc1.bias", cfg.inter, H);
        B.fc2 = untile(f, b + "mlp.linear_fc2.weight", b + "mlp.linear_fc2.bias", H, cfg.inter);
    }
    w.merger_ln_w = f32_of(f, p + "merger.norm.weight");
    w.merger_ln_b = f32_of(f, p + "merger.norm.bias");
    w.merger_fc1 = untile(f, p + "merger.linear_fc1.weight", p + "merger.linear_fc1.bias", H * M, H * M);
    w.merger_fc2 = untile(f, p + "merger.linear_fc2.weight", p + "merger.linear_fc2.bias", cfg.out, H * M);
    return w;
}

std::vector<float> vit_forward(const VitConfig& cfg, const VitWeights& w, const float* pixels, int gh, int gw) {
    const int n = gh * gw, H = cfg.hidden, M = cfg.merge * cfg.merge;
    if (gh % cfg.merge || gw % cfg.merge) throw std::runtime_error("vit: the grid is not a multiple of the merge size");
    std::vector<int> ph, pw;
    position_ids(gh, gw, cfg.merge, ph, pw);
    std::vector<float> x(static_cast<size_t>(n) * H);
    linear(pixels, n, w.patch, x.data());
    add_pos_embed(cfg, w, gh, gw, ph, pw, x.data());
    std::vector<float> cs, sn;
    rope_tables(cfg, ph, pw, cs, sn);
    std::vector<float> hn(x.size()), qkv(static_cast<size_t>(n) * 3 * H), att(x.size()), tmp(x.size());
    std::vector<float> ff(static_cast<size_t>(n) * cfg.inter);
    for (const VitBlock& B : w.blocks) {
        layer_norm(x.data(), n, H, B.ln1_w.data(), B.ln1_b.data(), cfg.eps, hn.data());
        linear(hn.data(), n, B.qkv, qkv.data());
        attention(cfg, qkv.data(), n, cs, sn, att.data());
        linear(att.data(), n, B.proj, tmp.data());
        for (size_t i = 0; i < x.size(); ++i) x[i] += tmp[i];
        layer_norm(x.data(), n, H, B.ln2_w.data(), B.ln2_b.data(), cfg.eps, hn.data());
        linear(hn.data(), n, B.fc1, ff.data());
#pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(ff.size()); ++i) ff[i] = gelu_tanh(ff[i]);
        linear(ff.data(), n, B.fc2, tmp.data());
        for (size_t i = 0; i < x.size(); ++i) x[i] += tmp[i];
    }
    // merger: LayerNorm per patch, 2x2 groups concatenated, fc1 -> exact GELU -> fc2
    layer_norm(x.data(), n, H, w.merger_ln_w.data(), w.merger_ln_b.data(), cfg.eps, hn.data());
    const int nm = n / M;
    std::vector<float> mid(static_cast<size_t>(nm) * H * M), y(static_cast<size_t>(nm) * cfg.out);
    linear(hn.data(), nm, w.merger_fc1, mid.data());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(mid.size()); ++i) mid[i] = gelu_erf(mid[i]);
    linear(mid.data(), nm, w.merger_fc2, y.data());
    return y;
}

}  // namespace open_qwen36::vision
