/// \file block_host.cpp
/// \brief The block prefill's host stages (see block_host.hpp).
///
/// Written for the compiler's vectoriser: float, contiguous inner loops over a
/// head's dims, the per-head work of a block spread over OpenMP threads
/// (heads are independent across every token of the block, so each thread
/// owns its heads and walks the tokens). Reductions that decide a norm or a
/// softmax accumulate in double.
#include "open_qwen36/block_host.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "open_qwen36/q4nx_file.hpp"   // bf16_to_f32 / f32_to_bf16

namespace open_qwen36 {
namespace host {

namespace {

float bf16r(float x) { return bf16_to_f32(f32_to_bf16(x)); }
float silu(float x) { return x / (1.0f + std::exp(-x)); }
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float softplus(float x) { return x > 0 ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x)); }

// x[d] / sqrt(mean(x^2) + eps) * w[d]
void rms_vec(const float* x, size_t d, const float* w, double eps, float* out) {
    double ss = 0;
    for (size_t j = 0; j < d; ++j) ss += static_cast<double>(x[j]) * x[j];
    const float r = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(d) + eps));
    for (size_t j = 0; j < d; ++j) out[j] = x[j] * r * w[j];
}

// the partial rotation over the first 2 * half dims, half-split (Qwen's layout)
void rope(float* x, size_t half, const double* inv_freq, double pos) {
    for (size_t i = 0; i < half; ++i) {
        const double a = pos * inv_freq[i];
        const float c = static_cast<float>(std::cos(a)), s = static_cast<float>(std::sin(a));
        const float x1 = x[i], x2 = x[half + i];
        x[i] = x1 * c - x2 * s;
        x[half + i] = x2 * c + x1 * s;
    }
}

}  // namespace

void rmsnorm_rows(const float* x, size_t T, size_t d, const float* w, double eps, float* out) {
#pragma omp parallel for
    for (long long t = 0; t < static_cast<long long>(T); ++t) rms_vec(x + t * d, d, w, eps, out + t * d);
}

void transpose(const float* y, size_t N, size_t T, float* out) {
    constexpr size_t B = 32;
#pragma omp parallel for
    for (long long nb = 0; nb < static_cast<long long>(N); nb += B)
        for (size_t tb = 0; tb < T; tb += B)
            for (size_t n = nb; n < std::min(static_cast<size_t>(nb) + B, N); ++n)
                for (size_t t = tb; t < std::min(tb + B, T); ++t) out[t * N + n] = y[n * T + t];
}

void tile_x(const float* x, size_t T, size_t K, uint16_t* out) {
    // [T,K] fp32 -> bf16, pre-tiled [K,T] in "k,n" order: K_TILE 64 x tile_n 32 tiles, each
    // tile in (8 x 8) MAC sub-tiles -- the layout gemm_q4_prefill.py streams its activation in
    constexpr size_t TK = 64, MAC = 8, TN = 32;
    if (K % TK || T % TN) throw std::runtime_error("open_qwen36: tile_x: K or T does not tile by (64, 32)");
    const size_t NB = T / TN;
#pragma omp parallel for
    for (long long kb = 0; kb < static_cast<long long>(K / TK); ++kb)
        for (size_t nb = 0; nb < NB; ++nb) {
            uint16_t* w = out + (kb * NB + nb) * TK * TN;
            for (size_t si = 0; si < TK / MAC; ++si)
                for (size_t ti = 0; ti < TN / MAC; ++ti)
                    for (size_t s = 0; s < MAC; ++s)
                        for (size_t t = 0; t < MAC; ++t)
                            *w++ = f32_to_bf16(x[(nb * TN + ti * MAC + t) * K + kb * TK + si * MAC + s]);
        }
}

void deltanet_block(const DeltaGeom& g, const float* qkv, const float* z, const float* xn, const float* convw,
                    const float* Wa, const float* Wb, const float* A, const float* dtb, const float* nw,
                    uint16_t* conv_state, float* S, float* og, double* phase_ms) {
    const auto tp0 = std::chrono::steady_clock::now();
    const size_t dim = g.head_dim, key_w = g.key_heads * dim, vw = g.value_heads * dim, nch = 2 * key_w + vw;
    if (g.value_heads % g.key_heads || g.t_real > g.T || g.s_rows < dim || g.lanes < g.value_heads)
        throw std::runtime_error("open_qwen36: deltanet_block: inconsistent geometry");
    const size_t grp = g.value_heads / g.key_heads, R = g.t_real;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(dim));
    std::fill(og, og + g.T * vw, 0.f);

    // ---- phase 1, per token: the conv (state rows carried), q / k normalised, alpha / beta
    // The conv is a fixed taps-wide window over the bf16-rounded rows, not a recurrence, so
    // every token's work is independent: row r of token t's window is qkv row t - pre + r,
    // and the carried state stands in where that runs before the block.
    const size_t pre = g.taps - 1;
    std::vector<float> carry(pre * nch);
    for (size_t r = 0; r < pre; ++r)
        for (size_t j = 0; j < nch; ++j) carry[r * nch + j] = bf16_to_f32(conv_state[r * nch + j]);
    std::vector<float> Q(R * key_w), Kk(R * key_w), V(R * vw), decay(R * g.value_heads), beta(R * g.value_heads);
#pragma omp parallel
    {
        std::vector<float> c(nch), al(g.lanes), be(g.lanes);
#pragma omp for
        for (long long tt = 0; tt < static_cast<long long>(R); ++tt) {
            const size_t t = static_cast<size_t>(tt);
            for (size_t j = 0; j < nch; ++j) {
                float acc = 0;
                for (size_t r = 0; r < g.taps; ++r) {
                    const long long s = tt - static_cast<long long>(pre) + static_cast<long long>(r);
                    acc += convw[r * nch + j] *
                           (s < 0 ? carry[(static_cast<size_t>(s) + pre) * nch + j] : bf16r(qkv[static_cast<size_t>(s) * nch + j]));
                }
                c[j] = silu(acc);
            }
            for (size_t hh = 0; hh < g.key_heads; ++hh)
                for (int which = 0; which < 2; ++which) {
                    const float* src = c.data() + which * key_w + hh * dim;
                    float* dst = (which ? Kk : Q).data() + t * key_w + hh * dim;
                    double ss = 0;
                    for (size_t j = 0; j < dim; ++j) ss += static_cast<double>(src[j]) * src[j];
                    const float r = static_cast<float>(1.0 / std::sqrt(ss + 1e-6));   // dn_glue's L2 norm
                    for (size_t j = 0; j < dim; ++j) dst[j] = src[j] * r;
                }
            std::copy(c.begin() + 2 * key_w, c.end(), V.begin() + t * vw);
            std::fill(al.begin(), al.end(), 0.f);
            std::fill(be.begin(), be.end(), 0.f);
            const float* x = xn + t * g.hid;
            for (size_t i = 0; i < g.hid; ++i) {
                const float xi = x[i];
                const float* wa = Wa + i * g.lanes;
                const float* wb = Wb + i * g.lanes;
                for (size_t h = 0; h < g.lanes; ++h) {
                    al[h] += xi * wa[h];
                    be[h] += xi * wb[h];
                }
            }
            for (size_t h = 0; h < g.value_heads; ++h) {
                decay[t * g.value_heads + h] = std::exp(A[h] * softplus(al[h] + dtb[h]));
                beta[t * g.value_heads + h] = sigmoid(be[h]);
            }
        }
    }
    // the window the next block starts from: the last `pre` rows, short blocks keeping what
    // the shift would have left in front of them
    for (size_t r = 0; r < pre; ++r) {
        const long long s = static_cast<long long>(R) - static_cast<long long>(pre) + static_cast<long long>(r);
        if (s < 0) {
            std::copy(conv_state + (R + r) * nch, conv_state + (R + r + 1) * nch, conv_state + r * nch);
        } else {
            const float* row = qkv + static_cast<size_t>(s) * nch;
            for (size_t j = 0; j < nch; ++j) conv_state[r * nch + j] = f32_to_bf16(row[j]);
        }
    }

    // ---- phase 2, per head over every token: the gated delta rule on S (in place), the gated norm
    const auto tp1 = std::chrono::steady_clock::now();
#pragma omp parallel for
    for (long long h = 0; h < static_cast<long long>(g.value_heads); ++h) {
        std::vector<float> tv(dim), delta(dim), o(dim), on(dim);
        float* Sh = S + h * g.s_rows * dim;
        for (size_t t = 0; t < R; ++t) {
            const float* kk = Kk.data() + t * key_w + (h / grp) * dim;
            const float* qq = Q.data() + t * key_w + (h / grp) * dim;
            const float* v = V.data() + t * vw + h * dim;
            const float dc = decay[t * g.value_heads + h], bt = beta[t * g.value_heads + h];
            std::fill(tv.begin(), tv.end(), 0.f);
            for (size_t i = 0; i < dim; ++i) {
                float* Si = Sh + i * dim;
                const float ki = kk[i];
                for (size_t j = 0; j < dim; ++j) {
                    Si[j] *= dc;
                    tv[j] += ki * Si[j];
                }
            }
            for (size_t j = 0; j < dim; ++j) delta[j] = bt * (v[j] - tv[j]);
            std::fill(o.begin(), o.end(), 0.f);
            for (size_t i = 0; i < dim; ++i) {
                float* Si = Sh + i * dim;
                const float ki = kk[i], qi = qq[i];
                for (size_t j = 0; j < dim; ++j) {
                    Si[j] += ki * delta[j];
                    o[j] += Si[j] * qi;
                }
            }
            for (size_t j = 0; j < dim; ++j) o[j] *= inv_sqrt;
            rms_vec(o.data(), dim, nw, g.eps, on.data());
            float* out = og + t * vw + h * dim;
            const float* zz = z + t * vw + h * dim;
            for (size_t j = 0; j < dim; ++j) out[j] = on[j] * silu(zz[j]);
        }
    }
    if (phase_ms) {
        using ms = std::chrono::duration<double, std::milli>;
        const auto tp2 = std::chrono::steady_clock::now();
        phase_ms[0] += ms(tp1 - tp0).count();
        phase_ms[1] += ms(tp2 - tp1).count();
    }
}

void attention_block(const AttnGeom& g, const float* q, const float* k, const float* v, const float* gate,
                     const float* qn, const float* kn, const double* inv_freq, uint16_t* kv, size_t kv_row_elems,
                     float* og) {
    const size_t qw = g.nh * g.hd, kvw = g.kvh * g.hd, half = g.rot / 2, rows = g.pos0 + g.t_real;
    if (g.nh % g.kvh || g.t_real > g.T || g.rot > g.hd || kv_row_elems < 2 * kvw)
        throw std::runtime_error("open_qwen36: attention_block: inconsistent geometry");
    const size_t grp = g.nh / g.kvh, R = g.t_real;
    const float scale = 1.0f / std::sqrt(static_cast<float>(g.hd));
    std::fill(og, og + g.T * qw, 0.f);

    // ---- phase 1: the cache window as floats (old rows from the cache, the block's rows
    // normed, roped, written to the cache in bf16), the block's queries normed and roped
    std::vector<float> K(rows * kvw), V(rows * kvw), Q(R * qw);
    for (size_t r = 0; r < g.pos0; ++r)
        for (size_t j = 0; j < kvw; ++j) {
            K[r * kvw + j] = bf16_to_f32(kv[r * kv_row_elems + j]);
            V[r * kvw + j] = bf16_to_f32(kv[r * kv_row_elems + kvw + j]);
        }
    for (size_t t = 0; t < R; ++t) {
        const size_t p = g.pos0 + t;
        for (size_t h = 0; h < g.nh; ++h) {
            float* dst = Q.data() + t * qw + h * g.hd;
            rms_vec(q + t * qw + h * g.hd, g.hd, qn, g.eps, dst);
            rope(dst, half, inv_freq, static_cast<double>(p));
        }
        std::vector<float> kh(kvw);
        for (size_t h = 0; h < g.kvh; ++h) {
            rms_vec(k + t * kvw + h * g.hd, g.hd, kn, g.eps, kh.data() + h * g.hd);
            rope(kh.data() + h * g.hd, half, inv_freq, static_cast<double>(p));
        }
        for (size_t j = 0; j < kvw; ++j) {
            const uint16_t kb = f32_to_bf16(kh[j]), vb = f32_to_bf16(v[t * kvw + j]);
            kv[p * kv_row_elems + j] = kb;
            kv[p * kv_row_elems + kvw + j] = vb;
            K[p * kvw + j] = bf16_to_f32(kb);
            V[p * kvw + j] = bf16_to_f32(vb);
        }
    }

    // ---- phase 2: every (head, token) pair attends over the causal window
    const long long pairs = static_cast<long long>(g.nh * R);
#pragma omp parallel for schedule(dynamic, 8)
    for (long long pr = 0; pr < pairs; ++pr) {
        const size_t h = static_cast<size_t>(pr) / R, t = static_cast<size_t>(pr) % R, p = g.pos0 + t;
        const float* qv = Q.data() + t * qw + h * g.hd;
        const size_t kh_off = (h / grp) * g.hd;
        std::vector<float> s(p + 1);
        float mx = -1e30f;
        for (size_t r = 0; r <= p; ++r) {
            const float* kr = K.data() + r * kvw + kh_off;
            float acc = 0;
            for (size_t j = 0; j < g.hd; ++j) acc += kr[j] * qv[j];
            s[r] = acc * scale;
            mx = std::max(mx, s[r]);
        }
        double denom = 0;
        for (size_t r = 0; r <= p; ++r) {
            s[r] = std::exp(s[r] - mx);
            denom += s[r];
        }
        const float inv = static_cast<float>(1.0 / denom);
        std::vector<float> o(g.hd, 0.f);
        for (size_t r = 0; r <= p; ++r) {
            const float* vr = V.data() + r * kvw + kh_off;
            const float a = s[r] * inv;
            for (size_t j = 0; j < g.hd; ++j) o[j] += a * vr[j];
        }
        float* out = og + t * qw + h * g.hd;
        const float* gt = gate + t * qw + h * g.hd;
        for (size_t j = 0; j < g.hd; ++j) out[j] = o[j] * sigmoid(gt[j]);
    }
}

void attention_prep(const AttnGeom& g, const float* q, const float* k, const float* v, const float* qn,
                    const float* kn, const double* inv_freq, uint16_t* kv, size_t kv_row_elems, float* Q) {
    const size_t qw = g.nh * g.hd, kvw = g.kvh * g.hd, half = g.rot / 2, R = g.t_real;
    if (g.nh % g.kvh || g.t_real > g.T || g.rot > g.hd || kv_row_elems < 2 * kvw)
        throw std::runtime_error("open_qwen36: attention_prep: inconsistent geometry");
    const float scale = 1.0f / std::sqrt(static_cast<float>(g.hd));
    std::fill(Q, Q + g.T * qw, 0.f);
#pragma omp parallel for
    for (long long tt = 0; tt < static_cast<long long>(R); ++tt) {
        const size_t t = static_cast<size_t>(tt), p = g.pos0 + t;
        for (size_t h = 0; h < g.nh; ++h) {
            float* dst = Q + t * qw + h * g.hd;
            rms_vec(q + t * qw + h * g.hd, g.hd, qn, g.eps, dst);
            rope(dst, half, inv_freq, static_cast<double>(p));
            for (size_t j = 0; j < g.hd; ++j) dst[j] *= scale;
        }
        std::vector<float> kh(kvw);
        for (size_t h = 0; h < g.kvh; ++h) {
            rms_vec(k + t * kvw + h * g.hd, g.hd, kn, g.eps, kh.data() + h * g.hd);
            rope(kh.data() + h * g.hd, half, inv_freq, static_cast<double>(p));
        }
        for (size_t j = 0; j < kvw; ++j) {
            kv[p * kv_row_elems + j] = f32_to_bf16(kh[j]);
            kv[p * kv_row_elems + kvw + j] = f32_to_bf16(v[t * kvw + j]);
        }
    }
}

void tile_rows_as_bt(const uint16_t* rows, size_t stride, size_t n_real, size_t n, size_t k, uint16_t* out) {
    constexpr size_t TK = 64, MAC = 8, TN = 32;
    if (k % TK || n % TN) throw std::runtime_error("open_qwen36: tile_rows_as_bt: k or n does not tile by (64, 32)");
    const size_t NB = n / TN;
#pragma omp parallel for
    for (long long kb = 0; kb < static_cast<long long>(k / TK); ++kb)
        for (size_t nb = 0; nb < NB; ++nb) {
            uint16_t* w = out + (kb * NB + nb) * TK * TN;
            for (size_t si = 0; si < TK / MAC; ++si)
                for (size_t ti = 0; ti < TN / MAC; ++ti)
                    for (size_t s = 0; s < MAC; ++s)
                        for (size_t t = 0; t < MAC; ++t) {
                            const size_t r = nb * TN + ti * MAC + t;
                            *w++ = r < n_real ? rows[r * stride + kb * TK + si * MAC + s] : 0;
                        }
        }
}

void tile_rows_as_b(const uint16_t* rows, size_t stride, size_t k_real, size_t k, size_t n, uint16_t* out) {
    constexpr size_t TK = 64, MAC = 8, TN = 32;
    if (k % TK || n % TN) throw std::runtime_error("open_qwen36: tile_rows_as_b: k or n does not tile by (64, 32)");
    const size_t NB = n / TN;
#pragma omp parallel for
    for (long long kb = 0; kb < static_cast<long long>(k / TK); ++kb)
        for (size_t nb = 0; nb < NB; ++nb) {
            uint16_t* w = out + (kb * NB + nb) * TK * TN;
            for (size_t si = 0; si < TK / MAC; ++si)
                for (size_t ti = 0; ti < TN / MAC; ++ti)
                    for (size_t s = 0; s < MAC; ++s) {
                        const size_t r = kb * TK + si * MAC + s;
                        for (size_t t = 0; t < MAC; ++t)
                            *w++ = r < k_real ? rows[r * stride + nb * TN + ti * MAC + t] : 0;
                    }
        }
}

void softmax_chunk(size_t M, size_t L, size_t hd, size_t c0, const float* s, const size_t* pos, float* m, float* l,
                   float* acc, uint16_t* p) {
#pragma omp parallel for
    for (long long rr = 0; rr < static_cast<long long>(M); ++rr) {
        const size_t r = static_cast<size_t>(rr);
        const float* sr = s + r * L;
        uint16_t* pr = p + r * L;
        const size_t valid = pos[r] >= c0 ? std::min(L, pos[r] - c0 + 1) : 0;
        if (valid == 0) {                       // the whole chunk is past this row's position
            std::fill(pr, pr + L, uint16_t{0});
            continue;
        }
        float mc = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < valid; ++j) mc = std::max(mc, sr[j]);
        const float m_new = std::max(m[r], mc);
        if (m[r] != m_new && l[r] != 0.f) {    // an earlier chunk's max is beaten: rescale what it accumulated
            const float a = std::exp(m[r] - m_new);
            l[r] *= a;
            float* ar = acc + r * hd;
            for (size_t j = 0; j < hd; ++j) ar[j] *= a;
        }
        double sum = 0;
        for (size_t j = 0; j < valid; ++j) {
            const uint16_t b = f32_to_bf16(std::exp(sr[j] - m_new));
            pr[j] = b;
            sum += bf16_to_f32(b);
        }
        std::fill(pr + valid, pr + L, uint16_t{0});
        l[r] += static_cast<float>(sum);
        m[r] = m_new;
    }
}

void router_block(size_t T, size_t hid, size_t E, size_t topk, const float* xm, const float* Wr, float* probs,
                  int32_t* idx, float* w) {
    if (topk > E) throw std::runtime_error("open_qwen36: router_block: topk past the expert count");
#pragma omp parallel for
    for (long long t = 0; t < static_cast<long long>(T); ++t) {
        std::vector<float> lg(E, 0.f);
        std::vector<char> taken(E, 0);
        const float* x = xm + t * hid;
        for (size_t i = 0; i < hid; ++i) {
            const float xi = x[i];
            const float* wr = Wr + i * E;
            for (size_t e = 0; e < E; ++e) lg[e] += xi * wr[e];
        }
        float mx = -1e30f;
        for (size_t e = 0; e < E; ++e) mx = std::max(mx, lg[e]);
        double denom = 0;
        for (size_t e = 0; e < E; ++e) {
            lg[e] = std::exp(lg[e] - mx);
            denom += lg[e];
        }
        const float inv = static_cast<float>(1.0 / denom);
        for (size_t e = 0; e < E; ++e) {
            lg[e] *= inv;
            probs[t * E + e] = lg[e];
        }
        double wsum = 0;
        for (size_t s = 0; s < topk; ++s) {
            size_t best = E;
            for (size_t e = 0; e < E; ++e)
                if (!taken[e] && (best == E || lg[e] > lg[best])) best = e;
            taken[best] = 1;
            idx[t * topk + s] = static_cast<int32_t>(best);
            w[t * topk + s] = lg[best];
            wsum += lg[best];
        }
        for (size_t s = 0; s < topk; ++s) w[t * topk + s] = static_cast<float>(w[t * topk + s] / wsum);
    }
}

}  // namespace host
}  // namespace open_qwen36
