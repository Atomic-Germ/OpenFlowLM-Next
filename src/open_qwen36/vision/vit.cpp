#include "open_qwen36/vision/vit.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

// The tower is host arithmetic, so the projections are worth vectorising. AVX2
// is chosen at RUNTIME, not by raising the binary's baseline: this is the only
// translation unit that uses it, and a machine without it runs the scalar loop
// it always ran. x86 only - nothing here is needed on ARM.
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define OFLM_VIT_AVX2 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

#include "gguf/gguf_file.hpp"
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

// Qwen3-VL-4B-Instruct-NPU2 uses a different tiling from the 35B: its linears are
// declared two-dimensional as [elements / 32768, 32768] and the order inside is tiles of
// 64 output rows by 512 input columns, row-major within a tile and row-major over the
// tiles, with NOTHING padded. Settled against Qwen/Qwen3-VL-4B-Instruct's own
// safetensors, all 315 tensors (see specs OPEN-VISION-VIT-FLAT).
constexpr int kFlatN = 64, kFlatK = 512, kFlatRow = kFlatN * kFlatK;

bool is_flat_tiled(const TensorMeta& m) {
    return m.shape.size() == 2 && m.shape[1] == static_cast<size_t>(kFlatRow);
}

/// The flat form -> bf16 [out, in]. Sizes are exact, so a mismatch is an error, not padding.
Linear untile_flat(const Q4nxFile& f, const std::string& wname, const std::string& bname, int out, int in) {
    const TensorMeta& m = f.meta(wname);
    const size_t want = static_cast<size_t>(out) * in;
    if (m.shape[0] * static_cast<size_t>(kFlatRow) != want)
        throw std::runtime_error("vit: " + wname + " holds " + std::to_string(m.shape[0] * kFlatRow) +
                                 " elements where [" + std::to_string(out) + ", " + std::to_string(in) +
                                 "] needs " + std::to_string(want));
    if (out % kFlatN || in % kFlatK)
        throw std::runtime_error("vit: " + wname + " [" + std::to_string(out) + ", " + std::to_string(in) +
                                 "] is not a whole number of " + std::to_string(kFlatN) + "x" +
                                 std::to_string(kFlatK) + " tiles");
    size_t nbytes = 0;
    const uint16_t* t = reinterpret_cast<const uint16_t*>(f.raw(wname, &nbytes));
    Linear L;
    L.out = out;
    L.in = in;
    L.w.resize(want);
    const size_t kt = static_cast<size_t>(in) / kFlatK;
    for (int o = 0; o < out; ++o) {
        const size_t tn = static_cast<size_t>(o) / kFlatN, rn = static_cast<size_t>(o) % kFlatN;
        uint16_t* dst = L.w.data() + static_cast<size_t>(o) * in;
        for (size_t tk = 0; tk < kt; ++tk)
            std::memcpy(dst + tk * kFlatK, t + ((tn * kt + tk) * kFlatN + rn) * kFlatK,
                        static_cast<size_t>(kFlatK) * 2);
    }
    if (bname.empty()) {
        L.b.assign(static_cast<size_t>(out), 0.f);
        return L;
    }
    L.b = f32_of(f, bname);
    if (L.b.size() != static_cast<size_t>(out)) throw std::runtime_error("vit: " + bname + " has the wrong length");
    return L;
}

/// [nt, kt, 64 * 256] bf16 (zero-padded tiles) -> bf16 [out, in], the padding dropped.
/// An empty bname gives a zero bias, so a bias-less linear needs no special case downstream.
Linear untile(const Q4nxFile& f, const std::string& wname, const std::string& bname, int out, int in) {
    const TensorMeta& m = f.meta(wname);
    if (m.dtype == "BF16" && is_flat_tiled(m)) return untile_flat(f, wname, bname, out, in);
    if (m.dtype == "BF16" && m.shape.size() == 2 && m.shape[0] == static_cast<size_t>(out) &&
        m.shape[1] == static_cast<size_t>(in)) {
        // Stored at its natural shape (Qwen3-VL keeps pos_embed that way).
        size_t nbytes = 0;
        const uint16_t* t = reinterpret_cast<const uint16_t*>(f.raw(wname, &nbytes));
        Linear L;
        L.out = out;
        L.in = in;
        L.w.assign(t, t + static_cast<size_t>(out) * in);
        if (bname.empty()) L.b.assign(static_cast<size_t>(out), 0.f);
        else {
            L.b = f32_of(f, bname);
            if (L.b.size() != static_cast<size_t>(out))
                throw std::runtime_error("vit: " + bname + " has the wrong length");
        }
        return L;
    }
    if (m.dtype != "BF16" || m.shape.size() != 3 || m.shape[2] != static_cast<size_t>(kTileN * kTileK))
        throw std::runtime_error("vit: " + wname + " is not a tiled bf16 [nt, kt, 16384] tensor");
    const size_t nt = m.shape[0], kt = m.shape[1];
    // exactly the padding the converter applies, not merely enough of it: a tensor with
    // more tiles than [out, in] needs is a tensor this loader has the wrong shape for
    const size_t pad = kTileK;   // both dims round up to max(kTileN, kTileK)
    if (nt != (out + pad - 1) / pad * pad / kTileN || kt != (in + pad - 1) / pad * pad / kTileK)
        throw std::runtime_error("vit: " + wname + " is [" + std::to_string(nt) + ", " + std::to_string(kt) +
                                 "] tiles where [" + std::to_string(out) + ", " + std::to_string(in) +
                                 "] padded to 256 needs [" + std::to_string((out + pad - 1) / pad * pad / kTileN) +
                                 ", " + std::to_string((in + pad - 1) / pad * pad / kTileK) + "]");
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
    if (bname.empty()) {
        L.b.assign(static_cast<size_t>(out), 0.f);
        return L;
    }
    L.b = f32_of(f, bname);
    if (L.b.size() != static_cast<size_t>(out)) throw std::runtime_error("vit: " + bname + " has the wrong length");
    return L;
}

/// Eight dot products of one x row against eight f32 weight rows, AVX2 + FMA.
/// Only the reduction order differs from the scalar loop below it, so the two
/// disagree by float rounding on the last bits and nothing else.
#if defined(OFLM_VIT_AVX2)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
void dot8_avx2(const float* xr, const float* wf, int in, float* acc) {
    __m256 a[8];
    for (int j = 0; j < 8; ++j) a[j] = _mm256_setzero_ps();
    int k = 0;
    for (; k + 8 <= in; k += 8) {
        const __m256 xv = _mm256_loadu_ps(xr + k);
        for (int j = 0; j < 8; ++j)
            a[j] = _mm256_fmadd_ps(xv, _mm256_loadu_ps(wf + static_cast<size_t>(j) * in + k), a[j]);
    }
    for (int j = 0; j < 8; ++j) {
        __m128 lo = _mm256_castps256_ps128(a[j]);
        lo = _mm_add_ps(lo, _mm256_extractf128_ps(a[j], 1));
        lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
        float s = _mm_cvtss_f32(lo);
        for (int t = k; t < in; ++t) s += xr[t] * wf[static_cast<size_t>(j) * in + t];
        acc[j] = s;
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
void widen_avx2(const uint16_t* wr, float* wd, int in) {
    int k = 0;
    for (; k + 8 <= in; k += 8) {
        const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(wr + k));
        _mm256_storeu_ps(wd + k, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16)));
    }
    for (; k < in; ++k) wd[k] = bf16f(wr[k]);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
float dot_avx2(const float* a, const float* b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    for (; i + 8 <= n; i += 8)
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    s0 = _mm256_add_ps(s0, s1);
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    float r = _mm_cvtss_f32(lo);
    for (; i < n; ++i) r += a[i] * b[i];
    return r;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
void axpy_avx2(float* acc, const float* x, float p, int n) {
    const __m256 pv = _mm256_set1_ps(p);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(acc + i, _mm256_fmadd_ps(pv, _mm256_loadu_ps(x + i), _mm256_loadu_ps(acc + i)));
    for (; i < n; ++i) acc[i] += p * x[i];
}

bool have_avx2() {
#if defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuidex(r, 7, 0);
    const bool avx2 = (r[1] & (1 << 5)) != 0;          // EBX bit 5
    __cpuid(r, 1);
    const bool fma = (r[2] & (1 << 12)) != 0;          // ECX bit 12
    const bool osxsave = (r[2] & (1 << 27)) != 0;
    if (!(avx2 && fma && osxsave)) return false;
    return (_xgetbv(0) & 0x6) == 0x6;                  // XMM and YMM state enabled
#else
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
}
#endif  // OFLM_VIT_AVX2

/// The two shapes attention() reduces over, vectorised where the CPU allows it.
/// The scalar forms use four accumulators rather than one: the old single chain
/// made every QK dot a serial add at three or four cycles a multiply.
inline float dot_f32(const float* a, const float* b, int n) {
#if defined(OFLM_VIT_AVX2)
    static const bool avx2 = have_avx2();
    if (avx2) return dot_avx2(a, b, n);
#endif
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    float r = (s0 + s1) + (s2 + s3);
    for (; i < n; ++i) r += a[i] * b[i];
    return r;
}

inline void axpy_f32(float* acc, const float* x, float p, int n) {
#if defined(OFLM_VIT_AVX2)
    static const bool avx2 = have_avx2();
    if (avx2) { axpy_avx2(acc, x, p, n); return; }
#endif
    for (int i = 0; i < n; ++i) acc[i] += p * x[i];
}

/// y[n, out] = x[n, in] . W^T + b. Eight output columns at a time: their W rows are
/// widened to f32 once, then a panel of x rows streams past them. Parallel over column
/// blocks. ldy is the output row stride, so q, k and v can be written into one [n, 3H]
/// buffer.
///
/// The panel is what keeps this off memory. Sweeping all n rows per column block reads
/// the whole activation matrix once per block - 160 times over for a 1280-wide
/// projection - and at a 40x56 grid that matrix is 11 MB, so it comes back from L3 every
/// time. A 128-row panel is under a megabyte and stays in L2 across the sweep.
void linear(const float* x, int n, const Linear& L, float* y, int ldy = 0) {
    const int in = L.in, out = L.out;
    if (ldy == 0) ldy = out;
    const int nb = (out + 7) / 8;
    constexpr int kPanel = 128;
#if defined(OFLM_VIT_AVX2)
    static const bool avx2 = have_avx2();
#endif
#pragma omp parallel
    {
        std::vector<float> wf(static_cast<size_t>(8) * in);
        for (int p = 0; p < n; p += kPanel) {
            const int pn = std::min(kPanel, n - p);
#pragma omp for schedule(dynamic, 4)
            for (int blk = 0; blk < nb; ++blk) {
                const int o0 = blk * 8, oc = std::min(8, out - o0);
                for (int j = 0; j < oc; ++j) {
                    const uint16_t* wr = L.w.data() + static_cast<size_t>(o0 + j) * in;
                    float* wd = wf.data() + static_cast<size_t>(j) * in;
#if defined(OFLM_VIT_AVX2)
                    if (avx2) { widen_avx2(wr, wd, in); continue; }
#endif
                    for (int k = 0; k < in; ++k) wd[k] = bf16f(wr[k]);
                }
                for (int i = p; i < p + pn; ++i) {
                    const float* xr = x + static_cast<size_t>(i) * in;
                    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#if defined(OFLM_VIT_AVX2)
                    if (avx2 && oc == 8) {
                        dot8_avx2(xr, wf.data(), in, acc);
                    } else
#endif
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
                    float* yr = y + static_cast<size_t>(i) * ldy + o0;
                    for (int j = 0; j < oc; ++j) yr[j] = acc[j] + L.b[o0 + j];
                }
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

/// Qwen2.5-VL's norm: no mean subtraction, no bias, eps fixed at 1e-6 in the module.
void rms_norm(const float* x, int n, int d, const float* w, float eps, float* y) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const float* xr = x + static_cast<size_t>(i) * d;
        float* yr = y + static_cast<size_t>(i) * d;
        double ms = 0;
        for (int k = 0; k < d; ++k) ms += static_cast<double>(xr[k]) * xr[k];
        const float inv = static_cast<float>(1.0 / std::sqrt(ms / d + eps));
        for (int k = 0; k < d; ++k) yr[k] = xr[k] * inv * w[k];
    }
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

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

/// Bidirectional attention inside each [cu[i], cu[i+1]) segment, 32 query rows at a time.
/// The whole-image tower passes {0, n}; the windowed one passes its window boundaries, so
/// the work list is (segment, query block) pairs rather than query blocks - a windowed
/// layer has many short segments and a full layer one long one, and collapsing over heads
/// x query blocks alone starves on the first.
void attention(const VitConfig& cfg, const float* qkv, int n, const std::vector<float>& cs, const std::vector<float>& sn,
               const std::vector<int>& cu, float* o) {
    const int NH = cfg.heads, HD = cfg.head_dim, H = cfg.hidden, half = HD / 2;
    const size_t row = static_cast<size_t>(3) * H;   // one token's [q | k | v]
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    // q, k (with RoPE) and v, per head contiguous: [NH][n][HD].
    //
    // v is re-laid for the same reason q and k are. Read in place it comes out of
    // the interleaved qkv buffer at a stride of 3 * hidden floats - 15 KB on
    // Qwen2.5-VL - and the AV loop walks the whole segment once per query, so a
    // full-attention block re-reads it n times with no prefetcher able to follow.
    std::vector<float> q(static_cast<size_t>(NH) * n * HD), k(q.size()), v(q.size());
#pragma omp parallel for schedule(static)
    for (int t = 0; t < n; ++t) {
        const float* c = cs.data() + static_cast<size_t>(t) * HD;
        const float* s = sn.data() + static_cast<size_t>(t) * HD;
        for (int h = 0; h < NH; ++h) {
            const float* qs = qkv + t * row + h * HD;
            const float* ks = qkv + t * row + H + h * HD;
            const float* vs = qkv + t * row + 2 * H + h * HD;
            float* qd = q.data() + (static_cast<size_t>(h) * n + t) * HD;
            float* kd = k.data() + (static_cast<size_t>(h) * n + t) * HD;
            float* vd = v.data() + (static_cast<size_t>(h) * n + t) * HD;
            for (int i = 0; i < HD; ++i) {
                const float rq = i < half ? -qs[i + half] : qs[i - half];
                const float rk = i < half ? -ks[i + half] : ks[i - half];
                qd[i] = qs[i] * c[i] + rq * s[i];
                kd[i] = ks[i] * c[i] + rk * s[i];
            }
            std::memcpy(vd, vs, static_cast<size_t>(HD) * sizeof(float));
        }
    }
    const int QB = 32;
    struct Work { int seg0, seg_len, q0, qc; };
    std::vector<Work> work;
    int longest = 0;
    for (size_t s = 0; s + 1 < cu.size(); ++s) {
        const int a = cu[s], len = cu[s + 1] - a;
        longest = std::max(longest, len);
        for (int q0 = a; q0 < a + len; q0 += QB) work.push_back({a, len, q0, std::min(QB, a + len - q0)});
    }
    const int nw = static_cast<int>(work.size());
#pragma omp parallel
    {
        std::vector<float> sc(static_cast<size_t>(QB) * longest);
#pragma omp for schedule(dynamic, 1) collapse(2)
        for (int h = 0; h < NH; ++h)
            for (int wi = 0; wi < nw; ++wi) {
                const Work& W = work[wi];
                const int m = W.seg_len;
                const float* kh = k.data() + (static_cast<size_t>(h) * n + W.seg0) * HD;
                const float* vh = v.data() + (static_cast<size_t>(h) * n + W.seg0) * HD;
                for (int i = 0; i < W.qc; ++i) {
                    const float* qr = q.data() + (static_cast<size_t>(h) * n + W.q0 + i) * HD;
                    float* sr = sc.data() + static_cast<size_t>(i) * m;
                    float mx = -1e30f;
                    for (int t = 0; t < m; ++t) {
                        const float* kr = kh + static_cast<size_t>(t) * HD;
                        float a = dot_f32(qr, kr, HD) * scale;
                        sr[t] = a;
                        mx = std::max(mx, a);
                    }
                    float sum = 0;
                    for (int t = 0; t < m; ++t) { sr[t] = std::exp(sr[t] - mx); sum += sr[t]; }
                    const float inv = 1.0f / sum;
                    float* orow = o + static_cast<size_t>(W.q0 + i) * H + h * HD;
                    float acc[128] = {};
                    for (int t = 0; t < m; ++t)
                        axpy_f32(acc, vh + static_cast<size_t>(t) * HD, sr[t] * inv, HD);
                    for (int d = 0; d < HD; ++d) orow[d] = acc[d];
                }
            }
    }
}

}  // namespace

namespace {

const char* const kHfKeys =
    "depth, hidden_size, num_heads, intermediate_size, out_hidden_size, patch_size, "
    "temporal_patch_size, spatial_merge_size, num_position_embeddings";

/// This is Qwen3-VL's full-attention tower without deepstack. A config describing anything
/// else is named, not approximated - dropping a part gives image embeddings that look
/// plausible and are wrong. Mirrors replica_vit.py's _refuse_towers_we_do_not_run.
void refuse_towers_we_do_not_run(const nlohmann::json& v) {
    const auto ds = v.find("deepstack_visual_indexes");
    if (ds != v.end() && ds->is_array() && !ds->empty())
        throw std::runtime_error("vit: vision_config deepstack_visual_indexes " + ds->dump() +
                                 " - this tower feeds those layers through extra mergers into the first "
                                 "decoder layers, which the host tower does not implement");
    if (v.value("window_size", 0) != 0 || v.contains("fullatt_block_indexes"))
        throw std::runtime_error("vit: vision_config window_size / fullatt_block_indexes - a windowed tower "
                                 "(Qwen2.5-VL) is a different design from the full-attention one implemented here");
    const std::string act = v.value("hidden_act", std::string("gelu_pytorch_tanh"));
    if (act != "gelu_pytorch_tanh")
        throw std::runtime_error("vit: vision_config hidden_act '" + act + "' - the tower's MLP is GELU-tanh");
}

}  // namespace

VitConfig VitConfig::from_model_dir(const std::string& model_dir) {
    std::ifstream f(model_dir + "/config.json");
    if (!f) throw std::runtime_error("vit: cannot open " + model_dir + "/config.json");
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return from_config_text(text);
}

VitConfig VitConfig::from_config_text(const std::string& config_json) {
    const nlohmann::json j = nlohmann::json::parse(config_json);
    const auto it = j.find("vision_config");
    if (it == j.end() || !it->is_object() || it->empty())
        throw std::runtime_error(
            "vit: config.json has no vision_config, so this container does not say what its tower looks "
            "like - Qwen3-VL-4B-Instruct-NPU2 is like this, its closed engine hardcodes the numbers. "
            "Looked for QWEN3_6_MOE_VISION_NUM_LAYERS, QWEN3_5_VISION_NUM_LAYERS, and the plain keys: " +
            std::string(kHfKeys));
    const nlohmann::json& v = *it;
    refuse_towers_we_do_not_run(v);
    VitConfig c;
    // OFLM prefixes the keys per family: QWEN3_6_MOE_* on the 35B, QWEN3_5_* on Qwen3.5 (same tower).
    const char* prefix = v.contains("QWEN3_6_MOE_VISION_NUM_LAYERS") ? "QWEN3_6_MOE_"
                         : v.contains("QWEN3_5_VISION_NUM_LAYERS") ? "QWEN3_5_" : nullptr;
    if (prefix) {
        auto g = [&](const char* k) { return v.at(std::string(prefix) + k); };
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
        c.channels = 3;
    } else if (v.contains("depth")) {
        auto h = [&](const char* k) {
            if (!v.contains(k))
                throw std::runtime_error("vit: vision_config has no " + std::string(k) +
                                         " in the transformers-shaped block");
            return v.at(k);
        };
        c.depth = h("depth");
        c.hidden = h("hidden_size");
        c.heads = h("num_heads");
        if (c.heads <= 0 || c.hidden % c.heads)
            throw std::runtime_error("vit: vision_config hidden_size " + std::to_string(c.hidden) +
                                     " is not a multiple of num_heads " + std::to_string(c.heads));
        // transformers has no head_dim or epsilon for this tower: hidden/heads, LayerNorm's default.
        c.head_dim = c.hidden / c.heads;
        c.inter = h("intermediate_size");
        c.out = h("out_hidden_size");
        c.patch = h("patch_size");
        c.temporal = h("temporal_patch_size");
        c.merge = h("spatial_merge_size");
        c.npos = h("num_position_embeddings");
        c.eps = v.value("layer_norm_eps", 1e-6f);
        c.channels = v.value("in_channels", v.value("in_chans", 3));
    } else {
        throw std::runtime_error("vit: vision_config carries none of the key sets this tower is read from: "
                                 "QWEN3_6_MOE_VISION_NUM_LAYERS, QWEN3_5_VISION_NUM_LAYERS, or " +
                                 std::string(kHfKeys));
    }
    if (c.hidden != c.heads * c.head_dim) throw std::runtime_error("vit: heads x head_dim != hidden");
    return c;
}

/// Qwen3-VL-4B-Instruct-NPU2 carries no vision_config at all, so the weight file is the
/// only description of its tower on disk. Everything but two numbers is determined by the
/// tensor shapes, and nothing in this container is padded, so an element count fixes a
/// width exactly once `hidden` is known -- and `hidden` is known, because patch_embed
/// keeps its natural [hidden, C, T, P, P] shape.
///
/// `heads` and `deepstack` are arguments because they are NOT in the file at any tiling:
/// qkv is [3 * hidden, hidden] for every head count, and the merger names say how many
/// extra mergers there are and never which blocks they hang off. They come from the
/// model's published config. Every other number is derived and cross-checked here, so a
/// wrong argument is refused rather than believed.
VitConfig VitConfig::qwen3vl_from_tensors(const std::string& vision_q4nx_path, int heads,
                                          const std::vector<int>& deepstack) {
    Q4nxFile f(vision_q4nx_path);
    const std::string p = "model.visual.";
    VitConfig c;
    c.family = VitFamily::Qwen3VL;

    const std::string pe = p + "patch_embed.proj.weight";
    if (!f.has(pe)) throw std::runtime_error("vit: " + vision_q4nx_path + " has no " + pe);
    const TensorMeta& m = f.meta(pe);
    if (m.dtype != "BF16" || m.shape.size() != 5)
        throw std::runtime_error("vit: " + pe + " is not bf16 [hidden, C, T, P, P]");
    c.hidden = static_cast<int>(m.shape[0]);
    c.channels = static_cast<int>(m.shape[1]);
    c.temporal = static_cast<int>(m.shape[2]);
    c.patch = static_cast<int>(m.shape[3]);
    if (c.channels != 3 || m.shape[3] != m.shape[4])
        throw std::runtime_error("vit: " + pe + " is not 3 channels of square patches");

    auto elems = [&](const std::string& name) -> size_t {
        if (!f.has(name)) throw std::runtime_error("vit: " + vision_q4nx_path + " has no " + name);
        const TensorMeta& t = f.meta(name);
        size_t n = 1;
        for (size_t d : t.shape) n *= d;
        return n;
    };

    c.depth = 0;
    while (f.has(p + "blocks." + std::to_string(c.depth) + ".attn.qkv.weight")) ++c.depth;
    if (!c.depth) throw std::runtime_error("vit: " + vision_q4nx_path + " has no blocks.N.* tensors");

    const size_t fc1 = elems(p + "blocks.0.mlp.linear_fc1.weight");
    if (fc1 % static_cast<size_t>(c.hidden))
        throw std::runtime_error("vit: mlp.linear_fc1 is not a multiple of hidden");
    c.inter = static_cast<int>(fc1 / c.hidden);

    const size_t sq = elems(p + "merger.linear_fc1.weight");
    size_t width = 0;
    while (width * width < sq) ++width;                       // the merger fc1 is square
    if (width * width != sq) throw std::runtime_error("vit: merger.linear_fc1 is not square");
    const size_t merge_sq = width / static_cast<size_t>(c.hidden);
    if (width % static_cast<size_t>(c.hidden) || merge_sq * c.hidden != width)
        throw std::runtime_error("vit: the merger width is not hidden times a square merge factor");
    c.merge = 0;
    while (static_cast<size_t>(c.merge) * c.merge < merge_sq) ++c.merge;
    if (static_cast<size_t>(c.merge) * c.merge != merge_sq)
        throw std::runtime_error("vit: the merge factor is not square");

    const size_t fc2 = elems(p + "merger.linear_fc2.weight");
    if (fc2 % width) throw std::runtime_error("vit: merger.linear_fc2 is not a multiple of the merged width");
    c.out = static_cast<int>(fc2 / width);
    c.npos = static_cast<int>(f.meta(p + "pos_embed.weight").shape[0]);

    int have = 0;
    while (f.has(p + "deepstack_merger_list." + std::to_string(have) + ".linear_fc1.weight")) ++have;
    if (have != static_cast<int>(deepstack.size()))
        throw std::runtime_error("vit: " + vision_q4nx_path + " holds " + std::to_string(have) +
                                 " deepstack mergers but " + std::to_string(deepstack.size()) +
                                 " indexes were given");
    for (int i : deepstack)
        if (i < 0 || i >= c.depth)
            throw std::runtime_error("vit: deepstack index " + std::to_string(i) + " is outside the tower's " +
                                     std::to_string(c.depth) + " blocks");
    c.deepstack = deepstack;

    if (heads <= 0 || c.hidden % heads)
        throw std::runtime_error("vit: head count " + std::to_string(heads) + " does not divide hidden " +
                                 std::to_string(c.hidden));
    c.heads = heads;
    c.head_dim = c.hidden / heads;
    c.eps = 1e-6f;
    return c;
}

namespace {

/// Qwen2.5-VL's window and full-attention blocks.
///
/// The mmproj states the PATTERN (`clip.vision.n_wa_pattern`: every Nth block attends over
/// the whole image) but never the window's edge, which llama.cpp keeps as an architecture
/// constant. config.json's `vision_config` has both (`window_size`, and the explicit
/// `fullatt_block_indexes` list), and a GGUF install keeps or writes one -- so read them
/// there, and derive the list from the pattern when only the pattern is available.
bool qwen25_attention(const std::string& config_json_path, int depth, int patch, int merge,
                      int pattern, int& window, std::vector<int>& fullatt) {
    std::ifstream f(config_json_path);
    if (f) {
        const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const nlohmann::json j = nlohmann::json::parse(text, nullptr, false);
        if (j.is_object() && j.contains("vision_config") && j["vision_config"].is_object()) {
            const nlohmann::json& v = j["vision_config"];
            window = v.value("window_size", 0);
            if (v.contains("fullatt_block_indexes") && v["fullatt_block_indexes"].is_array())
                for (const auto& e : v["fullatt_block_indexes"]) fullatt.push_back(e.get<int>());
        }
    }
    if (fullatt.empty() && pattern > 0)
        for (int i = 0; i < depth; ++i)
            if ((i + 1) % pattern == 0) fullatt.push_back(i);
    if (window > 0) return true;
    if (pattern <= 0) return false;
    // Qwen2.5-VL's published window is 112 px = eight 14-px patches, and llama.cpp bakes
    // the same relation in. Checked below against patch*merge, so a different patch size
    // cannot silently inherit it.
    window = 8 * patch;
    return window % (patch * merge) == 0;
}

}  // namespace

/// Qwen2.5-VL's projector: the windowed tower, which llama.cpp also splits into one
/// [P, P, C, hidden] conv per temporal slice. Its metadata has no merge size and no FFN
/// width (`clip.vision.feed_forward_length` is 0), so those come from the shapes: the
/// merger's fc1 is [hidden*merge^2, hidden*merge^2] and a block's ffn_gate is [hidden, inter].
VitConfig qwen25_from_gguf(const GgufFile& g, const std::string& mmproj_path) {
    auto num = [&](const char* k) {
        if (!g.kv_has(k)) throw std::runtime_error("vit: " + mmproj_path + " has no " + k);
        return g.kv_u64(k);
    };
    VitConfig c;
    c.family = VitFamily::Qwen25VL;
    c.depth = static_cast<int>(num("clip.vision.block_count"));
    c.hidden = static_cast<int>(num("clip.vision.embedding_length"));
    c.heads = static_cast<int>(num("clip.vision.attention.head_count"));
    c.out = static_cast<int>(num("clip.vision.projection_dim"));
    c.patch = static_cast<int>(num("clip.vision.patch_size"));
    c.eps = g.kv_has("clip.vision.attention.layer_norm_epsilon")
                ? static_cast<float>(g.kv_f64("clip.vision.attention.layer_norm_epsilon"))
                : 1e-6f;
    c.npos = 0;                                  // 2-D RoPE; there is no learned table
    if (c.hidden <= 0 || c.heads <= 0 || c.hidden % c.heads)
        throw std::runtime_error("vit: clip.vision.embedding_length is not a multiple of head_count");
    c.head_dim = c.hidden / c.heads;

    const GgufFile::TensorInfo& pe = g.tensor("v.patch_embd.weight");
    if (pe.dims.size() != 4 || static_cast<int>(pe.dims[0]) != c.patch ||
        static_cast<int>(pe.dims[1]) != c.patch || static_cast<int>(pe.dims[3]) != c.hidden)
        throw std::runtime_error("vit: " + mmproj_path + ": v.patch_embd.weight is not [P, P, C, hidden]");
    c.channels = static_cast<int>(pe.dims[2]);
    c.temporal = 1;
    while (g.has("v.patch_embd.weight." + std::to_string(c.temporal))) ++c.temporal;

    const GgufFile::TensorInfo& gate = g.tensor("v.blk.0.ffn_gate.weight");   // [hidden, inter]
    if (gate.dims.size() != 2 || static_cast<int>(gate.dims[0]) != c.hidden)
        throw std::runtime_error("vit: v.blk.0.ffn_gate.weight is not [hidden, inter]");
    c.inter = static_cast<int>(gate.dims[1]);

    const GgufFile::TensorInfo& mm0 = g.tensor("mm.0.weight");                // [hidden*M, hidden*M]
    if (mm0.dims.size() != 2 || static_cast<int>(mm0.dims[0]) != static_cast<int>(mm0.dims[1]))
        throw std::runtime_error("vit: mm.0.weight is not square");
    const int width = static_cast<int>(mm0.dims[0]);
    if (width % c.hidden) throw std::runtime_error("vit: the merger width is not a multiple of hidden");
    int merge = 1;
    while (merge * merge * c.hidden < width) ++merge;
    if (merge * merge * c.hidden != width)
        throw std::runtime_error("vit: the merger width is not hidden times a square merge factor");
    c.merge = merge;

    const int pattern = g.kv_has("clip.vision.n_wa_pattern")
                            ? static_cast<int>(g.kv_u64("clip.vision.n_wa_pattern")) : 0;
    const std::string cfg = (std::filesystem::path(mmproj_path).parent_path() / "config.json").string();
    if (!qwen25_attention(cfg, c.depth, c.patch, c.merge, pattern, c.window, c.fullatt))
        throw std::runtime_error(
            "vit: " + mmproj_path + " is Qwen2.5-VL's windowed projector, which does not state its "
            "window's edge; config.json beside it must carry vision_config.window_size (and "
            "fullatt_block_indexes, or clip.vision.n_wa_pattern). Qwen2.5-VL-3B's are 112 and [7,15,23,31].");
    for (int i : c.fullatt)
        if (i < 0 || i >= c.depth)
            throw std::runtime_error("vit: fullatt block " + std::to_string(i) + " is outside the tower");
    if (c.window % (c.patch * c.merge))
        throw std::runtime_error("vit: window " + std::to_string(c.window) + " is not a whole number of merge units");
    return c;
}

/// The tower a llama.cpp `mmproj` describes, from the FILE.
///
/// Every number this reader needs is in the file: `clip.vision.*` states the geometry
/// (which is why the container path needed a hand-written `vision.json`), the patch
/// embed's shape states the channels and the temporal split, the position table states
/// its own row count, and the deepstack indexes are in the tensor names themselves.
/// Each is cross-checked against the others, so a projector that does not describe the
/// tower this reader runs is refused rather than believed.
VitConfig VitConfig::from_gguf(const std::string& mmproj_path) {
    GgufFile g(mmproj_path);
    const std::string arch = g.kv_has("general.architecture") ? g.kv_str("general.architecture") : std::string();
    if (arch != "clip")
        throw std::runtime_error("vit: " + mmproj_path + " is a '" + arch + "' file, not a vision projector ('clip')");
    if (!g.kv_has("clip.has_vision_encoder") || g.kv_u64("clip.has_vision_encoder") == 0)
        throw std::runtime_error("vit: " + mmproj_path +
                                 " carries no vision encoder (clip.has_vision_encoder is false)");
    const std::string proj = g.kv_has("clip.projector_type") ? g.kv_str("clip.projector_type") : std::string();
    if (proj != "qwen3vl_merger" && proj != "qwen2.5vl_merger")
        throw std::runtime_error("vit: " + mmproj_path + " is a '" + proj +
                                 "' projector; this reader runs 'qwen3vl_merger' and 'qwen2.5vl_merger'");
    if (proj == "qwen2.5vl_merger") return qwen25_from_gguf(g, mmproj_path);

    auto num = [&](const char* k) {
        if (!g.kv_has(k)) throw std::runtime_error("vit: " + mmproj_path + " has no " + k);
        return g.kv_u64(k);
    };
    VitConfig c;
    c.family = VitFamily::Qwen3VL;
    c.depth = static_cast<int>(num("clip.vision.block_count"));
    c.hidden = static_cast<int>(num("clip.vision.embedding_length"));
    c.heads = static_cast<int>(num("clip.vision.attention.head_count"));
    c.inter = static_cast<int>(num("clip.vision.feed_forward_length"));
    c.out = static_cast<int>(num("clip.vision.projection_dim"));
    c.patch = static_cast<int>(num("clip.vision.patch_size"));
    c.merge = static_cast<int>(num("clip.vision.spatial_merge_size"));
    c.eps = g.kv_has("clip.vision.attention.layer_norm_epsilon")
                ? static_cast<float>(g.kv_f64("clip.vision.attention.layer_norm_epsilon"))
                : 1e-6f;
    if (c.hidden <= 0 || c.heads <= 0 || c.hidden % c.heads)
        throw std::runtime_error("vit: clip.vision.embedding_length " + std::to_string(c.hidden) +
                                 " is not a multiple of attention.head_count " + std::to_string(c.heads));
    c.head_dim = c.hidden / c.heads;
    if (c.depth <= 0 || c.patch <= 0 || c.merge <= 0 || c.out <= 0 || c.inter <= 0)
        throw std::runtime_error("vit: " + mmproj_path + " states a tower with a zero dimension");

    // The patch embed is one [P, P, C, hidden] conv per temporal slice: how llama.cpp
    // splits this tower's Conv3d, so the slice count IS the temporal patch size.
    const GgufFile::TensorInfo& pe = g.tensor("v.patch_embd.weight");
    if (pe.dims.size() != 4)
        throw std::runtime_error("vit: " + mmproj_path + ": v.patch_embd.weight is not [P, P, C, hidden]");
    c.temporal = 1;
    while (g.has("v.patch_embd.weight." + std::to_string(c.temporal))) ++c.temporal;
    c.channels = static_cast<int>(pe.dims[2]);
    if (static_cast<int>(pe.dims[0]) != c.patch || static_cast<int>(pe.dims[1]) != c.patch ||
        static_cast<int>(pe.dims[3]) != c.hidden)
        throw std::runtime_error("vit: v.patch_embd.weight is [" + std::to_string(pe.dims[0]) + ", " +
                                 std::to_string(pe.dims[1]) + ", " + std::to_string(pe.dims[2]) + ", " +
                                 std::to_string(pe.dims[3]) + "] where clip.vision says patch " +
                                 std::to_string(c.patch) + " and hidden " + std::to_string(c.hidden));
    const GgufFile::TensorInfo& pos = g.tensor("v.position_embd.weight");
    if (pos.dims.size() != 2 || static_cast<int>(pos.dims[0]) != c.hidden)
        throw std::runtime_error("vit: v.position_embd.weight is not [hidden, positions]");
    c.npos = static_cast<int>(pos.dims[1]);

    // Which blocks the deepstack mergers hang off is in the tensor NAMES
    // (v.deepstack.5.*) -- the same statement clip.vision.is_deepstack_layers makes,
    // and unlike that metadata array it survives every writer.
    std::vector<int> taps;
    for (const std::string& n : g.tensor_names()) {
        const std::string pre = "v.deepstack.";
        if (n.size() <= pre.size() || n.compare(0, pre.size(), pre) != 0) continue;
        const std::string rest = n.substr(pre.size());
        const auto dot = rest.find('.');
        if (dot == std::string::npos) continue;
        int idx = -1;
        try {
            idx = std::stoi(rest.substr(0, dot));
        } catch (const std::logic_error&) {
            continue;
        }
        if (std::find(taps.begin(), taps.end(), idx) == taps.end()) taps.push_back(idx);
    }
    std::sort(taps.begin(), taps.end());
    for (int i : taps)
        if (i < 0 || i >= c.depth)
            throw std::runtime_error("vit: deepstack index " + std::to_string(i) + " is outside the tower's " +
                                     std::to_string(c.depth) + " blocks");
    c.deepstack = taps;
    return c;
}

std::string vision_weights_path(const std::string& model_dir) {
    std::string file = "vision_weight.q4nx";
    {
        std::ifstream cf(model_dir + "/config.json");
        auto j = nlohmann::json::parse(cf, nullptr, false);
        if (j.is_object()) file = j.value("vision_model_weight", file);
    }
    std::error_code ec;
    if (std::filesystem::is_regular_file(model_dir + "/" + file, ec)) return file;
    // A GGUF install has no container: the projector arrives as llama.cpp's mmproj*.gguf
    // beside the language model (oflm-add fetches it). Any OTHER gguf here is the model.
    std::filesystem::directory_iterator it(model_dir, ec), end;
    if (!ec) {
        std::vector<std::string> found;
        for (const auto& e : it) {
            const std::string n = e.path().filename().string();
            if (e.path().extension() == ".gguf" && n.rfind("mmproj", 0) == 0) found.push_back(n);
        }
        if (found.size() == 1) return found.front();
        if (found.size() > 1)
            throw std::runtime_error("vit: " + model_dir + " holds " + std::to_string(found.size()) +
                                     " mmproj GGUFs; name the one to use in config.json's vision_model_weight");
    }
    return file;   // whatever config.json asked for; the loader names it when it is missing
}

VitConfig VitConfig::for_model_dir(const std::string& model_dir) {
    std::ifstream f(model_dir + "/config.json");
    if (!f) throw std::runtime_error("vit: cannot open " + model_dir + "/config.json");
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const nlohmann::json j = nlohmann::json::parse(text);
    // A GGUF install's projector describes itself; there is no container to read and no
    // side file to find. Checked first because such a model's config.json has no
    // vision_config either.
    {
        const std::string wf = vision_weights_path(model_dir);
        if (wf.size() > 5 && wf.compare(wf.size() - 5, 5, ".gguf") == 0) return from_gguf(model_dir + "/" + wf);
    }
    const std::string mt = j.value("model_type", std::string());
    // Qwen2.5-VL's decoder derives as plain qwen2, so the kernel set's family cannot tell
    // the two apart - only the container's own model_type does.
    if (mt == "qwen2_5_vl" || mt == "qwen2_5_vl_text") return qwen25_from_config_text(text);
    // Qwen3-VL-4B-Instruct-NPU2 ships no vision_config at all, so there is nothing here to
    // read the tower from - but the weight file determines all of it bar the head count
    // and the tap indexes. Those two come from config.json, where q4nx-build should be
    // writing them; until it does, oflm-add can add them at install time. Naming them in
    // the refusal is the point: a guessed head count gives plausible wrong embeddings.
    if (!j.contains("vision_config") && j.contains("vision_model_weight")) {
        // config.json cannot hold these. The downloader compares every registry-listed
        // file against a REMOTE manifest's byte count and re-pulls anything that differs,
        // so a key added locally is reverted (and on this model that is a 4 GB re-pull).
        // vision.json is not in the registry's file list, so it survives; oflm-add writes
        // it at install time.
        nlohmann::json side;
        {
            std::ifstream sf(model_dir + "/vision.json");
            if (sf) {
                auto parsed = nlohmann::json::parse(sf, nullptr, false);
                if (parsed.is_object()) side = parsed;
            }
        }
        const nlohmann::json& hv = j.contains("vision_heads") ? j["vision_heads"] : side["vision_heads"];
        const nlohmann::json& dv = j.contains("vision_deepstack_indexes") ? j["vision_deepstack_indexes"]
                                                                         : side["vision_deepstack_indexes"];
        if (!hv.is_number_integer() || !dv.is_array())
            throw std::runtime_error(
                "vit: " + model_dir + " has no vision_config, and this container's weight file cannot say how "
                "many attention heads its tower has or which blocks its deepstack mergers hang off. Write "
                "\"vision_heads\" (an integer) and \"vision_deepstack_indexes\" (an array) into vision.json "
                "beside the model - for Qwen3-VL-4B they are 16 and [5, 11, 17]. Everything else is read from "
                "the weight file and checked. (config.json cannot hold them: the downloader re-pulls any "
                "registry-listed file whose size changes.)");
        std::vector<int> taps;
        for (const auto& e : dv) taps.push_back(e.get<int>());
        const int heads = hv.get<int>();
        return qwen3vl_from_tensors(model_dir + "/" + j.value("vision_model_weight", std::string("vision_weight.q4nx")),
                                    heads, taps);
    }
    return from_config_text(text);
}

VitConfig VitConfig::qwen25_from_model_dir(const std::string& model_dir) {
    std::ifstream f(model_dir + "/config.json");
    if (!f) throw std::runtime_error("vit: cannot open " + model_dir + "/config.json");
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return qwen25_from_config_text(text);
}

VitConfig VitConfig::qwen25_from_config_text(const std::string& config_json) {
    const nlohmann::json j = nlohmann::json::parse(config_json);
    const auto it = j.find("vision_config");
    if (it == j.end() || !it->is_object() || it->empty())
        throw std::runtime_error("vit: config.json has no vision_config, so this container does not say what its "
                                 "windowed tower looks like");
    const nlohmann::json& v = *it;
    if (!v.contains("window_size") || !v.contains("fullatt_block_indexes"))
        throw std::runtime_error("vit: vision_config has no window_size / fullatt_block_indexes - this reader is "
                                 "Qwen2.5-VL's windowed tower; the full-attention one is VitConfig::from_config_text");
    if (!v.contains("depth"))
        throw std::runtime_error("vit: vision_config has no depth - Qwen2.5-VL's container keeps its source block, so "
                                 "this reads the plain transformers keys: depth, hidden_size, num_heads, "
                                 "intermediate_size, out_hidden_size, patch_size, temporal_patch_size, "
                                 "spatial_merge_size, window_size, fullatt_block_indexes");
    auto h = [&](const char* k) {
        if (!v.contains(k)) throw std::runtime_error("vit: vision_config has no " + std::string(k));
        return v.at(k);
    };
    VitConfig c;
    c.family = VitFamily::Qwen25VL;
    c.depth = h("depth");
    c.hidden = h("hidden_size");
    c.heads = h("num_heads");
    if (c.heads <= 0 || c.hidden % c.heads)
        throw std::runtime_error("vit: vision_config hidden_size " + std::to_string(c.hidden) +
                                 " is not a multiple of num_heads " + std::to_string(c.heads));
    // Neither head_dim nor an epsilon is a key on this tower: hidden/heads, and RMSNorm's
    // 1e-6, which transformers hardcodes in the module rather than reading from the config.
    c.head_dim = c.hidden / c.heads;
    c.inter = h("intermediate_size");
    c.out = h("out_hidden_size");
    c.patch = h("patch_size");
    c.temporal = h("temporal_patch_size");
    c.merge = h("spatial_merge_size");
    c.npos = 0;                       // no learned position table; position is entirely 2-D RoPE
    c.eps = v.value("rms_norm_eps", 1e-6f);
    c.channels = v.value("in_channels", v.value("in_chans", 3));
    c.window = h("window_size");
    for (const auto& b : v.at("fullatt_block_indexes")) c.fullatt.push_back(b.get<int>());
    const std::string act = v.value("hidden_act", std::string("silu"));
    if (act != "silu")
        throw std::runtime_error("vit: vision_config hidden_act '" + act + "' - this tower's MLP is SwiGLU over SiLU");
    if (c.window_side() <= 0)
        throw std::runtime_error("vit: window_size " + std::to_string(c.window) + " is smaller than one merge unit");
    return c;
}

static VitWeights load_vit_qwen25(const Q4nxFile& f, const VitConfig& cfg) {
    const std::string p = "model.visual.";
    const int H = cfg.hidden, I = cfg.inter, M = cfg.merge * cfg.merge;
    VitWeights w;
    {
        // Conv3d(bias=False), stored flat as the Qwen3-VL one is; untile() is not involved
        const TensorMeta& m = f.meta(p + "patch_embed.proj.weight");
        size_t nb = 0;
        const uint16_t* t = reinterpret_cast<const uint16_t*>(f.raw(p + "patch_embed.proj.weight", &nb));
        w.patch.out = H;
        w.patch.in = cfg.patch_dim();
        if (nb != static_cast<size_t>(H) * w.patch.in * 2 || m.dtype != "BF16")
            throw std::runtime_error("vit: patch_embed.proj.weight is not bf16 [hidden, C*T*P*P]");
        w.patch.w.assign(t, t + static_cast<size_t>(H) * w.patch.in);
        w.patch.b.assign(static_cast<size_t>(H), 0.f);
    }
    w.blocks.resize(cfg.depth);
    for (int i = 0; i < cfg.depth; ++i) {
        const std::string b = p + std::to_string(i) + ".";   // no "blocks." segment on this family
        VitBlock& B = w.blocks[i];
        B.ln1_w = f32_of(f, b + "rmsnorm1.weight");
        B.ln2_w = f32_of(f, b + "rmsnorm2.weight");
        B.q = untile(f, b + "attn.q_proj.weight", b + "attn.q_proj.bias", H, H);
        B.k = untile(f, b + "attn.k_proj.weight", b + "attn.k_proj.bias", H, H);
        B.v = untile(f, b + "attn.v_proj.weight", b + "attn.v_proj.bias", H, H);
        B.proj = untile(f, b + "attn.o_proj.weight", b + "attn.o_proj.bias", H, H);
        B.gate = untile(f, b + "mlp.gate_proj.weight", b + "mlp.gate_proj.bias", I, H);
        B.up = untile(f, b + "mlp.up_proj.weight", b + "mlp.up_proj.bias", I, H);
        B.down = untile(f, b + "mlp.down_proj.weight", b + "mlp.down_proj.bias", H, I);
    }
    w.merger_ln_w = f32_of(f, p + "merger.ln_q.weight");
    if (w.merger_ln_w.size() != static_cast<size_t>(H))
        throw std::runtime_error("vit: merger.ln_q.weight is not [hidden] - it normalises before the 2x2 concat");
    w.merger_fc1 = untile(f, p + "merger.mlp.0.weight", p + "merger.mlp.0.bias", H * M, H * M);
    w.merger_fc2 = untile(f, p + "merger.mlp.2.weight", p + "merger.mlp.2.bias", cfg.out, H * M);
    // The shipped 3B holds one tensor this tower does not read: `identity`, a 5120 x 5120
    // bf16 identity matrix, which is the 50 MiB the file was larger than the tower accounts
    // for. The closed engine presumably multiplies by it to move data through vision_mm
    // where the arithmetic is a copy. Anything else unaccounted for is refused rather than
    // loaded, because a missing piece reads as plausible numbers, not as an error. Header
    // read 2026-09-13; see .claude/plans/qwen25vl-container-size.md.
    const size_t want = 6 + 16 * static_cast<size_t>(cfg.depth);
    const size_t have = f.tensor_count();
    if (have != want && !(have == want + 1 && f.has("identity")))
        throw std::runtime_error("vit: " + f.path() + " holds " + std::to_string(have) +
                                 " tensors where this tower reads " + std::to_string(want) +
                                 " (plus `identity`, which it skips) - list its names before loading it");
    return w;
}

namespace {

/// The whole tensor, as a tower Linear: llama.cpp stores a matmul weight [in, out]
/// (its fastest dim is the input), the tower keeps [out, in] bf16.
Linear gguf_linear(const GgufFile& g, const std::string& base, int out, int in) {
    Linear L;
    L.out = out;
    L.in = in;
    const GgufFile::TensorInfo& t = g.tensor(base + ".weight");
    if (t.dims.size() != 2 || static_cast<int>(t.dims[0]) != in || static_cast<int>(t.dims[1]) != out)
        throw std::runtime_error("vit: " + base + ".weight is not [" + std::to_string(in) + ", " +
                                 std::to_string(out) + "] (the GGUF stores [in, out])");
    const std::vector<float> src = g.f32_all(base + ".weight");
    L.w.resize(static_cast<size_t>(out) * in);
    for (int o = 0; o < out; ++o)
        for (int i = 0; i < in; ++i) L.w[static_cast<size_t>(o) * in + i] = f32_to_bf16(src[static_cast<size_t>(i) + static_cast<size_t>(in) * o]);
    L.b = g.f32_all(base + ".bias");
    if (L.b.size() != static_cast<size_t>(out))
        throw std::runtime_error("vit: " + base + ".bias is [" + std::to_string(L.b.size()) + "] where " +
                                 std::to_string(out) + " is needed");
    return L;
}

}  // namespace

/// The same tower, from a llama.cpp mmproj. The names are llama.cpp's (`v.blk.N.attn_qkv`,
/// `mm.0`, `v.deepstack.N.fc1`) and the tensors are PLAIN -- no 64x256 pre-tiling, because
/// nothing downstream of this is the closed engine's vision_mm. So the only work beyond
/// reading is a transpose for every matmul, and the patch embed's split back into the
/// Conv3d this tower expects.
namespace {

/// Qwen2.5-VL's tower from a projector: split q/k/v, RMSNorm (weight only), SwiGLU, and a
/// merger whose norm runs over `hidden` before the 2x2 concat -- the same shape the
/// container loader reads, with llama.cpp's names.
VitWeights load_vit_gguf_qwen25(const GgufFile& g, const VitConfig& cfg, const std::string& path) {
    const int H = cfg.hidden, I = cfg.inter, M = cfg.merge * cfg.merge, P = cfg.patch, C = cfg.channels,
              T = cfg.temporal;
    VitWeights w;
    {
        w.patch.out = H;
        w.patch.in = cfg.patch_dim();
        w.patch.w.assign(static_cast<size_t>(H) * w.patch.in, 0);
        for (int t = 0; t < T; ++t) {
            const std::string nm = t == 0 ? std::string("v.patch_embd.weight")
                                          : "v.patch_embd.weight." + std::to_string(t);
            const GgufFile::TensorInfo& ti = g.tensor(nm);
            if (ti.dims.size() != 4 || static_cast<int>(ti.dims[0]) != P || static_cast<int>(ti.dims[1]) != P ||
                static_cast<int>(ti.dims[2]) != C || static_cast<int>(ti.dims[3]) != H)
                throw std::runtime_error("vit: " + nm + " is not [" + std::to_string(P) + ", " +
                                         std::to_string(P) + ", " + std::to_string(C) + ", " +
                                         std::to_string(H) + "]");
            const std::vector<float> src = g.f32_all(nm);
            for (int h = 0; h < H; ++h)
                for (int c = 0; c < C; ++c)
                    for (int ph = 0; ph < P; ++ph)
                        for (int pw = 0; pw < P; ++pw)
                            w.patch.w[((static_cast<size_t>(h) * C + c) * T + t) * P * P + ph * P + pw] =
                                f32_to_bf16(src[static_cast<size_t>(pw) + P * (ph + static_cast<size_t>(P) * (c + static_cast<size_t>(C) * h))]);
        }
        w.patch.b.assign(static_cast<size_t>(H), 0.f);    // Conv3d(bias=False) on this tower
    }
    w.blocks.resize(cfg.depth);
    for (int i = 0; i < cfg.depth; ++i) {
        const std::string b = "v.blk." + std::to_string(i) + ".";
        VitBlock& B = w.blocks[i];
        B.ln1_w = g.f32_all(b + "ln1.weight");            // RMSNorm: no bias on this tower
        B.ln2_w = g.f32_all(b + "ln2.weight");
        B.q = gguf_linear(g, b + "attn_q", H, H);
        B.k = gguf_linear(g, b + "attn_k", H, H);
        B.v = gguf_linear(g, b + "attn_v", H, H);
        B.proj = gguf_linear(g, b + "attn_out", H, H);
        B.gate = gguf_linear(g, b + "ffn_gate", I, H);
        B.up = gguf_linear(g, b + "ffn_up", I, H);
        B.down = gguf_linear(g, b + "ffn_down", H, I);
    }
    w.merger_ln_w = g.f32_all("v.post_ln.weight");
    if (w.merger_ln_w.size() != static_cast<size_t>(H))
        throw std::runtime_error("vit: v.post_ln.weight is not [hidden] - it normalises before the 2x2 concat");
    w.merger_fc1 = gguf_linear(g, "mm.0", H * M, H * M);
    w.merger_fc2 = gguf_linear(g, "mm.2", cfg.out, H * M);

    // 16 per block (2 norms, 7 linears with bias), then the patch embed's slices, the
    // position table llama.cpp writes and this tower does not read (2-D RoPE), post_ln and
    // the merger's two linears.
    const size_t want = 16 * static_cast<size_t>(cfg.depth) + static_cast<size_t>(T) + 6;
    const size_t have = g.tensor_names().size();
    if (have != want)
        throw std::runtime_error("vit: " + path + " holds " + std::to_string(have) +
                                 " tensors where this tower reads " + std::to_string(want) +
                                 " - list its names before loading it");
    return w;
}

}  // namespace

VitWeights load_vit_gguf(const std::string& path, const VitConfig& cfg) {
    GgufFile g(path);
    if (cfg.family == VitFamily::Qwen25VL) return load_vit_gguf_qwen25(g, cfg, path);
    if (cfg.family != VitFamily::Qwen3VL)
        throw std::runtime_error("vit: a GGUF projector is read for the Qwen3-VL and Qwen2.5-VL towers");
    const int H = cfg.hidden, I = cfg.inter, M = cfg.merge * cfg.merge, P = cfg.patch, C = cfg.channels,
              T = cfg.temporal;
    VitWeights w;
    {
        // patch_embd is one [P, P, C, H] conv per temporal slice (fastest dim = patch
        // WIDTH); this tower's patch vector is (C, T, P_h, P_w), the Conv3d input order
        // the processor's reorder_patches emits (imageproc.cpp).
        w.patch.out = H;
        w.patch.in = cfg.patch_dim();
        w.patch.w.assign(static_cast<size_t>(H) * w.patch.in, 0);
        for (int t = 0; t < T; ++t) {
            const std::string nm = t == 0 ? std::string("v.patch_embd.weight")
                                          : "v.patch_embd.weight." + std::to_string(t);
            const GgufFile::TensorInfo& ti = g.tensor(nm);
            if (ti.dims.size() != 4 || static_cast<int>(ti.dims[0]) != P || static_cast<int>(ti.dims[1]) != P ||
                static_cast<int>(ti.dims[2]) != C || static_cast<int>(ti.dims[3]) != H)
                throw std::runtime_error("vit: " + nm + " is not [" + std::to_string(P) + ", " + std::to_string(P) +
                                         ", " + std::to_string(C) + ", " + std::to_string(H) + "]");
            const std::vector<float> src = g.f32_all(nm);
            for (int h = 0; h < H; ++h)
                for (int c = 0; c < C; ++c)
                    for (int ph = 0; ph < P; ++ph)
                        for (int pw = 0; pw < P; ++pw)
                            w.patch.w[((static_cast<size_t>(h) * C + c) * T + t) * P * P + ph * P + pw] =
                                f32_to_bf16(src[static_cast<size_t>(pw) + P * (ph + static_cast<size_t>(P) * (c + static_cast<size_t>(C) * h))]);
        }
        w.patch.b = g.f32_all("v.patch_embd.bias");
        if (w.patch.b.size() != static_cast<size_t>(H))
            throw std::runtime_error("vit: v.patch_embd.bias is not [hidden]");
    }
    w.pos = g.f32_all("v.position_embd.weight");
    if (w.pos.size() != static_cast<size_t>(cfg.npos) * H)
        throw std::runtime_error("vit: v.position_embd.weight has the wrong shape");

    w.blocks.resize(cfg.depth);
    for (int i = 0; i < cfg.depth; ++i) {
        const std::string b = "v.blk." + std::to_string(i) + ".";
        VitBlock& B = w.blocks[i];
        B.ln1_w = g.f32_all(b + "ln1.weight");
        B.ln1_b = g.f32_all(b + "ln1.bias");
        B.ln2_w = g.f32_all(b + "ln2.weight");
        B.ln2_b = g.f32_all(b + "ln2.bias");
        B.qkv = gguf_linear(g, b + "attn_qkv", 3 * H, H);   // [q | k | v] along the output
        B.proj = gguf_linear(g, b + "attn_out", H, H);
        B.fc1 = gguf_linear(g, b + "ffn_up", I, H);
        B.fc2 = gguf_linear(g, b + "ffn_down", H, I);
    }
    w.merger_ln_w = g.f32_all("v.post_ln.weight");   // over `hidden`, BEFORE the 2x2 concat
    w.merger_ln_b = g.f32_all("v.post_ln.bias");
    w.merger_fc1 = gguf_linear(g, "mm.0", H * M, H * M);
    w.merger_fc2 = gguf_linear(g, "mm.2", cfg.out, H * M);

    for (int idx : cfg.deepstack) {
        const std::string d = "v.deepstack." + std::to_string(idx) + ".";
        Merger m;
        m.ln_w = g.f32_all(d + "norm.weight");
        m.ln_b = g.f32_all(d + "norm.bias");
        if (m.ln_w.size() != static_cast<size_t>(H) * M)
            throw std::runtime_error("vit: " + d + "norm.weight is [" + std::to_string(m.ln_w.size()) +
                                     "] where a post-shuffle norm is [" + std::to_string(H * M) + "]");
        m.fc1 = gguf_linear(g, d + "fc1", H * M, H * M);
        m.fc2 = gguf_linear(g, d + "fc2", cfg.out, H * M);
        w.deepstack.push_back(std::move(m));
    }

    // Every tensor accounted for, as the container loader demands: a missing piece reads
    // as plausible numbers rather than as an error, so an extra name is refused instead
    // of skipped. 12 per block, 6 per deepstack merger, then the patch embed (one weight
    // per temporal slice plus its bias), the position table, post_ln, and mm.0/mm.2.
    const size_t want = 12 * static_cast<size_t>(cfg.depth) + 6 * cfg.deepstack.size() +
                        static_cast<size_t>(T) + 8;
    const size_t have = g.tensor_names().size();
    if (have != want)
        throw std::runtime_error("vit: " + path + " holds " + std::to_string(have) +
                                 " tensors where this tower reads " + std::to_string(want) +
                                 " - list its names before loading it");
    return w;
}

VitWeights load_vit(const std::string& path, const VitConfig& cfg) {
    if (path.size() > 5 && path.compare(path.size() - 5, 5, ".gguf") == 0) return load_vit_gguf(path, cfg);
    Q4nxFile f(path);
    if (cfg.family == VitFamily::Qwen25VL) return load_vit_qwen25(f, cfg);
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
    // The deepstack mergers. Their norm is H * M wide, not H: they reshape into merge
    // groups and normalise across the whole row, where the tower's own merger normalises
    // each patch first. Reading one as the other does not broadcast, so a mix-up fails
    // here rather than producing plausible numbers.
    for (size_t j = 0; j < cfg.deepstack.size(); ++j) {
        const std::string d = p + "deepstack_merger_list." + std::to_string(j) + ".";
        Merger m;
        m.ln_w = f32_of(f, d + "norm.weight");
        m.ln_b = f32_of(f, d + "norm.bias");
        if (m.ln_w.size() != static_cast<size_t>(H) * M)
            throw std::runtime_error("vit: " + d + "norm.weight is [" + std::to_string(m.ln_w.size()) +
                                     "] where a post-shuffle norm is [" + std::to_string(H * M) + "]");
        m.fc1 = untile(f, d + "linear_fc1.weight", d + "linear_fc1.bias", H * M, H * M);
        m.fc2 = untile(f, d + "linear_fc2.weight", d + "linear_fc2.bias", cfg.out, H * M);
        w.deepstack.push_back(std::move(m));
    }
    return w;
}

namespace {

/// A deepstack merger: reshape into merge groups FIRST, then one LayerNorm across the
/// whole merged row. That ordering is the only structural difference from the tower's own
/// merger, and it is why the two norms are different widths.
std::vector<float> postshuffle_merger(const VitConfig& cfg, const Merger& m, const float* x, int n) {
    const int H = cfg.hidden, M = cfg.merge * cfg.merge, nm = n / M;
    std::vector<float> hn(static_cast<size_t>(nm) * H * M);
    layer_norm(x, nm, H * M, m.ln_w.data(), m.ln_b.data(), cfg.eps, hn.data());
    std::vector<float> mid(static_cast<size_t>(nm) * H * M), y(static_cast<size_t>(nm) * cfg.out);
    linear(hn.data(), nm, m.fc1, mid.data());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(mid.size()); ++i) mid[i] = gelu_erf(mid[i]);
    linear(mid.data(), nm, m.fc2, y.data());
    return y;
}

std::vector<float> forward_qwen3vl(const VitConfig& cfg, const VitWeights& w, const float* pixels, int gh, int gw,
                                   std::vector<std::vector<float>>* deep) {
    const int n = gh * gw, H = cfg.hidden, M = cfg.merge * cfg.merge;
    std::vector<int> ph, pw;
    position_ids(gh, gw, cfg.merge, ph, pw);
    std::vector<float> x(static_cast<size_t>(n) * H);
    linear(pixels, n, w.patch, x.data());
    add_pos_embed(cfg, w, gh, gw, ph, pw, x.data());
    std::vector<float> cs, sn;
    rope_tables(cfg, ph, pw, cs, sn);
    std::vector<float> hn(x.size()), qkv(static_cast<size_t>(n) * 3 * H), att(x.size()), tmp(x.size());
    std::vector<float> ff(static_cast<size_t>(n) * cfg.inter);
    const std::vector<int> whole = {0, n};
    if (deep) deep->clear();
    for (size_t bi = 0; bi < w.blocks.size(); ++bi) {
        const VitBlock& B = w.blocks[bi];
        layer_norm(x.data(), n, H, B.ln1_w.data(), B.ln1_b.data(), cfg.eps, hn.data());
        linear(hn.data(), n, B.qkv, qkv.data());
        attention(cfg, qkv.data(), n, cs, sn, whole, att.data());
        linear(att.data(), n, B.proj, tmp.data());
        for (size_t i = 0; i < x.size(); ++i) x[i] += tmp[i];
        layer_norm(x.data(), n, H, B.ln2_w.data(), B.ln2_b.data(), cfg.eps, hn.data());
        linear(hn.data(), n, B.fc1, ff.data());
#pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(ff.size()); ++i) ff[i] = gelu_tanh(ff[i]);
        linear(ff.data(), n, B.fc2, tmp.data());
        for (size_t i = 0; i < x.size(); ++i) x[i] += tmp[i];
        // The tap is taken AFTER the block runs, in the order cfg.deepstack lists them.
        if (deep) {
            const auto at = std::find(cfg.deepstack.begin(), cfg.deepstack.end(), static_cast<int>(bi));
            if (at != cfg.deepstack.end())
                deep->push_back(postshuffle_merger(cfg, w.deepstack[at - cfg.deepstack.begin()], x.data(), n));
        }
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

/// Qwen2.5-VL. The tokens run the whole stack in window order and come back at the end.
std::vector<float> forward_qwen25(const VitConfig& cfg, const VitWeights& w, const float* pixels, int gh, int gw) {
    const int n = gh * gw, H = cfg.hidden;
    const int unit = cfg.merge * cfg.merge, nm = n / unit;
    std::vector<int> idx, cu_win;
    window_index(cfg, gh, gw, idx, cu_win);
    // the permutation acts on whole merge units, so widen it to the tokens inside them
    std::vector<int> tok(static_cast<size_t>(n));
    for (size_t j = 0; j < idx.size(); ++j)
        for (int u = 0; u < unit; ++u) tok[j * unit + u] = idx[j] * unit + u;

    std::vector<float> x0(static_cast<size_t>(n) * H), x(x0.size());
    linear(pixels, n, w.patch, x0.data());
    for (int i = 0; i < n; ++i)
        std::memcpy(x.data() + static_cast<size_t>(i) * H, x0.data() + static_cast<size_t>(tok[i]) * H, sizeof(float) * H);

    std::vector<int> ph, pw, ph_p(n), pw_p(n);
    position_ids(gh, gw, cfg.merge, ph, pw);
    for (int i = 0; i < n; ++i) { ph_p[i] = ph[tok[i]]; pw_p[i] = pw[tok[i]]; }
    std::vector<float> cs, sn;
    rope_tables(cfg, ph_p, pw_p, cs, sn);

    const std::vector<int> whole = {0, n};
    std::vector<char> is_full(w.blocks.size(), 0);
    for (int b : cfg.fullatt)
        if (b >= 0 && b < static_cast<int>(is_full.size())) is_full[b] = 1;

    std::vector<float> hn(x.size()), qkv(static_cast<size_t>(n) * 3 * H), att(x.size()), tmp(x.size());
    std::vector<float> g(static_cast<size_t>(n) * cfg.inter), up(g.size());
    for (size_t i = 0; i < w.blocks.size(); ++i) {
        const VitBlock& B = w.blocks[i];
        rms_norm(x.data(), n, H, B.ln1_w.data(), cfg.eps, hn.data());
        linear(hn.data(), n, B.q, qkv.data(), 3 * H);
        linear(hn.data(), n, B.k, qkv.data() + H, 3 * H);
        linear(hn.data(), n, B.v, qkv.data() + 2 * H, 3 * H);
        attention(cfg, qkv.data(), n, cs, sn, is_full[i] ? whole : cu_win, att.data());
        linear(att.data(), n, B.proj, tmp.data());
        for (size_t t = 0; t < x.size(); ++t) x[t] += tmp[t];
        rms_norm(x.data(), n, H, B.ln2_w.data(), cfg.eps, hn.data());
        linear(hn.data(), n, B.gate, g.data());
        linear(hn.data(), n, B.up, up.data());
#pragma omp parallel for schedule(static)
        for (int t = 0; t < static_cast<int>(g.size()); ++t) g[t] = silu(g[t]) * up[t];
        linear(g.data(), n, B.down, tmp.data());
        for (size_t t = 0; t < x.size(); ++t) x[t] += tmp[t];
    }
    rms_norm(x.data(), n, H, w.merger_ln_w.data(), cfg.eps, hn.data());
    std::vector<float> mid(static_cast<size_t>(nm) * H * unit), y(static_cast<size_t>(nm) * cfg.out);
    linear(hn.data(), nm, w.merger_fc1, mid.data());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(mid.size()); ++i) mid[i] = gelu_erf(mid[i]);
    linear(mid.data(), nm, w.merger_fc2, y.data());
    // and back to the processor's row order
    std::vector<float> out(y.size());
    for (int j = 0; j < nm; ++j)
        std::memcpy(out.data() + static_cast<size_t>(idx[j]) * cfg.out, y.data() + static_cast<size_t>(j) * cfg.out,
                    sizeof(float) * cfg.out);
    return out;
}

}  // namespace

void window_index(const VitConfig& cfg, int gh, int gw, std::vector<int>& index, std::vector<int>& cu) {
    const int merge = cfg.merge, side = cfg.window_side();
    if (side <= 0)
        throw std::runtime_error("vit: window_size " + std::to_string(cfg.window) + " is smaller than one merge unit");
    const int lh = gh / merge, lw = gw / merge;
    // transformers does NOT modulo this, so a grid that already divides grows a whole extra
    // row and column of windows. They hold no units and collapse out of cu below.
    const int nwh = (lh + side - lh % side) / side, nww = (lw + side - lw % side) / side;
    index.clear();
    index.reserve(static_cast<size_t>(lh) * lw);
    cu.assign(1, 0);
    int seen = 0;
    for (int wh = 0; wh < nwh; ++wh)
        for (int ww = 0; ww < nww; ++ww) {
            for (int i = 0; i < side; ++i) {
                const int r = wh * side + i;
                if (r >= lh) break;
                for (int j = 0; j < side; ++j) {
                    const int c = ww * side + j;
                    if (c >= lw) break;
                    index.push_back(r * lw + c);
                    ++seen;
                }
            }
            const int boundary = seen * merge * merge;
            if (boundary != cu.back()) cu.push_back(boundary);
        }
}

std::vector<float> vit_forward_deepstack(const VitConfig& cfg, const VitWeights& w, const float* pixels,
                                         int gh, int gw, std::vector<std::vector<float>>* deep) {
    if (gh % cfg.merge || gw % cfg.merge) throw std::runtime_error("vit: the grid is not a multiple of the merge size");
    if (cfg.head_dim > 128) throw std::runtime_error("vit: head_dim > 128 does not fit the attention accumulator");
    if (deep && !cfg.deepstack.empty() && w.deepstack.size() != cfg.deepstack.size())
        throw std::runtime_error("vit: " + std::to_string(cfg.deepstack.size()) + " deepstack indexes but " +
                                 std::to_string(w.deepstack.size()) + " mergers were loaded");
    if (cfg.family == VitFamily::Qwen25VL) {
        if (deep) deep->clear();
        if (!cfg.deepstack.empty())
            throw std::runtime_error("vit: the windowed tower has no deepstack path");
        return forward_qwen25(cfg, w, pixels, gh, gw);
    }
    return forward_qwen3vl(cfg, w, pixels, gh, gw, deep);
}

std::vector<float> vit_forward(const VitConfig& cfg, const VitWeights& w, const float* pixels, int gh, int gw) {
    return vit_forward_deepstack(cfg, w, pixels, gh, gw, nullptr);
}

}  // namespace open_qwen36::vision
