#include "open_qwen36/dflash2_host.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <immintrin.h>

namespace open_qwen36 {
namespace dflash2 {

namespace {

float bf16f(uint16_t u) {
    uint32_t w = static_cast<uint32_t>(u) << 16;
    float f;
    std::memcpy(&f, &w, 4);
    return f;
}

float dot(const float* a, const float* b, size_t n) {
#ifdef __AVX2__
    if (n % 16 == 0) {
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        for (size_t i = 0; i < n; i += 16) {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        }
        alignas(32) float t[8];
        _mm256_store_ps(t, _mm256_add_ps(s0, s1));
        return ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
    }
#endif
    float s = 0;
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

size_t argmax16(const float* s) {
    size_t best = 0;
    for (size_t j = 1; j < kTopK; ++j)
        if (s[j] > s[best]) best = j;
    return best;
}

}  // namespace

std::vector<int> walk(const Lattice& lat) {
    std::vector<int> out;
    size_t pred = 0;
    for (size_t i = 0; i < lat.rows; ++i) {
        const size_t best = argmax16(&lat.score[(i * kTopK + pred) * kTopK]);
        out.push_back(lat.cand[i * kTopK + best]);
        pred = best;
    }
    return out;
}

std::vector<float> walk_margins(const Lattice& lat) {
    std::vector<float> out;
    size_t pred = 0;
    for (size_t i = 0; i < lat.rows; ++i) {
        const float* s = &lat.score[(i * kTopK + pred) * kTopK];
        const size_t best = argmax16(s);
        float second = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < kTopK; ++j)
            if (j != best) second = std::max(second, s[j]);
        out.push_back(s[best] - second);
        pred = best;
    }
    return out;
}

void top16(const float* logits, size_t n, int32_t* ids, float* vals) {
    size_t have = 0;
    for (size_t v = 0; v < n; ++v) {
        const float x = logits[v];
        if (have == kTopK && !(x > vals[kTopK - 1])) continue;
        size_t pos = have < kTopK ? have++ : kTopK - 1;
        while (pos > 0 && x > vals[pos - 1]) {
            vals[pos] = vals[pos - 1];
            ids[pos] = ids[pos - 1];
            --pos;
        }
        vals[pos] = x;
        ids[pos] = static_cast<int32_t>(v);
    }
}

void build_lattice(int anchor, size_t k, const float* logits, size_t vocab, const float* hn, size_t hidden,
                   const float* sel_hidden, size_t rank, const uint16_t* prev, const uint16_t* next, Lattice& lat) {
    lat.rows = k;
    lat.cand.assign(k * kTopK, 0);
    lat.logit.assign(k * kTopK, 0.f);
    lat.score.assign(k * kTopK * kTopK, 0.f);
#pragma omp parallel for
    for (long i = 0; i < static_cast<long>(k); ++i)
        top16(logits + i * vocab, vocab, &lat.cand[i * kTopK], &lat.logit[i * kTopK]);
    std::vector<float> gate(k * rank);
#pragma omp parallel for
    for (long ir = 0; ir < static_cast<long>(k * rank); ++ir) {
        const size_t i = ir / rank, r = ir % rank;
        gate[ir] = dot(hn + i * hidden, sel_hidden + r * hidden, hidden);
    }
    for (size_t i = 0; i < k; ++i) {
        const size_t npred = i == 0 ? 1 : kTopK;
        for (size_t p = 0; p < npred; ++p) {
            const int pid = i == 0 ? anchor : lat.cand[(i - 1) * kTopK + p];
            std::vector<float> pv(rank);
            for (size_t r = 0; r < rank; ++r) pv[r] = bf16f(prev[static_cast<size_t>(pid) * rank + r]) * gate[i * rank + r];
            for (size_t s = 0; s < kTopK; ++s) {
                const uint16_t* nx = next + static_cast<size_t>(lat.cand[i * kTopK + s]) * rank;
                double acc = 0;
                for (size_t r = 0; r < rank; ++r) acc += static_cast<double>(bf16f(nx[r])) * pv[r];
                lat.score[(i * kTopK + p) * kTopK + s] = static_cast<float>(acc) + lat.logit[i * kTopK + s];
            }
        }
        if (i == 0)
            for (size_t p = 1; p < kTopK; ++p)
                std::memcpy(&lat.score[p * kTopK], &lat.score[0], kTopK * sizeof(float));
    }
}

void conv2(const float* x, size_t rows, size_t n, const float* dyn, const float* base, int side, size_t group,
           float* y) {
    const size_t ng = n / group;
    const float* b0 = base + (static_cast<size_t>(side) * 2 + 0) * n;
    const float* b1 = base + (static_cast<size_t>(side) * 2 + 1) * n;
    // back to front so y may alias x: row t reads x[t - 1] before row t - 1 is overwritten
    for (size_t t = rows; t-- > 0;) {
        const float* d0 = dyn + (t * 4 + static_cast<size_t>(side) * 2 + 0) * ng;
        const float* d1 = dyn + (t * 4 + static_cast<size_t>(side) * 2 + 1) * ng;
        const float* xt = x + t * n;
        const float* xp = t ? x + (t - 1) * n : nullptr;
        float* yt = y + t * n;
        for (size_t c = 0; c < n; ++c) {
            const float w0 = d0[c / group] + b0[c], w1 = d1[c / group] + b1[c];
            yt[c] = w0 * xt[c] + (xp ? w1 * xp[c] : 0.f);
        }
    }
}

void rope_neox(float* x, size_t heads, size_t hd, double pos, double theta) {
    const size_t half = hd / 2;
    std::vector<float> c(half), s(half);
    for (size_t i = 0; i < half; ++i) {
        const double ang = pos * std::pow(theta, -static_cast<double>(2 * i) / static_cast<double>(hd));
        c[i] = static_cast<float>(std::cos(ang));
        s[i] = static_cast<float>(std::sin(ang));
    }
    for (size_t h = 0; h < heads; ++h) {
        float* v = x + h * hd;
        for (size_t i = 0; i < half; ++i) {
            const float a = v[i], b = v[i + half];
            v[i] = a * c[i] - b * s[i];
            v[i + half] = b * c[i] + a * s[i];
        }
    }
}

float round_f16(float x) {
    if (!std::isfinite(x)) return x;
    const float ax = std::fabs(x);
    if (ax >= 65520.f) return std::copysign(std::numeric_limits<float>::infinity(), x);
    int e;
    std::frexp(ax, &e);                                       // ax = f 2^e, f in [0.5, 1)
    const int ulp_exp = std::max(e - 11, -24);                // 11 significant bits; subnormals below 2^-14
    const float q = std::ldexp(std::nearbyint(std::ldexp(ax, -ulp_exp)), ulp_exp);
    return std::copysign(q, x);
}

void attention(const float* q, size_t rows, size_t nh, size_t kvh, size_t hd, const float* kc, const float* vc,
               size_t L, long ctx_pos0, const float* kb, const float* vb, long p0, size_t window, float* out) {
    const size_t grp = nh / kvh, kvw = kvh * hd;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
#pragma omp parallel for
    for (long gt = 0; gt < static_cast<long>(kvh * rows); ++gt) {
        {
            const long g = gt / static_cast<long>(rows), t = gt % static_cast<long>(rows);
            const long P = p0 + t;
            long lo = 0;   // first context row this row sees
            if (window) lo = std::min<long>(static_cast<long>(L), std::max<long>(0, P - static_cast<long>(window) + 1 - ctx_pos0));
            const size_t nk = (L > static_cast<size_t>(lo) ? L - lo : 0) + rows;
            std::vector<float> sc(nk);
            for (size_t hl = 0; hl < grp; ++hl) {
                const size_t h = g * grp + hl;
                const float* qv = q + (t * nh + h) * hd;
                float mx = -std::numeric_limits<float>::infinity();
                for (size_t j = 0; j < nk; ++j) {
                    const float* kv = j + lo < L ? kc + (j + lo) * kvw + g * hd : kb + (j + lo - L) * kvw + g * hd;
                    sc[j] = dot(qv, kv, hd) * scale;
                    mx = std::max(mx, sc[j]);
                }
                double sum = 0;
                for (size_t j = 0; j < nk; ++j) {
                    sc[j] = std::exp(sc[j] - mx);
                    sum += sc[j];
                }
                const float inv = static_cast<float>(1.0 / sum);
                float* o = out + (t * nh + h) * hd;
                std::fill(o, o + hd, 0.f);
                for (size_t j = 0; j < nk; ++j) {
                    const float* vv = j + lo < L ? vc + (j + lo) * kvw + g * hd : vb + (j + lo - L) * kvw + g * hd;
                    const float p = sc[j] * inv;
                    for (size_t d = 0; d < hd; ++d) o[d] += p * vv[d];
                }
            }
        }
    }
}

namespace {

inline __m256 exp8(__m256 x) {
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.0f)), _mm256_set1_ps(88.0f));
    const __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), r);
    const __m256 r2 = _mm256_mul_ps(r, r);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(p, r2, _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

}  // namespace

void silu_mul(const float* g, const float* u, size_t n, float* out) {
#pragma omp parallel for if(n >= 65536)
    for (long i = 0; i < static_cast<long>(n); i += 8) {
        const __m256 v = _mm256_loadu_ps(g + i);
        const __m256 s = _mm256_div_ps(v, _mm256_add_ps(_mm256_set1_ps(1.0f), exp8(_mm256_sub_ps(_mm256_setzero_ps(), v))));
        _mm256_storeu_ps(out + i, _mm256_mul_ps(s, _mm256_loadu_ps(u + i)));
    }
}

void GemvPlan::index_bands() {
    slices = 0;
    for (const auto& it : items) slices = std::max<size_t>(slices, it.slice + 1);
    band_y.assign(n / 64 * slices, -1);
    for (const auto& it : items)
        for (int b = 0; b < it.nb; ++b) band_y[static_cast<size_t>(it.b0 + b) * slices + it.slice] = it.y + b * static_cast<int>(kYBand);
}

namespace {

inline void transpose8(__m256& r0, __m256& r1, __m256& r2, __m256& r3, __m256& r4, __m256& r5, __m256& r6, __m256& r7) {
    __m256 t0 = _mm256_unpacklo_ps(r0, r1), t1 = _mm256_unpackhi_ps(r0, r1), t2 = _mm256_unpacklo_ps(r2, r3),
           t3 = _mm256_unpackhi_ps(r2, r3), t4 = _mm256_unpacklo_ps(r4, r5), t5 = _mm256_unpackhi_ps(r4, r5),
           t6 = _mm256_unpacklo_ps(r6, r7), t7 = _mm256_unpackhi_ps(r6, r7);
    __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44), s1 = _mm256_shuffle_ps(t0, t2, 0xEE), s2 = _mm256_shuffle_ps(t1, t3, 0x44),
           s3 = _mm256_shuffle_ps(t1, t3, 0xEE), s4 = _mm256_shuffle_ps(t4, t6, 0x44), s5 = _mm256_shuffle_ps(t4, t6, 0xEE),
           s6 = _mm256_shuffle_ps(t5, t7, 0x44), s7 = _mm256_shuffle_ps(t5, t7, 0xEE);
    r0 = _mm256_permute2f128_ps(s0, s4, 0x20);
    r1 = _mm256_permute2f128_ps(s1, s5, 0x20);
    r2 = _mm256_permute2f128_ps(s2, s6, 0x20);
    r3 = _mm256_permute2f128_ps(s3, s7, 0x20);
    r4 = _mm256_permute2f128_ps(s0, s4, 0x31);
    r5 = _mm256_permute2f128_ps(s1, s5, 0x31);
    r6 = _mm256_permute2f128_ps(s2, s6, 0x31);
    r7 = _mm256_permute2f128_ps(s3, s7, 0x31);
}

}  // namespace

void hadamard8(const float* x, size_t K, float* out) {
    const size_t nbk = K / 1024;
#pragma omp parallel for if(nbk >= 16)
    for (long b = 0; b < static_cast<long>(nbk); ++b) {
        alignas(32) float T[1024 * 8];
        for (size_t j0 = 0; j0 < 1024; j0 += 8) {
            __m256 r[8];
            for (size_t t = 0; t < 8; ++t) r[t] = _mm256_loadu_ps(x + t * K + b * 1024 + j0);
            transpose8(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
            for (size_t i = 0; i < 8; ++i) _mm256_store_ps(T + (j0 + i) * 8, r[i]);
        }
        for (size_t h = 1; h < 1024; h <<= 1)
            for (size_t i = 0; i < 1024; i += 2 * h)
                for (size_t j = i; j < i + h; ++j) {
                    const __m256 a = _mm256_load_ps(T + j * 8), c = _mm256_load_ps(T + (j + h) * 8);
                    _mm256_store_ps(T + j * 8, _mm256_add_ps(a, c));
                    _mm256_store_ps(T + (j + h) * 8, _mm256_sub_ps(a, c));
                }
        const __m256 sc = _mm256_set1_ps(1.0f / 32.0f);
        for (size_t j0 = 0; j0 < 1024; j0 += 8) {
            __m256 r[8];
            for (size_t i = 0; i < 8; ++i) r[i] = _mm256_mul_ps(_mm256_load_ps(T + (j0 + i) * 8), sc);
            transpose8(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
            for (size_t t = 0; t < 8; ++t) _mm256_storeu_ps(out + t * K + b * 1024 + j0, r[t]);
        }
    }
}

void build_records(const float* xr, size_t K, uint8_t* rec) {
    const size_t G = K / 128;
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
#pragma omp parallel for if(G >= 128)
    for (long gi = 0; gi < static_cast<long>(G * 2); ++gi) {
        const size_t grp = gi / 2, quad = gi % 2;
        uint8_t* r = rec + gi * kRec;
        const float* src[4];
        for (size_t tok = 0; tok < 4; ++tok) src[tok] = xr + (quad * 4 + tok) * K + grp * 128;
        __m256 mx = _mm256_setzero_ps();
        for (size_t tok = 0; tok < 4; ++tok)
            for (size_t i = 0; i < 128; i += 8) mx = _mm256_max_ps(mx, _mm256_and_ps(_mm256_loadu_ps(src[tok] + i), absmask));
        alignas(32) float mxs[8];
        _mm256_store_ps(mxs, mx);
        float m = 0;
        for (int i = 0; i < 8; ++i) m = std::max(m, mxs[i]);
        int sh = 0;
        if (m > 0) {
            uint32_t bits;
            std::memcpy(&bits, &m, 4);
            const int e = static_cast<int>((bits >> 23) & 255) - 126;      // m = f 2^e, f in [0.5, 1)
            sh = std::min(126, std::max(-100, 15 - e));
        }
        const uint32_t sbits = static_cast<uint32_t>(sh + 127) << 23;
        float scf;
        std::memcpy(&scf, &sbits, 4);
        const __m256 sc = _mm256_set1_ps(scf);
        const __m256i hi = _mm256_set1_epi32(32767);
        int16_t* xi = reinterpret_cast<int16_t*>(r);
        int32_t* sums = reinterpret_cast<int32_t*>(r + 1024);
        for (size_t tok = 0; tok < 4; ++tok) {
            __m256i acc = _mm256_setzero_si256();
            for (size_t o = 0; o < 16; ++o) {
                __m256i v = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(src[tok] + o * 8), sc));
                v = _mm256_min_epi32(v, hi);
                acc = _mm256_add_epi32(acc, v);
                const __m256i p = _mm256_permute4x64_epi64(_mm256_packs_epi32(v, v), 0x08);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(xi + (o * 4 + tok) * 8), _mm256_castsi256_si128(p));
            }
            alignas(32) int32_t a8[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(a8), acc);
            const int32_t s = a8[0] + a8[1] + a8[2] + a8[3] + a8[4] + a8[5] + a8[6] + a8[7];
            for (size_t p = 0; p < 8; ++p) sums[tok * 8 + p] = s;
        }
        std::memset(r + 1152, 0, kRec - 1152);
        std::memcpy(r + 1152, &sh, 4);
    }
}

void build_tables(const GemvPlan& p, const uint8_t* rec, uint8_t* tab) {
    for (long i = 0; i < static_cast<long>(p.items.size()); ++i) {
        const GemvItem& it = p.items[i];
        uint8_t* e = tab + static_cast<size_t>(it.tab) * p.tab_bytes;
        const int32_t hdr[2] = {it.nb, it.nkt};
        std::memcpy(e, hdr, sizeof hdr);
        std::memcpy(e + kHdr, rec + static_cast<size_t>(it.kt0) * 4 * kRec, static_cast<size_t>(it.nkt) * 4 * kRec);
    }
}

void reduce_y(const GemvPlan& p, const float* y, float* out) {
    const size_t N = p.n, S = p.slices;
#pragma omp parallel for if(N >= 65536)
    for (long b = 0; b < static_cast<long>(N / 64); ++b) {
        const size_t row0 = static_cast<size_t>(b) * 64;
        for (size_t half = 0; half < 2; ++half)
            for (size_t quad = 0; quad < 2; ++quad)
                for (size_t j = 0; j < 4; ++j)
                    for (size_t tok = 0; tok < 4; ++tok) {
                        const size_t at = (((half * 2 + quad) * 4 + j) * 4 + tok) * 8;
                        __m256 acc = _mm256_loadu_ps(y + p.band_y[b * S] + at);
                        for (size_t s = 1; s < S; ++s) acc = _mm256_add_ps(acc, _mm256_loadu_ps(y + p.band_y[b * S + s] + at));
                        _mm256_storeu_ps(out + (quad * 4 + tok) * N + row0 + half * 32 + j * 8, acc);
                    }
    }
}

}  // namespace dflash2
}  // namespace open_qwen36
