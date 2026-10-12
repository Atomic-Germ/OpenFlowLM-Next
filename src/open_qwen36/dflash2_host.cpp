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

void build_records(const float* xr, size_t K, uint8_t* rec) {
    const size_t G = K / 128;
#pragma omp parallel for
    for (long gi = 0; gi < static_cast<long>(G * 2); ++gi) {
        const size_t grp = gi / 2, quad = gi % 2;
        uint8_t* r = rec + gi * kRec;
        double m = 0;
        for (size_t tok = 0; tok < 4; ++tok)
            for (size_t i = 0; i < 128; ++i)
                m = std::max(m, std::fabs(static_cast<double>(xr[(quad * 4 + tok) * K + grp * 128 + i])));
        int sh = 0;
        if (m > 0) {
            int e;
            std::frexp(m, &e);
            sh = std::min(126, std::max(-100, 15 - e));       // 14 - floor(log2 m)
        }
        const double sc = std::ldexp(1.0, sh);
        int16_t* xi = reinterpret_cast<int16_t*>(r);
        int32_t* sums = reinterpret_cast<int32_t*>(r + 1024);
        for (size_t tok = 0; tok < 4; ++tok) {
            int32_t s = 0;
            for (size_t i = 0; i < 128; ++i) {
                double v = std::nearbyint(static_cast<double>(xr[(quad * 4 + tok) * K + grp * 128 + i]) * sc);
                v = std::min(32767.0, std::max(-32768.0, v));
                const int16_t q = static_cast<int16_t>(v);
                xi[((i / 8) * 4 + tok) * 8 + i % 8] = q;
                s += q;
            }
            for (size_t p = 0; p < 8; ++p) sums[tok * 8 + p] = s;
        }
        std::memset(r + 1152, 0, kRec - 1152);
        std::memcpy(r + 1152, &sh, 4);
    }
}

void build_tables(const GemvPlan& p, const uint8_t* rec, uint8_t* tab) {
#pragma omp parallel for
    for (long i = 0; i < static_cast<long>(p.items.size()); ++i) {
        const GemvItem& it = p.items[i];
        uint8_t* e = tab + static_cast<size_t>(it.tab) * p.tab_bytes;
        const int32_t hdr[2] = {it.nb, it.nkt};
        std::memcpy(e, hdr, sizeof hdr);
        std::memcpy(e + kHdr, rec + static_cast<size_t>(it.kt0) * 4 * kRec, static_cast<size_t>(it.nkt) * 4 * kRec);
    }
}

void reduce_y(const GemvPlan& p, const float* y, float* out) {
    const size_t N = p.n;
    std::fill(out, out + kRows * N, 0.f);
    int max_slice = 0;
    for (const auto& it : p.items) max_slice = std::max(max_slice, it.slice);
    // one slice at a time: a slice's items cover disjoint bands, so its loop has no races
    for (int s = 0; s <= max_slice; ++s) {
#pragma omp parallel for
        for (long i = 0; i < static_cast<long>(p.items.size()); ++i) {
            const GemvItem& it = p.items[i];
            if (it.slice != s) continue;
            for (int b = 0; b < it.nb; ++b) {
                const float* yb = y + it.y + static_cast<size_t>(b) * kYBand;
                const size_t row0 = static_cast<size_t>(it.b0 + b) * 64;
                for (size_t half = 0; half < 2; ++half)
                    for (size_t quad = 0; quad < 2; ++quad)
                        for (size_t j = 0; j < 4; ++j)
                            for (size_t tok = 0; tok < 4; ++tok) {
                                const float* src = yb + (((half * 2 + quad) * 4 + j) * 4 + tok) * 8;
                                float* dst = out + (quad * 4 + tok) * N + row0 + half * 32 + j * 8;
                                for (size_t q = 0; q < 8; ++q) dst[q] += src[q];
                            }
            }
        }
    }
}

}  // namespace dflash2
}  // namespace open_qwen36
