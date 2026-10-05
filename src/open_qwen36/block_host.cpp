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
#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
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

namespace {

/// One 8x8 float tile, src rows `sstride` apart, dst rows `dstride` apart.
///
/// The scalar loop this replaces walks one side of every 32x32 block four bytes at a time --
/// 32 separate cache lines touched per source row -- and reached about 10 GB/s over sixteen
/// threads on a 2582-token prefill's 16 GB of transposes. The AVX2 network reads eight whole
/// 32-byte rows, shuffles them in registers and writes eight whole 32-byte rows, so both
/// sides move in vector-width runs and the only scattered access left is one cache line per
/// row of a tile. It is pure data movement, so the result is identical to the last bit.
inline void t8x8(const float* src, size_t sstride, float* dst, size_t dstride) {
    __m256 r0 = _mm256_loadu_ps(src + 0 * sstride), r1 = _mm256_loadu_ps(src + 1 * sstride);
    __m256 r2 = _mm256_loadu_ps(src + 2 * sstride), r3 = _mm256_loadu_ps(src + 3 * sstride);
    __m256 r4 = _mm256_loadu_ps(src + 4 * sstride), r5 = _mm256_loadu_ps(src + 5 * sstride);
    __m256 r6 = _mm256_loadu_ps(src + 6 * sstride), r7 = _mm256_loadu_ps(src + 7 * sstride);
    const __m256 u0 = _mm256_unpacklo_ps(r0, r1), u1 = _mm256_unpackhi_ps(r0, r1);
    const __m256 u2 = _mm256_unpacklo_ps(r2, r3), u3 = _mm256_unpackhi_ps(r2, r3);
    const __m256 u4 = _mm256_unpacklo_ps(r4, r5), u5 = _mm256_unpackhi_ps(r4, r5);
    const __m256 u6 = _mm256_unpacklo_ps(r6, r7), u7 = _mm256_unpackhi_ps(r6, r7);
    const __m256 s0 = _mm256_shuffle_ps(u0, u2, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s1 = _mm256_shuffle_ps(u0, u2, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s2 = _mm256_shuffle_ps(u1, u3, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s3 = _mm256_shuffle_ps(u1, u3, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s4 = _mm256_shuffle_ps(u4, u6, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s5 = _mm256_shuffle_ps(u4, u6, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s6 = _mm256_shuffle_ps(u5, u7, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s7 = _mm256_shuffle_ps(u5, u7, _MM_SHUFFLE(3, 2, 3, 2));
    _mm256_storeu_ps(dst + 0 * dstride, _mm256_permute2f128_ps(s0, s4, 0x20));
    _mm256_storeu_ps(dst + 1 * dstride, _mm256_permute2f128_ps(s1, s5, 0x20));
    _mm256_storeu_ps(dst + 2 * dstride, _mm256_permute2f128_ps(s2, s6, 0x20));
    _mm256_storeu_ps(dst + 3 * dstride, _mm256_permute2f128_ps(s3, s7, 0x20));
    _mm256_storeu_ps(dst + 4 * dstride, _mm256_permute2f128_ps(s0, s4, 0x31));
    _mm256_storeu_ps(dst + 5 * dstride, _mm256_permute2f128_ps(s1, s5, 0x31));
    _mm256_storeu_ps(dst + 6 * dstride, _mm256_permute2f128_ps(s2, s6, 0x31));
    _mm256_storeu_ps(dst + 7 * dstride, _mm256_permute2f128_ps(s3, s7, 0x31));
}

/// dst[t * width + n] = src[n * T + t] for n in [0, width), t in [0, T): one thread's slice of
/// the N axis, 8x8 tiles where both sides are whole tiles and scalar at the edges.
void transpose_range(const float* src, size_t T, size_t width, size_t n0, size_t n1, float* dst) {
    constexpr size_t B = 64;                       // cache block, a whole number of 8x8 tiles
    for (size_t nb = n0; nb < n1; nb += B)
        for (size_t tb = 0; tb < T; tb += B) {
            const size_t ne = std::min(nb + B, n1), te = std::min(tb + B, T);
            size_t n = nb;
            for (; n + 8 <= ne; n += 8) {
                size_t t = tb;
                for (; t + 8 <= te; t += 8) t8x8(src + n * T + t, T, dst + t * width + n, width);
                for (; t < te; ++t)
                    for (size_t i = 0; i < 8; ++i) dst[t * width + n + i] = src[(n + i) * T + t];
            }
            for (; n < ne; ++n)
                for (size_t t = tb; t < te; ++t) dst[t * width + n] = src[n * T + t];
        }
}

}  // namespace

void transpose(const float* y, size_t N, size_t T, float* out) {
    const long long nthr = omp_get_max_threads();
    const size_t chunk = (N / 8 + nthr - 1) / nthr * 8;   // whole tiles per thread
#pragma omp parallel for
    for (long long i = 0; i < nthr; ++i) {
        const size_t a = std::min(static_cast<size_t>(i) * chunk, N);
        transpose_range(y, T, N, a, std::min(a + chunk, N), out);
    }
}

void transpose_parts(const float* y, size_t T, const TransposePart* parts, size_t n_parts) {
    // One parallel region for every part, not one each: a full-attention layer asks for four
    // ranges and four regions is four thread wake-ups, which under OMP_WAIT_POLICY=PASSIVE is
    // four times the wake-up and four chances for the NPU to see the cores come up.
    // (MSVC's OpenMP ignores `collapse`, so the two axes are flattened by hand.)
    const long long nthr = omp_get_max_threads();
#pragma omp parallel for
    for (long long j = 0; j < static_cast<long long>(n_parts) * nthr; ++j) {
        const TransposePart& q = parts[j / nthr];
        const long long i = j % nthr;
        const size_t chunk = (q.width / 8 + nthr - 1) / nthr * 8;
        const size_t a = std::min(static_cast<size_t>(i) * chunk, q.width);
        transpose_range(y + q.off * T, T, q.width, a, std::min(a + chunk, q.width), q.dst);
    }
}

void split_rows(const float* y, size_t T, size_t N, const TransposePart* parts, size_t n_parts) {
#pragma omp parallel for
    for (long long t = 0; t < static_cast<long long>(T); ++t)
        for (size_t i = 0; i < n_parts; ++i) {
            const TransposePart& q = parts[i];
            std::memcpy(q.dst + static_cast<size_t>(t) * q.width, y + static_cast<size_t>(t) * N + q.off, q.width * 4);
        }
}

void hadamard_rows(float* x, size_t T, size_t K, size_t block, const float* signs) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(block));   // 1/32 for 1024: exact
#pragma omp parallel for
    for (long long t = 0; t < static_cast<long long>(T); ++t) {
        float* row = x + static_cast<size_t>(t) * K;
        if (signs)
            for (size_t j = 0; j < K; ++j) row[j] *= signs[j];
        for (size_t b0 = 0; b0 + block <= K; b0 += block) {
            float* v = row + b0;
            for (size_t h = 1; h < block; h <<= 1)
                for (size_t i = 0; i < block; i += 2 * h)
                    for (size_t j = i; j < i + h; ++j) {
                        const float a = v[j], c = v[j + h];
                        v[j] = a + c;
                        v[j + h] = a - c;
                    }
            for (size_t j = 0; j < block; ++j) v[j] *= scale;
        }
    }
}

namespace {

/// f32_to_bf16 on eight lanes, the same integer arithmetic (round to nearest even, no NaN
/// special case), packed to eight uint16 in order.
inline __m128i bf16x8(__m256 v) {
    const __m256i u = _mm256_castps_si256(v);
    const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1));
    const __m256i r = _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(u, _mm256_set1_epi32(0x7FFF)), lsb), 16);
    // every lane is now in [0, 0xFFFF], so the unsigned-saturating pack is exact
    return _mm_packus_epi32(_mm256_castsi256_si128(r), _mm256_extracti128_si256(r, 1));
}

/// bf16_to_f32(f32_to_bf16(x)) on eight lanes.
inline __m256 bf16r8(__m256 v) {
    const __m256i u = _mm256_castps_si256(v);
    const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1));
    const __m256i r = _mm256_add_epi32(_mm256_add_epi32(u, _mm256_set1_epi32(0x7FFF)), lsb);
    return _mm256_castsi256_ps(_mm256_and_si256(r, _mm256_set1_epi32(static_cast<int>(0xFFFF0000u))));
}

/// The 8x8 transpose of t8x8, register to register.
inline void tr8(__m256& r0, __m256& r1, __m256& r2, __m256& r3, __m256& r4, __m256& r5, __m256& r6, __m256& r7) {
    const __m256 u0 = _mm256_unpacklo_ps(r0, r1), u1 = _mm256_unpackhi_ps(r0, r1);
    const __m256 u2 = _mm256_unpacklo_ps(r2, r3), u3 = _mm256_unpackhi_ps(r2, r3);
    const __m256 u4 = _mm256_unpacklo_ps(r4, r5), u5 = _mm256_unpackhi_ps(r4, r5);
    const __m256 u6 = _mm256_unpacklo_ps(r6, r7), u7 = _mm256_unpackhi_ps(r6, r7);
    const __m256 s0 = _mm256_shuffle_ps(u0, u2, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s1 = _mm256_shuffle_ps(u0, u2, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s2 = _mm256_shuffle_ps(u1, u3, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s3 = _mm256_shuffle_ps(u1, u3, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s4 = _mm256_shuffle_ps(u4, u6, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s5 = _mm256_shuffle_ps(u4, u6, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s6 = _mm256_shuffle_ps(u5, u7, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s7 = _mm256_shuffle_ps(u5, u7, _MM_SHUFFLE(3, 2, 3, 2));
    r0 = _mm256_permute2f128_ps(s0, s4, 0x20);
    r1 = _mm256_permute2f128_ps(s1, s5, 0x20);
    r2 = _mm256_permute2f128_ps(s2, s6, 0x20);
    r3 = _mm256_permute2f128_ps(s3, s7, 0x20);
    r4 = _mm256_permute2f128_ps(s0, s4, 0x31);
    r5 = _mm256_permute2f128_ps(s1, s5, 0x31);
    r6 = _mm256_permute2f128_ps(s2, s6, 0x31);
    r7 = _mm256_permute2f128_ps(s3, s7, 0x31);
}

/// hadamard_rows' butterflies on one run of n floats (n a power of two, at least 8), the same
/// stages in the same order with the same operands: stage h maps (v[j], v[j + h]) to
/// (a + c, a - c). Stages 1, 2 and 4 stay inside one 8-lane vector (a lane swap, then the sum
/// on the low lane of each pair and the difference, taken as swapped - original so it is
/// a - c and not -(c - a), on the high one: the two differ in the sign of an exact zero);
/// stages 8 and up pair whole vectors, two stages per pass over the run.
inline void fwht_run(float* v, size_t n) {
    for (size_t j = 0; j < n; j += 8) {
        __m256 x = _mm256_loadu_ps(v + j);
        __m256 s = _mm256_permute_ps(x, 0xB1);                                  // h = 1
        x = _mm256_blend_ps(_mm256_add_ps(x, s), _mm256_sub_ps(s, x), 0xAA);
        s = _mm256_permute_ps(x, 0x4E);                                         // h = 2
        x = _mm256_blend_ps(_mm256_add_ps(x, s), _mm256_sub_ps(s, x), 0xCC);
        s = _mm256_permute2f128_ps(x, x, 0x01);                                 // h = 4
        x = _mm256_blend_ps(_mm256_add_ps(x, s), _mm256_sub_ps(s, x), 0xF0);
        _mm256_storeu_ps(v + j, x);
    }
    size_t h = 8;
    for (; 2 * h < n; h *= 4)                       // stages h and 2h in one pass
        for (size_t i = 0; i < n; i += 4 * h)
            for (size_t j = i; j < i + h; j += 8) {
                const __m256 a0 = _mm256_loadu_ps(v + j), a1 = _mm256_loadu_ps(v + j + h);
                const __m256 a2 = _mm256_loadu_ps(v + j + 2 * h), a3 = _mm256_loadu_ps(v + j + 3 * h);
                const __m256 b0 = _mm256_add_ps(a0, a1), b1 = _mm256_sub_ps(a0, a1);
                const __m256 b2 = _mm256_add_ps(a2, a3), b3 = _mm256_sub_ps(a2, a3);
                _mm256_storeu_ps(v + j, _mm256_add_ps(b0, b2));
                _mm256_storeu_ps(v + j + 2 * h, _mm256_sub_ps(b0, b2));
                _mm256_storeu_ps(v + j + h, _mm256_add_ps(b1, b3));
                _mm256_storeu_ps(v + j + 3 * h, _mm256_sub_ps(b1, b3));
            }
    if (h < n)                                      // an odd stage count leaves the last one
        for (size_t j = 0; j < h; j += 8) {
            const __m256 a = _mm256_loadu_ps(v + j), c = _mm256_loadu_ps(v + j + h);
            _mm256_storeu_ps(v + j, _mm256_add_ps(a, c));
            _mm256_storeu_ps(v + j + h, _mm256_sub_ps(a, c));
        }
}

}  // namespace

namespace {

/// hadamard_tile_x's body over any source: load(row, b0, v) writes the `block` floats of token
/// row `row` starting at column b0 into v, before the transform.
///
/// hadamard_rows into a copy and then tile_x walked x five times (copy read and write, the
/// transform's read and write, the tile's read) and the out buffer once. Here one task owns one
/// 8-token sub-group of a tile_x token group and one transform block: it loads its eight runs
/// into a private buffer, transforms them (fwht_run: hadamard_rows' butterflies, same stages,
/// same operands, so the same floats), scales them by the same exact power of two, and writes
/// each 8 x 8 MAC sub-tile as one register transpose and one 128-byte store. The source is
/// read once and out written once. The first form converted one scalar at a time out of a
/// 32-row buffer read at a 4 KB stride, ran the transform's first three stages as scalar loops
/// of one, two and four, and split K 5120 into only 40 tasks for 24 threads.
/// The GEMM's bfp16ebs8 activation block vector (OPEN-GEMM-T2, GQP_XBFP) from eight token rows of
/// eight k each: every row is one block. Each value first takes the bf16 rounding the bf16 tiles
/// give it (f32_to_bf16), then the conversion the bfp16 GEMM's core applies to its bf16 operand
/// (to_v64bfp16ebs8 under conv_even), as read byte for byte off the hardware by
/// open_kernels/designs/bfp_cvt: E is the block's largest biased exponent; a value's int8 mantissa
/// is its 8-bit significand shifted right by E - e + 1, rounded to nearest even and signed; and
/// if any mantissa of the block falls outside [-128, 127] the block takes E + 1 and every value is
/// rounded again. Out: per block, the exponent byte then the eight mantissas (72 bytes).
inline __m256i bfp16_mant(__m256i u, __m256i e, __m256i E) {
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i normal = _mm256_cmpgt_epi32(e, _mm256_setzero_si256());
    const __m256i sig =
        _mm256_and_si256(normal, _mm256_or_si256(_mm256_and_si256(u, _mm256_set1_epi32(0x7F)), _mm256_set1_epi32(0x80)));
    const __m256i sh = _mm256_min_epi32(_mm256_add_epi32(_mm256_sub_epi32(E, e), one), _mm256_set1_epi32(9));
    __m256i q = _mm256_srlv_epi32(sig, sh);
    const __m256i rem = _mm256_and_si256(sig, _mm256_sub_epi32(_mm256_sllv_epi32(one, sh), one));
    const __m256i half = _mm256_sllv_epi32(one, _mm256_sub_epi32(sh, one));
    const __m256i up = _mm256_or_si256(_mm256_cmpgt_epi32(rem, half),
                                       _mm256_and_si256(_mm256_cmpeq_epi32(rem, half),
                                                        _mm256_cmpeq_epi32(_mm256_and_si256(q, one), one)));
    q = _mm256_sub_epi32(q, up);
    const __m256i neg = _mm256_cmpgt_epi32(_mm256_and_si256(u, _mm256_set1_epi32(0x8000)), _mm256_setzero_si256());
    return _mm256_sub_epi32(_mm256_xor_si256(q, neg), neg);             // -q where the sign bit is set
}

inline void bfp16_v64(const __m256 (&r)[8], uint8_t* o) {
    const __m256i one = _mm256_set1_epi32(1), ff = _mm256_set1_epi32(0xFF);
    for (int t = 0; t < 8; ++t) {
        __m256i u = _mm256_castps_si256(r[t]);
        u = _mm256_srli_epi32(
            _mm256_add_epi32(_mm256_add_epi32(u, _mm256_set1_epi32(0x7FFF)), _mm256_and_si256(_mm256_srli_epi32(u, 16), one)),
            16);                                                           // bf16 bits, as f32_to_bf16
        const __m256i e = _mm256_and_si256(_mm256_srli_epi32(u, 7), ff);
        __m256i E = _mm256_max_epi32(e, _mm256_permute2x128_si256(e, e, 1));
        E = _mm256_max_epi32(E, _mm256_shuffle_epi32(E, 0x4E));
        E = _mm256_max_epi32(E, _mm256_shuffle_epi32(E, 0xB1));          // the block max, every lane
        __m256i q = bfp16_mant(u, e, E);
        const __m256i out = _mm256_or_si256(_mm256_cmpgt_epi32(q, _mm256_set1_epi32(127)),
                                            _mm256_cmpgt_epi32(_mm256_set1_epi32(-128), q));
        if (!_mm256_testz_si256(out, out)) {                               // rare: a value rounded out of int8
            E = _mm256_add_epi32(E, one);
            q = bfp16_mant(u, e, E);
        }
        const __m256i p16 = _mm256_packs_epi32(q, q);
        const __m256i p8 = _mm256_packs_epi16(p16, p16);
        const uint64_t mant = static_cast<uint32_t>(_mm_cvtsi128_si32(_mm256_castsi256_si128(p8))) |
                              (static_cast<uint64_t>(static_cast<uint32_t>(_mm_cvtsi128_si32(_mm256_extracti128_si256(p8, 1)))) << 32);
        o[9 * t] = static_cast<uint8_t>(_mm_cvtsi128_si32(_mm256_castsi256_si128(E)));
        std::memcpy(o + 9 * t + 1, &mant, 8);
    }
}

template <bool BFP, class Load>
void hadamard_tile(size_t T, size_t K, size_t block, void* out_v, size_t tk, const char* who, const Load& load) {
    constexpr size_t MAC = 8, TN = 32;
    const size_t TK = tk;
    if ((TK != 64 && TK != 128) || K % TK || T % TN || block % TK || K % block || (block & (block - 1)))
        throw std::runtime_error(std::string("open_qwen36: ") + who + ": K, T or the block does not tile by (" +
                                 std::to_string(TK) + ", 32)");
    if (BFP && TK != 128)
        throw std::runtime_error(std::string("open_qwen36: ") + who + ": bfp16 activation tiles are 128-k tiles");
    const size_t NB = T / TN, KBLK = K / block, SUB = TN / MAC;
    const size_t LD = block + 16;                   // rows a 4 KB multiple apart would share L1 sets
    uint16_t* out = static_cast<uint16_t*>(out_v);
    uint8_t* outb = static_cast<uint8_t*>(out_v);
    const size_t TB = TK * TN * 9 / 8;              // one bfp16 tile's bytes
    const __m256 scale = _mm256_set1_ps(1.0f / std::sqrt(static_cast<float>(block)));
#pragma omp parallel
    {
        std::vector<float> buf(MAC * LD);
#pragma omp for schedule(dynamic, 2)   // the cores are not all equally fast (4 Zen 5 + 8 Zen 5c here)
        for (long long task = 0; task < static_cast<long long>(NB * SUB * KBLK); ++task) {
            const size_t nb = static_cast<size_t>(task) / (SUB * KBLK), rem = static_cast<size_t>(task) % (SUB * KBLK);
            const size_t ti = rem / KBLK, b0 = (rem % KBLK) * block;
            for (size_t r = 0; r < MAC; ++r) {
                float* v = buf.data() + r * LD;
                load(nb * TN + ti * MAC + r, b0, v);
                fwht_run(v, block);
            }
            const float* b = buf.data();
            if constexpr (BFP) {
                // gemm_bfp_mm.cc's B tile: [token block pair ti / 2][k block si][ti % 2] block vectors,
                // each eight token rows of eight k -- the loaded rows themselves, no transpose
                for (size_t kl = 0; kl < block / TK; ++kl) {
                    uint8_t* w = outb + ((b0 / TK + kl) * NB + nb) * TB;
                    for (size_t si = 0; si < TK / MAC; ++si) {
                        const size_t c = kl * TK + si * MAC;
                        const __m256 rr[8] = {_mm256_mul_ps(_mm256_loadu_ps(b + 0 * LD + c), scale),
                                              _mm256_mul_ps(_mm256_loadu_ps(b + 1 * LD + c), scale),
                                              _mm256_mul_ps(_mm256_loadu_ps(b + 2 * LD + c), scale),
                                              _mm256_mul_ps(_mm256_loadu_ps(b + 3 * LD + c), scale),
                                              _mm256_mul_ps(_mm256_loadu_ps(b + 4 * LD + c), scale),
                                              _mm256_mul_ps(_mm256_loadu_ps(b + 5 * LD + c), scale),
                                              _mm256_mul_ps(_mm256_loadu_ps(b + 6 * LD + c), scale),
                                              _mm256_mul_ps(_mm256_loadu_ps(b + 7 * LD + c), scale)};
                        bfp16_v64(rr, w + ((ti / 2) * 2 * (TK / MAC) + 2 * si + (ti % 2)) * 72);
                    }
                }
                continue;
            }
            for (size_t kl = 0; kl < block / TK; ++kl) {
                uint16_t* w = out + ((b0 / TK + kl) * NB + nb) * TK * TN + ti * MAC * MAC;
                for (size_t si = 0; si < TK / MAC; ++si) {
                    const size_t c = kl * TK + si * MAC;
                    __m256 r0 = _mm256_mul_ps(_mm256_loadu_ps(b + 0 * LD + c), scale);
                    __m256 r1 = _mm256_mul_ps(_mm256_loadu_ps(b + 1 * LD + c), scale);
                    __m256 r2 = _mm256_mul_ps(_mm256_loadu_ps(b + 2 * LD + c), scale);
                    __m256 r3 = _mm256_mul_ps(_mm256_loadu_ps(b + 3 * LD + c), scale);
                    __m256 r4 = _mm256_mul_ps(_mm256_loadu_ps(b + 4 * LD + c), scale);
                    __m256 r5 = _mm256_mul_ps(_mm256_loadu_ps(b + 5 * LD + c), scale);
                    __m256 r6 = _mm256_mul_ps(_mm256_loadu_ps(b + 6 * LD + c), scale);
                    __m256 r7 = _mm256_mul_ps(_mm256_loadu_ps(b + 7 * LD + c), scale);
                    tr8(r0, r1, r2, r3, r4, r5, r6, r7);     // row s now holds tokens t = 0..7 of k c + s
                    __m128i* o = reinterpret_cast<__m128i*>(w + si * SUB * MAC * MAC);
                    _mm_storeu_si128(o + 0, bf16x8(r0));
                    _mm_storeu_si128(o + 1, bf16x8(r1));
                    _mm_storeu_si128(o + 2, bf16x8(r2));
                    _mm_storeu_si128(o + 3, bf16x8(r3));
                    _mm_storeu_si128(o + 4, bf16x8(r4));
                    _mm_storeu_si128(o + 5, bf16x8(r5));
                    _mm_storeu_si128(o + 6, bf16x8(r6));
                    _mm_storeu_si128(o + 7, bf16x8(r7));
                }
            }
        }
    }
}

}  // namespace

namespace {

struct LoadX {
    const float* x;
    size_t K, block;
    const float* signs;
    void operator()(size_t row, size_t b0, float* v) const {
        const float* src = x + row * K + b0;
        if (signs) {
            const float* sg = signs + b0;
            for (size_t j = 0; j < block; j += 8)
                _mm256_storeu_ps(v + j, _mm256_mul_ps(_mm256_loadu_ps(src + j), _mm256_loadu_ps(sg + j)));
        } else {
            std::memcpy(v, src, block * 4);
        }
    }
};

struct LoadSwiglu {
    const float* ug;
    size_t ff, ld, block;
    void operator()(size_t row, size_t b0, float* v) const {
        const float* u = ug + row * ld + b0;
        const float* g = u + ff;
        // the exact expression the FFN's own loop evaluates, one element at a time: that loop
        // is not vectorised (MSVC sees possible aliasing), so it calls the scalar expf, and a
        // vectorised form here would call the vector library's exp and round differently
#pragma loop(no_vector)
        for (size_t j = 0; j < block; ++j) v[j] = g[j] / (1.f + std::exp(-g[j])) * u[j];
    }
};

}  // namespace

void hadamard_tile_x(const float* x, size_t T, size_t K, size_t block, const float* signs, uint16_t* out,
                     size_t tk) {
    hadamard_tile<false>(T, K, block, out, tk, "hadamard_tile_x", LoadX{x, K, block, signs});
}

void hadamard_tile_swiglu(const float* ug, size_t T, size_t ff, size_t ld, size_t block, uint16_t* out, size_t tk) {
    hadamard_tile<false>(T, ff, block, out, tk, "hadamard_tile_swiglu", LoadSwiglu{ug, ff, ld, block});
}

void hadamard_tile_x_bfp(const float* x, size_t T, size_t K, size_t block, const float* signs, uint8_t* out) {
    hadamard_tile<true>(T, K, block, out, 128, "hadamard_tile_x_bfp", LoadX{x, K, block, signs});
}

void hadamard_tile_swiglu_bfp(const float* ug, size_t T, size_t ff, size_t ld, size_t block, uint8_t* out) {
    hadamard_tile<true>(T, ff, block, out, 128, "hadamard_tile_swiglu_bfp", LoadSwiglu{ug, ff, ld, block});
}
void tile_x(const float* x, size_t T, size_t K, uint16_t* out, size_t tk) {
    // [T,K] fp32 -> bf16, pre-tiled [K,T] in "k,n" order: K_TILE tk (64, or 128 for a GQP_KT=128
    // GEMM) x tile_n 32 tiles, each tile in (8 x 8) MAC sub-tiles -- the layout
    // gemm_q4_prefill.py streams its activation in
    constexpr size_t MAC = 8, TN = 32;
    const size_t TK = tk;
    if ((TK != 64 && TK != 128) || K % TK || T % TN)
        throw std::runtime_error("open_qwen36: tile_x: K or T does not tile by (" + std::to_string(TK) + ", 32)");
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

namespace {

template <class T>
T* grow(std::vector<T>& v, size_t n) {
    if (v.size() < n) v.resize(n);
    return v.data();
}

/// alpha / beta for one token over rows [i0, ie) and lanes [h, h + 8 NV): a[h..] += x[i] * Wa[i][h..]
/// and the same for b, every lane's sum over i in order, held in registers across the rows.
template <int NV>
inline void ab_rows(const float* x, size_t i0, size_t ie, const float* Wa, const float* Wb, size_t L, size_t h,
                    float* a, float* b) {
    __m256 ra[NV], rb[NV];
    for (int v = 0; v < NV; ++v) {
        ra[v] = _mm256_loadu_ps(a + h + 8 * v);
        rb[v] = _mm256_loadu_ps(b + h + 8 * v);
    }
    for (size_t i = i0; i < ie; ++i) {
        const __m256 xi = _mm256_set1_ps(x[i]);
        const float* wa = Wa + i * L + h;
        const float* wb = Wb + i * L + h;
        for (int v = 0; v < NV; ++v) {
            ra[v] = _mm256_add_ps(ra[v], _mm256_mul_ps(xi, _mm256_loadu_ps(wa + 8 * v)));
            rb[v] = _mm256_add_ps(rb[v], _mm256_mul_ps(xi, _mm256_loadu_ps(wb + 8 * v)));
        }
    }
    for (int v = 0; v < NV; ++v) {
        _mm256_storeu_ps(a + h + 8 * v, ra[v]);
        _mm256_storeu_ps(b + h + 8 * v, rb[v]);
    }
}

/// The delta rule's first pass for one head, columns [0, dim): tv[j] = sum over i in order of
/// k[i] * (S[i][j] * dc), from zero, and S[i][j] * dc left in S for the update pass to add to.
/// Sixteen columns at a time in registers, then eight, then one; every column's sum is the same
/// sequence of multiplies and adds whichever width it is in.
void rule_first_pass(float* Sh, size_t dim, const float* k, float dcs, float* tv) {
    const __m256 dc = _mm256_set1_ps(dcs);
    size_t jc = 0;
    for (; jc + 16 <= dim; jc += 16) {
        __m256 n0 = _mm256_setzero_ps(), n1 = _mm256_setzero_ps();
        for (size_t i = 0; i < dim; ++i) {
            float* Si = Sh + i * dim + jc;
            const __m256 ki = _mm256_set1_ps(k[i]);
            const __m256 p0 = _mm256_mul_ps(_mm256_loadu_ps(Si), dc), p1 = _mm256_mul_ps(_mm256_loadu_ps(Si + 8), dc);
            _mm256_storeu_ps(Si, p0);
            _mm256_storeu_ps(Si + 8, p1);
            n0 = _mm256_add_ps(n0, _mm256_mul_ps(ki, p0));
            n1 = _mm256_add_ps(n1, _mm256_mul_ps(ki, p1));
        }
        _mm256_storeu_ps(tv + jc, n0);
        _mm256_storeu_ps(tv + jc + 8, n1);
    }
    for (; jc + 8 <= dim; jc += 8) {
        __m256 n0 = _mm256_setzero_ps();
        for (size_t i = 0; i < dim; ++i) {
            float* Si = Sh + i * dim + jc;
            const __m256 p0 = _mm256_mul_ps(_mm256_loadu_ps(Si), dc);
            _mm256_storeu_ps(Si, p0);
            n0 = _mm256_add_ps(n0, _mm256_mul_ps(_mm256_set1_ps(k[i]), p0));
        }
        _mm256_storeu_ps(tv + jc, n0);
    }
    for (; jc < dim; ++jc) {
        float n = 0.f;
        for (size_t i = 0; i < dim; ++i) {
            const float p = Sh[i * dim + jc] * dcs;
            Sh[i * dim + jc] = p;
            n += k[i] * p;
        }
        tv[jc] = n;
    }
}

/// The update pass for one head and token, on S holding S_prev * dc (the decay already applied
/// by the pass before): s = that + k[i] * delta[j], o[j] += s * q[i]. Unless LAST, the next
/// token's first pass rides along -- tvn[j] += kn[i] * (s * dcn) -- and S keeps s * dcn for the
/// next update; the last token stores s itself, the state the block hands on. s * dcn is the
/// very product the next token's update would form as S[i][j] * dc, so it is formed once.
template <bool LAST>
void rule_update_pass(float* Sh, size_t dim, const float* k, const float* q, const float* kn, float dcns,
                      const float* delta, float* o, float* tvn) {
    const __m256 dcn = _mm256_set1_ps(dcns);
    size_t jc = 0;
    for (; jc + 16 <= dim; jc += 16) {
        const __m256 d0 = _mm256_loadu_ps(delta + jc), d1 = _mm256_loadu_ps(delta + jc + 8);
        __m256 o0 = _mm256_setzero_ps(), o1 = _mm256_setzero_ps();
        __m256 n0 = _mm256_setzero_ps(), n1 = _mm256_setzero_ps();
        for (size_t i = 0; i < dim; ++i) {
            float* Si = Sh + i * dim + jc;
            const __m256 ki = _mm256_set1_ps(k[i]), qi = _mm256_set1_ps(q[i]);
            const __m256 s0 = _mm256_add_ps(_mm256_loadu_ps(Si), _mm256_mul_ps(ki, d0));
            const __m256 s1 = _mm256_add_ps(_mm256_loadu_ps(Si + 8), _mm256_mul_ps(ki, d1));
            o0 = _mm256_add_ps(o0, _mm256_mul_ps(s0, qi));
            o1 = _mm256_add_ps(o1, _mm256_mul_ps(s1, qi));
            if constexpr (LAST) {
                _mm256_storeu_ps(Si, s0);
                _mm256_storeu_ps(Si + 8, s1);
            } else {
                const __m256 kni = _mm256_set1_ps(kn[i]);
                const __m256 p0 = _mm256_mul_ps(s0, dcn), p1 = _mm256_mul_ps(s1, dcn);
                _mm256_storeu_ps(Si, p0);
                _mm256_storeu_ps(Si + 8, p1);
                n0 = _mm256_add_ps(n0, _mm256_mul_ps(kni, p0));
                n1 = _mm256_add_ps(n1, _mm256_mul_ps(kni, p1));
            }
        }
        _mm256_storeu_ps(o + jc, o0);
        _mm256_storeu_ps(o + jc + 8, o1);
        if constexpr (!LAST) {
            _mm256_storeu_ps(tvn + jc, n0);
            _mm256_storeu_ps(tvn + jc + 8, n1);
        }
    }
    for (; jc + 8 <= dim; jc += 8) {
        const __m256 d0 = _mm256_loadu_ps(delta + jc);
        __m256 o0 = _mm256_setzero_ps(), n0 = _mm256_setzero_ps();
        for (size_t i = 0; i < dim; ++i) {
            float* Si = Sh + i * dim + jc;
            const __m256 s0 = _mm256_add_ps(_mm256_loadu_ps(Si), _mm256_mul_ps(_mm256_set1_ps(k[i]), d0));
            o0 = _mm256_add_ps(o0, _mm256_mul_ps(s0, _mm256_set1_ps(q[i])));
            if constexpr (LAST) {
                _mm256_storeu_ps(Si, s0);
            } else {
                const __m256 p0 = _mm256_mul_ps(s0, dcn);
                _mm256_storeu_ps(Si, p0);
                n0 = _mm256_add_ps(n0, _mm256_mul_ps(_mm256_set1_ps(kn[i]), p0));
            }
        }
        _mm256_storeu_ps(o + jc, o0);
        if constexpr (!LAST) _mm256_storeu_ps(tvn + jc, n0);
    }
    for (; jc < dim; ++jc) {
        float oo = 0.f, nn = 0.f;
        for (size_t i = 0; i < dim; ++i) {
            const float s = Sh[i * dim + jc] + k[i] * delta[jc];
            oo += s * q[i];
            if constexpr (LAST) {
                Sh[i * dim + jc] = s;
            } else {
                const float p = s * dcns;
                Sh[i * dim + jc] = p;
                nn += kn[i] * p;
            }
        }
        o[jc] = oo;
        if constexpr (!LAST) tvn[jc] = nn;
    }
}

}  // namespace

void deltanet_block(const DeltaGeom& g, const float* qkv, const float* z, const float* xn, const float* convw,
                    const float* Wa, const float* Wb, const float* A, const float* dtb, const float* nw,
                    uint16_t* conv_state, float* S, float* og, double* phase_ms) {
    const auto tp0 = std::chrono::steady_clock::now();
    const size_t dim = g.head_dim, key_w = g.key_heads * dim, vw = g.value_heads * dim, nch = 2 * key_w + vw;
    if (g.value_heads % g.key_heads || g.t_real > g.T || g.s_rows < dim || g.lanes < g.value_heads)
        throw std::runtime_error("open_qwen36: deltanet_block: inconsistent geometry");
    const size_t grp = g.value_heads / g.key_heads, R = g.t_real;
    const size_t qkv_ld = g.qkv_ld ? g.qkv_ld : nch, z_ld = g.z_ld ? g.z_ld : vw;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(dim));
    // og past t_real is zero; every row before it is written in full by phase 2
    std::fill(og + R * vw, og + g.T * vw, 0.f);

    // Scratch that lives across calls. Fresh vectors here were ~10 MB a call, zero-filled by one
    // thread and faulted in page by page from the OS (an allocation that size is its own
    // VirtualAlloc on Windows).
    struct Scratch {
        std::vector<float> carry, Q, Kk, V, decay, beta;
    };
    static thread_local Scratch sc;
    float* carry = grow(sc.carry, (g.taps - 1) * nch);
    float* Q = grow(sc.Q, R * key_w);
    float* Kk = grow(sc.Kk, R * key_w);
    float* V = grow(sc.V, R * vw);
    float* decay = grow(sc.decay, R * g.value_heads);
    float* beta = grow(sc.beta, R * g.value_heads);

    // ---- phase 1, per token: the conv (state rows carried), q / k normalised, alpha / beta
    // The conv is a fixed taps-wide window over the bf16-rounded rows, not a recurrence, so
    // every token's work is independent: row r of token t's window is qkv row t - pre + r,
    // and the carried state stands in where that runs before the block.
    const size_t pre = g.taps - 1;
    for (size_t r = 0; r < pre; ++r)
        for (size_t j = 0; j < nch; ++j) carry[r * nch + j] = bf16_to_f32(conv_state[r * nch + j]);
#pragma omp parallel
    {
        static thread_local std::vector<float> cbuf;
        float* c = grow(cbuf, nch);
#pragma omp for schedule(dynamic, 4)   // the cores are not all equally fast
        for (long long tt = 0; tt < static_cast<long long>(R); ++tt) {
            const size_t t = static_cast<size_t>(tt);
            // Eight channels at a time with the taps' partial sums held in a register: c starts
            // at zero and adds cw * v for r = 0, 1, ... exactly as the scalar loop did (separate
            // multiply and add, as MSVC compiles the scalar form: it does not contract to FMA),
            // and the bf16 round trip is the same integer arithmetic, so every bit is the same.
            size_t j = 0;
            for (; j + 8 <= nch; j += 8) {
                __m256 acc = _mm256_setzero_ps();
                for (size_t r = 0; r < g.taps; ++r) {
                    const long long s = tt - static_cast<long long>(pre) + static_cast<long long>(r);
                    const __m256 v = s < 0 ? _mm256_loadu_ps(carry + (static_cast<size_t>(s) + pre) * nch + j)
                                           : bf16r8(_mm256_loadu_ps(qkv + static_cast<size_t>(s) * qkv_ld + j));
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(convw + r * nch + j), v));
                }
                _mm256_storeu_ps(c + j, acc);
            }
            for (; j < nch; ++j) {
                float acc = 0.f;
                for (size_t r = 0; r < g.taps; ++r) {
                    const long long s = tt - static_cast<long long>(pre) + static_cast<long long>(r);
                    const float v = s < 0 ? carry[(static_cast<size_t>(s) + pre) * nch + j]
                                          : bf16r(qkv[static_cast<size_t>(s) * qkv_ld + j]);
                    acc += convw[r * nch + j] * v;
                }
                c[j] = acc;
            }
            for (j = 0; j < nch; ++j) c[j] = silu(c[j]);
            for (size_t hh = 0; hh < g.key_heads; ++hh)
                for (int which = 0; which < 2; ++which) {
                    const float* src = c + which * key_w + hh * dim;
                    float* dst = (which ? Kk : Q) + t * key_w + hh * dim;
                    double ss = 0;
                    for (size_t i = 0; i < dim; ++i) ss += static_cast<double>(src[i]) * src[i];
                    const float r = static_cast<float>(1.0 / std::sqrt(ss + 1e-6));   // the L2 norm dn_glue applies
                    for (size_t i = 0; i < dim; ++i) dst[i] = src[i] * r;
                }
            std::memcpy(V + t * vw, c + 2 * key_w, vw * 4);
        }
    }
    // alpha / beta: [R, hid] x [hid, lanes] twice, a few tokens per pass over Wa / Wb. Token by
    // token, every token streamed both matrices (2 x hid x lanes floats, ~2 MB at the 27B) out of
    // the cache again. Each token's sums still run over i in order from zero, the same multiply
    // and add, so the result is bit-identical. The lane loop is written out eight lanes at a time:
    // MSVC left the scalar form unvectorised.
    {
        // one block per thread where that fits (256 tokens over 24 threads: 11 each), at most 16
        const size_t nthr = static_cast<size_t>(std::max(1, omp_get_max_threads()));
        const size_t TB = std::min<size_t>(16, std::max<size_t>(1, (R + nthr - 1) / nthr));
        const long long nblk = static_cast<long long>((R + TB - 1) / TB);
        const size_t L = g.lanes, L8 = L / 8 * 8;
#pragma omp parallel
        {
            std::vector<float> al(TB * L), be(TB * L);
#pragma omp for
            for (long long bb = 0; bb < nblk; ++bb) {
                const size_t t0 = static_cast<size_t>(bb) * TB, nt = std::min(TB, R - t0);
                std::fill(al.begin(), al.end(), 0.f);
                std::fill(be.begin(), be.end(), 0.f);
                // 64 rows of Wa / Wb at a time (24 KB at 48 lanes, L1-resident) against every
                // token of the block, each token's sums held in registers across the rows and
                // stored once per row block: the same adds in the same order, a load and a store
                // per 64 multiply-adds instead of per one
                constexpr size_t IB = 64;
                for (size_t i0 = 0; i0 < g.hid; i0 += IB) {
                    const size_t ie = std::min(g.hid, i0 + IB);
                    for (size_t u = 0; u < nt; ++u) {
                        const float* x = xn + (t0 + u) * g.hid;
                        float* __restrict a = al.data() + u * L;
                        float* __restrict b = be.data() + u * L;
                        size_t h = 0;
                        for (; h + 32 <= L8; h += 32) ab_rows<4>(x, i0, ie, Wa, Wb, L, h, a, b);
                        for (; h + 16 <= L8; h += 16) ab_rows<2>(x, i0, ie, Wa, Wb, L, h, a, b);
                        for (; h < L8; h += 8) ab_rows<1>(x, i0, ie, Wa, Wb, L, h, a, b);
                        for (; h < L; ++h)
                            for (size_t i = i0; i < ie; ++i) {
                                a[h] += x[i] * Wa[i * L + h];
                                b[h] += x[i] * Wb[i * L + h];
                            }
                    }
                }
                for (size_t u = 0; u < nt; ++u)
                    for (size_t h = 0; h < g.value_heads; ++h) {
                        decay[(t0 + u) * g.value_heads + h] = std::exp(A[h] * softplus(al[u * L + h] + dtb[h]));
                        beta[(t0 + u) * g.value_heads + h] = sigmoid(be[u * L + h]);
                    }
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
            const float* row = qkv + static_cast<size_t>(s) * qkv_ld;
            for (size_t j = 0; j < nch; ++j) conv_state[r * nch + j] = f32_to_bf16(row[j]);
        }
    }

    // ---- phase 2, per head over every token: the gated delta rule on S (in place), the gated norm
    //
    // Per token t the rule is two passes over S: tv = sum_i k_t[i] * (S[i] * dc_t), then
    // S[i] = S[i] * dc_t + k_t[i] * delta and o += S[i] * q_t[i]. Token t + 1's first pass reads
    // exactly the S[i] that token t's second pass has just written, so it rides along: one
    // pass over S per token, with tv for t + 1 summed over i in the same order from zero, the
    // same multiplies and adds, so every float is the one the two-pass form produced. The
    // decayed S[i] * dc_{t+1} the ride-along forms is kept in S for token t + 1's update, which
    // needed that same product (four multiplies an element instead of five; the last token
    // stores the undecayed state). Sixteen columns at a time keep delta, o and the next tv in
    // registers across the i loop; S is read and written once per token instead of read twice
    // and written once, and tv / o no longer round-trip through memory on every row.
    const auto tp1 = std::chrono::steady_clock::now();
    if (R > 0) {
#pragma omp parallel
        {
            static thread_local std::vector<float> hbuf;
            float* tv = grow(hbuf, 5 * dim);
            float* tvn = tv + dim;
            float* delta = tv + 2 * dim;
            float* o = tv + 3 * dim;
            float* on = tv + 4 * dim;
            // A static split of the heads: handing them out dynamically, as the per-token and tile
            // loops are (the cores are not all equally fast), measured 12 % slower here; splitting a
            // head's columns across threads -- independent up to the norm -- measured 3x slower (two
            // threads then share the cache line of S where a row does not start on 64 bytes).
#pragma omp for
            for (long long h = 0; h < static_cast<long long>(g.value_heads); ++h) {
                float* Sh = S + h * g.s_rows * dim;
                const size_t kh = (static_cast<size_t>(h) / grp) * dim;
                rule_first_pass(Sh, dim, Kk + kh, decay[h], tv);
                for (size_t t = 0; t < R; ++t) {
                    const float* kk = Kk + t * key_w + kh;
                    const float* qq = Q + t * key_w + kh;
                    const float* v = V + t * vw + h * dim;
                    const float bt = beta[t * g.value_heads + h];
                    for (size_t j = 0; j < dim; ++j) delta[j] = bt * (v[j] - tv[j]);
                    if (t + 1 < R)
                        rule_update_pass<false>(Sh, dim, kk, qq, Kk + (t + 1) * key_w + kh,
                                                decay[(t + 1) * g.value_heads + h], delta, o, tvn);
                    else
                        rule_update_pass<true>(Sh, dim, kk, qq, nullptr, 0.f, delta, o, nullptr);
                    for (size_t j = 0; j < dim; ++j) o[j] *= inv_sqrt;
                    rms_vec(o, dim, nw, g.eps, on);
                    float* out = og + t * vw + h * dim;
                    const float* zz = z + t * z_ld + h * dim;
                    for (size_t j = 0; j < dim; ++j) out[j] = on[j] * silu(zz[j]);
                    std::swap(tv, tvn);
                }
            }
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
    const size_t q_ld = g.q_ld ? g.q_ld : qw, k_ld = g.k_ld ? g.k_ld : kvw, v_ld = g.v_ld ? g.v_ld : kvw;
    const size_t g_ld = g.g_ld ? g.g_ld : qw;
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
            rms_vec(q + t * q_ld + h * g.hd, g.hd, qn, g.eps, dst);
            rope(dst, half, inv_freq, static_cast<double>(p));
        }
        std::vector<float> kh(kvw);
        for (size_t h = 0; h < g.kvh; ++h) {
            rms_vec(k + t * k_ld + h * g.hd, g.hd, kn, g.eps, kh.data() + h * g.hd);
            rope(kh.data() + h * g.hd, half, inv_freq, static_cast<double>(p));
        }
        for (size_t j = 0; j < kvw; ++j) {
            const uint16_t kb = f32_to_bf16(kh[j]), vb = f32_to_bf16(v[t * v_ld + j]);
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
        const float* gt = gate + t * g_ld + h * g.hd;
        for (size_t j = 0; j < g.hd; ++j) out[j] = o[j] * sigmoid(gt[j]);
    }
}

void attention_prep(const AttnGeom& g, const float* q, const float* k, const float* v, const float* qn,
                    const float* kn, const double* inv_freq, uint16_t* kv, size_t kv_row_elems, float* Q) {
    const size_t qw = g.nh * g.hd, kvw = g.kvh * g.hd, half = g.rot / 2, R = g.t_real;
    if (g.nh % g.kvh || g.t_real > g.T || g.rot > g.hd || kv_row_elems < 2 * kvw)
        throw std::runtime_error("open_qwen36: attention_prep: inconsistent geometry");
    const size_t q_ld = g.q_ld ? g.q_ld : qw, k_ld = g.k_ld ? g.k_ld : kvw, v_ld = g.v_ld ? g.v_ld : kvw;
    const float scale = 1.0f / std::sqrt(static_cast<float>(g.hd));
    std::fill(Q, Q + g.T * qw, 0.f);
#pragma omp parallel
    {
    std::vector<float> kh(kvw);                 // once a thread, not once a token
#pragma omp for
    for (long long tt = 0; tt < static_cast<long long>(R); ++tt) {
        const size_t t = static_cast<size_t>(tt), p = g.pos0 + t;
        for (size_t h = 0; h < g.nh; ++h) {
            float* dst = Q + t * qw + h * g.hd;
            rms_vec(q + t * q_ld + h * g.hd, g.hd, qn, g.eps, dst);
            rope(dst, half, inv_freq, static_cast<double>(p));
            for (size_t j = 0; j < g.hd; ++j) dst[j] *= scale;
        }
        for (size_t h = 0; h < g.kvh; ++h) {
            rms_vec(k + t * k_ld + h * g.hd, g.hd, kn, g.eps, kh.data() + h * g.hd);
            rope(kh.data() + h * g.hd, half, inv_freq, static_cast<double>(p));
        }
        for (size_t j = 0; j < kvw; ++j) {
            kv[p * kv_row_elems + j] = f32_to_bf16(kh[j]);
            kv[p * kv_row_elems + kvw + j] = f32_to_bf16(v[t * v_ld + j]);
        }
    }
    }
}

void attn_group_queries(const float* Q, size_t T, size_t nh, size_t kvh, size_t hd, size_t gh, uint16_t* qb) {
    const size_t grp = nh / kvh, qw = nh * hd;
#pragma omp parallel for
    for (long long r = 0; r < static_cast<long long>(grp * T); ++r) {
        const size_t hl = static_cast<size_t>(r) / T, t = static_cast<size_t>(r) % T;
        const float* src = Q + t * qw + (gh * grp + hl) * hd;
        uint16_t* dst = qb + static_cast<size_t>(r) * hd;
        size_t j = 0;
        for (; j + 8 <= hd; j += 8) _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + j), bf16x8(_mm256_loadu_ps(src + j)));
        for (; j < hd; ++j) dst[j] = f32_to_bf16(src[j]);
    }
}

void attn_group_out(const float* acc, const float* lsum, const float* gate, size_t T, size_t t_real, size_t nh,
                    size_t kvh, size_t hd, size_t gh, size_t g_ld, float* og) {
    // attention_npu's epilogue, the same expression per element; it was one thread over
    // grp * t_real * hd sigmoids (~1.6 M exps a layer-block at the 27B)
    const size_t grp = nh / kvh, qw = nh * hd, gl = g_ld ? g_ld : qw;
#pragma omp parallel for
    for (long long rr = 0; rr < static_cast<long long>(grp * t_real); ++rr) {
        const size_t hl = static_cast<size_t>(rr) / t_real, t = static_cast<size_t>(rr) % t_real;
        const size_t r = hl * T + t, h = gh * grp + hl;
        const float inv = 1.0f / lsum[r];
        const float* gt = gate + t * gl + h * hd;
        float* out = og + t * qw + h * hd;
        for (size_t j = 0; j < hd; ++j) out[j] = acc[r * hd + j] * inv / (1.0f + std::exp(-gt[j]));
    }
}

void add_into(float* acc, const float* c, size_t n) {
#pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(n); ++i) acc[i] += c[i];
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
