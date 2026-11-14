/// \file bfp16.hpp
/// \brief bfp16ebs8 operands of gemm_bfp16 (open_kernels/designs/gemm_bfp16, pack.py is the NumPy twin):
///        eight values along K share one exponent byte, each a signed 8-bit mantissa, nine bytes a block.
///
/// Derived from ggml-xdna's src/npu/bfp16_pack.{h,cpp} (Cyrus Attoun, MIT). Header-only so the pool packer
/// (pools.cpp, no AVX2 flag) and the activation packer (block_host.cpp) share one layout.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace open_qwen36 {

constexpr size_t BFP16_BLOCK_BYTES = 9;

/// The kernel's compiled tile: A is m rows x k, B is n rows x k, over rows x cols cores.
struct Bfp16Tiling {
    size_t m = 64, k = 64, n = 128, cols = 8, rows = 4;
};

/// Eight floats -> exponent byte (the largest float exponent) + eight signed mantissas
/// q = round-half-even(x * 2^(133 - e)) saturated to [-128, 127]; decoded q * 2^(e - 133).
inline void bfp16_encode_block(const float* x, uint8_t* out) {
    uint32_t u[8];
    std::memcpy(u, x, 32);
    uint32_t maxe = 0;
    for (int i = 0; i < 8; i++) maxe = std::max(maxe, (u[i] >> 23) & 0xffu);
    out[0] = static_cast<uint8_t>(maxe);
    const int sexp = 133 - static_cast<int>(maxe) + 127;     // biased exponent of the scale 2^(133 - maxe)
#ifdef __AVX2__
    if (sexp >= 1 && sexp <= 254) {
        const __m256 scale = _mm256_castsi256_ps(_mm256_set1_epi32(sexp << 23));
        __m256i q = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x), scale));      // nearest even
        q = _mm256_min_epi32(_mm256_max_epi32(q, _mm256_set1_epi32(-128)), _mm256_set1_epi32(127));
        const __m128i q16 = _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(out + 1), _mm_packs_epi16(q16, q16));
        return;
    }
#endif
    const int sh = 133 - static_cast<int>(maxe);     // also zeros, and exponents so small the scale is not a normal float
    for (int i = 0; i < 8; i++) {
        const float q = std::nearbyint(std::ldexp(x[i], sh));
        out[1 + i] = static_cast<uint8_t>(static_cast<int8_t>(std::min(127.0f, std::max(-128.0f, q))));
    }
}

namespace bfp16_detail {
/// inside a tile: row pairs (16 rows) by block column, then the pair's 16 rows
inline size_t in_tile(size_t rr, size_t cb, const Bfp16Tiling& t) {
    const size_t tkb = t.k / 8;
    return ((rr / 16) * tkb + cb % tkb) * 16 + rr % 16;
}
}  // namespace bfp16_detail

/// Byte offset of the block holding columns [8 cb, 8 cb + 8) of row r of an A operand (K columns): for each
/// block of m rows, for each chunk of k, one m x k tile.
inline size_t bfp16_a_offset(size_t r, size_t cb, size_t K, const Bfp16Tiling& t) {
    const size_t nk = K / t.k, tile = (r / t.m) * nk + cb / (t.k / 8);
    return (tile * t.m * (t.k / 8) + bfp16_detail::in_tile(r % t.m, cb, t)) * BFP16_BLOCK_BYTES;
}

/// ... of a B operand (N rows, K columns): for each array column c, for each of its n-row groups (rows
/// (c + cols * g) * n onward), for each chunk of k, one n x k tile.
inline size_t bfp16_b_offset(size_t r, size_t cb, size_t N, size_t K, const Bfp16Tiling& t) {
    const size_t per_col = N / t.n / t.cols, nk = K / t.k, g = r / t.n;
    const size_t tile = ((g % t.cols) * per_col + g / t.cols) * nk + cb / (t.k / 8);
    return (tile * t.n * (t.k / 8) + bfp16_detail::in_tile(r % t.n, cb, t)) * BFP16_BLOCK_BYTES;
}

/// pack.py glu_row's twin (up row f is 8 further): ff chunk kc in n-tile (kc % per_col) * cols + kc / per_col keeps a column's H slots one run
inline size_t bfp16_glu_row(size_t f, size_t ff, const Bfp16Tiling& t) {
    const size_t per_col = 2 * ff / t.n / t.cols, half = t.n / 2, kc = f / half;
    const size_t ct = (kc % per_col) * t.cols + kc / per_col;
    return ct * t.n + 16 * ((f % half) / 8) + f % 8;
}

/// 9 x 911 words: a core tile's 8192 padded so every H slot (kc * rows + row) starts on a whole bfp16 block
constexpr size_t BFP16_H_SLOT_WORDS = 8199;
inline size_t bfp16_h_bytes(size_t T, size_t ff, const Bfp16Tiling& t) {
    return (T / t.m) * (2 * ff / t.n) * BFP16_H_SLOT_WORDS * 4;
}

}  // namespace open_qwen36
