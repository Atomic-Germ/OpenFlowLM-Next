/// \file tile-x-bench.cpp
/// \brief Where `host::tile_x`'s prefill time goes, in three separate places.
///
/// `tile_x` turns one GEMM's x [T, K] fp32 into the kernel's tiled bf16 activation layout.
/// PR #1's per-layer profile attributes ~3.0 s of a 35.0 s prefill block to it -- 49.5% of
/// all host time -- while the output transpose moves more elements in 0.96 s. Two candidate
/// explanations, and they call for opposite fixes:
///
///   1. the loop is scalar, one `f32_to_bf16` per element, source strided by K; or
///   2. the DESTINATION is the cost: the engine writes into `xb.map<uint16_t*>()`, a mapped
///      device buffer, and byte-granular writes to a mapping like that are far slower than
///      writes to ordinary memory.
///
/// So there are three columns: the shipped loop into host memory, the same loop into a real
/// XRT buffer mapping, and a vectorised loop into host memory. Comparing 1 against 2 answers
/// (2); comparing 1 against 3 answers (1). If (2) dominates, no amount of loop tuning helps
/// and the fix is in how the buffer is filled.
///
///   utilities/prefill-block/build-tile-x-bench.sh && <build>/tile_x_bench [reps]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <immintrin.h>
#include <omp.h>

#include "open_qwen36/block_host.hpp"
#include "open_qwen36/q4nx_file.hpp"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/experimental/xrt_ext.h"

namespace {

using open_qwen36::f32_to_bf16;

/// One 8x8 sub-tile of [T, K] fp32 into 64 contiguous bf16, vectorised.
///
/// The scalar walk reads x[(r0 + t) * K + c0 + s] for t, s in [0, 8), so it holds the
/// destination's inner order (s-major, then t) and reads the source strided by K. This loads
/// eight whole 32-byte source rows, transposes them in registers -- `t8x8`'s network, which
/// the output transpose already uses -- and then converts each of the eight resulting float
/// vectors to eight bf16 and stores 16 contiguous bytes. Same rounding as `f32_to_bf16`
/// (`u + 0x7FFF + ((u >> 16) & 1) >> 16`), so the output is bit-identical.
inline void tile8x8(const float* src, size_t sstride, uint16_t* dst) {
    const __m256 r0 = _mm256_loadu_ps(src + 0 * sstride), r1 = _mm256_loadu_ps(src + 1 * sstride);
    const __m256 r2 = _mm256_loadu_ps(src + 2 * sstride), r3 = _mm256_loadu_ps(src + 3 * sstride);
    const __m256 r4 = _mm256_loadu_ps(src + 4 * sstride), r5 = _mm256_loadu_ps(src + 5 * sstride);
    const __m256 r6 = _mm256_loadu_ps(src + 6 * sstride), r7 = _mm256_loadu_ps(src + 7 * sstride);
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
    const __m256 v[8] = {
        _mm256_permute2f128_ps(s0, s4, 0x20), _mm256_permute2f128_ps(s1, s5, 0x20),
        _mm256_permute2f128_ps(s2, s6, 0x20), _mm256_permute2f128_ps(s3, s7, 0x20),
        _mm256_permute2f128_ps(s0, s4, 0x31), _mm256_permute2f128_ps(s1, s5, 0x31),
        _mm256_permute2f128_ps(s2, s6, 0x31), _mm256_permute2f128_ps(s3, s7, 0x31),
    };
    for (int i = 0; i < 8; ++i) {
        const __m256i u = _mm256_castps_si256(v[i]);
        const __m256i odd = _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1));
        const __m256i r = _mm256_add_epi32(u, _mm256_add_epi32(odd, _mm256_set1_epi32(0x7FFF)));
        const __m256i h = _mm256_srli_epi32(r, 16);
        // 8 dwords -> 8 contiguous words: pack within lanes, then take the two low qwords
        const __m256i packed = _mm256_permute4x64_epi64(_mm256_packus_epi32(h, h), 0b1000);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i * 8), _mm256_castsi256_si128(packed));
    }
}

/// The candidate: the same tiling order as `host::tile_x`, one 8x8 sub-tile at a time.
void tile_x_avx2(const float* x, size_t T, size_t K, uint16_t* out) {
    constexpr size_t TK = 64, MAC = 8, TN = 32;
    const size_t NB = T / TN;
#pragma omp parallel for
    for (long long job = 0; job < static_cast<long long>(K / TK * NB); ++job) {
        const size_t kb = static_cast<size_t>(job) / NB, nb = static_cast<size_t>(job) % NB;
        uint16_t* w = out + (kb * NB + nb) * TK * TN;
        for (size_t si = 0; si < TK / MAC; ++si)
            for (size_t ti = 0; ti < TN / MAC; ++ti)
                tile8x8(x + (nb * TN + ti * MAC) * K + kb * TK + si * MAC, K, w + (si * (TN / MAC) + ti) * MAC * MAC);
    }
}

struct Shape {
    const char* what;
    size_t T, K;
};

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/// Best of `reps`: this is a host-memory loop, and a single rep on a busy box measures the
/// neighbours. The minimum is the implementation's own cost.
template <class F>
double best_ms(F fn, int reps) {
    double best = 1e18;
    for (int r = 0; r < reps; ++r) {
        const double t0 = now_ms();
        fn();
        best = std::min(best, now_ms() - t0);
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 10;
    // The shapes a 256-token Qwen3.8 block actually asks for: hidden -> qkv|z and out read K
    // = hidden, the FFN's down reads K = ffn, and the shared expert's up|gate reads hidden.
    const Shape shapes[] = {
        {"qkv|z (K=5120)", 256, 5120},
        {"down (K=17408)", 256, 17408},
    };
    std::printf("%d threads, best of %d reps\n\n", omp_get_max_threads(), reps);

    auto dev = xrt::device(0);   // the benchmark's own handle: never a second process's
    std::printf("%-18s %11s %14s %11s %9s\n", "shape", "host scalar", "device-mapped BO",
                "host avx2", "BO/host");
    bool all_identical = true;
    for (const Shape& s : shapes) {
        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(-8.f, 8.f);
        std::vector<float> x(s.T * s.K);
        for (float& v : x) v = dist(rng);
        std::vector<uint16_t> a(s.T * s.K), b(s.T * s.K), c(s.T * s.K);

        // the real target: a device buffer, sized and mapped the way Core::gemm_dispatch maps it
        xrt::bo bo = xrt::ext::bo(dev, s.T * s.K * 2);
        uint16_t* mapped = bo.map<uint16_t*>();

        open_qwen36::host::tile_x(x.data(), s.T, s.K, a.data());
        open_qwen36::host::tile_x(x.data(), s.T, s.K, mapped);
        bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, s.T * s.K * 2, 0);
        std::memcpy(c.data(), mapped, s.T * s.K * 2);
        tile_x_avx2(x.data(), s.T, s.K, b.data());

        const bool ok_dev = a == c, ok_avx = a == b;
        all_identical = all_identical && ok_dev && ok_avx;

        const double t_host = best_ms([&] { open_qwen36::host::tile_x(x.data(), s.T, s.K, a.data()); }, reps);
        const double t_dev = best_ms([&] {
            open_qwen36::host::tile_x(x.data(), s.T, s.K, mapped);
            bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, s.T * s.K * 2, 0);   // drain, as a real GEMM's sync does
        }, reps);
        const double t_avx = best_ms([&] { tile_x_avx2(x.data(), s.T, s.K, b.data()); }, reps);
        std::printf("%-18s %9.2fms %12.2fms %9.2fms %8.1fx%s%s\n", s.what, t_host, t_dev, t_avx,
                    t_dev / t_host, ok_dev ? "" : "  DEV-MISMATCH", ok_avx ? "" : "  AVX-MISMATCH");
    }
    std::printf("\n%s\n", all_identical
                              ? "PASS: every path produces the same tiled bits"
                              : "FAIL: an implementation disagrees with the reference");
    return all_identical ? 0 : 1;
}
