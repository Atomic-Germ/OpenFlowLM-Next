/// \file pools_test.cpp
/// \brief OPEN-PACK-PLAN: the two value-changing / reshaping pack ops Qwen3.5 needs
///        (`requant_q4_1`, `transpose`) produce the same bytes here as in
///        open_kernels/recipes/pack.py. No XRT, no hardware, no model file:
///        both sides build the SAME synthetic q8 chunks from the LCG below and
///        both assert the same FNV-1a hash of the result, so a divergence in
///        either implementation fails one of the two tests.
// Traces: OPEN-PACK-PLAN, OPEN-FAMILY-QWEN35 (canonical spec: specs/open-engine/spec.md)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "open_qwen36/pools.hpp"
#include "open_qwen36/q4nx_file.hpp"

using open_qwen36::pools::q8_half_tile;
using open_qwen36::pools::requant_q4_1_chunks;
using open_qwen36::pools::transpose_bytes;

namespace {

// What open_kernels/recipes/pack.py produces on the vectors below (FNV-1a 64 of the
// output bytes). specs/open-engine/tests/test_qwen35.py asserts the same two numbers,
// so if either implementation changes, one of the two tests fails.
constexpr uint64_t REQUANT_FNV1A = 0x4ac3babad266dd49ull;
constexpr uint64_t TRANSPOSE_FNV1A = 0xb27d0b6f7149fd83ull;
// The pool a container mixing one q8 and one q4_1 tensor packs to, through pools::apply.
// specs/open-engine/tests/test_pack_plan.py asserts the same number on the same bytes.
constexpr uint64_t MIXED_FNV1A = 0x548390807a90d2ebull;
// The pool the SAME q8 tensor packs to when the kernel set streams it at q8 (`q8_perm`,
// OPEN-QUANT-Q8). specs/open-engine/tests/test_quant_q8.py asserts this number.
constexpr uint64_t Q8_POOL_FNV1A = 0x8d4a3cf75e4cbffaull;

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    failures += !ok;
}

/// The shared test vector. Chunk c: 256 bf16 scales in [2^-9, 2^-8) (finite, positive,
/// so the requantizer sees real ranges) then 8192 int8 codes, all from one LCG per chunk.
/// specs/open-engine/tests/test_qwen35.py builds these exact bytes.
std::vector<uint8_t> q8_vector(size_t nch) {
    std::vector<uint8_t> out(nch * 8704);
    for (size_t c = 0; c < nch; ++c) {
        uint32_t s = 0x9E3779B9u * static_cast<uint32_t>(c + 1);
        uint8_t* p = out.data() + c * 8704;
        for (size_t i = 0; i < 256; ++i) {
            s = s * 1664525u + 1013904223u;
            const uint16_t h = static_cast<uint16_t>(0x3B00u | (s >> 24));   // bf16, exponent 0x76
            std::memcpy(p + 2 * i, &h, 2);
        }
        for (size_t i = 512; i < 8704; ++i) {
            s = s * 1664525u + 1013904223u;
            p[i] = static_cast<uint8_t>(s >> 24);
        }
    }
    return out;
}

uint64_t fnv1a(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

/// Read a q4_1 chunk's value at (row, block, lane) the way gemv_q4.h does: nibble * d + m.
float q4_read(const uint8_t* chunk, unsigned r, unsigned b, unsigned i) {
    const unsigned meta = b * 32 + r;
    uint16_t du, mu;
    std::memcpy(&du, chunk + 2 * meta, 2);
    std::memcpy(&mu, chunk + 512 + 2 * meta, 2);
    auto f32 = [](uint16_t h) {
        uint32_t u = static_cast<uint32_t>(h) << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    };
    const unsigned p = (r / 16) * 4096 + b * 512 + i * 16 + (r % 16);
    const uint8_t byte = chunk[1024 + (p >> 1)];
    const unsigned nib = (p & 1) ? (byte >> 4) : (byte & 0xF);
    return static_cast<float>(nib) * f32(du) + f32(mu);
}

float q8_read(const uint8_t* chunk, unsigned r, unsigned b, unsigned i) {
    uint16_t sh;
    std::memcpy(&sh, chunk + 2 * (b * 32 + r), 2);
    uint32_t u = static_cast<uint32_t>(sh) << 16;
    float sc;
    std::memcpy(&sc, &u, 4);
    const unsigned p = (r / 16) * 4096 + b * 512 + i * 16 + (r % 16);
    return static_cast<float>(reinterpret_cast<const int8_t*>(chunk + 512)[p]) * sc;
}

// ---- the mixed q8 / q4_1 container (OPEN-PACK-PLAN's q8-source rule)
constexpr size_t Q8_CH = 8704, Q4_CH = 5120, BAD_CH = 1280;
constexpr size_t NCH = 8;                    // (128 rows / 32) x (512 K / 256)
constexpr uint64_t IN_DIM = 512;
const char* Q8_NAME = "model.layers.0.linear_attn.ssm_out_proj.weight";
const char* Q4_NAME = "model.layers.0.mlp.down_proj.weight";
const char* BAD_NAME = "model.layers.0.mlp.up_proj.weight";

/// The plain byte stream test_pack_plan.py's `_lcg_bytes` builds.
std::vector<uint8_t> lcg_bytes(uint32_t seed, size_t n) {
    std::vector<uint8_t> out(n);
    uint32_t s = seed;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        out[i] = static_cast<uint8_t>(s >> 24);
    }
    return out;
}

/// pool chunk index -> file chunk index, recomputed here rather than reached into pools.cpp:
/// the test asserts the law independently (gemv_q4.h's band law).
std::vector<size_t> band_perm(size_t nch, size_t in_dim) {
    const size_t ncol = in_dim / 256, per_band = in_dim / 128;
    std::vector<size_t> p(nch);
    for (size_t c = 0; c < nch; ++c)
        p[c] = (64 * (c / per_band) + 32 * (c % 2)) / 32 * ncol + 256 * ((c % per_band) / 2) / 256;
    return p;
}

/// Write a synthetic `.q4nx` (safetensors: 8-byte header length, JSON header, data) with
/// one q8 tensor, one q4_1 tensor and one whose chunks are neither.
std::string write_mixed_container(const std::vector<uint8_t>& q8, const std::vector<uint8_t>& q4,
                                  const std::vector<uint8_t>& bad) {
    auto entry = [](const char* name, size_t nch, size_t ch, size_t off) {
        return std::string("\"") + name + "\":{\"dtype\":\"I8\",\"shape\":[" + std::to_string(nch) + "," +
               std::to_string(ch) + "],\"data_offsets\":[" + std::to_string(off) + "," +
               std::to_string(off + nch * ch) + "]}";
    };
    std::string hdr = "{" + entry(Q8_NAME, NCH, Q8_CH, 0) + "," +
                      entry(Q4_NAME, NCH, Q4_CH, q8.size()) + "," +
                      entry(BAD_NAME, NCH, BAD_CH, q8.size() + q4.size()) + "}";
    const std::string path = (std::filesystem::temp_directory_path() / "open_qwen36_pools_test.q4nx").string();
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    uint64_t n = hdr.size();
    f.write(reinterpret_cast<const char*>(&n), 8);
    f.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
    f.write(reinterpret_cast<const char*>(q8.data()), static_cast<std::streamsize>(q8.size()));
    f.write(reinterpret_cast<const char*>(q4.data()), static_cast<std::streamsize>(q4.size()));
    f.write(reinterpret_cast<const char*>(bad.data()), static_cast<std::streamsize>(bad.size()));
    f.close();
    return path;
}

open_qwen36::PackOp q8_perm_op(const char* tensor, uint64_t dst) {
    open_qwen36::PackOp op;
    op.op = "q8_perm";
    op.tensor = tensor;
    op.dst = dst;
    op.nch = 2 * NCH;            // POOL half-tiles: twice the source chunks
    op.in_dim = IN_DIM;
    return op;
}

open_qwen36::PackOp std_perm_op(const char* tensor, uint64_t dst) {
    open_qwen36::PackOp op;
    op.op = "std_perm";
    op.tensor = tensor;
    op.dst = dst;
    op.nch = NCH;
    op.in_dim = IN_DIM;
    return op;
}

void mixed_container_tests() {
    const std::vector<uint8_t> q8 = q8_vector(NCH);
    const std::vector<uint8_t> q4 = lcg_bytes(0x1234567u, NCH * Q4_CH);
    const std::vector<uint8_t> bad = lcg_bytes(0x89ABCDEu, NCH * BAD_CH);
    const std::string path = write_mixed_container(q8, q4, bad);
    open_qwen36::Q4nxFile f(path);

    check(f.chunk_bytes(Q8_NAME) == Q8_CH && f.chunk_bytes(Q4_NAME) == Q4_CH &&
              f.chunk_bytes(BAD_NAME) == BAD_CH,
          "the chunk format is read per tensor, not guessed for the whole file");

    std::vector<uint8_t> pool(2 * NCH * Q4_CH, 0);
    open_qwen36::pools::apply(std_perm_op(Q8_NAME, 0), f, 0, pool.data(), pool.size(), Q4_CH);
    open_qwen36::pools::apply(std_perm_op(Q4_NAME, NCH * Q4_CH), f, 0, pool.data(), pool.size(), Q4_CH);

    // (b) the q8 half is requant_q4_1 of the q8 chunks in the SAME band order
    std::vector<uint8_t> want8(NCH * Q4_CH);
    requant_q4_1_chunks(q8.data(), NCH, want8.data());
    const auto perm = band_perm(NCH, IN_DIM);
    bool ok8 = true, ok4 = true;
    for (size_t c = 0; c < NCH && ok8; ++c)
        ok8 = std::memcmp(pool.data() + c * Q4_CH, want8.data() + perm[c] * Q4_CH, Q4_CH) == 0;
    check(ok8, "std_perm: a q8 source packs as the re-quantized q4_1, same chunk order");

    // (c) the q4_1 half is the verbatim chunk copy it always was
    for (size_t c = 0; c < NCH && ok4; ++c)
        ok4 = std::memcmp(pool.data() + (NCH + c) * Q4_CH, q4.data() + perm[c] * Q4_CH, Q4_CH) == 0;
    check(ok4, "std_perm: a q4_1 source beside it is still copied chunk for chunk");

    // (a) both packers on the same bytes
    const uint64_t got = fnv1a(pool.data(), pool.size());
    std::printf("      mixed pool fnv1a = 0x%016llx\n", static_cast<unsigned long long>(got));
    check(got == MIXED_FNV1A, "mixed q8 / q4_1 pool: byte-identical to the NumPy packer");

    // the refusal names the tensor and the byte count
    std::string msg;
    try {
        open_qwen36::pools::apply(std_perm_op(BAD_NAME, 0), f, 0, pool.data(), pool.size(), Q4_CH);
    } catch (const std::exception& e) {
        msg = e.what();
    }
    check(msg.find("mlp.up_proj.weight") != std::string::npos && msg.find("1280") != std::string::npos &&
              msg.find("Q4_K") == std::string::npos,
          "a 1280-byte chunk tensor is refused, naming it and 1280 (\"" + msg + "\")");

    // ---- OPEN-QUANT-Q8: the same q8 tensor streamed AT q8, through q8_perm
    std::vector<uint8_t> q8pool(2 * NCH * Q4_CH, 0);
    open_qwen36::pools::apply(q8_perm_op(Q8_NAME, 0), f, 0, q8pool.data(), q8pool.size(), Q4_CH);

    // the split is a byte permutation: codes verbatim, scales gathered, tail zero
    bool split_ok = true;
    for (size_t c = 0; c < NCH && split_ok; ++c)
        for (unsigned h = 0; h < 2 && split_ok; ++h) {
            std::vector<uint8_t> want(Q4_CH);
            q8_half_tile(q8.data() + c * Q8_CH, h, want.data());
            split_ok = std::memcmp(want.data() + 256, q8.data() + c * Q8_CH + 512 + h * 4096, 4096) == 0;
            for (unsigned kb = 0; kb < 8 && split_ok; ++kb)
                for (unsigned r = 0; r < 16 && split_ok; ++r)
                    split_ok = std::memcmp(want.data() + 2 * (kb * 16 + r),
                                           q8.data() + c * Q8_CH + 2 * (kb * 32 + 16 * h + r), 2) == 0;
            for (size_t i = 4352; i < Q4_CH && split_ok; ++i) split_ok = want[i] == 0;
        }
    check(split_ok, "q8_perm: the 16-row half-tile is a byte permutation of the container chunk");

    // the band law, against a placement recomputed here
    bool law_ok = true;
    const size_t per_band = IN_DIM / 64, ncol = IN_DIM / 256;
    for (size_t c = 0; c < 2 * NCH && law_ok; ++c) {
        const size_t band = c / per_band, cc = c % per_band, part = cc % 4, kt = cc / 4;
        std::vector<uint8_t> want(Q4_CH);
        q8_half_tile(q8.data() + ((2 * band + part / 2) * ncol + kt) * Q8_CH,
                     static_cast<unsigned>(part % 2), want.data());
        law_ok = std::memcmp(q8pool.data() + c * Q4_CH, want.data(), Q4_CH) == 0;
    }
    check(law_ok, "q8_perm: half-tile c of a band holds rows 16*(c%4), k-tile c/4");

    const uint64_t gotq8 = fnv1a(q8pool.data(), q8pool.size());
    std::printf("      q8 pool fnv1a = 0x%016llx\n", static_cast<unsigned long long>(gotq8));
    check(gotq8 == Q8_POOL_FNV1A, "q8_perm pool: byte-identical to the NumPy packer");
    check(q8pool.size() == 2 * NCH * Q4_CH, "a q8 projection occupies twice the q4_1 bytes");

    // a container that disagrees with the manifest's role is refused, naming the tensor
    msg.clear();
    try {
        open_qwen36::pools::apply(q8_perm_op(Q4_NAME, 0), f, 0, q8pool.data(), q8pool.size(), Q4_CH);
    } catch (const std::exception& e) {
        msg = e.what();
    }
    check(msg.find("mlp.down_proj.weight") != std::string::npos && msg.find("5120") != std::string::npos &&
              msg.find("8704") != std::string::npos,
          "q8_perm over a q4_1 tensor is refused, naming it (\"" + msg + "\")");

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

}  // namespace

int main() {
    // ---- requant_q4_1 on 12 shared chunks
    const size_t NCH = 12;
    std::vector<uint8_t> src = q8_vector(NCH);
    std::vector<uint8_t> out(NCH * 5120);
    requant_q4_1_chunks(src.data(), NCH, out.data());

    // The bound OPEN-FAMILY-QWEN35 states: every value within d/2 of its q4_1 reading.
    double worst = 0.0;
    bool any_nonzero = false;
    for (size_t c = 0; c < NCH; ++c) {
        const uint8_t* q4 = out.data() + c * 5120;
        const uint8_t* q8 = src.data() + c * 8704;
        for (unsigned r = 0; r < 32; ++r)
            for (unsigned b = 0; b < 8; ++b) {
                uint16_t du;
                std::memcpy(&du, q4 + 2 * (b * 32 + r), 2);
                uint32_t u = static_cast<uint32_t>(du) << 16;
                float d;
                std::memcpy(&d, &u, 4);
                for (unsigned i = 0; i < 32; ++i) {
                    const double e = std::abs(static_cast<double>(q8_read(q8, r, b, i)) -
                                              static_cast<double>(q4_read(q4, r, b, i)));
                    if (e > 0.0) any_nonzero = true;
                    if (d > 0.0f) worst = std::max(worst, e / (0.5 * d));
                }
            }
    }
    check(worst <= 1.0 + 1e-9, "requant_q4_1: every value within d/2 of its reading (worst " +
                                   std::to_string(worst) + " x d/2)");
    check(any_nonzero, "requant_q4_1: it really quantizes (the readings are not the q8 values)");

    // The byte-equality gate. This constant is what open_kernels/recipes/pack.py's
    // requant_q4_1 produces on the same vector; tests/test_qwen35.py asserts it too.
    const uint64_t got = fnv1a(out.data(), out.size());
    const uint64_t want = REQUANT_FNV1A;
    std::printf("      requant_q4_1 fnv1a = 0x%016llx\n", static_cast<unsigned long long>(got));
    check(got == want, "requant_q4_1: byte-identical to the NumPy packer");

    // ---- transpose: [32, 64] of 2-byte values
    std::vector<uint8_t> t_src(32 * 64 * 2), t_dst(32 * 64 * 2);
    for (size_t i = 0; i < t_src.size(); ++i) t_src[i] = static_cast<uint8_t>((i * 37 + 11) & 0xFF);
    transpose_bytes(t_src.data(), 32, 64, 2, 32, t_dst.data());
    bool ok = true;
    for (uint64_t r = 0; r < 32 && ok; ++r)
        for (uint64_t c = 0; c < 64 && ok; ++c)
            ok = std::memcmp(t_dst.data() + (c * 32 + r) * 2, t_src.data() + (r * 64 + c) * 2, 2) == 0;
    check(ok, "transpose: [32, 64] bf16 -> [64, 32]");
    std::printf("      transpose fnv1a = 0x%016llx\n",
                static_cast<unsigned long long>(fnv1a(t_dst.data(), t_dst.size())));
    check(fnv1a(t_dst.data(), t_dst.size()) == TRANSPOSE_FNV1A, "transpose: byte-identical to the NumPy packer");

    // ---- the padded transpose the 16-head DeltaNet packs: [16, 64] -> [64, 32], columns
    // 16..31 zero. The same law the NumPy packer's dst_rows takes (specs/.../test_qwen35.py).
    std::vector<uint8_t> p_src(16 * 64 * 2), p_dst(64 * 32 * 2, 0xAB);
    for (size_t i = 0; i < p_src.size(); ++i) p_src[i] = static_cast<uint8_t>((i * 37 + 11) & 0xFF);
    transpose_bytes(p_src.data(), 16, 64, 2, 32, p_dst.data());
    bool pok = true;
    for (uint64_t c = 0; c < 64 && pok; ++c) {
        for (uint64_t r = 0; r < 16 && pok; ++r)
            pok = std::memcmp(p_dst.data() + (c * 32 + r) * 2, p_src.data() + (r * 64 + c) * 2, 2) == 0;
        for (uint64_t r = 16; r < 32 && pok; ++r)
            pok = p_dst[(c * 32 + r) * 2] == 0 && p_dst[(c * 32 + r) * 2 + 1] == 0;
    }
    check(pok, "transpose: [16, 64] bf16 -> [64, 32] with columns 16..31 zero");

    // ---- a container mixing q8 and q4_1 tensors, packed through pools::apply
    mixed_container_tests();

    std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
