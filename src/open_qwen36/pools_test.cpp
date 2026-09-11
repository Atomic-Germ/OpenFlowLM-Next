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
#include <utility>
#include <vector>

#include "open_qwen36/manifest.hpp"
#include "open_qwen36/pools.hpp"
#include "open_qwen36/q4nx_file.hpp"

using open_qwen36::pools::q8_half_tile;
using open_qwen36::pools::q4k_to_q4_1_chunks;
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
// The Q4_K chunks of q4k_vector(NCH) transcoded to q4_1, and the pool std_perm puts them
// in beside a q4_1 tensor (OPEN-QUANT-Q4K). tests/test_quant_q4k.py asserts both.
constexpr uint64_t Q4K_TRANSCODE_FNV1A = 0x685dc049ec1ca2d7ull;
constexpr uint64_t Q4K_POOL_FNV1A = 0xb02083912551d9d3ull;

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

/// Write a synthetic `.q4nx` (safetensors: 8-byte header length, JSON header, data) from
/// {tensor name, chunk width, bytes} triples -- the mixed q8 / q4_1 container and the Q4_K
/// one differ only in what they hold.
struct Tensor {
    const char* name;
    size_t ch;
    const std::vector<uint8_t>* data;
};

std::string write_container(const std::string& stem, const std::vector<Tensor>& ts) {
    std::string hdr = "{";
    size_t off = 0;
    for (size_t i = 0; i < ts.size(); ++i) {
        const size_t n = ts[i].data->size();
        hdr += (i ? "," : "") + std::string("\"") + ts[i].name + "\":{\"dtype\":\"I8\",\"shape\":[" +
               std::to_string(n / ts[i].ch) + "," + std::to_string(ts[i].ch) + "],\"data_offsets\":[" +
               std::to_string(off) + "," + std::to_string(off + n) + "]}";
        off += n;
    }
    hdr += "}";
    const std::string path = (std::filesystem::temp_directory_path() / (stem + ".q4nx")).string();
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    uint64_t n = hdr.size();
    f.write(reinterpret_cast<const char*>(&n), 8);
    f.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
    for (const Tensor& t : ts)
        f.write(reinterpret_cast<const char*>(t.data->data()), static_cast<std::streamsize>(t.data->size()));
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
    const std::string path = write_container("open_qwen36_pools_test",
                                             {{Q8_NAME, Q8_CH, &q8}, {Q4_NAME, Q4_CH, &q4}, {BAD_NAME, BAD_CH, &bad}});
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
    check(msg.find("mlp.up_proj.weight") != std::string::npos &&
              msg.find("1280 is a smaller chunk geometry") != std::string::npos,
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

// ---- OPEN-QUANT-Q4K: a Q4_K source (4736-byte chunks, what OFLM 1.0.3+ writes)
constexpr size_t Q4K_CH = 4736;
const char* Q4K_NAME = "model.layers.0.mlp.gate_proj.weight";

/// The Q4_K test vector specs/open-engine/tests/test_quant_q4k.py builds byte for byte:
/// per chunk one LCG seeded from the chunk index; 4608 plain bytes (uint8 scales, uint8
/// mins, nibbles), then 32 bf16 `S` with exponent 0x76 and 32 bf16 `M` with the same
/// exponent and the sign bit set -- a Q4_K min is stored already negated.
std::vector<uint8_t> q4k_vector(size_t nch) {
    std::vector<uint8_t> out(nch * Q4K_CH);
    for (size_t c = 0; c < nch; ++c) {
        uint32_t s = 0x9E3779B9u * static_cast<uint32_t>(c + 1);
        uint8_t* p = out.data() + c * Q4K_CH;
        for (size_t i = 0; i < 4608; ++i) {
            s = s * 1664525u + 1013904223u;
            p[i] = static_cast<uint8_t>(s >> 24);
        }
        for (size_t i = 0; i < 32; ++i) {
            s = s * 1664525u + 1013904223u;
            const uint16_t h = static_cast<uint16_t>(0x3B00u | (s >> 24));
            std::memcpy(p + 4608 + 2 * i, &h, 2);
        }
        for (size_t i = 0; i < 32; ++i) {
            s = s * 1664525u + 1013904223u;
            const uint16_t h = static_cast<uint16_t>(0xBB00u | (s >> 24));
            std::memcpy(p + 4672 + 2 * i, &h, 2);
        }
    }
    return out;
}

/// A Q4_K chunk's value at (row, block, lane), read straight off the layout rather than
/// through the transcode: value = S[r]*scales[b*32+r]*nib + M[r]*mins[b*32+r]. The two
/// terms come back separately because each is rounded once by the transcode, so the error
/// bound is 2^-8 * (|scale term| + |min term|) -- and where the two cancel, that is much
/// larger than 2^-8 * |value|.
std::pair<double, double> q4k_terms(const uint8_t* chunk, unsigned r, unsigned b, unsigned i) {
    auto f32 = [](uint16_t h) {
        uint32_t u = static_cast<uint32_t>(h) << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return static_cast<double>(f);
    };
    uint16_t sh, mh;
    std::memcpy(&sh, chunk + 4608 + 2 * r, 2);
    std::memcpy(&mh, chunk + 4672 + 2 * r, 2);
    const unsigned k = b * 32 + i;
    const uint8_t byte = chunk[512 + k * 16 + r / 2];
    const unsigned nib = (r % 2) ? (byte >> 4) : (byte & 0xF);
    return {nib * f32(sh) * chunk[b * 32 + r], f32(mh) * chunk[256 + b * 32 + r]};
}

void q4k_container_tests() {
    const std::vector<uint8_t> q4k = q4k_vector(NCH);
    const std::vector<uint8_t> q4 = lcg_bytes(0x1234567u, NCH * Q4_CH);
    const std::vector<uint8_t> bad = lcg_bytes(0x89ABCDEu, NCH * BAD_CH);
    const std::string path = write_container("open_qwen36_pools_test_q4k",
                                             {{Q4K_NAME, Q4K_CH, &q4k}, {Q4_NAME, Q4_CH, &q4}, {BAD_NAME, BAD_CH, &bad}});
    open_qwen36::Q4nxFile f(path);
    check(f.chunk_bytes(Q4K_NAME) == Q4K_CH, "a Q4_K tensor's chunk width is read per tensor");

    // the transcode itself: the reading is the container's value up to the bf16 rounding of
    // the two products -- a half-ulp each, 2^-8 relative
    std::vector<uint8_t> tr(NCH * Q4_CH);
    q4k_to_q4_1_chunks(q4k.data(), NCH, tr.data());
    double worst = 0.0;
    bool moved = false;
    for (size_t c = 0; c < NCH; ++c)
        for (unsigned r = 0; r < 32; ++r)
            for (unsigned b = 0; b < 8; ++b)
                for (unsigned i = 0; i < 32; ++i) {
                    const auto t = q4k_terms(q4k.data() + c * Q4K_CH, r, b, i);
                    const double want = t.first + t.second;
                    const double got = q4_read(tr.data() + c * Q4_CH, r, b, i);
                    const double tol = (0x1p-8 + 0x1p-20) * (std::abs(t.first) + std::abs(t.second));
                    worst = std::max(worst, std::abs(got - want) / (tol + 1e-300));
                    moved = moved || got != want;
                }
    check(worst <= 1.0, "q4k_to_q4_1: within a bf16 half-ulp of the container's value (worst " +
                            std::to_string(worst) + " x the bound)");
    check(moved, "q4k_to_q4_1: the rounding is really exercised");

    const uint64_t gott = fnv1a(tr.data(), tr.size());
    std::printf("      q4k transcode fnv1a = 0x%016llx\n", static_cast<unsigned long long>(gott));
    check(gott == Q4K_TRANSCODE_FNV1A, "q4k_to_q4_1: byte-identical to the NumPy transcode");

    // as a pack source: the permutation is applied to transcoded chunks, exactly as to
    // file chunks, and a q4_1 tensor beside it is still a verbatim copy
    std::vector<uint8_t> pool(2 * NCH * Q4_CH, 0);
    open_qwen36::pools::apply(std_perm_op(Q4K_NAME, 0), f, 0, pool.data(), pool.size(), Q4_CH);
    open_qwen36::pools::apply(std_perm_op(Q4_NAME, NCH * Q4_CH), f, 0, pool.data(), pool.size(), Q4_CH);
    const auto perm = band_perm(NCH, IN_DIM);
    bool okk = true, ok4 = true;
    for (size_t c = 0; c < NCH && okk; ++c)
        okk = std::memcmp(pool.data() + c * Q4_CH, tr.data() + perm[c] * Q4_CH, Q4_CH) == 0;
    check(okk, "std_perm: a Q4_K source packs as the transcoded q4_1, same chunk order");
    for (size_t c = 0; c < NCH && ok4; ++c)
        ok4 = std::memcmp(pool.data() + (NCH + c) * Q4_CH, q4.data() + perm[c] * Q4_CH, Q4_CH) == 0;
    check(ok4, "std_perm: a q4_1 source beside a Q4_K one is still copied chunk for chunk");

    const uint64_t got = fnv1a(pool.data(), pool.size());
    std::printf("      q4k pool fnv1a = 0x%016llx\n", static_cast<unsigned long long>(got));
    check(got == Q4K_POOL_FNV1A, "Q4_K / q4_1 pool: byte-identical to the NumPy packer");

    // a kernel set that streams this projection at q8 is not satisfied by Q4_K
    std::string msg;
    try {
        open_qwen36::pools::apply(q8_perm_op(Q4K_NAME, 0), f, 0, pool.data(), pool.size(), Q4_CH);
    } catch (const std::exception& e) {
        msg = e.what();
    }
    check(msg.find("mlp.gate_proj.weight") != std::string::npos && msg.find("4736") != std::string::npos &&
              msg.find("8704") != std::string::npos,
          "q8_perm over a Q4_K tensor is refused, naming it (\"" + msg + "\")");

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

/// A non-unit `RowGlobal::scale` (longrope's attention factor, OPEN-FAMILY-PHI3): the
/// whole-table builder `pools::build_ptab` matches `recipes/pack.py`'s `ptab(..., scale=)`
/// byte for byte. This is the C++ side of a Python-only gap: `test_phi3.py`'s
/// `test_ptab_applies_the_scale_to_cos_and_sin` checked only `pack.ptab`, so a Python/C++
/// mismatch on the scaled path could have reached hardware unnoticed.
constexpr uint64_t PTAB_SCALE_FNV1A = 0x091cd4029a9681a8ull;

void ptab_scale_tests() {
    open_qwen36::Manifest m;
    m.rotary_dim = 8;
    m.ptab_row = 1024;
    open_qwen36::RowGlobal g;
    g.inv_freq = {0.5, 0.25, 0.125, 0.0625};
    g.scale = 1.19;
    g.window = 0;
    const size_t rows = 6;
    std::vector<uint8_t> t(rows * m.ptab_row);
    open_qwen36::pools::build_ptab(m, g, rows, t.data());

    // cos/sin at row p, pair i: scale * cos(p * inv_freq[i]) / sin(...), at +512 / +512+2*half
    const size_t half = g.inv_freq.size();
    bool ok = true;
    for (size_t p = 0; p < rows && ok; ++p) {
        for (size_t i = 0; i < half && ok; ++i) {
            float c, s;
            std::memcpy(&c, t.data() + p * m.ptab_row + 512 + 4 * i, 4);
            std::memcpy(&s, t.data() + p * m.ptab_row + 512 + 4 * half + 4 * i, 4);
            const double ang = static_cast<double>(p) * g.inv_freq[i];
            ok = std::abs(c - static_cast<float>(g.scale * std::cos(ang))) < 1e-6f &&
                 std::abs(s - static_cast<float>(g.scale * std::sin(ang))) < 1e-6f;
        }
    }
    check(ok, "build_ptab: a non-unit scale multiplies both cos and sin");
    const uint64_t got = fnv1a(t.data(), t.size());
    std::printf("      ptab scale fnv1a = 0x%016llx\n", static_cast<unsigned long long>(got));
    check(got == PTAB_SCALE_FNV1A, "build_ptab: byte-identical to recipes/pack.py's ptab(..., scale=)");

    // unit scale (every other family) is the unscaled table -- the key does not move it
    open_qwen36::RowGlobal g1 = g;
    g1.scale = 1.0;
    std::vector<uint8_t> t1(rows * m.ptab_row);
    open_qwen36::pools::build_ptab(m, g1, rows, t1.data());
    bool differs = std::memcmp(t.data(), t1.data(), t.size()) != 0;
    check(differs, "build_ptab: a non-unit scale actually changes the bytes (not silently ignored)");
}

/// Phi-3's longrope, the full mechanism: row r reads `long_inv_freq` once r >= switch_row,
/// `inv_freq` before it, together with the attention scale (OPEN-FAMILY-PHI3). The C++ side
/// of `test_phi3.py`'s `test_ptab_switches_table_per_row_at_the_threshold` /
/// `test_numpy_and_cpp_agree_on_a_switched_and_scaled_ptab`: a C++-only regression in the
/// switch itself (as opposed to a plain scale) could otherwise pass every Python test.
constexpr uint64_t PTAB_SWITCH_FNV1A = 0x7000741bf5ffa40bull;

void ptab_switch_tests() {
    open_qwen36::Manifest m;
    m.rotary_dim = 8;
    m.ptab_row = 1024;
    open_qwen36::RowGlobal g;
    g.inv_freq = {0.5, 0.25, 0.125, 0.0625};
    g.long_inv_freq = {3.0, 1.5, 0.75, 0.375};
    g.switch_row = 4;
    g.scale = 1.19;
    const size_t rows = 10, half = g.inv_freq.size();
    std::vector<uint8_t> t(rows * m.ptab_row);
    open_qwen36::pools::build_ptab(m, g, rows, t.data());

    bool ok = true;
    for (size_t p = 0; p < rows && ok; ++p) {
        const std::vector<double>& freq = (p >= g.switch_row) ? g.long_inv_freq : g.inv_freq;
        for (size_t i = 0; i < half && ok; ++i) {
            float c, s;
            std::memcpy(&c, t.data() + p * m.ptab_row + 512 + 4 * i, 4);
            std::memcpy(&s, t.data() + p * m.ptab_row + 512 + 4 * half + 4 * i, 4);
            const double ang = static_cast<double>(p) * freq[i];
            ok = std::abs(c - static_cast<float>(g.scale * std::cos(ang))) < 1e-6f &&
                 std::abs(s - static_cast<float>(g.scale * std::sin(ang))) < 1e-6f;
        }
    }
    check(ok, "build_ptab: row r reads long_inv_freq once r >= switch_row, inv_freq before it");
    const uint64_t got = fnv1a(t.data(), t.size());
    std::printf("      ptab switch fnv1a = 0x%016llx\n", static_cast<unsigned long long>(got));
    check(got == PTAB_SWITCH_FNV1A, "build_ptab: byte-identical to recipes/pack.py's switched ptab");

    // kSwitchNever (every other family): switch_row so large that row >= switch_row never
    // holds, so long_inv_freq (here left empty, as an unused RowGlobal always has it) is
    // never read even though the field exists on the struct.
    open_qwen36::RowGlobal g2;
    g2.inv_freq = g.inv_freq;
    check(g2.switch_row == open_qwen36::RowGlobal::kSwitchNever, "RowGlobal: switch_row defaults to kSwitchNever");
    std::vector<uint8_t> t2(rows * m.ptab_row);
    open_qwen36::pools::build_ptab(m, g2, rows, t2.data());   // must not read g2.long_inv_freq (empty)
    for (size_t p = 0; p < rows && ok; ++p)
        for (size_t i = 0; i < half && ok; ++i) {
            float c;
            std::memcpy(&c, t2.data() + p * m.ptab_row + 512 + 4 * i, 4);
            ok = std::abs(c - static_cast<float>(std::cos(static_cast<double>(p) * g.inv_freq[i]))) < 1e-6f;
        }
    check(ok, "build_ptab: kSwitchNever reads inv_freq for every row, long_inv_freq never touched");
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
    q4k_container_tests();
    ptab_scale_tests();
    ptab_switch_tests();

    std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
