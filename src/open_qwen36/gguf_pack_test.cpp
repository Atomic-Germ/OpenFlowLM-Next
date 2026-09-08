/// \file gguf_pack_test.cpp
/// \brief No-XRT unit test for the GGUF-direct path: a synthetic GGUF goes
///        through pools::apply's std_perm_gguf and the packed pool bytes must
///        dequantize BIT-EXACTLY to the GGUF blocks' values (the pack is a byte
///        permutation plus an exact fp16 -> f32 widening of the scales;
///        open_kernels/gguf_pool.py holds the same law in NumPy).
///
/// Also covers GgufFile::embed_row (Q8_0) and pack_norm's f32 -> bf16.
///
///     cmake -S . -B build && cmake --build build && ctest --test-dir build
#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

#include "open_qwen36/gguf_file.hpp"
#include "open_qwen36/pools.hpp"

namespace fs = std::filesystem;

using namespace open_qwen36;

namespace {

void put_u32(std::vector<uint8_t>& f, uint32_t v) {
    f.insert(f.end(), reinterpret_cast<const uint8_t*>(&v), reinterpret_cast<const uint8_t*>(&v) + 4);
}
void put_u64(std::vector<uint8_t>& f, uint64_t v) {
    f.insert(f.end(), reinterpret_cast<const uint8_t*>(&v), reinterpret_cast<const uint8_t*>(&v) + 8);
}
void put_str(std::vector<uint8_t>& f, const std::string& s) {
    put_u64(f, s.size());
    f.insert(f.end(), s.begin(), s.end());
}
void put_kv_str(std::vector<uint8_t>& f, const std::string& k, const std::string& v) {
    put_str(f, k);
    put_u32(f, 8);                                   // GGUF_TYPE_STRING
    put_str(f, v);
}
void put_kv_u32(std::vector<uint8_t>& f, const std::string& k, uint32_t v) {
    put_str(f, k);
    put_u32(f, 4);                                   // GGUF_TYPE_UINT32
    put_u32(f, v);
}

uint16_t f32_to_f16(float f) {                       // truncating (test values only)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<uint16_t>(((u >> 16) & 0x8000) | ((((u >> 23) - 127 + 15) & 0x1F) << 10) |
                                 ((u >> 13) & 0x3FF));
}

/// GGUF blocks for a [rows, cols] tensor and the reference dequant (computed
/// from the EXACT fp16 scales that hit the file).
struct Blocks {
    std::vector<uint8_t> bytes;
    std::vector<float> values;                       // rows x cols
    unsigned rows, cols, blk;
};

Blocks make_blocks(unsigned rows, unsigned cols, const std::string& type, std::mt19937_64& rng) {
    const unsigned nb = cols / 32;
    const bool q4 = type != "Q8_0";
    const bool has_min = type == "Q4_1";
    const unsigned blk = q4 ? (has_min ? 20u : 18u) : 34u;
    Blocks out;
    out.rows = rows;
    out.cols = cols;
    out.blk = blk;
    out.bytes.assign(static_cast<size_t>(rows) * nb * blk, 0);
    out.values.resize(static_cast<size_t>(rows) * cols);
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned b = 0; b < nb; ++b) {
            uint8_t* p = out.bytes.data() + (static_cast<size_t>(r) * nb + b) * blk;
            const uint16_t dh = f32_to_f16(0.01f + 0.02f * static_cast<float>(rng() % 1000) / 1000.f);
            const uint16_t mh = has_min ? f32_to_f16(-0.05f + 0.04f * static_cast<float>(rng() % 1000) / 1000.f) : 0;
            std::memcpy(p, &dh, 2);
            if (has_min) std::memcpy(p + 2, &mh, 2);
            const float df = fp16_to_f32(dh);        // the exact value that hits the file
            const float mf = has_min ? fp16_to_f32(mh) : 0.f;
            for (unsigned j = 0; j < 32; ++j) {
                if (q4) {
                    const uint8_t q = static_cast<uint8_t>(rng() % 16);
                    if (j < 16) p[has_min ? 4 + j : 2 + j] |= q;
                    else p[has_min ? 4 + (j - 16) : 2 + (j - 16)] |= static_cast<uint8_t>(q << 4);
                    out.values[static_cast<size_t>(r) * cols + b * 32 + j] = df * q + mf;
                } else {
                    const int8_t q = static_cast<int8_t>(rng() % 256) - 128;
                    p[2 + j] = static_cast<uint8_t>(q);
                    out.values[static_cast<size_t>(r) * cols + b * 32 + j] = df * q;
                }
            }
        }
    }
    return out;
}

std::string write_gguf(const fs::path& dir, const Blocks& mm, const Blocks& emb, const std::vector<float>& norm) {
    std::vector<uint8_t> f;
    f.insert(f.end(), {'G', 'G', 'U', 'F'});
    put_u32(f, 3);                                   // v3: no alignment field, fixed 32
    put_u64(f, 3);                                   // tensors
    put_u64(f, 2);                                   // kv pairs
    put_kv_str(f, "general.architecture", "llama");
    put_kv_u32(f, "llama.embedding_length", 512);
    // tensor infos with placeholder offsets (patched once the data layout is known)
    const auto put_tensor = [&](const std::string& name, const std::vector<uint64_t>& dims, uint32_t type) {
        put_str(f, name);
        put_u32(f, static_cast<uint32_t>(dims.size()));
        for (uint64_t d : dims) put_u64(f, d);
        put_u32(f, type);
        put_u64(f, 0);
    };
    put_tensor("blk.0.attn_q.weight", {512, 64}, 3);         // Q4_1
    put_tensor("token_embd.weight", {256, 32}, 8);           // Q8_0
    put_tensor("blk.0.attn_norm.weight", {512}, 0);          // F32
    // data section: 32-aligned right after the header; the tensor offsets are
    // RELATIVE to it (GGUF semantics)
    const auto align32 = [](size_t n) { return (n + 31) / 32 * 32; };
    const size_t off = align32(f.size());
    std::vector<uint64_t> offs;                  // relative offsets, patched into the header
    size_t rel = 0;
    auto emit = [&](const std::vector<uint8_t>& b) {
        offs.push_back(rel);
        rel = align32(rel + b.size());
    };
    emit(mm.bytes);
    emit(emb.bytes);
    offs.push_back(rel);
    // patch the three tensor offsets
    const size_t n_off_fields = 3;
    for (size_t t = 0; t < n_off_fields; ++t) {
        // the t-th tensor info's offset field: walk the header
        size_t p = 4 + 4 + 8 + 8;
        // skip the kv pairs
        for (int kv = 0; kv < 2; ++kv) {
            uint64_t ln;
            std::memcpy(&ln, f.data() + p, 8);
            p += 8 + ln;                             // key
            uint32_t ty;
            std::memcpy(&ty, f.data() + p, 4);
            p += 4;
            if (ty == 8) {                           // string value
                std::memcpy(&ln, f.data() + p, 8);
                p += 8 + ln;
            } else {                                 // u32 value
                p += 4;
            }
        }
        for (size_t i = 0; i < t; ++i) {
            uint64_t ln;
            uint32_t nd;
            std::memcpy(&ln, f.data() + p, 8);
            p += 8 + ln;                             // name
            std::memcpy(&nd, f.data() + p, 4);
            p += 4 + 8 * nd;                         // dims
            p += 4;                                  // type
            p += 8;                                  // the offset field itself
        }
        uint64_t ln;
        uint32_t nd;
        std::memcpy(&ln, f.data() + p, 8);
        p += 8 + ln;                                 // name
        std::memcpy(&nd, f.data() + p, 4);
        p += 4 + 8 * nd + 4;                         // dims, type -> the offset field
        std::memcpy(f.data() + p, &offs[t], 8);
    }
    while (f.size() < off + offs[0]) f.push_back(0);   // pad to the first tensor
    f.insert(f.end(), mm.bytes.begin(), mm.bytes.end());
    while (f.size() < off + offs[1]) f.push_back(0);
    f.insert(f.end(), emb.bytes.begin(), emb.bytes.end());
    while (f.size() < off + offs[2]) f.push_back(0);
    for (float v : norm) {
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
        f.insert(f.end(), b, b + 4);
    }
    const fs::path out = dir / "test.gguf";
    std::ofstream of(out, std::ios::binary);
    of.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(f.size()));
    return out.string();
}

uint32_t f32_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

}  // namespace

int main() {
    std::mt19937_64 rng(42);
    const Blocks mm = make_blocks(64, 512, "Q4_1", rng);
    std::vector<float> norm(512);
    for (float& v : norm) v = 0.9f + 0.1f * static_cast<float>(rng() % 1000) / 1000.f;
    const Blocks emb = make_blocks(32, 256, "Q8_0", rng);
    const fs::path dir = fs::temp_directory_path() / "gguf_pack_test";
    fs::create_directories(dir);
    GgufFile g(write_gguf(dir, mm, emb, norm));

    // ---- header sanity
    if (g.kv_str("general.architecture") != "llama" || g.kv_u64("llama.embedding_length") != 512 ||
        !g.has("model.layers.0.self_attn.q_proj.weight")) {   // found through the HF-name map
        std::printf("FAIL header\n");
        return 1;
    }
    if (GgufFile::gguf_name("model.embed_tokens.weight") != "token_embd.weight" ||
        GgufFile::gguf_name("lm_head.weight") != "output.weight" ||
        GgufFile::gguf_name("model.layers.3.mlp.down_proj.weight") != "blk.3.ffn_down.weight" ||
        GgufFile::gguf_name("model.layers.1.input_layernorm.weight") != "blk.1.attn_norm.weight" ||
        GgufFile::gguf_name("model.layers.2.self_attn.q_norm.weight") != "blk.2.attn_q_norm.weight" ||
        GgufFile::gguf_name("model.norm.weight") != "output_norm.weight") {
        std::printf("FAIL name map\n");
        return 1;
    }
    std::printf("ok    header (kv, tensor index, HF-name map)\n");

    // ---- std_perm_gguf: the pool bytes dequantize to the GGUF values bit-exactly
    PackOp op;
    op.op = "std_perm_gguf";
    op.tensor = "model.layers.0.self_attn.q_proj.weight";   // the manifest HF-style name
    op.in_dim = 512;
    op.nch = 64ull * 512 / (32 * 256);               // 4 chunks of 32x256
    std::vector<uint8_t> pool(static_cast<size_t>(op.nch) * 6144, 0);
    pools::apply(op, g, 0, pool.data(), pool.size(), 6144);

    size_t bad = 0;
    for (size_t c = 0; c < op.nch; ++c) {
        const uint8_t* d = pool.data() + c * 6144;
        // the band law: rows0 = 64*(c/per_band) + 32*(c%2), cols0 = 256*((c%per_band)/2)
        const size_t per_band = 4;
        const size_t rows0 = 64 * (c / per_band) + 32 * (c % 2), cols0 = 256 * ((c % per_band) / 2);
        for (size_t r = 0; r < 32; ++r) {
            for (size_t k = 0; k < 256; ++k) {
                const size_t kb = k / 32;
                const float df = *reinterpret_cast<const float*>(d + 4 * (kb * 32 + r));
                const float mf = *reinterpret_cast<const float*>(d + 1024 + 4 * (kb * 32 + r));
                const size_t p = (r / 16) * 4096 + k * 16 + (r % 16);
                const uint8_t byte = d[2048 + (p >> 1)];
                const uint8_t nib = (p & 1) ? byte >> 4 : byte & 0xF;
                const float got = nib * df + mf;
                const float want = mm.values[(rows0 + r) * mm.cols + cols0 + k];
                if (f32_bits(got) != f32_bits(want)) {
                    if (bad < 5)
                        std::printf("  chunk %zu r=%zu k=%zu: got %.9g want %.9g\n", c, r, k, got, want);
                    ++bad;
                }
            }
        }
    }
    std::printf("%s std_perm_gguf (%zu of %u values differ)\n", bad ? "FAIL" : "ok", bad, op.nch * 32u * 256u);
    if (bad) return 1;

    // ---- pack_norm: f32 -> bf16 (RNE)
    {
        std::vector<uint8_t> nb(norm.size() * 2);
        pools::pack_norm(g, "blk.0.attn_norm.weight", nb.size(), nb.data());
        for (size_t i = 0; i < norm.size(); ++i) {
            uint32_t u;
            std::memcpy(&u, &norm[i], 4);
            const uint16_t want = static_cast<uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
            uint16_t got;
            std::memcpy(&got, nb.data() + 2 * i, 2);
            if (got != want) {
                std::printf("FAIL pack_norm[%zu]: got %04x want %04x\n", i, got, want);
                return 1;
            }
        }
        std::printf("ok    pack_norm f32 -> bf16 (RNE)\n");
    }

    // ---- embed_row (Q8_0, bit-exact)
    {
        std::vector<float> row(256);
        g.embed_row("token_embd.weight", 7, 256, row.data());
        for (size_t i = 0; i < 256; ++i) {
            if (f32_bits(row[i]) != f32_bits(emb.values[7 * 256 + i])) {
                std::printf("FAIL embed_row(Q8_0)[%zu]: got %.9g want %.9g\n", i, row[i], emb.values[7 * 256 + i]);
                return 1;
            }
        }
        std::printf("ok    embed_row Q8_0 (bit-exact)\n");
    }

    std::printf("GGUF-PACK OK\n");
    return 0;
}
