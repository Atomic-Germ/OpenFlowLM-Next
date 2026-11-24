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
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gguf/gguf_file.hpp"
#include "open_qwen36/pools.hpp"
#include "open_qwen36/q4nx_file.hpp"

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
    unsigned rows, cols, blk, type;
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
    out.type = type == "Q4_0" ? 2u : type == "Q4_1" ? 3u : 8u;
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
                    out.values[static_cast<size_t>(r) * cols + b * 32 + j] = df *
                        (has_min ? static_cast<float>(q) : static_cast<float>(static_cast<int>(q) - 8)) + mf;
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

/// Q6_K tensor blocks and the reference dequant. Q6_K is the one super-block
/// quant whose embed_row had a real bug (the `qh` pointer advanced 16 B per
/// 128-value half instead of 32, skewing every second half of a super-block),
/// so this test is deliberately self-contained: it synthesizes the block with a
/// known code layout and checks GgufFile::embed_row against it.
struct BlocksQ6K {
    unsigned rows, cols;                 // cols is a multiple of 256
    std::vector<uint8_t> bytes;          // rows * (cols/256) * 210
    std::vector<float> values;           // rows * cols
};

BlocksQ6K make_blocks_q6k(unsigned rows, unsigned cols, std::mt19937_64& rng) {
    const unsigned nbs = cols / 256;                 // super-blocks per row
    BlocksQ6K b;
    b.rows = rows;
    b.cols = cols;
    b.bytes.assign(static_cast<size_t>(rows) * nbs * 210, 0);
    b.values.resize(static_cast<size_t>(rows) * cols);
    const uint16_t dh = f32_to_f16(0.5f);            // exactly representable
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned sb = 0; sb < nbs; ++sb) {
            uint8_t* p = b.bytes.data() + (static_cast<size_t>(r) * nbs + sb) * 210;
            std::memcpy(p + 208, &dh, 2);            // d
            std::memset(p + 192, 1, 16);             // scales: all 1 (int8), so y = d * 1 * (code-32)
            float* y = b.values.data() + static_cast<size_t>(r) * cols + sb * 256;
            for (int v = 0; v < 256; ++v) {
                const uint8_t code = static_cast<uint8_t>(rng() % 64);
                // ql: one byte per value's LOW 4 bits; low nibble for the first 64 of a
                // 128-half, high nibble for the next 64 (the dequant's q1/q2 vs q3/q4).
                const unsigned ql_byte = (v >= 128 ? 64u : 0u) + static_cast<unsigned>(v & 63);
                if ((v % 128) >= 64) p[ql_byte] |= static_cast<uint8_t>((code & 0xF) << 4);
                else p[ql_byte] |= static_cast<uint8_t>(code & 0xF);
                // qh: 32 bytes per 128-half, each covers 4 values (bits 0/2/4/6).
                const unsigned rel = static_cast<unsigned>((v >= 128 ? v - 128 : v) % 128);
                const unsigned l = rel % 32, which = rel / 32;
                p[128 + (v >= 128 ? 32u + l : l)] |=
                    static_cast<uint8_t>(((code >> 4) & 0x3) << (2 * which));
                y[v] = 0.5f * static_cast<float>(static_cast<int>(code) - 32);
            }
        }
    }
    return b;
}

/// Q5_0 / Q5_1: 22 / 24 bytes per 32 values, with the fifth bits in a trailing
/// u32 (bit j for the low-nibble of nibble byte j, bit j+12 for its high one).
/// llama.cpp emits the values INTERLEAVED (2*j, 2*j+1); this fixture encodes
/// that law independently of the reader it tests.
struct BlocksQ5 {
    unsigned rows, cols, blk, type;
    std::vector<uint8_t> bytes;
    std::vector<float> values;
};

BlocksQ5 make_blocks_q5(unsigned rows, unsigned cols, unsigned type, std::mt19937_64& rng) {
    BlocksQ5 b;
    b.rows = rows;
    b.cols = cols;
    b.type = type;
    b.blk = type == 6 ? 22 : 24;                      // q5_0 / q5_1
    b.bytes.assign(static_cast<size_t>(rows) * (cols / 32) * b.blk, 0);
    b.values.resize(static_cast<size_t>(rows) * cols);
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned bl = 0; bl < cols / 32; ++bl) {
            uint8_t* p = b.bytes.data() + (static_cast<size_t>(r) * (cols / 32) + bl) * b.blk;
            const uint16_t dh = f32_to_f16(0.01f + 0.02f * static_cast<float>(rng() % 1000) / 1000.f);
            const uint16_t mh = type == 7 ? f32_to_f16(-0.05f + 0.04f * static_cast<float>(rng() % 1000) / 1000.f) : 0;
            std::memcpy(p, &dh, 2);
            if (type == 7) std::memcpy(p + 2, &mh, 2);
            const float d = fp16_to_f32(dh), m = type == 7 ? fp16_to_f32(mh) : 0.f;
            uint32_t qh = 0;
            for (unsigned j = 0; j < 16; ++j) {
                const uint8_t lo = static_cast<uint8_t>(rng() % 16);
                const uint8_t hi = static_cast<uint8_t>(rng() % 16);
                const bool blo = (rng() & 1) != 0, bhi = (rng() & 1) != 0;
                p[(type == 7 ? 4 : 2) + j] = static_cast<uint8_t>(lo | (hi << 4));
                if (blo) qh |= 1u << j;
                if (bhi) qh |= 1u << (j + 16);       // read back with (qh >> (j+12)) & 0x10
                const int v0 = lo | (blo ? 0x10 : 0), v1 = hi | (bhi ? 0x10 : 0);
                float* y = b.values.data() + static_cast<size_t>(r) * cols + bl * 32 + 2 * j;
                y[0] = type == 7 ? d * static_cast<float>(v0) + m : d * static_cast<float>(v0 - 16);
                y[1] = type == 7 ? d * static_cast<float>(v1) + m : d * static_cast<float>(v1 - 16);
            }
            std::memcpy(p + b.blk - 4, &qh, 4);
        }
    }
    return b;
}

/// A one-tensor GGUF carrying a single Q6_K tensor (rows x 256).
std::string write_gguf_q6k(const fs::path& dir, const BlocksQ6K& b) {
    std::vector<uint8_t> f;
    f.insert(f.end(), {'G', 'G', 'U', 'F'});
    put_u32(f, 3);
    put_u64(f, 1);                                    // tensors
    put_u64(f, 1);                                    // kv pairs
    put_kv_str(f, "general.architecture", "llama");
    const auto align32 = [](size_t n) { return (n + 31) / 32 * 32; };
    put_str(f, "blk.0.attn_q.weight");
    put_u32(f, 2);
    put_u64(f, b.cols);                               // cols (fastest)
    put_u64(f, b.rows);
    put_u32(f, 14);                                   // GGML_TYPE_Q6_K
    put_u64(f, 0);                                    // RELATIVE offset (0: data sits at the data base)
    const size_t data_off = align32(f.size());        // == the GGUF's data base
    while (f.size() < data_off) f.push_back(0);
    f.insert(f.end(), b.bytes.begin(), b.bytes.end());
    const fs::path out = dir / "test-q6k.gguf";
    std::ofstream of(out, std::ios::binary);
    of.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(f.size()));
    return out.string();
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
    put_tensor("blk.0.attn_q.weight", {512, 64}, mm.type);
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
    const fs::path out = dir / ("test-" + std::to_string(mm.type) + ".gguf");
    std::ofstream of(out, std::ios::binary);
    of.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(f.size()));
    return out.string();
}

uint32_t f32_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

/// A GGUF of arbitrary tensors (the MoE test needs 3-D expert tensors and a 2-D router;
/// write_gguf above is fixed at three tensors).
struct SynthTensor {
    std::string name;
    std::vector<uint64_t> dims;                  // fastest first, as GGUF stores them
    uint32_t type;
    std::vector<uint8_t> bytes;
};

std::string write_gguf_tensors(const fs::path& dir, const std::string& file,
                               const std::vector<SynthTensor>& ts,
                               uint32_t split_no = 0, uint32_t split_count = 0) {
    const bool split = split_count > 1;
    std::vector<uint8_t> f;
    f.insert(f.end(), {'G', 'G', 'U', 'F'});
    put_u32(f, 3);
    put_u64(f, ts.size());
    put_u64(f, split ? 3 : 1);
    put_kv_str(f, "general.architecture", "qwen35moe");
    if (split) {
        put_kv_u32(f, "split.no", split_no);
        put_kv_u32(f, "split.count", split_count);
    }
    for (const auto& t : ts) {
        put_str(f, t.name);
        put_u32(f, static_cast<uint32_t>(t.dims.size()));
        for (uint64_t d : t.dims) put_u64(f, d);
        put_u32(f, t.type);
        put_u64(f, 0);                            // patched below
    }
    const auto align32 = [](size_t n) { return (n + 31) / 32 * 32; };
    const size_t data_off = align32(f.size());
    std::vector<uint64_t> offs;
    uint64_t rel = 0;
    for (const auto& t : ts) {
        offs.push_back(rel);
        rel = align32(rel + t.bytes.size());
    }
    // walk the header to the offset field of each tensor info
    size_t p = 4 + 4 + 8 + 8;
    for (uint32_t kv = 0; kv < (split ? 3u : 1u); ++kv) {
        uint64_t ln;
        uint32_t ty;
        std::memcpy(&ln, f.data() + p, 8);
        p += 8 + ln;
        std::memcpy(&ty, f.data() + p, 4);
        p += 4;
        if (ty == 8) {
            std::memcpy(&ln, f.data() + p, 8);
            p += 8 + ln;
        } else {
            p += 4;
        }
    }
    for (size_t i = 0; i < ts.size(); ++i) {
        uint64_t ln;
        uint32_t nd;
        std::memcpy(&ln, f.data() + p, 8);
        p += 8 + ln;
        std::memcpy(&nd, f.data() + p, 4);
        p += 4 + 8 * nd + 4;
        std::memcpy(f.data() + p, &offs[i], 8);
        p += 8;
    }
    while (f.size() < data_off) f.push_back(0);
    for (size_t i = 0; i < ts.size(); ++i) {
        while (f.size() < data_off + offs[i]) f.push_back(0);
        f.insert(f.end(), ts[i].bytes.begin(), ts[i].bytes.end());
    }
    const fs::path out = dir / file;
    std::ofstream of(out, std::ios::binary);
    of.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(f.size()));
    return out.string();
}

/// The q4_1 value a pool chunk holds at (row r, column k), in the bf16-scale law (5120 B).
float pool_q4_1_value(const uint8_t* d, size_t r, size_t k) {
    const size_t kb = k / 32;
    uint16_t db, mb;
    std::memcpy(&db, d + 2 * (kb * 32 + r), 2);
    std::memcpy(&mb, d + 512 + 2 * (kb * 32 + r), 2);
    const size_t p0 = (r / 16) * 4096 + k * 16 + (r % 16);
    const uint8_t byte = d[1024 + (p0 >> 1)];
    const uint8_t nib = (p0 & 1) ? byte >> 4 : byte & 0xF;
    return nib * bf16_to_f32(db) + bf16_to_f32(mb);
}

/// The same value from a Q4_1 source block, with the fp16 scales rounded to bf16 exactly
/// as the packer writes them into a 5120-byte chunk. (The 6144 f32 law is bit-exact; the
/// q4_1 law the GGUF-direct MoE uses is the one with this one rounding.)
float source_q4_1_value(const Blocks& b, size_t row, size_t col) {
    const size_t nb = b.cols / 32;
    const uint8_t* p = b.bytes.data() + (row * nb + col / 32) * b.blk;
    uint16_t dh, mh;
    std::memcpy(&dh, p, 2);
    std::memcpy(&mh, p + 2, 2);
    const float d = bf16_to_f32(f32_to_bf16(fp16_to_f32(dh)));
    const float m = bf16_to_f32(f32_to_bf16(fp16_to_f32(mh)));
    const uint8_t byte = p[4 + (col % 16)];
    const uint8_t q = (col % 32 < 16) ? (byte & 0xF) : (byte >> 4);
    return d * q + m;
}

// A container in memory: just the tensors' bytes, as a .q4nx stores small weights (bf16 or f32).
class MemFile : public WeightFile {
public:
    std::map<std::string, std::vector<uint8_t>> t;
    bool has(const std::string& name) const override { return t.count(name) != 0; }
    const uint8_t* raw(const std::string& name, size_t* nbytes) const override {
        const auto& b = t.at(name);
        if (nbytes) *nbytes = b.size();
        return b.data();
    }
    void embed_row(const std::string&, size_t, size_t, float*) const override { throw std::logic_error("unused"); }
    std::vector<float> bf16(const std::string&) const override { throw std::logic_error("unused"); }
    std::vector<float> f32(const std::string&) const override { throw std::logic_error("unused"); }
    const TensorMeta& meta(const std::string&) const override { throw std::logic_error("unused"); }
    void bf16_row(const std::string&, size_t, size_t, float*) const override { throw std::logic_error("unused"); }
    void drop_pages() override {}
};

template <typename T>
std::vector<uint8_t> bytes_of(const std::vector<T>& v) {
    std::vector<uint8_t> b(v.size() * sizeof(T));
    std::memcpy(b.data(), v.data(), b.size());
    return b;
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
        GgufFile::gguf_name("model.layers.2.post_attention_layernorm.weight") != "blk.2.ffn_norm.weight" ||
        GgufFile::gguf_name("model.layers.2.pre_feedforward_layernorm.weight") != "blk.2.ffn_norm.weight" ||
        GgufFile::gguf_name("model.layers.2.post_feedforward_layernorm.weight") != "blk.2.post_ffw_norm.weight" ||
        GgufFile::gguf_name("model.norm.weight") != "output_norm.weight" ||
        GgufFile::gguf_name("model.layers.2.post_attention_layernorm.weight", true) != "blk.2.post_attention_norm.weight" ||
        GgufFile::gguf_name("model.layers.2.pre_feedforward_layernorm.weight", true) != "blk.2.ffn_norm.weight" ||
        GgufFile::gguf_name("model.layers.2.post_feedforward_layernorm.weight", true) != "blk.2.post_ffw_norm.weight" ||
        GgufFile::gguf_name("model.layers.2.input_layernorm.weight", true) != "blk.2.attn_norm.weight") {
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

    // Q4_0 has a two-byte header (d only), unlike Q4_1's d+m header. Exercise
    // the same pack path separately so its code offset cannot drift to +4.
    {
        const Blocks mm0 = make_blocks(64, 512, "Q4_0", rng);
        GgufFile g0(write_gguf(dir, mm0, emb, norm));
        std::vector<uint8_t> p0(static_cast<size_t>(op.nch) * 6144, 0);
        pools::apply(op, g0, 0, p0.data(), p0.size(), 6144);
        size_t bad0 = 0;
        for (size_t c = 0; c < op.nch; ++c) {
            const uint8_t* d = p0.data() + c * 6144;
            const size_t per_band = 4;
            const size_t rows0 = 64 * (c / per_band) + 32 * (c % 2);
            const size_t cols0 = 256 * ((c % per_band) / 2);
            for (size_t r = 0; r < 32; ++r)
                for (size_t k = 0; k < 256; ++k) {
                    const size_t kb = k / 32;
                    const float df = *reinterpret_cast<const float*>(d + 4 * (kb * 32 + r));
                    const size_t p = (r / 16) * 4096 + k * 16 + (r % 16);
                    const uint8_t byte = d[2048 + (p >> 1)];
                    const uint8_t nib = (p & 1) ? byte >> 4 : byte & 0xF;
                    if (f32_bits(nib * df - 8.f * df) != f32_bits(mm0.values[(rows0 + r) * mm0.cols + cols0 + k]))
                        ++bad0;
                }
        }
        std::printf("%s std_perm_gguf Q4_0 (%zu values differ)\n", bad0 ? "FAIL" : "ok", bad0);
        if (bad0) return 1;
    }

    // ---- expert_stripes_gguf / expert_down_gguf: 3-D [in, n, experts] sources
    {
        const unsigned hid = 512, ff = 256, E = 3;
        const unsigned ns = ff / 128, ncol = hid / 256;
        Blocks up = make_blocks(ff * E, hid, "Q4_1", rng);      // dims [hid, ff, E]
        Blocks gate = make_blocks(ff * E, hid, "Q4_1", rng);
        Blocks down = make_blocks(hid * E, ff, "Q4_1", rng);    // dims [ff, hid, E]
        GgufFile gm(write_gguf_tensors(dir, "test-moe.gguf",
            {{"blk.0.ffn_up_exps.weight", {hid, ff, E}, 3, up.bytes},
             {"blk.0.ffn_gate_exps.weight", {hid, ff, E}, 3, gate.bytes},
             {"blk.0.ffn_down_exps.weight", {ff, hid, E}, 3, down.bytes}}));

        // the container's MoE names (model.layer.{l}., not the dense model.layers.{l}.)
        // must resolve to llama.cpp's *_exps / *_shexp tensors
        if (GgufFile::gguf_name("model.layer.0.mlp.up_exps_proj.weight") != "blk.0.ffn_up_exps.weight" ||
            GgufFile::gguf_name("model.layer.0.mlp.share_down_exps_proj.weight") != "blk.0.ffn_down_shexp.weight" ||
            GgufFile::gguf_name("model.layer.0.moe_router.weight") != "blk.0.ffn_gate_inp.weight" ||
            GgufFile::gguf_name("model.layer.0.shared_expert_gate.weight") != "blk.0.ffn_gate_inp_shexp.weight") {
            std::printf("FAIL MoE name map\n");
            return 1;
        }

        const size_t stripe = static_cast<size_t>(128) * hid / 8192 * 5120;
        const size_t B = static_cast<size_t>(hid) * ff / 8192 * 5120;
        PackOp es;
        es.op = "expert_stripes_gguf";
        es.up = "model.layer.0.mlp.up_exps_proj.weight";
        es.gate = "model.layer.0.mlp.gate_exps_proj.weight";
        es.in_dim = hid;
        es.experts = E;
        es.stripes = ns;
        es.stripe_bytes = stripe;
        std::vector<uint8_t> sp(E * 2 * ns * stripe, 0);
        pools::apply(es, gm, 0, sp.data(), sp.size(), 5120);
        size_t bads = 0;
        for (unsigned e = 0; e < E; ++e)
            for (unsigned k = 0; k < ns; ++k) {
                const uint8_t* ud = sp.data() + (2 * ns * e + 2 * k) * stripe;
                const uint8_t* gd = ud + stripe;
                for (size_t c = 0; c < stripe / 5120; ++c) {
                    const size_t tp = ncol * (c % 4) + c / 4;
                    const size_t row0 = e * ff + k * 128 + 32 * (tp / ncol);
                    const size_t col0 = 256 * (tp % ncol);
                    for (size_t r = 0; r < 32; ++r)
                        for (size_t kk = 0; kk < 256; ++kk) {
                            const size_t row = row0 + r, col = col0 + kk;
                            if (f32_bits(pool_q4_1_value(ud + c * 5120, r, kk)) !=
                                f32_bits(source_q4_1_value(up, row, col))) ++bads;
                            if (f32_bits(pool_q4_1_value(gd + c * 5120, r, kk)) !=
                                f32_bits(source_q4_1_value(gate, row, col))) ++bads;
                        }
                }
            }
        std::printf("%s expert_stripes_gguf (%zu values differ)\n", bads ? "FAIL" : "ok", bads);
        if (bads) return 1;

        PackOp ed;
        ed.op = "expert_down_gguf";
        ed.tensor = "model.layer.0.mlp.down_exps_proj.weight";
        ed.experts = E;
        ed.expert_bytes = B;
        std::vector<uint8_t> dp(E * B, 0);
        pools::apply(ed, gm, 0, dp.data(), dp.size(), 5120);
        size_t badd = 0;
        const size_t nchd = B / 5120;
        for (unsigned e = 0; e < E; ++e)
            for (size_t c = 0; c < nchd; ++c) {
                const size_t rt = 4 * (c / 8) + (c % 4), cg = (c / 4) % 2;
                const size_t fc = 2 * rt + cg;
                const size_t row0 = e * hid + 32 * fc;    // ncol == 1 (ff == 256)
                for (size_t r = 0; r < 32; ++r)
                    for (size_t kk = 0; kk < 256; ++kk)
                        if (f32_bits(pool_q4_1_value(dp.data() + e * B + c * 5120, r, kk)) !=
                            f32_bits(source_q4_1_value(down, row0 + r, kk))) ++badd;
            }
        std::printf("%s expert_down_gguf (%zu values differ)\n", badd ? "FAIL" : "ok", badd);
        if (badd) return 1;
    }

    // ---- embed_row over a 3-D expert tensor: a flat row past the first expert
    {
        const unsigned cols = 256, n = 4, E = 2;
        Blocks ex = make_blocks(n * E, cols, "Q8_0", rng);
        GgufFile ge(write_gguf_tensors(dir, "test-exps.gguf",
            {{"blk.0.ffn_up_exps.weight", {cols, n, E}, 8, ex.bytes}}));
        std::vector<float> row(cols);
        ge.embed_row("model.layer.0.mlp.up_exps_proj.weight", n + 2, cols, row.data());
        size_t bad = 0;
        for (unsigned i = 0; i < cols; ++i)
            if (f32_bits(row[i]) != f32_bits(ex.values[(n + 2) * cols + i])) ++bad;
        std::printf("%s embed_row 3-D expert (row past the first expert)\n", bad ? "FAIL" : "ok");
        if (bad) return 1;
    }

    // ---- transpose from a GGUF f16 tensor (the MoE router): decode, don't reinterpret
    {
        const unsigned E = 4, hid = 64;
        std::vector<uint8_t> f16(static_cast<size_t>(E) * hid * 2);
        std::vector<float> vals(static_cast<size_t>(E) * hid);
        for (size_t i = 0; i < vals.size(); ++i) {
            const float v = -2.f + 4.f * static_cast<float>(rng() % 1000) / 1000.f;
            const uint16_t h = f32_to_f16(v);    // truncating, so the decode is what we compare
            std::memcpy(f16.data() + 2 * i, &h, 2);
            vals[i] = fp16_to_f32(h);
        }
        GgufFile gr(write_gguf_tensors(dir, "test-router.gguf",
            {{"blk.0.ffn_gate_inp.weight", {hid, E}, 1, f16}}));   // [E, hid] rows
        PackOp rt;
        rt.op = "transpose";
        rt.tensor = "model.layer.0.moe_router.weight";
        rt.rows = E;
        rt.cols = hid;
        rt.elem = 2;
        std::vector<uint8_t> out(static_cast<size_t>(E) * hid * 2, 0);
        pools::apply(rt, gr, 0, out.data(), out.size(), 5120);
        bool bad = false;
        for (unsigned r = 0; r < E && !bad; ++r)
            for (unsigned c = 0; c < hid && !bad; ++c) {
                uint16_t got;
                std::memcpy(&got, out.data() + 2 * (c * E + r), 2);
                if (f32_bits(bf16_to_f32(got)) !=
                    f32_bits(bf16_to_f32(f32_to_bf16(vals[r * hid + c])))) {
                    std::printf("FAIL transpose f16 router [%u,%u]\n", r, c);
                    bad = true;
                }
            }
        std::printf("%s transpose GGUF f16 (router)\n", bad ? "FAIL" : "ok");
        if (bad) return 1;
    }

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

    // ---- lmhead_q8 from a raw GGUF Q8_0 head: 128-row bands, K tiles, then four 32-row
    // quarters; each fp16 scale widened exactly and rounded to bf16, codes copied. The second
    // pass gives every block a subnormal scale (exponent 0, mantissa 1..1023), which a pure
    // Q8_0 head quantized from BF16 has wherever a block's amax < 127 * 2^-14.
    {
        const Blocks head = make_blocks(256, 512, "Q8_0", rng);
        Blocks sub = head;
        for (size_t b = 0; b < sub.bytes.size() / 34; ++b) {
            const uint16_t h = static_cast<uint16_t>(1 + (b * 37) % 0x3FF);
            std::memcpy(sub.bytes.data() + b * 34, &h, 2);
        }
        PackOp h;
        h.op = "lmhead_q8";
        h.tensor = "model.embed_tokens.weight";
        h.in_dim = 512;
        h.chunk_bytes = 8704;
        for (const auto& [label, src] : {std::pair<const char*, const Blocks*>{"normal", &head},
                                         std::pair<const char*, const Blocks*>{"subnormal", &sub}}) {
            GgufFile q8(write_gguf_tensors(dir, std::string("test-q8-head-") + label + ".gguf",
                                           {{"token_embd.weight", {512, 256}, 8, src->bytes}}));
            std::vector<uint8_t> hp(16 * 8704, 0);
            pools::apply(h, q8, 0, hp.data(), hp.size(), 6144);
            size_t bad_scale = 0, bad_code = 0;
            for (size_t band = 0; band < 2; ++band)
                for (size_t kt = 0; kt < 2; ++kt)
                    for (size_t quarter = 0; quarter < 4; ++quarter)
                        for (size_t row = 0; row < 32; ++row)
                            for (size_t kb = 0; kb < 8; ++kb) {
                                const size_t logical = band * 128 + quarter * 32 + row;
                                const uint8_t* raw = src->bytes.data() + (logical * 16 + kt * 8 + kb) * 34;
                                uint16_t half;
                                std::memcpy(&half, raw, 2);
                                const float value = fp16_to_f32(half);
                                uint32_t bits;
                                std::memcpy(&bits, &value, 4);
                                const uint16_t want = static_cast<uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
                                const uint8_t* chunk = hp.data() + ((band * 2 + kt) * 4 + quarter) * 8704;
                                uint16_t got;
                                std::memcpy(&got, chunk + 2 * (kb * 32 + row), 2);
                                bad_scale += got != want;
                                for (size_t k = 0; k < 32; ++k) {
                                    const size_t off = (row / 16) * 4096 + kb * 512 + k * 16 + row % 16;
                                    bad_code += chunk[512 + off] != raw[2 + k];
                                }
                            }
            std::printf("%s lmhead_q8 GGUF Q8_0, %s fp16 scales (%zu scales, %zu codes differ)\n",
                        bad_scale || bad_code ? "FAIL" : "ok   ", label, bad_scale, bad_code);
            if (bad_scale || bad_code) return 1;
        }
    }

    // ---- pack_norm with div_running's product (Gemma 4's layer_output_scale). The same case,
    // and the same expected bits, as test_gemma4.py's test_put_div_running_matches_pools_cpp.
    {
        const std::vector<uint16_t> norm16 = {0x3F80, 0x3FC0, 0xC030, 0x3DCD, 0x4049, 0x3F81};
        const std::vector<uint16_t> want = {0x4099, 0x40E5, 0xC152, 0x3EF4, 0x4170, 0x409A};
        MemFile mf;
        mf.t["norm"] = bytes_of(norm16);
        mf.t["scale.0"] = bytes_of(std::vector<uint16_t>{0x3F33});   // bf16
        mf.t["scale.1"] = bytes_of(std::vector<float>{0.3f});         // f32
        double div = 1.0;
        for (int j = 0; j < 2; ++j) div *= pools::scalar_value(mf, "scale." + std::to_string(j));
        std::vector<uint16_t> got(norm16.size());
        pools::pack_norm(mf, "norm", got.size() * 2, reinterpret_cast<uint8_t*>(got.data()), div);
        if (got != want) {
            for (size_t i = 0; i < got.size(); ++i)
                std::printf("FAIL pack_norm div[%zu]: got %04x want %04x\n", i, got[i], want[i]);
            return 1;
        }
        std::printf("ok    pack_norm divided by the running scale product\n");
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

    // ---- embed_row (Q6_K, the qh-advance regression: every super-block half)
    {
        BlocksQ6K q6 = make_blocks_q6k(4, 512, rng);      // 4 rows x 2 super-blocks
        GgufFile g6(write_gguf_q6k(dir, q6));
        std::vector<float> row(512);
        bool bad6 = false;
        for (unsigned r = 0; r < 4 && !bad6; ++r) {
            g6.embed_row("blk.0.attn_q.weight", r, 512, row.data());
            for (size_t i = 0; i < 512; ++i)
                if (f32_bits(row[i]) != f32_bits(q6.values[static_cast<size_t>(r) * 512 + i])) {
                    std::printf("FAIL embed_row(Q6_K)[r%u i%zu]: got %.9g want %.9g\n", r, i, row[i],
                                q6.values[static_cast<size_t>(r) * 512 + i]);
                    bad6 = true;
                    break;
                }
        }
        std::printf("%s embed_row Q6_K (both super-block halves bit-exact)\n", bad6 ? "FAIL" : "ok");
        if (bad6) return 1;
    }

    // ---- fl_q4: the generic GGUF expert pack, with a padded destination slot
    // (LFM2.5-8B-A1B's 1792 real rows into a 2048-row band-aligned slot). The
    // op reads only `rows` real rows from each expert; the rest of the slot stays
    // zero, so a padded band contributes nothing to the GEMV.
    {
        const unsigned cols = 256, real = 128, padded = 192, E = 2;  // real 2 bands, padded 3
        Blocks ex = make_blocks(real * E, cols, "Q4_0", rng);
        GgufFile gf(write_gguf_tensors(dir, "test-flq4.gguf",
            {{"blk.0.ffn_gate_exps.weight", {cols, real, E}, 2, ex.bytes}}));
        // one 2-D reference file per expert (its own rows, so std_perm_gguf can
        // read them without a chunk0 offset into a 3-D tensor)
        std::vector<std::unique_ptr<GgufFile>> refs;
        for (unsigned e = 0; e < E; ++e) {
            const size_t row_bytes = static_cast<size_t>(cols / 32) * 18;   // Q4_0: 18 B per 32-value block
            std::vector<uint8_t> bytes(ex.bytes.begin() + static_cast<size_t>(e) * real * row_bytes,
                                       ex.bytes.begin() + static_cast<size_t>(e + 1) * real * row_bytes);
            refs.emplace_back(new GgufFile(write_gguf_tensors(dir, "test-flq4-ref" + std::to_string(e) + ".gguf",
                {{"blk.0.ffn_gate_exps.weight", {cols, real}, 2, bytes}})));
        }
        PackOp op;
        op.op = "fl_q4";
        op.tensor = "blk.0.ffn_gate_exps.weight";
        op.rows = real;
        op.in_dim = cols;
        op.experts = E;
        op.n_total = real;                 // source stride: expert e starts at row e*real
        const size_t band = 2 * (cols / 256) * 5120;   // one 64-row band
        op.expert_bytes = (padded / 64) * band;
        // the real packer zeroes the whole pool before apply (pools.cpp), so
        // the padded lanes start at zero and stay zero
        std::vector<uint8_t> out(E * op.expert_bytes, 0);
        pools::apply(op, gf, 0, out.data(), out.size(), 5120);

        // fl_q4's byte law for the real rows IS std_perm_gguf's (both use
        // gguf_chunk's std_perm raster). Pack each expert's rows through the
        // reference op with the matching chunk0 and compare byte-for-byte.
        const size_t nch = static_cast<size_t>(real) * cols / (32 * 256);
        size_t bad = 0;
        for (unsigned e = 0; e < E; ++e) {
            PackOp ref;
            ref.op = "std_perm_gguf";
            ref.tensor = op.tensor;
            ref.in_dim = cols;
            ref.nch = nch;
            std::vector<uint8_t> plain(nch * 5120, 0);
            pools::apply(ref, *refs[e], 0, plain.data(), plain.size(), 5120);

            const uint8_t* dst = out.data() + e * op.expert_bytes;
            for (size_t c = 0; c < nch; ++c) {
                const uint8_t* want = plain.data() + c * 5120;
                const size_t b = c / 2, half = c % 2;
                const uint8_t* got = dst + b * band + half * 5120;
                if (std::memcmp(got, want, 5120) != 0) ++bad;
            }
            // the pad bands (rows real..padded-1) stay zero bytes.
            for (size_t i = (real / 64) * band; i < op.expert_bytes; ++i)
                if (dst[i] != 0) { ++bad; break; }
        }
        std::printf("%s fl_q4 GGUF expert pack (padded slot, exact codes + zero pad)\n", bad ? "FAIL" : "ok");
        if (bad) return 1;
    }

    // ---- Q5_0 / Q5_1: the 5-bit families Unsloth mixes into UD-Q4_K_XL
    // (Q4_K on the gates/ups, Q5_1 on the MoE downs). A synthetic block checks
    // the reader's interleaved fifth-bit order; a second GGUF carrying the same
    // logical values as Q4_1 checks that the requant pack path sees those values.
    for (unsigned ty : {6u, 7u}) {
        const std::string label = ty == 6 ? "Q5_0" : "Q5_1";
        BlocksQ5 q5 = make_blocks_q5(4, 256, ty, rng);
        GgufFile g5e(write_gguf_tensors(dir, "test-" + label + "-emb.gguf",
            {{"token_embd.weight", {256, 4}, ty, q5.bytes}}));
        std::vector<float> row(256);
        bool bad = false;
        for (unsigned r = 0; r < 4 && !bad; ++r) {
            g5e.embed_row("token_embd.weight", r, 256, row.data());
            for (unsigned i = 0; i < 256; ++i)
                if (f32_bits(row[i]) != f32_bits(q5.values[static_cast<size_t>(r) * 256 + i])) {
                    std::printf("FAIL embed_row(%s)[r%u i%u]: got %.9g want %.9g\n",
                                label.c_str(), r, i, row[i], q5.values[static_cast<size_t>(r) * 256 + i]);
                    bad = true;
                    break;
                }
        }
        std::printf("%s embed_row %s (fifth bits, interleaved order)\n", bad ? "FAIL" : "ok", label.c_str());
        if (bad) return 1;

        const unsigned rows = 64, cols = 512, nb = cols / 32;
        const uint16_t dh = f32_to_f16(1.f), mh = f32_to_f16(ty == 6 ? -16.f : 0.f);
        const size_t b5 = ty == 6 ? 22 : 24;
        std::vector<uint8_t> bytes5(static_cast<size_t>(rows) * nb * b5, 0);
        std::vector<uint8_t> bytes41(static_cast<size_t>(rows) * nb * 20, 0);
        for (unsigned r = 0; r < rows; ++r) {
            for (unsigned b = 0; b < nb; ++b) {
                uint8_t* p5 = bytes5.data() + (static_cast<size_t>(r) * nb + b) * b5;
                uint8_t* p4 = bytes41.data() + (static_cast<size_t>(r) * nb + b) * 20;
                std::memcpy(p5, &dh, 2);
                if (ty == 7) std::memcpy(p5 + 2, &mh, 2);
                std::memcpy(p4, &dh, 2);
                std::memcpy(p4 + 2, &mh, 2);
                for (unsigned j = 0; j < 16; ++j) {
                    const uint8_t lo = static_cast<uint8_t>((2 * j) % 16);
                    const uint8_t hi = static_cast<uint8_t>((2 * j + 1) % 16);
                    p5[(ty == 7 ? 4 : 2) + j] = static_cast<uint8_t>(lo | (hi << 4));
                    p4[4 + j] = static_cast<uint8_t>(j | (j << 4));
                }
                // qh = 0: the fifth bits add nothing, so both files hold 0..15.
            }
        }
        GgufFile g5p(write_gguf_tensors(dir, "test-" + label + "-pack.gguf",
            {{"blk.0.attn_q.weight", {cols, rows}, ty, bytes5}}));
        GgufFile g41(write_gguf_tensors(dir, "test-" + label + "-ref.gguf",
            {{"blk.0.attn_q.weight", {cols, rows}, 3, bytes41}}));
        PackOp op5;
        op5.op = "std_perm_gguf";
        op5.tensor = "blk.0.attn_q.weight";
        op5.in_dim = cols;
        op5.nch = 64ull * cols / (32 * 256);
        std::vector<uint8_t> p5(static_cast<size_t>(op5.nch) * 6144, 0), p41(p5.size(), 0);
        pools::apply(op5, g5p, 0, p5.data(), p5.size(), 6144);
        pools::apply(op5, g41, 0, p41.data(), p41.size(), 6144);
        if (p5 != p41) {
            size_t at = 0;
            while (at < p5.size() && p5[at] == p41[at]) ++at;
            std::printf("FAIL std_perm_gguf %s requant: byte %zu %02x != %02x\n",
                        label.c_str(), at, p5[at], p41[at]);
            return 1;
        }
        std::printf("ok    std_perm_gguf %s requant (same pool bytes as Q4_1)\n", label.c_str());
    }

    // ---- split GGUF: two shards, installed as model.gguf + numbered sibling,
    // and the original Name-00001-of-000NN.gguf spelling. Both open as ONE file.
    {
        Blocks s1 = make_blocks(2, 64, "Q8_0", rng), s2 = make_blocks(2, 64, "Q8_0", rng);
        SynthTensor a{"blk.0.attn_q.weight", {64, 2}, 8, s1.bytes};
        SynthTensor b{"blk.1.attn_q.weight", {64, 2}, 8, s2.bytes};
        write_gguf_tensors(dir, "model.gguf", {a}, 1, 2);
        write_gguf_tensors(dir, "model.gguf-00002-of-00002", {b}, 2, 2);
        GgufFile gsplit((dir / "model.gguf").string());
        if (gsplit.kv_str("split.count") != "2" || gsplit.tensor_names().size() != 2 ||
            gsplit.tensor("blk.0.attn_q.weight").shard != 0 ||
            gsplit.tensor("blk.1.attn_q.weight").shard != 1) {
            std::printf("FAIL split GGUF index (metadata / tensor shards)\n");
            return 1;
        }
        std::vector<float> row(64);
        bool bad = false;
        for (size_t t = 0; t < 2 && !bad; ++t) {
            const Blocks& want = t ? s2 : s1;
            const char* name = t ? "blk.1.attn_q.weight" : "blk.0.attn_q.weight";
            for (unsigned r = 0; r < 2 && !bad; ++r) {
                gsplit.embed_row(name, r, 64, row.data());
                for (unsigned i = 0; i < 64; ++i)
                    if (f32_bits(row[i]) != f32_bits(want.values[static_cast<size_t>(r) * 64 + i])) {
                        std::printf("FAIL split embed_row(%s)[r%u i%u]\n", name, r, i);
                        bad = true;
                        break;
                    }
            }
        }
        std::printf("%s split GGUF (installed model.gguf + sibling; rows from both shards)\n",
                    bad ? "FAIL" : "ok");
        if (bad) return 1;

        write_gguf_tensors(dir, "Source-00001-of-00002.gguf", {a}, 1, 2);
        write_gguf_tensors(dir, "Source-00002-of-00002.gguf", {b}, 2, 2);
        GgufFile graw((dir / "Source-00001-of-00002.gguf").string());
        bool badraw = false;
        for (size_t t = 0; t < 2 && !badraw; ++t) {
            const Blocks& want = t ? s2 : s1;
            const char* name = t ? "blk.1.attn_q.weight" : "blk.0.attn_q.weight";
            for (size_t r = 0; r < 2 && !badraw; ++r) {
                graw.embed_row(name, r, 64, row.data());
                for (size_t i = 0; i < 64; ++i)
                    if (f32_bits(row[i]) != f32_bits(want.values[r * 64 + i])) {
                        badraw = true;
                        break;
                    }
            }
        }
        std::printf("%s split GGUF (source Name-NNNNN-of-NNNNN spelling)\n", badraw ? "FAIL" : "ok");
        if (badraw) return 1;

        // Unsloth numbers split.no from ZERO. Same file names, same order; only
        // the metadata convention differs, and the reader must accept it.
        write_gguf_tensors(dir, "Unsloth-00001-of-00002.gguf", {a}, 0, 2);
        write_gguf_tensors(dir, "Unsloth-00002-of-00002.gguf", {b}, 1, 2);
        GgufFile guns((dir / "Unsloth-00001-of-00002.gguf").string());
        bool badzero = guns.tensor_names().size() != 2;
        for (size_t t = 0; t < 2 && !badzero; ++t) {
            const Blocks& want = t ? s2 : s1;
            const char* name = t ? "blk.1.attn_q.weight" : "blk.0.attn_q.weight";
            guns.embed_row(name, 0, 64, row.data());
            for (unsigned i = 0; i < 64; ++i)
                if (f32_bits(row[i]) != f32_bits(want.values[i])) { badzero = true; break; }
        }
        std::printf("%s split GGUF (Unsloth 0-based split.no)\n", badzero ? "FAIL" : "ok");
        if (badzero) return 1;
    }

    std::printf("GGUF-PACK OK\n");
    return 0;
}
