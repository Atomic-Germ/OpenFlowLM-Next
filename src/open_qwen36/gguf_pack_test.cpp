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

/// Q5_K tensor blocks and the reference dequant (llama.cpp's q5_k: a
/// super-block of 256 values in 176 B -- fp16 d, fp16 dmin, 12 B of 6-bit
/// scales, 32 B of the fifth bit, 128 B of 4-bit codes).
struct BlocksQ5K {
    unsigned rows, cols;
    std::vector<uint8_t> bytes;          // rows * (cols/256) * 176
    std::vector<float> values;           // rows * cols
};

BlocksQ5K make_blocks_q5k(unsigned rows, unsigned cols, std::mt19937_64& rng) {
    const unsigned nbs = cols / 256;
    BlocksQ5K b;
    b.rows = rows;
    b.cols = cols;
    b.bytes.assign(static_cast<size_t>(rows) * nbs * 176, 0);
    b.values.resize(static_cast<size_t>(rows) * cols);
    const uint16_t dh = f32_to_f16(0.5f);            // exactly representable
    const uint16_t mh = f32_to_f16(0.01f);           // exactly representable
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned sb = 0; sb < nbs; ++sb) {
            uint8_t* p = b.bytes.data() + (static_cast<size_t>(r) * nbs + sb) * 176;
            std::memcpy(p, &dh, 2);
            std::memcpy(p + 2, &mh, 2);
            // the 4-bit codes: per 64-value half, the low nibbles of 32 bytes then the high
            for (int h = 0; h < 4; ++h)
                for (int l = 0; l < 32; ++l) {
                    const uint8_t q = static_cast<uint8_t>(rng() % 16);
                    p[48 + h * 32 + l] |= (h & 1) ? static_cast<uint8_t>(q << 4) : q;
                }
            // the fifth bit: byte (v % 32) holds the bits of the values v, v+32, v+64, ...
            for (int v = 0; v < 256; ++v)
                if (rng() % 2) p[16 + v % 32] |= static_cast<uint8_t>(1u << (v / 32));
            // the scales: eight 6-bit pairs, packed as the reader unpacks them
            // (the first four pairs in the low six bits of bytes 0..3 (scale) and 4..7
            // (min); the last four split a byte's high nibbles and top two bits)
            for (int j = 0; j < 8; ++j) {
                const uint8_t sc = static_cast<uint8_t>(rng() % 64), m = static_cast<uint8_t>(rng() % 64);
                if (j < 4) {
                    p[4 + j] |= static_cast<uint8_t>(sc & 63);                  // the scales array is p+4
                    p[8 + j] |= static_cast<uint8_t>(m & 63);
                } else {
                    const int j4 = j - 4;
                    p[12 + j4] |= static_cast<uint8_t>(sc & 0x0F);             // the scale's low four bits
                    p[4 + j4] |= static_cast<uint8_t>((sc >> 4) & 0x03) << 6;  // its top two bits
                    p[12 + j4] |= static_cast<uint8_t>((m & 0x0F) << 4);      // the min's low four bits
                    p[8 + j4] |= static_cast<uint8_t>((m >> 4) & 0x03) << 6;   // its top two bits
                }
            }
            // the reference, from the bytes that hit the file
            float* y = b.values.data() + static_cast<size_t>(r) * cols + sb * 256;
            for (int v = 0; v < 256; ++v) {
                const int h = (v / 64) * 64;
                const uint8_t byte = p[48 + h / 2 + (v % 32)];
                const uint8_t nib = (v % 64) < 32 ? (byte & 0xF) : (byte >> 4);
                const uint8_t fifth = (p[16 + v % 32] >> (v / 32)) & 1;
                const int j = v / 32;
                uint8_t sc, m;
                if (j < 4) {
                    sc = p[4 + j] & 63;
                    m = p[8 + j] & 63;
                } else {
                    const int j4 = j - 4;
                    sc = static_cast<uint8_t>((p[12 + j4] & 0x0F) | ((p[4 + j4] >> 6) << 4));
                    m = static_cast<uint8_t>(((p[12 + j4] >> 4) & 0x0F) | ((p[8 + j4] >> 6) << 4));
                }
                // the exact fp16 values that hit the file (f32_to_f16 truncates)
                y[v] = fp16_to_f32(dh) * sc * (nib | (fifth << 4)) - fp16_to_f32(mh) * m;
            }
        }
    }
    return b;
}

/// A one-tensor GGUF carrying a single Q5_K tensor (rows x 256).
std::string write_gguf_q5k(const fs::path& dir, const BlocksQ5K& b) {
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
    put_u32(f, 13);                                   // GGML_TYPE_Q5_K
    put_u64(f, 0);
    const size_t data_off = align32(f.size());
    while (f.size() < data_off) f.push_back(0);
    f.insert(f.end(), b.bytes.begin(), b.bytes.end());
    const fs::path out = dir / "test-q5k.gguf";
    std::ofstream of(out, std::ios::binary);
    of.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(f.size()));
    return out.string();
}

/// IQ4_XS tensor blocks and the reference dequant (llama.cpp's iq4_xs: a
/// super-block of 256 values in 136 B -- fp16 d, 2 B of 2-bit scale highs, 4 B
/// of 4-bit scale lows, 128 B of 4-bit codes; eight groups of 32 values, each
/// with a signed 6-bit scale, and the same sixteen-entry codebook as iq4_nl).
struct BlocksIQ4XS {
    unsigned rows, cols;
    std::vector<uint8_t> bytes;          // rows * (cols/256) * 136
    std::vector<float> values;           // rows * cols
};

BlocksIQ4XS make_blocks_iq4xs(unsigned rows, unsigned cols, std::mt19937_64& rng) {
    static const int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                  1, 13, 25, 38, 53, 69, 89, 113};
    const unsigned nbs = cols / 256;
    BlocksIQ4XS b;
    b.rows = rows;
    b.cols = cols;
    b.bytes.assign(static_cast<size_t>(rows) * nbs * 136, 0);
    b.values.resize(static_cast<size_t>(rows) * cols);
    const uint16_t dh = f32_to_f16(0.25f);            // exactly representable
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned sb = 0; sb < nbs; ++sb) {
            uint8_t* p = b.bytes.data() + (static_cast<size_t>(r) * nbs + sb) * 136;
            std::memcpy(p, &dh, 2);                    // d
            // the codes: 16 bytes per group of 32 values
            for (int g = 0; g < 8; ++g)
                for (int j = 0; j < 16; ++j) p[8 + g * 16 + j] = static_cast<uint8_t>(rng() % 256);
            // the scales: eight signed 6-bit values, the low four bits in scales_l
            // (two per byte) and the high two bits in scales_h (two bits per group)
            uint8_t sc[8];
            for (int g = 0; g < 8; ++g) sc[g] = static_cast<uint8_t>(rng() % 64);
            for (int g = 0; g < 8; ++g) {
                const uint8_t lo = sc[g] & 0xF, hi = (sc[g] >> 4) & 0x3;
                if (g & 1) p[4 + g / 2] |= static_cast<uint8_t>(lo << 4);
                else p[4 + g / 2] |= lo;
                p[2] |= static_cast<uint8_t>(hi << (2 * g));
            }
            // the reference, from the bytes that hit the file
            float* y = b.values.data() + static_cast<size_t>(r) * cols + sb * 256;
            for (int g = 0; g < 8; ++g) {
                const uint8_t lo = (g & 1) ? static_cast<uint8_t>(p[4 + g / 2] >> 4)
                                           : static_cast<uint8_t>(p[4 + g / 2] & 0xF);
                const uint8_t hi = static_cast<uint8_t>((p[2] >> (2 * g)) & 0x3);
                const float dl = 0.25f * static_cast<float>(static_cast<int>(lo | (hi << 4)) - 32);
                for (int j = 0; j < 16; ++j) {
                    const uint8_t q = p[8 + g * 16 + j];
                    y[g * 32 + j] = dl * kv[q & 0xF];
                    y[g * 32 + 16 + j] = dl * kv[q >> 4];
                }
            }
        }
    }
    return b;
}

/// A one-tensor GGUF carrying a single IQ4_XS tensor (rows x 256).
std::string write_gguf_iq4xs(const fs::path& dir, const BlocksIQ4XS& b) {
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
    put_u32(f, 23);                                   // GGML_TYPE_IQ4_XS
    put_u64(f, 0);                                    // RELATIVE offset (0: data sits at the data base)
    const size_t data_off = align32(f.size());        // == the GGUF's data base
    while (f.size() < data_off) f.push_back(0);
    f.insert(f.end(), b.bytes.begin(), b.bytes.end());
    const fs::path out = dir / "test-iq4xs.gguf";
    std::ofstream of(out, std::ios::binary);
    of.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(f.size()));
    return out.string();
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

    // ---- embed_row (IQ4_XS: the 6-bit group scales and the codebook)
    {
        BlocksIQ4XS iq = make_blocks_iq4xs(4, 512, rng);      // 4 rows x 2 super-blocks
        GgufFile giq(write_gguf_iq4xs(dir, iq));
        std::vector<float> row(512);
        bool badiq = false;
        for (unsigned r = 0; r < 4 && !badiq; ++r) {
            giq.embed_row("blk.0.attn_q.weight", r, 512, row.data());
            for (size_t i = 0; i < 512; ++i)
                if (f32_bits(row[i]) != f32_bits(iq.values[static_cast<size_t>(r) * 512 + i])) {
                    std::printf("FAIL embed_row(IQ4_XS)[r%u i%zu]: got %.9g want %.9g\n", r, i, row[i],
                                iq.values[static_cast<size_t>(r) * 512 + i]);
                    badiq = true;
                    break;
                }
        }
        std::printf("%s embed_row IQ4_XS (group scales and codebook bit-exact)\n", badiq ? "FAIL" : "ok");
        if (badiq) return 1;
    }

    // ---- embed_row (Q5_K: the sixth-bit codes and the five-bit values)
    {
        BlocksQ5K q5 = make_blocks_q5k(4, 512, rng);      // 4 rows x 2 super-blocks
        GgufFile g5(write_gguf_q5k(dir, q5));
        std::vector<float> row(512);
        bool bad5 = false;
        for (unsigned r = 0; r < 4 && !bad5; ++r) {
            g5.embed_row("blk.0.attn_q.weight", r, 512, row.data());
            for (size_t i = 0; i < 512; ++i)
                if (f32_bits(row[i]) != f32_bits(q5.values[static_cast<size_t>(r) * 512 + i])) {
                    std::printf("FAIL embed_row(Q5_K)[r%u i%zu]: got %.9g want %.9g\n", r, i, row[i],
                                q5.values[static_cast<size_t>(r) * 512 + i]);
                    bad5 = true;
                    break;
                }
        }
        std::printf("%s embed_row Q5_K (the fifth bit and the 6-bit scales bit-exact)\n", bad5 ? "FAIL" : "ok");
        if (bad5) return 1;
    }

    std::printf("GGUF-PACK OK\n");
    return 0;
}
