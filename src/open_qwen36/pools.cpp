/// \file pools.cpp
/// \brief The packing-plan interpreter (see pools.hpp).
#include "open_qwen36/pools.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gguf/gguf_file.hpp"
#include "open_qwen36/bfp16.hpp"
#include "open_qwen36/q4nx_file.hpp"

namespace open_qwen36 {
namespace pools {

namespace {

std::string with_layer(const std::string& name, int layer) {
    std::string s = name;
    const std::string key = "{l}";
    for (size_t p = s.find(key); p != std::string::npos; p = s.find(key, p)) s.replace(p, key.size(), std::to_string(layer));
    return s;
}

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("pools: " + what); }

void bounds(const PackOp& op, uint64_t nbytes, size_t dst_bytes) {
    if (op.dst + nbytes > dst_bytes)
        fail(op.op + " " + (op.tensor.empty() ? op.up : op.tensor) + " writes " + std::to_string(nbytes) + " B at " +
             std::to_string(op.dst) + ", past the " + std::to_string(dst_bytes) + " B buffer");
}

/// Both band laws derive the file raster's column count as `in_dim / 256`, which FLOORS.
/// At a width that is not a whole number of 256-column k-tiles -- GPT-OSS's container ships
/// 2944 -- the k-tile index then runs past it and pool chunks alias onto each other: 1409
/// distinct file chunks selected for 1472 slots, a corrupt pool with nothing raised. The
/// recipe is not supposed to derive such a width (OPEN-WIDTH-PAD pads to the lcm), and this
/// is the check that a manifest carrying one does not pack silently anyway.
/// recipes/pack.py raises the same way, from `band_rowblock_ktile` and `q8_perm`.
void ktiles_or_fail(const char* who, size_t in_dim) {
    if (in_dim % 256)
        fail(std::string(who) + ": in_dim=" + std::to_string(in_dim) + " is not a whole number of "
             "256-column k-tiles (" + std::to_string(in_dim % 256) + " over); the file raster's "
             "column count would floor and the pool chunks would alias onto each other");
}

/// pool chunk index -> file chunk index for a standard [out, in] matmul tensor:
/// a band is 64 rows x in_dim = in_dim/128 chunks; inside its band chunk i covers
/// row half i%2 and k-tile i/2 (gemv_q4.h's band law); file chunk f covers rows
/// 32*(f/ncol), cols 256*(f%ncol). Same law as recipes/pack.py (which documents
/// its equivalence with the form phlegm verified against OFLM's captured pools).
std::vector<size_t> std_perm(size_t nch, size_t in_dim) {
    ktiles_or_fail("std_perm", in_dim);
    size_t ncol = in_dim / 256, per_band = in_dim / 128;
    std::vector<size_t> perm(nch);
    for (size_t c = 0; c < nch; ++c) {
        size_t rows0 = 64 * (c / per_band) + 32 * (c % 2);
        size_t cols0 = 256 * ((c % per_band) / 2);
        perm[c] = (rows0 / 32) * ncol + cols0 / 256;
    }
    return perm;
}

/// Two adjacent 2560-byte chunks -> one 5120-byte pool chunk. Eight byte-slice copies and
/// no arithmetic: a 2560-byte chunk is q4_1 over 32 rows x 128 columns, four 32-column
/// blocks instead of eight, so the meta index `b*32 + r` puts the low half's blocks at
/// metas 0..127 and the high half's at 128..255, and the nibble raster's leading term
/// `(r/16) * 512*nb` splits each source into two 1024-byte planes by row half -- which is
/// why the nibbles interleave A0 B0 A1 B1 rather than concatenating.
/// recipes/pack.py `fuse_chunks` is the same eight copies in NumPy.
void fuse_chunk(const uint8_t* a, const uint8_t* b, uint8_t* out) {
    std::memcpy(out + 0, a + 0, 256);            // d, columns 0..127
    std::memcpy(out + 256, b + 0, 256);          // d, columns 128..255
    std::memcpy(out + 512, a + 256, 256);        // m
    std::memcpy(out + 768, b + 256, 256);
    std::memcpy(out + 1024, a + 512, 1024);      // nibbles, rows 0..15
    std::memcpy(out + 2048, b + 512, 1024);
    std::memcpy(out + 3072, a + 1536, 1024);     // nibbles, rows 16..31
    std::memcpy(out + 4096, b + 1536, 1024);
}

/// (32-row block, 128-column block) -> file chunk index in the supertile raster
/// q4nx-build writes for GPT-OSS: row block `rb` is supertile `rb/rg` at position `rb%rg`,
/// and its column block `q` lands at `(rb/rg * ncol128 + q) * rg + rb%rg`. Every other
/// converter writes the plain `rb * ncol + q`, which this reduces to at rg = 1.
size_t supertile_index(size_t rb, size_t q, size_t ncol128, size_t rg) {
    return ((rb / rg) * ncol128 + q) * rg + rb % rg;
}

const uint8_t* raw(const WeightFile& m, const std::string& name, size_t need, size_t* got = nullptr) {
    size_t n = 0;
    const uint8_t* p = m.raw(name, &n);
    if (n < need) fail(name + " is " + std::to_string(n) + " B, the plan needs " + std::to_string(need));
    if (got) *got = n;
    return p;
}

constexpr size_t Q8_CHUNK = 8704;    // 256 bf16 scales then 8192 int8 codes
constexpr size_t Q4_CHUNK = 5120;    // 256 bf16 d, 256 bf16 m, then 4096 B of nibbles
constexpr size_t GGUF_CHUNK = 6144;  // the GGUF-direct twin's f32-scale law: 1024 B d, 1024 B m, codes
constexpr size_t Q4_HALF = 2560;     // GPT-OSS's chunk: the same layout over 32 rows x 128 columns
constexpr size_t Q4K_CHUNK = 4736;   // 256 uint8 scales, 256 uint8 mins, 4096 B of nibbles, 32 bf16 S, 32 bf16 M
constexpr size_t Q8H_SCALES = 256;   // a half-tile's 128 bf16 scales
constexpr size_t Q8H_CODES = 4096;   // a half-tile's 4096 int8 codes
constexpr unsigned Q8H_ROWS = 16;

/// pool half-tile index -> (file chunk index, half) for a q8 projection: a band is 64 rows
/// x in_dim = in_dim/64 half-tiles; half-tile c inside its band covers rows 16*(c%4) of the
/// band and k-tile c/4, so its source is file chunk (2*band + (c%4)/2) at half (c%4)%2
/// (gemv_q8.h's band law; recipes/pack.py q8_perm is the same law in NumPy).
std::vector<std::pair<size_t, unsigned>> q8_perm(size_t nch, size_t in_dim) {
    ktiles_or_fail("q8_perm", in_dim);
    const size_t ncol = in_dim / 256, per_band = in_dim / 64;
    std::vector<std::pair<size_t, unsigned>> perm(nch);
    for (size_t c = 0; c < nch; ++c) {
        const size_t band = c / per_band, cc = c % per_band, part = cc % 4, kt = cc / 4;
        perm[c] = {(2 * band + part / 2) * ncol + kt, static_cast<unsigned>(part % 2)};
    }
    return perm;
}

float bf16_to_f32(uint16_t h) {
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

uint32_t bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return u;
}

/// f32 -> bf16 toward -inf: a positive magnitude truncates, a negative one grows.
uint16_t bf16_floor(float x) {
    const uint32_t u = bits(x);
    return static_cast<uint16_t>((u >> 31) ? ((u + 0xFFFFu) >> 16) : (u >> 16));
}

/// f32 -> bf16 toward +inf.
uint16_t bf16_ceil(float x) {
    const uint32_t u = bits(x);
    return static_cast<uint16_t>((u >> 31) ? (u >> 16) : ((u + 0xFFFFu) >> 16));
}

/// f32 -> bf16, round to nearest even. The directed roundings above make a re-quantized
/// block's range cover its source; a Q4_K scale is not a range end but a value being
/// re-expressed, so it takes the nearest bf16.
uint16_t bf16_rne(float x) {
    const uint32_t u = bits(x);
    return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

/// Code index of (row, block, lane) inside a chunk: the raster both formats use.
inline unsigned code_index(unsigned r, unsigned b, unsigned i) {
    return (r / 16) * 4096 + b * 512 + i * 16 + (r % 16);
}

/// What a chunk size that is neither 5120 nor 8704 probably is, for the refusal message.
std::string chunk_guess(size_t ch) {
    if (ch == Q4_HALF)
        return "GPT-OSS's 32-row x 128-column chunk, which the std_fuse op reads -- the file raster "
               "is a supertile as well as half-width, so this tensor needs that op rather than this one";
    if (ch == 1280) return "a smaller chunk geometry (" + std::to_string(ch * 8192 / Q4_CHUNK) +
                           " values per chunk instead of 8192)";
    return "not a chunk format this packer knows";
}

/// `nch` q4_1 chunks of `name` starting at chunk `chunk0`, whatever the container stores.
/// A q4_1 tensor is a view into the mapping; a q8 one is re-quantized into `tmp` first
/// (OPEN-PACK-PLAN: a q8 source is accepted transparently by every q4 pack op, because the
/// two formats hold the SAME 32-row x 256-column tile, so no chunk index law changes).
/// Is this 5120-byte tensor the SIGNED quantiser (w = d * int4(q), every min zero) rather
/// than q4_1 (w = d * q + min)? Both share the chunk and nothing in the container says
/// which, so the signal is the mins: a real q4_1 tensor does not have 256 exactly-zero
/// ones, because a min is a block's own minimum. recipes/pack.py is_signed_q4 has the
/// same rule, and the replica reads through it, so the two cannot disagree.
bool signed_q4(const Q4nxFile& m, const std::string& name) {
    const uint8_t* p = raw(m, name, Q4_CHUNK) + 512;
    for (unsigned i = 0; i < 256; ++i) {
        uint16_t mn;
        std::memcpy(&mn, p + 2 * i, 2);
        if (mn) return false;
    }
    return true;
}

const uint8_t* q4_source(const WeightFile& m, const std::string& name, size_t chunk0, size_t nch, size_t ch,
                         std::vector<uint8_t>& tmp) {
    // q4_1 / q8 / Q4_K chunks only come from a q4nx container (GGUF goes through std_perm_gguf).
    const auto* q = dynamic_cast<const Q4nxFile*>(&m);
    if (!q) fail(name + " is not in a q4nx container; only std_perm_gguf packs GGUF tensors");
    const size_t src_ch = q->chunk_bytes(name);
    if (src_ch == Q4_CHUNK && ch == Q4_CHUNK && signed_q4(*q, name)) {
        const uint8_t* src = raw(m, name, (chunk0 + nch) * Q4_CHUNK) + chunk0 * Q4_CHUNK;
        tmp.resize(nch * Q4_CHUNK);
        q4_0_to_q4_1_chunks(src, nch, tmp.data());
        return tmp.data();
    }
    if (src_ch == ch) return raw(m, name, (chunk0 + nch) * ch) + chunk0 * ch;
    if (src_ch == Q8_CHUNK && ch == Q4_CHUNK) {
        const uint8_t* src = raw(m, name, (chunk0 + nch) * Q8_CHUNK) + chunk0 * Q8_CHUNK;
        tmp.resize(nch * Q4_CHUNK);
        requant_q4_1_chunks(src, nch, tmp.data());
        return tmp.data();
    }
    if (src_ch == Q4K_CHUNK && ch == Q4_CHUNK) {
        const uint8_t* src = raw(m, name, (chunk0 + nch) * Q4K_CHUNK) + chunk0 * Q4K_CHUNK;
        tmp.resize(nch * Q4_CHUNK);
        q4k_to_q4_1_chunks(src, nch, tmp.data());
        return tmp.data();
    }
    fail(name + " has " + std::to_string(src_ch) + "-byte quant chunks; the packer reads " +
         std::to_string(ch) + " (q4_1), " + std::to_string(Q8_CHUNK) + " (q8) and " +
         std::to_string(Q4K_CHUNK) + " (Q4_K); " + std::to_string(src_ch) + " is " + chunk_guess(src_ch));
}

/// One pool chunk (the pool's own chunk law: f32 scales for GGUF_CHUNK, bf16 for Q4_CHUNK)
/// out of a GGUF tensor's FILE RASTER: rows [row0, row0 + 32) and the eight 32-value k
/// blocks at [blk0, blk0 + 8). `row0` indexes the WHOLE tensor, so an expert tensor's
/// expert e starts at row e * rows_per_expert -- which is what lets the MoE pack ops read
/// their 3-D [k, n, experts] tensors with the same per-chunk law std_perm_gguf uses for a
/// 2-D one.
///
/// Q4_0 / Q4_1 is a byte permutation plus an exact widening of the fp16 scales; every other
/// accepted type is decoded a row at a time and re-quantized to the q4_1 law. The chunk is
/// zeroed here, so a row or column block past the tensor stays zero (the pool's padding).
void gguf_chunk(const GgufFile& g, const std::string& name, const GgufFile::TensorInfo& t,
                size_t row0, size_t blk0, size_t rows, size_t in_dim, size_t ch, double fold,
                std::vector<float>& rd, uint8_t* d) {
    const bool kq = fold != 1.0 || t.type == GgufFile::Type::Q8_0 || t.type == GgufFile::Type::Q4_K ||
                    t.type == GgufFile::Type::Q5_K || t.type == GgufFile::Type::Q6_K ||
                    t.type == GgufFile::Type::IQ4_NL || t.type == GgufFile::Type::IQ4_XS;
    if (!kq && t.type != GgufFile::Type::Q4_0 && t.type != GgufFile::Type::Q4_1)
        fail(name + " is " + GgufFile::type_name(t.type) + "; std_perm_gguf packs Q4_0/Q4_1 and "
             "requantizes Q8_0/Q4_K/Q5_K/Q6_K/IQ4_NL/IQ4_XS (convert the others with q4nx-build)");
    if (ch != Q4_CHUNK && ch != GGUF_CHUNK)
        fail(name + ": pool chunk " + std::to_string(ch) +
             " is neither the q4_1 law nor the GGUF-direct twin's f32 one");
    const bool has_min = t.type == GgufFile::Type::Q4_1 || kq;
    const size_t blk = has_min ? 20 : 18;
    const bool f32_law = ch == GGUF_CHUNK;
    const size_t M_OFF = f32_law ? 1024 : 512, CODES = f32_law ? 2048 : 1024, SD = f32_law ? 4 : 2;
    const size_t nb = in_dim / 32;
    size_t n = 0;
    const uint8_t* src = raw(g, name, 0, &n);
    if (kq) rd.resize(in_dim);
    const bool granite_qk = g.kv_has("general.architecture") && g.kv_str("general.architecture") == "granite" &&
                            (name.find(".self_attn.q_proj.") != std::string::npos ||
                             name.find(".self_attn.k_proj.") != std::string::npos);
    const uint64_t granite_hd = granite_qk
        ? (g.kv_has("granite.attention.key_length") ? g.kv_u64("granite.attention.key_length")
                                                    : g.kv_u64("granite.embedding_length") / g.kv_u64("granite.attention.head_count"))
        : 0;
    std::memset(d, 0, ch);
    for (size_t r = 0; r < 32; ++r) {
        uint64_t row = row0 + r;
        if (row >= rows) break;                           // padded rows: zeroed scales/codes
        if (granite_hd) {
            const uint64_t hb = (row / granite_hd) * granite_hd, ih = row % granite_hd, half = granite_hd / 2;
            row = hb + (ih < half ? 2 * ih : 2 * (ih - half) + 1);
        }
        const uint8_t rb = static_cast<uint8_t>(r / 16), rl = static_cast<uint8_t>(r % 16);
        if (kq) g.embed_row(name, row, in_dim, rd.data());   // exact f32 row, once
        const uint8_t* srow = kq ? nullptr : src + row * nb * blk;
        for (size_t kb = 0; kb < 8; ++kb) {
            if (blk0 + kb >= nb) break;                  // padded columns stay zero
            // scales: fp16 d (and m) -> f32, plane index j = kb*32 + r
            uint16_t ud, um = 0;
            if (kq) {
                // re-quantize the block to the pool's q4_1 law (fp16 d/m, 4-bit codes)
                float vs[32];
                const float* v = rd.data() + (blk0 + kb) * 32;
                for (size_t i = 0; i < 32; ++i) vs[i] = fold ? v[i] * static_cast<float>(fold) : v[i];
                float lo = vs[0], hi = vs[0];
                for (size_t i = 0; i < 32; ++i) {
                    lo = std::min(lo, vs[i]);
                    hi = std::max(hi, vs[i]);
                }
                const float mraw = lo, draw = (hi - lo) / 15.f;
                ud = f32_to_fp16(draw);
                um = f32_to_fp16(mraw);
                const float df = fp16_to_f32(ud), mf = fp16_to_f32(um);
                const float inv = df > 0.f ? 1.f / df : 0.f;
                if (f32_law) {
                    std::memcpy(d + SD * (kb * 32 + r), &df, 4);
                    std::memcpy(d + M_OFF + SD * (kb * 32 + r), &mf, 4);
                } else {
                    const uint16_t db = f32_to_bf16(df), mb = f32_to_bf16(mf);
                    std::memcpy(d + SD * (kb * 32 + r), &db, 2);
                    std::memcpy(d + M_OFF + SD * (kb * 32 + r), &mb, 2);
                }
                // codes: value i at nibble (kb*32+i, r), i+16 at (p0+256)
                uint8_t nib[16];
                for (size_t i = 0; i < 16; ++i) {
                    const int q0 = std::lround((vs[i] - mf) * inv);
                    const int q1 = std::lround((vs[i + 16] - mf) * inv);
                    nib[i] = static_cast<uint8_t>(((q1 < 0 ? 0 : q1 > 15 ? 15 : q1) << 4) |
                                                  (q0 < 0 ? 0 : q0 > 15 ? 15 : q0));
                }
                for (size_t i = 0; i < 16; ++i) {
                    const size_t p0 = rb * 4096 + (kb * 32 + i) * 16 + rl;
                    d[CODES + (p0 >> 1)] |= (p0 & 1) ? static_cast<uint8_t>(nib[i] << 4)
                                                     : (nib[i] & 0xF);
                    const size_t p1 = p0 + 256;              // i + 16: the block's high half
                    d[CODES + (p1 >> 1)] |= (p1 & 1) ? (nib[i] & 0xF0) : (nib[i] >> 4);
                }
                continue;
            }
            const uint8_t* b = srow + (blk0 + kb) * blk;
            std::memcpy(&ud, b, 2);
            if (has_min) std::memcpy(&um, b + 2, 2);
            const float df = fp16_to_f32(ud);
            const float mf = has_min ? fp16_to_f32(um) : -8.f * df;
            if (f32_law) {
                std::memcpy(d + SD * (kb * 32 + r), &df, 4);
                std::memcpy(d + M_OFF + SD * (kb * 32 + r), &mf, 4);
            } else {
                const uint16_t db = f32_to_bf16(df), mb = f32_to_bf16(mf);
                std::memcpy(d + SD * (kb * 32 + r), &db, 2);
                std::memcpy(d + M_OFF + SD * (kb * 32 + r), &mb, 2);
            }
            // codes: block nibbles (two K per byte) -> two rows per byte at one K
            for (size_t i = 0; i < 16; ++i) {
                const uint8_t byte = b[(has_min ? 4 : 2) + i];
                const uint8_t lo = byte & 0xF, hi = byte >> 4;   // GGUF: value i | value i+16
                const size_t p0 = rb * 4096 + (kb * 32 + i) * 16 + rl;
                d[CODES + (p0 >> 1)] |= (p0 & 1) ? lo << 4 : lo;   // value j = lo
                const size_t p1 = p0 + 256;              // i + 16: the block's high half
                d[CODES + (p1 >> 1)] |= (p1 & 1) ? hi << 4 : hi;   // value j + 16 = hi
            }
        }
    }
}

}  // namespace

void split_q4_1_chunks(const uint8_t* src, size_t nch, bool hi, uint8_t* dst) {
    for (size_t c = 0; c < nch; ++c) {
        const uint8_t* s = src + c * Q8_CHUNK;
        uint8_t* o = dst + c * Q4_CHUNK;
        std::memset(o, 0, Q4_CHUNK);
        const int8_t* code = reinterpret_cast<const int8_t*>(s + 512);
        uint8_t* nib = o + 1024;
        for (unsigned meta = 0; meta < 256; ++meta) {
            uint16_t sh;
            std::memcpy(&sh, s + 2 * meta, 2);
            const float scale = bf16_to_f32(sh);
            // x16 and x-128 move only the exponent, so the truncation is exact
            const float d = hi ? scale * 16.0f : scale;
            const float mn = hi ? scale * -128.0f : 0.0f;
            uint32_t du, mu;
            std::memcpy(&du, &d, 4);
            std::memcpy(&mu, &mn, 4);
            const uint16_t d16 = static_cast<uint16_t>(du >> 16), m16 = static_cast<uint16_t>(mu >> 16);
            std::memcpy(o + 2 * meta, &d16, 2);
            std::memcpy(o + 512 + 2 * meta, &m16, 2);
        }
        for (unsigned r = 0; r < 32; ++r)
            for (unsigned b = 0; b < 8; ++b)
                for (unsigned i = 0; i < 32; ++i) {
                    const unsigned p = code_index(r, b, i);
                    const int v = code[p];
                    const int q = hi ? (v >> 4) + 8 : (v & 15);     // >> of a negative int floors
                    nib[p >> 1] |= static_cast<uint8_t>((p & 1) ? (q << 4) : q);
                }
    }
}

/// A consts tensor's values as bf16, decoded from whatever the file holds.
///
/// The q4nx container stores these weights bf16 (the kernel's layout), so the
/// bytes are read in place. A GGUF stores them f32 -- or, for the SSM
/// projections, quantized (llama.cpp quantizes what it quantizes) -- so they
/// are decoded and converted to the same bf16 values the container holds.
/// Returns false when the file's bytes are already the wanted ones.
bool consts_as_bf16(const WeightFile& f, const std::string& name, size_t count, std::vector<uint8_t>& out) {
    size_t n = 0;
    f.raw(name, &n);                                  // the file's own bytes: a quantized
                                                      // tensor is shorter than its bf16 form
    // Two bytes per value is bf16 ONLY in a container (which stores these weights bf16).
    // A GGUF's f16 tensor has the same size and would be read as bf16 -- a silent
    // reinterpretation of every value -- so ask the file which type it is.
    const auto* g = dynamic_cast<const GgufFile*>(&f);
    if (n == count * 2 && (!g || g->tensor(name).type == GgufFile::Type::BF16)) return false;
    const std::vector<float> v = f.f32(name);
    if (v.size() != count)
        fail(name + " is " + std::to_string(v.size()) + " values, not " + std::to_string(count));
    out.resize(count * 2);
    for (size_t i = 0; i < count; ++i) {
        const uint16_t b = f32_to_bf16(v[i]);
        std::memcpy(out.data() + 2 * i, &b, 2);
    }
    return true;
}

void pack_norm(const WeightFile& f, const std::string& name, size_t bytes, uint8_t* dst, double div) {
    // A small weight (a layernorm) as bf16: the q4nx container stores it bf16;
    // a GGUF may store it f32 / f16 / bf16 -- convert exactly as the container does.
    // `div` divides in double before the single rounding, so a folded scale costs no extra one.
    size_t n = 0;
    const uint8_t* src = raw(f, name, 0, &n);
    auto* g = dynamic_cast<const GgufFile*>(&f);
    auto bf16_div = [&](size_t count) {
        for (size_t i = 0; i < count; ++i) {
            uint16_t u;
            std::memcpy(&u, src + 2 * i, 2);
            const uint16_t b = f32_to_bf16(static_cast<float>(bf16_to_f32(u) / div));
            std::memcpy(dst + 2 * i, &b, 2);
        }
    };
    if (!g) {
        if (n != bytes) fail(name + " is " + std::to_string(n) + " B, the slot holds " + std::to_string(bytes));
        if (div == 1.0) std::memcpy(dst, src, n);
        else bf16_div(n / 2);
        return;
    }
    switch (g->tensor(name).type) {
        case GgufFile::Type::BF16: {
            if (n != bytes) fail(name + " is " + std::to_string(n) + " B, the slot holds " + std::to_string(bytes));
            if (div == 1.0) std::memcpy(dst, src, n);
            else bf16_div(n / 2);
            break;
        }
        case GgufFile::Type::F16: {
            if (n != bytes) fail(name + " is " + std::to_string(n) + " B of fp16, the slot holds " + std::to_string(bytes) + " B of bf16");
            for (size_t i = 0; i < bytes / 2; ++i) {
                uint16_t u;
                std::memcpy(&u, src + 2 * i, 2);
                const float v = static_cast<float>(fp16_to_f32(u) / div);
                const uint16_t b = f32_to_bf16(v);
                std::memcpy(dst + 2 * i, &b, 2);
            }
            break;
        }
        case GgufFile::Type::F32: {
            if (n == bytes) {
                // The slot holds f32 values (the SSM's per-head scalars): copy them.
                std::memcpy(dst, src, n);
                break;
            }
            // A slot wider than the tensor is a width padded past the model's (GPT-OSS: 2880 values in a
            // 3072-lane norm): the values, then bf16 zeros. Wider tensors than slots still refuse.
            if (n > 2 * bytes || n % 4) fail(name + " is " + std::to_string(n) + " B of f32, the slot holds " + std::to_string(bytes) + " B of bf16");
            if (n < 2 * bytes) std::memset(dst + n / 2, 0, bytes - n / 2);
            for (size_t i = 0; i < n / 4; ++i) {
                float v;
                std::memcpy(&v, src + 4 * i, 4);
                const uint16_t b = f32_to_bf16(div == 1.0 ? v : static_cast<float>(v / div));
                std::memcpy(dst + 2 * i, &b, 2);
            }
            break;
        }
        default:
            fail(name + " is " + GgufFile::type_name(g->tensor(name).type) + "; a layernorm must be f32 / f16 / bf16");
    }
}

double scalar_value(const WeightFile& f, const std::string& name) {
    size_t n = 0;
    const uint8_t* src = raw(f, name, 0, &n);
    auto* g = dynamic_cast<const GgufFile*>(&f);
    const bool f32 = g ? g->tensor(name).type == GgufFile::Type::F32 : n == 4;
    const bool f16 = g && g->tensor(name).type == GgufFile::Type::F16;
    if ((f32 && n != 4) || (!f32 && n != 2)) fail(name + " is " + std::to_string(n) + " B, not one value");
    if (f32) { float v; std::memcpy(&v, src, 4); return v; }
    uint16_t u;
    std::memcpy(&u, src, 2);
    return f16 ? fp16_to_f32(u) : bf16_to_f32(u);
}

namespace {
// row r, column k of a tensor's chunks (32-row blocks major, then 256-k tiles), as the q8 / q4_1 GEMMs read it
float q8_value(const uint8_t* src, size_t ncol, size_t r, size_t k) {
    const uint8_t* c = src + ((r / 32) * ncol + k / 256) * Q8_CHUNK;
    const unsigned rl = static_cast<unsigned>(r % 32), kin = static_cast<unsigned>(k % 256);
    uint16_t s;
    std::memcpy(&s, c + 2 * ((kin / 32) * 32 + rl), 2);
    return static_cast<float>(static_cast<int8_t>(c[512 + code_index(rl, kin / 32, kin % 32)])) * bf16_to_f32(s);
}
float q4_1_value(const uint8_t* src, size_t ncol, size_t r, size_t k) {
    const uint8_t* c = src + ((r / 32) * ncol + k / 256) * Q4_CHUNK;
    const unsigned rl = static_cast<unsigned>(r % 32), kin = static_cast<unsigned>(k % 256);
    uint16_t d, m;
    std::memcpy(&d, c + 2 * ((kin / 32) * 32 + rl), 2);
    std::memcpy(&m, c + 512 + 2 * ((kin / 32) * 32 + rl), 2);
    const unsigned p = (rl / 16) * 4096 + kin * 16 + rl % 16;
    const unsigned n = (c[1024 + (p >> 1)] >> (4 * (p & 1))) & 15u;
    return bf16_to_f32(m) + static_cast<float>(n) * bf16_to_f32(d);
}
}  // namespace

void bf16_gemm_pack(const uint8_t* src, bool q8, size_t rows, size_t cols, uint16_t* dst) {
    const size_t ncol = cols / 256, nkt = cols / 64;
#pragma omp parallel for
    for (long long bb = 0; bb < static_cast<long long>(rows / 64); ++bb) {
        const size_t b = static_cast<size_t>(bb);
        for (size_t kt = 0; kt < nkt; ++kt) {
            uint16_t* e = dst + (b * nkt + kt) * 4096;
            for (size_t z = 0; z < 16; ++z)
                for (size_t i = 0; i < 8; ++i)
                    for (size_t r = 0; r < 4; ++r)
                        for (size_t kk = 0; kk < 8; ++kk) {
                            const size_t row = 64 * b + 4 * z + r, k = 64 * kt + 8 * i + kk;
                            e[(z * 8 + i) * 32 + r * 8 + kk] =
                                f32_to_bf16(q8 ? q8_value(src, ncol, row, k) : q4_1_value(src, ncol, row, k));
                        }
        }
    }
}

void bfp16_gemm_pack(const uint8_t* src, size_t rows, size_t cols, size_t row0, size_t n_total, uint8_t* dst,
                     const std::string& glu) {
    const Bfp16Tiling t;
    const size_t ncol = cols / 256;
    const size_t up = glu == "up" ? 8 : 0;
#pragma omp parallel for
    for (long long rr = 0; rr < static_cast<long long>(rows); ++rr) {
        const size_t r = static_cast<size_t>(rr);
        const size_t br = glu.empty() ? row0 + r : bfp16_glu_row(r, n_total / 2, t) + up;
        float w[8];
        for (size_t cb = 0; cb < cols / 8; ++cb) {
            for (size_t i = 0; i < 8; ++i) w[i] = q4_1_value(src, ncol, r, cb * 8 + i);
            bfp16_encode_block(w, dst + bfp16_b_offset(br, cb, n_total, cols, t));
        }
    }
}

namespace {
// 8 values sharing their largest exponent, 8-bit two's-complement mantissas, rounded as pack.py's encoder
void bfp16_encode8(const float* x, uint8_t* out) {
    uint32_t b[8];
    int e[8], mx = 0;
    for (int i = 0; i < 8; ++i) {
        std::memcpy(&b[i], &x[i], 4);
        e[i] = static_cast<int>((b[i] >> 23) & 0xFF);
        mx = std::max(mx, e[i]);
    }
    out[0] = static_cast<uint8_t>(mx);
    for (int i = 0; i < 8; ++i) {
        const int64_t mant = static_cast<int64_t>(b[i] & 0x7FFFFF) | (e[i] ? (int64_t{1} << 23) : 0);
        const int64_t s = (b[i] >> 31) ? -mant : mant;
        const int sh = 17 + std::min(mx - e[i], 31);
        const int64_t v = std::clamp<int64_t>((s + (int64_t{1} << (sh - 1))) >> sh, -128, 127);
        out[1 + i] = static_cast<uint8_t>(static_cast<int8_t>(v));
    }
}
}  // namespace

void bfp16_dit_pack(const uint8_t* src, size_t rows, size_t cols, uint8_t* dst) {
    if (rows % 128 || cols % 256) fail("bfp16_dit: [" + std::to_string(rows) + ", " + std::to_string(cols) +
                                       "] is not 128-row tiles of 256-wide chunks");
    const size_t ncol = cols / 256, nkt = cols / 64;
    // tile (n tile, k tile) of 128 x 64, then [n-block 16][k-block 8][n 8], a block being 8 k of one row
#pragma omp parallel for
    for (long long nt = 0; nt < static_cast<long long>(rows / 128); ++nt)
        for (size_t kt = 0; kt < nkt; ++kt) {
            uint8_t* tile = dst + (static_cast<size_t>(nt) * nkt + kt) * 9216;
            float v[8];
            for (size_t nb = 0; nb < 16; ++nb)
                for (size_t kb = 0; kb < 8; ++kb)
                    for (size_t ni = 0; ni < 8; ++ni) {
                        const size_t n = static_cast<size_t>(nt) * 128 + nb * 8 + ni, k0 = kt * 64 + kb * 8;
                        for (size_t i = 0; i < 8; ++i) v[i] = q4_1_value(src, ncol, n, k0 + i);
                        bfp16_encode8(v, tile + ((nb * 8 + kb) * 8 + ni) * 9);
                    }
        }
}

// ---- GPT-OSS (OPEN-FAMILY-GPTOSS): GGUF Q8_0 / MXFP4 -> the q8 GEMV's half-tiles, and Q8_0 -> the
// f32-scale lm_head chunks. recipes/gptoss_pack.py writes the same bytes (tests/test_gptoss_pack.py).
//
// half-tile, 16 rows x 256 K, 5120 B: bf16 scale[kb*16 + r] at [0:256), int8 code[k*16 + r] at
// [256:4352), f32 bias[r] at [4352:4416) on k-tile-0 half-tiles only (gemv_q8.h GEMV_Q8_PAD_BIAS);
// a 64-row band is K/64 half-tiles, half-tile c the rows' part c % 4 and k-tile c / 4. Rows and
// columns past the tensor's are zero. Q8_0: code = q, scale = bf16(fp16 d), round to nearest even.
// MXFP4: code = 2x the e2m1 level (KVALUES, an integer), scale = 2^(e-128), exact in bf16.
namespace {
constexpr size_t GX_TILE = 5120;
constexpr size_t GX_BIAS = 4352;
constexpr size_t GX_HEAD = 9216;
constexpr int8_t GX_KVALUES[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

float e8m0_half(uint8_t e) {
    const uint32_t u = e < 2 ? (0x00200000u << e) : (static_cast<uint32_t>(e - 1) << 23);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

void gx_tiles(const uint8_t* src, bool mxfp4, size_t rows, size_t cols, size_t rows_pad, size_t cols_pad,
              const float* bias, uint8_t* out) {
    if (rows > rows_pad || cols > cols_pad || rows_pad % 64 || cols_pad % 256 || cols % 32)
        fail("gx tiles: [" + std::to_string(rows) + ", " + std::to_string(cols) + "] does not pad to [" +
             std::to_string(rows_pad) + ", " + std::to_string(cols_pad) + "]");
    const size_t nkt = cols_pad / 256, nb = rows_pad / 64, bpr = cols / 32, bb = mxfp4 ? 17 : 34;
#pragma omp parallel for schedule(dynamic)
    for (long long bl = 0; bl < static_cast<long long>(nb); ++bl) {
        const size_t b = static_cast<size_t>(bl);
        for (size_t kt = 0; kt < nkt; ++kt)
            for (size_t part = 0; part < 4; ++part) {
                uint8_t* t = out + ((b * nkt + kt) * 4 + part) * GX_TILE;
                std::memset(t, 0, GX_TILE);
                for (size_t r = 0; r < 16; ++r) {
                    const size_t row = 64 * b + 16 * part + r;
                    if (row >= rows) continue;
                    for (size_t kb = 0; kb < 8; ++kb) {
                        const size_t blk = kt * 8 + kb;
                        if (blk >= bpr) continue;
                        const uint8_t* B = src + (row * bpr + blk) * bb;
                        float scale;
                        if (mxfp4) {
                            scale = e8m0_half(B[0]);
                        } else {
                            uint16_t h;
                            std::memcpy(&h, B, 2);
                            scale = fp16_to_f32(h);
                        }
                        const uint16_t sb = bf16_rne(scale);
                        std::memcpy(t + 2 * (kb * 16 + r), &sb, 2);
                        for (size_t i = 0; i < 32; ++i) {
                            const int8_t code = mxfp4 ? GX_KVALUES[i < 16 ? (B[1 + i] & 15) : (B[1 + i - 16] >> 4)]
                                                      : static_cast<int8_t>(B[2 + i]);
                            t[Q8H_SCALES + (kb * 32 + i) * 16 + r] = static_cast<uint8_t>(code);
                        }
                    }
                    if (kt == 0 && bias) std::memcpy(t + GX_BIAS + 4 * r, bias + row, 4);
                }
            }
    }
}

// mx4 half-tile (gemv_mx4.h): the experts' MXFP4 kept as nibbles, 16 rows x 256 K in 2368 B --
// four 64-K groups g of 576 B (bf16 scale[kb' * 16 + r] for blocks 2g + kb' in 64 B, then each
// block's 256 B of nibbles: four 8-K octets, 32 B for rows 0..7 then 32 B for rows 8..15, nibble
// j of a half -- byte j / 2, low nibble first -- being k = j / 8, row = j % 8), then f32 bias[16]
// at 2304 on k-tile 0. A nibble is the e2m1 code re-signed for the kernel's decode
// (GX_MX4_NIB: p, -p for a negative, -8 for -12); the scale is MXFP4's 2^(e-128) as in gx_tiles.
constexpr size_t GX_MX4_TILE = 2368;
constexpr size_t GX_MX4_BIAS = 2304;
constexpr uint8_t GX_MX4_NIB[16] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 15, 14, 13, 12, 11, 10, 8};

void gx_mx4_tiles(const uint8_t* src, size_t rows, size_t cols, size_t rows_pad, size_t cols_pad, const float* bias,
                  uint8_t* out) {
    if (rows > rows_pad || cols > cols_pad || rows_pad % 64 || cols_pad % 256 || cols % 32)
        fail("gx mx4 tiles: [" + std::to_string(rows) + ", " + std::to_string(cols) + "] does not pad to [" +
             std::to_string(rows_pad) + ", " + std::to_string(cols_pad) + "]");
    const size_t nkt = cols_pad / 256, nb = rows_pad / 64, bpr = cols / 32;
#pragma omp parallel for schedule(dynamic)
    for (long long bl = 0; bl < static_cast<long long>(nb); ++bl) {
        const size_t b = static_cast<size_t>(bl);
        for (size_t kt = 0; kt < nkt; ++kt)
            for (size_t part = 0; part < 4; ++part) {
                uint8_t* t = out + ((b * nkt + kt) * 4 + part) * GX_MX4_TILE;
                std::memset(t, 0, GX_MX4_TILE);
                for (size_t r = 0; r < 16; ++r) {
                    const size_t row = 64 * b + 16 * part + r;
                    if (row >= rows) continue;
                    for (size_t kb = 0; kb < 8; ++kb) {
                        const size_t blk = kt * 8 + kb;
                        if (blk >= bpr) continue;
                        const uint8_t* B = src + (row * bpr + blk) * 17;
                        uint8_t* g = t + (kb / 2) * 576;
                        const uint16_t sb = bf16_rne(e8m0_half(B[0]));
                        std::memcpy(g + 2 * ((kb % 2) * 16 + r), &sb, 2);
                        uint8_t* nib = g + 64 + (kb % 2) * 256 + (r / 8) * 32;
                        for (size_t i = 0; i < 32; ++i) {
                            const uint8_t n = i < 16 ? (B[1 + i] & 15) : (B[1 + i - 16] >> 4);
                            const size_t j = (i % 8) * 8 + r % 8;          // k = 8 oc + i % 8, oc = i / 8
                            nib[(i / 8) * 64 + j / 2] |= static_cast<uint8_t>(GX_MX4_NIB[n] << (4 * (j % 2)));
                        }
                    }
                    if (kt == 0 && bias) std::memcpy(t + GX_MX4_BIAS + 4 * r, bias + row, 4);
                }
            }
    }
}

size_t gx_band_bytes(size_t cols_pad, size_t tile = GX_TILE) { return cols_pad / 64 * tile; }

// The block prefill's expert GEMM B (OPEN-PREFILL-GPTOSS; gemm_bfp16 GB_B=mx4, mx4_b.cc and pack.py
// pack_b_mx4): per expert an N x K operand in gemm_bfp16's tile order -- array column c, its n-groups
// c, c + 8, ..., each group's K / 64 tiles -- each tile 128 rows x 64 k in 4352 B: x = e + 5 per
// (32-K block, row) as [kb32][sp][half][r] (row 16 sp + 8 half + r), then per (sp, kb, half) 32 B of
// re-signed nibbles, value j = row 16 sp + 8 half + j / 8, k = 8 kb + j % 8 (byte j / 2, low nibble
// first). `row_of(n)` names the source row of B row n, or -1 for a zero row.
constexpr size_t GP_MX4_TILE = 4352;

template <class RowOf>
void gp_mx4_pack(const uint8_t* src, size_t cols, size_t N, size_t K, RowOf row_of, uint8_t* dst) {
    const size_t nk = K / 64, ng = N / 128, per_col = ng / 8, bpr = cols / 32;
#pragma omp parallel for schedule(dynamic)
    for (long long gl = 0; gl < static_cast<long long>(ng); ++gl) {
        const size_t g = static_cast<size_t>(gl);
        for (size_t kc = 0; kc < nk; ++kc) {
            uint8_t* t = dst + (((g % 8) * per_col + g / 8) * nk + kc) * GP_MX4_TILE;
            std::memset(t, 0, GP_MX4_TILE);
            for (size_t rr = 0; rr < 128; ++rr) {
                const long src_row = row_of(g * 128 + rr);
                if (src_row < 0) continue;
                const size_t sp = rr / 16, half = (rr / 8) % 2, r = rr % 8;
                for (size_t kb32 = 0; kb32 < 2; ++kb32) {
                    const size_t blk = kc * 2 + kb32;
                    if (blk >= bpr) continue;
                    const uint8_t* B = src + (static_cast<size_t>(src_row) * bpr + blk) * 17;
                    t[(kb32 * 8 + sp) * 16 + half * 8 + r] = static_cast<uint8_t>(B[0] + 5);
                    uint8_t v[32];                     // value i of the block: qs[i] & 15, value i + 16: qs[i] >> 4
                    for (size_t i = 0; i < 16; ++i) {
                        v[i] = GX_MX4_NIB[B[1 + i] & 15];
                        v[i + 16] = GX_MX4_NIB[B[1 + i] >> 4];
                    }
                    // the row's 8 values of each 8-K block are 4 whole bytes of its 32-B run (row r at bytes 4r..4r+3)
                    for (size_t kq = 0; kq < 4; ++kq) {
                        uint8_t* o = t + 256 + (((sp * 8 + kb32 * 4 + kq) * 2 + half) * 32) + r * 4;
                        for (size_t m = 0; m < 4; ++m)
                            o[m] = static_cast<uint8_t>(v[kq * 8 + 2 * m] | (v[kq * 8 + 2 * m + 1] << 4));
                    }
                }
            }
        }
    }
}

/// A GGUF tensor of `type` and the shape the op expects ([experts,] rows, cols), and its block bytes.
const GgufFile& gx_gguf(const WeightFile& m, const std::string& name, GgufFile::Type type, size_t* rows,
                        size_t* cols, size_t experts) {
    const auto* g = dynamic_cast<const GgufFile*>(&m);
    if (!g) fail(name + ": the gx ops read a GGUF");
    const auto& t = g->tensor(name);
    if (t.type != type)
        fail(name + " is " + GgufFile::type_name(t.type) + ", the plan wants " + GgufFile::type_name(type));
    if (t.dims.size() != (experts ? 3u : 2u) || (experts && t.dims[2] != experts))
        fail(name + ": not a " + std::string(experts ? "[experts, rows, cols]" : "[rows, cols]") + " tensor");
    *cols = t.dims[0];
    *rows = t.dims[1];
    return *g;
}

/// An F32 bias, as floats ([rows] or [experts, rows]).
std::vector<float> gx_bias(const WeightFile& m, const std::string& name, size_t values) {
    const auto* g = dynamic_cast<const GgufFile*>(&m);
    if (!g || g->tensor(name).type != GgufFile::Type::F32) fail(name + ": a gx bias must be a GGUF F32 tensor");
    std::vector<float> v(values);
    std::memcpy(v.data(), raw(m, name, values * 4), values * 4);
    return v;
}
}  // namespace

uint64_t op_bytes(const PackOp& op, size_t chunk_bytes) {
    if (op.op == "bfp16_dit") return op.nch * 9216;      // a 32 x 256 chunk's 8192 values at 9 bits
    if (op.op == "bfp16_gemm" || op.op == "gp_bfp16") return op.n_total * op.in_dim / 8 * BFP16_BLOCK_BYTES;
    if (op.op == "gp_mx4") return op.experts * op.expert_bytes;
    return op.op == "bf16_gemm" ? op.nch * 2 * 8192 : op.nch * chunk_bytes;
}

void requant_q4_1_chunks(const uint8_t* src, size_t nch, uint8_t* dst) {
    for (size_t c = 0; c < nch; ++c) {
        const uint8_t* s = src + c * Q8_CHUNK;
        uint8_t* o = dst + c * Q4_CHUNK;
        std::memset(o, 0, Q4_CHUNK);
        const int8_t* code = reinterpret_cast<const int8_t*>(s + 512);
        uint8_t* nib = o + 1024;
        for (unsigned r = 0; r < 32; ++r) {
            for (unsigned b = 0; b < 8; ++b) {
                const unsigned meta = b * 32 + r;          // the scale slot of (row, block)
                uint16_t sh;
                std::memcpy(&sh, s + 2 * meta, 2);
                const float scale = bf16_to_f32(sh);
                float v[32];
                float mn = 0.0f, mx = 0.0f;
                for (unsigned i = 0; i < 32; ++i) {
                    v[i] = static_cast<float>(code[code_index(r, b, i)]) * scale;
                    if (i == 0) {
                        mn = mx = v[0];
                    } else {
                        if (v[i] < mn) mn = v[i];
                        if (v[i] > mx) mx = v[i];
                    }
                }
                const uint16_t mu = bf16_floor(mn);
                const float mf = bf16_to_f32(mu);
                const uint16_t du = bf16_ceil((mx - mf) / 15.0f);
                const float d = bf16_to_f32(du);
                const float inv = d > 0.0f ? 1.0f / d : 0.0f;
                std::memcpy(o + 2 * meta, &du, 2);        // d[256]
                std::memcpy(o + 512 + 2 * meta, &mu, 2);  // m[256]
                for (unsigned i = 0; i < 32; ++i) {
                    const float t = (v[i] - mf) * inv;
                    int q = static_cast<int>(t + 0.5f);
                    if (q < 0) q = 0;
                    if (q > 15) q = 15;
                    const unsigned p = code_index(r, b, i);
                    nib[p >> 1] |= static_cast<uint8_t>((p & 1) ? (q << 4) : q);
                }
            }
        }
    }
}

void q4_0_to_q4_1_chunks(const uint8_t* src, size_t nch, uint8_t* dst) {
    // int4(q) == (q ^ 8) - 8, so flipping bit 3 of every nibble turns two's complement
    // into offset binary and d * int4(q) becomes d * q' + (-8 * d). Writing -8 * d into
    // the min slot leaves the GEMV reading exactly the right values. Both halves are
    // exact: the flip is a relabelling and -8 * d only moves a bf16 exponent.
    // recipes/pack.py q4_0_to_q4_1 does the same and must agree byte for byte.
    for (size_t c = 0; c < nch; ++c) {
        const uint8_t* s = src + c * Q4_CHUNK;
        uint8_t* o = dst + c * Q4_CHUNK;
        std::memcpy(o, s, 512);                            // the scales are unchanged
        for (unsigned i = 0; i < 256; ++i) {
            uint16_t d;
            std::memcpy(&d, s + 2 * i, 2);
            const uint16_t mn = bf16_rne(-8.0f * bf16_to_f32(d));
            std::memcpy(o + 512 + 2 * i, &mn, 2);
        }
        for (size_t k = 1024; k < Q4_CHUNK; ++k) o[k] = static_cast<uint8_t>(s[k] ^ 0x88);
    }
}

void q4k_to_q4_1_chunks(const uint8_t* src, size_t nch, uint8_t* dst) {
    for (size_t c = 0; c < nch; ++c) {
        const uint8_t* s = src + c * Q4K_CHUNK;
        uint8_t* o = dst + c * Q4_CHUNK;
        // d[g*32 + r] = bf16(S[r] * scales[g*32 + r]), m likewise from M and mins -- the
        // two formats already agree on the meta index, so this is a multiply in place.
        for (unsigned i = 0; i < 256; ++i) {
            uint16_t sh, mh;
            std::memcpy(&sh, s + 4608 + 2 * (i % 32), 2);
            std::memcpy(&mh, s + 4672 + 2 * (i % 32), 2);
            const uint16_t d = bf16_rne(bf16_to_f32(sh) * static_cast<float>(s[i]));
            const uint16_t mn = bf16_rne(bf16_to_f32(mh) * static_cast<float>(s[256 + i]));
            std::memcpy(o + 2 * i, &d, 2);
            std::memcpy(o + 512 + 2 * i, &mn, 2);
        }
        // Q4_K holds a column's 32 rows in 16 contiguous bytes; the pool splits rows 0-15
        // and 16-31 into two 2048-byte planes. Nibble values and parity are unchanged.
        for (unsigned k = 0; k < 256; ++k)
            for (unsigned h = 0; h < 2; ++h)
                std::memcpy(o + 1024 + h * 2048 + k * 8, s + 512 + k * 16 + h * 8, 8);
    }
}

void q8_half_tile(const uint8_t* chunk, unsigned half, uint8_t* dst) {
    std::memset(dst, 0, Q4_CHUNK);
    for (unsigned kb = 0; kb < 8; ++kb)
        for (unsigned r = 0; r < Q8H_ROWS; ++r)
            std::memcpy(dst + 2 * (kb * Q8H_ROWS + r), chunk + 2 * (kb * 32 + Q8H_ROWS * half + r), 2);
    std::memcpy(dst + Q8H_SCALES, chunk + 512 + static_cast<size_t>(half) * Q8H_CODES, Q8H_CODES);
}

/// [rows, cols] -> [cols, dst_rows], columns `rows`..`dst_rows`-1 zeroed. dst_rows == rows
/// is the plain transpose; a wider one is the 16-head DeltaNet's alpha / beta going into
/// dn_glue's 32-lane accumulator (open_kernels/recipes/qwen35.py).
void transpose_bytes(const uint8_t* src, uint64_t rows, uint64_t cols, uint64_t elem, uint64_t dst_rows,
                     uint8_t* dst) {
    std::memset(dst, 0, static_cast<size_t>(cols * dst_rows * elem));
    for (uint64_t r = 0; r < rows; ++r)
        for (uint64_t c = 0; c < cols; ++c)
            std::memcpy(dst + (c * dst_rows + r) * elem, src + (r * cols + c) * elem, elem);
}

void apply(const PackOp& op, const WeightFile& m, int layer, uint8_t* dst, size_t dst_bytes, size_t ch) {
    if (op.op == "std_perm") {
        const std::string name = with_layer(op.tensor, layer);
        if (op.nch == 0 || op.in_dim == 0) fail("std_perm " + name + " without nch / in_dim");
        bounds(op, op.nch * ch, dst_bytes);
        std::vector<uint8_t> tmp;
        const uint8_t* src = nullptr;
        if (!op.split.empty()) {
            // one half of a q8 projection's exact q4_1 split (split_q4_1_chunks)
            const auto* q4 = dynamic_cast<const Q4nxFile*>(&m);
            if (!q4) fail("std_perm " + name + " split " + op.split + ": only a q4nx container stores q8 chunks to split");
            if (q4->chunk_bytes(name) != Q8_CHUNK || ch != Q4_CHUNK)
                fail("std_perm " + name + " split " + op.split + ": the source must be q8 and the pool q4_1");
            const uint8_t* q8 = raw(m, name, (op.chunk0 + op.nch) * Q8_CHUNK) + op.chunk0 * Q8_CHUNK;
            tmp.resize(op.nch * Q4_CHUNK);
            split_q4_1_chunks(q8, op.nch, op.split == "hi", tmp.data());
            src = tmp.data();
        } else {
            src = q4_source(m, name, op.chunk0, op.nch, ch, tmp);
        }
        auto perm = std_perm(op.nch, op.in_dim);
        for (size_t c = 0; c < op.nch; ++c) std::memcpy(dst + op.dst + c * ch, src + perm[c] * ch, ch);
    } else if (op.op == "std_fuse") {
        // OPEN-PACK-CHUNK-FUSE: a std_perm band whose source chunks are half-width. The
        // container holds 32 rows x 128 columns per chunk, in the supertile raster, so each
        // pool chunk is the k-tile's two 128-column halves fused; a column block past the
        // container's own width is synthesised as zeros, which is also the pad from the
        // container's K to the pool's. recipes/pack.py `std_fuse` is the same interpreter.
        const std::string name = with_layer(op.tensor, layer);
        if (op.nch == 0 || op.in_dim == 0 || op.src_dim == 0)
            fail("std_fuse " + name + " without nch / in_dim / src_dim");
        const size_t rg = op.rg ? op.rg : 4;
        if (op.in_dim % 256)
            fail("std_fuse " + name + ": in_dim=" + std::to_string(op.in_dim) + " is not a whole "
                 "number of 256-column k-tiles; the pool chunks would alias onto each other");
        if (op.src_dim % 128)
            fail("std_fuse " + name + ": src_dim=" + std::to_string(op.src_dim) + " is not a whole "
                 "number of 128-column chunks");
        if (op.src_dim > op.in_dim)
            fail("std_fuse " + name + ": the container is " + std::to_string(op.src_dim) +
                 " wide and the pool only " + std::to_string(op.in_dim) + "; a pool narrower than "
                 "the container would drop columns");
        const auto* q4 = dynamic_cast<const Q4nxFile*>(&m);
        if (!q4) fail(name + ": std_fuse reads a q4nx container's supertile raster");
        const size_t src_ch = q4->chunk_bytes(name);
        if (src_ch != Q4_HALF)
            fail(name + ": std_fuse reads " + std::to_string(Q4_HALF) + "-byte chunks (32 rows x "
                 "128 columns) and the container stores it in " + std::to_string(src_ch) + "-byte ones");
        bounds(op, op.nch * ch, dst_bytes);

        const size_t per_band = op.in_dim / 128, ncol128 = op.src_dim / 128;
        size_t nrb = 0;
        for (size_t c = 0; c < op.nch; ++c) nrb = std::max(nrb, 2 * (c / per_band) + c % 2 + 1);
        if (nrb % rg)
            fail("std_fuse " + name + ": " + std::to_string(nrb) + " row blocks is not a whole "
                 "number of " + std::to_string(rg) + "-row-block supertiles");
        const size_t nsrc = nrb * ncol128;
        size_t have = 0;
        const uint8_t* src = raw(m, name, nsrc * Q4_HALF, &have);
        if (have != nsrc * Q4_HALF)
            fail(name + ": " + std::to_string(have / Q4_HALF) + " chunks of " + std::to_string(Q4_HALF) +
                 " B, but a " + std::to_string(op.src_dim) + "-wide tensor covering " +
                 std::to_string(op.nch) + " pool chunks needs exactly " + std::to_string(nsrc));

        const std::vector<uint8_t> zero(Q4_HALF, 0);      // the synthesised column block
        for (size_t c = 0; c < op.nch; ++c) {
            const size_t rb = 2 * (c / per_band) + c % 2, kt = (c % per_band) / 2;
            const uint8_t* lo = 2 * kt < ncol128 ? src + supertile_index(rb, 2 * kt, ncol128, rg) * Q4_HALF
                                                 : zero.data();
            const uint8_t* hi = 2 * kt + 1 < ncol128 ? src + supertile_index(rb, 2 * kt + 1, ncol128, rg) * Q4_HALF
                                                     : zero.data();
            fuse_chunk(lo, hi, dst + op.dst + c * ch);
        }
    } else if (op.op == "std_perm_gguf") {
        // A GGUF [out, in] matmul tensor (Q4_0 / Q4_1 blocks, row-major, fp16
        // scales) into the pool's band order of f32-scale chunks (6144 B): the
        // codes permute into the chunk's 16-lane interleave, the fp16 block
        // scales widen EXACTLY to f32 (open_kernels/gguf_pool.py). Zero loss.
        const std::string name = with_layer(op.tensor, layer);
        if (op.nch == 0 || op.in_dim == 0) fail("std_perm_gguf " + name + " without nch / in_dim");
        const GgufFile& g = dynamic_cast<const GgufFile&>(m);
        const GgufFile::TensorInfo& t = g.tensor(name);
        // Q8_0 / K-quant / i-quant tensors (typical GGUF lm_head / tied embeddings, and
        // the IQ4_XS weights most GGUF downloads ship) are dequantized per row and
        // re-quantized here to the pool's q4 layout with fp16 scales -- the same
        // conversion q4nx-build does offline. The matmul body of the repo stays
        // untouched (Q4_1 etc. go through the byte-exact path below).
        // `fold` != 1 means the plan wants the values scaled before they enter the
        // pool (Granite's attention_multiplier: a container is folded by q4nx-build,
        // a raw GGUF is not). Scaling needs the decoded values, so it forces the
        // requant path even for a Q4_0/Q4_1 source.
        const bool fold = op.fold != 1.0;
        const bool kq = fold || t.type == GgufFile::Type::Q8_0 || t.type == GgufFile::Type::Q4_K ||
                        t.type == GgufFile::Type::Q5_K || t.type == GgufFile::Type::Q6_K ||
                        t.type == GgufFile::Type::IQ4_NL || t.type == GgufFile::Type::IQ4_XS;
        if (!kq && t.type != GgufFile::Type::Q4_0 && t.type != GgufFile::Type::Q4_1)
            fail(name + " is " + GgufFile::type_name(t.type) + "; std_perm_gguf packs Q4_0/Q4_1 and "
                 "requantizes Q8_0/Q4_K/Q5_K/Q6_K/IQ4_NL/IQ4_XS (convert the others with q4nx-build)");
        // Which chunk law the POOL wants (m.chunk_bytes): the GGUF-direct twin
        // widens the block scales exactly to f32 (planes of 1024 B, codes at
        // 2048), and the q4_1 twin this file also serves for the f32-numerics
        // bisect keeps the q4nx law -- the same codes with the scales rounded
        // to bf16 (planes of 512 B, codes at 1024). The GGUF scale was fp16 to
        // begin with, so the bf16 rounding is the only quality step, and the
        // twin that reads it is the one every shipped q4nx model runs on.
        if (ch != Q4_CHUNK && ch != GGUF_CHUNK)
            fail("std_perm_gguf " + name + ": pool chunk " + std::to_string(ch) +
                 " is neither the q4_1 law nor the GGUF-direct twin's f32 one");
        const size_t nb = op.in_dim / 32;                        // k blocks per row
        size_t n = 0;
        raw(m, name, 0, &n);
        // The tensor's rows: the file holds the whole projection, and the plan
        // may read a piece of it (the qwen35 family's gated q is two halves of
        // one tensor, the second op's chunk0 pointing at the second half).
        const uint64_t rows = t.dims.size() > 1 ? t.dims[1] : (t.dims.empty() ? 0 : t.dims[0]);
        const size_t ncol = op.in_dim / 256;
        const uint64_t op_rows = op.nch / ncol * 32;
        size_t chunk0 = op.chunk0;
        if (!chunk0 && rows > op_rows) {
            // Fused GGUF tensors (Phi-3: attn_qkv is [q | k | v], ffn_up is [gate | up])
            // when the manifest names the split projections with chunk0 == 0.
            if (name.find(".self_attn.k_proj.") != std::string::npos && rows > 2 * op_rows)
                chunk0 = static_cast<size_t>((rows - 2 * op_rows) / 32) * ncol;
            else if (name.find(".self_attn.v_proj.") != std::string::npos)
                chunk0 = static_cast<size_t>((rows - op_rows) / 32) * ncol;
            else if (name.find(".mlp.up_proj.") != std::string::npos && rows == 2 * op_rows)
                chunk0 = op.nch;
        }
        const uint64_t out_dim = chunk0 ? rows : op_rows;
        const size_t src_blk = t.type == GgufFile::Type::Q4_1 ? 20 : 18;
        if (!kq && n < static_cast<size_t>(out_dim) * nb * src_blk)
            fail(name + " is " + std::to_string(n) + " B, the plan needs " +
                 std::to_string(static_cast<uint64_t>(out_dim) * nb * src_blk));
        bounds(op, op.nch * ch, dst_bytes);
        auto perm = std_perm(op.nch, op.in_dim);
        // A FUSED q|gate projection is ordered differently by the two sources: a .q4nx
        // container holds [all q | all gate], while llama.cpp writes it per head --
        // [q_head0 | gate_head0 | q_head1 | ...] -- so a pool chunk that should hold q's
        // rows reads the gate's instead (Qwen3.5-0.8B: cos 0.14 for the whole tensor).
        // `qgate_rows` is the rows per head-half, which is what the interleave steps by.
        const size_t qgate_hpb = op.qgate_rows ? op.qgate_rows / 32 : 0;
        // The requant path decodes a row at a time: with a K-quant source that is the
        // whole load-time cost of a GGUF model (a Q4_K_M 8B took minutes). Every chunk
        // is independent -- it reads its own rows of the mapping and writes its own
        // slot -- so the loop is parallel; the decode buffer moves inside it.
#pragma omp parallel for schedule(dynamic) if (kq)
        for (size_t c = 0; c < op.nch; ++c) {
            std::vector<float> rd;                       // this chunk's decode buffer
            uint8_t* d = dst + op.dst + c * ch;
            // the file's chunk this pool chunk reads: the op's range (chunk0)
            // and the permutation inside it
            size_t fc = chunk0 + perm[c];
            if (qgate_hpb) {
                // relative to THIS op's half: the gate op's chunk0 is already the second
                // half, so its first rowblock is chunk0/ncol, not 0. The file's rowblock
                // for that half is then (r / hpb) * 2*hpb + (r % hpb) [+ hpb for the gate].
                const size_t rb0 = chunk0 / ncol;
                const size_t r = fc / ncol - rb0, kt = fc % ncol;
                fc = ((r / qgate_hpb) * 2 * qgate_hpb + (r % qgate_hpb) +
                      (op.qgate_gate ? qgate_hpb : 0)) * ncol + kt;
            }
            gguf_chunk(g, name, t, 32 * (fc / ncol), (fc % ncol) * 8, rows, op.in_dim, ch, op.fold,
                       rd, d);
        }
    } else if (op.op == "bf16_gemm") {
        // `nch` SOURCE chunks from `chunk0`: q8 read exactly unless `via` q4_1 asks for the q4 ops' reading
        const std::string name = with_layer(op.tensor, layer);
        if (op.nch == 0 || op.in_dim == 0 || op.in_dim % 256) fail("bf16_gemm " + name + " without nch / in_dim");
        const size_t ncol = op.in_dim / 256;
        if (op.nch % ncol || (op.nch / ncol) % 2)
            fail("bf16_gemm " + name + ": " + std::to_string(op.nch) + " chunks is not whole 64-row bands of a " +
                 std::to_string(op.in_dim) + "-wide tensor");
        bounds(op, op_bytes(op, ch), dst_bytes);
        const size_t rows = op.nch / ncol * 32;
        uint16_t* out = reinterpret_cast<uint16_t*>(dst + op.dst);
        // q8 / q4_1 SOURCE chunks only exist in a q4nx container (a GGUF tensor
        // goes through std_perm_gguf), so resolve the chunk size through the
        // concrete type. apply() takes WeightFile& so one path serves both
        // containers; this op is q4nx-only.
        const auto* q4 = dynamic_cast<const Q4nxFile*>(&m);
        if (!q4) fail("bf16_gemm " + name + " is not in a q4nx container; only std_perm_gguf packs GGUF tensors");
        if (q4->chunk_bytes(name) == Q8_CHUNK && op.via != "q4_1") {
            bf16_gemm_pack(raw(m, name, (op.chunk0 + op.nch) * Q8_CHUNK) + op.chunk0 * Q8_CHUNK, true, rows, op.in_dim, out);
        } else {
            std::vector<uint8_t> tmp;
            bf16_gemm_pack(q4_source(m, name, op.chunk0, op.nch, Q4_CHUNK, tmp), false, rows, op.in_dim, out);
        }
    } else if (op.op == "bfp16_gemm") {
        // `nch` q4_1 SOURCE chunks from `chunk0`, rows [row0, row0 + rows) of an `n_total`-row weight (OPEN-PREFILL-BFP16)
        const std::string name = with_layer(op.tensor, layer);
        const Bfp16Tiling t;
        if (op.in_dim % 256 || op.in_dim % t.k) fail("bfp16_gemm " + name + ": in_dim " + std::to_string(op.in_dim) + " does not tile");
        const size_t ncol = op.in_dim / 256;
        if (op.nch % ncol) fail("bfp16_gemm " + name + ": " + std::to_string(op.nch) + " chunks is not whole rows of a " +
                                std::to_string(op.in_dim) + "-wide tensor");
        const size_t rows = op.nch / ncol * 32;
        if (op.row0 % t.n || rows % 32 || op.row0 + rows > op.n_total || op.n_total % (t.n * t.cols))
            fail("bfp16_gemm " + name + ": rows [" + std::to_string(op.row0) + ", " + std::to_string(op.row0 + rows) +
                 ") of " + std::to_string(op.n_total) + " do not tile by " + std::to_string(t.n) + " rows x " +
                 std::to_string(t.cols) + " columns");
        if (!op.glu.empty() && (op.row0 || 2 * rows != op.n_total || rows % (t.n / 2)))
            fail("bfp16_gemm " + name + ": a glu half is rows [0, ff) of a 2 ff-row weight, ff whole " +
                 std::to_string(t.n / 2) + "-row chunks");
        bounds(op, op_bytes(op, ch), dst_bytes);
        const auto* q4 = dynamic_cast<const Q4nxFile*>(&m);
        if (!q4) fail("bfp16_gemm " + name + " is not in a q4nx container; only std_perm_gguf packs GGUF tensors");
        if (q4->chunk_bytes(name) == Q8_CHUNK)
            fail("bfp16_gemm " + name + ": the container stores this tensor at q8; only q4_1 sources are packed to bfp16");
        std::vector<uint8_t> tmp;
        bfp16_gemm_pack(q4_source(m, name, op.chunk0, op.nch, Q4_CHUNK, tmp), rows, op.in_dim, op.row0, op.n_total,
                        dst + op.dst, op.glu);
    } else if (op.op == "bfp16_dit") {
        // `nch` SOURCE chunks from `chunk0`, as the q4 ops read them (OPEN-PREFILL-GEMM8)
        const std::string name = with_layer(op.tensor, layer);
        if (op.nch == 0 || op.in_dim == 0 || op.in_dim % 256) fail("bfp16_dit " + name + " without nch / in_dim");
        const size_t ncol = op.in_dim / 256;
        if (op.nch % ncol || (op.nch / ncol) % 4)
            fail("bfp16_dit " + name + ": " + std::to_string(op.nch) + " chunks is not whole 128-row tiles of a " +
                 std::to_string(op.in_dim) + "-wide tensor");
        bounds(op, op_bytes(op, ch), dst_bytes);
        std::vector<uint8_t> tmp;
        bfp16_dit_pack(q4_source(m, name, op.chunk0, op.nch, Q4_CHUNK, tmp), op.nch / ncol * 32, op.in_dim, dst + op.dst);
    } else if (op.op == "q8_perm") {
        // The projection stays at q8: `nch` counts POOL half-tiles (5120 B each, twice the
        // q4_1 bytes of the same tensor) and `chunk0` is a SOURCE file-chunk offset, as it
        // is for std_perm, so the fused [q | gate] split reads the same way in both formats.
        const std::string name = with_layer(op.tensor, layer);
        if (op.nch == 0 || op.in_dim == 0) fail("q8_perm " + name + " without nch / in_dim");
        if (op.nch % 2) fail("q8_perm " + name + ": " + std::to_string(op.nch) +
                             " half-tiles is not a whole number of chunks");
        bounds(op, op.nch * ch, dst_bytes);
        const auto* q4 = dynamic_cast<const Q4nxFile*>(&m);
        if (!q4) fail(name + ": q8_perm requires a q4nx container");
        const size_t src_ch = q4->chunk_bytes(name);
        if (src_ch != Q8_CHUNK)
            fail(name + ": the kernel set streams this projection at q8 (" + std::to_string(Q8_CHUNK) +
                 "-byte chunks) but the container stores it in " + std::to_string(src_ch) +
                 "-byte chunks; re-export the kernels for this container, or force the q4_1 fallback");
        const size_t nsrc = op.nch / 2;
        const uint8_t* src = raw(m, name, (op.chunk0 + nsrc) * Q8_CHUNK) + op.chunk0 * Q8_CHUNK;
        const auto perm = q8_perm(op.nch, op.in_dim);
        for (size_t c = 0; c < op.nch; ++c)
            q8_half_tile(src + perm[c].first * Q8_CHUNK, perm[c].second, dst + op.dst + c * ch);
    } else if (op.op == "expert_stripes") {
        // up / gate as interleaved [up_k | gate_k] stripes per expert, each stripe's chunks
        // transposed (pool chunk c <- file chunk ncol*(c%4) + c/4).
        const std::string un = with_layer(op.up, layer), gn = with_layer(op.gate, layer);
        const uint64_t S = op.stripe_bytes, ns = op.stripes, E = op.experts;
        if (!S || !ns || !E || !op.in_dim) fail("expert_stripes without stripe_bytes / stripes / experts / in_dim");
        bounds(op, E * 2 * ns * S, dst_bytes);
        const size_t ncol = op.in_dim / 256, nchs = S / ch;
        std::vector<size_t> tp(nchs);
        for (size_t c = 0; c < nchs; ++c) tp[c] = ncol * (c % 4) + c / 4;
        std::vector<uint8_t> utmp, gtmp;
        for (uint64_t e = 0; e < E; ++e) {
            for (uint64_t k = 0; k < ns; ++k) {
                const uint8_t* us = q4_source(m, un, (ns * e + k) * nchs, nchs, ch, utmp);
                const uint8_t* gs = q4_source(m, gn, (ns * e + k) * nchs, nchs, ch, gtmp);
                uint8_t* ud = dst + op.dst + (2 * ns * e + 2 * k) * S;
                uint8_t* gd = ud + S;
                for (size_t c = 0; c < nchs; ++c) {
                    std::memcpy(ud + c * ch, us + tp[c] * ch, ch);
                    std::memcpy(gd + c * ch, gs + tp[c] * ch, ch);
                }
            }
        }
    } else if (op.op == "expert_down") {
        // down slices: pool chunk c <- file chunk 2*rt + cg, rt = 4*(c/8) + c%4, cg = (c/4)%2
        const std::string name = with_layer(op.tensor, layer);
        const uint64_t B = op.expert_bytes, E = op.experts;
        if (!B || !E) fail("expert_down without expert_bytes / experts");
        bounds(op, E * B, dst_bytes);
        const size_t nchs = B / ch;
        std::vector<uint8_t> tmp;
        for (uint64_t e = 0; e < E; ++e) {
            const uint8_t* ds = q4_source(m, name, e * nchs, nchs, ch, tmp);
            uint8_t* dd = dst + op.dst + e * B;
            for (size_t c = 0; c < nchs; ++c) {
                size_t rt = 4 * (c / 8) + (c % 4), cg = (c / 4) % 2;
                std::memcpy(dd + c * ch, ds + (2 * rt + cg) * ch, ch);
            }
        }
    } else if (op.op == "expert_stripes_gguf") {
        // The GGUF twin of expert_stripes: the same interleaved [up_k | gate_k] stripes,
        // read from llama.cpp's 3-D [in, n, experts] expert tensors. A container flattens
        // the experts into one [E*n, k] tensor, so there a file chunk index is expert-major
        // and here it is (row of the 3-D tensor, k-tile) -- the destination layout, and
        // every kernel that reads it, is unchanged.
        const std::string un = with_layer(op.up, layer), gn = with_layer(op.gate, layer);
        const uint64_t S = op.stripe_bytes, ns = op.stripes, E = op.experts;
        if (!S || !ns || !E || !op.in_dim)
            fail("expert_stripes_gguf without stripe_bytes / stripes / experts / in_dim");
        bounds(op, E * 2 * ns * S, dst_bytes);
        const auto& g = dynamic_cast<const GgufFile&>(m);
        const auto& tu = g.tensor(un);
        const auto& tg = g.tensor(gn);
        const size_t ncol = op.in_dim / 256, nchs = S / ch;
        if (!ncol || S % ch)
            fail("expert_stripes_gguf " + un + ": " + std::to_string(S) + "-byte stripes are not whole " +
                 std::to_string(ch) + "-byte chunks of a " + std::to_string(op.in_dim) + "-wide tensor");
        // one expert's rows: the stripes cover `ns` x 128 rows, and the file's chunk
        // raster is (rows / 32) x ncol chunks, expert-major.
        const size_t rows_per = ns * 128;
        const size_t rows = rows_per * E;
        for (const auto* t : {&tu, &tg}) {
            if (t->dims.size() < 3)
                fail("expert_stripes_gguf: an expert tensor is [in, n, experts], this one has " +
                     std::to_string(t->dims.size()) + " dim(s)");
            if (t->dims[0] != op.in_dim || t->dims[1] != rows_per || t->dims[2] != E)
                fail("expert_stripes_gguf " + un + ": tensor is [" + std::to_string(t->dims[0]) + ", " +
                     std::to_string(t->dims[1]) + ", " + std::to_string(t->dims[2]) + "], the plan packs [" +
                     std::to_string(op.in_dim) + ", " + std::to_string(rows_per) + ", " + std::to_string(E) + "]");
        }
        if (nchs != 4 * ncol)
            fail("expert_stripes_gguf " + un + ": a " + std::to_string(S) + "-byte stripe is " +
                 std::to_string(nchs) + " chunks, a 128-row x " + std::to_string(op.in_dim) +
                 "-wide stripe is " + std::to_string(4 * ncol));
        std::vector<size_t> tp(nchs);
        for (size_t c = 0; c < nchs; ++c) tp[c] = ncol * (c % 4) + c / 4;
        std::vector<float> rd;
        for (uint64_t e = 0; e < E; ++e) {
            for (uint64_t k = 0; k < ns; ++k) {
                uint8_t* ud = dst + op.dst + (2 * ns * e + 2 * k) * S;
                uint8_t* gd = ud + S;
                for (size_t c = 0; c < nchs; ++c) {
                    const size_t fc = k * nchs + tp[c];
                    const size_t row0 = e * rows_per + 32 * (fc / ncol);
                    const size_t blk0 = (fc % ncol) * 8;
                    gguf_chunk(g, un, tu, row0, blk0, rows, op.in_dim, ch, 1.0, rd, ud + c * ch);
                    gguf_chunk(g, gn, tg, row0, blk0, rows, op.in_dim, ch, 1.0, rd, gd + c * ch);
                }
            }
        }
    } else if (op.op == "expert_down_gguf") {
        // The GGUF twin of expert_down: the same RS=4 down law over llama.cpp's 3-D
        // [in, n, experts] down tensor. The tensor's own K is its fastest dim, which is
        // what the down GEMV streams -- so it comes from the file, not from the plan.
        const std::string name = with_layer(op.tensor, layer);
        const uint64_t B = op.expert_bytes, E = op.experts;
        if (!B || !E) fail("expert_down_gguf without expert_bytes / experts");
        bounds(op, E * B, dst_bytes);
        const auto& g = dynamic_cast<const GgufFile&>(m);
        const auto& t = g.tensor(name);
        if (t.dims.size() < 3)
            fail("expert_down_gguf " + name + ": an expert tensor is [in, n, experts], this one has " +
                 std::to_string(t.dims.size()) + " dim(s)");
        const size_t in_dim = t.dims[0], rows_per = t.dims[1];
        if (t.dims[2] != E)
            fail("expert_down_gguf " + name + ": the tensor has " + std::to_string(t.dims[2]) +
                 " experts, the plan packs " + std::to_string(E));
        if (in_dim % 256)
            fail("expert_down_gguf " + name + ": in_dim=" + std::to_string(in_dim) +
                 " is not a whole number of 256-column k-tiles");
        if (B % ch) fail("expert_down_gguf " + name + ": " + std::to_string(B) + " B is not whole chunks");
        const size_t ncol = in_dim / 256, nchs = B / ch;
        if (nchs * 32 != rows_per * ncol)
            fail("expert_down_gguf " + name + ": " + std::to_string(B) + " B is " + std::to_string(nchs) +
                 " chunks, the tensor's [" + std::to_string(in_dim) + ", " + std::to_string(rows_per) +
                 "] is " + std::to_string(rows_per * ncol / 32));
        std::vector<float> rd;
        for (uint64_t e = 0; e < E; ++e) {
            uint8_t* dd = dst + op.dst + e * B;
            for (size_t c = 0; c < nchs; ++c) {
                const size_t rt = 4 * (c / 8) + (c % 4), cg = (c / 4) % 2;
                const size_t fc = 2 * rt + cg;
                const size_t row0 = e * rows_per + 32 * (fc / ncol);
                const size_t blk0 = (fc % ncol) * 8;
                gguf_chunk(g, name, t, row0, blk0, rows_per * E, in_dim, ch, 1.0, rd, dd + c * ch);
            }
        }
    } else if (op.op == "put") {
        const std::string name = with_layer(op.tensor, layer);
        // div_running: the product of a per-layer scalar over every layer before this one
        // (Gemma 4's layer_output_scale; the residual is carried divided by it, which the
        // RMSNorms that read it cannot see, so the block outputs added to it are divided too).
        // This is direction-exact under RMSNorm, NOT what llama.cpp computes (it multiplies
        // the layer output by s_l instead); the two agree on ranks and logits order and
        // differ by one enormous positive scalar on the residual stream. See
        // recipes/gemma4.py's comment on the layer_output_scale run before touching this;
        // this is also bit-identical with pack.py's _div_running (double product, one rounding).
        double div = 1.0;
        if (!op.div_running.empty())
            for (int j = 0; j < layer; ++j) {
                std::string s = op.div_running;
                s.replace(s.find("{j}"), 3, std::to_string(j));
                div *= scalar_value(m, s);
            }
        if (!(div > 0.0) || !std::isfinite(div)) fail(name + ": div_running product " + std::to_string(div));
        if (dynamic_cast<const GgufFile*>(&m) || div != 1.0) {
            // a GGUF's small weights may be f32/f16: convert to the blob's bf16 layout
            bounds(op, op.cap, dst_bytes);
            pack_norm(m, name, op.cap, dst + op.dst, div);
        } else {
            size_t n = 0;
            const uint8_t* src = raw(m, name, 0, &n);
            if (n > op.cap) fail(name + " is " + std::to_string(n) + " B, its slot holds " + std::to_string(op.cap));
            bounds(op, n, dst_bytes);
            std::memcpy(dst + op.dst, src, n);
        }
    } else if (op.op == "lmhead_q8") {
        // 128-row supertile order. nk = K / 256 k-tiles, a band is 4 row quarters x nk k-tiles,
        // and the file holds chunk (rowblock32, ktile) at rowblock32 * nk + ktile:
        //   pool k <- file (4*(k / per_band) + k % 4) * nk + (k % per_band) / 4,  per_band = 4*nk.
        const std::string name = with_layer(op.tensor, layer);
        const size_t CH8 = op.chunk_bytes;
        if (!CH8) fail("lmhead_q8 without chunk_bytes");
        if (!op.in_dim) fail("lmhead_q8 " + name + " without in_dim");
        const size_t nk = op.in_dim / 256, per_band = 4 * nk;
        if (!nk) fail("lmhead_q8 " + name + ": in_dim " + std::to_string(op.in_dim) + " is under one 256-wide k-tile");
        size_t n = 0;
        const uint8_t* src = raw(m, name, 0, &n);
        if (const auto* q = dynamic_cast<const Q4nxFile*>(&m)) {
            // A q4nx container stores the head already in the engine's q8 chunk
            // raster (Q8NX layout: 256 bf16 scales then 8192 int8 codes, in the
            // supertile order mapped below) -- a straight copy with the supertile
            // permutation is all that is needed.
            size_t nch = n / CH8;
            if (n % CH8)
                fail(name + " is " + std::to_string(n) + " B, not a whole number of " +
                     std::to_string(CH8) + "-byte q8 chunks");
            bounds(op, nch * CH8, dst_bytes);
            for (size_t k = 0; k < nch; ++k) {
                size_t s = k / per_band, r = k % per_band;
                size_t fch = (4 * s + r % 4) * nk + r / 4;
                std::memcpy(dst + op.dst + k * CH8, src + fch * CH8, CH8);
            }
        } else {
            // A raw GGUF stores the head row-major in its own quant. Two cases:
            //   - Q8_0: the source already carries per-32-block int8 codes and an
            //     fp16 scale, so this is a pure remap -- widen the scale to bf16
            //     (the cast the converter does) and permute into the supertile
            //     raster. No requantization, so the head is bit-identical to a
            //     q4nx-build conversion of the same Q8_0 tensor.
            //   - anything else (Q6_K, Q4_K, Q4_0, ...): dequantize each row to
            //     f32 and requantize to int8 + bf16 scale per 32-value group,
            //     matching the pool ops' kq path.
            const GgufFile& g = dynamic_cast<const GgufFile&>(m);
            const GgufFile::TensorInfo& t = g.tensor(name);
            const uint64_t rows = t.dims.size() > 1 ? t.dims[1]
                                : (t.dims.empty() ? 0 : t.dims[0]);
            const size_t ncol = op.in_dim / 256;   // ktiles per row
            if (rows == 0 || ncol == 0) fail(name + " has no rows / ktiles");
            const bool q8_0 = (t.type == GgufFile::Type::Q8_0);
            std::vector<float> rowbuf(op.in_dim);
            // Q8_0 raw block size: 2 B fp16 d + 32 B int8 codes = 34 B per 32 values.
            const size_t q8blk = 34, blocks_per_row = op.in_dim / 32;
            for (uint64_t row = 0; row < rows; ++row) {
                if (q8_0) {
                    // read the raw Q8_0 blocks directly (embed_row would discard codes)
                    const uint8_t* base = g.raw(name, nullptr);
                    // offset into the tensor's data: row * blocks_per_row * 34
                    const uint8_t* rp = base + row * blocks_per_row * q8blk;
                    for (size_t kt = 0; kt < ncol; ++kt) {
                        // The POOL is in supertile raster, not the file's: the q4nx
                        // branch above maps pool chunk k to file chunk
                        // (4*(k/per_band) + k%4)*nk + (k%per_band)/4, so a file chunk
                        // rb*nk + kt belongs at pool chunk (rb/4)*(4*nk) + 4*kt + (rb%4).
                        // Writing the file's own raster here scrambles every 128-row
                        // band: the head then produces plausible, wrong logits (Qwen3.5-0.8B
                        // GGUF-direct: logit cos 0.33 while every layer's hidden state was
                        // at 0.998 -- the container path permutes, which is why it works).
                        const size_t rb = static_cast<size_t>(row) / 32;
                        const size_t k = (rb / 4) * (4 * ncol) + 4 * kt + (rb % 4);
                        if (k * CH8 + CH8 > dst_bytes) continue;
                        const size_t base_col = kt * 256;
                        const unsigned rr = static_cast<unsigned>(row % 32);
                        for (unsigned b = 0; b < 8; ++b) {
                            const uint8_t* blk = rp + (base_col / 32 + b) * q8blk;
                            uint16_t d16;
                            std::memcpy(&d16, blk, 2);
                            // Widen fp16 -> f32 exactly, then RNE to bf16 (the converter's cast). A
                            // hand-rolled widening here lost subnormal scales: it never added the
                            // exponent bias on that branch, and the unsigned exponent wrapped, so a
                            // subnormal d became inf / NaN / ~1e38. A pure Q8_0 head quantized from
                            // BF16 has them wherever a block's amax < 127 * 2^-14.
                            const float dval = fp16_to_f32(d16);
                            uint16_t scale = bf16_rne(dval);
                            std::memcpy(dst + op.dst + k * CH8 + 2 * (b * 32 + rr), &scale, 2);
                            for (unsigned i = 0; i < 32; ++i) {
                                int code = static_cast<int8_t>(blk[2 + i]);
                                const unsigned ci = code_index(rr, b, i);
                                dst[op.dst + k * CH8 + 512 + ci] = static_cast<uint8_t>(code);
                            }
                        }
                    }
                } else {
                    g.embed_row(name, row, op.in_dim, rowbuf.data());
                    for (size_t kt = 0; kt < ncol; ++kt) {
                        // The POOL is in supertile raster, not the file's: the q4nx
                        // branch above maps pool chunk k to file chunk
                        // (4*(k/per_band) + k%4)*nk + (k%per_band)/4, so a file chunk
                        // rb*nk + kt belongs at pool chunk (rb/4)*(4*nk) + 4*kt + (rb%4).
                        // Writing the file's own raster here scrambles every 128-row
                        // band: the head then produces plausible, wrong logits (Qwen3.5-0.8B
                        // GGUF-direct: logit cos 0.33 while every layer's hidden state was
                        // at 0.998 -- the container path permutes, which is why it works).
                        const size_t rb = static_cast<size_t>(row) / 32;
                        const size_t k = (rb / 4) * (4 * ncol) + 4 * kt + (rb % 4);
                        if (k * CH8 + CH8 > dst_bytes) continue;
                        const size_t base_col = kt * 256;
                        const unsigned rr = static_cast<unsigned>(row % 32);
                        for (unsigned b = 0; b < 8; ++b) {
                            float vals[32];
                            float mx = 0.f;
                            bool finite = true;
                            for (unsigned i = 0; i < 32; ++i) {
                                vals[i] = rowbuf[base_col + b * 32 + i];
                                if (!std::isfinite(vals[i])) { finite = false; break; }
                                const float a = vals[i] < 0.f ? -vals[i] : vals[i];
                                if (a > mx) mx = a;
                            }
                            if (!finite) continue;
                            uint16_t scale = (mx > 0.f) ? bf16_rne(mx / 127.f) : 0;
                            const float sf = bf16_to_f32(scale);
                            const float inv = (sf > 0.f) ? 1.f / sf : 0.f;
                            std::memcpy(dst + op.dst + k * CH8 + 2 * (b * 32 + rr), &scale, 2);
                            for (unsigned i = 0; i < 32; ++i) {
                                int q = (int)std::lround(vals[i] * inv);
                                if (q < -127) q = -127;
                                if (q > 127) q = 127;
                                const unsigned ci = code_index(rr, b, i);
                                dst[op.dst + k * CH8 + 512 + ci] = static_cast<uint8_t>(q);
                            }
                        }
                    }
                }
            }
        }

    } else if (op.op == "transpose_banked") {
        const std::string name = with_layer(op.tensor, layer);
        const uint64_t rows = op.rows, cols = op.cols, elem = op.elem;
        if (!rows || !cols || !elem) fail("transpose_banked " + name + " without rows / cols / elem");
        // Check against the actual destination before multiplying dimensions.
        // The bank width is a format constant: no wider GEMV vector is implied.
        const uint64_t banks = rows / 32 + (rows % 32 != 0);
        if (op.dst > dst_bytes || banks > (dst_bytes - op.dst) / 32 / elem / cols)
            fail("transpose_banked " + name + ": destination too small");
        size_t n = 0;
        const uint8_t* src = raw(m, name, rows * cols * elem, &n);
        if (n != rows * cols * elem)
            fail("transpose_banked " + name + ": tensor size does not match rows / cols / elem");
        uint8_t* out = dst + op.dst;
        std::memset(out, 0, banks * cols * 32 * elem);
        for (uint64_t r = 0; r < rows; ++r)
            for (uint64_t c = 0; c < cols; ++c)
                std::memcpy(out + (((r / 32) * cols + c) * 32 + r % 32) * elem,
                            src + (r * cols + c) * elem, elem);
    } else if (op.op == "transpose") {
        const std::string name = with_layer(op.tensor, layer);
        const uint64_t rows = op.rows, cols = op.cols, elem = op.elem ? op.elem : 2;
        const uint64_t dr = op.dst_rows ? op.dst_rows : rows;
        if (!rows || !cols) fail("transpose " + name + " without rows / cols");
        if (dr < rows) fail("transpose " + name + ": dst_rows " + std::to_string(dr) +
                            " is narrower than rows " + std::to_string(rows));
        size_t n = 0;
        const uint8_t* src = nullptr;
        std::vector<uint8_t> tmp;
        if (elem == 2 && consts_as_bf16(m, name, rows * cols, tmp)) src = tmp.data();
        if (!src) {
            src = raw(m, name, rows * cols * elem, &n);
            if (n != rows * cols * elem)
                fail(name + " is " + std::to_string(n) + " B, not a [" + std::to_string(rows) + ", " +
                     std::to_string(cols) + "] tensor of " + std::to_string(elem) + "-byte values");
        }
        bounds(op, cols * dr * elem, dst_bytes);
        transpose_bytes(src, rows, cols, elem, dr, dst + op.dst);
    } else if (op.op == "conv_transpose") {
        // conv1d bf16 [taps][groups*width] -> [groups][taps][width]
        const std::string name = with_layer(op.tensor, layer);
        const uint64_t taps = op.taps, groups = op.groups, width = op.width;
        if (!taps || !groups || !width) fail("conv_transpose without taps / groups / width");
        size_t n = 0;
        const uint8_t* src = nullptr;
        std::vector<uint8_t> tmp;
        if (consts_as_bf16(m, name, taps * groups * width, tmp)) src = tmp.data();
        if (!src) {
            src = raw(m, name, taps * groups * width * 2, &n);
            if (n != taps * groups * width * 2) fail(name + " is not bf16[" + std::to_string(taps) + ", " + std::to_string(groups * width) + "]");
        }
        // A GGUF stores the filter CHANNELS-major: llama.cpp's fastest dim is the tap, so
        // `ssm_conv1d.weight` is [channels][taps], while a .q4nx container stores it
        // taps-major. The shuffle below is written for taps-major, so re-order first --
        // without this the flat values are read taps-major and every channel gets another
        // channel's filter. (Qwen3.5-0.8B GGUF-direct: layer 0 came out at cos 0.12 against
        // the same weights through a container, and this was the only differing tensor.)
        if (auto* gg = dynamic_cast<const GgufFile*>(&m)) {
            const auto& d = gg->tensor(name).dims;
            if (d.size() == 2 && static_cast<uint64_t>(d[0]) == taps &&
                static_cast<uint64_t>(d[1]) == groups * width) {
                std::vector<uint8_t> t(taps * groups * width * 2);
                for (uint64_t c = 0; c < groups * width; ++c)
                    for (uint64_t tp = 0; tp < taps; ++tp)
                        std::memcpy(&t[(tp * groups * width + c) * 2], src + (c * taps + tp) * 2, 2);
                tmp.swap(t);
                src = tmp.data();
            }
        }
        bounds(op, taps * groups * width * 2, dst_bytes);
        for (uint64_t g = 0; g < groups; ++g)
            for (uint64_t t = 0; t < taps; ++t)
                std::memcpy(dst + op.dst + (g * taps + t) * width * 2, src + (t * groups * width + g * width) * 2, width * 2);
    } else if (op.op == "gx_q8") {
        // GPT-OSS's attention projections: Q8_0 [out, in] -> half-tiles, the bias in the pad
        const std::string name = with_layer(op.tensor, layer);
        size_t rows, cols;
        gx_gguf(m, name, GgufFile::Type::Q8_0, &rows, &cols, 0);
        const size_t nbytes = op.rows / 64 * gx_band_bytes(op.in_dim);
        bounds(op, nbytes, dst_bytes);
        std::vector<float> bias;
        if (!op.bias.empty()) bias = gx_bias(m, with_layer(op.bias, layer), rows);
        gx_tiles(raw(m, name, rows * cols / 32 * 34), false, rows, cols, op.rows, op.in_dim,
                 bias.empty() ? nullptr : bias.data(), dst + op.dst);
    } else if (op.op == "gx_ug" || op.op == "gx_down" || op.op == "gx_ug_mx4" || op.op == "gx_down_mx4") {
        // GPT-OSS's routed experts, MXFP4 [E, rows, cols]: expert e at dst + e * expert_bytes, as q8
        // half-tiles or (_mx4) as the GGUF's own nibbles in mx4 half-tiles. gx_ug interleaves up and
        // gate per band, [band][up | gate], the order gx_moe's cores stream them in.
        const bool ug = op.op == "gx_ug" || op.op == "gx_ug_mx4";
        const bool mx4 = op.op == "gx_ug_mx4" || op.op == "gx_down_mx4";
        const std::string un = with_layer(ug ? op.up : op.tensor, layer);
        size_t rows, cols;
        gx_gguf(m, un, GgufFile::Type::MXFP4, &rows, &cols, op.experts);
        const size_t per = rows * cols / 32 * 17, band = gx_band_bytes(op.in_dim, mx4 ? GX_MX4_TILE : GX_TILE);
        const size_t nb = op.rows / 64;
        auto tiles = [&](const uint8_t* s, const float* bias, uint8_t* d) {
            if (mx4) gx_mx4_tiles(s, rows, cols, op.rows, op.in_dim, bias, d);
            else gx_tiles(s, true, rows, cols, op.rows, op.in_dim, bias, d);
        };
        const size_t one = nb * band * (ug ? 2 : 1);
        if (one > op.expert_bytes) fail(op.op + " " + un + ": an expert is " + std::to_string(one) + " B, its slot " +
                                        std::to_string(op.expert_bytes));
        bounds(op, (op.experts - 1) * op.expert_bytes + one, dst_bytes);
        const uint8_t* us = raw(m, un, op.experts * per);
        const uint8_t* gs = nullptr;
        std::vector<float> ub, gb;
        if (ug) {
            const std::string gn = with_layer(op.gate, layer);
            size_t gr, gc;
            gx_gguf(m, gn, GgufFile::Type::MXFP4, &gr, &gc, op.experts);
            if (gr != rows || gc != cols) fail(op.op + ": " + un + " and " + gn + " differ in shape");
            gs = raw(m, gn, op.experts * per);
            if (!op.up_bias.empty()) ub = gx_bias(m, with_layer(op.up_bias, layer), op.experts * rows);
            if (!op.gate_bias.empty()) gb = gx_bias(m, with_layer(op.gate_bias, layer), op.experts * rows);
        } else if (!op.bias.empty()) {
            fail(op.op + " " + un + ": the down bias is the route's (bsum), not the pool's");
        }
        std::vector<uint8_t> tu(nb * band), tg(ug ? nb * band : 0);
        for (uint64_t e = 0; e < op.experts; ++e) {
            uint8_t* d = dst + op.dst + e * op.expert_bytes;
            if (!ug) {
                tiles(us + e * per, nullptr, d);
                continue;
            }
            tiles(us + e * per, ub.empty() ? nullptr : ub.data() + e * rows, tu.data());
            tiles(gs + e * per, gb.empty() ? nullptr : gb.data() + e * rows, tg.data());
            for (size_t b = 0; b < nb; ++b) {
                std::memcpy(d + 2 * b * band, tu.data() + b * band, band);
                std::memcpy(d + (2 * b + 1) * band, tg.data() + b * band, band);
            }
        }
    } else if (op.op == "gp_bfp16") {
        // the block prefill's attention projections (OPEN-PREFILL-GPTOSS): a GGUF tensor's rows (its
        // own blocks, Q8_0 exactly, GgufFile::embed_row) as rows [row0, row0 + rows) of an n_total-row
        // gemm_bfp16 B operand of K = in_dim -- bfp16_gemm's law with K any multiple of the 64-wide
        // tile, not of a q4 chunk's 256
        const std::string name = with_layer(op.tensor, layer);
        const Bfp16Tiling t;
        const auto* g = dynamic_cast<const GgufFile*>(&m);
        if (!g) fail("gp_bfp16 " + name + " reads a GGUF");
        const GgufFile::TensorInfo& ti = g->tensor(name);
        const size_t rows = ti.dims.size() >= 2 ? ti.dims[1] : 1u;
        if (ti.dims.empty() || ti.dims[0] != op.in_dim || op.in_dim % t.k || op.row0 % t.n ||
            op.row0 + rows > op.n_total || op.n_total % (t.n * t.cols))
            fail("gp_bfp16 " + name + ": [" + std::to_string(rows) + ", " + std::to_string(ti.dims.empty() ? 0 : ti.dims[0]) +
                 "] at row " + std::to_string(op.row0) + " does not tile an [" + std::to_string(op.n_total) + ", " +
                 std::to_string(op.in_dim) + "] operand");
        bounds(op, op_bytes(op, ch), dst_bytes);
        uint8_t* out = dst + op.dst;
#pragma omp parallel
        {
            std::vector<float> w(op.in_dim);
#pragma omp for
            for (long long rr = 0; rr < static_cast<long long>(rows); ++rr) {
                const size_t r = static_cast<size_t>(rr);
                g->embed_row(name, r, op.in_dim, w.data());
                uint8_t* row = out + bfp16_b_offset(op.row0 + r, 0, op.n_total, op.in_dim, t);
                for (size_t cb = 0; cb < op.in_dim / 8; ++cb) {
                    float* b = w.data() + cb * 8;
                    if (op.split == "lo") {
                        // what bfp16 leaves of the block: the hi half's remainder, encoded in its own block
                        uint8_t hi[BFP16_BLOCK_BYTES];
                        bfp16_encode_block(b, hi);
                        const float sc = std::ldexp(1.0f, static_cast<int>(hi[0]) - 133);
                        for (size_t i = 0; i < 8; ++i) b[i] -= static_cast<float>(static_cast<int8_t>(hi[1 + i])) * sc;
                    }
                    bfp16_encode_block(b, row + bfp16_b_row_block(0, cb, t));
                }
            }
        }
    } else if (op.op == "gp_mx4") {
        // the block prefill's routed experts (OPEN-PREFILL-GPTOSS): MXFP4 [E, rows, cols] as gemm_bfp16 GB_B=mx4
        // operands of n_total x in_dim, expert e at dst + e * expert_bytes. gate | up: B rows [0, rows) are the
        // gate's, [n_total / 2, n_total / 2 + rows) the up's; down: rows [0, rows). Every other row is zero.
        const bool glu = op.tensor.empty();
        const std::string gn = with_layer(glu ? op.gate : op.tensor, layer);
        size_t rows, cols;
        gx_gguf(m, gn, GgufFile::Type::MXFP4, &rows, &cols, op.experts);
        const size_t N = op.n_total, K = op.in_dim, per = rows * cols / 32 * 17;
        if (N % 1024 || K % 64 || K < cols || (glu ? 2 * rows > N : rows > N) ||
            N / 128 * (K / 64) * GP_MX4_TILE > op.expert_bytes)
            fail("gp_mx4 " + gn + ": [" + std::to_string(rows) + ", " + std::to_string(cols) + "] experts do not fit a " +
                 std::to_string(N) + " x " + std::to_string(K) + " operand in " + std::to_string(op.expert_bytes) + " B");
        bounds(op, op.experts * op.expert_bytes, dst_bytes);
        const uint8_t* gs = raw(m, gn, op.experts * per);
        const uint8_t* us = nullptr;
        if (glu) {
            const std::string un = with_layer(op.up, layer);
            size_t ur, uc;
            gx_gguf(m, un, GgufFile::Type::MXFP4, &ur, &uc, op.experts);
            if (ur != rows || uc != cols) fail("gp_mx4: " + gn + " and " + un + " differ in shape");
            us = raw(m, un, op.experts * per);
        }
        for (uint64_t e = 0; e < op.experts; ++e) {
            uint8_t* d = dst + op.dst + e * op.expert_bytes;
            if (!glu) {
                gp_mx4_pack(gs + e * per, cols, N, K, [&](size_t n) { return n < rows ? static_cast<long>(n) : -1L; }, d);
                continue;
            }
            // one operand from two tensors: pack the gate's rows, then the up's into the second half of a copy, and
            // merge (each tile row is wholly one tensor's, so OR-ing the zeroed rows of the other in is exact)
            std::vector<uint8_t> tmp(N / 128 * (K / 64) * GP_MX4_TILE);
            gp_mx4_pack(gs + e * per, cols, N, K, [&](size_t n) { return n < rows ? static_cast<long>(n) : -1L; }, d);
            gp_mx4_pack(us + e * per, cols, N, K,
                        [&](size_t n) { return n >= N / 2 && n - N / 2 < rows ? static_cast<long>(n - N / 2) : -1L; },
                        tmp.data());
            for (size_t i = 0; i < tmp.size(); ++i) d[i] |= tmp[i];
        }
    } else if (op.op == "gx_lmhead") {
        // the head, Q8_0 [vocab, hidden] -> lm_head_q8.h's f32-scale chunks (32 rows x 256 K, 9216 B):
        // f32 scale[kb*32 + r] at [0:1024), code[(r/16)*4096 + k*16 + r%16] at [1024:9216); 128-row
        // bands of 4 quarters x K/256 k-tiles, chunk c of a band the quarter c % 4 and k-tile c / 4
        const std::string name = with_layer(op.tensor, layer);
        size_t rows, cols;
        gx_gguf(m, name, GgufFile::Type::Q8_0, &rows, &cols, 0);
        if (rows > op.rows || cols > op.in_dim || op.rows % 128 || op.in_dim % 256 || cols % 32)
            fail("gx_lmhead " + name + ": [" + std::to_string(rows) + ", " + std::to_string(cols) +
                 "] does not pad to [" + std::to_string(op.rows) + ", " + std::to_string(op.in_dim) + "]");
        const size_t nB = op.rows / 128, nkt = op.in_dim / 256, bpr = cols / 32;
        bounds(op, nB * nkt * 4 * GX_HEAD, dst_bytes);
        const uint8_t* src = raw(m, name, rows * cols / 32 * 34);
        uint8_t* out = dst + op.dst;
#pragma omp parallel for schedule(dynamic)
        for (long long Bl = 0; Bl < static_cast<long long>(nB); ++Bl) {
            const size_t B = static_cast<size_t>(Bl);
            for (size_t kt = 0; kt < nkt; ++kt)
                for (size_t q = 0; q < 4; ++q) {
                    uint8_t* c = out + ((B * nkt + kt) * 4 + q) * GX_HEAD;
                    std::memset(c, 0, GX_HEAD);
                    for (size_t r = 0; r < 32; ++r) {
                        const size_t row = 128 * B + 32 * q + r;
                        if (row >= rows) continue;
                        for (size_t kb = 0; kb < 8; ++kb) {
                            const size_t blk = kt * 8 + kb;
                            if (blk >= bpr) continue;
                            const uint8_t* b = src + (row * bpr + blk) * 34;
                            uint16_t h;
                            std::memcpy(&h, b, 2);
                            const float d = fp16_to_f32(h);
                            std::memcpy(c + 4 * (kb * 32 + r), &d, 4);
                            for (size_t i = 0; i < 32; ++i)
                                c[1024 + (r / 16) * 4096 + (kb * 32 + i) * 16 + r % 16] = b[2 + i];
                        }
                    }
                }
        }
    } else {
        fail("unknown pack op " + op.op);
    }
}

void pack_pool(const Manifest& m, const LayerType& lt, const WeightFile& f, int layer, uint8_t* dst) {
    std::memset(dst, 0, m.pool_bytes);
    for (const auto& op : lt.pool) apply(op, f, layer, dst, m.pool_bytes, m.chunk_bytes);
}

void pack_consts(const Manifest& m, const LayerType& lt, const WeightFile& f, int layer, uint8_t* dst) {
    std::memset(dst, 0, lt.consts_bytes);
    for (const auto& op : lt.consts) apply(op, f, layer, dst, lt.consts_bytes, m.chunk_bytes);
}

void pack_lmhead(const Manifest& m, const WeightFile& f, uint8_t* out) {
    std::memset(out, 0, m.lmhead_pool_bytes);
    for (const auto& op : m.lmhead_ops) apply(op, f, 0, out, m.lmhead_pool_bytes, m.chunk_bytes);
}

// Which of (t, h, w) rotary pair i takes. transformers' apply_interleaved_mrope: every pair
// starts as t; pairs a, a + 3, a + 6, ... below 3 * section[a] take axis a for a = h, w.
// The chunked layout is [t x s0 | h x s1 | w x s2].
static int mrope_axis(size_t i, const std::vector<int>& section, bool interleaved) {
    if (section.size() != 3) return 0;
    if (interleaved) {
        const int a = static_cast<int>(i % 3);
        return (a != 0 && i < 3 * static_cast<size_t>(section[a])) ? a : 0;
    }
    const size_t s0 = static_cast<size_t>(section[0]), s1 = s0 + static_cast<size_t>(section[1]);
    return i < s0 ? 0 : (i < s1 ? 1 : 2);
}

void build_ptab_record(const Manifest& m, const RowGlobal& g, size_t row, const double pos[3],
                       const std::vector<int>& section, bool interleaved, uint8_t* r) {
    // RoPE over the first rotary_dim dims of a head, half-split pairs (i, i + rot/2), the recipe's theta:
    // [i32 pos | i32 nf | cos f32[rot/2] @512 | sin f32[rot/2] right after] (attn.h reads [cos | sin] at +512)
    // the table's own pairs and record size (Gemma 4 has two tables of different widths);
    // a table that names neither is the layout's, which is every other family
    const size_t half = (g.rotary_dim ? g.rotary_dim : m.rotary_dim) / 2;
    const size_t row_bytes = g.per_row ? g.per_row : m.ptab_row;
    if (512 + 8 * half > row_bytes) fail("the rotary dim does not fit the position record");
    std::memset(r, 0, row_bytes);
    uint64_t start, nf64;
    stream_patch::attn_window(row, g.window, &start, &nf64);
    int32_t valid = static_cast<int32_t>(row - start), nf = static_cast<int32_t>(nf64);
    std::memcpy(r, &valid, 4);
    std::memcpy(r + 4, &nf, 4);
    // Phi-3's longrope: row `row` takes long_inv_freq once it reaches switch_row (HF's own
    // seq_len = pos + 1 > original_max_position_embeddings rule, applied per row since this
    // engine computes one row per token). kSwitchNever means every other family: always inv_freq.
    const std::vector<double>& freq = (row >= g.switch_row) ? g.long_inv_freq : g.inv_freq;
    for (size_t i = 0; i < half; ++i) {
        double ang = pos[mrope_axis(i, section, interleaved)] * freq[i];
        float c = static_cast<float>(g.scale * std::cos(ang)), s = static_cast<float>(g.scale * std::sin(ang));
        std::memcpy(r + 512 + 4 * i, &c, 4);
        std::memcpy(r + 512 + 4 * half + 4 * i, &s, 4);
    }
}

void build_ptab(const Manifest& m, const RowGlobal& g, size_t rows, uint8_t* t) {
    for (size_t p = 0; p < rows; ++p) {
        const double pos[3] = {static_cast<double>(p), static_cast<double>(p), static_cast<double>(p)};
        build_ptab_record(m, g, p, pos, {}, false, t + p * (g.per_row ? g.per_row : m.ptab_row));
    }
}

}  // namespace pools
}  // namespace open_qwen36
