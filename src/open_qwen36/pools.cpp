/// \file pools.cpp
/// \brief The packing-plan interpreter (see pools.hpp).
#include "open_qwen36/pools.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

/// pool chunk index -> file chunk index for a standard [out, in] matmul tensor:
/// a band is 64 rows x in_dim = in_dim/128 chunks; inside its band chunk i covers
/// row half i%2 and k-tile i/2 (gemv_q4.h's band law); file chunk f covers rows
/// 32*(f/ncol), cols 256*(f%ncol). Same law as recipes/pack.py (which documents
/// its equivalence with the form phlegm verified against OFLM's captured pools).
std::vector<size_t> std_perm(size_t nch, size_t in_dim) {
    size_t ncol = in_dim / 256, per_band = in_dim / 128;
    std::vector<size_t> perm(nch);
    for (size_t c = 0; c < nch; ++c) {
        size_t rows0 = 64 * (c / per_band) + 32 * (c % 2);
        size_t cols0 = 256 * ((c % per_band) / 2);
        perm[c] = (rows0 / 32) * ncol + cols0 / 256;
    }
    return perm;
}

constexpr size_t Q8_CHUNK = 8704;    // 256 bf16 scales then 8192 int8 codes
constexpr size_t Q4_CHUNK = 5120;    // 256 bf16 d, 256 bf16 m, then 4096 B of nibbles
constexpr size_t Q4K_CHUNK = 4736;   // 256 uint8 scales, 256 uint8 mins, 4096 B of nibbles, 32 bf16 S, 32 bf16 M
constexpr size_t Q8H_SCALES = 256;   // a half-tile's 128 bf16 scales
constexpr size_t Q8H_CODES = 4096;   // a half-tile's 4096 int8 codes
constexpr unsigned Q8H_ROWS = 16;

/// pool half-tile index -> (file chunk index, half) for a q8 projection: a band is 64 rows
/// x in_dim = in_dim/64 half-tiles; half-tile c inside its band covers rows 16*(c%4) of the
/// band and k-tile c/4, so its source is file chunk (2*band + (c%4)/2) at half (c%4)%2
/// (gemv_q8.h's band law; recipes/pack.py q8_perm is the same law in NumPy).
std::vector<std::pair<size_t, unsigned>> q8_perm(size_t nch, size_t in_dim) {
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

const uint8_t* raw(const Q4nxFile& m, const std::string& name, size_t need, size_t* got = nullptr) {
    size_t n = 0;
    const uint8_t* p = m.raw(name, &n);
    if (n < need) fail(name + " is " + std::to_string(n) + " B, the plan needs " + std::to_string(need));
    if (got) *got = n;
    return p;
}

/// What a chunk size the packer does not read probably is, for the refusal message.
std::string chunk_guess(size_t ch) {
    if (ch == 1280 || ch == 2560) return "a smaller chunk geometry (" + std::to_string(ch * 8192 / Q4_CHUNK) +
                                         " values per chunk instead of 8192)";
    return "not a chunk format this packer knows";
}

/// `nch` q4_1 chunks of `name` starting at chunk `chunk0`, whatever the container stores.
/// A q4_1 tensor is a view into the mapping; a q8 one is re-quantized into `tmp` first
/// (OPEN-PACK-PLAN: a q8 source is accepted transparently by every q4 pack op, because the
/// two formats hold the SAME 32-row x 256-column tile, so no chunk index law changes).
const uint8_t* q4_source(const Q4nxFile& m, const std::string& name, size_t chunk0, size_t nch, size_t ch,
                         std::vector<uint8_t>& tmp) {
    const size_t src_ch = m.chunk_bytes(name);
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

}  // namespace

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

void apply(const PackOp& op, const Q4nxFile& m, int layer, uint8_t* dst, size_t dst_bytes, size_t ch) {
    if (op.op == "std_perm") {
        const std::string name = with_layer(op.tensor, layer);
        if (op.nch == 0 || op.in_dim == 0) fail("std_perm " + name + " without nch / in_dim");
        bounds(op, op.nch * ch, dst_bytes);
        std::vector<uint8_t> tmp;
        const uint8_t* src = q4_source(m, name, op.chunk0, op.nch, ch, tmp);
        auto perm = std_perm(op.nch, op.in_dim);
        for (size_t c = 0; c < op.nch; ++c) std::memcpy(dst + op.dst + c * ch, src + perm[c] * ch, ch);
    } else if (op.op == "q8_perm") {
        // The projection stays at q8: `nch` counts POOL half-tiles (5120 B each, twice the
        // q4_1 bytes of the same tensor) and `chunk0` is a SOURCE file-chunk offset, as it
        // is for std_perm, so the fused [q | gate] split reads the same way in both formats.
        const std::string name = with_layer(op.tensor, layer);
        if (op.nch == 0 || op.in_dim == 0) fail("q8_perm " + name + " without nch / in_dim");
        if (op.nch % 2) fail("q8_perm " + name + ": " + std::to_string(op.nch) +
                             " half-tiles is not a whole number of chunks");
        bounds(op, op.nch * ch, dst_bytes);
        const size_t src_ch = m.chunk_bytes(name);
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
    } else if (op.op == "put") {
        const std::string name = with_layer(op.tensor, layer);
        size_t n = 0;
        const uint8_t* src = raw(m, name, 0, &n);
        if (n > op.cap) fail(name + " is " + std::to_string(n) + " B, its slot holds " + std::to_string(op.cap));
        bounds(op, n, dst_bytes);
        std::memcpy(dst + op.dst, src, n);
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
        size_t nch = n / CH8;
        bounds(op, nch * CH8, dst_bytes);
        for (size_t k = 0; k < nch; ++k) {
            size_t s = k / per_band, r = k % per_band;
            size_t fch = (4 * s + r % 4) * nk + r / 4;
            std::memcpy(dst + op.dst + k * CH8, src + fch * CH8, CH8);
        }
    } else if (op.op == "transpose") {
        const std::string name = with_layer(op.tensor, layer);
        const uint64_t rows = op.rows, cols = op.cols, elem = op.elem ? op.elem : 2;
        const uint64_t dr = op.dst_rows ? op.dst_rows : rows;
        if (!rows || !cols) fail("transpose " + name + " without rows / cols");
        if (dr < rows) fail("transpose " + name + ": dst_rows " + std::to_string(dr) +
                            " is narrower than rows " + std::to_string(rows));
        size_t n = 0;
        const uint8_t* src = raw(m, name, rows * cols * elem, &n);
        if (n != rows * cols * elem)
            fail(name + " is " + std::to_string(n) + " B, not a [" + std::to_string(rows) + ", " +
                 std::to_string(cols) + "] tensor of " + std::to_string(elem) + "-byte values");
        bounds(op, cols * dr * elem, dst_bytes);
        transpose_bytes(src, rows, cols, elem, dr, dst + op.dst);
    } else if (op.op == "conv_transpose") {
        // conv1d bf16 [taps][groups*width] -> [groups][taps][width]
        const std::string name = with_layer(op.tensor, layer);
        const uint64_t taps = op.taps, groups = op.groups, width = op.width;
        if (!taps || !groups || !width) fail("conv_transpose without taps / groups / width");
        size_t n = 0;
        const uint8_t* src = raw(m, name, taps * groups * width * 2, &n);
        if (n != taps * groups * width * 2) fail(name + " is not bf16[" + std::to_string(taps) + ", " + std::to_string(groups * width) + "]");
        bounds(op, n, dst_bytes);
        for (uint64_t g = 0; g < groups; ++g)
            for (uint64_t t = 0; t < taps; ++t)
                std::memcpy(dst + op.dst + (g * taps + t) * width * 2, src + (t * groups * width + g * width) * 2, width * 2);
    } else {
        fail("unknown pack op " + op.op);
    }
}

void pack_pool(const Manifest& m, const LayerType& lt, const Q4nxFile& f, int layer, uint8_t* dst) {
    std::memset(dst, 0, m.pool_bytes);
    for (const auto& op : lt.pool) apply(op, f, layer, dst, m.pool_bytes, m.chunk_bytes);
}

void pack_consts(const Manifest& m, const LayerType& lt, const Q4nxFile& f, int layer, uint8_t* dst) {
    std::memset(dst, 0, lt.consts_bytes);
    for (const auto& op : lt.consts) apply(op, f, layer, dst, lt.consts_bytes, m.chunk_bytes);
}

void pack_lmhead(const Manifest& m, const Q4nxFile& f, uint8_t* out) {
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
    const size_t half = m.rotary_dim / 2;
    if (512 + 8 * half > m.ptab_row) fail("the rotary dim does not fit the position record");
    std::memset(r, 0, m.ptab_row);
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
        build_ptab_record(m, g, p, pos, {}, false, t + p * m.ptab_row);
    }
}

}  // namespace pools
}  // namespace open_qwen36
