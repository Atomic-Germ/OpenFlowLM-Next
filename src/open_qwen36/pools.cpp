/// \file pools.cpp
/// \brief The packing-plan interpreter (see pools.hpp).
#include "open_qwen36/pools.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "open_qwen36/gguf_file.hpp"
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

/// pool chunk index -> file chunk index for a standard [out, in] matmul tensor:
/// a band is 64 rows x in_dim = in_dim/128 chunks; inside its band chunk i covers
/// row half i%2 and k-tile i/2 (gemv_q4.h's band law); file chunk f covers rows
/// 32*(f/ncol), cols 256*(f%ncol). Same law as recipes/pack.py (which documents
/// its equivalence with the form phlegm verified against FLM's captured pools).
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

const uint8_t* raw(const WeightFile& m, const std::string& name, size_t need, size_t* got = nullptr) {
    size_t n = 0;
    const uint8_t* p = m.raw(name, &n);
    if (n < need) fail(name + " is " + std::to_string(n) + " B, the plan needs " + std::to_string(need));
    if (got) *got = n;
    return p;
}

}  // namespace

void pack_norm(const WeightFile& f, const std::string& name, size_t bytes, uint8_t* dst) {
    // A small weight (a layernorm) as bf16: the q4nx container stores it bf16;
    // a GGUF may store it f32 / f16 / bf16 -- convert exactly as the container does.
    size_t n = 0;
    const uint8_t* src = raw(f, name, 0, &n);
    auto* g = dynamic_cast<const GgufFile*>(&f);
    if (!g) {
        if (n != bytes) fail(name + " is " + std::to_string(n) + " B, the slot holds " + std::to_string(bytes));
        std::memcpy(dst, src, n);
        return;
    }
    switch (g->tensor(name).type) {
        case GgufFile::Type::BF16: {
            if (n != bytes) fail(name + " is " + std::to_string(n) + " B, the slot holds " + std::to_string(bytes));
            std::memcpy(dst, src, n);
            break;
        }
        case GgufFile::Type::F16: {
            if (n != bytes) fail(name + " is " + std::to_string(n) + " B of fp16, the slot holds " + std::to_string(bytes) + " B of bf16");
            for (size_t i = 0; i < bytes / 2; ++i) {
                uint16_t u;
                std::memcpy(&u, src + 2 * i, 2);
                const float v = fp16_to_f32(u);
                const uint16_t b = f32_to_bf16(v);
                std::memcpy(dst + 2 * i, &b, 2);
            }
            break;
        }
        case GgufFile::Type::F32: {
            if (n != 2 * bytes) fail(name + " is " + std::to_string(n) + " B of f32, the slot holds " + std::to_string(bytes) + " B of bf16");
            for (size_t i = 0; i < bytes / 2; ++i) {
                float v;
                std::memcpy(&v, src + 4 * i, 4);
                const uint16_t b = f32_to_bf16(v);
                std::memcpy(dst + 2 * i, &b, 2);
            }
            break;
        }
        default:
            fail(name + " is " + GgufFile::type_name(g->tensor(name).type) + "; a layernorm must be f32 / f16 / bf16");
    }
}

void apply(const PackOp& op, const WeightFile& m, int layer, uint8_t* dst, size_t dst_bytes, size_t ch) {
    if (op.op == "std_perm") {
        const std::string name = with_layer(op.tensor, layer);
        if (op.nch == 0 || op.in_dim == 0) fail("std_perm " + name + " without nch / in_dim");
        bounds(op, op.nch * ch, dst_bytes);
        const uint8_t* src = raw(m, name, (op.chunk0 + op.nch) * ch) + op.chunk0 * ch;
        auto perm = std_perm(op.nch, op.in_dim);
        for (size_t c = 0; c < op.nch; ++c) std::memcpy(dst + op.dst + c * ch, src + perm[c] * ch, ch);
    } else if (op.op == "std_perm_gguf") {
        // A GGUF [out, in] matmul tensor (Q4_0 / Q4_1 blocks, row-major, fp16
        // scales) into the pool's band order of f32-scale chunks (6144 B): the
        // codes permute into the chunk's 16-lane interleave, the fp16 block
        // scales widen EXACTLY to f32 (open_kernels/gguf_pool.py). Zero loss.
        const std::string name = with_layer(op.tensor, layer);
        if (op.nch == 0 || op.in_dim == 0) fail("std_perm_gguf " + name + " without nch / in_dim");
        const GgufFile& g = dynamic_cast<const GgufFile&>(m);
        const GgufFile::TensorInfo& t = g.tensor(name);
        if (t.type != GgufFile::Type::Q4_0 && t.type != GgufFile::Type::Q4_1)
            fail(name + " is " + GgufFile::type_name(t.type) + "; std_perm_gguf packs Q4_0/Q4_1 only "
                 "(convert K-quants with q4nx-build)");
        const bool has_min = t.type == GgufFile::Type::Q4_1;
        const size_t blk = has_min ? 20 : 18;
        const size_t nb = op.in_dim / 32;                        // k blocks per row
        size_t n = 0;
        const uint8_t* src = raw(m, name, 0, &n);
        const uint64_t out_dim = op.nch / (op.in_dim / 256) * 32;
        if (n < static_cast<size_t>(out_dim) * nb * blk)
            fail(name + " is " + std::to_string(n) + " B, the plan needs " +
                 std::to_string(static_cast<uint64_t>(out_dim) * nb * blk));
        bounds(op, op.nch * ch, dst_bytes);
        auto perm = std_perm(op.nch, op.in_dim);
        const size_t ncol = op.in_dim / 256;
        for (size_t c = 0; c < op.nch; ++c) {
            uint8_t* d = dst + op.dst + c * ch;
            const size_t row0 = 32 * (perm[c] / ncol);           // rows
            const size_t blk0 = (perm[c] % ncol) * 8;            // k blocks (32 values each)
            for (size_t r = 0; r < 32; ++r) {
                const uint64_t row = row0 + r;
                if (row >= out_dim) break;                       // padded rows: zeroed scales/codes
                const uint8_t* srow = src + row * nb * blk;
                const uint8_t rb = static_cast<uint8_t>(r / 16), rl = static_cast<uint8_t>(r % 16);
                for (size_t kb = 0; kb < 8; ++kb) {
                    if (blk0 + kb >= nb) break;                  // padded columns stay zero
                    const uint8_t* b = srow + (blk0 + kb) * blk;
                    // scales: fp16 d (and m) -> f32, plane index j = kb*32 + r
                    uint16_t ud, um = 0;
                    std::memcpy(&ud, b, 2);
                    if (has_min) std::memcpy(&um, b + 2, 2);
                    const float df = fp16_to_f32(ud), mf = has_min ? fp16_to_f32(um) : 0.f;
                    std::memcpy(d + 4 * (kb * 32 + r), &df, 4);
                    std::memcpy(d + 1024 + 4 * (kb * 32 + r), &mf, 4);
                    // codes: block nibbles (two K per byte) -> two rows per byte at one K
                    for (size_t i = 0; i < 16; ++i) {
                        const uint8_t byte = b[4 + i];
                        const uint8_t lo = byte & 0xF, hi = byte >> 4;   // GGUF: value i | value i+16
                        const size_t p0 = rb * 4096 + (kb * 32 + i) * 16 + rl;
                        d[2048 + (p0 >> 1)] |= (p0 & 1) ? lo << 4 : lo;   // value j = lo
                        const size_t p1 = p0 + 256;              // i + 16: the block's high half
                        d[2048 + (p1 >> 1)] |= (p1 & 1) ? hi << 4 : hi;   // value j + 16 = hi
                    }
                }
            }
        }
    } else if (op.op == "expert_stripes") {
        // up / gate as interleaved [up_k | gate_k] stripes per expert, each stripe's chunks
        // transposed (pool chunk c <- file chunk ncol*(c%4) + c/4).
        const std::string un = with_layer(op.up, layer), gn = with_layer(op.gate, layer);
        const uint64_t S = op.stripe_bytes, ns = op.stripes, E = op.experts;
        if (!S || !ns || !E || !op.in_dim) fail("expert_stripes without stripe_bytes / stripes / experts / in_dim");
        bounds(op, E * 2 * ns * S, dst_bytes);
        const uint8_t* up = raw(m, un, E * ns * S);
        const uint8_t* gt = raw(m, gn, E * ns * S);
        const size_t ncol = op.in_dim / 256, nchs = S / ch;
        std::vector<size_t> tp(nchs);
        for (size_t c = 0; c < nchs; ++c) tp[c] = ncol * (c % 4) + c / 4;
        for (uint64_t e = 0; e < E; ++e) {
            for (uint64_t k = 0; k < ns; ++k) {
                const uint8_t* us = up + (ns * e + k) * S;
                const uint8_t* gs = gt + (ns * e + k) * S;
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
        const uint8_t* dn = raw(m, name, E * B);
        const size_t nchs = B / ch;
        for (uint64_t e = 0; e < E; ++e) {
            const uint8_t* ds = dn + e * B;
            uint8_t* dd = dst + op.dst + e * B;
            for (size_t c = 0; c < nchs; ++c) {
                size_t rt = 4 * (c / 8) + (c % 4), cg = (c / 4) % 2;
                std::memcpy(dd + c * ch, ds + (2 * rt + cg) * ch, ch);
            }
        }
    } else if (op.op == "put") {
        const std::string name = with_layer(op.tensor, layer);
        if (dynamic_cast<const GgufFile*>(&m)) {
            // a GGUF's small weights may be f32/f16: convert to the blob's bf16 layout
            bounds(op, op.cap, dst_bytes);
            pack_norm(m, name, op.cap, dst + op.dst);
        } else {
            size_t n = 0;
            const uint8_t* src = raw(m, name, 0, &n);
            if (n > op.cap) fail(name + " is " + std::to_string(n) + " B, its slot holds " + std::to_string(op.cap));
            bounds(op, n, dst_bytes);
            std::memcpy(dst + op.dst, src, n);
        }
    } else if (op.op == "lmhead_q8") {
        // 128-row supertile order: pool chunk k <- file chunk (4*(k/32) + (k%4))*8 + ((k%32)/4)
        const std::string name = with_layer(op.tensor, layer);
        const size_t CH8 = op.chunk_bytes;
        if (!CH8) fail("lmhead_q8 without chunk_bytes");
        size_t n = 0;
        const uint8_t* src = raw(m, name, 0, &n);
        size_t nch = n / CH8;
        bounds(op, nch * CH8, dst_bytes);
        for (size_t k = 0; k < nch; ++k) {
            size_t s = k / 32, r = k % 32;
            size_t fch = (4 * s + r % 4) * 8 + r / 4;
            std::memcpy(dst + op.dst + k * CH8, src + fch * CH8, CH8);
        }
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

void build_ptab(const Manifest& m, const RowGlobal& g, size_t rows, uint8_t* t) {
    // RoPE over the first rotary_dim dims of a head, half-split pairs (i, i + rot/2), the recipe's theta:
    // [i32 pos | i32 nf | cos f32[rot/2] @512 | sin f32[rot/2] right after] (attn.h reads [cos | sin] at +512)
    const size_t half = m.rotary_dim / 2;
    if (512 + 8 * half > m.ptab_row) fail("the rotary dim does not fit the position record");
    std::memset(t, 0, rows * m.ptab_row);
    for (size_t p = 0; p < rows; ++p) {
        uint8_t* r = t + p * m.ptab_row;
        uint64_t start, nf64;
        stream_patch::attn_window(p, g.window, &start, &nf64);
        int32_t valid = static_cast<int32_t>(p - start), nf = static_cast<int32_t>(nf64);
        std::memcpy(r, &valid, 4);
        std::memcpy(r + 4, &nf, 4);
        for (size_t i = 0; i < half; ++i) {
            double ang = static_cast<double>(p) * g.inv_freq[i];
            float c = static_cast<float>(std::cos(ang)), s = static_cast<float>(std::sin(ang));
            std::memcpy(r + 512 + 4 * i, &c, 4);
            std::memcpy(r + 512 + 4 * half + 4 * i, &s, 4);
        }
    }
}

}  // namespace pools
}  // namespace open_qwen36
