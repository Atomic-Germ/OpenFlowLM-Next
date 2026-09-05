/// \file pools.cpp
/// \brief Pool / consts / ptab packers (see pools.hpp). Port of
///        open_kernels/model/pools.py and make_decode.py's layer_consts.
#include "open_qwen36/pools.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace open_qwen36 {
namespace pools {

using namespace layout;

namespace {

std::string lname(int layer, const char* suffix) {
    return "model.layer." + std::to_string(layer) + "." + suffix;
}

/// pool chunk index -> file chunk index for a standard [out, in] matmul tensor:
/// pool chunk c covers rows 64*(c/per_band) + 32*(c%2) and cols
/// 1024*((c/8) % (in/1024)) + 256*((c/2)%4); file chunk f covers rows
/// 32*(f/ncol), cols 256*(f%ncol).
std::vector<size_t> std_perm(size_t nch, size_t in_dim) {
    size_t ncol = in_dim / 256, per_band = in_dim / 128, kgroups = in_dim >= 1024 ? in_dim / 1024 : 1;
    std::vector<size_t> perm(nch);
    for (size_t c = 0; c < nch; ++c) {
        size_t rows0 = 64 * (c / per_band) + 32 * (c % 2);
        size_t cols0 = (1024 * ((c / 8) % kgroups) + 256 * ((c / 2) % 4)) % in_dim;
        perm[c] = (rows0 / 32) * ncol + cols0 / 256;
    }
    return perm;
}

/// Copy `nch` chunks of `src` into `dst` in `perm` order.
void permute_chunks(const uint8_t* src, size_t src_bytes, const std::vector<size_t>& perm, uint8_t* dst) {
    if (src_bytes < perm.size() * kChunk)
        throw std::runtime_error("pools: tensor smaller than its pool region (" + std::to_string(src_bytes) + " B)");
    for (size_t c = 0; c < perm.size(); ++c) std::memcpy(dst + c * kChunk, src + perm[c] * kChunk, kChunk);
}

void permute_tensor(const Q4nxFile& m, const std::string& name, size_t nch, size_t in_dim, uint8_t* dst) {
    size_t n = 0;
    const uint8_t* src = m.raw(name, &n);
    permute_chunks(src, n, std_perm(nch, in_dim), dst);
}

void put(const Q4nxFile& m, const std::string& name, uint8_t* dst, size_t cap) {
    size_t n = 0;
    const uint8_t* src = m.raw(name, &n);
    if (n > cap) throw std::runtime_error("pools: " + name + " is " + std::to_string(n) + " B, slot holds " + std::to_string(cap));
    std::memcpy(dst, src, n);
}

}  // namespace

void build_layer_pool(const Q4nxFile& m, int layer, bool full_attn, uint8_t* pool) {
    std::memset(pool, 0, kPoolBytes);
    // Routed experts: up/gate as interleaved stripes, each stripe's 32 chunks
    // transposed (pool chunk c <- file chunk 8*(c%4) + c/4); down slices with
    // pool chunk c <- file chunk 2*rt + cg, rt = 4*(c/8) + c%4, cg = (c/4)%2.
    size_t nup = 0, ngt = 0, ndn = 0;
    const uint8_t* up = m.raw(lname(layer, "mlp.up_exps_proj.weight"), &nup);
    const uint8_t* gt = m.raw(lname(layer, "mlp.gate_exps_proj.weight"), &ngt);
    const uint8_t* dn = m.raw(lname(layer, "mlp.down_exps_proj.weight"), &ndn);
    if (nup < 256 * 4 * kStripe || ngt < 256 * 4 * kStripe || ndn < 256 * 655360)
        throw std::runtime_error("pools: layer " + std::to_string(layer) + " expert tensors are short");
    for (size_t e = 0; e < 256; ++e) {
        for (size_t k = 0; k < 4; ++k) {
            const uint8_t* us = up + (4 * e + k) * kStripe;
            const uint8_t* gs = gt + (4 * e + k) * kStripe;
            uint8_t* ud = pool + (8 * e + 2 * k) * kStripe;
            uint8_t* gd = pool + (8 * e + 2 * k + 1) * kStripe;
            for (size_t c = 0; c < 32; ++c) {
                size_t f = 8 * (c % 4) + c / 4;
                std::memcpy(ud + c * kChunk, us + f * kChunk, kChunk);
                std::memcpy(gd + c * kChunk, gs + f * kChunk, kChunk);
            }
        }
        const uint8_t* ds = dn + e * 655360;
        uint8_t* dd = pool + 335544320 + e * 655360;
        for (size_t c = 0; c < 128; ++c) {
            size_t rt = 4 * (c / 8) + (c % 4), cg = (c / 4) % 2;
            std::memcpy(dd + c * kChunk, ds + (2 * rt + cg) * kChunk, kChunk);
        }
    }
    permute_tensor(m, lname(layer, "mlp.share_up_exps_proj.weight"), 128, 2048, pool + 503316480);
    permute_tensor(m, lname(layer, "mlp.share_gate_exps_proj.weight"), 128, 2048, pool + 503971840);
    permute_tensor(m, lname(layer, "mlp.share_down_exps_proj.weight"), 128, 512, pool + 504627200);
    if (full_attn) {
        // q_proj is the fused [q 4096 | gate 4096] rows; the pool splits the halves.
        size_t nq = 0;
        const uint8_t* qg = m.raw(lname(layer, "self_attn.q_proj.weight"), &nq);
        if (nq < 2048 * kChunk) throw std::runtime_error("pools: q_proj short");
        auto p1024 = std_perm(1024, 2048);
        permute_chunks(qg, 1024 * kChunk, p1024, pool + 505282560);
        permute_chunks(qg + 1024 * kChunk, 1024 * kChunk, p1024, pool + 511836160);
        permute_tensor(m, lname(layer, "self_attn.k_proj.weight"), 128, 2048, pool + 510525440);
        permute_tensor(m, lname(layer, "self_attn.v_proj.weight"), 128, 2048, pool + 511180800);
        permute_tensor(m, lname(layer, "self_attn.o_proj.weight"), 1024, 4096, pool + 517079040);
    } else {
        permute_tensor(m, lname(layer, "linear_attn.qkv_proj.weight"), 2048, 2048, pool + 505282560);
        permute_tensor(m, lname(layer, "self_attn.gate_proj.weight"), 1024, 2048, pool + 515768320);
    }
}

void build_lmhead_pool(const Q4nxFile& m, uint8_t* out) {
    // 128-row supertile order: pool chunk k <- file chunk (4*(k/32) + (k%4))*8 + ((k%32)/4)
    constexpr size_t CH8 = 8704;
    size_t n = 0;
    const uint8_t* raw = m.raw("lm_head.weight", &n);
    size_t nch = n / CH8;
    if (nch * CH8 > kLmheadPoolBytes) throw std::runtime_error("pools: lm_head larger than its pool");
    std::memset(out, 0, kLmheadPoolBytes);
    for (size_t k = 0; k < nch; ++k) {
        size_t s = k / 32, r = k % 32;
        size_t f = (4 * s + r % 4) * 8 + r / 4;
        std::memcpy(out + k * CH8, raw + f * CH8, CH8);
    }
}

void build_consts(const Q4nxFile& m, int layer, bool full_attn, uint8_t* c) {
    if (full_attn) {
        std::memset(c, 0, CA_BYTES);
        put(m, lname(layer, "input_layernorm.weight"), c + CA_LNW, 4096);
        put(m, lname(layer, "post_attention_layernorm.weight"), c + CA_POSTLN, 4096);
        put(m, lname(layer, "self_attn.q_norm.weight"), c + CA_META, 512);        // effective q norm, bf16[256]
        put(m, lname(layer, "self_attn.k_norm.weight"), c + CA_META + 512, 512);
        put(m, lname(layer, "moe_router.weight"), c + CA_RW, 1048576);
        put(m, lname(layer, "shared_expert_gate.weight"), c + CA_SGW, 4096);
        return;
    }
    std::memset(c, 0, C_BYTES);
    put(m, lname(layer, "input_layernorm.weight"), c + C_LNW, 4096);
    // The glue's side blob: alpha / beta projections, the [a | dt_bias] record,
    // conv1d transposed to [8 groups][4 taps][1024]. (Its first 4 KB is the xn
    // slot the kernel fills from `act`, which is why the region starts at 4096.)
    uint8_t* side = c + C_SIDE;
    put(m, lname(layer, "linear_attn.ssm_alpha_proj.weight"), side, 131072);
    put(m, lname(layer, "linear_attn.ssm_beta_proj.weight"), side + 131072, 131072);
    put(m, lname(layer, "linear_attn.ssm_a"), side + 262144, 128);          // f32[32]
    put(m, lname(layer, "linear_attn.ssm_dt.bias"), side + 262144 + 128, 128);
    {
        size_t n = 0;
        const uint8_t* convw = m.raw(lname(layer, "linear_attn.ssm_conv1d.weight"), &n);  // bf16 [4][8192]
        if (n != 65536) throw std::runtime_error("pools: conv1d weight is not bf16[4,8192]");
        uint8_t* t = side + 266240;
        for (size_t g = 0; g < 8; ++g)
            for (size_t tap = 0; tap < 4; ++tap)
                std::memcpy(t + (g * 4 + tap) * 2048, convw + (tap * 8192 + g * 1024) * 2, 2048);
    }
    put(m, lname(layer, "linear_attn.ssm_norm.weight"), c + C_NW, 256);      // bf16[128] in a 4 KB element
    put(m, lname(layer, "post_attention_layernorm.weight"), c + C_POSTLN, 4096);
    put(m, lname(layer, "moe_router.weight"), c + C_RW, 1048576);
    put(m, lname(layer, "shared_expert_gate.weight"), c + C_SGW, 4096);
    permute_tensor(m, lname(layer, "linear_attn.ssm_out_proj.weight"), 1024, 4096, c + C_WOUT);
}

void build_ptab(size_t rows, uint8_t* t) {
    // partial RoPE: rotary dim 64 of 256, theta 1e7 — the same freqs as replica.py
    std::memset(t, 0, rows * PTAB_ROW);
    for (size_t p = 0; p < rows; ++p) {
        uint8_t* r = t + p * PTAB_ROW;
        int32_t pos = static_cast<int32_t>(p), nf = static_cast<int32_t>(p ? p : 1);
        std::memcpy(r, &pos, 4);
        std::memcpy(r + 4, &nf, 4);
        for (size_t i = 0; i < 32; ++i) {
            double ang = static_cast<double>(p) * std::pow(1e7, -static_cast<double>(i) / 32.0);
            float c = static_cast<float>(std::cos(ang)), s = static_cast<float>(std::sin(ang));
            std::memcpy(r + 512 + 4 * i, &c, 4);
            std::memcpy(r + 640 + 4 * i, &s, 4);
        }
    }
}

}  // namespace pools
}  // namespace open_qwen36
