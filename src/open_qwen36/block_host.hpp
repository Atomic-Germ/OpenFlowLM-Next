/// \file block_host.hpp
/// \brief The block prefill's host stages: what runs on the CPU between the
///        GEMM dispatches of a 256-token block (OPEN-PREFILL-BATCH).
///
/// The 35B's DeltaNet recurrence, its attention over the KV rows and its
/// router are all on the host in the block route; the kernels do the
/// projections (whole-array GEMMs) and, one token at a time, the MoE block.
/// Every function here is the numpy reference open_kernels/model/replica_block.py
/// written out in C++ over T tokens with the state carried through the first
/// t_real of them: the kernels keep the conv state and the KV rows in bf16, so
/// those are rounded exactly as the device would; everything else accumulates
/// in double. block_host_test.cpp holds these to that reference's fixture.
#pragma once

#include <cstddef>
#include <cstdint>

namespace open_qwen36 {
namespace host {

/// out[t] = x[t] / sqrt(mean(x[t]^2) + eps) * w, rows of d.
void rmsnorm_rows(const float* x, size_t T, size_t d, const float* w, double eps, float* out);
/// y [N, T] row-major (the GEMM's own output order) -> out [T, N].
void transpose(const float* y, size_t N, size_t T, float* out);
/// x [T, K] fp32 -> the GEMM's tiled bf16 activation layout ([K, T] "k,n" order, 64 x 32
/// tiles of 8 x 8 MAC sub-tiles, gemm_q4_prefill.py); out holds K * T bf16 bits.
void tile_x(const float* x, size_t T, size_t K, uint16_t* out);

struct DeltaGeom {
    size_t T = 0, t_real = 0, hid = 0;
    size_t key_heads = 0, value_heads = 0, head_dim = 0, taps = 0;
    size_t lanes = 0;       ///< columns of the packed alpha / beta projection (the value heads, padded)
    size_t s_rows = 0;      ///< rows of one head's S in the state buffer (head_dim real, the rest zero)
    double eps = 1e-6;      ///< the gated norm's eps
};

/// The linear-attention layer's middle over a block. qkv [T, 2*key_w + vw] and z [T, vw]
/// are the fused projection's outputs (pre-activation), xn [T, hid] the normed layer
/// input; convw [taps, nch], Wa / Wb [hid, lanes], A / dtb [value_heads], nw [head_dim].
/// conv_state [taps-1, nch] (bf16) and S [value_heads, s_rows, head_dim] (f32) are the
/// layer's state, updated in place through the first t_real tokens. og [T, vw] out
/// (zero past t_real). phase_ms, if given, gets the two halves' wall time: the
/// per-token one then the delta rule.
void deltanet_block(const DeltaGeom& g, const float* qkv, const float* z, const float* xn, const float* convw,
                    const float* Wa, const float* Wb, const float* A, const float* dtb, const float* nw,
                    uint16_t* conv_state, float* S, float* og, double* phase_ms = nullptr);

struct AttnGeom {
    size_t T = 0, t_real = 0, nh = 0, kvh = 0, hd = 0, rot = 0;
    size_t pos0 = 0;        ///< the block's first position (and KV row)
    double eps = 1e-6;      ///< the q / k norms' eps
};

/// The full-attention layer's middle over a block. q / gate [T, nh*hd], k / v [T, kvh*hd]
/// are the fused projection's outputs; qn / kn [hd]; inv_freq [rot/2]. kv is the layer's
/// cache: rows of kv_row_elems bf16, [K_t | V_t] each kvh*hd wide, rows [0, pos0) valid on
/// entry and [pos0, pos0 + t_real) written. og [T, nh*hd] out: the gated attention
/// output (zero past t_real).
void attention_block(const AttnGeom& g, const float* q, const float* k, const float* v, const float* gate,
                     const float* qn, const float* kn, const double* inv_freq, uint16_t* kv, size_t kv_row_elems,
                     float* og);

/// The host half of the block attention when its products run on the NPU (OPEN-PREFILL-ATTN):
/// what attention_block does before its products and nothing after. Q [T, nh*hd] fp32 out,
/// normed and roped with 1/sqrt(hd) folded in (a power of two at every head dim here, so the
/// bf16 the kernel sees rounds exactly as the unscaled value would); the block's k / v normed,
/// roped and written to the cache rows [pos0, pos0 + t_real) in bf16. Rows of Q past t_real are zero.
void attention_prep(const AttnGeom& g, const float* q, const float* k, const float* v, const float* qn,
                    const float* kn, const double* inv_freq, uint16_t* kv, size_t kv_row_elems, float* Q);

/// rows [n, k] bf16, `stride` elements apart (the cache's K or V half), as the GEMM's tiled B for
/// B = rows^T [k, n] -- tile_x's layout from a bf16 source. Rows at or past n_real read as zero;
/// n a multiple of 32, k of 64.
void tile_rows_as_bt(const uint16_t* rows, size_t stride, size_t n_real, size_t n, size_t k, uint16_t* out);
/// rows [k, n] bf16, `stride` apart, as the tiled B for B = rows itself. Rows past k_real read as zero.
void tile_rows_as_b(const uint16_t* rows, size_t stride, size_t k_real, size_t k, size_t n, uint16_t* out);

/// One window chunk of the causal row softmax, with the running (max, sum) carried across chunks:
/// s [M, L] fp32 scores whose column 0 is window row c0; row r's query sits at pos[r], so columns
/// c0 + j > pos[r] are masked. Writes p [M, L] bf16 = exp(s - m_new) (zero where masked), scales
/// acc [M, hd] and l [M] by exp(m_old - m_new), and folds the bf16-rounded p into l -- what the
/// kernel will multiply is what the denominator counts. Start with m = -inf, l = 0, acc = 0.
void softmax_chunk(size_t M, size_t L, size_t hd, size_t c0, const float* s, const size_t* pos, float* m, float* l,
                   float* acc, uint16_t* p);

/// The MoE router over a block: softmax of xm [T, hid] @ Wr [hid, E] into probs [T, E], the
/// top-k by probability (lowest index on a tie, as the kernel's router_fin picks) into
/// idx [T, topk], renormalised into w [T, topk].
void router_block(size_t T, size_t hid, size_t E, size_t topk, const float* xm, const float* Wr, float* probs,
                  int32_t* idx, float* w);

}  // namespace host
}  // namespace open_qwen36
