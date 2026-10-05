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
#include <vector>

namespace open_qwen36 {
namespace host {

/// out[t] = x[t] / sqrt(mean(x[t]^2) + eps) * w, rows of d.
void rmsnorm_rows(const float* x, size_t T, size_t d, const float* w, double eps, float* out);
/// y [N, T] row-major (the GEMM's own output order) -> out [T, N].
void transpose(const float* y, size_t N, size_t T, float* out);

/// One column range of a GEMM's output and where it goes: columns [off, off + width) of
/// y [N, T] land in dst as [T, width].
struct TransposePart {
    float* dst = nullptr;
    size_t off = 0, width = 0;
};
/// y [N, T] transposed straight into the ranges the caller is going to read it as, so a
/// fused projection needs no [T, N] copy in between. The ranges may not overlap.
void transpose_parts(const float* y, size_t T, const TransposePart* parts, size_t n_parts);
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

/// What one verify block's DeltaNet produced per token, enough to put the layer's state back
/// to any prefix of it (OPEN-SPEC-VERIFY). The block-start state is in here too: a rollback
/// replays forwards from it, so it is the tape's business to keep it rather than the caller's.
struct DeltaTape {
    std::vector<uint16_t> conv_state0;  ///< [taps-1, nch] bf16: the conv state the block started from
    std::vector<float> S0;              ///< [value_heads, s_rows, head_dim]: the S it started from
    std::vector<float> key;             ///< [t_real, key_heads * head_dim], normed (phase 1's k)
    std::vector<float> val;             ///< [t_real, value_heads * head_dim]
    std::vector<float> decay, beta;     ///< [t_real, value_heads] each
    size_t t_real = 0;                  ///< tokens the block ran: the largest k a rollback takes
};

/// The linear-attention layer's middle over a block. qkv [T, 2*key_w + vw] and z [T, vw]
/// are the fused projection's outputs (pre-activation), xn [T, hid] the normed layer
/// input; convw [taps, nch], Wa / Wb [hid, lanes], A / dtb [value_heads], nw [head_dim].
/// conv_state [taps-1, nch] (bf16) and S [value_heads, s_rows, head_dim] (f32) are the
/// layer's state, updated in place through the first t_real tokens. og [T, vw] out
/// (zero past t_real). phase_ms, if given, gets the two halves' wall time: the
/// per-token one then the delta rule. tape, if given, records what deltanet_rollback needs.
void deltanet_block(const DeltaGeom& g, const float* qkv, const float* z, const float* xn, const float* convw,
                    const float* Wa, const float* Wb, const float* A, const float* dtb, const float* nw,
                    uint16_t* conv_state, float* S, float* og, double* phase_ms = nullptr,
                    DeltaTape* tape = nullptr);

/// The layer's DeltaNet state after the first `k` tokens of the block `tape` came from: what
/// a decode that had stopped at the accepted prefix would have left (OPEN-SPEC-VERIFY). The
/// recurrence cannot be run backwards, so this restores the state the block STARTED from -- in
/// the tape, which is why it holds it -- and replays k of the tape's tokens, k rank-1 updates
/// per head and no projection. `qkv` is the same pointer the block was given (the conv window
/// is k of its rows). conv_state and S are overwritten.
///
/// The other two pieces of state need no function. A full-attention layer's KV rows are written
/// per token from that token's own k / v, so rows [pos0, pos0 + k) are already what k sequential
/// tokens would have written and rolling back is moving the position; and the rows past them are
/// read by nothing until a later block overwrites them.
void deltanet_rollback(const DeltaGeom& g, const DeltaTape& tape, size_t k, const float* qkv,
                       uint16_t* conv_state, float* S);

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
