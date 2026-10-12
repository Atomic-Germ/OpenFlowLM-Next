/// \file dflash2_host.hpp
/// \brief The DFlash 2 drafter's host stages (OPEN-DRAFT-DFLASH2): everything between its GEMV
///        dispatches, XRT-free so dflash2_test.cpp can hold it to utilities/dflash2_ref.py.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace open_qwen36 {
namespace dflash2 {

constexpr size_t kTopK = 16;
constexpr size_t kRows = 8;            ///< block rows per pass: [anchor, MASK x 7], truncated to k drafts

/// The selector lattice of one draft (common/speculative.cpp's DFlash 2 path): per draft row i,
/// 16 candidate ids (the head's top 16, descending) and score[i][p][s] = next[cand_s] . (prev[pred_p]
/// * gate_i) + logit[cand_s]; row 0's only predecessor is the anchor, so all its p columns are equal.
struct Lattice {
    size_t rows = 0;
    std::vector<int32_t> cand;         ///< [rows][16]
    std::vector<float> logit;          ///< [rows][16] the candidates' head logits
    std::vector<float> score;          ///< [rows][16 pred][16 slot]
};

/// The greedy walk: row 0 reads predecessor column 0, row i the column of the slot row i-1 chose;
/// the highest score wins, the lowest slot on a tie. One token per row.
std::vector<int> walk(const Lattice& lat);
/// The winning score's margin over the runner-up at each row along the walk (near-tie diagnostics).
std::vector<float> walk_margins(const Lattice& lat);

/// Top-16 of logits [n] by value, descending, the lower id first on a tie.
void top16(const float* logits, size_t n, int32_t* ids, float* vals);

/// The lattice for rows 1..k of a block: logits [k][vocab] and hn [k][hidden] (the final-normed rows),
/// sel_hidden [rank][hidden], prev / next [vocab][rank] as bf16 bits.
void build_lattice(int anchor, size_t k, const float* logits, size_t vocab, const float* hn, size_t hidden,
                   const float* sel_hidden, size_t rank, const uint16_t* prev, const uint16_t* next, Lattice& lat);

/// DFlash 2's dynamic 2-tap grouped conv over the block's rows: y[t] = w0[t] x[t] + w1[t] x[t-1]
/// (row 0 has no predecessor), w_tap[t][c] = dyn[t][side][tap][c / group] + base[side][tap][c].
/// x, y [rows][n] (may alias), dyn [rows][2][2][n / group], base [2][2][n].
void conv2(const float* x, size_t rows, size_t n, const float* dyn, const float* base, int side, size_t group,
           float* y);

/// NeoX RoPE over all head_dim dims: pairs (i, i + hd/2) at angle pos * theta^(-2i/hd).
void rope_neox(float* x, size_t heads, size_t hd, double pos, double theta);

/// x rounded to the nearest IEEE half (the drafter's KV cache holds f16 values, as llama.cpp's).
float round_f16(float x);

/// Attention of the block's query rows over [context | block]: q [rows][nh][hd] (normed, roped),
/// kc / vc the context rows [L][kvh][hd] at positions ctx_pos0 .. ctx_pos0 + L - 1, kb / vb the
/// block's own rows [rows][kvh][hd]. Row t sits at position p0 + t and sees context positions p
/// with p0 + t - p < window (0 = all) and every block row. out [rows][nh][hd].
void attention(const float* q, size_t rows, size_t nh, size_t kvh, size_t hd, const float* kc, const float* vc,
               size_t L, long ctx_pos0, const float* kb, const float* vb, long p0, size_t window, float* out);

// ---- gemv_t2n's host half (open_kernels/designs/gemv_t2n/t2n_host.py is the same in NumPy)

constexpr size_t kRec = 1216;          ///< one (k-tile, 128-group, quad) record
constexpr size_t kHdr = 64;
constexpr size_t kYBand = 512;

struct GemvItem {
    int core = 0, slice = 0, b0 = 0, nb = 0, kt0 = 0, nkt = 0, tab = 0, y = 0;
};
struct GemvPlan {
    size_t n = 0, k = 0, tab_bytes = 0, n_tabs = 0, y_floats = 0;
    std::vector<GemvItem> items;
};

/// xr [8][K] (already through H/32 per 1024 block) -> records [K/256][2 group][2 quad][kRec].
void build_records(const float* xr, size_t K, uint8_t* rec);
/// One table element per item: header (nb, nkt), then the item's k-tiles' records. tab [n_tabs][tab_bytes].
void build_tables(const GemvPlan& p, const uint8_t* rec, uint8_t* tab);
/// The kernel's partial y -> out [8][N] (every item's bands added into place).
void reduce_y(const GemvPlan& p, const float* y, float* out);

}  // namespace dflash2
}  // namespace open_qwen36
