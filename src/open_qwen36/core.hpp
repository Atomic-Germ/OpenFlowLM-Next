/// \file core.hpp
/// \brief The resident open-kernel decode engine: device, kernels, weights and
///        per-layer state held for the process lifetime; one `step()` per token.
///
/// The core is an interpreter of the kernel set's manifest.json
/// (manifest.hpp, written by open_kernels/export_qwen36_kernels.py from the
/// family recipe): the contexts and kernels to load, the per-layer buffers to
/// allocate and pack, and per layer TYPE the verb sequence to run --
/// `run <kernel> <buffers...>` and `moeroute2 <kernel>` (read the router's
/// top-k out of `act`, re-point the expert fills) -- then the tail (final
/// norm, lm_head). Kernels marked `attnpos` have their KV window length and
/// row / RoPE-record offsets patched once per token. No model constant lives
/// in this file; a new model in the family is a new manifest.
///
/// This is the host half of the open path that phlegm ran as a batch `.cfg`
/// program and planned as `OpenBackend`. It has no dependency on the OFLM app
/// headers so it can be built and tested on its own (cli.cpp); engine.hpp
/// adapts it to the app's `causal_lm` seam.
///
/// Prefill is decode-as-prefill -- the prompt through `step()` one token at a
/// time, exact for this architecture -- unless the kernel set carries the
/// block route (manifest.hpp's GemmBlockProgram): then `step_gemm_block()`
/// takes 256 tokens at a time through the projections as GEMMs, with the
/// stages between them on the host (block_host.hpp) and, on the MoE
/// families, the expert block still one token at a time.
#pragma once

#include <chrono>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include "open_qwen36/block_host.hpp"
#include "open_qwen36/manifest.hpp"
#include "open_qwen36/pools.hpp"
#include "open_qwen36/q4nx_file.hpp"
#include "stream_patch.hpp"

namespace open_qwen36 {

struct CoreConfig {
    std::string model_dir;   ///< holds config.json + model.q4nx (+ tokenizer files)
    std::string kernel_dir;  ///< holds manifest.json and the xclbin / insts.bin files it names
    int num_layers = -1;     ///< -1 = all of them; a prefix otherwise (testing)
    size_t max_ctx = 4096;   ///< KV rows per attention layer and RoPE records: the context capacity
    unsigned timeout_ms = 60000;  ///< per dispatch; 0 blocks
    bool verbose = true;
};

/// Everything a request needs to be resumed later (the app's checkpoint/restore).
struct Snapshot {
    int pos = 0;
    int64_t mrope_pos = 0;                     ///< the (t, h, w) counter (M-RoPE, once a request has an image)
    bool mrope_on = false;
    std::vector<std::vector<uint8_t>> states;  ///< per linear layer: the state BO
    std::vector<std::vector<uint8_t>> kv;      ///< per attention layer: rows [0, pos)
};

/// One kernel's dispatches over a block: what it cost where it actually runs.
struct DispatchStat {
    int calls = 0;
    double ms = 0, min_ms = 1e18;
    void add(double t) {
        ++calls;
        ms += t;
        min_ms = t < min_ms ? t : min_ms;
    }
};

struct StepTiming {
    double part0_ms = 0, part1_ms = 0, route_ms = 0, lmhead_ms = 0, total_ms = 0;
    // The block route's stages, split finely enough to say which one to work on.
    // part1_ms is mid + tail; route_ms is the four moe_* below.
    double mid_ms = 0;        ///< the DeltaNet recurrence, or the attention itself
    double dn_conv_ms = 0;    ///< of mid: DeltaNet's per-token half (conv, q/k norms, alpha/beta)
    double dn_rule_ms = 0;    ///< of mid: DeltaNet's delta rule on S, per head over the block
    double attn_ms = 0;       ///< of mid: the attention layers' host half
    double gemm_tile_ms = 0;  ///< x into the GEMM's tiled bf16 activation layout
    double gemm_tr_ms = 0;    ///< the GEMM's [N, T] output back to [T, N], allocation included
    double tail_ms = 0;       ///< residual, post-norm, router
    double state_ms = 0;      ///< the state BO syncs (the KV read grows with position)
    double moe_prep_ms = 0;   ///< xm / the router record / the residual into act
    double moe_patch_ms = 0;  ///< moe2_apply and the instruction sync
    double moe_run_ms = 0;    ///< the mx dispatch itself
    double moe_read_ms = 0;   ///< xres back
    double shared_ms = 0;     ///< the shared expert over the block (its GEMMs are in part0)
};

class Core {
public:
    /// Reads the manifest, checks it against the model's config.json, opens the
    /// device (or borrows `dev`), registers the xclbins and loads the
    /// instruction streams. Weights come with load_weights().
    Core(const CoreConfig& cfg, xrt::device* dev = nullptr);
    ~Core();
    Core(const Core&) = delete;
    Core& operator=(const Core&) = delete;

    /// Pack every layer's pools and consts straight into resident device
    /// buffers. Minutes on first touch of a 22 GB file, ~1 min warm.
    void load_weights(const std::function<void(int done, int total)>& progress = {});

    /// Start a new context: zero the linear states, position 0.
    void reset();
    /// One decode step for `token` at the current position. Logits (f32,
    /// vocab) are computed only when asked for; read them with logits().
    void step(int token, bool want_logits);
    /// One step whose input is a hidden vector instead of a token -- an image token's
    /// embedding from the vision tower -- at the M-RoPE position `mpos` = (t, h, w). The
    /// (t, h, w) counter is not advanced; the caller does that per image (mrope_advance).
    /// One pre-computed row (an image patch) instead of a token id.
    ///
    /// `deepstack` is Qwen3-VL's: `n_deepstack` rows of `hidden` floats, feature j being
    /// added onto the residual AFTER decoder layer j has run - which is where transformers
    /// puts it, and is not the same as folding feature 0 into the input embedding. nullptr
    /// for every other VLM.
    void step_embed(const float* x, bool want_logits, const int64_t mpos[3],
                    const float* deepstack = nullptr, int n_deepstack = 0);
    const std::vector<float>& logits() const { return logits_host_; }

    /// M-RoPE (Qwen3-VL, config.json rope_parameters.mrope_section): once a request has
    /// an image, every later token's rotary record is written from a (t, h, w) counter
    /// rather than its KV row -- a text token takes (c, c, c) and advances c by one, an
    /// image's tokens take (c, c + row, c + col) and the image advances c by
    /// max(rows, cols). Until mrope_begin() the prebuilt records (row p at position p)
    /// serve, which is the text-only path unchanged.
    bool has_mrope() const { return mrope_section_.size() == 3; }
    void mrope_begin();
    void mrope_advance(int64_t n) { mrope_pos_ += n; }
    int64_t mrope_pos() const { return mrope_pos_; }
    /// True once mrope_begin() has fired for this request -- the gemm-block route
    /// writes no position records, so it can't serve a prompt that has had an image.
    bool mrope_active() const { return mrope_on_; }
    /// config.json's image_token_id (-1 when the model has none).
    int image_token_id() const { return image_token_id_; }
    /// Time every kernel the layer's block route names under conditions the route puts them
    /// in, to say where a dispatch's time goes. Writes its table to stderr; needs
    /// load_weights(), and the buffers' contents do not affect the timing.
    ///
    /// What the probes found on the 35B (2026-09-12), each against the same kernel run back
    /// to back with nothing else happening:
    ///   cycling every layer's weights   free
    ///   a 30 ms idle gap                +0.1 to +0.7 ms
    ///   the expert patch + its sync     free (the sync of an 800 KB stream is 0.03 ms)
    ///   the host reading the output     free
    ///   a change of hardware context    +2.8 ms, flat, whatever the kernel's size
    ///   ~48 MB of host memory churn     +3.7 ms on the smallest, +14.7 on the widest
    /// The last two together reproduce what the same dispatches cost in a real block
    /// (OFLM_OPEN_DISPATCH_LOG prints that). Churn scales with what the kernel streams,
    /// which is what dirty cache lines being written back during the stream would do.
    void bench_dispatch(int layer, int reps);
    /// The same for decode's dispatches -- the per-token program, which the block route never
    /// runs. Three probes: each kernel repeated on one layer, the same cycling every layer of
    /// its type, and the real 40-layer walk in order. The walk's total is the floor a decode
    /// step cannot go below. Call it after a step so the attnpos and route patches hold real
    /// values; it leaves the state and the KV window meaningless, so exit afterwards.
    void bench_decode(int reps);
    /// One kernel, on one layer, over and over, with the arguments its own program gives it.
    /// For a half-program kernel (lx0, ax0) this HANGS unless the build is self-contained --
    /// the second half is what drains its fifos -- so it is for timing a truncated build
    /// (LX_STOP), not for the shipped kernels; bench_decode replays whole layers instead.
    void bench_kernel(const std::string& name, int reps, int layer, int warm_token);
    /// Per-kernel dispatch counts and times since the last call, then cleared. Empty unless
    /// dispatch accounting is on (OFLM_OPEN_DISPATCH_LOG).
    std::map<std::string, DispatchStat> take_dispatch_stats();
    /// The block route's token block (manifest.hpp's GemmBlockProgram), or 0
    /// when the loaded kernel set has none / its layer types disagree.
    size_t gemm_block_t() const { return gemm_block_t_; }

    /// `n` bytes of a layer's `act` scratch at `off`, straight off the device. Bring-up
    /// only: it is how you tell a stage that computes the wrong thing from a stage that
    /// never ran, without inferring either from the logits.
    void read_act(int layer, size_t off, size_t n, uint8_t* dst);
    /// T = gemm_block_t() tokens through every layer on the block route: the
    /// projections as whole-array GEMM dispatches, the stages between them
    /// per layer-type kind (dense: T single-token attention dispatches and
    /// host norms / SwiGLU; linear: the DeltaNet recurrence on the host; full:
    /// attention on the host; the MoE block one token at a time). The caller
    /// pads a short tail with any in-range id and passes the REAL count as
    /// `t_real`: only those tokens touch the state, and only they advance the
    /// position. Logits, like step(), only for the last real token, only when
    /// asked.
    void step_gemm_block(const std::vector<int>& ids, size_t t_real, bool want_logits);
    /// The WHOLE prompt on the block route, layer-major: every T-wide block through
    /// layer l's projections and host stages before layer l + 1, with the layer's MoE
    /// run ONCE over every token of the prompt instead of once per block. Same
    /// operations in the same order per layer, so bit-exact against the block-major
    /// loop above; what changes is how many times each expert's 1.97 MB is streamed
    /// (at 2582 tokens: ~81 of its tokens a layer instead of ~8, so the visits an
    /// expert costs fall by the same factor). The caller passes the prompt unpadded;
    /// the tail block is padded here with the last id, exactly as the caller did.
    /// Only for the MoE kinds (linear / full) -- layer_major_ok() says so.
    void step_gemm_prompt(const std::vector<int>& ids, bool want_logits);
    /// Whether step_gemm_prompt() can run this kernel set: a block route whose every
    /// layer is a MoE kind, and OFLM_OPEN_LAYER_MAJOR not set to 0.
    bool layer_major_ok() const;
    /// Validation: logits for EVERY real token of the next blocks (one lm_head pass each),
    /// read back with block_logits() -- what a position-for-position diff against the
    /// sequential path needs. Off by default; costs a tail per token.
    void set_block_logits_all(bool on) { block_logits_all_ = on; }
    const std::vector<std::vector<float>>& block_logits() const { return block_logits_; }

    int position() const { return pos_; }
    /// Test hook: place the next token at `pos` without decoding up to it.
    void seek(int pos);
    size_t max_ctx() const { return cfg_.max_ctx; }
    int num_layers() const { return nl_; }
    bool is_attention_layer(int l) const { return types_[l]->state_kind == "kv"; }
    const StepTiming& last_timing() const { return timing_; }
    const Manifest& manifest() const { return man_; }
    size_t vocab() const { return man_.vocab; }
    size_t real_vocab() const { return man_.real_vocab; }

    Snapshot checkpoint() const;
    void restore(const Snapshot& s);

    /// One cached row of an attention layer's K or V (bf16, kv_row / 4 elements).
    void kv_row(int layer, int row, bool value, uint16_t* out);

    const Q4nxFile& file() const { return *file_; }

private:
    struct Kern {
        std::string name;
        std::string patch;
        std::unique_ptr<xrt::kernel> k;
        std::unique_ptr<xrt::bo> instr;
        std::vector<uint32_t> words;
        std::vector<stream_patch::MoePatch> moe2;    ///< moeroute2 / moebatch: the routed-expert fills
        size_t slots = 0;                            ///< moebatch: expert slots the stream carries
        std::vector<stream_patch::AttnPatch> attn;
        stream_patch::AttnGeometry geom;     ///< attnpos: the manifest's rows plus this kernel's window
        uint32_t* iw() { return instr->map<uint32_t*>(); }
    };

    CoreConfig cfg_;
    Manifest man_;
    std::unique_ptr<Q4nxFile> file_;
    int nl_ = 0;
    std::vector<const LayerType*> types_;      ///< per layer

    std::unique_ptr<xrt::device> owned_dev_;
    xrt::device* dev_ = nullptr;
    std::map<std::string, std::unique_ptr<xrt::hw_context>> ctxs_;
    std::map<std::string, Kern> kerns_;

    std::vector<xrt::bo> pools_, consts_, act_, state_;   ///< per layer
    std::map<std::string, xrt::bo> globals_;              ///< the manifest's globals (xres, ptab, lmpool, gact, ...)
    bool weights_loaded_ = false;
    int pos_ = 0;
    /// Dispatch bookkeeping, for the intermittent dx timeout. A failure needs to say
    /// which layer, how far into the run, and how long the host sat between dispatches -
    /// "kernel dx at position N" alone does not separate a hung command from a late one.
    uint64_t dispatches_ = 0;
    std::chrono::steady_clock::time_point last_done_{};
    std::vector<int> mrope_section_;          ///< empty: no M-RoPE (every model but the VLMs)
    bool mrope_interleaved_ = false;
    int image_token_id_ = -1;
    bool mrope_on_ = false;
    int64_t mrope_pos_ = 0;
    size_t ptab_dirty_ = 0;                    ///< rows [0, dirty) hold per-request records; reset() restores them

    // ---- the block route (manifest.hpp's GemmBlockProgram)
    size_t gemm_block_t_ = 0;    ///< common gemm_block.t across every loaded layer type, or 0
    bool moe_batch_on_ = true;   ///< the token-batched expert kernel where the set carries it (OFLM_OPEN_MOE_BATCH=0 off)
    bool attn_block_on_ = true;  ///< the attention products on the NPU where the set carries them (OFLM_OPEN_ATTN_BLOCK=0 off)
    bool layer_major_on_ = true; ///< the whole prompt through each layer before the next (OFLM_OPEN_LAYER_MAJOR=0 off)
    bool dispatch_log_ = false;  ///< OFLM_OPEN_DISPATCH_LOG: keep per-kernel dispatch times
    std::vector<float> gout_, sg_ug_, sg_y_;   ///< the block route's GEMM outputs, kept across layers
    std::map<std::string, DispatchStat> dispatch_stats_;
    // Per weight name, per layer: a dedicated buffer holding a contiguous run of
    // the packed pool / consts bytes (the GEMM kernels read their weight from
    // byte 0 of their own buffer; an XRT sub-buffer view is untested here).
    // Built once in load_weights() from the same host bytes the pool upload uses.
    std::map<std::string, std::vector<xrt::bo>> gemm_w_;
    // dense: the two norm weights (bf16) the host RMSNorm reads, captured from consts
    std::vector<std::vector<uint16_t>> ln_w_bf16_, post_ln_w_bf16_;
    // linear / full: the small per-layer tensors the host stages read, straight from the file
    struct HostConsts {
        std::vector<float> ln, postln, router;          ///< [hid], [hid], [hid, E]
        std::vector<float> convw, Wa, Wb, A, dtb, nw;   ///< linear: [taps, nch], [hid, lanes] x2, [heads] x2, [head_dim]
        size_t lanes = 0;
        std::vector<float> qn, kn;                      ///< full: [hd] x2
        std::vector<float> sgw;                         ///< the shared expert's sigmoid gate, [hid]
    };
    std::vector<HostConsts> hc_;                       ///< per layer, filled for a linear / full route
    /// What a MoE layer's attention half leaves for its expert block: the residual, its
    /// post-attention norm and the router's top-k. Rows are a block on the block-major
    /// route and the whole padded prompt on the layer-major one; held here rather than
    /// on the stack so the 20-70 MB is allocated once per request, not once per layer.
    struct MoeStage {
        std::vector<float> res, xm, probs, w;
        std::vector<int32_t> idx;
    };
    MoeStage moe_;
    /// Size moe_ for `rows` tokens. Never shrinks: a request's blocks are all the same width.
    void moe_stage_resize(size_t rows);
    bool block_logits_all_ = false;
    std::vector<std::vector<float>> block_logits_;     ///< per real token of the last block, when asked

    std::vector<float> logits_host_;
    StepTiming timing_;

    xrt::hw_context& context(const std::string& name);
    void load_kernel(const std::string& name, const KernelDesc& d);
    xrt::bo alloc(size_t bytes, const uint8_t* init = nullptr, size_t init_bytes = 0);
    xrt::bo& buffer(const std::string& name, int layer);
    void step_impl(int token, const float* x, bool want_logits, const int64_t* mpos,
                   const float* deepstack = nullptr, int n_deepstack = 0);
    /// Write KV row `row`'s position record from (t, h, w) into every position table.
    void write_record(size_t row, const double pos[3]);
    double run(Kern& k, const std::vector<std::string>& args, int layer);
    /// run()'s two halves for the bench: building the xrt::run and setting its arguments,
    /// then start() to wait(). The sum is what run() returns.
    std::pair<double, double> run_split(Kern& k, const std::vector<std::string>& args, int layer);
    void route(Kern& k, int layer, uint64_t act_off);
    void log(const std::string& s) const;

    // ---- the block route's helpers (core.cpp)
    /// The byte {offset, length} of pack op `idx` of `lt.pool` (from "pool") or `lt.consts`
    /// ("consts"): a band-law (std_perm) projection, length = nch * chunk_bytes.
    std::pair<size_t, size_t> op_region(const LayerType& lt, const std::string& from, size_t idx) const;
    /// The consts tensor whose name ends in `suffix`, with the layer index filled in.
    std::string const_tensor(const LayerType& lt, const std::string& suffix, int layer) const;
    /// One GEMM step over x [T, K] (f32 row-major): tile, upload, run, download y as [T, N].
    /// `out` is grown if it is short and then fully overwritten; pass a buffer that lives
    /// across layers, so the 12 MB the widest GEMM returns is allocated once, not 40 times
    /// a block.
    void gemm(const Step& s, const std::vector<float>& x, size_t T, size_t K, size_t N, int layer,
              std::vector<float>& out);
    /// The same dispatch without the transpose: y stays [N, T] in the output buffer and the
    /// mapping is returned, so a caller that is going to slice the output can transpose
    /// straight into its own arrays. Valid until the next GEMM on the same buffer.
    const float* gemm_run(const Step& s, const std::vector<float>& x, size_t T, size_t K, size_t N, int layer);
    /// The tail (final norm, lm_head) for one residual row into logits_host_.
    void tail_logits(const float* row);
    /// Host-side shuttle of one token's `act_bytes` slice between a GLOBAL
    /// T-wide scratch buffer (`wide`, e.g. "gact") and an ordinary T=1
    /// per-layer scratch buffer (`scratch1`, e.g. "act").
    /// Move one token's slice between the T-wide scratch and a layer's own `act`.
    /// `region_bytes` limits it to [region_off, +region_bytes) of the slice; 0 moves all of it.
    void shuttle_buf(xrt::bo& wide, xrt::bo& scratch1, size_t token, size_t act_bytes, bool wide_to_scratch,
                     size_t region_off = 0, size_t region_bytes = 0);
    /// out[t,:] = x[t,:] / sqrt(mean(x[t,:]^2) + eps) * w[:], reduction and
    /// the final multiply both in fp64. w is bf16 (hidden elements).
    static void rmsnorm_host(const std::vector<double>& x, size_t T, size_t hid,
                             const std::vector<uint16_t>& w_bf16, double eps, std::vector<float>& out);
    /// [T,K] fp32 -> bf16, pre-tiled into [K,T] "k,n" order (K_TILE=64, MAC 8x8, tile_n 32)
    /// -- the layout gemm_q4_prefill.py streams its activation in.
    static void tile_gemm_x(const std::vector<float>& x_tk, size_t T, size_t K, std::vector<uint16_t>& out);
    /// The scalar original of tile_gemm_x, kept only as the reference host::tile_x is checked against.
    static void tile_gemm_x_reference(const std::vector<float>& x_tk, size_t T, size_t K, std::vector<uint16_t>& out);
    /// One dense layer of the route (0167/#32): entry RMSNorm -> GEMM qkv3 -> T dxB
    /// dispatches -> GEMM o -> residual + post-attn RMSNorm -> GEMM gate, up -> host
    /// SwiGLU -> GEMM down -> residual. `xres` is T*hidden fp64, updated in place.
    void step_gemm_block_layer(int l, std::vector<double>& xres, size_t T);
    /// The MoE families' block (kinds linear / full): xres as f32 [T, hidden].
    void step_block_moe(const std::vector<int>& ids, size_t t_real, bool want_logits);
    /// The shared expert over a whole block: up|gate then down as GEMMs, silu and the
    /// sigmoid gate on the host, added into res [T, hid] in place.
    void shared_expert_block(int l, const float* xm, float* res, size_t T, size_t t_real);
    /// A linear-attention layer of the block route, everything up to the MoE: GEMM qkv|z
    /// -> host DeltaNet (state in place through t_real tokens) -> GEMM out -> residual,
    /// norm, router -> the shared expert. `xres` is THIS block's T rows; the router's
    /// output lands in moe_ at row `row0`. The DeltaNet state is read off the device only
    /// on the layer's `first` block and written back only on its `last`: in between the
    /// host map is the only thing that touches it. The MoE itself is block_layer_moe().
    void block_layer_linear(int l, float* xres, size_t T, size_t t_real, size_t row0, bool first, bool last);
    /// A full-attention layer: GEMM q|k|v|gate -> host attention over the KV rows (rows
    /// [pos0, pos0 + t_real) written) -> GEMM o -> the same tail. `pos0` is the block's
    /// first position, which is pos_ on the block-major route and pos_ + row0 on the
    /// layer-major one; the cached rows below it are pulled off the device on `first` only.
    void block_layer_full(int l, float* xres, size_t T, size_t t_real, size_t pos0, size_t row0, bool first);
    /// The routed experts for whatever the layer's attention half staged in moe_: the
    /// token-batched kernel over rows [0, t_real) of `rows`, or mx one token at a time.
    /// `xres` (rows * hidden) is overwritten with the layer's output.
    void block_layer_moe(int l, size_t rows, size_t t_real, float* xres);
    /// The MoE block for one token on the sequential kernel (lx1 / ax1): xm, the router
    /// record and the residual into `act`, route + run, the new residual out of `xres`.
    void moe_token(int l, const float* xm, const float* res, const float* probs, const int32_t* idx, const float* w,
                   float* out);
    /// The routed experts over the whole block on the token-batched kernel (OPEN-MOE-BATCH):
    /// xm / res [T, hid], the router's idx / w [T, topk] for the first t_real tokens; out [T, hid]
    /// = res + the weighted expert outputs (padding rows carried as res).
    void moe_block(int l, const float* xm, const float* res, const int32_t* idx, const float* w, size_t T, size_t t_real,
                   float* out);
    /// The full-attention layer's attention over the block as GEMM dispatches (OPEN-PREFILL-ATTN):
    /// per kv head, the group's queries against the window's K rows for the scores, the row
    /// softmax on the host, then against the V rows. Q [T, nh*hd] as attention_prep leaves it,
    /// kv the layer's cache with the block's rows already written; og [T, nh*hd] out, gated.
    void attention_npu(int l, const host::AttnGeom& g, const float* Q, const float* gate, const uint16_t* kv,
                       size_t kv_row_elems, float* og);
};

}  // namespace open_qwen36
