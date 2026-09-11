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
/// Prefill is decode-as-prefill: the prompt goes through `step()` one token at
/// a time from zeroed state with logits skipped, which is exact for this
/// architecture (each layer's state update sees one token at a time) and is
/// the only prefill the open kernels have.
#pragma once

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

struct StepTiming {
    double part0_ms = 0, part1_ms = 0, route_ms = 0, lmhead_ms = 0, total_ms = 0;
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
    void step_embed(const float* x, bool want_logits, const int64_t mpos[3]);
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
    /// 0167/#32: the GEMM-route batched-prefill block size (manifest.hpp's
    /// GemmBlockProgram), or 0 when the loaded kernel set has none / its
    /// layer types disagree -- refuses rather than guesses.
    size_t gemm_block_t() const { return gemm_block_t_; }
    /// T tokens through every layer as 5 GEMM dispatches (q|k|v fused,
    /// o_proj, gate_proj, up_proj, down_proj) plus T single-token attnpos-
    /// patched attention dispatches between GEMM A' and GEMM O, with
    /// HOST-side fp64 RMSNorm/residual/SwiGLU between every GEMM stage (see
    /// manifest.hpp's GemmBlockProgram docstring for the full chain).
    /// `ids.size()` must equal gemm_block_t(); the caller pads a short tail
    /// with any in-range token id (hardware-proven exact: the real columns'
    /// output does not depend on what the padding columns carry) and passes
    /// the REAL count as `t_real` so position only advances by the real
    /// tokens. Logits, like step(), only for the (t_real-1)th token, only
    /// when asked.
    void step_gemm_block(const std::vector<int>& ids, size_t t_real, bool want_logits);

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
        std::vector<stream_patch::MoePatch> moe2;
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
    std::vector<int> mrope_section_;          ///< empty: no M-RoPE (every model but the VLMs)
    bool mrope_interleaved_ = false;
    int image_token_id_ = -1;
    bool mrope_on_ = false;
    int64_t mrope_pos_ = 0;
    size_t ptab_dirty_ = 0;                    ///< rows [0, dirty) hold per-request records; reset() restores them

    // ---- 0167/#32: the GEMM-route block (see manifest.hpp's GemmBlockProgram)
    size_t gemm_block_t_ = 0;    ///< common gemm_block.t across every loaded layer type, or 0
    // Per-layer, dedicated (byte-0-based) weight buffers built ONCE in
    // load_weights() by copying the already-packed pool bytes at the q/k/v,
    // o, up, gate, down PackOp offsets (Manifest::layer_type(l).pool[0..6],
    // fixed order per open_kernels/recipes/dense.py's pack_plan/programs) --
    // NOT sub-ranges of pools_[l] passed with an offset: the GEMM kernels
    // (gemm_q4_prefill.py) were traced expecting their "w" arg to start at
    // byte 0 of its own bound buffer, and an XRT device-side sub-buffer view
    // is an untested mechanism in this tree -- so this pays a one-time
    // ~1.9 GB extra resident-memory cost (5 buffers/layer x 40 layers) for a
    // mechanism already proven on hardware, rather than an unvalidated one.
    // q/k/v are concatenated because they are ALREADY byte-contiguous in
    // the existing pool layout (POOL_K immediately follows POOL_Q+QB, POOL_V
    // immediately follows POOL_K+KB -- confirmed against a real manifest.json,
    // not assumed), so this is a single memcpy, no interleaving.
    std::vector<xrt::bo> gqkv3_w_, go_w_, ggate_w_, gup_w_, gdown_w_;   ///< per layer, only when gemm_block.t > 0
    // Host-resident copies of input_layernorm.weight / post_attention_layernorm.weight
    // (bf16, hidden elements each), captured once from the SAME host buffer
    // pack_consts() just wrote (before its device sync) -- this route's
    // RMSNorm runs on the HOST in fp64 (rmsnorm_host() below), so it needs
    // these as host floats, not a device buffer.
    std::vector<std::vector<uint16_t>> ln_w_bf16_, post_ln_w_bf16_;    ///< per layer, only when gemm_block.t > 0

    std::vector<float> logits_host_;
    StepTiming timing_;

    xrt::hw_context& context(const std::string& name);
    void load_kernel(const std::string& name, const KernelDesc& d);
    xrt::bo alloc(size_t bytes, const uint8_t* init = nullptr, size_t init_bytes = 0);
    xrt::bo& buffer(const std::string& name, int layer);
    void step_impl(int token, const float* x, bool want_logits, const int64_t* mpos);
    /// Write KV row `row`'s position record from (t, h, w) into every position table.
    void write_record(size_t row, const double pos[3]);
    double run(Kern& k, const std::vector<std::string>& args, int layer);
    void route(Kern& k, int layer, uint64_t act_off);
    void log(const std::string& s) const;

    // ---- 0167/#32: the GEMM-route block's own helpers (core.cpp)
    /// The byte {offset, length} of `lt.pool[idx]` (idx: 0=q 1=k 2=v 3=o
    /// 4=up 5=gate 6=down, per open_kernels/recipes/dense.py's pack_plan
    /// order) inside a layer's packed pool buffer. length = nch * chunk_bytes,
    /// the SAME formula gemm_q4_prefill.py's own `pool_bytes = n_weight * k *
    /// 5 // 8` computes (cross-checked against a real manifest.json).
    std::pair<size_t, size_t> pool_region(const LayerType& lt, int idx) const;
    /// Host-side shuttle of one token's `act_bytes` slice between a GLOBAL
    /// T-wide scratch buffer (`wide`, e.g. "gact") and an ordinary T=1
    /// per-layer scratch buffer (`scratch1`, e.g. "act") -- a map+memcpy+sync
    /// round trip, the same idiom open_kernels/harness/run_kernel.cpp's
    /// `copy` directive uses, validated on hardware before this was wired
    /// into the engine.
    void shuttle_buf(xrt::bo& wide, xrt::bo& scratch1, size_t token, size_t act_bytes, bool wide_to_scratch);
    /// out[t,:] = x[t,:] / sqrt(mean(x[t,:]^2) + eps) * w[:], reduction and
    /// the final multiply both in fp64 (trap 11: fp32 is not a safe
    /// reduction width at this hidden size). w is bf16 (hidden elements);
    /// out is written as fp32 (ready for tile_gemm_x).
    static void rmsnorm_host(const std::vector<double>& x, size_t T, size_t hid,
                             const std::vector<uint16_t>& w_bf16, double eps, std::vector<float>& out);
    /// [T,K] fp32 -> bf16, pre-tiled into [K,T] "k,n" order (K_TILE=64,
    /// MAC 8x8) -- a local port of the SAME algorithm
    /// open_npue/npue_pack.cpp's tile_b implements (internal linkage there,
    /// not callable from here -- copied rather than exported, see this
    /// route's own history in tasks/0167 for why). Writes K*T bf16 elements
    /// (raw bits) to `out`.
    static void tile_gemm_x(const std::vector<float>& x_tk, size_t T, size_t K, std::vector<uint16_t>& out);
    /// One layer of the GEMM-route chain (manifest.hpp's GemmBlockProgram
    /// docstring): entry RMSNorm -> GEMM A' (qkv3) -> T dxB dispatches at
    /// positions [pos_, pos_+T) -> GEMM O -> residual + post-attn RMSNorm ->
    /// GEMM gate + GEMM up -> host SwiGLU -> GEMM down -> residual. `xres` is
    /// T*hidden fp64, updated in place (this layer's output becomes the next
    /// layer's input) -- positions come from the member `pos_`, unchanged by
    /// this call (the caller advances it once per BLOCK, not per layer).
    void step_gemm_block_layer(int l, std::vector<double>& xres, size_t T);
};

}  // namespace open_qwen36
