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
/// program and planned as `OpenBackend`. It has no dependency on the FLM app
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

#include "open_qwen36/gguf_file.hpp"
#include "open_qwen36/manifest.hpp"
#include "open_qwen36/pools.hpp"
#include "open_qwen36/q4nx_file.hpp"
#include "open_qwen36/weight_file.hpp"
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
    std::vector<std::vector<uint8_t>> states;  ///< per linear layer: the state BO
    std::vector<std::vector<uint8_t>> kv;      ///< per attention layer: rows [0, pos)
};

struct StepTiming {
    double part0_ms = 0, part1_ms = 0, route_ms = 0, lmhead_ms = 0, total_ms = 0;
};

/// The config.json fields a manifest checks, derived from a GGUF's metadata
/// (the llama-family `<arch>.embedding_length` keys). A key the model does not
/// carry is refused by name; families whose config carries GGUF-invisible
/// fields (Granite's folded multipliers) need their config.json.
nlohmann::json derive_config(const GgufFile& g);

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
    const std::vector<float>& logits() const { return logits_host_; }

    int position() const { return pos_; }
    /// Test hook: place the next token at `pos` without decoding up to it.
    void seek(int pos);
    size_t max_ctx() const { return cfg_.max_ctx; }
    int num_layers() const { return nl_; }
    bool is_attention_layer(int l) const { return types_[l]->state_kind == "kv"; }
    const StepTiming& last_timing() const { return timing_; }
    const Manifest& manifest() const { return man_; }
    /// The selected weights view: the manifest itself (q4nx) or its GGUF-direct twin.
    const Manifest& weights() const { return w_ ? *w_ : man_; }
    bool is_gguf() const { return gguf_; }
    size_t vocab() const { return weights().vocab; }
    size_t real_vocab() const { return weights().real_vocab; }

    Snapshot checkpoint() const;
    void restore(const Snapshot& s);

    /// One cached row of an attention layer's K or V (bf16, kv_row / 4 elements).
    void kv_row(int layer, int row, bool value, uint16_t* out);

    const WeightFile& file() const { return *file_; }

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
    const Manifest* w_ = nullptr;              ///< the weights view (man_ or man_.gguf.get())
    bool gguf_ = false;                        ///< the weights are a GGUF (model.gguf)
    std::unique_ptr<WeightFile> file_;
    int nl_ = 0;
    std::vector<const LayerType*> types_;      ///< per layer

    std::unique_ptr<xrt::device> owned_dev_;
    xrt::device* dev_ = nullptr;
    std::map<std::string, std::unique_ptr<xrt::hw_context>> ctxs_;
    std::map<std::string, Kern> kerns_;

    std::vector<xrt::bo> pools_, consts_, act_, state_;   ///< per layer
    std::map<std::string, xrt::bo> globals_;              ///< the manifest's globals (xres, ptab, lmpool, ...)
    bool weights_loaded_ = false;
    int pos_ = 0;
    std::vector<float> logits_host_;
    StepTiming timing_;

    xrt::hw_context& context(const std::string& name);
    void load_kernel(const std::string& name, const KernelDesc& d);
    xrt::bo alloc(size_t bytes, const uint8_t* init = nullptr, size_t init_bytes = 0);
    xrt::bo& buffer(const std::string& name, int layer);
    double run(Kern& k, const std::vector<std::string>& args, int layer);
    void route(Kern& k, int layer, uint64_t act_off);
    void log(const std::string& s) const;
};

}  // namespace open_qwen36
