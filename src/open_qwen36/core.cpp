/// \file core.cpp
/// \brief The resident open-kernel decode engine: a manifest interpreter (see core.hpp).
#include "open_qwen36/core.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_xclbin.h"

namespace open_qwen36 {

namespace fs = std::filesystem;

namespace {

// 0167/#32: the GEMM route's host math reuses this file's own
// open_qwen36::bf16_to_f32 / open_qwen36::f32_to_bf16 (q4nx_file.hpp) --
// both already round-to-nearest-even, matching open_npue/npue_pack.cpp's
// bf16_rne and ml_dtypes.bfloat16's cast exactly (checked: same bit
// arithmetic, `u + 0x7FFF + ((u>>16)&1)`). NOT open_npue's own tile_b,
// which has internal (anonymous-namespace) linkage and is not declared in
// its header, so it is not callable from here; its algorithm is reproduced
// in tile_gemm_x() below instead, using these two conversions.

constexpr int kOpcode = 3;
constexpr size_t kBoAlign = 1u << 20;  // XDNA wants 1 MB-aligned buffer sizes
size_t padup(size_t n) { return (n + kBoAlign - 1) / kBoAlign * kBoAlign; }

std::vector<uint8_t> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("open_qwen36: cannot read " + p.string());
    std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> v(static_cast<size_t>(n));
    if (n > 0 && !f.read(reinterpret_cast<char*>(v.data()), n)) throw std::runtime_error("open_qwen36: short read " + p.string());
    return v;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

void Core::log(const std::string& s) const {
    if (cfg_.verbose) std::fprintf(stderr, "open_qwen36: %s\n", s.c_str());
}

Core::Core(const CoreConfig& cfg, xrt::device* dev) : cfg_(cfg) {
    // ---- the kernel set's manifest, and the model it must agree with
    man_ = Manifest::load((fs::path(cfg_.kernel_dir) / "manifest.json").string());
    fs::path md(cfg_.model_dir);
    std::ifstream cf(md / "config.json");
    if (!cf) throw std::runtime_error("open_qwen36: no config.json in " + cfg_.model_dir);
    auto j = nlohmann::json::parse(cf, nullptr, false);
    if (!j.is_object()) throw std::runtime_error("open_qwen36: bad config.json in " + cfg_.model_dir);
    man_.check_model(j, md.filename().string());
    int total = static_cast<int>(man_.layers.size());
    nl_ = cfg_.num_layers > 0 && cfg_.num_layers < total ? cfg_.num_layers : total;
    types_.resize(nl_);
    for (int l = 0; l < nl_; ++l) types_[l] = &man_.layer_type(l);
    file_ = std::make_unique<Q4nxFile>((md / "model.q4nx").string());
    int nattn = 0;
    for (int l = 0; l < nl_; ++l) nattn += is_attention_layer(l);
    log("model " + md.filename().string() + " (" + man_.family + ", " + man_.spec_hash.substr(0, 19) + "): " +
        std::to_string(nl_) + " of " + std::to_string(total) + " layers, " + std::to_string(nattn) +
        " attention, context capacity " + std::to_string(cfg_.max_ctx));

    // ---- device, contexts, kernels (only the kernels the running layers' programs and the tail name)
    if (dev) {
        dev_ = dev;
    } else {
        owned_dev_ = std::make_unique<xrt::device>(0u);
        dev_ = owned_dev_.get();
    }
    std::map<std::string, bool> wanted;
    for (int l = 0; l < nl_; ++l) {
        for (const auto& s : types_[l]->program) wanted[s.kernel] = true;
        // 0167/#32: the GEMM-route block's 5 GEMM kernels, plus "dxB"
        // (the attention half, driven directly by Core rather than via a
        // Step -- see manifest.hpp's GemmBlockProgram) which the manifest
        // parser already required to exist whenever gemm_block is present.
        for (const auto& s : types_[l]->gemm_block.program) wanted[s.kernel] = true;
        if (types_[l]->gemm_block.t) wanted["dxB"] = true;
    }
    for (const auto& s : man_.tail) wanted[s.kernel] = true;
    for (const auto& [name, d] : man_.kernels)
        if (wanted.count(name)) load_kernel(name, d);
    logits_host_.assign(man_.vocab, 0.f);

    // 0167/#32: the GEMM-route block size every loaded layer type agrees on.
    // Disagreement (a mixed dense_local/dense manifest where only one carries
    // a gemm_block program) or no gemm_block program anywhere both read as
    // "unsupported" (0), never a guess at which layer type's T applies --
    // step_gemm_block() refuses. gemm_block_t_ == 0 is not an error (most
    // kernel sets have no gemm_block program), but it silently means every
    // prefill runs one token at a time even on a model that DOES have one, if
    // the kernel_dir actually loaded is a stale copy without it
    // (Engine::find_kernels() prefers <model>/open_kernels over
    // FLM_OPEN_KERNELS_DIR/FLM_XCLBIN_PATH -- this project lost real time to
    // exactly that before this log line existed).
    gemm_block_t_ = nl_ > 0 ? types_[0]->gemm_block.t : 0;
    for (int l = 1; l < nl_; ++l)
        if (types_[l]->gemm_block.t != gemm_block_t_) { gemm_block_t_ = 0; break; }
    log("GEMM-route prefill block size (0167/#32): " + std::to_string(gemm_block_t_) +
        (gemm_block_t_ ? "" : " (no gemm_block program in this kernel set, or its layer types disagree)"));
}

Core::~Core() = default;

xrt::hw_context& Core::context(const std::string& name) {
    auto it = ctxs_.find(name);
    if (it != ctxs_.end()) return *it->second;
    fs::path p = fs::path(cfg_.kernel_dir) / man_.contexts.at(name);
    if (!fs::exists(p)) throw std::runtime_error("open_qwen36: missing kernel " + p.string());
    xrt::xclbin xcl(p.string());
    auto uuid = dev_->register_xclbin(xcl);
    auto ctx = std::make_unique<xrt::hw_context>(*dev_, uuid);
    return *(ctxs_[name] = std::move(ctx));
}

void Core::load_kernel(const std::string& name, const KernelDesc& d) {
    fs::path p = fs::path(cfg_.kernel_dir) / d.insts;
    if (!fs::exists(p)) throw std::runtime_error("open_qwen36: missing instruction stream " + p.string());
    Kern& k = kerns_[name];
    k.name = name;
    k.patch = d.patch;
    k.k = std::make_unique<xrt::kernel>(context(d.context), "MLIR_AIE");
    auto insts = read_file(p);
    if (insts.empty() || insts.size() % 4) throw std::runtime_error("open_qwen36: " + p.string() + " is not word-sized");
    k.words.resize(insts.size() / 4);
    std::memcpy(k.words.data(), insts.data(), insts.size());
    k.instr = std::make_unique<xrt::bo>(*dev_, insts.size(), xrt::bo::flags::cacheable, k.k->group_id(1));
    std::memcpy(k.instr->map<void*>(), insts.data(), insts.size());
    k.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    if (d.patch == "moeroute2") k.moe2 = stream_patch::moe2_table(k.words, name, man_.moe);
    else if (d.patch == "attnpos") {
        k.attn = stream_patch::attn_table(k.words, name, man_.attn);
        k.geom = man_.attn;
        k.geom.window = d.window;
    }
}

xrt::bo Core::alloc(size_t bytes, const uint8_t* init, size_t init_bytes) {
    xrt::bo bo = xrt::ext::bo(*dev_, padup(bytes));
    auto* m = bo.map<uint8_t*>();
    std::memset(m, 0, padup(bytes));
    if (init) std::memcpy(m, init, init_bytes);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return bo;
}

void Core::load_weights(const std::function<void(int, int)>& progress) {
    auto t0 = std::chrono::steady_clock::now();
    pools_.clear(); consts_.clear(); act_.clear(); state_.clear(); globals_.clear();
    gqkv3_w_.clear(); go_w_.clear(); ggate_w_.clear(); gup_w_.clear(); gdown_w_.clear();
    ln_w_bf16_.clear(); post_ln_w_bf16_.clear();
    pools_.reserve(nl_); consts_.reserve(nl_); act_.reserve(nl_); state_.reserve(nl_);
    if (gemm_block_t_) {
        gqkv3_w_.resize(nl_); go_w_.resize(nl_); ggate_w_.resize(nl_); gup_w_.resize(nl_); gdown_w_.resize(nl_);
        ln_w_bf16_.resize(nl_); post_ln_w_bf16_.resize(nl_);
    }
    for (int l = 0; l < nl_; ++l) {
        const LayerType& lt = *types_[l];
        xrt::bo pool = xrt::ext::bo(*dev_, man_.pool_bytes);
        uint8_t* pool_host = pool.map<uint8_t*>();
        pools::pack_pool(man_, lt, *file_, l, pool_host);
        // 0167/#32: the GEMM route's 5 per-layer weight buffers,
        // built ONCE here from the SAME freshly-packed host bytes pool.sync()
        // is about to upload -- see core.hpp's field comment for why these
        // are dedicated buffers rather than a sub-range of `pool`. q/k/v are
        // concatenated because lt.pool[0..2] (q,k,v) are already byte-
        // contiguous in this layout (dense.py's pack_plan lays them out back
        // to back with no gaps); this is checked, not assumed.
        if (lt.gemm_block.t) {
            auto [qo, qb] = pool_region(lt, 0);
            auto [ko, kb] = pool_region(lt, 1);
            auto [vo, vb] = pool_region(lt, 2);
            if (ko != qo + qb || vo != ko + kb)
                throw std::runtime_error("open_qwen36: layer " + std::to_string(l) +
                                         ": q/k/v pool regions are not contiguous -- the GEMM route's "
                                         "qkv3 weight buffer cannot be built by a single memcpy");
            xrt::bo w = xrt::ext::bo(*dev_, qb + kb + vb);
            std::memcpy(w.map<uint8_t*>(), pool_host + qo, qb + kb + vb);
            w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            gqkv3_w_[l] = std::move(w);
            auto mk = [&](int idx, std::vector<xrt::bo>& dst) {
                auto [off, bytes] = pool_region(lt, idx);
                xrt::bo b = xrt::ext::bo(*dev_, bytes);
                std::memcpy(b.map<uint8_t*>(), pool_host + off, bytes);
                b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                dst[l] = std::move(b);
            };
            mk(3, go_w_); mk(4, gup_w_); mk(5, ggate_w_); mk(6, gdown_w_);
        }
        pool.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        pools_.push_back(std::move(pool));
        xrt::bo c = xrt::ext::bo(*dev_, padup(lt.consts_bytes));
        std::memset(c.map<uint8_t*>(), 0, padup(lt.consts_bytes));
        uint8_t* c_host = c.map<uint8_t*>();
        pools::pack_consts(man_, lt, *file_, l, c_host);
        // 0167/#32: this route's host RMSNorm needs the two norm
        // weights as host floats. consts.pack always puts input_layernorm at
        // byte 0 and post_attention_layernorm right after it, both ELN =
        // hidden*2 bytes bf16 (open_kernels/recipes/dense.py's DenseLayout;
        // CD_LNW=0, CD_POSTLN=eln) -- checked against a real manifest.json,
        // not assumed. Captured from the SAME host buffer pack_consts() just
        // wrote, before its device sync, so this costs no extra I/O.
        if (lt.gemm_block.t) {
            ln_w_bf16_[l].resize(man_.hidden);
            post_ln_w_bf16_[l].resize(man_.hidden);
            std::memcpy(ln_w_bf16_[l].data(), c_host, man_.hidden * 2);
            std::memcpy(post_ln_w_bf16_[l].data(), c_host + man_.hidden * 2, man_.hidden * 2);
        }
        c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        consts_.push_back(std::move(c));
        act_.push_back(alloc(lt.act_bytes));
        state_.push_back(alloc(lt.state_kind == "kv" ? cfg_.max_ctx * lt.state_row : lt.state_bytes));
        if (progress) progress(l + 1, nl_ + 1);
        if ((l + 1) % 10 == 0 || l + 1 == nl_)
            log(std::to_string(l + 1) + "/" + std::to_string(nl_) + " layers resident (" +
                std::to_string(static_cast<int>(ms_since(t0) / 1000)) + " s)");
    }
    // ---- the globals: the lm_head pool and the final norm's weight from the file, the ptab
    // computed, everything else zero (xres, zero, xresf, hn, logits)
    for (const auto& [name, bytes] : man_.globals) {
        if (name == "lmpool") {
            xrt::bo lm = xrt::ext::bo(*dev_, bytes);
            pools::pack_lmhead(man_, *file_, lm.map<uint8_t*>());
            lm.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            globals_[name] = std::move(lm);
        } else if (name == "normw") {
            size_t n = 0;
            const uint8_t* nw = file_->raw(man_.norm_tensor, &n);
            if (n != man_.norm_bytes) throw std::runtime_error("open_qwen36: " + man_.norm_tensor + " is not " + std::to_string(man_.norm_bytes) + " B");
            globals_[name] = alloc(bytes, nw, n);
        } else {
            globals_[name] = alloc(bytes);
        }
    }
    for (const auto& [name, rg] : man_.per_row_globals) {
        std::vector<uint8_t> pt(cfg_.max_ctx * rg.per_row);
        pools::build_ptab(man_, rg, cfg_.max_ctx, pt.data());
        globals_[name] = alloc(pt.size(), pt.data(), pt.size());
    }
    file_->drop_pages();  // the packers are done with the container; keep only what the steps touch
    if (progress) progress(nl_ + 1, nl_ + 1);
    weights_loaded_ = true;
    pos_ = 0;
    log("weights resident: " + std::to_string(nl_) + " pools + lm_head, " +
        std::to_string(static_cast<int>(ms_since(t0) / 1000)) + " s");
}

void Core::reset() {
    if (!weights_loaded_) throw std::runtime_error("open_qwen36: reset before load_weights");
    // The linear layers' state must start at zero. The KV rows need not: the
    // window read is [0, max(pos, 1)) and row 0 at position 0 is a dummy the
    // kernel masks.
    for (int l = 0; l < nl_; ++l) {
        const LayerType& lt = *types_[l];
        if (lt.state_kind != "linear") continue;
        std::memset(state_[l].map<uint8_t*>(), 0, lt.state_bytes);
        state_[l].sync(XCL_BO_SYNC_BO_TO_DEVICE, lt.state_bytes, 0);
    }
    pos_ = 0;
}

xrt::bo& Core::buffer(const std::string& name, int layer) {
    if (name == "pool") return pools_[layer];
    if (name == "consts") return consts_[layer];
    if (name == "act") return act_[layer];
    if (name == "state") return state_[layer];
    // 0167/#32: the GEMM-route block's per-layer weight buffers
    // (built once in load_weights(), see its own comment for why they are
    // dedicated buffers rather than a sub-range of `pool`).
    auto gemm_w = [&](std::vector<xrt::bo>& v, const char* label) -> xrt::bo& {
        if (layer < 0 || static_cast<size_t>(layer) >= v.size() || !v[layer])
            throw std::runtime_error("open_qwen36: layer " + std::to_string(layer) + " has no '" + label + "' (gemm_block) buffer");
        return v[layer];
    };
    if (name == "gqkv3_w") return gemm_w(gqkv3_w_, "gqkv3_w");
    if (name == "go_w") return gemm_w(go_w_, "go_w");
    if (name == "ggate_w") return gemm_w(ggate_w_, "ggate_w");
    if (name == "gup_w") return gemm_w(gup_w_, "gup_w");
    if (name == "gdown_w") return gemm_w(gdown_w_, "gdown_w");
    auto it = globals_.find(name);
    if (it == globals_.end()) throw std::runtime_error("open_qwen36: the program names an unknown buffer '" + name + "'");
    return it->second;
}

double Core::run(Kern& k, const std::vector<std::string>& args, int layer) {
    auto t0 = std::chrono::steady_clock::now();
    xrt::run r(*k.k);
    r.set_arg(0, kOpcode);
    r.set_arg(1, *k.instr);
    r.set_arg(2, static_cast<int>(k.words.size()));
    int i = 3;
    for (const auto& a : args) r.set_arg(i++, buffer(a, layer));
    r.start();
    auto st = cfg_.timeout_ms ? r.wait(std::chrono::milliseconds(cfg_.timeout_ms)) : r.wait();
    if (st != ERT_CMD_STATE_COMPLETED)
        throw std::runtime_error("open_qwen36: kernel " + k.name + " at position " + std::to_string(pos_) +
                                 " ended in ERT state " + std::to_string(static_cast<int>(st)) +
                                 (st == ERT_CMD_STATE_TIMEOUT ? " (timeout)" : ""));
    return ms_since(t0);
}

void Core::route(Kern& k, int layer, uint64_t act_off) {
    auto t0 = std::chrono::steady_clock::now();
    if (k.moe2.empty()) throw std::runtime_error("open_qwen36: moeroute2 on " + k.name + ", which has no routed-expert table");
    xrt::bo& act = act_[layer];
    const size_t off = act_off + man_.rout_idx_off;
    act.sync(XCL_BO_SYNC_BO_FROM_DEVICE, 32, off);
    uint32_t idx[8];
    std::memcpy(idx, act.map<uint8_t*>() + off, 32);
    for (unsigned s = 0; s < man_.moe.topk; ++s)
        if (idx[s] >= man_.moe.experts) throw std::runtime_error("open_qwen36: router produced expert index " + std::to_string(idx[s]));
    stream_patch::moe2_apply(k.iw(), k.moe2, idx, man_.moe);
    k.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    timing_.route_ms += ms_since(t0);
}

void Core::step(int token, bool want_logits) {
    if (!weights_loaded_) throw std::runtime_error("open_qwen36: step before load_weights");
    if (static_cast<size_t>(pos_) >= cfg_.max_ctx)
        throw std::runtime_error("open_qwen36: position " + std::to_string(pos_) + " reached the context capacity " +
                                 std::to_string(cfg_.max_ctx));
    if (token < 0 || static_cast<size_t>(token) >= man_.vocab) throw std::runtime_error("open_qwen36: token id out of range");
    auto t0 = std::chrono::steady_clock::now();
    timing_ = StepTiming{};

    xrt::bo& xres = buffer("xres", 0);
    file_->bf16_row(man_.embed_tensor, static_cast<size_t>(token), man_.hidden, xres.map<float*>());
    xres.sync(XCL_BO_SYNC_BO_TO_DEVICE, man_.hidden * 4, 0);
    for (auto& [name, k] : kerns_) {
        if (k.patch != "attnpos") continue;
        stream_patch::attn_apply(k.iw(), k.attn, static_cast<uint64_t>(pos_), k.geom);
        k.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    for (int l = 0; l < nl_; ++l) {
        int nrun = 0;
        for (const Step& s : types_[l]->program) {
            Kern& k = kerns_.at(s.kernel);
            if (s.op == "run") {
                double ms = run(k, s.args, l);
                (nrun++ == 0 ? timing_.part0_ms : timing_.part1_ms) += ms;
            } else {
                route(k, l, s.act_off);
            }
        }
    }
    if (want_logits) {
        auto t1 = std::chrono::steady_clock::now();
        for (const Step& s : man_.tail) run(kerns_.at(s.kernel), s.args, 0);
        xrt::bo& lg = buffer("logits", 0);
        lg.sync(XCL_BO_SYNC_BO_FROM_DEVICE, man_.vocab * 4, 0);
        std::memcpy(logits_host_.data(), lg.map<uint8_t*>(), man_.vocab * 4);
        timing_.lmhead_ms = ms_since(t1);
    }
    ++pos_;
    timing_.total_ms = ms_since(t0);
}

// ============================================================================
// 0167/#32: the GEMM prefill route. See manifest.hpp's GemmBlockProgram
// docstring for the shape of the chain; core.hpp's field comments explain
// each buffer's lifetime and why it is per-layer vs global.
// ============================================================================

std::pair<size_t, size_t> Core::pool_region(const LayerType& lt, int idx) const {
    if (idx < 0 || static_cast<size_t>(idx) >= lt.pool.size())
        throw std::runtime_error("open_qwen36: pool_region: index " + std::to_string(idx) + " out of range (" +
                                 std::to_string(lt.pool.size()) + " pool ops)");
    const PackOp& op = lt.pool[static_cast<size_t>(idx)];
    return {static_cast<size_t>(op.dst), static_cast<size_t>(op.nch) * man_.chunk_bytes};
}

void Core::shuttle_buf(xrt::bo& wide, xrt::bo& scratch1, size_t token, size_t act_bytes, bool wide_to_scratch) {
    // `wide` is an explicit argument rather than a fixed per-layer buffer
    // because the GEMM route's T-wide attention scratch ("gact") is a
    // GLOBAL, not per-layer (act_bytes is uniform across every Granite dense
    // layer, so a per-layer copy would only cost memory).
    const size_t off = token * act_bytes;
    if (wide_to_scratch) {
        wide.sync(XCL_BO_SYNC_BO_FROM_DEVICE, act_bytes, off);
        std::memcpy(scratch1.map<uint8_t*>(), wide.map<uint8_t*>() + off, act_bytes);
        scratch1.sync(XCL_BO_SYNC_BO_TO_DEVICE, act_bytes, 0);
    } else {
        scratch1.sync(XCL_BO_SYNC_BO_FROM_DEVICE, act_bytes, 0);
        std::memcpy(wide.map<uint8_t*>() + off, scratch1.map<uint8_t*>(), act_bytes);
        wide.sync(XCL_BO_SYNC_BO_TO_DEVICE, act_bytes, off);
    }
}

void Core::rmsnorm_host(const std::vector<double>& x, size_t T, size_t hid, const std::vector<uint16_t>& w_bf16,
                        double eps, std::vector<float>& out) {
    // Reduction AND the final multiply both in fp64 (trap 11: a fp32
    // reduction over 2560+ terms is not a safe correctness metric at this
    // width). out[t,k] = x[t,k]/sqrt(mean_k(x^2)+eps)*w[k].
    out.assign(T * hid, 0.f);
    for (size_t t = 0; t < T; ++t) {
        const double* row = &x[t * hid];
        double ss = 0;
        for (size_t k = 0; k < hid; ++k) ss += row[k] * row[k];
        const double rms = std::sqrt(ss / static_cast<double>(hid) + eps);
        float* orow = &out[t * hid];
        for (size_t k = 0; k < hid; ++k) {
            const double w = static_cast<double>(bf16_to_f32(w_bf16[k]));
            orow[k] = static_cast<float>((row[k] / rms) * w);
        }
    }
}

void Core::tile_gemm_x(const std::vector<float>& x_tk, size_t T, size_t K, std::vector<uint16_t>& out) {
    // [T,K] fp32 -> bf16, pre-tiled [K,T] "k,n" order (K_TILE=64, MAC 8x8,
    // tile_n=32 -- gemm_q4_prefill.py's own GQP_TILE_N default), matching
    // open_npue/npue_pack.cpp's tile_b algorithm exactly (copied, not
    // called -- that function has internal linkage in its own translation
    // unit). `x_tk` is [T,K] row-major (T rows of K elements, this route's
    // own natural RMSNorm-output layout); the transpose to logical [K,T] is
    // done by indexing, not a separate pass.
    constexpr size_t TK = 64, MAC = 8, TN = 32;
    if (K % TK || T % TN)
        throw std::runtime_error("open_qwen36: gemm-route tile_gemm_x: K=" + std::to_string(K) + " or T=" +
                                 std::to_string(T) + " does not tile by (" + std::to_string(TK) + "," + std::to_string(TN) + ")");
    out.assign(K * T, 0);
    size_t w = 0;
    for (size_t kb = 0; kb < K / TK; ++kb)
        for (size_t nb = 0; nb < T / TN; ++nb)
            for (size_t si = 0; si < TK / MAC; ++si)
                for (size_t ti = 0; ti < TN / MAC; ++ti)
                    for (size_t s = 0; s < MAC; ++s)
                        for (size_t t = 0; t < MAC; ++t) {
                            const size_t r = kb * TK + si * MAC + s;  // K index
                            const size_t c = nb * TN + ti * MAC + t;  // T index
                            out[w++] = f32_to_bf16(x_tk[c * K + r]);
                        }
}

void Core::step_gemm_block_layer(int l, std::vector<double>& xres, size_t T) {
    const LayerType& lt = *types_[l];
    const GemmBlockProgram& gb = lt.gemm_block;
    const size_t hid = man_.hidden, qw = gb.qw, kvw = gb.kvw, ff = gb.ff;
    const size_t n_qkv3 = qw + 2 * kvw;

    // One GEMM dispatch: tile `x` [T,K] -> upload -> run -> download `y` [N,T].
    // Timing: part0_ms sums ALL 5 GEMM dispatches
    // (qkv3, o, gate, up, down); route_ms (otherwise unused by a dense/Granite
    // layer type -- no MoE routing here) is repurposed for the T attention
    // (dxB) dispatches below, so the two are cleanly separable instead of
    // both landing in part1_ms.
    auto run_gemm = [&](size_t idx, const std::vector<float>& x, size_t K, size_t N, std::vector<float>& y_out) {
        const Step& s = gb.program[idx];
        std::vector<uint16_t> xt;
        tile_gemm_x(x, T, K, xt);
        xrt::bo& xb = buffer(s.args[1], 0);
        std::memcpy(xb.map<uint8_t*>(), xt.data(), xt.size() * 2);
        xb.sync(XCL_BO_SYNC_BO_TO_DEVICE, xt.size() * 2, 0);
        Kern& k = kerns_.at(s.kernel);
        timing_.part0_ms += run(k, s.args, l);
        xrt::bo& yb = buffer(s.args[2], 0);
        yb.sync(XCL_BO_SYNC_BO_FROM_DEVICE, N * T * 4, 0);
        y_out.assign(N * T, 0.f);
        std::memcpy(y_out.data(), yb.map<uint8_t*>(), N * T * 4);
    };

    // ---- entry RMSNorm, GEMM A' (qkv3, real q|k|v pool weight, ONE dispatch) ----
    std::vector<float> xnorm;
    rmsnorm_host(xres, T, hid, ln_w_bf16_[l], gb.eps, xnorm);
    std::vector<float> y_qkv3;  // [n_qkv3, T] row-major f32
    run_gemm(0, xnorm, hid, n_qkv3, y_qkv3);

    // ---- T single-token dxB dispatches, position-patched, through a GLOBAL
    // T-wide "gact" scratch buffer, shuttled one token at a time via
    // shuttle_buf() -- proven on hardware before this was wired in. ----
    xrt::bo& gact = buffer("gact", 0);
    xrt::bo& act1 = buffer("act", l);
    const size_t AD = lt.act_bytes;
    {
        // Fill gact's Q/K/V region for every token from y_qkv3's columns
        // (f32 bytes, matching the T=1 "act" buffer's own AD_Q/AD_KVN format
        // -- the SAME format Core::step() writes there today).
        std::vector<uint8_t> host_gact(T * AD, 0);
        for (size_t tk = 0; tk < T; ++tk) {
            uint8_t* base = host_gact.data() + tk * AD;
            float* qd = reinterpret_cast<float*>(base + gb.ad_q);
            float* kd = reinterpret_cast<float*>(base + gb.ad_kvn);
            float* vd = reinterpret_cast<float*>(base + gb.ad_kvn + kvw * 4);
            for (size_t c = 0; c < qw; ++c) qd[c] = y_qkv3[c * T + tk];
            for (size_t c = 0; c < kvw; ++c) kd[c] = y_qkv3[(qw + c) * T + tk];
            for (size_t c = 0; c < kvw; ++c) vd[c] = y_qkv3[(qw + kvw + c) * T + tk];
        }
        std::memcpy(gact.map<uint8_t*>(), host_gact.data(), T * AD);
        gact.sync(XCL_BO_SYNC_BO_TO_DEVICE, T * AD, 0);
    }
    {
        Kern& dxb = kerns_.at("dxB");
        const std::vector<std::string> attn_args = {"pool", "xres", "consts", "state", "act", "ptab"};
        for (size_t tk = 0; tk < T; ++tk) {
            const uint64_t pos = static_cast<uint64_t>(pos_) + tk;
            stream_patch::attn_apply(dxb.iw(), dxb.attn, pos, dxb.geom);
            dxb.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            shuttle_buf(gact, act1, tk, AD, /*wide_to_scratch=*/true);
            timing_.route_ms += run(dxb, attn_args, l);
            shuttle_buf(gact, act1, tk, AD, /*wide_to_scratch=*/false);
        }
    }
    // ---- read back AD_OG (bf16, qw elements/token) as [T,qw] f32 -----------
    std::vector<float> og(T * qw, 0.f);
    {
        gact.sync(XCL_BO_SYNC_BO_FROM_DEVICE, T * AD, 0);
        const uint8_t* base = gact.map<uint8_t*>();
        for (size_t tk = 0; tk < T; ++tk) {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(base + tk * AD + gb.ad_og);
            for (size_t c = 0; c < qw; ++c) og[tk * qw + c] = bf16_to_f32(src[c]);
        }
    }

    // ---- GEMM O (o_proj, real weight, reusing the "qkv"-shaped context) ---
    std::vector<float> y_o;  // [hid, T]
    run_gemm(1, og, qw, hid, y_o);

    // ---- host: residual add, post-attention RMSNorm ------------------------
    std::vector<double> res1(T * hid);
    for (size_t tk = 0; tk < T; ++tk)
        for (size_t c = 0; c < hid; ++c) res1[tk * hid + c] = xres[tk * hid + c] + static_cast<double>(y_o[c * T + tk]);
    std::vector<float> xm;
    rmsnorm_host(res1, T, hid, post_ln_w_bf16_[l], gb.eps, xm);

    // ---- GEMM gate_proj + up_proj (SAME context, zero switch between them) -
    std::vector<float> y_gate, y_up;  // both [ff, T]
    run_gemm(2, xm, hid, ff, y_gate);
    run_gemm(3, xm, hid, ff, y_up);

    // ---- host SwiGLU: silu(gate) * up ---------------------------------------
    std::vector<float> h(T * ff);
    for (size_t tk = 0; tk < T; ++tk)
        for (size_t c = 0; c < ff; ++c) {
            const double g = static_cast<double>(y_gate[c * T + tk]);
            const double u = static_cast<double>(y_up[c * T + tk]);
            h[tk * ff + c] = static_cast<float>((g / (1.0 + std::exp(-g))) * u);
        }

    // ---- GEMM down_proj, then residual -> next layer's xres -----------------
    std::vector<float> y_down;  // [hid, T]
    run_gemm(4, h, ff, hid, y_down);
    for (size_t tk = 0; tk < T; ++tk)
        for (size_t c = 0; c < hid; ++c) xres[tk * hid + c] = res1[tk * hid + c] + static_cast<double>(y_down[c * T + tk]);
}

void Core::step_gemm_block(const std::vector<int>& ids, size_t t_real, bool want_logits) {
    if (!weights_loaded_) throw std::runtime_error("open_qwen36: step_gemm_block before load_weights");
    const size_t T = ids.size();
    if (T == 0) return;
    if (gemm_block_t_ == 0 || T != gemm_block_t_)
        throw std::runtime_error("open_qwen36: step_gemm_block called with " + std::to_string(T) +
                                 " tokens, but this kernel set's gemm-route block size is " +
                                 std::to_string(gemm_block_t_) + " (0 = no gemm_block program loaded)");
    if (t_real == 0 || t_real > T) throw std::runtime_error("open_qwen36: step_gemm_block: t_real must be in (0, T]");
    // Positions [pos_, pos_+T) are all touched (padding columns included --
    // hardware-proven exact: the real columns' output does not depend on
    // what the padding columns carry), so the CAPACITY check must cover T,
    // not just the real tokens -- even though only t_real of them advance
    // pos_ afterward.
    if (static_cast<size_t>(pos_) + T > cfg_.max_ctx)
        throw std::runtime_error("open_qwen36: gemm-block [" + std::to_string(pos_) + ", " +
                                 std::to_string(pos_ + T) + ") would exceed the context capacity " +
                                 std::to_string(cfg_.max_ctx));
    for (int tok : ids)
        if (tok < 0 || static_cast<size_t>(tok) >= man_.vocab) throw std::runtime_error("open_qwen36: token id out of range");

    auto t0 = std::chrono::steady_clock::now();
    timing_ = StepTiming{};

    // ---- embed all T tokens (padding included) into a HOST-resident fp64
    // xres[T,hidden] -- this route's running residual stream lives on the
    // HOST between GEMM dispatches (RMSNorm/residual/SwiGLU are host-side,
    // not fused on-core), so there is no device-resident T-wide buffer for
    // it, unlike the per-layer weight/activation buffers below. ------------
    std::vector<double> xres(T * man_.hidden);
    {
        std::vector<float> row(man_.hidden);
        for (size_t tk = 0; tk < T; ++tk) {
            file_->bf16_row(man_.embed_tensor, static_cast<size_t>(ids[tk]), man_.hidden, row.data());
            for (size_t c = 0; c < man_.hidden; ++c) xres[tk * man_.hidden + c] = static_cast<double>(row[c]);
        }
    }

    for (int l = 0; l < nl_; ++l) {
        if (types_[l]->gemm_block.t == 0)
            throw std::runtime_error("open_qwen36: layer " + std::to_string(l) + " (" + types_[l]->name + ") has no gemm_block program");
        step_gemm_block_layer(l, xres, T);
    }
    pos_ += static_cast<int>(t_real);

    if (want_logits) {
        // Prefill wants only the (t_real-1)th (the true last REAL) token's
        // logits -- matches step()'s own "prefill wants only the final
        // token's logits" contract, generalized past T-1 for a padded final
        // block.
        auto t1 = std::chrono::steady_clock::now();
        xrt::bo& xres1 = buffer("xres", 0);
        const size_t last = t_real - 1;
        std::vector<float> last_row(man_.hidden);
        for (size_t c = 0; c < man_.hidden; ++c) last_row[c] = static_cast<float>(xres[last * man_.hidden + c]);
        std::memcpy(xres1.map<uint8_t*>(), last_row.data(), man_.hidden * 4);
        xres1.sync(XCL_BO_SYNC_BO_TO_DEVICE, man_.hidden * 4, 0);
        for (const Step& s : man_.tail) run(kerns_.at(s.kernel), s.args, 0);
        xrt::bo& lg = buffer("logits", 0);
        lg.sync(XCL_BO_SYNC_BO_FROM_DEVICE, man_.vocab * 4, 0);
        std::memcpy(logits_host_.data(), lg.map<uint8_t*>(), man_.vocab * 4);
        timing_.lmhead_ms = ms_since(t1);
    }
    timing_.total_ms = ms_since(t0);
}

void Core::seek(int pos) {
    if (pos < 0 || static_cast<size_t>(pos) >= cfg_.max_ctx) throw std::runtime_error("open_qwen36: seek out of range");
    pos_ = pos;
}

Snapshot Core::checkpoint() const {
    Snapshot s;
    s.pos = pos_;
    for (int l = 0; l < nl_; ++l) {
        const LayerType& lt = *types_[l];
        xrt::bo& bo = const_cast<xrt::bo&>(state_[l]);
        if (lt.state_kind == "kv") {
            size_t n = static_cast<size_t>(pos_) * lt.state_row;
            std::vector<uint8_t> rows(n);
            if (n) {
                bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, n, 0);
                std::memcpy(rows.data(), bo.map<uint8_t*>(), n);
            }
            s.kv.push_back(std::move(rows));
        } else {
            std::vector<uint8_t> st(lt.state_bytes);
            bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, lt.state_bytes, 0);
            std::memcpy(st.data(), bo.map<uint8_t*>(), lt.state_bytes);
            s.states.push_back(std::move(st));
        }
    }
    return s;
}

void Core::restore(const Snapshot& s) {
    size_t il = 0, ia = 0;
    for (int l = 0; l < nl_; ++l) {
        const LayerType& lt = *types_[l];
        if (lt.state_kind == "kv") {
            const auto& rows = s.kv.at(ia++);
            if (!rows.empty()) {
                std::memcpy(state_[l].map<uint8_t*>(), rows.data(), rows.size());
                state_[l].sync(XCL_BO_SYNC_BO_TO_DEVICE, rows.size(), 0);
            }
        } else {
            const auto& st = s.states.at(il++);
            if (st.size() != lt.state_bytes) throw std::runtime_error("open_qwen36: snapshot state size mismatch");
            std::memcpy(state_[l].map<uint8_t*>(), st.data(), lt.state_bytes);
            state_[l].sync(XCL_BO_SYNC_BO_TO_DEVICE, lt.state_bytes, 0);
        }
    }
    pos_ = s.pos;
}

void Core::kv_row(int layer, int row, bool value, uint16_t* out) {
    if (layer < 0 || layer >= nl_ || !is_attention_layer(layer)) throw std::runtime_error("open_qwen36: layer " + std::to_string(layer) + " has no KV cache");
    if (row < 0 || static_cast<size_t>(row) >= cfg_.max_ctx) throw std::runtime_error("open_qwen36: KV row out of range");
    const size_t kv_row = types_[layer]->state_row;
    size_t off = static_cast<size_t>(row) * kv_row + (value ? kv_row / 2 : 0);
    state_[layer].sync(XCL_BO_SYNC_BO_FROM_DEVICE, kv_row / 2, off);
    std::memcpy(out, state_[layer].map<uint8_t*>() + off, kv_row / 2);
}

}  // namespace open_qwen36
