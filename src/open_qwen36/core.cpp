/// \file core.cpp
/// \brief The resident open-kernel decode engine: a manifest interpreter (see core.hpp).
#include "open_qwen36/core.hpp"

#include <chrono>
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
    // The VLM bits: the image token the app expands per merged patch, and how the
    // rotary pairs split over (t, h, w). Absent on text-only models.
    image_token_id_ = j.value("image_token_id", -1);
    if (j.contains("rope_parameters") && j["rope_parameters"].is_object()) {
        const auto& rp = j["rope_parameters"];
        if (rp.contains("mrope_section") && rp["mrope_section"].is_array() && rp["mrope_section"].size() == 3) {
            size_t sum = 0;
            for (const auto& v : rp["mrope_section"]) { mrope_section_.push_back(v.get<int>()); sum += v.get<int>(); }
            mrope_interleaved_ = rp.value("mrope_interleaved", false);
            if (sum != man_.rotary_dim / 2)
                throw std::runtime_error("open_qwen36: mrope_section sums to " + std::to_string(sum) + ", not the " +
                                         std::to_string(man_.rotary_dim / 2) + " rotary pairs");
        }
    }
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
    for (int l = 0; l < nl_; ++l)
        for (const auto& s : types_[l]->program) wanted[s.kernel] = true;
    for (const auto& s : man_.tail) wanted[s.kernel] = true;
    for (const auto& [name, d] : man_.kernels)
        if (wanted.count(name)) load_kernel(name, d);
    logits_host_.assign(man_.vocab, 0.f);
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
    pools_.reserve(nl_); consts_.reserve(nl_); act_.reserve(nl_); state_.reserve(nl_);
    for (int l = 0; l < nl_; ++l) {
        const LayerType& lt = *types_[l];
        xrt::bo pool = xrt::ext::bo(*dev_, man_.pool_bytes);
        pools::pack_pool(man_, lt, *file_, l, pool.map<uint8_t*>());
        pool.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        pools_.push_back(std::move(pool));
        xrt::bo c = xrt::ext::bo(*dev_, padup(lt.consts_bytes));
        std::memset(c.map<uint8_t*>(), 0, padup(lt.consts_bytes));
        pools::pack_consts(man_, lt, *file_, l, c.map<uint8_t*>());
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
    // A request with an image rewrote the position records of the rows it used
    // (write_record); the next request expects row p to say position p again.
    if (ptab_dirty_) {
        for (const auto& [name, rg] : man_.per_row_globals) {
            xrt::bo& bo = globals_.at(name);
            pools::build_ptab(man_, rg, ptab_dirty_, bo.map<uint8_t*>());
            bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, ptab_dirty_ * rg.per_row, 0);
        }
        ptab_dirty_ = 0;
    }
    mrope_on_ = false;
    mrope_pos_ = 0;
    pos_ = 0;
}

xrt::bo& Core::buffer(const std::string& name, int layer) {
    if (name == "pool") return pools_[layer];
    if (name == "consts") return consts_[layer];
    if (name == "act") return act_[layer];
    if (name == "state") return state_[layer];
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

void Core::step(int token, bool want_logits) { step_impl(token, nullptr, want_logits, nullptr); }

void Core::step_embed(const float* x, bool want_logits, const int64_t mpos[3]) {
    if (!has_mrope()) throw std::runtime_error("open_qwen36: step_embed on a model without M-RoPE");
    step_impl(-1, x, want_logits, mpos);
}

void Core::mrope_begin() {
    if (mrope_on_) return;
    mrope_on_ = true;
    mrope_pos_ = pos_;
}

void Core::write_record(size_t row, const double pos[3]) {
    for (const auto& [name, rg] : man_.per_row_globals) {
        xrt::bo& bo = globals_.at(name);
        uint8_t* r = bo.map<uint8_t*>() + row * rg.per_row;
        pools::build_ptab_record(man_, rg, row, pos, mrope_section_, mrope_interleaved_, r);
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, rg.per_row, row * rg.per_row);
    }
    if (row + 1 > ptab_dirty_) ptab_dirty_ = row + 1;
}

void Core::step_impl(int token, const float* x, bool want_logits, const int64_t* mpos) {
    if (!weights_loaded_) throw std::runtime_error("open_qwen36: step before load_weights");
    if (static_cast<size_t>(pos_) >= cfg_.max_ctx)
        throw std::runtime_error("open_qwen36: position " + std::to_string(pos_) + " reached the context capacity " +
                                 std::to_string(cfg_.max_ctx));
    if (!x && (token < 0 || static_cast<size_t>(token) >= man_.vocab)) throw std::runtime_error("open_qwen36: token id out of range");
    auto t0 = std::chrono::steady_clock::now();
    timing_ = StepTiming{};

    xrt::bo& xres = buffer("xres", 0);
    if (x)
        std::memcpy(xres.map<float*>(), x, man_.hidden * 4);
    else
        file_->bf16_row(man_.embed_tensor, static_cast<size_t>(token), man_.hidden, xres.map<float*>());
    xres.sync(XCL_BO_SYNC_BO_TO_DEVICE, man_.hidden * 4, 0);
    if (mpos) {
        const double p3[3] = {static_cast<double>(mpos[0]), static_cast<double>(mpos[1]), static_cast<double>(mpos[2])};
        write_record(static_cast<size_t>(pos_), p3);
    } else if (mrope_on_) {
        const double cpos = static_cast<double>(mrope_pos_);
        const double p3[3] = {cpos, cpos, cpos};
        write_record(static_cast<size_t>(pos_), p3);
    }
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
    if (!mpos && mrope_on_) ++mrope_pos_;      // a text token after an image: (c, c, c), then c + 1
    timing_.total_ms = ms_since(t0);
}

void Core::seek(int pos) {
    if (pos < 0 || static_cast<size_t>(pos) >= cfg_.max_ctx) throw std::runtime_error("open_qwen36: seek out of range");
    pos_ = pos;
}

Snapshot Core::checkpoint() const {
    Snapshot s;
    s.pos = pos_;
    s.mrope_pos = mrope_pos_;
    s.mrope_on = mrope_on_;
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
    mrope_pos_ = s.mrope_pos;
    mrope_on_ = s.mrope_on;
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
