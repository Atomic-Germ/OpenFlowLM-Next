/// \file core.cpp
/// \brief The resident open-kernel decode engine (see core.hpp).
#include "open_qwen36/core.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_xclbin.h"

namespace open_qwen36 {

namespace fs = std::filesystem;
using namespace layout;

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
    // ---- the model: config.json for the layer schedule, the container for the weights
    fs::path md(cfg_.model_dir);
    std::ifstream cf(md / "config.json");
    if (!cf) throw std::runtime_error("open_qwen36: no config.json in " + cfg_.model_dir);
    auto j = nlohmann::json::parse(cf, nullptr, false);
    if (!j.is_object()) throw std::runtime_error("open_qwen36: bad config.json in " + cfg_.model_dir);
    int total = j.value("num_hidden_layers", 0);
    interval_ = j.value("full_attention_interval", 4);
    if (total <= 0 || interval_ <= 0) throw std::runtime_error("open_qwen36: config.json lacks num_hidden_layers / full_attention_interval");
    if (j.value("hidden_size", 0) != static_cast<int>(HIDDEN) || j.value("vocab_size", 0) != static_cast<int>(VOCAB) ||
        j.value("num_experts", 0) != 256 || j.value("moe_intermediate_size", 0) != 512)
        throw std::runtime_error("open_qwen36: the open kernels are built for hidden 2048 / vocab 248320 / 256 experts x 512; this config differs");
    nl_ = cfg_.num_layers > 0 && cfg_.num_layers < total ? cfg_.num_layers : total;
    full_.resize(nl_);
    for (int l = 0; l < nl_; ++l) {
        full_[l] = ((l + 1) % interval_ == 0);
        (full_[l] ? has_attn_ : has_lin_) = true;
    }
    file_ = std::make_unique<Q4nxFile>((md / "model.q4nx").string());
    log("model " + md.filename().string() + ": " + std::to_string(nl_) + " of " + std::to_string(total) +
        " layers, attention every " + std::to_string(interval_) + "th, context capacity " + std::to_string(cfg_.max_ctx));

    // ---- device, contexts, kernels
    if (dev) {
        dev_ = dev;
    } else {
        owned_dev_ = std::make_unique<xrt::device>(0u);
        dev_ = owned_dev_.get();
    }
    if (has_lin_) {
        auto& cx = load_context(ctx_x_, "lx0");
        load_kernel(lx0_, cx, "lx0");
        load_kernel(lx1_, cx, "lx1");
        lx1_.moe2 = stream_patch::moe2_table(lx1_.words, "lx1");
    }
    if (has_attn_) {
        auto& cy = load_context(ctx_y_, "ax0");
        load_kernel(ax0_, cy, "ax0");
        load_kernel(ax1_, cy, "ax1");
        ax0_.attn = stream_patch::attn_table(ax0_.words, "ax0");
        ax1_.moe2 = stream_patch::moe2_table(ax1_.words, "ax1");
    }
    load_kernel(ln_, load_context(ctx_l_, "ln"), "ln");
    load_kernel(lm_, load_context(ctx_k_, "lm_head_q8"), "lm_head_q8");
    logits_host_.assign(VOCAB, 0.f);
}

Core::~Core() = default;

xrt::hw_context& Core::load_context(std::unique_ptr<xrt::hw_context>& ctx, const std::string& design) {
    fs::path p = fs::path(cfg_.kernel_dir) / design / "final.xclbin";
    if (!fs::exists(p)) throw std::runtime_error("open_qwen36: missing kernel " + p.string());
    xrt::xclbin xcl(p.string());
    auto uuid = dev_->register_xclbin(xcl);
    ctx = std::make_unique<xrt::hw_context>(*dev_, uuid);
    return *ctx;
}

void Core::load_kernel(Kern& k, xrt::hw_context& ctx, const std::string& design) {
    fs::path p = fs::path(cfg_.kernel_dir) / design / "insts.bin";
    if (!fs::exists(p)) throw std::runtime_error("open_qwen36: missing instruction stream " + p.string());
    k.name = design;
    k.k = std::make_unique<xrt::kernel>(ctx, "MLIR_AIE");
    auto insts = read_file(p);
    if (insts.empty() || insts.size() % 4) throw std::runtime_error("open_qwen36: " + p.string() + " is not word-sized");
    k.words.resize(insts.size() / 4);
    std::memcpy(k.words.data(), insts.data(), insts.size());
    k.instr = std::make_unique<xrt::bo>(*dev_, insts.size(), xrt::bo::flags::cacheable, k.k->group_id(1));
    std::memcpy(k.instr->map<void*>(), insts.data(), insts.size());
    k.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
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
    pools_.clear(); consts_.clear(); act_.clear(); state_.clear();
    pools_.reserve(nl_); consts_.reserve(nl_); act_.reserve(nl_); state_.reserve(nl_);
    for (int l = 0; l < nl_; ++l) {
        xrt::bo pool = xrt::ext::bo(*dev_, kPoolBytes);
        pools::build_layer_pool(*file_, l, full_[l], pool.map<uint8_t*>());
        pool.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        pools_.push_back(std::move(pool));
        size_t cb = full_[l] ? CA_BYTES : C_BYTES;
        xrt::bo c = xrt::ext::bo(*dev_, padup(cb));
        std::memset(c.map<uint8_t*>(), 0, padup(cb));
        pools::build_consts(*file_, l, full_[l], c.map<uint8_t*>());
        c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        consts_.push_back(std::move(c));
        act_.push_back(alloc(full_[l] ? AA_BYTES : A_BYTES));
        state_.push_back(alloc(full_[l] ? cfg_.max_ctx * KV_ROW : STATE_BYTES));
        if (progress) progress(l + 1, nl_ + 1);
        if ((l + 1) % 10 == 0 || l + 1 == nl_)
            log(std::to_string(l + 1) + "/" + std::to_string(nl_) + " layers resident (" +
                std::to_string(static_cast<int>(ms_since(t0) / 1000)) + " s)");
    }
    {
        xrt::bo lm = xrt::ext::bo(*dev_, kLmheadPoolBytes);
        pools::build_lmhead_pool(*file_, lm.map<uint8_t*>());
        lm.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        lmpool_ = std::move(lm);
    }
    {
        std::vector<uint8_t> pt(cfg_.max_ctx * PTAB_ROW);
        pools::build_ptab(cfg_.max_ctx, pt.data());
        ptab_ = alloc(pt.size(), pt.data(), pt.size());
    }
    {
        size_t n = 0;
        const uint8_t* nw = file_->raw("model.norm.weight", &n);
        if (n != HIDDEN * 2) throw std::runtime_error("open_qwen36: model.norm.weight is not bf16[2048]");
        normw_ = alloc(n, nw, n);
    }
    xres_ = alloc(HIDDEN * 4);
    zero_ = alloc(HIDDEN * 4);
    xresf_ = alloc(HIDDEN * 4);
    hn_ = alloc(HIDDEN * 2);
    logits_ = alloc(LOGITS_BYTES);
    file_->drop_pages();  // the packers are done with the container; keep only what the steps touch
    if (progress) progress(nl_ + 1, nl_ + 1);
    weights_loaded_ = true;
    pos_ = 0;
    log("weights resident: " + std::to_string(nl_) + " pools + lm_head, " +
        std::to_string(static_cast<int>(ms_since(t0) / 1000)) + " s");
}

void Core::reset() {
    if (!weights_loaded_) throw std::runtime_error("open_qwen36: reset before load_weights");
    // The linear layers' conv state and S must start at zero. The KV rows
    // need not: the window read is [0, max(pos, 1)) and row 0 at position 0
    // is a dummy the kernel masks.
    for (int l = 0; l < nl_; ++l) {
        if (full_[l]) continue;
        std::memset(state_[l].map<uint8_t*>(), 0, STATE_BYTES);
        state_[l].sync(XCL_BO_SYNC_BO_TO_DEVICE, STATE_BYTES, 0);
    }
    pos_ = 0;
}

double Core::run(Kern& k, std::initializer_list<xrt::bo*> bufs) {
    auto t0 = std::chrono::steady_clock::now();
    xrt::run r(*k.k);
    r.set_arg(0, kOpcode);
    r.set_arg(1, *k.instr);
    r.set_arg(2, static_cast<int>(k.words.size()));
    int i = 3;
    for (xrt::bo* b : bufs) r.set_arg(i++, *b);
    r.start();
    auto st = cfg_.timeout_ms ? r.wait(std::chrono::milliseconds(cfg_.timeout_ms)) : r.wait();
    if (st != ERT_CMD_STATE_COMPLETED)
        throw std::runtime_error("open_qwen36: kernel " + k.name + " at position " + std::to_string(pos_) +
                                 " ended in ERT state " + std::to_string(static_cast<int>(st)) +
                                 (st == ERT_CMD_STATE_TIMEOUT ? " (timeout)" : ""));
    return ms_since(t0);
}

void Core::route_and_run_part1(Kern& part1, xrt::bo& act, size_t rout_off, std::initializer_list<xrt::bo*> bufs) {
    auto t0 = std::chrono::steady_clock::now();
    act.sync(XCL_BO_SYNC_BO_FROM_DEVICE, 32, rout_off + ROUT_IDX_OFF);
    uint32_t idx[8];
    std::memcpy(idx, act.map<uint8_t*>() + rout_off + ROUT_IDX_OFF, 32);
    for (uint32_t e : idx)
        if (e >= 256) throw std::runtime_error("open_qwen36: router produced expert index " + std::to_string(e));
    stream_patch::moe2_apply(part1.iw(), part1.moe2, idx);
    part1.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    timing_.route_ms += ms_since(t0);
    timing_.part1_ms += run(part1, bufs);
}

void Core::step(int token, bool want_logits) {
    if (!weights_loaded_) throw std::runtime_error("open_qwen36: step before load_weights");
    if (static_cast<size_t>(pos_) >= cfg_.max_ctx)
        throw std::runtime_error("open_qwen36: position " + std::to_string(pos_) + " reached the context capacity " +
                                 std::to_string(cfg_.max_ctx));
    if (token < 0 || static_cast<size_t>(token) >= VOCAB) throw std::runtime_error("open_qwen36: token id out of range");
    auto t0 = std::chrono::steady_clock::now();
    timing_ = StepTiming{};

    file_->bf16_row("model.embed_tokens.weight", static_cast<size_t>(token), HIDDEN, xres_.map<float*>());
    xres_.sync(XCL_BO_SYNC_BO_TO_DEVICE, HIDDEN * 4, 0);
    if (has_attn_) {
        stream_patch::attn_apply(ax0_.iw(), ax0_.attn, static_cast<uint64_t>(pos_));
        ax0_.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    for (int l = 0; l < nl_; ++l) {
        if (full_[l]) {
            timing_.part0_ms += run(ax0_, {&pools_[l], &xres_, &consts_[l], &state_[l], &act_[l], &ptab_});
            route_and_run_part1(ax1_, act_[l], AA_ROUT, {&pools_[l], &xres_, &consts_[l], &state_[l], &act_[l], &ptab_});
        } else {
            timing_.part0_ms += run(lx0_, {&pools_[l], &xres_, &consts_[l], &state_[l], &act_[l]});
            route_and_run_part1(lx1_, act_[l], A_ROUT, {&pools_[l], &xres_, &consts_[l], &state_[l], &act_[l]});
        }
    }
    if (want_logits) {
        auto t1 = std::chrono::steady_clock::now();
        run(ln_, {&xres_, &zero_, &normw_, &xresf_, &hn_});
        run(lm_, {&lmpool_, &hn_, &logits_});
        logits_.sync(XCL_BO_SYNC_BO_FROM_DEVICE, LOGITS_BYTES, 0);
        std::memcpy(logits_host_.data(), logits_.map<uint8_t*>(), LOGITS_BYTES);
        timing_.lmhead_ms = ms_since(t1);
    }
    ++pos_;
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
        xrt::bo& bo = const_cast<xrt::bo&>(state_[l]);
        if (full_[l]) {
            size_t n = static_cast<size_t>(pos_) * KV_ROW;
            std::vector<uint8_t> rows(n);
            if (n) {
                bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, n, 0);
                std::memcpy(rows.data(), bo.map<uint8_t*>(), n);
            }
            s.kv.push_back(std::move(rows));
        } else {
            std::vector<uint8_t> st(STATE_BYTES);
            bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, STATE_BYTES, 0);
            std::memcpy(st.data(), bo.map<uint8_t*>(), STATE_BYTES);
            s.states.push_back(std::move(st));
        }
    }
    return s;
}

void Core::restore(const Snapshot& s) {
    size_t il = 0, ia = 0;
    for (int l = 0; l < nl_; ++l) {
        if (full_[l]) {
            const auto& rows = s.kv.at(ia++);
            if (!rows.empty()) {
                std::memcpy(state_[l].map<uint8_t*>(), rows.data(), rows.size());
                state_[l].sync(XCL_BO_SYNC_BO_TO_DEVICE, rows.size(), 0);
            }
        } else {
            const auto& st = s.states.at(il++);
            std::memcpy(state_[l].map<uint8_t*>(), st.data(), STATE_BYTES);
            state_[l].sync(XCL_BO_SYNC_BO_TO_DEVICE, STATE_BYTES, 0);
        }
    }
    pos_ = s.pos;
}

void Core::kv_row(int layer, int row, bool value, uint16_t* out) {
    if (layer < 0 || layer >= nl_ || !full_[layer]) throw std::runtime_error("open_qwen36: layer " + std::to_string(layer) + " has no KV cache");
    if (row < 0 || static_cast<size_t>(row) >= cfg_.max_ctx) throw std::runtime_error("open_qwen36: KV row out of range");
    size_t off = static_cast<size_t>(row) * KV_ROW + (value ? KV_ROW / 2 : 0);
    state_[layer].sync(XCL_BO_SYNC_BO_FROM_DEVICE, KV_ROW / 2, off);
    std::memcpy(out, state_[layer].map<uint8_t*>() + off, KV_ROW / 2);
}

}  // namespace open_qwen36
