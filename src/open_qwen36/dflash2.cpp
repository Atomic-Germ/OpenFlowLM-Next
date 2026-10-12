/// \file dflash2.cpp
/// \brief The DFlash 2 block drafter on the NPU (see dflash2.hpp).
#include "open_qwen36/dflash2.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <immintrin.h>

#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_xclbin.h"

#include "open_qwen36/block_host.hpp"
#include "open_qwen36/pools.hpp"

namespace open_qwen36 {
namespace dflash2 {

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

constexpr int kOpcode = 3;
constexpr size_t kBoAlign = 1u << 20;
size_t padup(size_t n) { return (n + kBoAlign - 1) / kBoAlign * kBoAlign; }

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

json read_json(const fs::path& p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("dflash2: cannot read " + p.string());
    auto j = json::parse(f, nullptr, false);
    if (j.is_discarded()) throw std::runtime_error("dflash2: bad JSON in " + p.string());
    return j;
}

std::vector<uint8_t> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("dflash2: cannot read " + p.string());
    std::vector<uint8_t> v(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size()));
    return v;
}

PackOp parse_op(const json& j) {
    PackOp p;
    p.op = j.at("op").get<std::string>();
    p.tensor = j.value("tensor", "");
    p.dst = j.value("dst", 0ull);
    p.nch = j.value("nch", 0ull);
    p.in_dim = j.value("in_dim", 0ull);
    p.chunk0 = j.value("chunk0", 0ull);
    p.chunk_bytes = j.value("chunk_bytes", 0ull);
    return p;
}

void rms_rows(const float* x, size_t T, size_t d, const std::vector<float>& w, double eps, float* out) {
    host::rmsnorm_rows(x, T, d, w.data(), eps, out);
}

float silu(float v) { return v / (1.0f + std::exp(-v)); }

}  // namespace

Drafter::Drafter(const std::string& model_dir, const std::string& kernel_dir, Core& target, bool verbose)
    : target_(target), verbose_(verbose), model_dir_(model_dir), kernel_dir_(kernel_dir) {
    const json c = read_json(fs::path(model_dir) / "config.json");
    if (c.value("model_type", "") != "dflash2")
        throw std::runtime_error("dflash2: " + model_dir + " is not a DFlash 2 drafter container (model_type)");
    cfg_.hidden = c.at("hidden_size");
    cfg_.layers = c.at("num_hidden_layers");
    cfg_.nh = c.at("num_attention_heads");
    cfg_.kvh = c.at("num_key_value_heads");
    cfg_.hd = c.at("head_dim");
    cfg_.ff = c.at("intermediate_size");
    cfg_.vocab = c.at("vocab_size");
    cfg_.rank = c.at("selector_rank");
    cfg_.group = c.at("conv_group_size");
    cfg_.window = c.value("sliding_window", 0);
    cfg_.eps = c.at("rms_norm_eps");
    cfg_.theta = c.at("rope_theta");
    cfg_.mask_id = c.at("mask_token_id");
    cfg_.tap_layers = c.at("target_layer_ids").get<std::vector<int>>();
    cfg_.n_conv = 2 * static_cast<size_t>(c.at("conv_kernel_size")) * cfg_.hidden / cfg_.group;
    if (c.at("conv_kernel_size") != 2 || static_cast<size_t>(c.at("selector_top_k")) != kTopK)
        throw std::runtime_error("dflash2: the engine runs a 2-tap conv and a top-16 selector");
    if (cfg_.hidden != target.manifest().hidden || cfg_.vocab != target.vocab())
        throw std::runtime_error("dflash2: the drafter's hidden / vocab do not match the target's");

    man_ = read_json(fs::path(kernel_dir) / "manifest.json");
    if (man_.value("family", "") != "dflash2")
        throw std::runtime_error("dflash2: " + kernel_dir + " is not a dflash2 kernel set");
    const json& spec = man_.at("spec");
    if (spec.at("hidden") != cfg_.hidden || spec.at("intermediate") != cfg_.ff || spec.at("num_heads") != cfg_.nh ||
        spec.at("num_kv_heads") != cfg_.kvh || spec.at("head_dim") != cfg_.hd)
        throw std::runtime_error("dflash2: the kernel set was built for another drafter geometry");

    xrt::xclbin xcl((fs::path(kernel_dir) / man_.at("contexts").at("draft").get<std::string>()).string());
    auto uuid = target_.device().register_xclbin(xcl);
    ctx_ = std::make_unique<xrt::hw_context>(target_.device(), uuid);
    for (const auto& [op, g] : man_.at("layout").at("draft").at("gemv").items()) load_gemv(op, g);
    file_ = std::make_unique<Q4nxFile>((fs::path(model_dir) / "model.q4nx").string());
    max_rows_ = (cfg_.window ? cfg_.window : 8192) + 256;
    if (verbose_)
        std::fprintf(stderr, "dflash2: drafter %s (%zu layers, taps", fs::path(model_dir).filename().string().c_str(),
                     cfg_.layers);
    if (verbose_) {
        for (int t : cfg_.tap_layers) std::fprintf(stderr, " %d", t);
        std::fprintf(stderr, "), %zu GEMV streams from %s\n", gemv_.size(), kernel_dir.c_str());
    }
}

Drafter::~Drafter() = default;

void Drafter::load_gemv(const std::string& op, const json& g) {
    Gemv& v = gemv_[op];
    v.plan.n = g.at("n");
    v.plan.k = g.at("k");
    v.plan.tab_bytes = g.at("tab_bytes");
    v.plan.n_tabs = g.at("n_tabs");
    v.plan.y_floats = g.at("y_floats");
    if (g.at("tokens") != kRows) throw std::runtime_error("dflash2: " + op + " was built for another row count");
    int core = 0;
    for (const auto& items : g.at("items")) {
        for (const auto& it : items) {
            GemvItem x;
            x.core = core;
            x.slice = it.at("slice");
            x.b0 = it.at("b0");
            x.nb = it.at("nb");
            x.kt0 = it.at("kt0");
            x.nkt = it.at("nkt");
            x.tab = it.at("tab");
            x.y = it.at("y");
            v.plan.items.push_back(x);
        }
        ++core;
    }
    v.plan.index_bands();
    const std::string name = g.at("kernel");
    const json& kd = man_.at("kernels").at(name);
    auto insts = read_file(fs::path(kernel_dir_) / kd.at("insts").get<std::string>());
    if (insts.empty() || insts.size() % 4) throw std::runtime_error("dflash2: " + name + " insts is not word-sized");
    v.k = std::make_unique<xrt::kernel>(*ctx_, "MLIR_AIE");
    v.words = static_cast<uint32_t>(insts.size() / 4);
    v.instr = std::make_unique<xrt::bo>(target_.device(), insts.size(), xrt::bo::flags::cacheable, v.k->group_id(1));
    std::memcpy(v.instr->map<void*>(), insts.data(), insts.size());
    v.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    v.tab = xrt::ext::bo(target_.device(), padup(v.plan.n_tabs * v.plan.tab_bytes));
    v.y = xrt::ext::bo(target_.device(), padup(v.plan.y_floats * 4));
    v.rec.resize(v.plan.k / 256 * 4 * kRec);
    v.xr.resize(kRows * v.plan.k);
}

xrt::bo Drafter::pack(const json& spec, int layer) {
    const size_t bytes = spec.at("bytes");
    xrt::bo bo = xrt::ext::bo(target_.device(), padup(bytes));
    auto* m = bo.map<uint8_t*>();
    std::memset(m, 0, padup(bytes));
    for (const auto& o : spec.at("pack")) pools::apply(parse_op(o), *file_, layer, m, bytes, 5120);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return bo;
}

void Drafter::load_weights() {
    const auto t0 = std::chrono::steady_clock::now();
    const json& pk = man_.at("pack");
    for (const auto& [name, spec] : pk.at("weights").items()) w_.emplace(name, pack(spec, 0));
    lw_.assign(cfg_.layers, {});
    for (size_t l = 0; l < cfg_.layers; ++l)
        for (const auto& [name, spec] : pk.at("layer_weights").items()) lw_[l].emplace(name, pack(spec, static_cast<int>(l)));
    auto f32 = [&](const std::string& n) { return file_->f32(n); };
    lc_.assign(cfg_.layers, {});
    for (size_t l = 0; l < cfg_.layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        lc_[l] = {f32(p + "attn_norm.weight"), f32(p + "ffn_norm.weight"), f32(p + "attn_q_norm.weight"),
                  f32(p + "attn_k_norm.weight"), f32(p + "attn_conv_base"), f32(p + "ffn_conv_base")};
    }
    enc_norm_ = f32("enc.output_norm.weight");
    out_norm_ = f32("output_norm.weight");
    sel_hidden_ = f32("selector_hidden.weight");
    auto bf = [&](const std::string& n, std::vector<uint16_t>& dst) {
        size_t nb = 0;
        const uint8_t* r = file_->raw(n, &nb);
        dst.resize(nb / 2);
        std::memcpy(dst.data(), r, nb);
    };
    bf("selector_predecessor.weight", sel_prev_);
    bf("selector_successor.weight", sel_next_);
    if (sel_prev_.size() != cfg_.vocab * cfg_.rank || sel_next_.size() != cfg_.vocab * cfg_.rank ||
        sel_hidden_.size() != cfg_.rank * cfg_.hidden)
        throw std::runtime_error("dflash2: selector tables of the wrong size");
    reset();
    if (verbose_) std::fprintf(stderr, "dflash2: weights packed in %.1f s\n", ms_since(t0) / 1000.0);
}

void Drafter::reset() {
    const size_t kvw = cfg_.kvh * cfg_.hd;
    kc_.assign(cfg_.layers, std::vector<float>(max_rows_ * kvw, 0.f));
    vc_.assign(cfg_.layers, std::vector<float>(max_rows_ * kvw, 0.f));
    base_pos_ = next_pos_ = 0;
}

void Drafter::truncate(long pos) {
    if (pos < base_pos_) throw std::runtime_error("dflash2: cannot truncate below the cache window");
    next_pos_ = std::min(next_pos_, pos);
}

void Drafter::write_kv(int layer, long pos, const float* k, const float* v) {
    const size_t kvw = cfg_.kvh * cfg_.hd;
    std::memcpy(&kc_[layer][static_cast<size_t>(pos - base_pos_) * kvw], k, kvw * sizeof(float));
    std::memcpy(&vc_[layer][static_cast<size_t>(pos - base_pos_) * kvw], v, kvw * sizeof(float));
}

void Drafter::run_gemv(const std::string& op, xrt::bo& w, const float* x, float* out) {
    Gemv& g = gemv_.at(op);
    const auto t0 = std::chrono::steady_clock::now();
    hadamard8(x, g.plan.k, g.xr.data());
    build_records(g.xr.data(), g.plan.k, g.rec.data());
    build_tables(g.plan, g.rec.data(), g.tab.map<uint8_t*>());
    g.tab.sync(XCL_BO_SYNC_BO_TO_DEVICE, g.plan.n_tabs * g.plan.tab_bytes, 0);
    const auto t1 = std::chrono::steady_clock::now();
    xrt::run r(*g.k);
    r.set_arg(0, kOpcode);
    r.set_arg(1, *g.instr);
    r.set_arg(2, static_cast<int>(g.words));
    r.set_arg(3, w);
    r.set_arg(4, g.tab);
    r.set_arg(5, g.y);
    r.start();
    const auto st = r.wait(std::chrono::milliseconds(20000));
    if (st != ERT_CMD_STATE_COMPLETED)
        throw std::runtime_error("dflash2: gemv " + op + " did not complete (ERT state " + std::to_string(static_cast<int>(st)) + ")");
    const double run_ms = ms_since(t1);
    const auto t2 = std::chrono::steady_clock::now();
    g.y.sync(XCL_BO_SYNC_BO_FROM_DEVICE, g.plan.y_floats * 4, 0);
    _mm_mfence();
    const auto t3 = std::chrono::steady_clock::now();
    reduce_y(g.plan, g.y.map<const float*>(), out);
    const auto t4 = std::chrono::steady_clock::now();
    ++timing_.dispatches;
    timing_.gemv_ms += run_ms;
    timing_.prep_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    timing_.sync_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();
    timing_.reduce_ms += std::chrono::duration<double, std::milli>(t4 - t3).count();
    timing_.host_ms += ms_since(t0) - run_ms;
    if (op == "head") timing_.head_ms += run_ms;
}

void Drafter::inject(const float* taps, size_t n) {
    const size_t H = cfg_.hidden, tw = cfg_.tap_width(), kvw = cfg_.kvh * cfg_.hd, L = cfg_.layers;
    std::vector<float> x(kRows * tw), gc(kRows * H), g(kRows * H), kv(kRows * L * 2 * kvw), kn(kvw);
    for (size_t r0 = 0; r0 < n; r0 += kRows) {
        const size_t m = std::min(kRows, n - r0);
        std::fill(x.begin(), x.end(), 0.f);
        std::memcpy(x.data(), taps + r0 * tw, m * tw * sizeof(float));
        run_gemv("fc", w_.at("fc"), x.data(), gc.data());
        rms_rows(gc.data(), kRows, H, enc_norm_, cfg_.eps, g.data());
        run_gemv("kv", w_.at("kv"), g.data(), kv.data());
        for (size_t r = 0; r < m; ++r) {
            const long pos = next_pos_;
            if (static_cast<size_t>(pos - base_pos_) >= max_rows_) {
                // slide: keep the newest window rows at the front
                const size_t keep = cfg_.window ? cfg_.window : max_rows_ / 2;
                const size_t drop = static_cast<size_t>(pos - base_pos_) - keep;
                for (size_t l = 0; l < L; ++l) {
                    std::memmove(kc_[l].data(), kc_[l].data() + drop * kvw, keep * kvw * sizeof(float));
                    std::memmove(vc_[l].data(), vc_[l].data() + drop * kvw, keep * kvw * sizeof(float));
                }
                base_pos_ += static_cast<long>(drop);
            }
            for (size_t l = 0; l < L; ++l) {
                const float* kr = &kv[r * L * 2 * kvw + l * 2 * kvw];
                const float* vr = kr + kvw;
                rms_rows(kr, cfg_.kvh, cfg_.hd, lc_[l].k_norm, cfg_.eps, kn.data());
                rope_neox(kn.data(), cfg_.kvh, cfg_.hd, static_cast<double>(pos), cfg_.theta);
                std::vector<float> vv(vr, vr + kvw);
                for (auto& e : kn) e = round_f16(e);
                for (auto& e : vv) e = round_f16(e);
                write_kv(static_cast<int>(l), pos, kn.data(), vv.data());
            }
            ++next_pos_;
        }
    }
}

std::vector<int> Drafter::draft(int anchor, size_t k, Lattice* lat_out) {
    if (k == 0 || k >= kRows) throw std::runtime_error("dflash2: a draft is 1..7 tokens");
    timing_ = {};
    const auto t0 = std::chrono::steady_clock::now();
    const size_t H = cfg_.hidden, R = kRows, nh = cfg_.nh, kvh = cfg_.kvh, hd = cfg_.hd, ff = cfg_.ff;
    const size_t qw = nh * hd, kvw = kvh * hd, nconv = cfg_.n_conv;
    const long p0 = next_pos_;
    std::vector<float> x(R * H), h(R * H), dyn(R * nconv), qkv(R * (qw + 2 * kvw)), q(R * qw), kb(R * kvw),
        vb(R * kvw), att(R * qw), o(R * H), x2(R * H), gu(R * 2 * ff), a(R * ff), tmp(R * qw);
    const std::string& emb = target_.manifest().embed_tensor;
    for (size_t t = 0; t < R; ++t)
        target_.file().bf16_row(emb, static_cast<size_t>(t == 0 ? anchor : cfg_.mask_id), H, &x[t * H]);
    for (size_t l = 0; l < cfg_.layers; ++l) {
        const LayerConsts& c = lc_[l];
        auto& lw = lw_[l];
        rms_rows(x.data(), R, H, c.attn_norm, cfg_.eps, h.data());
        run_gemv("conv", lw.at("conv_attn"), h.data(), dyn.data());
        conv2(h.data(), R, H, dyn.data(), c.attn_cb.data(), 0, cfg_.group, h.data());
        run_gemv("qkv", lw.at("qkv"), h.data(), qkv.data());
        const auto ta = std::chrono::steady_clock::now();
        for (size_t t = 0; t < R; ++t) {
            const float* row = &qkv[t * (qw + 2 * kvw)];
            std::memcpy(&tmp[t * qw], row, qw * sizeof(float));
            std::memcpy(&kb[t * kvw], row + qw, kvw * sizeof(float));
            std::memcpy(&vb[t * kvw], row + qw + kvw, kvw * sizeof(float));
        }
        rms_rows(tmp.data(), R * nh, hd, c.q_norm, cfg_.eps, q.data());
        rms_rows(kb.data(), R * kvh, hd, c.k_norm, cfg_.eps, tmp.data());
        std::memcpy(kb.data(), tmp.data(), R * kvw * sizeof(float));
        for (size_t t = 0; t < R; ++t) {
            rope_neox(&q[t * qw], nh, hd, static_cast<double>(p0 + static_cast<long>(t)), cfg_.theta);
            rope_neox(&kb[t * kvw], kvh, hd, static_cast<double>(p0 + static_cast<long>(t)), cfg_.theta);
        }
        for (auto& e : kb) e = round_f16(e);
        for (auto& e : vb) e = round_f16(e);
        attention(q.data(), R, nh, kvh, hd, kc_[l].data(), vc_[l].data(), static_cast<size_t>(next_pos_ - base_pos_),
                  base_pos_, kb.data(), vb.data(), p0, cfg_.window, att.data());
        timing_.attn_ms += ms_since(ta);
        run_gemv("o", lw.at("o"), att.data(), o.data());
        conv2(o.data(), R, H, dyn.data(), c.attn_cb.data(), 1, cfg_.group, o.data());
        for (size_t i = 0; i < R * H; ++i) x2[i] = o[i] + x[i];
        rms_rows(x2.data(), R, H, c.ffn_norm, cfg_.eps, h.data());
        run_gemv("conv", lw.at("conv_ffn"), h.data(), dyn.data());
        conv2(h.data(), R, H, dyn.data(), c.ffn_cb.data(), 0, cfg_.group, h.data());
        run_gemv("gu", lw.at("gu"), h.data(), gu.data());
        for (size_t t = 0; t < R; ++t) silu_mul(&gu[t * 2 * ff], &gu[t * 2 * ff + ff], ff, &a[t * ff]);
        run_gemv("down", lw.at("down"), a.data(), o.data());
        conv2(o.data(), R, H, dyn.data(), c.ffn_cb.data(), 1, cfg_.group, o.data());
        for (size_t i = 0; i < R * H; ++i) x[i] = o[i] + x2[i];
    }
    rms_rows(x.data(), R, H, out_norm_, cfg_.eps, h.data());
    std::vector<float> logits(R * cfg_.vocab);
    run_gemv("head", target_.global_bo("lmpool"), h.data(), logits.data());
    const auto ts = std::chrono::steady_clock::now();
    Lattice lat;
    build_lattice(anchor, R - 1, &logits[cfg_.vocab], cfg_.vocab, &h[H], H, sel_hidden_.data(), cfg_.rank,
                  sel_prev_.data(), sel_next_.data(), lat);
    std::vector<int> d = walk(lat);
    timing_.select_ms = ms_since(ts);
    d.resize(k);
    if (lat_out) *lat_out = std::move(lat);
    timing_.total_ms = ms_since(t0);
    return d;
}

}  // namespace dflash2
}  // namespace open_qwen36
