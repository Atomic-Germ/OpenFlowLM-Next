/// \file manifest.cpp
/// \brief manifest.json parsing and the model check (see manifest.hpp).
#include "open_qwen36/manifest.hpp"

#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <utility>

namespace open_qwen36 {

using nlohmann::json;

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& what) {
    throw std::runtime_error("open_qwen36: " + where + ": " + what);
}

const json& need(const json& j, const char* key, const std::string& where) {
    if (!j.is_object() || !j.contains(key)) fail(where, std::string("lacks '") + key + "'");
    return j[key];
}

template <class T>
T get(const json& j, const char* key, const std::string& where) {
    try {
        return need(j, key, where).get<T>();
    } catch (const json::exception& e) {
        fail(where, std::string("'") + key + "': " + e.what());
    }
}

PackOp parse_op(const json& j, const std::string& where) {
    PackOp p;
    p.op = get<std::string>(j, "op", where);
    p.tensor = j.value("tensor", "");
    p.up = j.value("up", "");
    p.gate = j.value("gate", "");
    p.dst = j.value("dst", 0ull);
    p.cap = j.value("cap", 0ull);
    p.nch = j.value("nch", 0ull);
    p.in_dim = j.value("in_dim", 0ull);
    p.chunk0 = j.value("chunk0", 0ull);
    p.experts = j.value("experts", 0ull);
    p.stripes = j.value("stripes", 0ull);
    p.stripe_bytes = j.value("stripe_bytes", 0ull);
    p.expert_bytes = j.value("expert_bytes", 0ull);
    p.taps = j.value("taps", 0ull);
    p.groups = j.value("groups", 0ull);
    p.width = j.value("width", 0ull);
    p.chunk_bytes = j.value("chunk_bytes", 0ull);
    p.rows = j.value("rows", 0ull);
    p.cols = j.value("cols", 0ull);
    p.elem = j.value("elem", 0ull);
    p.dst_rows = j.value("dst_rows", 0ull);
    // The same fields pools::apply needs, checked here so a bad manifest is named
    // at load rather than surfacing as a "pools:" error part-way through packing.
    auto need_all = [&](std::initializer_list<std::pair<const char*, uint64_t>> fs) {
        for (const auto& [name, v] : fs)
            if (v == 0) fail(where, p.op + " " + (p.tensor.empty() ? p.up : p.tensor) + " without " + name);
    };
    if (p.op == "std_perm" || p.op == "q8_perm" || p.op == "put" || p.op == "expert_down" ||
        p.op == "conv_transpose" || p.op == "lmhead_q8" || p.op == "transpose") {
        if (p.tensor.empty()) fail(where, p.op + " without a tensor");
    } else if (p.op == "expert_stripes") {
        if (p.up.empty() || p.gate.empty()) fail(where, "expert_stripes without up / gate");
    } else {
        fail(where, "unknown pack op '" + p.op + "'");
    }
    if (p.op == "std_perm" || p.op == "q8_perm") need_all({{"nch", p.nch}, {"in_dim", p.in_dim}});
    else if (p.op == "transpose") need_all({{"rows", p.rows}, {"cols", p.cols}, {"elem", p.elem}});
    else if (p.op == "expert_stripes") need_all({{"stripe_bytes", p.stripe_bytes}, {"stripes", p.stripes}, {"experts", p.experts}, {"in_dim", p.in_dim}});
    else if (p.op == "expert_down") need_all({{"expert_bytes", p.expert_bytes}, {"experts", p.experts}});
    else if (p.op == "put") need_all({{"cap", p.cap}});
    else if (p.op == "lmhead_q8") need_all({{"chunk_bytes", p.chunk_bytes}, {"in_dim", p.in_dim}});
    else if (p.op == "conv_transpose") need_all({{"taps", p.taps}, {"groups", p.groups}, {"width", p.width}});
    return p;
}

std::vector<Step> parse_program(const json& j, const std::string& where) {
    std::vector<Step> out;
    if (!j.is_array()) fail(where, "program is not a list");
    for (const auto& s : j) {
        Step st;
        st.op = get<std::string>(s, "op", where);
        st.kernel = get<std::string>(s, "kernel", where);
        if (st.op == "run") {
            st.args = get<std::vector<std::string>>(s, "args", where);
            if (st.args.empty() || st.args.size() > 8) fail(where, "run " + st.kernel + ": " + std::to_string(st.args.size()) + " buffer args (1..8)");
        } else if (st.op == "moeroute2") {
            st.act_off = get<uint64_t>(s, "act_off", where);
        } else {
            fail(where, "unknown program op '" + st.op + "'");
        }
        out.push_back(std::move(st));
    }
    return out;
}

/// A program step names a kernel this manifest declares, and a moeroute2 step names
/// one built with the routed-expert patch table (Core::route needs it).
void check_step(const Manifest& m, const Step& s, const std::string& where, const char* what) {
    auto it = m.kernels.find(s.kernel);
    if (it == m.kernels.end()) fail(where, std::string(what) + " names unknown kernel " + s.kernel);
    if (s.op == "moeroute2" && it->second.patch != "moeroute2")
        fail(where, std::string(what) + ": moeroute2 on kernel " + s.kernel + ", whose patch is '" +
                        (it->second.patch.empty() ? std::string("none") : it->second.patch) + "'");
}

}  // namespace

Manifest Manifest::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("open_qwen36: no manifest at " + path);
    json j = json::parse(f, nullptr, false);
    if (j.is_discarded()) throw std::runtime_error("open_qwen36: " + path + " is not JSON");
    return parse(j, path);
}

Manifest Manifest::parse(const json& j, const std::string& where) {
    Manifest m;
    m.version = get<int>(j, "manifest_version", where);
    if (m.version != 1) fail(where, "manifest_version " + std::to_string(m.version) + " (this engine reads 1)");
    m.family = get<std::string>(j, "family", where);
    m.spec_hash = j.value("spec_hash", "");
    m.build_key = j.value("build_key", "");
    m.max_ctx_default = j.value("max_ctx_default", 4096ull);

    const json& lay = need(j, "layout", where);
    const std::string lw = where + " layout";
    m.hidden = get<size_t>(lay, "hidden", lw);
    m.vocab = get<size_t>(lay, "vocab", lw);
    m.real_vocab = lay.value("real_vocab", m.vocab);
    m.chunk_bytes = get<size_t>(lay, "chunk_bytes", lw);
    m.pool_bytes = get<size_t>(lay, "pool_bytes", lw);
    m.lmhead_pool_bytes = get<size_t>(lay, "lmhead_pool_bytes", lw);
    m.lmhead_chunk_bytes = lay.value("lmhead_chunk_bytes", 0ull);
    m.kv_row = get<size_t>(lay, "kv_row", lw);
    m.ptab_row = get<size_t>(lay, "ptab_row", lw);
    m.rotary_dim = get<size_t>(lay, "rotary_dim", lw);
    m.rope_theta = get<double>(lay, "rope_theta", lw);
    m.rope_inv_freq = get<std::vector<double>>(lay, "rope_inv_freq", lw);
    if (m.rope_inv_freq.size() != m.rotary_dim / 2) fail(lw, "rope_inv_freq has " + std::to_string(m.rope_inv_freq.size()) + " values, the rotary dim wants " + std::to_string(m.rotary_dim / 2));
    m.rout_idx_off = lay.value("rout_idx_off", 1024ull);
    m.has_moe = lay.contains("moe");
    if (m.has_moe) {
    const json& moe = lay["moe"];
    const std::string mw = lw + ".moe";
    m.moe.experts = get<unsigned>(moe, "experts", mw);
    m.moe.topk = get<unsigned>(moe, "topk", mw);
    m.moe.stripe = get<uint64_t>(moe, "stripe", mw);
    m.moe.up_bytes = get<uint64_t>(moe, "up_bytes", mw);
    m.moe.down_core = get<uint64_t>(moe, "down_core", mw);
    m.moe.pool_down = get<uint64_t>(moe, "pool_down", mw);
    m.moe.share_up = get<uint64_t>(moe, "share_up", mw);
    m.moe.share_gate = get<uint64_t>(moe, "share_gate", mw);
    m.moe.share_down = get<uint64_t>(moe, "share_down", mw);
    if (m.moe.topk > 8 || m.moe.topk == 0) fail(mw, "topk " + std::to_string(m.moe.topk) + " (the router record holds 8)");
    }
    m.attn.kv_row = m.kv_row;
    m.attn.ptab_row = m.ptab_row;

    m.layers = get<std::vector<std::string>>(j, "layers", where);
    if (m.layers.empty()) fail(where, "no layers");
    for (const auto& [k, v] : need(j, "contexts", where).items()) m.contexts[k] = v.get<std::string>();
    for (const auto& [k, v] : need(j, "kernels", where).items()) {
        KernelDesc d;
        d.context = get<std::string>(v, "context", where + " kernel " + k);
        d.insts = get<std::string>(v, "insts", where + " kernel " + k);
        d.patch = v.value("patch", "");
        d.window = v.value("window", 0ull);
        if (!m.contexts.count(d.context)) fail(where, "kernel " + k + " names unknown context " + d.context);
        if (!d.patch.empty() && d.patch != "moeroute2" && d.patch != "attnpos" && d.patch != "moebatch")
            fail(where, "kernel " + k + ": unknown patch " + d.patch);
        if ((d.patch == "moeroute2" || d.patch == "moebatch") && !m.has_moe)
            fail(where, "kernel " + k + " wants " + d.patch + " but layout.moe is absent");
        m.kernels[k] = d;
    }
    for (const auto& [name, v] : need(j, "layer_types", where).items()) {
        const std::string tw = where + " layer type " + name;
        LayerType t;
        t.name = name;
        const json& b = need(v, "buffers", tw);
        t.consts_bytes = get<uint64_t>(b, "consts", tw);
        t.act_bytes = get<uint64_t>(b, "act", tw);
        const json& st = need(b, "state", tw);
        t.state_kind = get<std::string>(st, "kind", tw);
        if (t.state_kind == "linear") t.state_bytes = get<uint64_t>(st, "bytes", tw);
        else if (t.state_kind == "kv") t.state_row = get<uint64_t>(st, "row", tw);
        else fail(tw, "unknown state kind " + t.state_kind);
        t.program = parse_program(need(v, "program", tw), tw);
        for (const auto& s : t.program) check_step(m, s, tw, "program");
        // the block prefill route, independent of the sequential `program`
        // above (manifest.hpp's GemmBlockProgram): a per-kind fixed step list
        if (v.contains("gemm_block")) {
            const json& gj = v["gemm_block"];
            const std::string gw = tw + " gemm_block";
            GemmBlockProgram& g = t.gemm_block;
            g.t = get<uint64_t>(gj, "t", gw);
            if (g.t == 0) fail(gw, "t must be > 0 when gemm_block is present");
            g.kind = gj.value("kind", "dense");
            g.eps = get<double>(gj, "eps", gw);
            g.program = parse_program(need(gj, "program", gw), gw + ".program");
            for (const auto& s : g.program) {
                check_step(m, s, gw, "gemm_block.program");
                if (s.op != "run" || s.args.size() != 3)
                    fail(gw, "gemm_block.program step " + s.kernel + " must be a run with exactly 3 args (weight, x, y), has " +
                                 std::to_string(s.args.size()));
            }
            auto parse_weights = [&](const char* key, std::map<std::string, GemmWeight>& into) {
                if (!gj.contains(key)) return;
                for (const auto& [name, wj] : gj[key].items()) {
                    GemmWeight w;
                    w.from = get<std::string>(wj, "from", gw + "." + key + "." + name);
                    if (w.from != "pool" && w.from != "consts") fail(gw, "weight " + name + ": from must be pool or consts");
                    w.ops = get<std::vector<size_t>>(wj, "ops", gw + "." + key + "." + name);
                    if (w.ops.empty()) fail(gw, "weight " + name + " names no pack ops");
                    into[name] = w;
                }
            };
            parse_weights("weights", g.weights);
            if (g.kind == "dense") {
                if (g.program.size() != 5)
                    fail(gw, "a dense route has exactly 5 steps (qkv3, o, gate, up, down), has " + std::to_string(g.program.size()));
                g.qw = get<uint64_t>(gj, "qw", gw);
                g.kvw = get<uint64_t>(gj, "kvw", gw);
                g.ff = get<uint64_t>(gj, "ff", gw);
                g.ad_q = get<uint64_t>(gj, "ad_q", gw);
                g.ad_kvn = get<uint64_t>(gj, "ad_kvn", gw);
                g.ad_og = get<uint64_t>(gj, "ad_og", gw);
                if (!m.kernels.count("dxB")) fail(gw, "gemm_block present but this manifest declares no dxB kernel");
                // #39's hand-built sets carry no weights map: the dense recipe's pack order is q k v o up gate down
                if (g.weights.empty())
                    g.weights = {{"gqkv3_w", {"pool", {0, 1, 2}}}, {"go_w", {"pool", {3}}}, {"gup_w", {"pool", {4}}},
                                 {"ggate_w", {"pool", {5}}}, {"gdown_w", {"pool", {6}}}};
            } else if (g.kind == "linear" || g.kind == "full") {
                if (!m.has_moe) fail(gw, "a " + g.kind + " route needs layout.moe (the per-token MoE tail)");
                if (g.program.size() != 2)
                    fail(gw, "a " + g.kind + " route has exactly 2 steps (the fused input projection, the output projection), has " +
                                 std::to_string(g.program.size()));
                g.a_xm = get<uint64_t>(gj, "a_xm", gw);
                g.a_rout = get<uint64_t>(gj, "a_rout", gw);
                g.a_res = get<uint64_t>(gj, "a_res", gw);
                g.moe_kernel = get<std::string>(gj, "moe_kernel", gw);
                g.moe_args = get<std::vector<std::string>>(gj, "moe_args", gw);
                auto mk = m.kernels.find(g.moe_kernel);
                if (mk == m.kernels.end()) fail(gw, "moe_kernel names unknown kernel " + g.moe_kernel);
                if (mk->second.patch != "moeroute2") fail(gw, "moe_kernel " + g.moe_kernel + " is not built with the moeroute2 patch table");
                if (g.moe_args.empty()) fail(gw, "moe_args is empty");
                // the shared expert over the whole block: up|gate then down. mx is built
                // routed-only, so a MoE route without these two steps has no shared expert
                // at all -- required, not optional.
                g.shared_ff = get<uint64_t>(gj, "shared_ff", gw);
                g.shared_program = parse_program(need(gj, "shared_program", gw), gw + ".shared_program");
                if (g.shared_program.size() != 2)
                    fail(gw, "shared_program has exactly 2 steps (up|gate, down), has " + std::to_string(g.shared_program.size()));
                for (const auto& s : g.shared_program) {
                    check_step(m, s, gw, "gemm_block.shared_program");
                    if (s.op != "run" || s.args.size() != 3)
                        fail(gw, "gemm_block.shared_program step " + s.kernel + " must be a run with exactly 3 args, has " +
                                     std::to_string(s.args.size()));
                }
                parse_weights("shared_weights", g.shared_weights);
                // the token-batched expert kernel: optional (an older set runs mx per token)
                if (gj.contains("moe_batch")) {
                    const json& bj = gj["moe_batch"];
                    const std::string bw = gw + ".moe_batch";
                    MoeBatch& b = g.moe_batch;
                    b.nt = get<uint64_t>(bj, "nt", bw);
                    b.args = get<std::vector<std::string>>(bj, "args", bw);
                    if (b.nt == 0 || b.args.size() != 4 || b.args[0] != "pool")
                        fail(bw, "wants nt > 0 and four args (pool, x, h, y)");
                    for (const auto& [slots_s, kname] : need(bj, "kernels", bw).items()) {
                        const size_t slots = static_cast<size_t>(std::stoull(slots_s));
                        if (slots == 0 || slots % 8) fail(bw, "slot count " + slots_s + " is not a positive multiple of 8");
                        auto it = m.kernels.find(kname.get<std::string>());
                        if (it == m.kernels.end()) fail(bw, "names unknown kernel " + kname.get<std::string>());
                        if (it->second.patch != "moebatch") fail(bw, "kernel " + it->first + " is not built with the moebatch patch table");
                        b.kernels[slots] = it->first;
                    }
                    if (b.kernels.empty()) fail(bw, "names no streams");
                }
                // the attention products on the NPU: optional, full attention only
                if (g.kind == "full" && gj.contains("attn_block")) {
                    const json& aj = gj["attn_block"];
                    const std::string aw = gw + ".attn_block";
                    AttnBlock& a = g.attn_block;
                    a.m = get<uint64_t>(aj, "m", aw);
                    a.hd = get<uint64_t>(aj, "hd", aw);
                    a.l_max = get<uint64_t>(aj, "l_max", aw);
                    a.args = get<std::vector<std::string>>(aj, "args", aw);
                    if (a.m == 0 || a.m % 256 || a.hd == 0 || a.hd % 256 || a.l_max == 0 || a.l_max % 256)
                        fail(aw, "wants m, hd and l_max as positive multiples of 256");
                    if (a.args.size() != 3) fail(aw, "wants three args (a, b, c)");
                    auto streams = [&](const char* key, std::map<size_t, std::string>& into) {
                        for (const auto& [rows_s, kname] : need(aj, key, aw).items()) {
                            const size_t rows = static_cast<size_t>(std::stoull(rows_s));
                            if (rows == 0 || rows % 256 || rows > a.l_max)
                                fail(aw, std::string(key) + ": window " + rows_s + " is not a positive multiple of 256 within l_max");
                            auto it = m.kernels.find(kname.get<std::string>());
                            if (it == m.kernels.end()) fail(aw, std::string(key) + " names unknown kernel " + kname.get<std::string>());
                            into[rows] = it->first;
                        }
                    };
                    streams("kernels_s", a.kernels_s);
                    streams("kernels_pv", a.kernels_pv);
                    if (a.kernels_s.empty() || !a.kernels_s.count(a.l_max)) fail(aw, "kernels_s must reach l_max");
                    std::vector<size_t> ks, kpv;
                    for (const auto& kv : a.kernels_s) ks.push_back(kv.first);
                    for (const auto& kv : a.kernels_pv) kpv.push_back(kv.first);
                    if (ks != kpv) fail(aw, "kernels_s and kernels_pv cover different windows");
                }
                if (g.kind == "linear") {
                    g.qkv_dim = get<uint64_t>(gj, "qkv_dim", gw);
                    g.vw = get<uint64_t>(gj, "vw", gw);
                    g.key_heads = get<uint64_t>(gj, "key_heads", gw);
                    g.value_heads = get<uint64_t>(gj, "value_heads", gw);
                    g.head_dim = get<uint64_t>(gj, "head_dim", gw);
                    g.conv_kernel = get<uint64_t>(gj, "conv_kernel", gw);
                    g.state_s_off = get<uint64_t>(gj, "state_s_off", gw);
                    g.s_head_bytes = get<uint64_t>(gj, "s_head_bytes", gw);
                    g.s_rows = get<uint64_t>(gj, "s_rows", gw);
                    if (t.state_kind != "linear") fail(gw, "a linear route on a layer type whose state is not linear");
                    if (g.qkv_dim != 2 * g.key_heads * g.head_dim + g.value_heads * g.head_dim || g.vw != g.value_heads * g.head_dim)
                        fail(gw, "qkv_dim / vw disagree with the head counts");
                } else {
                    g.qw = get<uint64_t>(gj, "qw", gw);
                    g.kvw = get<uint64_t>(gj, "kvw", gw);
                    g.nh = get<uint64_t>(gj, "nh", gw);
                    g.kvh = get<uint64_t>(gj, "kvh", gw);
                    g.hd = get<uint64_t>(gj, "hd", gw);
                    g.rot = get<uint64_t>(gj, "rot", gw);
                    if (t.state_kind != "kv") fail(gw, "a full route on a layer type whose state is not kv");
                    if (g.qw != g.nh * g.hd || g.kvw != g.kvh * g.hd || g.rot > g.hd || g.rot != m.rotary_dim)
                        fail(gw, "qw / kvw / rot disagree with the heads and the layout rotary dim");
                }
            } else {
                fail(gw, "unknown kind " + g.kind + " (dense | linear | full)");
            }
            for (const auto& s : g.program)
                if (!g.weights.count(s.args[0]))
                    fail(gw, "step " + s.kernel + " reads weight buffer " + s.args[0] + ", which weights does not define");
            for (const auto& s : g.shared_program)
                if (!g.shared_weights.count(s.args[0]))
                    fail(gw, "shared step " + s.kernel + " reads weight buffer " + s.args[0] + ", which shared_weights does not define");
        }
        const json& pk = need(v, "pack", tw);
        for (const auto& o : need(pk, "pool", tw)) t.pool.push_back(parse_op(o, tw + " pack.pool"));
        for (const auto& o : need(pk, "consts", tw)) t.consts.push_back(parse_op(o, tw + " pack.consts"));
        for (const auto* ws : {&t.gemm_block.weights, &t.gemm_block.shared_weights})
            for (const auto& [name, w] : *ws) {
                const size_t n = w.from == "pool" ? t.pool.size() : t.consts.size();
                for (size_t idx : w.ops)
                    if (idx >= n) fail(tw, "gemm_block weight " + name + " names " + w.from + " op " + std::to_string(idx) + " of " + std::to_string(n));
            }
        m.layer_types[name] = std::move(t);
    }
    for (const auto& l : m.layers)
        if (!m.layer_types.count(l)) fail(where, "layers names unknown layer type " + l);
    m.tail = parse_program(need(j, "tail", where), where + " tail");
    for (const auto& s : m.tail) check_step(m, s, where, "tail");
    for (const auto& [k, v] : need(j, "globals", where).items()) {
        if (v.is_number()) {
            m.globals[k] = v.get<uint64_t>();
        } else if (v.is_object() && v.contains("per_row")) {
            RowGlobal rg;
            rg.per_row = v["per_row"].get<uint64_t>();
            rg.inv_freq = v.value("inv_freq", m.rope_inv_freq);
            rg.scale = v.value("scale", 1.0);
            rg.window = v.value("window", 0ull);
            if (rg.inv_freq.size() != m.rotary_dim / 2) fail(where, "global " + k + ": inv_freq has " + std::to_string(rg.inv_freq.size()) + " values");
            const bool has_long = v.contains("long_inv_freq"), has_switch = v.contains("switch_row");
            if (has_long != has_switch)
                fail(where, "global " + k + ": long_inv_freq and switch_row must be given together");
            if (has_long) {
                rg.long_inv_freq = v["long_inv_freq"].get<std::vector<double>>();
                rg.switch_row = v["switch_row"].get<uint64_t>();
                if (rg.long_inv_freq.size() != m.rotary_dim / 2)
                    fail(where, "global " + k + ": long_inv_freq has " + std::to_string(rg.long_inv_freq.size()) + " values");
            }
            m.per_row_globals[k] = rg;
        } else {
            fail(where, "global " + k + " is neither a size nor {per_row}");
        }
    }
    // the globals come after the layer types, so the batched expert kernel's x / h / y are checked here
    for (const auto& [name, t] : m.layer_types) {
        for (size_t i = 1; i < t.gemm_block.moe_batch.args.size(); ++i)
            if (!m.globals.count(t.gemm_block.moe_batch.args[i]))
                fail(where, "layer type " + name + " gemm_block.moe_batch: arg " + t.gemm_block.moe_batch.args[i] + " is not a declared global");
        for (const auto& a : t.gemm_block.attn_block.args)
            if (!m.globals.count(a))
                fail(where, "layer type " + name + " gemm_block.attn_block: arg " + a + " is not a declared global");
    }
    const json& pack = need(j, "pack", where);
    m.embed_tensor = get<std::string>(need(pack, "embed", where), "tensor", where + " pack.embed");
    m.norm_tensor = get<std::string>(need(pack, "norm", where), "tensor", where + " pack.norm");
    m.norm_bytes = get<size_t>(need(pack, "norm", where), "bytes", where + " pack.norm");
    for (const auto& o : need(need(pack, "lm_head", where), "ops", where + " pack.lm_head")) m.lmhead_ops.push_back(parse_op(o, where + " pack.lm_head"));
    if (m.lmhead_ops.empty()) fail(where, "pack.lm_head has no ops");
    m.hf_config_check = need(j, "hf_config_check", where);
    if (j.contains("hf_config_defaults")) {
        if (!j["hf_config_defaults"].is_object()) fail(where, "hf_config_defaults is not an object");
        m.hf_config_defaults = j["hf_config_defaults"];
    }
    return m;
}

const LayerType& Manifest::layer_type(size_t layer) const {
    if (layer >= layers.size()) throw std::runtime_error("open_qwen36: layer " + std::to_string(layer) + " beyond the manifest");
    return layer_types.at(layers[layer]);
}

std::vector<std::string> Manifest::files() const {
    std::vector<std::string> f;
    for (const auto& [k, v] : contexts) f.push_back(v);
    for (const auto& [k, v] : kernels) f.push_back(v.insts);
    return f;
}

void Manifest::check_model(const json& config, const std::string& where) const {
    if (!config.is_object()) fail(where, "config.json is not an object");
    auto lacks = [&](const std::string& key) { fail(where, "config.json lacks '" + key + "'"); };
    for (const auto& [key, want] : hf_config_check.items()) {
        if (key == "model_type") {
            if (!config.contains(key)) lacks(key);
            bool ok = false;
            for (const auto& t : want) ok |= (config[key] == t);
            if (!ok) fail(where, "config.json model_type " + config[key].dump() + " is not one this kernel set serves (" + want.dump() + ")");
        } else if (key == "layer_types") {
            std::vector<std::string> got;
            if (config.contains("layer_types")) {
                got = config["layer_types"].get<std::vector<std::string>>();
            } else if (config.contains("full_attention_interval") && config.contains("num_hidden_layers")) {
                int iv = config["full_attention_interval"].get<int>(), n = config["num_hidden_layers"].get<int>();
                if (iv <= 0) fail(where, "config.json full_attention_interval must be positive");
                for (int l = 0; l < n; ++l) got.push_back((l + 1) % iv == 0 ? "full_attention" : "linear_attention");
            } else {
                lacks("layer_types");
            }
            if (got != want.get<std::vector<std::string>>())
                fail(where, "config.json layer_types differ from the kernel set's (" + std::to_string(got.size()) + " layers)");
        } else {
            // an absent key means its default when the manifest names one (Phi-3's optional
            // fields); the comparison is the same either way, so the check stays two-way
            const bool present = config.contains(key);
            if (!present && !hf_config_defaults.contains(key)) lacks(key);
            const json& got = present ? config[key] : hf_config_defaults[key];
            if (got != want)
                fail(where, "config.json " + key + " = " + got.dump() + (present ? "" : " (absent: its default)") +
                                 ", the kernel set was built for " + want.dump());
        }
    }
}

}  // namespace open_qwen36
