/// \file manifest_test.cpp
/// \brief OPEN-MANIFEST: the manifest parser reads the recipe's output, and a
///        model whose config.json disagrees with it is refused with the key named.
///        No XRT, no hardware: `manifest_test <fixtures/manifest_qwen36.json>`.
// Traces: OPEN-MANIFEST (canonical spec: specs/open-engine/spec.md)
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#include "open_qwen36/manifest.hpp"

using open_qwen36::Manifest;
using nlohmann::json;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    failures += !ok;
}

/// Expects check_model to throw with `needle` in the message.
void refused(const Manifest& m, const json& cfg, const std::string& needle, const std::string& what) {
    try {
        m.check_model(cfg, "test");
        check(false, what + " (accepted)");
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        check(msg.find(needle) != std::string::npos, what + ": " + msg);
    }
}

/// Expects Manifest::parse to throw with `needle` after `edit` breaks the manifest at `path`.
template <class Edit>
void refused_manifest(const char* path, const std::string& needle, const std::string& what, Edit edit) {
    std::ifstream f(path);
    json j = json::parse(f);
    edit(j);
    try {
        Manifest::parse(j, "edited");
        check(false, what + " (accepted)");
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        check(msg.find(needle) != std::string::npos, what + ": " + msg);
    }
}

json matching_config(const Manifest& m) {
    json cfg = m.hf_config_check;
    cfg["model_type"] = m.hf_config_check["model_type"][0];
    return cfg;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: manifest_test <manifest_qwen36.json> [<manifest_qwen3_4b.json>]\n");
        return 2;
    }
    Manifest m;
    try {
        m = Manifest::load(argv[1]);
    } catch (const std::exception& e) {
        std::printf("FAIL  load: %s\n", e.what());
        return 1;
    }
    // ---- what the recipe wrote for the 27B
    check(m.version == 1 && m.family == "qwen36moe", "version 1, family qwen36moe");
    check(m.layers.size() == 40 && m.layers[3] == "full_attention" && m.layers[0] == "linear_attention", "40 layers, attention every 4th");
    check(m.hidden == 2048 && m.vocab == 248320 && m.real_vocab == 248070, "hidden / vocab / real vocab");
    check(m.pool_bytes == 536870912 && m.lmhead_pool_bytes == 542113792 && m.chunk_bytes == 5120, "pool sizes");
    check(m.kv_row == 2048 && m.ptab_row == 1024 && m.rout_idx_off == 1024 && m.rotary_dim == 64, "kv row, ptab row, router idx, rotary dim");
    check(m.moe.stripe == 163840 && m.moe.up_bytes == 655360 && m.moe.down_core == 81920 && m.moe.pool_down == 335544320 &&
          m.moe.share_up == 503316480 && m.moe.share_gate == 503971840 && m.moe.share_down == 504627200 && m.moe.topk == 8,
          "MoE pool geometry");
    check(m.contexts.count("lx") && m.contexts.count("ax") && m.contexts.count("ln") && m.contexts.count("lm"), "four contexts");
    check(m.kernels.at("ax0").patch == "attnpos" && m.kernels.at("lx1").patch == "moeroute2" && m.kernels.at("ln").patch.empty(), "kernel patch kinds");
    const auto& lin = m.layer_types.at("linear_attention");
    const auto& full = m.layer_types.at("full_attention");
    check(lin.consts_bytes == 11882496 && lin.act_bytes == 190464 && lin.state_kind == "linear" && lin.state_bytes == 2342912, "linear layer buffers");
    check(full.consts_bytes == 1062912 && full.act_bytes == 98304 && full.state_kind == "kv" && full.state_row == 2048, "attention layer buffers");
    check(lin.program.size() == 3 && lin.program[0].op == "run" && lin.program[0].kernel == "lx0" && lin.program[0].args.size() == 5 &&
          lin.program[1].op == "moeroute2" && lin.program[1].act_off == 176128 && lin.program[2].kernel == "lx1", "linear program");
    check(full.program.size() == 3 && full.program[0].args.size() == 6 && full.program[0].args[5] == "ptab" &&
          full.program[1].act_off == 83968, "attention program");
    check(lin.pool.size() == 7 && full.pool.size() == 10 && lin.consts.size() == 11 && full.consts.size() == 6, "packing plans");
    check(lin.pool[0].op == "expert_stripes" && lin.pool[0].experts == 256 && lin.pool[1].op == "expert_down" &&
          lin.pool[5].op == "std_perm" && lin.pool[5].dst == 505282560 && lin.pool[5].nch == 2048, "pool plan ops");
    check(full.pool[8].op == "std_perm" && full.pool[8].chunk0 == 1024 && full.pool[8].dst == 511836160, "the fused q|gate split");
    check(m.tail.size() == 2 && m.tail[0].kernel == "ln" && m.tail[1].kernel == "lm" && m.tail[1].args.size() == 3, "tail program");
    check(m.globals.at("logits") == 248320 * 4 && m.globals.at("lmpool") == 542113792 && m.per_row_globals.at("ptab").per_row == 1024 &&
          m.per_row_globals.at("ptab").inv_freq.size() == 32 && m.per_row_globals.at("ptab").window == 0, "globals");
    check(m.embed_tensor == "model.embed_tokens.weight" && m.norm_tensor == "model.norm.weight" && m.lmhead_ops.size() == 1 &&
          m.lmhead_ops[0].op == "lmhead_q8" && m.lmhead_ops[0].tensor == "lm_head.weight" && m.lmhead_ops[0].chunk_bytes == 8704, "tensor names");
    check(m.has_moe, "the 27B manifest carries the MoE geometry");
    check(m.files().size() == 10, "10 files named (4 xclbin + 6 insts)");

    // ---- the model check
    json ok = matching_config(m);
    try {
        m.check_model(ok, "test");
        check(true, "a matching config.json is accepted");
    } catch (const std::exception& e) {
        check(false, std::string("a matching config.json is accepted: ") + e.what());
    }
    json interval = ok;
    interval.erase("layer_types");
    interval["full_attention_interval"] = 4;
    try {
        m.check_model(interval, "test");
        check(true, "full_attention_interval in place of layer_types is accepted");
    } catch (const std::exception& e) {
        check(false, std::string("full_attention_interval: ") + e.what());
    }
    json bad = ok; bad["hidden_size"] = 2560;
    refused(m, bad, "hidden_size", "hidden_size 2560 is refused by name");
    bad = ok; bad["model_type"] = "llama";
    refused(m, bad, "model_type", "model_type llama is refused");
    bad = ok; bad.erase("num_experts");
    refused(m, bad, "lacks 'num_experts'", "a missing key is named");
    bad = interval; bad["full_attention_interval"] = 5;
    refused(m, bad, "layer_types", "a different attention interval is refused");
    bad = ok; bad["num_hidden_layers"] = 24;
    refused(m, bad, "num_hidden_layers", "a 24-layer slice config is refused (the manifest is the 40-layer set)");

    // ---- a broken manifest
    json j = json::parse(std::string("{\"manifest_version\": 2}"));
    try {
        Manifest::parse(j, "broken");
        check(false, "manifest_version 2 is refused");
    } catch (const std::runtime_error& e) {
        check(std::string(e.what()).find("manifest_version 2") != std::string::npos, std::string("manifest_version 2 is refused: ") + e.what());
    }
    // A pack op missing a size pools::apply needs, and a moeroute2 step on a kernel
    // without the routed-expert table: both named at load, not part-way through a run.
    refused_manifest(argv[1], "nch", "a std_perm without nch is refused at load", [](json& j) {
        for (auto& lt : j["layer_types"])
            for (auto& o : lt["pack"]["pool"])
                if (o["op"] == "std_perm") o.erase("nch");
    });
    refused_manifest(argv[1], "chunk_bytes", "an lm_head op without chunk_bytes is refused at load",
                     [](json& j) { j["pack"]["lm_head"]["ops"][0].erase("chunk_bytes"); });
    refused_manifest(argv[1], "moeroute2 on kernel", "a moeroute2 step on an unpatched kernel is refused at load",
                     [](json& j) { j["kernels"]["lx1"].erase("patch"); });
    // ---- the dense family's manifest: no MoE geometry, one run per layer, a q4 head
    if (argc >= 3) {
        Manifest d;
        try {
            d = Manifest::load(argv[2]);
            check(d.family == "qwen3" && d.layers.size() == 36 && d.layers[0] == "dense", "qwen3: 36 dense layers");
            check(!d.has_moe && d.rout_idx_off == 1024, "qwen3: no MoE geometry");
            check(d.hidden == 2560 && d.vocab == 151936 && d.real_vocab == 151669 && d.kv_row == 4096 && d.ptab_row == 2048 &&
                  d.rotary_dim == 128, "qwen3: layout");
            const auto& lt = d.layer_types.at("dense");
            check(lt.program.size() == 1 && lt.program[0].op == "run" && lt.program[0].kernel == "dx" && lt.program[0].args.size() == 6 &&
                  lt.state_kind == "kv" && lt.state_row == 4096, "qwen3: one run per layer");
            check(d.kernels.at("dx").patch == "attnpos" && d.kernels.count("lm") && d.contexts.size() == 3, "qwen3: kernels");
            check(lt.pool.size() == 7 && lt.pool[0].op == "std_perm" && lt.pool[0].in_dim == 2560 && lt.consts.size() == 4,
                  "qwen3: packing plan");
            check(d.lmhead_ops.size() == 1 && d.lmhead_ops[0].op == "std_perm" && d.lmhead_ops[0].nch == 47480, "qwen3: q4 head");
            json ok = matching_config(d);
            d.check_model(ok, "qwen3");
            check(true, "qwen3: a matching config.json is accepted");
            json bad = ok; bad["intermediate_size"] = 12288;
            refused(d, bad, "intermediate_size", "qwen3: an 8B config is refused by name");
        } catch (const std::exception& e) {
            check(false, std::string("qwen3 fixture: ") + e.what());
        }
    }
    // ---- Gemma 3: two layer types on one stream, windows, two position tables
    if (argc >= 4) {
        try {
            Manifest g = Manifest::load(argv[3]);
            check(g.family == "gemma3" && g.layers.size() == 34 && g.layers[0] == "dense_local" && g.layers[5] == "dense", "gemma3: 5:1 local / global layers");
            check(g.kernels.at("dx").window == 0 && g.kernels.at("dx_local").window == 1024 &&
                  g.kernels.at("dx").insts == g.kernels.at("dx_local").insts, "gemma3: dx / dx_local share a stream, own windows");
            check(g.per_row_globals.size() == 2 && g.per_row_globals.at("ptab").window == 0 &&
                  g.per_row_globals.at("ptab_local").window == 1024 && g.per_row_globals.at("ptab_local").inv_freq[0] == 1.0 &&
                  g.per_row_globals.at("ptab").inv_freq[0] == 0.125, "gemma3: two position tables");
            check(g.layer_types.at("dense_local").program[0].args.back() == "ptab_local" &&
                  g.layer_types.at("dense").program[0].args.back() == "ptab", "gemma3: each layer type binds its table");
            check(g.layer_types.at("dense").consts.size() == 6 && g.hidden == 2560 && g.vocab == 262208 && g.real_vocab == 262145, "gemma3: consts and vocab");
            uint64_t s0, n0, s1, n1;
            stream_patch::attn_window(1500, 1024, &s0, &n0);
            stream_patch::attn_window(0, 1024, &s1, &n1);
            check(s0 == 477 && n0 == 1023 && s1 == 0 && n1 == 1, "attn_window: [477, 1500) at 1500; a dummy row at 0");
        } catch (const std::exception& e) {
            check(false, std::string("gemma3 fixture: ") + e.what());
        }
    }
    // ---- HunYuan dense: a vocabulary that is not a whole number of head bands
    if (argc >= 5) {
        try {
            Manifest h = Manifest::load(argv[4]);
            check(h.family == "hunyuan" && h.layers.size() == 32 && h.layers[0] == "dense", "hunyuan: 32 dense layers");
            check(h.hidden == 4096 && h.kv_row == 4096 && h.rotary_dim == 128, "hunyuan: layout");
            // the head is padded to whole 64-row bands (128192); the ids stop at the tokenizer's count
            check(h.vocab == 128192 && h.real_vocab == 128166 && h.lmhead_ops[0].nch == 64096, "hunyuan: padded head, real vocab");
            check(h.layer_types.at("dense").consts.size() == 4, "hunyuan: ln, post-ln and the two qk norms");
            json ok = matching_config(h);
            check(ok["vocab_size"] == 128167, "hunyuan: config.json is checked against the model's own vocab_size");
            h.check_model(ok, "hunyuan");
            check(true, "hunyuan: a matching config.json is accepted");
            json bad = ok; bad["vocab_size"] = 128192;
            refused(h, bad, "vocab_size", "hunyuan: the padded count is refused as a config.json vocab_size");
        } catch (const std::exception& e) {
            check(false, std::string("hunyuan fixture: ") + e.what());
        }
    }
    std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
