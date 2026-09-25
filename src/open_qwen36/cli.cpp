/// \file cli.cpp
/// \brief Drive the open Qwen3.6 engine without the OFLM app: token ids in,
///        greedy token ids and logits out. The test surface for core.cpp and
///        the way to run the engine on a box where the app itself does not
///        build (this one: no Boost / vcpkg / tokenizers-cpp).
///
///   open_qwen36_cli --model <dir> --kernels <dir (manifest.json + xclbins)> --ids 1,2,3 [--max-tokens N]
///       [--layers N] [--max-ctx N] [--dump-logits <prefix>] [--twice]
///       [--at-position P] [--ids-file <path>] [--gemm-block] [--prefill-logits]
///       [--dump-act <layer>:<off>:<bytes>:<path>]   bring-up: a slice of a layer's act scratch
///
/// The prompt ids are prefilled by sequential decode (logits skipped), then
/// greedy decode runs for --max-tokens. Each produced id is printed on its
/// own line as `token <id>` so a wrapper can detokenize (tools/chat.py).
/// --dump-logits writes `<prefix>_t<i>.bin` (f32[vocab]) for every position
/// with logits, which compare_decode.py can score. --twice runs the whole
/// request a second time on the same resident engine (state reset check).
/// --at-position P first seeks to position P with no cache rows in between
/// (a capacity check: the attention window then spans P rows).
/// --det-step N [--det-full] steps the --ids (forced, from --at-position) N times on one
/// resident engine and compares every step's logits (--det-full: every buffer) bit for
/// bit against a reference pass (OPEN-REQUEST-ISOLATION); exit 1 if any rep moved.
/// OFLM_ROUTE_CHECK=1 re-reads each router record and counts reads that changed.
///
/// 0167/#32: --gemm-block prefills via Core::step_gemm_block() (T tokens per
/// layer as 5 whole-array GEMM dispatches plus T attention dispatches, the
/// whole prompt padded up to a multiple of the kernel set's block size)
/// instead of one step() per token. --prefill-logits forces logits at every
/// PREFILL position reached (not just the true last one), written to
/// `<prefix>_p<position>.bin` (position-indexed, independent of
/// --dump-logits' own `_t<i>` decode-loop numbering) so a --gemm-block run
/// and a plain run over the SAME --ids can be compared position for position:
/// run once without --gemm-block, once with, then diff `_p<position>.bin`
/// pairs (correlation / argmax / top-5).
#include <algorithm>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "open_qwen36/core.hpp"

using open_qwen36::Core;
using open_qwen36::CoreConfig;

namespace {

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> ids;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) ids.push_back(std::atoi(s.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return ids;
}

int argmax(const std::vector<float>& v, size_t n) {
    int best = 0;
    for (size_t i = 1; i < n; ++i)
        if (v[i] > v[best]) best = static_cast<int>(i);
    return best;
}

void dump(const std::string& prefix, int t, const std::vector<float>& v) {
    std::string p = prefix + "_t" + std::to_string(t) + ".bin";
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
    std::fprintf(stderr, "wrote %s\n", p.c_str());
}

/// 0167/#32: position-indexed (not sequence-indexed) so a --gemm-block run
/// and a plain run dump the SAME filename for the SAME absolute prompt
/// position, however many dispatches produced it.
void dump_pos(const std::string& prefix, int pos, const std::vector<float>& v) {
    std::string p = prefix + "_p" + std::to_string(pos) + ".bin";
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
    std::fprintf(stderr, "wrote %s\n", p.c_str());
}

struct Args {
    CoreConfig cfg;
    std::vector<int> ids;
    int max_tokens = 16;
    std::string dump_prefix;
    bool twice = false;
    int repeat = 1;
    int gap_ms = 0;
    int at_position = 0;
    bool gemm_block = false;        // 0167/#32: prefill via step_gemm_block()
    bool block_major = false;       // --block-major: force the old block-at-a-time schedule
    bool prefill_logits = false;    // 0167/#32: logits (dump_pos) at every prefill position reached
    std::string dump_act;           // bring-up: "<layer>:<off>:<bytes>:<path>"
    int bench = 0;                  // --bench N[:LAYER]: time the route's dispatches instead of running a prompt
    int bench_layer = 0;            // which layer's route -- the 35B's two types run different GEMM shapes
    int bench_decode = 0;           // --bench-decode N: the same for the per-token program
    int det_step = 0;               // --det-step N: the --ids steps from --at-position, N times, compared
    bool det_full = false;          // --det-full: compare every buffer a step writes, not only the logits
    std::string bench_kernel;       // --bench-kernel NAME:REPS[:LAYER]: one kernel, over and over
    std::string pmode = "performance";   // --pmode: the NPU power mode to set first ("none" leaves it)
};

/// Set the NPU power mode, exactly as the app does for `run` / `serve` / `bench`
/// (src/src/main.cpp). The app has always done this and this CLI never did, so a
/// standalone measurement silently inherited whatever mode the last app run left
/// behind -- and the mode is worth about 2x on this hardware, which made open and
/// closed numbers taken minutes apart incomparable for reasons neither engine
/// controlled. Anything measured here now starts from a stated mode.
void set_power_mode(const std::string& mode) {
    if (mode == "none") return;
    if (mode != "default" && mode != "powersaver" && mode != "balanced" && mode != "performance" && mode != "turbo") {
        std::fprintf(stderr, "invalid --pmode %s (default, powersaver, balanced, performance, turbo, none)\n", mode.c_str());
        std::exit(2);
    }
#ifdef _WIN32
    const std::string cmd = "cd \"C:\\Windows\\System32\\AMD\" && .\\xrt-smi.exe configure --pmode " + mode + " > NUL 2>&1";
#else
    const std::string cmd = "xrt-smi configure --pmode " + mode + " > /dev/null 2>&1";
#endif
    const int rc = std::system(cmd.c_str());
    std::fprintf(stderr, "NPU power mode: %s%s\n", mode.c_str(), rc == 0 ? "" : " (xrt-smi failed; mode is whatever it was)");
}

/// Write a slice of a layer's act scratch to a file, so a bring-up run can compare one
/// stage's output against the reference instead of inferring it from the logits.
void dump_act_slice(Core& core, const std::string& spec) {
    int layer = 0; long long off = 0, n = 0;
    char path[512] = {0};
    if (std::sscanf(spec.c_str(), "%d:%lld:%lld:%511s", &layer, &off, &n, path) != 4 || n <= 0) {
        std::fprintf(stderr, "--dump-act wants <layer>:<off>:<bytes>:<path>, got %s\n", spec.c_str());
        return;
    }
    std::vector<uint8_t> buf(static_cast<size_t>(n));
    core.read_act(layer, static_cast<size_t>(off), buf.size(), buf.data());
    std::FILE* f = std::fopen(path, "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", path); return; }
    std::fwrite(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    std::fprintf(stderr, "wrote %lld B of layer %d act at %lld -> %s\n", n, layer, off, path);
}

Args parse(int argc, char** argv) {
    Args a;
    a.cfg.model_dir = std::getenv("OFLM_MODEL_DIR") ? std::getenv("OFLM_MODEL_DIR") : "";
    a.cfg.kernel_dir = std::getenv("OFLM_OPEN_KERNELS_DIR") ? std::getenv("OFLM_OPEN_KERNELS_DIR") : "";
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", k.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (k == "--model") a.cfg.model_dir = val();
        else if (k == "--kernels") a.cfg.kernel_dir = val();
        else if (k == "--ids") a.ids = parse_ids(val());
        else if (k == "--ids-file") {
            std::ifstream f(val());
            std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            a.ids = parse_ids(s);
        } else if (k == "--max-tokens") a.max_tokens = std::atoi(val().c_str());
        else if (k == "--layers") a.cfg.num_layers = std::atoi(val().c_str());
        else if (k == "--max-ctx") a.cfg.max_ctx = static_cast<size_t>(std::atoll(val().c_str()));
        else if (k == "--dump-logits") a.dump_prefix = val();
        else if (k == "--twice") a.twice = true;
        else if (k == "--repeat") a.repeat = std::atoi(val().c_str());
        else if (k == "--at-position") a.at_position = std::atoi(val().c_str());
        else if (k == "--dump-act") a.dump_act = val();            // "<layer>:<off>:<n>:<path>"
        else if (k == "--quiet") a.cfg.verbose = false;
        else if (k == "--gemm-block") a.gemm_block = true;
        else if (k == "--block-major") a.block_major = true;
        else if (k == "--prefill-logits") a.prefill_logits = true;
        else if (k == "--bench") {
            const std::string v = val();
            const size_t c = v.find(':');
            a.bench = std::atoi(v.substr(0, c).c_str());
            if (c != std::string::npos) a.bench_layer = std::atoi(v.substr(c + 1).c_str());
        }
        else if (k == "--bench-decode") a.bench_decode = std::atoi(val().c_str());
        else if (k == "--det-step") a.det_step = std::atoi(val().c_str());
        else if (k == "--det-full") a.det_full = true;
        else if (k == "--bench-kernel") a.bench_kernel = val();
        else if (k == "--pmode") a.pmode = val();
        // The server has a host gap right here that the CLI does not: between the last
        // prefill dispatch and the first decode one it samples, detokenises and writes
        // the first SSE chunk to a socket. Every dx timeout so far has landed on that
        // dispatch, so this makes the gap reproducible without the server.
        else if (k == "--gap-ms") a.gap_ms = std::atoi(val().c_str());
        else if (k == "--timeout-ms") a.cfg.timeout_ms = static_cast<unsigned>(std::strtoul(val().c_str(), nullptr, 10));
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); std::exit(2); }
    }
    if (a.cfg.model_dir.empty() || a.cfg.kernel_dir.empty() || a.ids.empty()) {
        std::fprintf(stderr, "usage: open_qwen36_cli --model <dir> --kernels <dir> --ids 1,2,3 [--max-tokens N] "
                             "[--layers N] [--max-ctx N] [--dump-logits <prefix>] [--twice] [--at-position P] "
                             "[--gemm-block] [--block-major] [--prefill-logits] [--bench N[:LAYER]] [--bench-decode N] "
                             "[--bench-kernel NAME:REPS[:LAYER]] [--pmode MODE]\n");
        std::exit(2);
    }
    return a;
}

/// Prefill + greedy decode; returns the produced ids.
std::vector<int> request(Core& core, const Args& a) {
    using clock = std::chrono::steady_clock;
    core.reset();
    if (a.at_position > 0) {
        // Capacity check: jump the position so the next token's attention
        // window spans P rows (the rows in between are the zeroed buffer).
        std::fprintf(stderr, "seeking to position %d\n", a.at_position);
        core.seek(a.at_position);
    }
    auto t0 = clock::now();
    int dumped = 0;
    // 0167/#32: --gemm-block prefills the WHOLE prompt in GT-wide blocks via
    // step_gemm_block() (never falling back to step() for a short tail -- the
    // last block is padded with a repeated in-range id, hardware-proven exact
    // for the real columns). --prefill-logits forces logits (want_logits) at
    // every position reached, dumped by ABSOLUTE position (dump_pos), so a
    // --gemm-block run and a plain run can be diffed position for position
    // over the SAME --ids.
    // Stage lines shared by both schedules: which stage to work on, when the line above
    // says the route is too slow.
    auto stages = [&](const char* what, size_t n0, size_t n1, size_t t_real) {
        const auto& tm = core.last_timing();
        std::fprintf(stderr, "  %s [%zu,%zu) t_real=%zu: %.1f ms (GEMM %.1f, host %.1f, per-token %.1f, lm_head %.1f)\n",
                     what, n0, n1, t_real, tm.total_ms, tm.part0_ms, tm.part1_ms, tm.route_ms, tm.lmhead_ms);
        std::fprintf(stderr,
                     "      mid %.1f (dn conv %.1f, dn rule %.1f, attn %.1f), prenorm %.1f, gemm tile %.1f,"
                     " gemm tr %.1f, tail %.1f, shared %.1f, state %.1f, sync %.1f, setup %.1f"
                     " | moe prep %.1f, patch %.1f, run %.1f, read %.1f | residue %.1f\n",
                     tm.mid_ms, tm.dn_conv_ms, tm.dn_rule_ms, tm.attn_ms, tm.prenorm_ms, tm.gemm_tile_ms,
                     tm.gemm_tr_ms, tm.tail_ms, tm.shared_ms, tm.state_ms, tm.sync_ms, tm.setup_ms,
                     tm.moe_prep_ms, tm.moe_patch_ms, tm.moe_run_ms, tm.moe_read_ms,
                     tm.total_ms - tm.part0_ms - tm.part1_ms - tm.route_ms - tm.lmhead_ms);
        for (const auto& [kn, d] : core.take_dispatch_stats())
            std::fprintf(stderr, "      %-22s %4d calls %8.1f ms total %7.3f mean %7.3f min\n", kn.c_str(),
                         d.calls, d.ms, d.ms / d.calls, d.min_ms);
    };
    size_t i = 0;
    // B(2): the whole prompt through each layer before the next, so a layer's MoE streams
    // each expert once for every token of the prompt that wants it instead of once per
    // block. Bit-exact against the block-at-a-time loop below, which --block-major forces.
    if (a.gemm_block && core.layer_major_ok() && !a.block_major) {
        core.set_block_logits_all(a.prefill_logits && !a.dump_prefix.empty());
        core.step_gemm_prompt(a.ids, true);
        stages("gemm-prompt", 0, a.ids.size(), a.ids.size());
        for (size_t t = 0; t < core.block_logits().size(); ++t)
            dump_pos(a.dump_prefix, static_cast<int>(t), core.block_logits()[t]);
        if (!a.dump_prefix.empty() && core.block_logits().empty())
            dump_pos(a.dump_prefix, static_cast<int>(a.ids.size() - 1), core.logits());
        if (!a.dump_prefix.empty()) dump(a.dump_prefix, dumped++, core.logits());
        i = a.ids.size();
    } else if (a.gemm_block) {
        size_t GT = core.gemm_block_t();
        if (GT == 0) { std::fprintf(stderr, "ERROR: --gemm-block given but this kernel set has no gemm_block program\n"); std::exit(2); }
        core.set_block_logits_all(a.prefill_logits && !a.dump_prefix.empty());
        while (i < a.ids.size()) {
            size_t t_real = std::min(GT, a.ids.size() - i);
            std::vector<int> blk(a.ids.begin() + static_cast<long>(i), a.ids.begin() + static_cast<long>(i + t_real));
            blk.resize(GT, blk.empty() ? 0 : blk.back());
            bool want = a.prefill_logits || (i + t_real == a.ids.size());
            core.step_gemm_block(blk, t_real, want);
            stages("gemm-block", i, i + GT, t_real);
            // every real position of the block, like the sequential path's --prefill-logits
            for (size_t t = 0; t < core.block_logits().size(); ++t)
                dump_pos(a.dump_prefix, static_cast<int>(i + t), core.block_logits()[t]);
            if (!a.dump_prefix.empty() && want && core.block_logits().empty())
                dump_pos(a.dump_prefix, static_cast<int>(i + t_real - 1), core.logits());
            if (!a.dump_prefix.empty() && i + t_real == a.ids.size()) dump(a.dump_prefix, dumped++, core.logits());
            i += t_real;
        }
    }
    for (; i < a.ids.size(); ++i) {
        bool last = i + 1 == a.ids.size();
        bool want = a.prefill_logits || last;
        core.step(a.ids[i], want);
        if (!a.dump_prefix.empty() && want) dump_pos(a.dump_prefix, static_cast<int>(i), core.logits());
        if (!a.dump_prefix.empty() && last) dump(a.dump_prefix, dumped++, core.logits());  // preserve the original _t<i> convention
    }
    if (a.gap_ms > 0) {
        std::fprintf(stderr, "idling %d ms before the first decode dispatch\n", a.gap_ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(a.gap_ms));
    }
    double prefill_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    std::fprintf(stderr, "prefill %zu tokens: %.0f ms (%.0f ms/token)\n", a.ids.size(), prefill_ms, prefill_ms / a.ids.size());

    std::vector<int> out;
    auto t1 = clock::now();
    core.take_dispatch_stats();   // prefill's dispatches are not this loop's
    int tok = argmax(core.logits(), core.real_vocab());
    for (int n = 0; n < a.max_tokens; ++n) {
        out.push_back(tok);
        std::printf("token %d\n", tok);
        std::fflush(stdout);
        if (n + 1 == a.max_tokens) break;
        core.step(tok, true);
        if (!a.dump_prefix.empty()) dump(a.dump_prefix, dumped++, core.logits());
        const auto& tm = core.last_timing();
        std::fprintf(stderr, "  step @%d: %.1f ms (part0 %.1f, route %.2f, part1 %.1f, lm_head %.1f)\n", core.position() - 1,
                     tm.total_ms, tm.part0_ms, tm.route_ms, tm.part1_ms, tm.lmhead_ms);
        for (const auto& [kn, d] : core.take_dispatch_stats())
            std::fprintf(stderr, "      %-22s %4d calls %8.1f ms total %7.3f mean %7.3f min\n", kn.c_str(),
                         d.calls, d.ms, d.ms / d.calls, d.min_ms);
        tok = argmax(core.logits(), core.real_vocab());
    }
    double dec_ms = std::chrono::duration<double, std::milli>(clock::now() - t1).count();
    if (core.late_route_reads())
        std::fprintf(stderr, "late router reads caught so far: %llu\n",
                     static_cast<unsigned long long>(core.late_route_reads()));
    if (out.size() > 1)
        std::fprintf(stderr, "decode %zu tokens: %.0f ms/token (%.2f tok/s)\n", out.size() - 1, dec_ms / (out.size() - 1),
                     1000.0 * (out.size() - 1) / dec_ms);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    // The server reads this (engine.cpp); the CLI is the tool you reach for when a
    // dispatch times out, so it should honour the same knob - and a small value is how
    // the timeout path itself gets exercised without waiting for a real one.
    if (const char* tm = std::getenv("OFLM_OPEN_TIMEOUT_MS"))
        a.cfg.timeout_ms = static_cast<unsigned>(std::strtoul(tm, nullptr, 10));
    set_power_mode(a.pmode);
    try {
        auto t0 = std::chrono::steady_clock::now();
        Core core(a.cfg);
        core.load_weights();
        std::fprintf(stderr, "resident after %.1f s\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        if (a.bench) {
            core.bench_dispatch(a.bench_layer, a.bench);
            std::printf("DONE\n");
            return 0;
        }
        if (!a.bench_kernel.empty()) {
            const size_t c1 = a.bench_kernel.find(':');
            if (c1 == std::string::npos) { std::fprintf(stderr, "--bench-kernel wants NAME:REPS[:LAYER]\n"); std::exit(2); }
            const size_t c2 = a.bench_kernel.find(':', c1 + 1);
            const std::string name = a.bench_kernel.substr(0, c1);
            const int reps = std::atoi(a.bench_kernel.substr(c1 + 1, c2 - c1 - 1).c_str());
            const int layer = c2 == std::string::npos ? 0 : std::atoi(a.bench_kernel.substr(c2 + 1).c_str());
            core.bench_kernel(name, reps, layer, a.ids[0]);
            std::printf("DONE\n");
            return 0;
        }
        if (a.bench_decode) {
            // --at-position applies here too, and it is the whole point of the probe: ax0 is the
            // only decode dispatch whose transfers grow with the window, so a bench taken at
            // position 1 (which is all this path could measure before) says nothing about what a
            // step costs at a real context length. seek() moves the position; the rows in between
            // are the zeroed buffer, which is the right window SIZE and meaningless values -- the
            // bench times dispatches and never reads a logit.
            if (a.at_position > 0) {
                std::fprintf(stderr, "seeking to position %d\n", a.at_position);
                core.seek(a.at_position);
            }
            core.step(a.ids[0], true);       // one real step first: the attnpos and route patches
            core.bench_decode(a.bench_decode);
            std::printf("DONE\n");
            return 0;
        }
        if (a.det_step) {
            if (a.at_position > 0) core.seek(a.at_position);
            const int bad = core.det_step(a.det_step, a.ids, a.det_full);
            std::printf(bad ? "NONDETERMINISTIC\n" : "DONE\n");
            return bad ? 1 : 0;
        }
        std::vector<int> first = request(core, a);
        if (!a.dump_act.empty()) dump_act_slice(core, a.dump_act);
        int reps = a.twice ? 2 : a.repeat;
        for (int r = 1; r < reps; ++r) {
            // The app checkpoints after the prompt and restores before the next
            // request; do the same so the snapshot path is exercised too.
            auto snap = core.checkpoint();
            core.restore(snap);
            std::vector<int> again = request(core, a);
            bool same = first == again;
            std::fprintf(stderr, "request %d %s the first\n", r + 1, same ? "REPRODUCED" : "DIFFERS FROM");
            if (!same) return 1;
        }
        std::printf("DONE\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
