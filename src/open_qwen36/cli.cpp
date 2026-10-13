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
/// --spec decodes speculatively (OPEN-DECODE-SPECULATIVE): same tokens as without it. --spec-method
/// names the drafter (default lookup); refused when the kernel set cannot serve it.
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
#include <cmath>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "open_qwen36/core.hpp"
#include "open_qwen36/spec.hpp"

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
    std::vector<std::string> dump_act;   // bring-up: "<layer>:<off>:<bytes>:<path>", repeatable
    int bench = 0;                  // --bench N[:LAYER]: time the route's dispatches instead of running a prompt
    int bench_layer = 0;            // which layer's route -- the 35B's two types run different GEMM shapes
    int bench_decode = 0;           // --bench-decode N: the same for the per-token program
    int det_step = 0;               // --det-step N: the --ids steps from --at-position, N times, compared
    bool det_full = false;          // --det-full: compare every buffer a step writes, not only the logits
    int bench_step = 0;             // --bench-step N: the REAL step, min of N, at --at-position
    std::string bench_kernel;       // --bench-kernel NAME:REPS[:LAYER]: one kernel, over and over
    std::string pmode = "performance";   // --pmode: the NPU power mode to set first ("none" leaves it)
    bool spec = false;              // --spec: speculative greedy decode (OPEN-DECODE-SPECULATIVE)
    int spec_width = 0;             // --spec-width N: positions per verify pass (0 = the kernel set's, else 4)
    bool spec_ref = false;          // --spec-ref: the reference verifier even when the set has a verify program
    bool spec_log = false;          // --spec-log: one line per pass
    std::string spec_method = "lookup";   // --spec-method M: this CLI has no registry entry to read the method from
    int verify_check = 0;           // --verify-check N: N greedy positions by step(), then by verify() passes, compared
    int bench_verify = 0;           // --bench-verify N: N verify passes and N steps after the prompt, timed
    int verify_probe = 0;           // --verify-probe L (with OFLM_OPEN_VERIFY_STOP=L): act of layers < L, verify vs step
    std::string decode_route;        // Stage 2.7: --decode-route {npu,host,auto} (default: the env's initial route)
    std::string decode_schedule;    // Stage 2.7: --decode-schedule "6n,6h,6n" -- per-step routes in the decode loop
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
    if (const char* pm = std::getenv("OFLM_OPEN_PREFILL_MODE")) a.cfg.prefill_mode = pm;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", k.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (k == "--model") a.cfg.model_dir = val();
        else if (k == "--kernels") a.cfg.kernel_dir = val();
        else if (k == "--prefill-mode") a.cfg.prefill_mode = val();
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
        else if (k == "--dump-act") a.dump_act.push_back(val());   // "<layer>:<off>:<n>:<path>"
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
        else if (k == "--verify-check") a.verify_check = std::atoi(val().c_str());
        else if (k == "--bench-verify") a.bench_verify = std::atoi(val().c_str());
        else if (k == "--verify-probe") a.verify_probe = std::atoi(val().c_str());
        else if (k == "--det-full") a.det_full = true;
        else if (k == "--bench-step") a.bench_step = std::atoi(val().c_str());
        else if (k == "--bench-kernel") a.bench_kernel = val();
        else if (k == "--pmode") a.pmode = val();
        else if (k == "--spec") a.spec = true;
        else if (k == "--spec-width") { a.spec = true; a.spec_width = std::atoi(val().c_str()); }
        else if (k == "--spec-ref") { a.spec = true; a.spec_ref = true; }
        else if (k == "--spec-log") a.spec_log = true;
        else if (k == "--spec-method") { a.spec = true; a.spec_method = val(); }
        // The server has a host gap right here that the CLI does not: between the last
        // prefill dispatch and the first decode one it samples, detokenises and writes
        // the first SSE chunk to a socket. Every dx timeout so far has landed on that
        // dispatch, so this makes the gap reproducible without the server.
        else if (k == "--gap-ms") a.gap_ms = std::atoi(val().c_str());
        else if (k == "--timeout-ms") a.cfg.timeout_ms = static_cast<unsigned>(std::strtoul(val().c_str(), nullptr, 10));
        else if (k == "--decode-route") a.decode_route = val();          // Stage 2.7
        else if (k == "--decode-schedule") a.decode_schedule = val();     // Stage 2.7
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); std::exit(2); }
    }
    if (a.cfg.model_dir.empty() || a.cfg.kernel_dir.empty() || a.ids.empty()) {
        std::fprintf(stderr, "usage: open_qwen36_cli --model <dir> --kernels <dir> --ids 1,2,3 [--max-tokens N] "
                             "[--layers N] [--max-ctx N] [--dump-logits <prefix>] [--twice] [--at-position P] "
                             "[--gemm-block] [--block-major] [--prefill-logits] [--bench N[:LAYER]] [--bench-decode N] "
                             "[--bench-step N] [--bench-kernel NAME:REPS[:LAYER]] [--pmode MODE] [--prefill-mode fast|lean] "
                             "[--spec] [--spec-width N] [--spec-ref] [--spec-log] [--spec-method M]\n");
        std::exit(2);
    }
    return a;
}

// greedy tokens do not depend on the draft, so this prints the same `token <id>` lines as the plain loop
std::vector<int> decode_spec(Core& core, const Args& a, int tok) {
    namespace sp = open_qwen36::spec;
    using clock = std::chrono::steady_clock;
    const bool fast = core.verify_width() > 0 && !a.spec_ref;
    size_t W = a.spec_width > 0 ? static_cast<size_t>(a.spec_width) : (fast ? core.verify_width() : size_t{4});
    if (fast) W = std::min(W, core.verify_width());
    if (W < 2) { std::fprintf(stderr, "--spec-width must be >= 2\n"); std::exit(2); }
    std::fprintf(stderr, "speculative decode: width %zu, method %s, %s\n", W, a.spec_method.c_str(),
                 fast ? "the kernel set's verify pass" : "reference verify (step + checkpoint/restore)");
    const std::unique_ptr<sp::Drafter> drafter = sp::make_drafter(a.spec_method);
    sp::ReferenceVerify<Core> ref(core);
    sp::CoreVerify<Core> cv(core);
    sp::Stats st;
    std::vector<int> hist = a.ids;      // every token the engine has consumed
    const size_t real = core.real_vocab();
    auto never = [](int) { return false; };
    std::vector<int> out{tok};
    std::printf("token %d\n", tok);
    std::fflush(stdout);
    double pass_ms[2] = {0, 0};          // plain steps, drafted passes
    while (out.size() < static_cast<size_t>(a.max_tokens)) {
        const size_t left = static_cast<size_t>(a.max_tokens) - out.size();
        std::vector<int> ids{tok};
        for (int d : drafter->propose(hist, tok, std::min(W - 1, left - 1))) ids.push_back(d);
        const auto tp = clock::now();
        const size_t before = out.size();
        size_t m = 1;
        if (ids.size() == 1) {
            core.step(tok, true);
            out.push_back(argmax(core.logits(), real));
        } else if (fast) {
            // the CLI's own argmax rule on the pass's logits (they are step()'s, bit for bit)
            m = sp::run_pass(cv, ids, left, [&](size_t i) { return sp::argmax(cv.logits(i).data(), real); }, never, out);
        } else {
            m = sp::run_pass(ref, ids, left, [&](size_t i) { return sp::argmax(ref.logits(i).data(), real); }, never, out);
        }
        const double ms = std::chrono::duration<double, std::milli>(clock::now() - tp).count();
        pass_ms[ids.size() > 1] += ms;
        st.add(ids.size(), m);
        if (a.spec_log)
            std::fprintf(stderr, "  pass @%d: %zu positions, %zu kept, %.1f ms\n", core.position() - static_cast<int>(m),
                         ids.size(), m, ms);
        for (size_t i = before; i < out.size(); ++i) std::printf("token %d\n", out[i]);
        std::fflush(stdout);
        drafter->observe(ids.data(), m);
        hist.insert(hist.end(), ids.begin(), ids.begin() + static_cast<long>(m));
        tok = out.back();
    }
    std::fprintf(stderr,
                 "spec: %zu passes (%zu drafted), %zu positions, %zu draft tokens kept, %.2f tokens/pass;"
                 " plain steps %.0f ms, drafted passes %.0f ms\n",
                 st.passes, st.drafted, st.positions, st.accepted, st.passes ? double(st.emitted) / st.passes : 0.0,
                 pass_ms[0], pass_ms[1]);
    return out;
}

// OPEN-DECODE-SPECULATIVE-VERIFY gate: every fourth pass is a plain step so the decode stream reads the slot accept() left
int g_verify_bad = 0;

void verify_check(Core& core, const Args& a) {
    const int n = a.verify_check;
    const size_t V = core.vocab();
    auto snap = core.checkpoint();
    std::vector<int> tokens{argmax(core.logits(), core.real_vocab())};
    std::vector<std::vector<float>> ref;
    for (int i = 0; i < n; ++i) {
        core.step(tokens.back(), true);
        ref.push_back(core.logits());
        tokens.push_back(argmax(core.logits(), core.real_vocab()));
    }
    core.restore(snap);
    const int p0 = core.position();
    int bad = 0, passes = 0, checked = 0;
    double vms = 0, sms = 0;
    int nv = 0, ns = 0;
    auto cmp = [&](const std::vector<float>& got, int i, const char* what) {
        ++checked;
        if (std::memcmp(got.data(), ref[i].data(), V * 4) == 0) return;
        float md = 0;
        size_t nd = 0;
        for (size_t v = 0; v < V; ++v) {
            const float d = std::fabs(got[v] - ref[i][v]);
            if (got[v] != ref[i][v]) ++nd;
            md = std::max(md, d);
        }
        std::fprintf(stderr, "  MISMATCH %s position %d (+%d): %zu of %zu logits differ, max |diff| %g\n", what,
                     p0 + i, i, nd, V, md);
        ++bad;
    };
    int i = 0;
    for (int k = 0; i + 1 < n; ++k) {
        if (k % 4 == 3) {                          // a plain step after an accept
            core.step(tokens[i], true);
            sms += core.last_timing().total_ms;
            ++ns;
            cmp(core.logits(), i, "step");
            ++i;
            continue;
        }
        const bool right = k % 3 != 2;
        const int draft = right ? tokens[i + 1] : (tokens[i + 1] + 1) % static_cast<int>(core.real_vocab());
        const int ids[2] = {tokens[i], draft};
        int am[2] = {-1, -1};
        core.verify(ids, 2, am);
        vms += core.last_timing().total_ms;
        ++nv;
        ++passes;
        cmp(core.verify_logits(0), i, "verify[0]");
        if (am[0] != tokens[i + 1]) {
            std::fprintf(stderr, "  ARGMAX position %d: verify %d, step %d\n", p0 + i, am[0], tokens[i + 1]);
            ++bad;
        }
        if (right) {
            cmp(core.verify_logits(1), i + 1, "verify[1]");
            if (am[1] != tokens[i + 2]) {
                std::fprintf(stderr, "  ARGMAX position %d: verify %d, step %d\n", p0 + i + 1, am[1], tokens[i + 2]);
                ++bad;
            }
            core.accept(2);
            i += 2;
        } else {
            core.accept(1);
            i += 1;
        }
        if (core.position() != p0 + i) {
            std::fprintf(stderr, "  POSITION after accept: %d, want %d\n", core.position(), p0 + i);
            ++bad;
        }
    }
    std::fprintf(stderr, "verify check: %d positions from %d, %d verify passes (%.1f ms mean), %d steps (%.1f ms mean), "
                         "%d logit rows compared: %s (%d bad)\n",
                 i, p0, passes, nv ? vms / nv : 0.0, ns, ns ? sms / ns : 0.0, checked, bad ? "FAIL" : "PASS", bad);
    std::printf(bad ? "VERIFY CHECK FAIL\n" : "VERIFY CHECK PASS\n");
    g_verify_bad = bad;
}

// both positions of each pass are accepted, so the position advances by two
void bench_verify(Core& core, const Args& a) {
    const int n = a.bench_verify;
    int tok = argmax(core.logits(), core.real_vocab());
    std::vector<double> v2, v1;
    for (int r = 0; r < n; ++r) {
        const int ids[2] = {tok, tok};
        int am[2];
        core.verify(ids, 2, am);
        v2.push_back(core.last_timing().total_ms);
        {
            const auto& t = core.last_timing();
            std::fprintf(stderr, "  verify @%d: %.1f ms (layers blocked %.1f, tail %.1f, host read+argmax %.2f, embed %.2f, patch %.2f)\n",
                         core.position(), t.total_ms, t.part0_ms, t.part1_ms, t.route_ms, t.embed_ms, t.patch_ms);
        }
        core.accept(2);
        tok = am[1];
    }
    for (int r = 0; r < n; ++r) {
        core.step(tok, true);
        v1.push_back(core.last_timing().total_ms);
        {
            const auto& t = core.last_timing();
            std::fprintf(stderr, "  step @%d: %.1f ms (part0 %.1f, part1 %.1f, lm_head+read %.1f, embed %.2f, patch %.2f)\n",
                         core.position() - 1, t.total_ms, t.part0_ms, t.part1_ms, t.lmhead_ms, t.embed_ms, t.patch_ms);
        }
        tok = argmax(core.logits(), core.real_vocab());
    }
    auto stat = [](std::vector<double> v, const char* what) {
        std::sort(v.begin(), v.end());
        std::fprintf(stderr, "bench-verify %s: min %.1f ms, median %.1f ms over %zu\n", what, v.front(), v[v.size() / 2], v.size());
        std::printf("BENCH %s min %.2f median %.2f\n", what, v.front(), v[v.size() / 2]);
    };
    // alternating, as a loop with draftless passes does: each switch of hardware context costs
    std::vector<double> a2, a1;
    for (int r = 0; r < n; ++r) {
        const int ids[2] = {tok, tok};
        int am[2];
        core.verify(ids, 2, am);
        a2.push_back(core.last_timing().total_ms);
        core.accept(2);
        core.step(am[1], true);
        a1.push_back(core.last_timing().total_ms);
        tok = argmax(core.logits(), core.real_vocab());
    }
    stat(v2, "verify2");
    stat(v1, "step1");
    stat(a2, "verify2-after-step");
    stat(a1, "step1-after-verify");
    core.dump_step_trace("--bench-verify");
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
                     " | moe prep %.1f, patch %.1f, run %.1f, read %.1f | qkv scatter %.1f | residue %.1f\n",
                     tm.mid_ms, tm.dn_conv_ms, tm.dn_rule_ms, tm.attn_ms, tm.prenorm_ms, tm.gemm_tile_ms,
                     tm.gemm_tr_ms, tm.tail_ms, tm.shared_ms, tm.state_ms, tm.sync_ms, tm.setup_ms,
                     tm.moe_prep_ms, tm.moe_patch_ms, tm.moe_run_ms, tm.moe_read_ms,
                     tm.qkv_scatter_ms,
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
    if (a.verify_check > 0) {
        verify_check(core, a);
        return out;
    }
    if (a.bench_verify > 0) {
        bench_verify(core, a);
        return out;
    }
    if (a.verify_probe > 0) {
        // bring-up: layer acts after a step vs after a verify pass stopped before layer L (OFLM_OPEN_VERIFY_STOP=L)
        const int L = a.verify_probe;
        auto snap = core.checkpoint();
        const int tok = argmax(core.logits(), core.real_vocab());
        core.step(tok, false);
        std::vector<std::vector<uint8_t>> ref(L);
        auto bytes = [&](int l) { return static_cast<size_t>(core.manifest().layer_type(l).act_bytes); };
        for (int l = 0; l < L; ++l) { ref[l].resize(bytes(l)); core.read_act(l, 0, bytes(l), ref[l].data()); }
        core.restore(snap);
        const int ids[2] = {tok, tok};
        int am[2];
        core.verify(ids, 2, am);
        for (int l = 0; l < L; ++l) {
            std::vector<uint8_t> got(bytes(l));
            core.read_act(l, 0, bytes(l), got.data());
            size_t nd = 0, first = got.size(), last = 0;
            std::string granules;
            for (size_t o = 0; o < got.size(); ++o)
                if (got[o] != ref[l][o]) { ++nd; first = std::min(first, o); last = o; }
            for (size_t g = 0; g < got.size(); g += 4096) {
                bool d = std::memcmp(got.data() + g, ref[l].data() + g, std::min<size_t>(4096, got.size() - g)) != 0;
                granules += d ? 'x' : '.';
            }
            std::fprintf(stderr, "probe layer %d (%s): %zu bytes differ, first %zu last %zu\n  %s\n", l,
                         core.is_attention_layer(l) ? "full" : "linear", nd, first, last, granules.c_str());
        }
        std::printf("PROBE DONE\n");
        return out;
    }
    auto t1 = clock::now();
    core.take_dispatch_stats();   // prefill's dispatches are not this loop's
    int tok = argmax(core.logits(), core.real_vocab());
    if (a.spec) {
        out = decode_spec(core, a, tok);
        const double dec_ms = std::chrono::duration<double, std::milli>(clock::now() - t1).count();
        if (out.size() > 1)
            std::fprintf(stderr, "decode %zu tokens: %.0f ms/token (%.2f tok/s)\n", out.size() - 1, dec_ms / (out.size() - 1),
                         1000.0 * (out.size() - 1) / dec_ms);
        return out;
    }
    // Stage 2.7 Gate B: the per-step decode schedule ("6n,6h,6n"), the tool for
    // the one-process mid-stream switch test. Entries are (count, route) pairs
    // applied to the decode loop's steps in order; beyond the schedule the last
    // entry's route stays. Letters: n=npu, h=host, a=auto.
    std::vector<std::pair<int, std::string>> sched;
    if (!a.decode_schedule.empty()) {
        size_t at = 0;
        while (at <= a.decode_schedule.size()) {
            const size_t comma = a.decode_schedule.find(',', at);
            const std::string seg = a.decode_schedule.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
            const size_t x = seg.find_first_of("nha");
            if (x == std::string::npos || x == 0 || seg.find_first_of("nha", x + 1) != std::string::npos)
                { std::fprintf(stderr, "--decode-schedule wants <N><n|h|a> segments, got '%s'\n", seg.c_str()); std::exit(2); }
            sched.push_back({std::atoi(seg.substr(0, x).c_str()), seg.substr(x)});
            if (comma == std::string::npos) break;
            at = comma + 1;
        }
    }
    auto sched_route = [&sched](int n) -> const char* {
        auto full = [](const std::string& s) -> const char* {
            if (s == "n") return "npu";
            if (s == "h") return "host";
            if (s == "a") return "auto";
            return s.c_str();
        };
        int acc = 0;
        for (const auto& [cnt, r] : sched)
            if (n < (acc += cnt)) return full(r);
        return sched.empty() ? nullptr : full(sched.back().second);
    };
    for (int n = 0; n < a.max_tokens; ++n) {
        out.push_back(tok);
        std::printf("token %d\n", tok);
        std::fflush(stdout);
        if (n + 1 == a.max_tokens) break;
        if (const char* r = sched_route(n)) core.set_decode_route(r);
        const std::string rt = core.decode_route_at(core.position());
        core.step(tok, true);
        if (!a.dump_prefix.empty()) dump(a.dump_prefix, dumped++, core.logits());
        const auto& tm = core.last_timing();
        std::fprintf(stderr,
                     "  step @%d [%s]: %.1f ms (part0 %.1f, route %.2f, part1 %.1f, lm_head %.1f; embed %.2f, patch %.2f,"
                     " dispatches %.1f)\n",
                     core.position() - 1, rt.c_str(), tm.total_ms, tm.part0_ms, tm.route_ms, tm.part1_ms, tm.lmhead_ms,
                     tm.embed_ms, tm.patch_ms, tm.dispatch_ms);
        for (const auto& [kn, d] : core.take_dispatch_stats())
            std::fprintf(stderr, "      %-22s %4d calls %8.1f ms total %7.3f mean %7.3f min\n", kn.c_str(),
                         d.calls, d.ms, d.ms / d.calls, d.min_ms);
        tok = argmax(core.logits(), core.real_vocab());
    }
    double dec_ms = std::chrono::duration<double, std::milli>(clock::now() - t1).count();
    if (core.late_route_reads())
        std::fprintf(stderr, "late router reads caught so far: %llu\n",
                     static_cast<unsigned long long>(core.late_route_reads()));
    core.dump_step_trace("the decode loop");        // OFLM_OPEN_STEP_TRACE=1; silent otherwise
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
        if (a.spec) {
            const open_qwen36::spec::Availability av =
                open_qwen36::spec::spec_availability({{"speculative", {{"method", a.spec_method}}}}, &core.manifest());
            if (!av.ok()) {
                std::fprintf(stderr, "--spec: speculative decode is not available with %s: %s\n", a.cfg.kernel_dir.c_str(),
                             av.reason.c_str());
                return 2;
            }
        }
        core.load_weights();
        // Stage 2.7: the runtime decode route (env only set the initial one).
        // Stage 2.10 P6: auto is the default when nothing named a route --
        // NPU-first since the fast attention promotion -- but the 2.6 env's
        // initial host route still wins for the compat/debug path, so nothing
        // switches silently under an env user.
        if (!a.decode_route.empty()) core.set_decode_route(a.decode_route);
        else if (!std::getenv("OFLM_OPEN_HOST_ATTN_DECODE")) core.set_decode_route("auto");
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
            core.dump_step_trace("bench_decode's serial replays");   // and clears, so the next table is clean
            // and the number bench_decode's serial sum is the floor for: what a real step of
            // those same dispatches costs when the route issues them (OPEN-DECODE-PIPELINE).
            core.bench_step(a.bench_step ? a.bench_step : 10, a.ids[0]);
            core.dump_step_trace("--bench-step");
            std::printf("DONE\n");
            return 0;
        }
        if (a.bench_step) {
            if (a.at_position > 0) {
                std::fprintf(stderr, "seeking to position %d\n", a.at_position);
                core.seek(a.at_position);
            }
            core.step(a.ids[0], true);       // one real step first: the attnpos and route patches
            core.dump_step_trace("the warm-up step");
            core.bench_step(a.bench_step, a.ids[0]);
            core.dump_step_trace("--bench-step");
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
        if (a.verify_check > 0) return g_verify_bad ? 1 : 0;
        for (const std::string& d : a.dump_act) dump_act_slice(core, d);   // all from the same last step
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
