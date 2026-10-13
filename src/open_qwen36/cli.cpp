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
#include "open_qwen36/dflash2.hpp"
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
    int verify_n = 0;               // --verify-n n: the bench's positions a pass (default: the kernel set's width)
    int verify_probe = 0;           // --verify-probe L (with OFLM_OPEN_VERIFY_STOP=L): act of layers < L, verify vs step
    std::vector<int> taps;          // --taps L,L,...: keep the residual after these layers (OPEN-DECODE-TAPS)
    std::string taps_out;           // --taps-out PATH: the ring's rows after the run, f32 [rows][taps * hidden], PATH.txt says which
    std::string residual_out;       // --residual-out PATH: the decode buffer after every step(), f32 [steps][hidden]
    // OPEN-DRAFT-DFLASH2 gate: the drafter on dumped taps (tf3's <run>/<pid>.jsonl + <pid>.taps)
    std::string draft_dir, draft_kernels, draft_run, draft_out;
    std::vector<std::string> draft_pids;
    int draft_stride = 2, draft_max_steps = 0, draft_bench = 0;
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
        else if (k == "--taps") {
            std::string v = val();
            if (v == "dflash2") v = "5,19,33,47,61";
            for (size_t i = 0; i < v.size();) {
                const size_t c = v.find(',', i);
                a.taps.push_back(std::atoi(v.substr(i, c - i).c_str()));
                i = c == std::string::npos ? v.size() : c + 1;
            }
        } else if (k == "--taps-out") a.taps_out = val();
        else if (k == "--residual-out") a.residual_out = val();
        else if (k == "--bench-verify") a.bench_verify = std::atoi(val().c_str());
        else if (k == "--verify-n") a.verify_n = std::atoi(val().c_str());
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
        else if (k == "--draft") a.draft_dir = val();
        else if (k == "--draft-kernels") a.draft_kernels = val();
        else if (k == "--draft-run") a.draft_run = val();
        else if (k == "--draft-out") a.draft_out = val();
        else if (k == "--draft-stride") a.draft_stride = std::atoi(val().c_str());
        else if (k == "--draft-max-steps") a.draft_max_steps = std::atoi(val().c_str());
        else if (k == "--draft-bench") a.draft_bench = std::atoi(val().c_str());
        else if (k == "--draft-pids") {
            const std::string v = val();
            for (size_t i0 = 0; i0 < v.size();) {
                size_t j = v.find(',', i0);
                if (j == std::string::npos) j = v.size();
                if (j > i0) a.draft_pids.push_back(v.substr(i0, j - i0));
                i0 = j + 1;
            }
        }
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); std::exit(2); }
    }
    if (!a.draft_run.empty() && a.ids.empty()) a.ids = {0};
    if (a.cfg.model_dir.empty() || a.cfg.kernel_dir.empty() || a.ids.empty()) {
        std::fprintf(stderr, "usage: open_qwen36_cli --model <dir> --kernels <dir> --ids 1,2,3 [--max-tokens N] "
                             "[--layers N] [--max-ctx N] [--dump-logits <prefix>] [--twice] [--at-position P] "
                             "[--gemm-block] [--block-major] [--prefill-logits] [--bench N[:LAYER]] [--bench-decode N] "
                             "[--bench-step N] [--bench-kernel NAME:REPS[:LAYER]] [--pmode MODE] "
                             "[--spec] [--spec-width N] [--spec-ref] [--spec-log] [--spec-method M]\n");
        std::exit(2);
    }
    return a;
}

std::unique_ptr<open_qwen36::dflash2::Proposer> g_proposer;   // --spec-method dflash2: built once, after the weights load

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
    const std::unique_ptr<sp::Drafter> own = g_proposer ? nullptr : sp::make_drafter(a.spec_method);
    sp::Drafter* const drafter = g_proposer ? static_cast<sp::Drafter*>(g_proposer.get()) : own.get();
    const open_qwen36::dflash2::Proposer::Stats ps0 = g_proposer ? g_proposer->stats() : open_qwen36::dflash2::Proposer::Stats{};
    std::vector<size_t> accept_hist(W + 1, 0);
    double propose_ms = 0;
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
        const auto tq = clock::now();
        for (int d : drafter->propose(hist, tok, std::min(W - 1, left - 1))) ids.push_back(d);
        propose_ms += std::chrono::duration<double, std::milli>(clock::now() - tq).count();
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
        ++accept_hist[std::min(m, W)];
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
    std::fprintf(stderr, "spec: tokens per pass histogram:");
    for (size_t m = 1; m <= W; ++m) std::fprintf(stderr, " %zu:%zu", m, accept_hist[m]);
    std::fprintf(stderr, "\nspec: proposing %.0f ms in all (%.1f ms a pass)\n", propose_ms, st.passes ? propose_ms / st.passes : 0.0);
    if (g_proposer) {
        const auto& ps = g_proposer->stats();
        const double n = static_cast<double>(std::max<size_t>(1, ps.drafts - ps0.drafts));
        std::fprintf(stderr,
                     "spec: dflash2 %zu drafts (%zu skipped, %zu cache restarts), %zu cache rows; per draft: block pass %.1f ms,"
                     " cache rows %.1f ms (gathering the taps %.2f ms)\n",
                     ps.drafts - ps0.drafts, ps.skipped - ps0.skipped, ps.restarts - ps0.restarts, ps.rows - ps0.rows,
                     (ps.draft_ms - ps0.draft_ms) / n, (ps.inject_ms - ps0.inject_ms) / n, (ps.stage_ms - ps0.stage_ms) / n);
    }
    return out;
}

// OPEN-DECODE-SPECULATIVE-VERIFY gate: every fourth pass is a plain step so the decode stream reads the slot accept() left
int g_verify_bad = 0;

void verify_check(Core& core, const Args& a) {
    const int n = a.verify_check;
    const size_t V = core.vocab();
    auto snap = core.checkpoint();
    std::vector<int> tokens{argmax(core.logits(), core.real_vocab())};
    std::vector<std::vector<float>> ref, ref_taps;
    const size_t TW = core.tap_width();
    const int p_start = core.position();
    int tap_bad = 0;
    for (int i = 0; i < n; ++i) {
        core.step(tokens.back(), true);
        if (TW) {
            const float* t = core.tap_row(p_start + i);
            if (!t) { std::fprintf(stderr, "  TAPS missing after step at position %d\n", p_start + i); ++tap_bad; ref_taps.emplace_back(TW); }
            else ref_taps.emplace_back(t, t + TW);
        }
        ref.push_back(core.logits());
        tokens.push_back(argmax(core.logits(), core.real_vocab()));
    }
    core.restore(snap);
    const int p0 = core.position();
    int bad = tap_bad, passes = 0, checked = 0;
    double vms = 0, sms = 0;
    int nv = 0, ns = 0;
    long tap_rows = 0;
    auto cmp_taps = [&](int i, const char* what) {
        if (!TW) return;
        ++tap_rows;
        const float* t = core.tap_row(p0 + i);
        if (t && std::memcmp(t, ref_taps[i].data(), TW * 4) == 0) return;
        std::fprintf(stderr, "  TAPS MISMATCH %s position %d (+%d)%s\n", what, p0 + i, i, t ? "" : ": none held");
        ++bad;
    };
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
    // passes of 2 .. width positions in turn; every third one has a wrong draft at a rotating position, so accept()
    // takes 1 .. W positions and the next pass starts from each state slot
    const int VW = static_cast<int>(core.verify_width());
    int i = 0;
    for (int k = 0; i + 1 < n; ++k) {
        const int W = 2 + (k / 4) % (VW - 1);
        if (k % 4 == 3 || i + W > n) {             // a plain step after an accept, and the tail
            core.step(tokens[i], true);
            sms += core.last_timing().total_ms;
            ++ns;
            cmp(core.logits(), i, "step");
            cmp_taps(i, "step");
            ++i;
            continue;
        }
        const int wrong = k % 3 == 2 ? 1 + (k / 3) % (W - 1) : W;
        std::vector<int> ids(W), am(W, -1);
        for (int j = 0; j < W; ++j)
            ids[j] = j == wrong ? (tokens[i + j] + 1) % static_cast<int>(core.real_vocab()) : tokens[i + j];
        core.verify(ids.data(), W, am.data());
        vms += core.last_timing().total_ms;
        ++nv;
        ++passes;
        for (int j = 0; j < wrong; ++j) {
            char what[32];
            std::snprintf(what, sizeof what, "verify[%d]", j);
            cmp(core.verify_logits(j), i + j, what);
            cmp_taps(i + j, what);
            if (am[j] != tokens[i + j + 1]) {
                std::fprintf(stderr, "  ARGMAX position %d: verify %d, step %d\n", p0 + i + j, am[j], tokens[i + j + 1]);
                ++bad;
            }
        }
        core.accept(wrong);
        if (TW)
            for (int j = wrong; j < W; ++j)
                if (core.tap_row(p0 + i + j)) {
                    std::fprintf(stderr, "  TAPS held for dropped row %d of the pass at position %d\n", j, p0 + i);
                    ++bad;
                }
        i += wrong;
        if (core.position() != p0 + i) {
            std::fprintf(stderr, "  POSITION after accept: %d, want %d\n", core.position(), p0 + i);
            ++bad;
        }
    }
    std::fprintf(stderr, "verify check: %d positions from %d, %d verify passes (%.1f ms mean), %d steps (%.1f ms mean), "
                         "%d logit rows and %ld tap rows compared: %s (%d bad)\n",
                 i, p0, passes, nv ? vms / nv : 0.0, ns, ns ? sms / ns : 0.0, checked, tap_rows, bad ? "FAIL" : "PASS", bad);
    std::printf(bad ? "VERIFY CHECK FAIL\n" : "VERIFY CHECK PASS\n");
    g_verify_bad = bad;
}

// every position of each pass is accepted, so the position advances by the width
void bench_verify(Core& core, const Args& a) {
    const int n = a.bench_verify;
    const size_t W = a.verify_n > 0 ? static_cast<size_t>(a.verify_n) : core.verify_width();
    int tok = argmax(core.logits(), core.real_vocab());
    std::vector<double> v2, v1;
    std::vector<int> ids(W), am(W);
    for (int r = 0; r < n; ++r) {
        std::fill(ids.begin(), ids.end(), tok);
        core.verify(ids.data(), W, am.data());
        v2.push_back(core.last_timing().total_ms);
        {
            const auto& t = core.last_timing();
            std::fprintf(stderr, "  verify @%d: %.1f ms (layers blocked %.1f, tail %.1f, host read+argmax %.2f, embed %.2f, patch %.2f)\n",
                         core.position(), t.total_ms, t.part0_ms, t.part1_ms, t.route_ms, t.embed_ms, t.patch_ms);
        }
        core.accept(W);
        tok = am[W - 1];
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
        std::fill(ids.begin(), ids.end(), tok);
        core.verify(ids.data(), W, am.data());
        a2.push_back(core.last_timing().total_ms);
        core.accept(W);
        core.step(am[W - 1], true);
        a1.push_back(core.last_timing().total_ms);
        tok = argmax(core.logits(), core.real_vocab());
    }
    const std::string vn = "verify" + std::to_string(W);
    stat(v2, vn.c_str());
    stat(v1, "step1");
    stat(a2, (vn + "-after-step").c_str());
    stat(a1, "step1-after-verify");
    core.dump_step_trace("--bench-verify");
}

void append_residual(Core& core, const Args& a) {
    if (a.residual_out.empty()) return;
    std::vector<float> r(core.manifest().hidden);
    core.read_residual(r.data());
    std::FILE* f = std::fopen(a.residual_out.c_str(), core.position() == 1 ? "wb" : "ab");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", a.residual_out.c_str()); return; }
    std::fwrite(r.data(), 4, r.size(), f);
    std::fclose(f);
}

void dump_taps(Core& core, const Args& a) {
    const long hi = core.position();
    long lo = hi;
    while (lo > 0 && core.tap_row(lo - 1)) --lo;
    std::FILE* f = std::fopen(a.taps_out.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", a.taps_out.c_str()); return; }
    for (long p = lo; p < hi; ++p) std::fwrite(core.tap_row(p), 4, core.tap_width(), f);
    std::fclose(f);
    if (std::FILE* m = std::fopen((a.taps_out + ".txt").c_str(), "w")) {
        std::fprintf(m, "first_pos %ld rows %ld taps %zu hidden %zu\n", lo, hi - lo, core.tap_count(), core.manifest().hidden);
        std::fclose(m);
    }
    std::fprintf(stderr, "taps: positions [%ld, %ld) -> %s\n", lo, hi, a.taps_out.c_str());
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
        append_residual(core, a);
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
    for (int n = 0; n < a.max_tokens; ++n) {
        out.push_back(tok);
        std::printf("token %d\n", tok);
        std::fflush(stdout);
        if (n + 1 == a.max_tokens) break;
        core.step(tok, true);
        append_residual(core, a);
        if (!a.dump_prefix.empty()) dump(a.dump_prefix, dumped++, core.logits());
        const auto& tm = core.last_timing();
        std::fprintf(stderr,
                     "  step @%d: %.1f ms (part0 %.1f, route %.2f, part1 %.1f, lm_head %.1f; embed %.2f, patch %.2f,"
                     " dispatches %.1f)\n",
                     core.position() - 1, tm.total_ms, tm.part0_ms, tm.route_ms, tm.part1_ms, tm.lmhead_ms,
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

// OPEN-DRAFT-DFLASH2 gate: one 7-token walk per position on tf3's dumped taps; k = 3 and 7 are its prefixes
int draft_gate(Core& core, const Args& a) {
    namespace d2 = open_qwen36::dflash2;
    using nlohmann::json;
    d2::Drafter dr(a.draft_dir, a.draft_kernels, core, a.cfg.verbose);
    dr.load_weights();
    const size_t tw = dr.config().tap_width();
    std::ofstream out(a.draft_out.empty() ? "drafts.jsonl" : a.draft_out);
    std::vector<double> tot, gemv, head, attn, host, sel;
    for (const std::string& pid : a.draft_pids) {
        std::ifstream jf(a.draft_run + "/" + pid + ".jsonl");
        if (!jf) throw std::runtime_error("no " + a.draft_run + "/" + pid + ".jsonl");
        std::string line;
        std::getline(jf, line);
        const json hdr = json::parse(line);
        std::vector<int> g;
        while (std::getline(jf, line))
            if (!line.empty()) g.push_back(json::parse(line).at("g").get<int>());
        const std::vector<int> prompt = hdr.at("prompt_ids").get<std::vector<int>>();
        const long n0 = hdr.at("prompt_n").get<long>() - 1;
        std::ifstream tf(a.draft_run + "/" + pid + ".taps", std::ios::binary | std::ios::ate);
        if (!tf) throw std::runtime_error("no " + a.draft_run + "/" + pid + ".taps");
        std::vector<float> taps(static_cast<size_t>(tf.tellg()) / 4);
        tf.seekg(0);
        tf.read(reinterpret_cast<char*>(taps.data()), static_cast<std::streamsize>(taps.size() * 4));
        const long n_taps = static_cast<long>(taps.size() / tw);
        dr.reset();
        int done = 0;
        for (size_t s = 0; s < g.size(); s += static_cast<size_t>(std::max(1, a.draft_stride))) {
            if (a.draft_max_steps && done >= a.draft_max_steps) break;
            const long p0 = n0 + static_cast<long>(s);
            if (p0 > n_taps) break;
            if (dr.cached() < p0) dr.inject(&taps[static_cast<size_t>(dr.cached()) * tw], static_cast<size_t>(p0 - dr.cached()));
            const int anchor = s == 0 ? prompt.back() : g[s - 1];
            d2::Lattice lat;
            const std::vector<int> d = dr.draft(anchor, 7, &lat);
            const auto& t = dr.last_pass();
            tot.push_back(t.total_ms); gemv.push_back(t.gemv_ms); head.push_back(t.head_ms);
            attn.push_back(t.attn_ms); host.push_back(t.host_ms); sel.push_back(t.select_ms);
            out << json{{"pid", pid}, {"s", s}, {"p0", p0}, {"d", d}, {"m", d2::walk_margins(lat)},
                        {"c1", std::vector<int>(lat.cand.begin(), lat.cand.begin() + d2::kTopK)},
                        {"ms", t.total_ms}}.dump() << "\n";
            ++done;
        }
        out.flush();
        std::fprintf(stderr, "dflash2 gate: %s %d drafts\n", pid.c_str(), done);
        for (int r = 0; r < a.draft_bench; ++r) {
            const auto d = dr.draft(g.empty() ? prompt.back() : g[0], 7);
            const auto& t = dr.last_pass();
            std::fprintf(stderr, "bench pass %d at context %ld: total %.2f ms, gemv %.2f (head %.2f), attention %.2f, "
                                 "host around gemv %.2f (tables %.2f, y read-back %.2f, sum %.2f), selector %.2f, %d dispatches\n",
                         r, dr.cached(), t.total_ms, t.gemv_ms, t.head_ms, t.attn_ms, t.host_ms, t.prep_ms, t.sync_ms,
                         t.reduce_ms, t.select_ms, t.dispatches);
            std::string ops;
            for (const auto& [name, ms] : t.op_ms) ops += " " + name + " " + std::to_string(ms).substr(0, 5);
            std::fprintf(stderr, "  per op (ms):%s\n", ops.c_str());
            (void)d;
        }
        for (int r = 0; r < a.draft_bench && n_taps >= dr.cached() + 8; ++r) {
            const auto ti = std::chrono::steady_clock::now();
            dr.inject(&taps[static_cast<size_t>(dr.cached()) * tw], 8);
            std::fprintf(stderr, "bench inject of 8 rows: %.2f ms (fc + K|V for every layer)\n",
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ti).count());
        }
    }
    auto med = [](std::vector<double> v) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    std::fprintf(stderr, "dflash2 gate: %zu drafts, median pass %.2f ms (gemv %.2f, head %.2f, attention %.2f, "
                         "host around gemv %.2f, selector %.2f)\n",
                 tot.size(), med(tot), med(gemv), med(head), med(attn), med(host), med(sel));
    std::printf("DONE\n");
    return 0;
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
        open_qwen36::spec::Availability spec_av;
        if (a.spec) {
            nlohmann::json entry = {{"speculative", {{"method", a.spec_method}}}};
            if (!a.draft_dir.empty()) entry["speculative"]["draft"] = a.draft_dir;
            spec_av = open_qwen36::spec::spec_availability(
                entry, &core.manifest(),
                [&](const std::string& d) { return open_qwen36::spec::resolve_draft(d, "", a.draft_kernels); });
            if (!spec_av.ok()) {
                std::fprintf(stderr, "--spec: speculative decode is not available with %s: %s\n", a.cfg.kernel_dir.c_str(),
                             spec_av.reason.c_str());
                return 2;
            }
        }
        core.load_weights();
        if (!a.taps.empty()) core.set_taps(a.taps);
        if (a.spec && spec_av.method == "dflash2")
            g_proposer = std::make_unique<open_qwen36::dflash2::Proposer>(core, spec_av.draft.dir, spec_av.draft.kernel_dir,
                                                                           a.cfg.verbose);
        std::fprintf(stderr, "resident after %.1f s\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        if (!a.draft_run.empty()) return draft_gate(core, a);
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
        if (!a.taps_out.empty() && a.verify_check == 0) dump_taps(core, a);
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
