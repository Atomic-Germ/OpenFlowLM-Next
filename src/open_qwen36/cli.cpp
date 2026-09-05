/// \file cli.cpp
/// \brief Drive the open Qwen3.6 engine without the FLM app: token ids in,
///        greedy token ids and logits out. The test surface for core.cpp and
///        the way to run the engine on a box where the app itself does not
///        build (this one: no Boost / vcpkg / tokenizers-cpp).
///
///   open_qwen36_cli --model <dir> --kernels <dir> --ids 1,2,3 [--max-tokens N]
///       [--layers N] [--max-ctx N] [--dump-logits <prefix>] [--twice]
///       [--at-position P] [--ids-file <path>]
///
/// The prompt ids are prefilled by sequential decode (logits skipped), then
/// greedy decode runs for --max-tokens. Each produced id is printed on its
/// own line as `token <id>` so a wrapper can detokenize (tools/chat.py).
/// --dump-logits writes `<prefix>_t<i>.bin` (f32[248320]) for every position
/// with logits, which compare_decode.py can score. --twice runs the whole
/// request a second time on the same resident engine (state reset check).
/// --at-position P first seeks to position P with no cache rows in between
/// (a capacity check: the attention window then spans P rows).
#include <chrono>
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

struct Args {
    CoreConfig cfg;
    std::vector<int> ids;
    int max_tokens = 16;
    std::string dump_prefix;
    bool twice = false;
    int repeat = 1;
    int at_position = 0;
};

Args parse(int argc, char** argv) {
    Args a;
    a.cfg.model_dir = std::getenv("FLM_MODEL_DIR") ? std::getenv("FLM_MODEL_DIR") : "";
    a.cfg.kernel_dir = std::getenv("FLM_OPEN_KERNELS_DIR") ? std::getenv("FLM_OPEN_KERNELS_DIR") : "";
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
        else if (k == "--quiet") a.cfg.verbose = false;
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); std::exit(2); }
    }
    if (a.cfg.model_dir.empty() || a.cfg.kernel_dir.empty() || a.ids.empty()) {
        std::fprintf(stderr, "usage: open_qwen36_cli --model <dir> --kernels <dir> --ids 1,2,3 [--max-tokens N] "
                             "[--layers N] [--max-ctx N] [--dump-logits <prefix>] [--twice] [--at-position P]\n");
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
    for (size_t i = 0; i < a.ids.size(); ++i) {
        bool last = i + 1 == a.ids.size();
        core.step(a.ids[i], last);
        if (!a.dump_prefix.empty() && last) dump(a.dump_prefix, dumped++, core.logits());
    }
    double prefill_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    std::fprintf(stderr, "prefill %zu tokens: %.0f ms (%.0f ms/token)\n", a.ids.size(), prefill_ms, prefill_ms / a.ids.size());

    std::vector<int> out;
    auto t1 = clock::now();
    int tok = argmax(core.logits(), open_qwen36::layout::REAL_VOCAB);
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
        tok = argmax(core.logits(), open_qwen36::layout::REAL_VOCAB);
    }
    double dec_ms = std::chrono::duration<double, std::milli>(clock::now() - t1).count();
    if (out.size() > 1)
        std::fprintf(stderr, "decode %zu tokens: %.0f ms/token (%.2f tok/s)\n", out.size() - 1, dec_ms / (out.size() - 1),
                     1000.0 * (out.size() - 1) / dec_ms);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    try {
        auto t0 = std::chrono::steady_clock::now();
        Core core(a.cfg);
        core.load_weights();
        std::fprintf(stderr, "resident after %.1f s\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        std::vector<int> first = request(core, a);
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
