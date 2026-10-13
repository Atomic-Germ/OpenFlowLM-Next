// Traces: OPEN-DECODE-SPECULATIVE (canonical spec: specs/open-engine/spec.md)
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_qwen36/spec.hpp"

using namespace open_qwen36::spec;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

std::string str(const std::vector<int>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s + "]";
}

// greedy mostly follows the last two tokens (lookups hit), sometimes the state or rows (drafts get rejected)
struct FakeSnap {
    int pos = 0;
    std::vector<uint64_t> states;
    std::vector<std::vector<int>> kv;
};

class FakeCore {
public:
    static constexpr size_t V = 61;
    static constexpr size_t CAP = 4096;
    FakeCore() { reset(); }
    void reset() {
        pos_ = 0;
        h_ = 0x9E3779B97F4A7C15ull;
        kv_.assign(CAP, -1);
        steps_ = 0;
    }
    int position() const { return pos_; }
    void seek(int p) { pos_ = p; }
    void step(int tok, bool want_logits) {
        kv_[pos_] = tok;
        h_ = (h_ ^ static_cast<uint64_t>(tok + 1)) * 0x100000001B3ull;
        h_ ^= h_ >> 29;
        ++pos_;
        ++steps_;
        if (want_logits) logits_ = compute();
    }
    const std::vector<float>& logits() const { return logits_; }
    FakeSnap checkpoint() const {
        FakeSnap s;
        s.pos = pos_;
        s.states = {h_};
        s.kv.push_back(std::vector<int>(kv_.begin(), kv_.begin() + pos_));
        return s;
    }
    void restore(const FakeSnap& s) {
        if (!s.kv.empty()) std::copy(s.kv[0].begin(), s.kv[0].end(), kv_.begin());
        h_ = s.states.at(0);
        pos_ = s.pos;
    }
    size_t verify_width() const { return 4; }
    void verify(const int* ids, size_t n, int* argmax_out) {
        pos0_ = pos_;
        hs_.clear();
        vlogits_.assign(n, {});
        for (size_t i = 0; i < n; ++i) {
            step(ids[i], true);
            hs_.push_back(h_);
            vlogits_[i] = logits_;
            argmax_out[i] = argmax(logits_.data(), V);
        }
    }
    const std::vector<float>& verify_logits(size_t i) const { return vlogits_.at(i); }
    void accept(size_t m) {
        h_ = hs_.at(m - 1);
        pos_ = pos0_ + static_cast<int>(m);
    }
    uint64_t state() const { return h_; }
    size_t steps() const { return steps_; }

private:
    std::vector<float> compute() const {
        std::vector<float> lg(V);
        const int a = pos_ >= 1 ? kv_[pos_ - 1] : 0, b = pos_ >= 2 ? kv_[pos_ - 2] : 0;
        long rows = 0;
        for (int p = 0; p < pos_; ++p) rows += kv_[p];
        int win = (a * 7 + b * 3 + 1) % static_cast<int>(V);
        if (h_ % 17 == 0) win = static_cast<int>((h_ >> 11) % V);          // the recurrent state decides
        else if (rows % 29 == 0) win = (win + 5) % static_cast<int>(V);   // the attention rows decide
        for (size_t t = 0; t < V; ++t) lg[t] = static_cast<float>(((h_ >> (t % 40)) & 0xFF) / 512.0);
        lg[win] = 4.0f;
        if (h_ % 13 == 0) lg[(win + 1) % V] = 4.0f;   // an exact tie: the lower index must win
        return lg;
    }
    int pos_ = 0, pos0_ = 0;
    uint64_t h_ = 0;
    std::vector<int> kv_;
    std::vector<uint64_t> hs_;
    std::vector<std::vector<float>> vlogits_;
    std::vector<float> logits_;
    size_t steps_ = 0;
};


std::vector<int> plain(FakeCore& c, const std::vector<int>& prompt, size_t max_new, int stop) {
    c.reset();
    for (size_t i = 0; i < prompt.size(); ++i) c.step(prompt[i], i + 1 == prompt.size());
    std::vector<int> out;
    int tok = argmax(c.logits().data(), FakeCore::V);
    while (true) {
        out.push_back(tok);
        if (tok == stop || out.size() >= max_new) break;
        c.step(tok, true);
        tok = argmax(c.logits().data(), FakeCore::V);
    }
    return out;
}

std::vector<int> speculative(FakeCore& c, const std::vector<int>& prompt, size_t max_new, int stop, size_t width,
                             bool fast, Stats* st, std::vector<std::vector<int>>* passes = nullptr) {
    c.reset();
    for (size_t i = 0; i < prompt.size(); ++i) c.step(prompt[i], i + 1 == prompt.size());
    const std::unique_ptr<Drafter> drafter = make_drafter("lookup");
    std::vector<int> hist = prompt;
    std::vector<int> out;
    int tok = argmax(c.logits().data(), FakeCore::V);
    out.push_back(tok);
    ReferenceVerify<FakeCore> ref(c);
    CoreVerify<FakeCore> fv(c);
    while (tok != stop && out.size() < max_new) {
        std::vector<int> ids{tok};
        const size_t left = max_new - out.size();
        for (int d : drafter->propose(hist, tok, std::min(width - 1, left - 1))) ids.push_back(d);
        const size_t before = out.size();
        size_t m;
        auto is_stop = [&](int t) { return t == stop; };
        if (fast)
            m = run_pass(fv, ids, left,
                         [&](size_t i) {
                             const int t = argmax(fv.logits(i).data(), FakeCore::V);
                             return t == fv.argmax_at(i) ? t : -2;   // the two must agree
                         },
                         is_stop, out);
        else m = run_pass(ref, ids, left, [&](size_t i) { return argmax(ref.logits(i).data(), FakeCore::V); }, is_stop, out);
        if (st) st->add(ids.size(), m);
        if (passes) passes->emplace_back(out.end() - static_cast<long>(m), out.end());
        drafter->observe(ids.data(), m);
        hist.insert(hist.end(), ids.begin(), ids.begin() + static_cast<long>(m));
        tok = out.back();
        if (out.size() != before + m) return {};   // run_pass's count disagrees with what it appended
    }
    return out;
}

void drafter_tests() {
    auto d = [](std::vector<int> ctx, size_t k) {
        PromptLookup p;
        p.assign(ctx.data(), ctx.size());
        return p.draft(k);
    };
    check(d({5, 6, 7, 8, 9, 5, 6}, 3) == std::vector<int>({7, 8, 9}), "a 2-token suffix seen once drafts what followed it");
    check(d({5, 6, 1, 5, 6, 2, 5, 6}, 3) == std::vector<int>({2, 5, 6}), "the most recent earlier occurrence wins");
    check(d({1, 5, 6, 7, 2, 5, 6, 8, 1, 5, 6}, 3) == std::vector<int>({7, 2, 5}), "the longest matching suffix wins over a more recent shorter one");
    check(d({4, 9, 4}, 3).empty(), "a 1-token match proposes nothing (suffix length >= 2)");
    check(d({3, 4, 3, 4}, 3) == std::vector<int>({3, 4}), "a draft is cut at the end of the context");
    check(d({1, 2, 3, 1, 2}, 0).empty(), "k = 0 proposes nothing");
    check(d({1, 2, 3, 4, 9, 1, 2, 3, 4}, 1) == std::vector<int>({9}), "k caps the draft");
    check(d({7, 1, 2, 3, 4, 5, 8, 2, 3, 4, 5}, 2) == std::vector<int>({8, 2}), "nmax = 4: a 4-token match is as long as it looks");
    // incremental pushes, sync() and a rebuild agree on every prefix of a long random sequence
    std::mt19937 rng(7);
    std::vector<int> seq(3000);
    for (auto& t : seq) t = static_cast<int>(rng() % 9);
    PromptLookup inc, syn;
    bool same = true;
    for (size_t i = 0; i < seq.size(); ++i) {
        inc.push(seq[i]);
        if (i % 97 == 0) syn.sync(std::vector<int>(seq.begin(), seq.begin() + static_cast<long>(i + 1)));
        if (i % 97 == 0 || i + 1 == seq.size()) {
            PromptLookup fresh;
            fresh.assign(seq.data(), i + 1);
            syn.sync(std::vector<int>(seq.begin(), seq.begin() + static_cast<long>(i + 1)));
            for (size_t k = 1; k <= 5; ++k)
                same = same && inc.draft(k) == fresh.draft(k) && syn.draft(k) == fresh.draft(k);
        }
    }
    // sync() to a history that is NOT an extension rebuilds
    syn.sync({1, 2, 3, 1, 2});
    same = same && syn.draft(3) == std::vector<int>({3, 1, 2});
    check(same, "incremental push, sync() and a rebuild give the same drafts");

    const std::unique_ptr<Drafter> lk = make_drafter("lookup");
    check(lk && lk->propose({5, 6, 7, 8, 9, 5}, 6, 3) == std::vector<int>({7, 8, 9}),
          "make_drafter(\"lookup\") proposes from the fed tokens plus the next one");
    check(make_drafter("mtp") == nullptr, "make_drafter of a method this build does not have gives nothing");
}

void availability_tests() {
    using J = nlohmann::json;
    open_qwen36::Manifest with_verify, without_verify;
    with_verify.verify.width = 2;
    const J lookup = {{"name", "Ternary-Bonsai-2-27B-NPU2"}, {"speculative", {{"method", "lookup"}}}};
    const J undeclared = {{"name", "Qwen3.5-9B-NPU2"}};

    Availability a = spec_availability(lookup, &with_verify);
    check(a.status == Availability::kAvailable && a.method == "lookup",
          "declared lookup + a kernel set with a verify program -> available, method lookup");

    a = spec_availability(undeclared, &with_verify);
    check(a.status == Availability::kUnavailable && a.reason == "the registry entry declares no speculative method",
          "no `speculative` field -> unavailable: \"" + a.reason + "\"");
    a = spec_availability({{"speculative", nullptr}}, &with_verify);
    check(a.status == Availability::kUnavailable, "`speculative: null` -> unavailable, as if absent");

    a = spec_availability({{"speculative", {{"method", "dflash2"}}}}, &with_verify);
    check(a.status == Availability::kError &&
              a.reason == "unknown speculative method \"dflash2\" (this build knows: lookup)",
          "an unknown method -> error listing the known ones: \"" + a.reason + "\"");
    a = spec_availability({{"speculative", {{"draft", "qwen3.5:0.8b"}}}}, &with_verify);
    check(a.status == Availability::kError && a.reason.find("lookup") != std::string::npos,
          "`speculative` without a method -> error listing the known ones");

    a = spec_availability(lookup, &without_verify);
    check(a.status == Availability::kUnavailable &&
              a.reason == "the kernel set has no verify program, which method \"lookup\" needs",
          "declared lookup + a kernel set without a verify program -> unavailable: \"" + a.reason + "\"");
    a = spec_availability(lookup, nullptr);
    check(a.status == Availability::kUnavailable && a.reason == "no open kernel set was found",
          "declared lookup + no kernel set -> unavailable");
}

void loop_tests() {
    FakeCore c;
    std::mt19937 rng(11);
    size_t cases = 0, total_drafted = 0, total_accepted = 0;
    bool all_same = true, all_state = true, all_fewer = true;
    for (int pi = 0; pi < 12; ++pi) {
        std::vector<int> prompt(3 + pi * 5);
        for (auto& t : prompt) t = static_cast<int>(rng() % FakeCore::V);
        for (size_t width : {2u, 3u, 4u}) {
            for (int fast = 0; fast < 2; ++fast) {
                const size_t N = 150;
                std::vector<int> ref = plain(c, prompt, N, -1);
                const uint64_t ref_state = c.state();
                const int ref_pos = c.position();
                Stats st;
                std::vector<int> got = speculative(c, prompt, N, -1, width, fast != 0, &st);
                ++cases;
                if (got != ref) {
                    all_same = false;
                    std::printf("     prompt %d width %zu %s: plain %s\n     spec  %s\n", pi, width, fast ? "fast" : "ref",
                                str(ref).c_str(), str(got).c_str());
                }
                // plain decode stepped every emitted token but the last; so must the engine now
                all_state = all_state && c.state() == ref_state && c.position() == ref_pos;
                total_drafted += st.drafted;
                total_accepted += st.accepted;
                all_fewer = all_fewer && st.emitted + 1 == N && (st.accepted == 0 || st.passes < st.emitted);
            }
        }
    }
    check(all_same, "speculative greedy == plain greedy, " + std::to_string(cases) + " requests (widths 2-4, reference and native verify)");
    check(all_state, "after the request the engine's position and recurrent state equal plain decode's");
    check(total_drafted > 100 && total_accepted > 100,
          "the requests exercise drafting (" + std::to_string(total_drafted) + " drafted passes, " +
              std::to_string(total_accepted) + " accepted draft tokens)");
    check(all_fewer, "the pass counts add up: accepted drafts mean fewer passes than tokens");

    // a stop token cuts a pass: output ends at the first stop, the state is plain's at that point
    bool stop_ok = true;
    size_t stops_hit = 0;
    for (int pi = 0; pi < 12; ++pi) {
        std::vector<int> prompt(4 + pi * 3);
        for (auto& t : prompt) t = static_cast<int>(rng() % FakeCore::V);
        std::vector<int> free_run = plain(c, prompt, 120, -1);
        // pick a stop token that appears mid-run, after drafts have had a chance to be accepted
        const int stop = free_run[40 + pi];
        for (int fast = 0; fast < 2; ++fast) {
            std::vector<int> ref = plain(c, prompt, 120, stop);
            const uint64_t s = c.state();
            const int p = c.position();
            std::vector<int> got = speculative(c, prompt, 120, stop, 4, fast != 0, nullptr);
            stop_ok = stop_ok && got == ref && got.back() == stop && c.state() == s && c.position() == p;
            stops_hit += got.size() < 120;
        }
    }
    check(stop_ok && stops_hit > 0, "a stop token ends the pass at the stop, with plain decode's state (" + std::to_string(stops_hit) + " runs)");

    // ... and when the stop is an ACCEPTED draft token in the middle of a pass
    bool mid_ok = true;
    size_t mid_cases = 0;
    for (int pi = 0; pi < 400 && mid_cases < 8; ++pi) {
        std::vector<int> prompt(6 + pi);
        for (auto& t : prompt) t = static_cast<int>(rng() % FakeCore::V);
        std::vector<std::vector<int>> passes;
        std::vector<int> run = speculative(c, prompt, 150, -1, 4, true, nullptr, &passes);
        size_t at = 1;   // run[0] came from prefill
        bool done = false;
        for (const auto& ps : passes) {
            for (size_t j = 0; j + 1 < ps.size() && !done; ++j) {
                const auto first = run.begin() + static_cast<long>(at + j);
                if (std::find(run.begin(), first, ps[j]) != first) continue;   // not its first occurrence
                const int stop = ps[j];
                for (int fast = 0; fast < 2; ++fast) {
                    std::vector<int> ref = plain(c, prompt, 150, stop);
                    const uint64_t s = c.state();
                    const int p = c.position();
                    std::vector<int> got = speculative(c, prompt, 150, stop, 4, fast != 0, nullptr);
                    mid_ok = mid_ok && got == ref && ref.size() == at + j + 1 && c.state() == s && c.position() == p;
                }
                ++mid_cases;
                done = true;
            }
            if (done) break;
            at += ps.size();
        }
    }
    check(mid_ok && mid_cases >= 4, "a stop token accepted mid-pass ends the pass there (" + std::to_string(mid_cases) + " cases)");

    // a length cap cuts a pass the same way, at every length around a pass boundary
    bool cap_ok = true;
    std::vector<int> prompt{1, 2, 3, 4, 5, 6, 1, 2, 3};
    for (size_t N = 1; N <= 40; ++N)
        for (int fast = 0; fast < 2; ++fast) {
            std::vector<int> ref = plain(c, prompt, N, -1);
            const uint64_t s = c.state();
            const int p = c.position();
            std::vector<int> got = speculative(c, prompt, N, -1, 4, fast != 0, nullptr);
            cap_ok = cap_ok && got == ref && got.size() == N && c.state() == s && c.position() == p;
        }
    check(cap_ok, "max_emit caps a pass: exactly N tokens, plain decode's state, N = 1..40");

    // the reference verifier spends extra steps; the native one exactly one per position
    FakeCore f;
    std::vector<int> pr(20);
    for (auto& t : pr) t = static_cast<int>(rng() % FakeCore::V);
    Stats st;
    speculative(f, pr, 200, -1, 4, true, &st);
    check(f.steps() == pr.size() + st.positions, "native verify: one step per position run (" + std::to_string(f.steps()) + ")");
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    drafter_tests();
    availability_tests();
    loop_tests();
    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL OK", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
