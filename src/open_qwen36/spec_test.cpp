// Traces: OPEN-DECODE-SPECULATIVE (canonical spec: specs/open-engine/spec.md)
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
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

    a = spec_availability({{"speculative", {{"method", "mtp"}}}}, &with_verify);
    check(a.status == Availability::kError &&
              a.reason == "unknown speculative method \"mtp\" (this build knows: lookup, dflash2)",
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

// a drafter container and kernel set on disk, as oflm-add leaves them: <root>/<name>/{config.json,open_kernels/manifest.json}
std::string make_draft(const std::filesystem::path& root, const std::string& name, const nlohmann::json& cfg, const std::string& family) {
    namespace fs = std::filesystem;
    fs::create_directories(root / name / "open_kernels");
    std::ofstream(root / name / "config.json") << cfg.dump();
    std::ofstream(root / name / "open_kernels" / "manifest.json") << nlohmann::json{{"family", family}}.dump();
    return (root / name).string();
}

void draft_tests() {
    namespace fs = std::filesystem;
    using J = nlohmann::json;
    const fs::path root = fs::temp_directory_path() / "oflm_spec_test_draft";
    fs::remove_all(root);
    const J cfg = {{"model_type", "dflash2"}, {"hidden_size", 5120}, {"vocab_size", 248320}, {"target_layer_ids", {5, 19, 33, 47, 61}}};
    make_draft(root, "Bonsai2-drafter", cfg, "dflash2");
    J wrong_type = cfg;
    wrong_type["model_type"] = "qwen3";
    make_draft(root, "NotADrafter", wrong_type, "dflash2");
    make_draft(root, "NoKernels", cfg, "qwen3");

    DraftInfo d = resolve_draft("Bonsai2-drafter", root.string());
    check(!d.dir.empty() && d.hidden == 5120 && d.vocab == 248320 && d.tap_layers == std::vector<int>({5, 19, 33, 47, 61}) &&
              fs::path(d.kernel_dir) == root / "Bonsai2-drafter" / "open_kernels",
          "a drafter named beside the models resolves to its directory, geometry, taps and linked kernel set");
    d = resolve_draft((root / "Bonsai2-drafter").string(), "");
    check(!d.dir.empty(), "a drafter given as a directory resolves without a models root");
    d = resolve_draft("Bonsai2-drafter", root.string(), (root / "elsewhere").string());
    check(d.dir.empty() && d.why.find("no dflash2 kernel set") != std::string::npos, "a kernel override with no manifest -> why: \"" + d.why + "\"");
    d = resolve_draft("Missing", root.string());
    check(d.dir.empty() && d.why.find("is not installed") != std::string::npos, "a drafter that is not installed -> why: \"" + d.why + "\"");
    d = resolve_draft("NotADrafter", root.string());
    check(d.dir.empty() && d.why.find("not a DFlash 2 drafter container") != std::string::npos, "a container of another model type is refused");
    d = resolve_draft("NoKernels", root.string());
    check(d.dir.empty() && d.why.find("no dflash2 kernel set") != std::string::npos, "a drafter whose kernel set is another family is refused");

    // Manifest is move-only (the GGUF-direct view), so each variant is built fresh
    auto bonsai_like = [] {
        open_qwen36::Manifest m;
        m.verify.width = 4;
        m.hidden = 5120;
        m.vocab = 248320;
        m.layers.assign(64, "lin");
        return m;
    };
    open_qwen36::Manifest bonsai = bonsai_like();
    const DraftResolver res = [&](const std::string& n) { return resolve_draft(n, root.string()); };
    const J declared = {{"speculative", {{"method", "dflash2"}, {"draft", "Bonsai2-drafter"}}}};

    Availability a = spec_availability(declared, &bonsai, res);
    check(a.status == Availability::kAvailable && a.method == "dflash2" && !a.draft.dir.empty(),
          "dflash2 + verify + taps' geometry + the drafter installed -> available");
    a = spec_availability({{"speculative", {{"method", "dflash2"}}}}, &bonsai, res);
    check(a.status == Availability::kError && a.reason.find("needs a \"draft\"") != std::string::npos,
          "dflash2 without a draft -> error: \"" + a.reason + "\"");
    a = spec_availability({{"speculative", {{"method", "dflash2"}, {"draft", "Missing"}}}}, &bonsai, res);
    check(a.status == Availability::kUnavailable && a.reason.find("is not installed") != std::string::npos,
          "dflash2 with a drafter that is not installed -> unavailable: \"" + a.reason + "\"");
    open_qwen36::Manifest no_verify = bonsai_like();
    no_verify.verify.width = 0;
    a = spec_availability(declared, &no_verify, res);
    check(a.status == Availability::kUnavailable && a.reason.find("no verify program") != std::string::npos,
          "dflash2 on a kernel set without a verify program -> unavailable");
    open_qwen36::Manifest small = bonsai_like();
    small.hidden = 2560;
    a = spec_availability(declared, &small, res);
    check(a.status == Availability::kUnavailable && a.reason.find("trained for another target") != std::string::npos,
          "a drafter whose hidden size is not the target's -> unavailable: \"" + a.reason + "\"");
    open_qwen36::Manifest short_stack = bonsai_like();
    short_stack.layers.assign(40, "lin");
    a = spec_availability(declared, &short_stack, res);
    check(a.status == Availability::kUnavailable && a.reason.find("taps layer 47") != std::string::npos,
          "a drafter that taps a layer the target lacks -> unavailable: \"" + a.reason + "\"");
    a = spec_availability(declared, &bonsai);
    check(a.status == Availability::kUnavailable, "dflash2 from a caller that cannot look for a drafter -> unavailable");
    check(make_drafter("dflash2") == nullptr, "dflash2 is built by the engine (it needs the core), not make_drafter");
    fs::remove_all(root);
}

// a drafter that answers from a script and counts what it was asked
struct ScriptedDrafter : Drafter {
    std::vector<std::vector<int>> answers;
    size_t calls = 0, keeps = 0;
    explicit ScriptedDrafter(std::vector<std::vector<int>> a = {}) : answers(std::move(a)) {}
    std::vector<int> propose(const std::vector<int>&, int, size_t) override {
        if (calls < answers.size()) return answers[calls++];
        ++calls;
        return {};
    }
    void observe(const int*, size_t) override {}
    void keep_up(const std::vector<int>&) override { ++keeps; }
};

void stack_tests() {
    check(!Stacked::takes({}) && Stacked::takes({7}) && Stacked::takes({7, 8, 9}),
          "the stacking rule: a member that proposed nothing yields, one that proposed anything drafts");

    for (const bool eager : {false, true}) {
        auto a = std::make_unique<ScriptedDrafter>(std::vector<std::vector<int>>{{}, {5, 6}, {}});
        auto b = std::make_unique<ScriptedDrafter>(std::vector<std::vector<int>>{{9}, {}});
        ScriptedDrafter *pa = a.get(), *pb = b.get();
        Stacked s(eager);
        s.add("first", std::move(a));
        s.add("second", std::move(b));
        const std::vector<int> fed{1, 2, 3};
        const std::vector<int> d1 = s.propose(fed, 4, 3), d2 = s.propose(fed, 4, 3), d3 = s.propose(fed, 4, 3);
        check(d1 == std::vector<int>({9}), "first has nothing -> the second drafts");
        check(d2 == std::vector<int>({5, 6}), "first has a draft -> it drafts");
        check(pa->calls == 3 && pb->calls == 2, "the member behind one that drafted is not run: first 3 passes, second 2");
        check(d3.empty(), "nobody has a draft -> no draft");
        check(pb->keeps == (eager ? 1u : 0u), std::string(eager ? "eager: " : "deferred: ") +
                                                  "the later member catches up after an earlier one drafted: " + std::to_string(pb->keeps));
        check(pa->keeps == 0, "a member is never asked to catch up after its own draft");
        check(s.report(false) == "first 1 / second 1 / none 1 of 3 passes", "the mix: " + s.report(false));
        check(s.report() == "first 1 / second 1 / none 1 of 3 passes" && s.report() == "first 0 / second 0 / none 0 of 0 passes",
              "reading the mix starts the count over");
    }

    // the real thing: lookup in front, where the match length decides who drafts
    const std::vector<int> fed{7, 8, 1, 2, 9, 3, 4, 7};   // with next = 8 the suffix [7, 8] recurs, [4, 7, 8] does not
    for (const int min : {2, 3, 4}) {
        Stacked s;
        s.add("lookup", std::make_unique<PromptLookup>(min, 4));
        auto model = std::make_unique<ScriptedDrafter>(std::vector<std::vector<int>>{{40, 41, 42}});
        ScriptedDrafter* pm = model.get();
        s.add("model", std::move(model));
        const std::vector<int> d = s.propose(fed, 8, 3);
        if (min == 2)
            check(d == std::vector<int>({1, 2, 9}) && pm->calls == 0, "min match 2: a 2-token match drafts and the model is not run");
        else
            check(d == std::vector<int>({40, 41, 42}) && pm->calls == 1,
                  "min match " + std::to_string(min) + ": no match that long -> the model drafts");
    }
}

void list_tests() {
    using J = nlohmann::json;
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "oflm_spec_test_list";
    fs::remove_all(root);
    const J cfg = {{"model_type", "dflash2"}, {"hidden_size", 5120}, {"vocab_size", 248320}, {"target_layer_ids", {5, 19, 33, 47, 61}}};
    make_draft(root, "Drafter", cfg, "dflash2");
    auto bonsai_like = [] {
        open_qwen36::Manifest b;
        b.verify.width = 4;
        b.hidden = 5120;
        b.vocab = 248320;
        b.layers.assign(64, "lin");
        return b;
    };
    open_qwen36::Manifest m = bonsai_like();
    const DraftResolver res = [&](const std::string& n) { return resolve_draft(n, root.string()); };
    const J lookup = {{"method", "lookup"}}, dflash2 = {{"method", "dflash2"}, {"draft", "Drafter"}};

    Availability a = spec_availability({{"speculative", J::array({lookup, dflash2})}}, &m, res);
    check(a.ok() && a.method == "lookup,dflash2" && a.members.size() == 2 && a.members[0].method == "lookup" &&
              a.members[1].method == "dflash2" && !a.draft.dir.empty(),
          "a list of lookup then dflash2 -> available, in that order: " + a.method + " " + a.reason);
    a = spec_availability({{"speculative", J::array({lookup})}}, &m, res);
    check(a.ok() && a.method == "lookup", "a list of one is the single object");
    a = spec_availability({{"speculative", lookup}}, &m, res);
    check(a.ok() && a.method == "lookup" && a.members.size() == 1 && a.members[0].min_match == 0,
          "the single-object form of PR 201 still means a list of one");
    a = spec_availability({{"speculative", J::array({J{{"method", "lookup"}, {"min_match", 3}}, dflash2})}}, &m, res);
    check(a.ok() && a.members[0].min_match == 3 && a.members[1].min_match == 0, "lookup's min_match is carried");
    for (const J& bad : {J(0), J(9), J("4"), J(2.5)}) {
        a = spec_availability({{"speculative", J::array({J{{"method", "lookup"}, {"min_match", bad}}})}}, &m, res);
        check(a.status == Availability::kError && a.reason.find("min_match") != std::string::npos,
              "min_match " + bad.dump() + " -> error: " + a.reason);
    }
    a = spec_availability({{"speculative", J::array({lookup, J{{"method", "dflash2"}, {"draft", "Drafter"}, {"min_match", 3}}})}}, &m, res);
    check(a.status == Availability::kError && a.reason.find("min_match") != std::string::npos, "min_match on dflash2 -> error");
    a = spec_availability({{"speculative", J::array()}}, &m, res);
    check(a.status == Availability::kUnavailable && a.reason == "the registry entry declares no speculative method",
          "an empty list declares nothing");
    a = spec_availability({{"speculative", J::array({lookup, lookup})}}, &m, res);
    check(a.status == Availability::kError && a.reason.find("listed twice") != std::string::npos, "a method listed twice -> error");
    a = spec_availability({{"speculative", J::array({dflash2, lookup})}}, &m, res);
    check(a.status == Availability::kError && a.reason.find("always drafts") != std::string::npos,
          "lookup behind dflash2 would never run -> error: " + a.reason);
    a = spec_availability({{"speculative", J::array({lookup, J{{"method", "mtp"}}})}}, &m, res);
    check(a.status == Availability::kError && a.reason.find("unknown speculative method \"mtp\"") != std::string::npos,
          "an unknown method in a list -> error");
    a = spec_availability({{"speculative", J::array({lookup, J{{"method", "dflash2"}}})}}, &m, res);
    check(a.status == Availability::kError && a.reason.find("needs a \"draft\"") != std::string::npos,
          "dflash2 without a draft in a list -> error");
    a = spec_availability({{"speculative", J::array({lookup, J{{"method", "dflash2"}, {"draft", "Missing"}}})}}, &m, res);
    check(a.status == Availability::kUnavailable && a.reason.find("is not installed") != std::string::npos && a.method == "lookup,dflash2",
          "one listed drafter that is not installed makes the whole stack unavailable: " + a.reason);
    open_qwen36::Manifest nov = bonsai_like();
    nov.verify.width = 0;
    a = spec_availability({{"speculative", J::array({lookup, dflash2})}}, &nov, res);
    check(a.status == Availability::kUnavailable && a.reason == "the kernel set has no verify program, which method \"lookup\" needs",
          "no verify program -> unavailable, naming the first method: " + a.reason);
    a = spec_availability({{"speculative", J::array({lookup, dflash2})}}, nullptr, res);
    check(a.status == Availability::kUnavailable && a.reason == "no open kernel set was found", "a list with no kernel set -> unavailable");
    a = spec_availability({{"speculative", 7}}, &m, res);
    check(a.status == Availability::kError, "a `speculative` that is neither an object nor a list -> error");
    fs::remove_all(root);
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
    draft_tests();
    stack_tests();
    list_tests();
    loop_tests();
    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL OK", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
