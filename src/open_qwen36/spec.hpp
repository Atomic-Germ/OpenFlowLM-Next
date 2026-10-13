// Header-only and XRT-free so spec_test.cpp can drive it against a fake engine (OPEN-DECODE-SPECULATIVE).
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"
#include "open_qwen36/manifest.hpp"

namespace open_qwen36 {
namespace spec {

// A draft source behind the verify pass; the registry entry's `speculative.method` picks one (make_drafter).
class Drafter {
public:
    virtual ~Drafter() = default;
    // up to k tokens to follow `next`; `fed` is every token the target has consumed, position i at index i
    virtual std::vector<int> propose(const std::vector<int>& fed, int next, size_t k) = 0;
    // the pass kept `kept[0..m)` (`next`, then the accepted draft): where a drafter with its own state advances it
    virtual void observe(const int* kept, size_t m) = 0;
};

class PromptLookup : public Drafter {
public:
    static constexpr int kMaxN = 8;

    explicit PromptLookup(int nmin = 2, int nmax = 4) : nmin_(std::max(1, nmin)), nmax_(std::min(kMaxN, std::max(nmin_, nmax))) {}

    void clear() {
        ctx_.clear();
        idx_.clear();
    }
    // an n-gram is indexed only once it has a successor, so a hit always has a token to draft
    void push(int tok) {
        ctx_.push_back(tok);
        const size_t L = ctx_.size() - 1;   // index of the new token = successor position
        for (int n = nmin_; n <= nmax_; ++n) {
            if (L < static_cast<size_t>(n)) break;
            idx_[key(L - n, n)] = L - n;
        }
    }
    void assign(const int* ids, size_t n) {
        clear();
        for (size_t i = 0; i < n; ++i) push(ids[i]);
    }
    // the prefix test is a full compare: cheap next to a step
    void sync(const std::vector<int>& history) {
        if (history.size() >= ctx_.size() && std::equal(ctx_.begin(), ctx_.end(), history.begin())) {
            for (size_t i = ctx_.size(); i < history.size(); ++i) push(history[i]);
            return;
        }
        assign(history.data(), history.size());
    }
    const std::vector<int>& context() const { return ctx_; }

    // the longest recurring suffix wins over a more recent shorter one
    std::vector<int> draft(size_t k) const {
        std::vector<int> d;
        if (k == 0) return d;
        const size_t L = ctx_.size();
        for (int n = nmax_; n >= nmin_; --n) {
            if (L < static_cast<size_t>(n)) continue;
            auto it = idx_.find(key(L - n, n));
            if (it == idx_.end()) continue;
            const size_t from = it->second + n;
            const size_t to = std::min(L, from + k);
            d.assign(ctx_.begin() + static_cast<long>(from), ctx_.begin() + static_cast<long>(to));
            return d;
        }
        return d;
    }

    std::vector<int> propose(const std::vector<int>& fed, int next, size_t k) override {
        sync(fed);
        push(next);   // `next` is fed by every pass, so the lookup may keep it
        return draft(k);
    }
    void observe(const int*, size_t) override {}   // the next propose() catches up from `fed`

private:
    struct Key {
        std::array<int32_t, kMaxN> t;
        int n;
        bool operator==(const Key& o) const { return n == o.n && t == o.t; }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const {
            uint64_t h = 1469598103934665603ull ^ static_cast<uint64_t>(k.n);
            for (int i = 0; i < k.n; ++i) h = (h ^ static_cast<uint32_t>(k.t[i])) * 1099511628211ull;
            return static_cast<size_t>(h);
        }
    };
    Key key(size_t start, int n) const {
        Key k{};
        k.t.fill(-1);
        k.n = n;
        for (int i = 0; i < n; ++i) k.t[i] = ctx_[start + i];
        return k;
    }

    int nmin_, nmax_;
    std::vector<int> ctx_;
    std::unordered_map<Key, size_t, KeyHash> idx_;   // n-gram -> most recent start with a successor
};

// pick may be the app's stateful sampler: call it in position order and only as far as the walk goes
template <class V, class Pick, class Stop>
size_t run_pass(V& v, const std::vector<int>& ids, size_t max_emit, Pick&& pick, Stop&& stop, std::vector<int>& out) {
    const size_t n = ids.size();
    v.verify(ids.data(), n);
    size_t m = 0;
    for (size_t i = 0; i < n; ++i) {
        const int t = pick(i);
        out.push_back(t);
        m = i + 1;
        if (stop(t) || m >= max_emit) break;
        if (i + 1 < n && t != ids[i + 1]) break;   // the draft is wrong from here on
    }
    v.accept(m);
    return m;
}

// lowest index wins a tie: the CLI's greedy rule
inline int argmax(const float* v, size_t n) {
    int best = 0;
    for (size_t i = 1; i < n; ++i)
        if (v[i] > v[best]) best = static_cast<int>(i);
    return best;
}

// The snapshot is taken at position 0 so it copies no KV rows; rows a rejected position wrote die on rewind.
template <class CoreT>
class ReferenceVerify {
public:
    explicit ReferenceVerify(CoreT& c) : c_(c) {}

    void verify(const int* ids, size_t n) {
        pos0_ = c_.position();
        ids_.assign(ids, ids + n);
        snapped_ = n > 1;
        if (snapped_) {
            c_.seek(0);
            snap_ = c_.checkpoint();
            c_.seek(pos0_);
            snap_.pos = pos0_;
        }
        if (logits_.size() < n) logits_.resize(n);
        for (size_t i = 0; i < n; ++i) {
            c_.step(ids[i], true);
            logits_[i] = c_.logits();
        }
    }
    const std::vector<float>& logits(size_t i) const { return logits_[i]; }
    void accept(size_t m) {
        if (m >= ids_.size()) return;
        c_.restore(snap_);
        for (size_t i = 0; i < m; ++i) c_.step(ids_[i], false);
    }

private:
    CoreT& c_;
    int pos0_ = 0;
    bool snapped_ = false;
    std::vector<int> ids_;
    std::vector<std::vector<float>> logits_;
    decltype(std::declval<CoreT&>().checkpoint()) snap_{};
};

template <class CoreT>
class CoreVerify {
public:
    explicit CoreVerify(CoreT& c) : c_(c) {}
    void verify(const int* ids, size_t n) {
        am_.assign(n, -1);
        c_.verify(ids, n, am_.data());
    }
    const std::vector<float>& logits(size_t i) const { return c_.verify_logits(i); }
    int argmax_at(size_t i) const { return am_[i]; }
    void accept(size_t m) { c_.accept(m); }

private:
    CoreT& c_;
    std::vector<int> am_;
};

struct Stats {
    size_t passes = 0;      // verify passes plus plain steps
    size_t drafted = 0;
    size_t positions = 0;
    size_t emitted = 0;
    size_t accepted = 0;
    void add(size_t n_ids, size_t m) {
        ++passes;
        drafted += n_ids > 1;
        positions += n_ids;
        emitted += m;
        accepted += m - 1;
    }
};

struct Method {
    const char* name;
    bool needs_verify;   // the kernel set's verify program (OPEN-DECODE-SPECULATIVE-VERIFY)
    bool needs_draft;    // a drafter container named by the entry's `draft` (OPEN-REGISTRY-SPECULATIVE)
};
inline const std::vector<Method>& known_methods() {
    static const std::vector<Method> m{{"lookup", true, false}, {"dflash2", true, true}};
    return m;
}

inline std::unique_ptr<Drafter> make_drafter(const std::string& method) {
    if (method == "lookup") return std::make_unique<PromptLookup>();
    return nullptr;
}

// A drafter container on disk: what spec_availability needs to know about it without loading weights.
struct DraftInfo {
    std::string dir, kernel_dir, why;    // `why` says what is missing when `dir` or `kernel_dir` is empty
    size_t hidden = 0, vocab = 0;
    std::vector<int> tap_layers;         // the target layers (0-based) whose output the drafter reads
};
using DraftResolver = std::function<DraftInfo(const std::string& draft)>;

// `draft` is a directory, or a name under `models_root` (the registry's `draft`); the kernel set is
// `kernel_override`, else <dir>/open_kernels (where oflm-add links it)
inline DraftInfo resolve_draft(const std::string& draft, const std::string& models_root, const std::string& kernel_override = "") {
    namespace fs = std::filesystem;
    DraftInfo d;
    fs::path dir(draft);
    if (!fs::is_directory(dir) && !models_root.empty()) dir = fs::path(models_root) / draft;
    if (!fs::is_directory(dir)) {
        d.why = "the drafter container \"" + draft + "\" is not installed (oflm add <model> --speculative dflash2 --draft-model <dir>)";
        return d;
    }
    std::ifstream f(dir / "config.json");
    const nlohmann::json c = f ? nlohmann::json::parse(f, nullptr, false) : nlohmann::json();
    if (!c.is_object() || c.value("model_type", "") != "dflash2" || !c.contains("target_layer_ids") ||
        !c["target_layer_ids"].is_array()) {
        d.why = dir.string() + " is not a DFlash 2 drafter container (config.json model_type)";
        return d;
    }
    d.hidden = c.value("hidden_size", size_t{0});
    d.vocab = c.value("vocab_size", size_t{0});
    d.tap_layers = c["target_layer_ids"].get<std::vector<int>>();
    const fs::path k = kernel_override.empty() ? dir / "open_kernels" : fs::path(kernel_override);
    std::ifstream mf(k / "manifest.json");
    const nlohmann::json m = mf ? nlohmann::json::parse(mf, nullptr, false) : nlohmann::json();
    if (!m.is_object() || m.value("family", "") != "dflash2") {
        d.why = "the drafter " + dir.filename().string() + " has no dflash2 kernel set (looked in " + k.string() + ")";
        return d;
    }
    d.dir = dir.string();
    d.kernel_dir = k.string();
    return d;
}

struct Availability {
    enum Status { kAvailable, kUnavailable, kError };
    Status status = kUnavailable;
    std::string method;
    std::string reason;
    DraftInfo draft;   // set when the method has a drafter container
    bool ok() const { return status == kAvailable; }
};

// `entry` is the model's registry entry; `manifest` its kernel set's, null when none was found;
// `resolve` finds the drafter container a method with one names
inline Availability spec_availability(const nlohmann::json& entry, const Manifest* manifest,
                                      const DraftResolver& resolve = nullptr) {
    Availability a;
    const auto it = entry.is_object() ? entry.find("speculative") : entry.end();
    if (!entry.is_object() || it == entry.end() || it->is_null()) {
        a.reason = "the registry entry declares no speculative method";
        return a;
    }
    std::string names;
    for (const Method& m : known_methods()) names += std::string(names.empty() ? "" : ", ") + m.name;
    if (!it->is_object() || !it->contains("method") || !(*it)["method"].is_string()) {
        a.status = Availability::kError;
        a.reason = "the registry entry's \"speculative\" has no \"method\" (this build knows: " + names + ")";
        return a;
    }
    a.method = (*it)["method"].get<std::string>();
    const Method* m = nullptr;
    for (const Method& k : known_methods())
        if (a.method == k.name) m = &k;
    if (!m) {
        a.status = Availability::kError;
        a.reason = "unknown speculative method \"" + a.method + "\" (this build knows: " + names + ")";
        return a;
    }
    if (!manifest) {
        a.reason = "no open kernel set was found";
        return a;
    }
    if (m->needs_verify && manifest->verify.width < 2) {
        a.reason = "the kernel set has no verify program, which method \"" + a.method + "\" needs";
        return a;
    }
    if (m->needs_draft) {
        if (!it->contains("draft") || !(*it)["draft"].is_string() || (*it)["draft"].get<std::string>().empty()) {
            a.status = Availability::kError;
            a.reason = "method \"" + a.method + "\" needs a \"draft\" naming the drafter container";
            return a;
        }
        if (!resolve) {
            a.reason = "method \"" + a.method + "\" needs a drafter container and this caller cannot look for one";
            return a;
        }
        a.draft = resolve((*it)["draft"].get<std::string>());
        if (a.draft.dir.empty()) {
            a.reason = a.draft.why;
            return a;
        }
        // the verify pass writes the taps; the drafter reads them, so its geometry must be the target's
        if (a.draft.hidden != manifest->hidden || a.draft.vocab != manifest->vocab) {
            a.reason = "the drafter " + (*it)["draft"].get<std::string>() + " was trained for another target (hidden " +
                       std::to_string(a.draft.hidden) + ", vocab " + std::to_string(a.draft.vocab) + "; this model has " +
                       std::to_string(manifest->hidden) + ", " + std::to_string(manifest->vocab) + ")";
            return a;
        }
        for (int l : a.draft.tap_layers)
            if (l < 0 || static_cast<size_t>(l) >= manifest->layers.size()) {
                a.reason = "the drafter taps layer " + std::to_string(l) + " but this model has " +
                           std::to_string(manifest->layers.size()) + " layers";
                return a;
            }
    }
    a.status = Availability::kAvailable;
    return a;
}

}  // namespace spec
}  // namespace open_qwen36
