/// \file spec_lookup.hpp
/// \brief OPEN-SPEC-LOOKUP: prompt-lookup drafts for the L-row verify pass, and the per-cycle choice between a
///        lookup verify, Uno's two-pass cycle and a plain decode step. Token ids only; no XRT, no model.
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace open_qwen36 {

/// Up to `k` drafts for what follows hist[0, n): the tokens after the most recent earlier occurrence of the
/// longest suffix n-gram, n = nmax down to nmin. A continuation cut short by the end of the history is extended by
/// looking up again on the history plus the drafts so far (`extend`). Returns the n of the first match, 0 for none.
inline int lookup_drafts(const int* hist, size_t n, size_t k, int nmax, int nmin, std::vector<int>& out,
                         bool extend = true) {
    out.clear();
    if (nmin < 1) nmin = 1;
    auto at = [&](size_t i) { return i < n ? hist[i] : out[i - n]; };
    int first = 0;
    while (out.size() < k) {
        const size_t s = n + out.size();
        int got = 0;
        size_t from = 0;
        for (int g = nmax; g >= nmin && !got; --g) {
            const size_t gs = static_cast<size_t>(g);
            if (s < gs + 1) continue;
            for (size_t j = s - gs; j-- > 0;) {
                size_t i = 0;
                while (i < gs && at(j + i) == at(s - gs + i)) ++i;
                if (i == gs) { got = g; from = j + gs; break; }
            }
        }
        if (!got) break;
        if (!first) first = got;
        for (size_t c = from; c < s && out.size() < k; ++c) out.push_back(at(c));
        if (!extend) break;
    }
    return first;
}

enum class SpecMethod { plain = 0, lookup = 1, uno = 2 };

inline const char* spec_method_name(SpecMethod m) {
    return m == SpecMethod::plain ? "plain" : m == SpecMethod::lookup ? "lookup" : "uno";
}

/// The expected-gain rule: tokens per millisecond of each method from running means of what it committed and what
/// it cost here, a lookup's split by the n it matched. A method not run for a while drifts back to its prior so it is
/// tried again when the text changes.
struct SpecPolicy {
    enum class Mode { adaptive, lookup, lookup_uno, uno, plain };
    Mode mode = Mode::adaptive;
    double alpha = 0.3;                          ///< weight of the newest cycle in each running mean
    double revert = 0.1;                         ///< per cycle, how far an unused estimate moves back to its prior
    double prior_lookup[5] = {0, 0.3, 0.8, 1.0, 2.0};    ///< drafts accepted a lookup pass, by matched n
    double prior_uno = 1.0;                      ///< drafts accepted a Uno cycle (3.04 tokens a cycle at short context)
    double ratio_verify = 1.33, ratio_draft = 1.58;      ///< a pass's cost in plain steps until it has been timed
    double acc_lookup[5] = {0, 0.3, 0.8, 1.0, 2.0};
    double acc_uno = 1.0;
    double ms[3] = {0, 0, 0};                    ///< running mean ms: plain step, verify pass, draft pass
    bool timed[3] = {false, false, false};

    void reset_estimates() {
        std::copy(prior_lookup, prior_lookup + 5, acc_lookup);
        acc_uno = prior_uno;
        for (int i = 0; i < 3; ++i) { ms[i] = 0; timed[i] = false; }
    }
    double cost(int which) const {               // 0 plain step, 1 verify pass, 2 draft pass
        if (timed[which]) return ms[which];
        const double r[3] = {1.0, ratio_verify, ratio_draft};
        for (int i = 0; i < 3; ++i)
            if (timed[i]) return ms[i] / r[i] * r[which];
        return r[which];
    }
    double gain(SpecMethod m, int matched) const {   // committed tokens per ms
        if (m == SpecMethod::plain) return 1.0 / cost(0);
        if (m == SpecMethod::lookup) return (1.0 + acc_lookup[std::clamp(matched, 1, 4)]) / cost(1);
        return (2.0 + acc_uno) / (cost(2) + cost(1));
    }
    SpecMethod choose(int matched, bool lookup_room, bool uno_room) const {
        const bool can_lookup = matched > 0 && lookup_room, can_uno = uno_room;
        switch (mode) {
        case Mode::plain: return SpecMethod::plain;
        case Mode::uno: return can_uno ? SpecMethod::uno : SpecMethod::plain;
        case Mode::lookup: return can_lookup ? SpecMethod::lookup : SpecMethod::plain;
        case Mode::lookup_uno: return can_lookup ? SpecMethod::lookup : can_uno ? SpecMethod::uno : SpecMethod::plain;
        case Mode::adaptive: break;
        }
        SpecMethod best = SpecMethod::plain;
        double g = gain(SpecMethod::plain, 0);
        if (can_uno && gain(SpecMethod::uno, 0) > g) { best = SpecMethod::uno; g = gain(SpecMethod::uno, 0); }
        if (can_lookup && gain(SpecMethod::lookup, matched) > g) best = SpecMethod::lookup;
        return best;
    }
    /// After a cycle: `accepted` drafts it kept, the ms of its plain step / verify pass / draft pass (0 if none ran).
    void record(SpecMethod m, int matched, int accepted, double step_ms, double verify_ms, double draft_ms) {
        auto mix = [this](double& v, double x) { v += alpha * (x - v); };
        const double t[3] = {step_ms, verify_ms, draft_ms};
        for (int i = 0; i < 3; ++i)
            if (t[i] > 0) {
                if (timed[i]) mix(ms[i], t[i]);
                else { ms[i] = t[i]; timed[i] = true; }
            }
        for (int g = 1; g <= 4; ++g)
            if (!(m == SpecMethod::lookup && g == matched)) acc_lookup[g] += revert * (prior_lookup[g] - acc_lookup[g]);
        if (m != SpecMethod::uno) acc_uno += revert * (prior_uno - acc_uno);
        if (m == SpecMethod::lookup) mix(acc_lookup[std::clamp(matched, 1, 4)], accepted);
        if (m == SpecMethod::uno) mix(acc_uno, accepted);
    }
};

}  // namespace open_qwen36
