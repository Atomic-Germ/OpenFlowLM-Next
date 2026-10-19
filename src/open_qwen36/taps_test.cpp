// Traces: OPEN-DECODE-TAPS (canonical spec: specs/open-engine/spec.md)
#include <cstdio>
#include <string>
#include <vector>

#include "open_qwen36/tap_ring.hpp"

using open_qwen36::TapRing;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) std::printf("FAIL  %s\n", what.c_str());
    failures += !ok;
}

constexpr size_t kHidden = 6, kTaps = 3;

float val(long pos, size_t tap, size_t i, int gen) { return static_cast<float>(gen * 100000 + pos * 100 + static_cast<long>(tap) * 10 + static_cast<long>(i)) * 0.5f; }

// one pass of n positions from pos0, written tap by tap as a layer walk does
void run_pass(TapRing& r, long pos0, size_t n, int gen, bool verify) {
    r.begin(pos0, n);
    for (size_t t = 0; t < kTaps; ++t)
        for (size_t j = 0; j < n; ++j) {
            std::vector<float> row(kHidden);
            for (size_t i = 0; i < kHidden; ++i) row[i] = val(pos0 + static_cast<long>(j), t, i, gen);
            r.put(t, pos0 + static_cast<long>(j), row.data());
        }
    r.end(pos0, n, verify);
}

bool holds(const TapRing& r, long pos, int gen) {
    const float* p = r.row(pos);
    if (!p) return false;
    for (size_t t = 0; t < kTaps; ++t)
        for (size_t i = 0; i < kHidden; ++i)
            if (p[t * kHidden + i] != val(pos, t, i, gen)) return false;
    return true;
}

}  // namespace

int main() {
    TapRing r;
    check(!r.on() && r.row(0) == nullptr, "an unconfigured ring is off and holds nothing");
    r.configure(0, kHidden, 8);
    check(!r.on(), "no taps is off whatever the window");

    r.configure(kTaps, kHidden, 8);
    check(r.on() && r.width() == kTaps * kHidden, "a row is the taps back to back");

    // decode steps: the window slides, older positions drop out
    for (long p = 0; p < 20; ++p) run_pass(r, p, 1, 0, false);
    bool ok = true;
    for (long p = 12; p < 20; ++p) ok = ok && holds(r, p, 0);
    check(ok, "the last window positions of a step run hold their taps, exactly");
    ok = true;
    for (long p = 0; p < 12; ++p) ok = ok && r.row(p) == nullptr;
    check(ok, "positions older than the window hold nothing");
    check(r.row(20) == nullptr && r.row(-1) == nullptr, "a position never run holds nothing");

    // a verify pass of 4, accept(2): rows 2 and 3 are dead, rows 0 and 1 stay
    run_pass(r, 20, 4, 1, true);
    ok = true;
    for (long p = 20; p < 24; ++p) ok = ok && holds(r, p, 1);
    check(ok, "every row of a verify pass holds its taps");
    r.drop(2);
    check(holds(r, 20, 1) && holds(r, 21, 1), "accept(2) keeps the first two rows");
    check(r.row(22) == nullptr && r.row(23) == nullptr, "accept(2) drops the rest");
    r.drop(1);
    check(holds(r, 21, 1), "a second drop with no pass pending changes nothing");

    // the next pass rewrites position 22 on; the kept rows stay
    run_pass(r, 22, 3, 2, true);
    check(holds(r, 20, 1) && holds(r, 21, 1) && holds(r, 22, 2) && holds(r, 23, 2) && holds(r, 24, 2),
          "a pass after accept(2) holds its own rows and the kept ones");
    r.drop(3);
    check(holds(r, 24, 2), "accept(n) drops nothing");

    // a step after a verify pass: drop has no pass to act on
    run_pass(r, 25, 1, 3, false);
    r.drop(0);
    check(holds(r, 25, 3), "drop after a plain step is a no-op");

    // a pass that never ends (a dispatch timed out) leaves its rows empty and the others alone
    r.begin(26, 4);
    check(r.row(26) == nullptr && r.row(29) == nullptr && holds(r, 25, 3), "an unfinished pass holds nothing");
    r.clear();
    check(r.row(25) == nullptr, "clear empties the ring");

    // a rewind to 22 (Core::restore): the rows before it stay, from it on go
    TapRing w;
    w.configure(kTaps, kHidden, 8);
    for (long p = 18; p < 26; ++p) run_pass(w, p, 1, 6, false);
    w.truncate(22);
    check(holds(w, 18, 6) && holds(w, 21, 6) && w.row(22) == nullptr && w.row(25) == nullptr,
          "truncate(pos) drops the rows from pos on and keeps the ones before it");

    // block prefill of 20 positions from 5: only the last window (8) is written, like Core does
    TapRing q;
    q.configure(kTaps, kHidden, 8);
    for (long p = 0; p < 5; ++p) run_pass(q, p, 1, 0, false);
    const long pos0 = 5;
    const size_t N = 20, keep = 8;
    q.begin(pos0 + static_cast<long>(N - keep), keep);
    for (size_t t = 0; t < kTaps; ++t)
        for (size_t row = N - keep; row < N; ++row) {
            std::vector<float> v(kHidden);
            for (size_t i = 0; i < kHidden; ++i) v[i] = val(pos0 + static_cast<long>(row), t, i, 4);
            q.put(t, pos0 + static_cast<long>(row), v.data());
        }
    q.end(pos0 + static_cast<long>(N - keep), keep, false);
    ok = true;
    for (long p = 17; p < 25; ++p) ok = ok && holds(q, p, 4);
    check(ok, "a prefill longer than the window holds its last window");
    ok = true;
    for (long p = 0; p < 17; ++p) ok = ok && q.row(p) == nullptr;
    check(ok, "a prefill longer than the window evicts what the earlier positions held");

    // a prefill shorter than the window keeps the positions before it
    TapRing s;
    s.configure(kTaps, kHidden, 8);
    for (long p = 0; p < 3; ++p) run_pass(s, p, 1, 0, false);
    run_pass(s, 3, 4, 5, false);
    check(holds(s, 0, 0) && holds(s, 2, 0) && holds(s, 3, 5) && holds(s, 6, 5), "a short prefill keeps the earlier positions");

    std::printf(failures ? "OPEN-DECODE-TAPS FAILED (%d)\n" : "OPEN-DECODE-TAPS PASSED\n", failures);
    return failures ? 1 : 0;
}
