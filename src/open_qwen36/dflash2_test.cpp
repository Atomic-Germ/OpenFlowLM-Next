// OPEN-DRAFT-DFLASH2 unit test (no XRT): the selector walk, and gemv_t2n's host half against a CPU model of
// the kernel's contract (tables in, partial y out).
// Traces: OPEN-DRAFT-DFLASH2-WALK, OPEN-DRAFT-DFLASH2-TABLES
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "open_qwen36/dflash2_host.hpp"

using namespace open_qwen36::dflash2;

static int g_bad = 0;
#define CHECK(c, ...)                                  \
    do {                                               \
        if (!(c)) {                                    \
            std::fprintf(stderr, "FAIL %s:%d ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);         \
            std::fprintf(stderr, "\n");                \
            ++g_bad;                                   \
        }                                              \
    } while (0)

static float& S(Lattice& l, size_t i, size_t p, size_t s) { return l.score[(i * kTopK + p) * kTopK + s]; }

static void walk_tests() {
    Lattice l;
    l.rows = 3;
    l.cand.resize(3 * kTopK);
    l.logit.assign(3 * kTopK, 0.f);
    l.score.assign(3 * kTopK * kTopK, 0.f);
    for (size_t i = 0; i < 3; ++i)
        for (size_t s = 0; s < kTopK; ++s) l.cand[i * kTopK + s] = static_cast<int32_t>(1000 * (i + 1) + s);
    S(l, 0, 0, 3) = 5.f;         // row 0 reads predecessor column 0 only
    S(l, 0, 5, 7) = 9.f;
    S(l, 1, 3, 9) = 2.f;         // row 1 reads the column of the slot row 0 chose (3)
    S(l, 1, 0, 2) = 8.f;
    S(l, 2, 9, 4) = 1.f;         // a tie in row 2's column 9: the lower slot wins
    S(l, 2, 9, 11) = 1.f;
    const std::vector<int> d = walk(l);
    CHECK(d.size() == 3 && d[0] == 1003 && d[1] == 2009 && d[2] == 3004, "walk %d %d %d", d[0], d[1], d[2]);
    const std::vector<float> m = walk_margins(l);
    CHECK(m[0] == 5.f && m[1] == 2.f && m[2] == 0.f, "margins %g %g %g", m[0], m[1], m[2]);
    l.rows = 1;
    CHECK(walk(l) == std::vector<int>{1003}, "a one-row walk");
}

static void top16_tests() {
    std::vector<float> v(1000);
    for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>((i * 7919) % 1000);
    v[500] = v[17] = 2000.f;     // a tie at the top: the lower id first
    int32_t ids[kTopK];
    float vals[kTopK];
    top16(v.data(), v.size(), ids, vals);
    CHECK(ids[0] == 17 && ids[1] == 500, "top16 tie order %d %d", ids[0], ids[1]);
    for (size_t j = 1; j < kTopK; ++j) CHECK(vals[j] <= vals[j - 1], "top16 not descending at %zu", j);
    CHECK(vals[2] == 999.f && vals[kTopK - 1] == 986.f, "top16 values %g %g", vals[2], vals[kTopK - 1]);
}

static void conv_tests() {
    const size_t R = 3, n = 32, grp = 16, ng = n / grp;
    std::vector<float> x(R * n), dyn(R * 4 * ng), base(4 * n), y(R * n);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.1f * static_cast<float>(i % 7) - 0.3f;
    for (size_t i = 0; i < dyn.size(); ++i) dyn[i] = 0.01f * static_cast<float>(i);
    for (size_t i = 0; i < base.size(); ++i) base[i] = 1.0f + 0.001f * static_cast<float>(i);
    conv2(x.data(), R, n, dyn.data(), base.data(), 1, grp, y.data());
    for (size_t t = 0; t < R; ++t)
        for (size_t c = 0; c < n; ++c) {
            const float w0 = dyn[(t * 4 + 2) * ng + c / grp] + base[2 * n + c];
            const float w1 = dyn[(t * 4 + 3) * ng + c / grp] + base[3 * n + c];
            const float want = w0 * x[t * n + c] + (t ? w1 * x[(t - 1) * n + c] : 0.f);
            CHECK(std::fabs(y[t * n + c] - want) < 1e-6f, "conv2 t %zu c %zu", t, c);
        }
    std::vector<float> z = x;
    conv2(z.data(), R, n, dyn.data(), base.data(), 1, grp, z.data());
    CHECK(z == y, "conv2 in place");
}

// The kernel's contract (gemv_t2n.cc), on the CPU: D = sum(xi * (code - 1)) in int32, y += D 2^-shift * s.
static void emulate(const GemvPlan& p, const std::vector<uint8_t>& codes, const std::vector<float>& s,
                    const std::vector<uint8_t>& tab, std::vector<float>& y) {
    const size_t K = p.k;
    y.assign(p.y_floats, 0.f);
    for (const GemvItem& it : p.items) {
        const uint8_t* e = &tab[static_cast<size_t>(it.tab) * p.tab_bytes];
        int32_t hdr[2];
        std::memcpy(hdr, e, 8);
        CHECK(hdr[0] == it.nb && hdr[1] == it.nkt, "table header");
        for (int b = 0; b < it.nb; ++b)
            for (int kt = 0; kt < it.nkt; ++kt)
                for (size_t half = 0; half < 2; ++half)
                    for (size_t quad = 0; quad < 2; ++quad)
                        for (size_t g = 0; g < 2; ++g) {
                            const uint8_t* rec = e + kHdr + ((static_cast<size_t>(kt) * 2 + g) * 2 + quad) * kRec;
                            int32_t sh;
                            std::memcpy(&sh, rec + 1152, 4);
                            const int16_t* xi = reinterpret_cast<const int16_t*>(rec);
                            const int32_t* sums = reinterpret_cast<const int32_t*>(rec + 1024);
                            for (size_t j = 0; j < 4; ++j)
                                for (size_t tok = 0; tok < 4; ++tok)
                                    for (size_t pp = 0; pp < 8; ++pp) {
                                        const size_t row = static_cast<size_t>(it.b0 + b) * 64 + half * 32 + j * 8 + pp;
                                        const size_t k0 = (static_cast<size_t>(it.kt0 + kt) * 2 + g) * 128;
                                        int64_t acc = 0;
                                        for (size_t i = 0; i < 128; ++i)
                                            acc += static_cast<int64_t>(xi[((i / 8) * 4 + tok) * 8 + i % 8]) * codes[row * K + k0 + i];
                                        const int64_t D = acc - sums[tok * 8 + pp];
                                        const float sc = s[row * (K / 128) + k0 / 128];
                                        y[it.y + static_cast<size_t>(b) * kYBand + (((half * 2 + quad) * 4 + j) * 4 + tok) * 8 + pp] +=
                                            static_cast<float>(std::ldexp(static_cast<double>(D), -sh)) * sc;
                                    }
                        }
    }
}

static void table_tests() {
    const size_t N = 128, K = 2048;           // two bands, eight k-tiles in two slices of four
    GemvPlan p;
    p.n = N;
    p.k = K;
    p.tab_bytes = (kHdr + 4 * 4 * kRec + 2047) / 2048 * 2048;
    int tab = 0, y = 0;
    for (int slice = 0; slice < 2; ++slice)
        for (int b = 0; b < 2; ++b) {
            GemvItem it;
            it.core = tab;
            it.slice = slice;
            it.b0 = b;
            it.nb = 1;
            it.kt0 = 4 * slice;
            it.nkt = 4;
            it.tab = tab++;
            it.y = y;
            y += static_cast<int>(kYBand);
            p.items.push_back(it);
        }
    p.n_tabs = static_cast<size_t>(tab);
    p.y_floats = static_cast<size_t>(y);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd;
    std::vector<uint8_t> codes(N * K);
    std::vector<float> s(N * K / 128), x(kRows * K);
    for (auto& c : codes) c = static_cast<uint8_t>(rng() % 3);
    for (auto& v : s) v = 0.01f + 0.001f * static_cast<float>(rng() % 100);
    for (size_t t = 0; t < kRows; ++t)
        for (size_t i = 0; i < K; ++i) x[t * K + i] = nd(rng) * std::pow(10.f, static_cast<float>(t % 3) - 1.f);
    std::vector<uint8_t> rec(K / 256 * 4 * kRec), tb(p.n_tabs * p.tab_bytes, 0);
    build_records(x.data(), K, rec.data());
    build_tables(p, rec.data(), tb.data());
    std::vector<float> yk, out(kRows * N);
    emulate(p, codes, s, tb, yk);
    reduce_y(p, yk.data(), out.data());
    // the reference reads x as the records hold it: a quad shares one shift, so a row 100x smaller
    // than its quad's largest keeps ~8 bits, and that is the format, not the layout under test
    std::vector<double> xe(kRows * K);
    for (size_t grp = 0; grp < K / 128; ++grp)
        for (size_t quad = 0; quad < 2; ++quad) {
            const uint8_t* r = &rec[(grp * 2 + quad) * kRec];
            int32_t sh;
            std::memcpy(&sh, r + 1152, 4);
            const int16_t* xi = reinterpret_cast<const int16_t*>(r);
            for (size_t tok = 0; tok < 4; ++tok)
                for (size_t i = 0; i < 128; ++i) {
                    const size_t at = (quad * 4 + tok) * K + grp * 128 + i;
                    xe[at] = std::ldexp(static_cast<double>(xi[((i / 8) * 4 + tok) * 8 + i % 8]), -sh);
                    CHECK(std::fabs(xe[at] - x[at]) <= std::ldexp(0.5, -sh), "record value off by more than half a step");
                }
        }
    double worst = 0;
    for (size_t t = 0; t < kRows; ++t)
        for (size_t r = 0; r < N; ++r) {
            double ref = 0, mag = 0;
            for (size_t i = 0; i < K; ++i) {
                const double w = (static_cast<double>(codes[r * K + i]) - 1.0) * s[r * (K / 128) + i / 128];
                ref += w * xe[t * K + i];
                mag += std::fabs(w * xe[t * K + i]);
            }
            worst = std::max(worst, std::fabs(out[t * N + r] - ref) / (mag + 1e-30));
        }
    CHECK(worst < 1e-4, "tables + reduce against fp64: worst error %.3g of the row's magnitude", worst);
}

int main() {
    walk_tests();
    top16_tests();
    conv_tests();
    table_tests();
    std::printf(g_bad ? "dflash2_test: %d FAILED\n" : "dflash2_test: all passed\n", g_bad);
    return g_bad ? 1 : 0;
}
