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
    p.index_bands();
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

// block_host's hadamard_rows, the order the NPU preps and the block route use: stages h = 1.., then * 1/32
static void hadamard_ref(float* x, size_t K) {
    for (size_t b0 = 0; b0 + 1024 <= K; b0 += 1024) {
        float* v = x + b0;
        for (size_t h = 1; h < 1024; h <<= 1)
            for (size_t i = 0; i < 1024; i += 2 * h)
                for (size_t j = i; j < i + h; ++j) {
                    const float a = v[j], c = v[j + h];
                    v[j] = a + c;
                    v[j + h] = a - c;
                }
        for (size_t j = 0; j < 1024; ++j) v[j] *= 1.0f / 32.0f;
    }
}

static void hadamard_tests() {
    const size_t K = 3072;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd;
    std::vector<float> x(kRows * K), want, got(kRows * K);
    for (auto& v : x) v = nd(rng);
    want = x;
    for (size_t t = 0; t < kRows; ++t) hadamard_ref(&want[t * K], K);
    hadamard8(x.data(), K, got.data());
    CHECK(std::memcmp(want.data(), got.data(), want.size() * sizeof(float)) == 0, "hadamard8 is not bit-identical to the scalar transform");
}

static void silu_tests() {
    const size_t n = 4096;
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 6.f);
    std::vector<float> g(n), u(n), out(n);
    for (size_t i = 0; i < n; ++i) { g[i] = nd(rng); u[i] = nd(rng); }
    g[0] = -100.f; g[1] = 100.f; g[2] = 0.f;
    silu_mul(g.data(), u.data(), n, out.data());
    double worst = 0;
    for (size_t i = 0; i < n; ++i) {
        const double want = static_cast<double>(g[i]) / (1.0 + std::exp(-static_cast<double>(g[i]))) * u[i];
        worst = std::max(worst, std::fabs(out[i] - want) / (std::fabs(want) + 1e-6));
    }
    CHECK(worst < 1e-5, "silu_mul worst relative error %.3g", worst);
}

// the attention as it was before the key rows were shared between queries: one (row, head) at a time
static void attention_one_by_one(const float* q, size_t rows, size_t nh, size_t kvh, size_t hd, const float* kc, const float* vc,
                                 size_t L, long ctx_pos0, const float* kb, const float* vb, long p0, size_t window, float* out) {
    const size_t grp = nh / kvh, kvw = kvh * hd;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    auto dot = [](const float* a, const float* b, size_t n) {
        float s = 0;
        for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
        return s;
    };
    for (size_t g = 0; g < kvh; ++g)
        for (size_t t = 0; t < rows; ++t) {
            const long P = p0 + static_cast<long>(t);
            long lo = 0;
            if (window) lo = std::min<long>(static_cast<long>(L), std::max<long>(0, P - static_cast<long>(window) + 1 - ctx_pos0));
            const size_t nk = (L > static_cast<size_t>(lo) ? L - lo : 0) + rows;
            std::vector<double> sc(nk);
            for (size_t hl = 0; hl < grp; ++hl) {
                const size_t h = g * grp + hl;
                const float* qv = q + (t * nh + h) * hd;
                double mx = -1e300;
                for (size_t j = 0; j < nk; ++j) {
                    const float* kv = j + lo < L ? kc + (j + lo) * kvw + g * hd : kb + (j + lo - L) * kvw + g * hd;
                    sc[j] = static_cast<double>(dot(qv, kv, hd)) * scale;
                    mx = std::max(mx, sc[j]);
                }
                double sum = 0;
                for (size_t j = 0; j < nk; ++j) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
                float* o = out + (t * nh + h) * hd;
                for (size_t d = 0; d < hd; ++d) o[d] = 0;
                std::vector<double> acc(hd, 0.0);
                for (size_t j = 0; j < nk; ++j) {
                    const float* vv = j + lo < L ? vc + (j + lo) * kvw + g * hd : vb + (j + lo - L) * kvw + g * hd;
                    for (size_t d = 0; d < hd; ++d) acc[d] += sc[j] / sum * vv[d];
                }
                for (size_t d = 0; d < hd; ++d) o[d] = static_cast<float>(acc[d]);
            }
        }
}

static void attention_tests() {
    std::mt19937 rng(21);
    std::normal_distribution<float> nd(0.f, 1.f);
    const size_t nh = 8, kvh = 2, hd = 32;
    struct Case { size_t rows, L; long ctx0; long p0; size_t window; };
    const Case cases[] = {{8, 0, 0, 0, 0}, {8, 5, 0, 5, 0}, {8, 100, 0, 100, 0}, {8, 100, 0, 100, 64}, {8, 300, 52, 352, 128},
                          {4, 40, 0, 40, 16}, {7, 33, 10, 43, 20}, {1, 17, 0, 17, 0}, {8, 64, 0, 64, 8}};
    for (const Case& c : cases) {
        std::vector<float> q(c.rows * nh * hd), kc(c.L * kvh * hd), vc(c.L * kvh * hd), kb(c.rows * kvh * hd), vb(c.rows * kvh * hd);
        for (auto* v : {&q, &kc, &vc, &kb, &vb})
            for (auto& x : *v) x = nd(rng);
        std::vector<float> got(c.rows * nh * hd), want(got.size());
        attention(q.data(), c.rows, nh, kvh, hd, kc.data(), vc.data(), c.L, c.ctx0, kb.data(), vb.data(), c.p0, c.window, got.data());
        attention_one_by_one(q.data(), c.rows, nh, kvh, hd, kc.data(), vc.data(), c.L, c.ctx0, kb.data(), vb.data(), c.p0, c.window, want.data());
        double worst = 0, mag = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            worst = std::max(worst, static_cast<double>(std::fabs(got[i] - want[i])));
            mag = std::max(mag, static_cast<double>(std::fabs(want[i])));
        }
        CHECK(worst < 1e-5 * (mag + 1e-3), "attention rows %zu L %zu window %zu: worst error %.3g of %.3g", c.rows, c.L, c.window, worst, mag);
    }
}

static void cold_fill_tests() {
    bool deepen = true;
    ColdFill f;
    CHECK(f.window(2048, &deepen) == 2048 && !deepen, "cold 0: always the whole window");
    f.cold = 256;
    f.after = 3;
    CHECK(f.window(2048, &deepen) == 256 && !deepen, "a cold cache takes the last 256");
    CHECK(f.window(200, &deepen) == 200 && !deepen, "a window smaller than the cold fill is taken whole");
    f.drafted();
    f.drafted();
    CHECK(f.window(2048, &deepen) == 256 && !deepen, "two drafts of three: still small");
    f.drafted();
    CHECK(f.window(2048, &deepen) == 2048 && deepen, "the third draft: refill with the whole window");
    CHECK(f.window(2048, &deepen) == 2048 && !deepen, "the refill is asked for once");
    f.went_cold();
    CHECK(f.window(2048, &deepen) == 256 && !deepen && f.drafts == 0, "a new request is cold again");
}

static void sync_tests() {
    auto all = [](long) { return true; };
    std::vector<int> fed(100);
    for (size_t i = 0; i < fed.size(); ++i) fed[i] = static_cast<int>(i * 7 + 1);
    const std::vector<int> none;
    auto cache = [&](long from, long to) { return std::vector<int>(fed.begin() + from, fed.begin() + to); };

    SyncPlan p = plan_sync(none, 0, fed, 2048, all);
    CHECK(p.can_draft && p.restart && p.from == 0 && p.to == 100, "an empty cache takes every position");
    p = plan_sync(cache(0, 90), 0, fed, 2048, all);
    CHECK(p.can_draft && !p.restart && p.keep == 90 && p.from == 90 && p.to == 100, "a cache that is the history takes the new rows");
    p = plan_sync(cache(0, 100), 0, fed, 2048, all);
    CHECK(p.can_draft && !p.restart && p.keep == 100 && p.from == 100 && p.to == 100, "a cache that is up to date takes nothing");
    std::vector<int> other = cache(0, 100);
    other[60] = -5;
    p = plan_sync(other, 0, fed, 2048, all);
    CHECK(p.can_draft && !p.restart && p.keep == 60 && p.from == 60 && p.to == 100, "a different token at 60 rewinds the cache to 60");
    std::vector<int> shorter(fed.begin(), fed.begin() + 40);
    p = plan_sync(cache(0, 100), 0, shorter, 2048, all);
    CHECK(p.can_draft && !p.restart && p.keep == 40 && p.from == 40 && p.to == 40, "a request that rewound below the cache drops the rest");

    std::vector<int> big(5000);
    for (size_t i = 0; i < big.size(); ++i) big[i] = static_cast<int>(i % 977);
    p = plan_sync(none, 0, big, 2048, all);
    CHECK(p.can_draft && p.restart && p.from == 5000 - 2048 && p.to == 5000, "a long prompt injects the last window only");
    std::vector<int> c1000(big.begin(), big.begin() + 1000);
    p = plan_sync(c1000, 0, big, 2048, all);
    CHECK(p.can_draft && p.restart && p.from == 5000 - 2048, "a cache that ends before the window restarts at its start");
    std::vector<int> c4000(big.begin() + 1952, big.begin() + 4000);
    p = plan_sync(c4000, 1952, big, 2048, all);
    CHECK(p.can_draft && !p.restart && p.keep == 4000 && p.from == 4000 && p.to == 5000, "a cache inside the window is extended");

    p = plan_sync(cache(0, 90), 0, fed, 2048, [](long q) { return q != 95; });
    CHECK(p.can_draft && p.restart && p.from == 96 && p.to == 100, "a position the target lost restarts the cache after it");
    p = plan_sync(cache(0, 100), 0, fed, 2048, [](long q) { return q != 95; });
    CHECK(p.can_draft && !p.restart && p.from == 100, "a lost position below an up-to-date cache is not needed");
    p = plan_sync(none, 0, fed, 2048, [](long q) { return q != 99; });
    CHECK(!p.can_draft, "no taps for the last position and nothing cached: no draft");
    p = plan_sync(none, 0, std::vector<int>{}, 2048, all);
    CHECK(!p.can_draft, "no history: no draft");
    p = plan_sync(cache(10, 50), 10, fed, 2048, all);
    CHECK(p.can_draft && !p.restart && p.keep == 50 && p.from == 50, "a cache that begins late is extended from its end");
}

int main() {
    attention_tests();
    sync_tests();
    cold_fill_tests();
    silu_tests();
    hadamard_tests();
    walk_tests();
    top16_tests();
    conv_tests();
    table_tests();
    std::printf(g_bad ? "dflash2_test: %d FAILED\n" : "dflash2_test: all passed\n", g_bad);
    return g_bad ? 1 : 0;
}
