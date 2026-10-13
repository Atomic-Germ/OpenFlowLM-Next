// Traces: OPEN-DECODE-SPECULATIVE-VERIFY (canonical spec: specs/open-engine/spec.md)
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "open_qwen36/verify_slots.hpp"
#include "stream_patch.hpp"

using open_qwen36::VerifySlots;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) std::printf("FAIL  %s\n", what.c_str());
    failures += !ok;
}

struct FakeEngine {
    VerifySlots vs;
    std::vector<int> slot_state;   // per slot: the state after this many positions (-1 = never written)
    int pos = 0;

    explicit FakeEngine(unsigned width) : slot_state(width + 1, -1) {
        vs.width = width;
        vs.reset();
        slot_state[0] = 0;
    }
    void step() {
        vs.drop();
        check(slot_state[vs.cur] == pos, "step reads the state after its position");
        slot_state[vs.cur] = ++pos;
    }
    void verify(size_t n) {
        vs.begin(pos, n);
        if (n == 1) {                  // a decode step
            check(slot_state[vs.cur] == pos, "1-position verify reads the committed state");
            slot_state[vs.cur] = pos + 1;
            pos += 1;
            return;
        }
        check(slot_state[vs.perm[0]] == pos, "the pass reads the committed state");
        for (size_t j = 0; j < n; ++j) {
            check(vs.perm[j + 1] != vs.cur, "a pass never writes the committed slot");
            check(slot_state[vs.perm[j]] == pos + static_cast<int>(j), "position j reads the state after j");
            slot_state[vs.perm[j + 1]] = pos + static_cast<int>(j) + 1;
        }
        pos += static_cast<int>(n);
    }
    void accept(size_t m) {
        const int p0 = vs.pos0;
        pos = vs.accept(m);
        check(pos == p0 + static_cast<int>(m), "accept(m) moves the position to pos0 + m");
        check(slot_state[vs.cur] == pos, "accept(m): the committed slot holds the state after pos0 + m");
    }
};

void bookkeeping(unsigned width, unsigned seed) {
    FakeEngine e(width);
    std::mt19937 rng(seed);
    for (int it = 0; it < 2000; ++it) {
        const unsigned r = rng() % 8;
        if (r == 0) {
            e.step();
        } else {
            const size_t n = 1 + rng() % width;
            e.verify(n);
            e.accept(1 + rng() % n);
        }
    }
    // a refused accept: none pending, or m out of range
    bool threw = false;
    try { e.vs.accept(1); } catch (const std::exception&) { threw = true; }
    check(threw, "accept with no pass pending is refused");
    e.verify(width);
    threw = false;
    try { e.vs.accept(width + 1); } catch (const std::exception&) { threw = true; }
    check(threw, "accept(m > n) is refused");
    e.accept(width);
    // a step between the pass and the accept leaves nothing to accept
    e.verify(width);
    e.pos = e.vs.pos0;                 // the fake rolls its own position back, as Core::step would see it
    e.vs.drop();
    threw = false;
    try { e.vs.accept(1); } catch (const std::exception&) { threw = true; }
    check(threw, "a dropped pass cannot be accepted");
}

// 4-word header, then per patch a BD blockwrite (op 1) and a 0x81 DDR patch on its register
std::vector<uint32_t> stream(const std::vector<std::pair<uint32_t, uint32_t>>& patches) {
    std::vector<uint32_t> w = {0, 0, 0, 0};
    uint32_t reg = 0x1D000;
    for (const auto& [arg, off] : patches) {
        std::vector<uint32_t> bd(12, 0);
        bd[0] = 1;
        bd[2] = reg;
        bd[4] = 1024;                  // the BD length in words
        w.insert(w.end(), bd.begin(), bd.end());
        std::vector<uint32_t> p(12, 0);
        p[0] = 0x81;
        p[6] = reg + 4;
        p[8] = arg;
        p[10] = off | (arg >= 5 ? 0x80000000u : 0u);
        w.insert(w.end(), p.begin(), p.end());
        reg += 0x20;
    }
    return w;
}

void slot_patch() {
    const uint64_t S = 3211264;
    // a verify stream: reads slot 0, writes slots 1 and 2; a pool fill (arg 0) is left alone
    std::vector<uint32_t> w = stream({{3, 61440}, {0, 4096}, {3, static_cast<uint32_t>(S + 61440)},
                                      {3, static_cast<uint32_t>(2 * S + 128)}, {3, 0}});
    auto t = stream_patch::slot_table(w, "fake", 3, S, 3);
    check(t.size() == 4, "slot_table finds the four state transfers");
    const unsigned perm[3] = {2, 0, 1};
    stream_patch::slot_apply(w.data(), t, perm, S);
    check(w[t[0].word] == 2 * S + 61440, "slot 0 -> 2");
    check(w[t[1].word] == 61440, "slot 1 -> 0");
    check(w[t[2].word] == S + 128, "slot 2 -> 1, offset kept");
    check(w[t[3].word] == 2 * S, "slot 0 at offset 0 -> 2");
    bool threw = false;
    try { stream_patch::slot_table(stream({{3, static_cast<uint32_t>(3 * S)}}), "fake", 3, S, 3); }
    catch (const std::exception&) { threw = true; }
    check(threw, "a transfer past the last slot is refused");
}

void attnv_patch() {
    stream_patch::AttnGeometry g;
    g.kv_row = 4096;
    g.ptab_row = 2048;
    g.rb_win = 4;
    // position 0: row drain (row 1), record (ptab row 1), window; position 1: the same at row 2
    std::vector<uint32_t> w = stream({{3, 4096}, {5, 2048}, {3, 0}, {3, 8192}, {5, 4096}, {3, 0}});
    auto t = stream_patch::attnv_table(w, "fake", 2, g);
    check(t.size() == 8, "attnv_table: four patches (window length, window offset, row, record) per position");
    stream_patch::attnv_apply(w.data(), t, 513, g);
    for (const auto& p : t) {
        const uint64_t ps = 513 + p.q;
        const uint64_t nb = (ps + 3) / 4;
        const uint32_t v = w[p.word] & 0x7fffffffu;
        if (p.kind == 0) check(v == nb * 4 * 4096 / 4, "window length of position " + std::to_string(p.q));
        if (p.kind == 1) check(v == ps * 4096, "row drain of position " + std::to_string(p.q));
        if (p.kind == 2) check(v == ps * 2048 && (w[p.word] & 0x80000000u), "record of position " + std::to_string(p.q));
        if (p.kind == 3) check(v == 0, "window start of position " + std::to_string(p.q));
    }
}

}  // namespace

int main() {
    bookkeeping(2, 1);
    bookkeeping(2, 7);
    bookkeeping(4, 3);
    slot_patch();
    attnv_patch();
    std::printf(failures ? "verify_slots_test: %d FAILED\n" : "verify_slots_test: all passed\n", failures);
    return failures ? 1 : 0;
}
