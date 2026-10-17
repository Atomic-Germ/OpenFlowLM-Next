// Traces: OPEN-SPEC-LOOKUP (canonical spec: specs/open-engine/spec.md)
// The engine's drafter against the hand-derived cases the Python drafter is tested on (tests/test_spec_lookup.py).
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "open_qwen36/spec_lookup.hpp"

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: spec_lookup_test <spec_lookup_cases.json>\n"); return 2; }
    std::ifstream f(argv[1]);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 2; }
    const auto j = nlohmann::json::parse(f);
    int bad = 0, n = 0;
    std::vector<int> drafts;
    for (const auto& c : j.at("cases")) {
        const auto hist = c.at("hist").get<std::vector<int>>();
        const int got = open_qwen36::lookup_drafts(hist.data(), hist.size(), c.at("k").get<size_t>(), c.at("nmax").get<int>(),
                                                   c.at("nmin").get<int>(), drafts, c.at("extend").get<bool>());
        const bool ok = got == c.at("n").get<int>() && drafts == c.at("drafts").get<std::vector<int>>();
        bad += !ok;
        ++n;
        std::printf("%s  %s\n", ok ? "ok  " : "FAIL", c.at("what").get<std::string>().c_str());
    }
    std::printf(bad ? "SPEC-LOOKUP FAIL (%d of %d)\n" : "SPEC-LOOKUP OK (%d of %d failed)\n", bad, n);
    return bad ? 1 : 0;
}
