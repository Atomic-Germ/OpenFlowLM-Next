// Traces: OPEN-UNO-DECODE (canonical spec: specs/open-engine/spec.md)
#include "modules/sampler.hpp"
#include <iostream>
#include <vector>

static int failures = 0;

static int pick(std::vector<float> logits) {
    sampler_config cfg;
    cfg.top_k = 1;
    Sampler s(static_cast<int>(logits.size()), cfg);
    std::vector<bf16> x(logits.begin(), logits.end());
    buffer<bf16> b(x);
    return s.sample(b);
}

static void expect(const char* what, int got, int want) {
    if (got != want) {
        std::cerr << "FAIL " << what << ": got " << got << ", want " << want << "\n";
        ++failures;
    }
}

int main() {
    std::vector<float> v(32, -1.0f);
    v[1] = v[8] = 3.0f;
    expect("tie across lanes (ids 1 and 8)", pick(v), 1);

    v.assign(32, -1.0f);
    v[9] = v[2] = 3.0f;
    expect("tie, higher id in the lower lane (ids 2 and 9)", pick(v), 2);

    v.assign(35, -1.0f);
    v[5] = v[33] = 3.0f;
    expect("tie between the vector body and the tail", pick(v), 5);

    v.assign(32, -1.0f);
    v[17] = 4.0f;
    v[3] = 3.0f;
    expect("a strict maximum still wins", pick(v), 17);

    if (failures == 0) std::cout << "sampler_greedy: all checks passed\n";
    return failures ? 1 : 0;
}
