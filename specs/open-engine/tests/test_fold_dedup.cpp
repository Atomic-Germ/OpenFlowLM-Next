#include "open_qwen36/prefill_precision.hpp"
#include <array>
#include <cassert>
#include <cstring>
#include <limits>
#include <string>

using namespace open_qwen36;
using namespace open_qwen36::prefill_precision;

struct Clock {
    using time_point = std::chrono::steady_clock::time_point;
    static inline int reads = 0;
    static time_point now() { return time_point(std::chrono::milliseconds(++reads)); }
};

static void equal(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    assert(a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

struct Result {
    std::vector<float> output;
    std::array<std::vector<float>, 4> inputs;
};

// Dispatch owns one mapped buffer, overwritten by every product. The fold must
// preserve all three copies and never write to this buffer, even after return.
static Result run(Workspace& ws, const std::vector<float>& x, size_t outputs,
                  bool candidate, bool cancellation, FoldPhaseTiming* timing = nullptr) {
    Result r;
    const auto unused_low = ws.low.low;
    std::array<std::vector<float>, 4> products;
    std::vector<float> mapped(outputs, -91), previous = mapped;
    int calls = 0;
    const float* low_pointer = nullptr;
    const float* out = ws.run<Clock>(x.data(), x.size(), outputs,
        [&](bool low_weight, const float* input) {
            equal(mapped, previous);
            const int p = calls++;
            assert(p < 4 && low_weight == (p >= 2));
            assert((p % 2 == 0) == (input == x.data()));
            if (p == 1) low_pointer = input;
            if (p == 3 && candidate) assert(input == low_pointer);
            r.inputs[p].assign(input, input + x.size());
            for (size_t i = 0; i < outputs; ++i) {
                if (cancellation) {
                    // Grouped gives zero; a left-associated four-term sum gives one.
                    const float terms[] = {1.e20f, 1.f, -1.e20f, 1.f};
                    mapped[i] = terms[p];
                } else {
                    float sum = 0;
                    for (size_t k = 0; k < x.size(); ++k) {
                        const float value = bf16_to_f32(f32_to_bf16(input[k]));
                        const float weight = float(int((k + i * 3) % 13) - 6)
                            * (low_weight ? 0.00390625f : 1.f);
                        sum += value * weight;
                    }
                    mapped[i] = sum;
                }
            }
            previous = mapped;
            products[p] = mapped;
            return mapped.data();
        }, timing, candidate);
    assert(calls == 4);
    if (candidate) equal(ws.low.low, unused_low);
    equal(mapped, previous);
    assert(out == ws.output.data() && out != mapped.data());
    std::vector<float> high(outputs), low(outputs), expected(outputs);
    for (size_t i = 0; i < outputs; ++i) {
        high[i] = products[0][i] + products[1][i];
        low[i] = products[2][i] + products[3][i];
        expected[i] = high[i] + low[i];
    }
    equal(ws.high.output, high);
    equal(ws.low.output, low);
    equal(ws.output, expected);
    r.output.assign(out, out + outputs);
    std::fill(mapped.begin(), mapped.end(), -123.f);
    equal(r.output, ws.output);
    equal(r.inputs[0], x);
    equal(r.inputs[2], x);
    equal(r.inputs[1], r.inputs[3]);
    for (size_t i = 0; i < x.size(); ++i)
        assert(r.inputs[1][i] == x[i] - bf16_to_f32(f32_to_bf16(x[i])));
    return r;
}

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string test = argv[1];
    if (test == "numeric") {
        Workspace control, candidate;
        // Full blocks, partial blocks and mixed changing input/output sizes reuse
        // the same workspaces, including switches back to the control route.
        for (size_t rows : {256, 1, 17, 255, 257, 2, 256}) {
            for (size_t k : {1, 7, 16}) {
                std::vector<float> x(rows * k);
                for (size_t i = 0; i < x.size(); ++i)
                    x[i] = float(int(i % 127) - 63) * 0.031371f;
                x[0] = -0.f;
                const float adversarial[] = {0.f, -0.f, 1.00390625f, -1.00390625f,
                    std::numeric_limits<float>::denorm_min(),
                    -std::numeric_limits<float>::denorm_min(),
                    bf16_to_f32(0x7f7f), -bf16_to_f32(0x7f7f)};
                for (size_t i = 0; i < x.size() && i < 8; ++i) x[i] = adversarial[i];
                for (bool cancellation : {false, true}) {
                    auto a = run(control, x, rows * 3, false, cancellation);
                    auto b = run(candidate, x, rows * 3, true, cancellation);
                    equal(a.output, b.output);
                    for (int p = 0; p < 4; ++p) equal(a.inputs[p], b.inputs[p]);
                    if (cancellation) for (float value : b.output) assert(value == 0.f);
                    equal(b.output, run(candidate, x, rows * 3, false, cancellation).output);
                }
            }
        }
    } else if (test == "validation") {
        for (bool candidate : {false, true}) {
            Workspace ws;
            for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                                  std::numeric_limits<float>::infinity(),
                                  -std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::max(),
                                  -std::numeric_limits<float>::max()}) {
                std::vector<float> x(17, 1.003f);
                x.back() = invalid;
                int calls = 0;
                bool rejected = false;
                try {
                    ws.run(x.data(), x.size(), 1,
                        [&](bool, const float*) { ++calls; return x.data(); }, nullptr, candidate);
                } catch (const std::runtime_error& e) {
                    rejected = std::string(e.what()) == "paired GEMM requires finite BF16-range inputs";
                }
                assert(rejected && calls == 0);
                run(ws, std::vector<float>(3, 1.003f), 2, candidate, true);
            }
        }
    } else if (test == "timing") {
        for (bool candidate : {false, true}) {
            Workspace ws;
            Clock::reads = 0;
            run(ws, std::vector<float>(17, 1.003f), 7, candidate, true);
            assert(Clock::reads == 0);
            FoldPhaseTiming timing;
            run(ws, std::vector<float>(17, 1.003f), 7, candidate, true, &timing);
            // Two clock reads per window: 3 copies, 3 sums, 1 or 2 splits.
            assert(Clock::reads == (candidate ? 14 : 16));
            assert(timing.calls == 1);
            assert(timing.split_ms == (candidate ? 1 : 2));
            assert(timing.copy_ms == 3 && timing.accumulate_ms == 3);
        }
    } else return 1;
}
