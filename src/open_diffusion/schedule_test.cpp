// Traces: OPEN-DIFFUSION-STEPS (canonical spec: specs/open-diffusion/spec.md)
//
// open_diffusion_schedule_test [MODEL_DIR]
//
// schedule.hpp makes the sigmas, timestep features and Euler dts for any step count. For
// the bundle's own count it must give the bytes export_bundle.py wrote (tf_<R>.bin,
// dt_<R>.bin, from klein_pipeline.py): the dts exactly, the features within half a bf16 step
// at 1.0, 2^-8 (schedule.hpp says why not exactly). A wrong feature or dt is silent: the image just
// drifts. Every resolution in bundle.json is checked. MODEL_DIR defaults to the
// installed model; the test fails, naming the path, when it is absent.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "schedule.hpp"

namespace fs = std::filesystem;
namespace sch = open_diffusion::schedule;

static std::vector<char> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return {};
    std::vector<char> d(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(d.data(), static_cast<std::streamsize>(d.size()));
    return d;
}

static float bf16_value(uint16_t w) {
    uint32_t u = static_cast<uint32_t>(w) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// tol 0: every word identical. Otherwise each word, read as a bf16 value, within tol of
// the bundle's (absolute: a cosine near 0 is many ulps from its neighbour).
static int compare(const char* what, const std::vector<uint16_t>& got, const std::vector<char>& want,
                   double tol = 0) {
    if (want.size() != got.size() * 2) {
        std::printf("FAIL  %s: %zu bytes, the bundle's has %zu\n", what, got.size() * 2, want.size());
        return 1;
    }
    size_t diff = 0, far = 0, first = 0;
    double worst = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        uint16_t w;
        std::memcpy(&w, want.data() + 2 * i, 2);
        if (w == got[i]) continue;
        ++diff;
        double d = std::fabs(static_cast<double>(bf16_value(w)) - bf16_value(got[i]));
        worst = std::max(worst, d);
        if ((tol == 0 || d > tol) && far++ == 0) first = i;
    }
    if (far) {
        std::printf("FAIL  %s: %zu of %zu words off by more than %g (first at %zu)\n", what, far,
                    got.size(), tol, first);
        return 1;
    }
    std::printf("ok    %s: %zu words, %zu differ, by at most %g\n", what, got.size(), diff, worst);
    return 0;
}

int main(int argc, char** argv) {
    fs::path model;
    if (argc > 1) {
        model = argv[1];
    } else {
        const char* root = std::getenv("OFLM_MODEL_PATH");
        const char* home = std::getenv("USERPROFILE");
        model = root ? fs::path(root) : fs::path(home ? home : ".") / ".oflm";
        model /= fs::path("models") / "FLUX.2-klein-4B-NPU2";
    }
    std::ifstream bf(model / "bundle.json");
    if (!bf) {
        std::fprintf(stderr, "FAIL: %s is missing (install flux2-klein:4b or pass MODEL_DIR)\n",
                     (model / "bundle.json").string().c_str());
        return 1;
    }
    nlohmann::json bundle = nlohmann::json::parse(bf);
    int failures = 0, checked = 0;
    for (auto& [r, file] : bundle.at("resolutions").items()) {
        std::ifstream sf(model / file.get<std::string>());
        nlohmann::json sched = nlohmann::json::parse(sf);
        const int steps = sched.at("steps").get<int>();
        const int tokens = sched.at("image_tokens").get<int>();
        const auto& init = sched.at("init");
        auto sig = sch::sigmas(tokens, steps);
        std::string tag = r + ", " + std::to_string(steps) + " steps";
        failures += compare(("TF " + tag).c_str(), sch::timestep_features(sig, steps),
                            read_file(model / init.at("TF").get<std::string>()), 1.0 / 256);
        failures += compare(("DT " + tag).c_str(), sch::dt_params(sig, steps),
                            read_file(model / init.at("DT").get<std::string>()));
        checked += 2;
    }
    // the terminal sigma, and a count the bundle does not have, keep their shape
    auto s8 = sch::sigmas(4096, 8);
    bool shape = s8.size() == 9 && s8.front() == 1.0f && s8.back() == 0.0f;
    for (size_t i = 1; i < s8.size(); ++i) shape = shape && s8[i] < s8[i - 1];
    std::printf("%s  8 steps: 9 sigmas, 1 down to 0, strictly decreasing\n", shape ? "ok  " : "FAIL");
    failures += shape ? 0 : 1;
    ++checked;
    std::printf("\n%s (%d checks, %d failures)\n", failures ? "FAILED" : "PASS", checked, failures);
    return failures ? 1 : 0;
}
