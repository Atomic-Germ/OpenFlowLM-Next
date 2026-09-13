// vit_test: the C++ vision tower against replica_vit.py's fixture (OPEN-VISION-VIT-REF).
//   python open_kernels/model/replica_vit.py --grid 16 16 --no-hf --fixture DIR
//   vit_test <model_dir> DIR
// Prints the correlation and the max error against the numpy reference (which is itself
// checked against transformers), and the forward's wall time.
//
//   vit_test --configs <fixtures_dir>   (OPEN-VISION-VIT-CONFIG, needs no container)
// Reads the checked-in container configs and checks the geometry against the same
// expectations tests/test_vision_config.py holds replica_vit.py to.
//
//   vit_test --window-index             (OPEN-VISION-VIT-WINDOWED, needs nothing at all)
// Qwen2.5-VL's window permutation against the values worked out by hand from the geometry
// -- the same ones tests/test_vision_vit_windowed.py pins the numpy reference to.
//
//   vit_test --windowed <model_dir> <fixture>   (OPEN-VISION-VIT-WINDOWED, numeric)
//   python open_kernels/model/replica_vit_qwen25.py --fixture DIR   writes both.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "open_qwen36/vision/vit.hpp"

static std::vector<float> read_f32(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(2); }
    f.seekg(0, std::ios::end);
    std::vector<float> v(static_cast<size_t>(f.tellg()) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
    return v;
}

static int g_failed = 0;

static void check(bool ok, const std::string& what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

static std::string read_text(const std::string& p) {
    std::ifstream f(p);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(2); }
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

/// The geometry of a checked-in config, or the refusal it raised.
static int config_tests(const std::string& fixtures) {
    using namespace open_qwen36::vision;
    auto cfg_of = [&](const char* name) { return VitConfig::from_config_text(read_text(fixtures + "/config_" + name + ".json")); };
    auto refusal = [&](const char* name) {
        try {
            cfg_of(name);
        } catch (const std::exception& e) {
            return std::string(e.what());
        }
        return std::string();
    };
    const VitConfig a = cfg_of("qwen36_35b");
    check(a.depth == 27 && a.hidden == 1152 && a.heads == 16 && a.head_dim == 72,
          "QWEN3_6_MOE_* keys: 27 x 1152, 16 heads x 72");
    check(a.inter == 4304 && a.out == 2048 && a.patch == 16 && a.temporal == 2 && a.merge == 2 && a.npos == 2304,
          "QWEN3_6_MOE_* keys: MLP 4304 -> 2048, patch 16, 2304 positions");
    const VitConfig b = cfg_of("qwen35_9b");
    check(b.hidden == b.heads * b.head_dim && b.depth > 0, "QWEN3_5_* keys read");
    const VitConfig h = cfg_of("qwen35_0p8b");                 // Qwen3.5-0.8B's HF config
    check(h.depth == 12 && h.hidden == 768 && h.heads == 12 && h.head_dim == 64,
          "plain transformers keys: head_dim comes from hidden_size / num_heads");
    check(h.inter == 3072 && h.out == 1024 && h.npos == 2304 && h.channels == 3 && h.eps == 1e-6f,
          "plain transformers keys: no epsilon in the block, LayerNorm's default");
    const VitConfig hc = cfg_of("qwen35_0p8b_container");      // the container OFLM ships for it
    check(h.depth == hc.depth && h.hidden == hc.hidden && h.heads == hc.heads && h.head_dim == hc.head_dim &&
              h.inter == hc.inter && h.out == hc.out && h.patch == hc.patch && h.npos == hc.npos,
          "both shapes of Qwen3.5-0.8B give the same tower");
    check(refusal("qwen3vl_4b").find("vision_config") != std::string::npos,
          "a container with no vision_config names what was looked for");
    check(refusal("qwen3vl_4b_hf").find("deepstack") != std::string::npos,
          "a deepstack tower is refused by name");
    check(refusal("qwen25vl_3b").find("window") != std::string::npos,
          "a windowed tower is refused by name");
    return g_failed == 0 ? 0 : 1;
}

/// Qwen2.5-VL's window permutation, against numbers derived from the geometry by hand.
static int window_index_tests() {
    using namespace open_qwen36::vision;
    VitConfig c;
    c.family = VitFamily::Qwen25VL;
    c.merge = 2;
    c.patch = 14;
    c.window = 112;
    check(c.window_side() == 4, "112 px / 2 patches per merge unit / 14 px per patch = 4 units a side");

    std::vector<int> idx, cu;
    window_index(c, 12, 10, idx, cu);
    const std::vector<int> want = {0,  1,  2,  3,  5,  6,  7,  8, 10, 11, 12, 13, 15, 16, 17, 18,
                                  4,  9,  14, 19, 20, 21, 22, 23, 25, 26, 27, 28, 24, 29};
    check(idx == want, "12 x 10 patches -> 6 x 5 merge units in four windows, two partial");
    check(cu == std::vector<int>({0, 64, 80, 112, 120}), "its segment boundaries are in patches: 64, 80, 112, 120");

    window_index(c, 8, 8, idx, cu);
    std::vector<int> ident(16);
    for (int i = 0; i < 16; ++i) ident[i] = i;
    check(idx == ident && cu == std::vector<int>({0, 64}),
          "a grid that divides still pads whole empty windows, which collapse away");

    bool perm = true;
    for (const auto& gr : std::vector<std::pair<int, int>>{{12, 10}, {8, 8}, {2, 2}, {16, 24}, {6, 18}}) {
        window_index(c, gr.first, gr.second, idx, cu);
        std::vector<int> sorted = idx;
        std::sort(sorted.begin(), sorted.end());
        for (size_t i = 0; i < sorted.size(); ++i) perm = perm && sorted[i] == static_cast<int>(i);
        perm = perm && static_cast<int>(sorted.size()) == gr.first / 2 * (gr.second / 2);
        perm = perm && cu.front() == 0 && cu.back() == gr.first * gr.second;
        for (size_t i = 1; i < cu.size(); ++i) perm = perm && cu[i] > cu[i - 1];
    }
    check(perm, "on five grids it is a permutation of the merge units with increasing boundaries");
    return g_failed == 0 ? 0 : 1;
}

/// OPEN-VISION-VIT-FLAT: Qwen3-VL's tower off the shipped container, deepstack included,
/// against replica_deepstack.py's fixture. The config comes from the WEIGHT FILE, because
/// this container has no vision_config at all; only the head count and the tap indexes are
/// given, and the loader checks everything else it derives.
///
///   python open_kernels/model/replica_deepstack.py --model-dir DIR --indexes 5 11 17 --fixture FX
///   vit_test --deepstack DIR FX 16 5,11,17
static int deepstack_test(const std::string& md, const std::string& fx, int heads,
                          const std::vector<int>& taps) {
    using namespace open_qwen36::vision;
    nlohmann::json g;
    std::ifstream(fx + "/grid.json") >> g;
    const int gh = g.at("h"), gw = g.at("w");
    const VitConfig cfg = VitConfig::qwen3vl_from_tensors(md + "/vision_weight.q4nx", heads, taps);
    std::printf("from the weight file alone: depth %d hidden %d inter %d out %d npos %d merge %d "
                "patch %d temporal %d\n",
                cfg.depth, cfg.hidden, cfg.inter, cfg.out, cfg.npos, cfg.merge, cfg.patch, cfg.temporal);
    auto t0 = std::chrono::steady_clock::now();
    const VitWeights w = load_vit(md + "/vision_weight.q4nx", cfg);
    std::printf("weights: %d blocks + %zu deepstack mergers in %.1f s\n", cfg.depth, w.deepstack.size(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    const std::vector<float> px = read_f32(fx + "/pixels.bin");
    std::vector<std::vector<float>> deep;
    t0 = std::chrono::steady_clock::now();
    const std::vector<float> y = vit_forward_deepstack(cfg, w, px.data(), gh, gw, &deep);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    auto compare = [&](const std::vector<float>& got, const std::string& path, const char* label) {
        const std::vector<float> ref = read_f32(path);
        if (got.size() != ref.size()) {
            std::printf("FAIL %-14s %zu vs ref %zu\n", label, got.size(), ref.size());
            ++g_failed;
            return;
        }
        double sy = 0, sr = 0, syy = 0, srr = 0, syr = 0, maxe = 0, maxr = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            sy += got[i]; sr += ref[i];
            syy += double(got[i]) * got[i]; srr += double(ref[i]) * ref[i]; syr += double(got[i]) * ref[i];
            maxe = std::max(maxe, std::fabs(double(got[i]) - ref[i]));
            maxr = std::max(maxr, std::fabs(double(ref[i])));
        }
        const double nn = static_cast<double>(got.size());
        const double corr = (syr - sy * sr / nn) / std::sqrt((syy - sy * sy / nn) * (srr - sr * sr / nn));
        const bool ok = corr > 0.99999 && maxe / maxr < 1e-3;
        std::printf("%s %-14s corr %.8f  rel %.2e\n", ok ? "ok  " : "FAIL", label, corr, maxe / maxr);
        if (!ok) ++g_failed;
    };

    std::printf("%d x %d patches in %.2f s\n", gh, gw, secs);
    compare(y, fx + "/ref.bin", "merged");
    for (size_t j = 0; j < deep.size(); ++j)
        compare(deep[j], fx + "/deep" + std::to_string(j) + ".bin", ("deepstack[" + std::to_string(j) + "]").c_str());
    if (deep.size() != taps.size()) {
        std::printf("FAIL %zu features for %zu taps\n", deep.size(), taps.size());
        ++g_failed;
    }
    std::printf("%s\n", g_failed ? "FAILED" : "PASS");
    return g_failed == 0 ? 0 : 1;
}

static std::vector<int> parse_ints(const std::string& csv) {
    std::vector<int> out;
    size_t i = 0;
    while (i < csv.size()) {
        size_t j = csv.find(',', i);
        if (j == std::string::npos) j = csv.size();
        out.push_back(std::atoi(csv.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--window-index") return window_index_tests();
    if (argc >= 3 && std::string(argv[1]) == "--configs") return config_tests(argv[2]);
    if (argc >= 6 && std::string(argv[1]) == "--deepstack")
        return deepstack_test(argv[2], argv[3], std::atoi(argv[4]), parse_ints(argv[5]));
    const bool windowed = argc >= 4 && std::string(argv[1]) == "--windowed";
    if (argc < 3) {
        std::fprintf(stderr, "usage: vit_test <model_dir> <fixture_dir> | vit_test --windowed <model_dir> <fixture_dir>"
                             " | vit_test --deepstack <model_dir> <fixture_dir> <heads> <taps,csv>"
                             " | vit_test --configs <fixtures_dir> | vit_test --window-index\n");
        return 2;
    }
    using namespace open_qwen36::vision;
    const std::string md = argv[windowed ? 2 : 1], fx = argv[windowed ? 3 : 2];
    nlohmann::json g;
    std::ifstream(fx + "/grid.json") >> g;
    const int gh = g.at("h"), gw = g.at("w");
    const VitConfig cfg = windowed ? VitConfig::qwen25_from_model_dir(md) : VitConfig::from_model_dir(md);
    auto t0 = std::chrono::steady_clock::now();
    const VitWeights w = load_vit(md + (windowed ? "/vision_weights.q4nx" : "/vision_weight.q4nx"), cfg);
    auto t1 = std::chrono::steady_clock::now();
    std::printf("weights: %d blocks in %.1f s\n", cfg.depth, std::chrono::duration<double>(t1 - t0).count());
    const std::vector<float> px = read_f32(fx + "/pixels.bin"), ref = read_f32(fx + "/ref.bin");
    if (px.size() != static_cast<size_t>(gh * gw) * cfg.patch_dim()) { std::fprintf(stderr, "pixels.bin size mismatch\n"); return 2; }
    t0 = std::chrono::steady_clock::now();
    const std::vector<float> y = vit_forward(cfg, w, px.data(), gh, gw);
    t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    if (y.size() != ref.size()) { std::fprintf(stderr, "output %zu vs ref %zu\n", y.size(), ref.size()); return 1; }
    double sy = 0, sr = 0, syy = 0, srr = 0, syr = 0, maxe = 0, maxr = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        sy += y[i]; sr += ref[i]; syy += double(y[i]) * y[i]; srr += double(ref[i]) * ref[i]; syr += double(y[i]) * ref[i];
        maxe = std::max(maxe, std::fabs(double(y[i]) - ref[i]));
        maxr = std::max(maxr, std::fabs(double(ref[i])));
    }
    const double nn = static_cast<double>(y.size());
    const double corr = (syr - sy * sr / nn) / std::sqrt((syy - sy * sy / nn) * (srr - sr * sr / nn));
    std::printf("%dx%d patches -> %zu x %d in %.2f s: corr %.8f  max|err| %.3e  max|ref| %.3e  rel %.2e\n",
                gh, gw, y.size() / cfg.out, cfg.out, secs, corr, maxe, maxr, maxe / maxr);
    return (corr > 0.99999 && maxe / maxr < 1e-3) ? 0 : 1;
}
