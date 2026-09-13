// vit_test: the C++ vision tower against replica_vit.py's fixture (OPEN-VISION-VIT-REF).
//   python open_kernels/model/replica_vit.py --grid 16 16 --no-hf --fixture DIR
//   vit_test <model_dir> DIR
// Prints the correlation and the max error against the numpy reference (which is itself
// checked against transformers), and the forward's wall time.
//
//   vit_test --configs <fixtures_dir>   (OPEN-VISION-VIT-CONFIG, needs no container)
// Reads the checked-in container configs and checks the geometry against the same
// expectations tests/test_vision_config.py holds replica_vit.py to.
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

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--configs") return config_tests(argv[2]);
    if (argc < 3) { std::fprintf(stderr, "usage: vit_test <model_dir> <fixture_dir> | vit_test --configs <fixtures_dir>\n"); return 2; }
    using namespace open_qwen36::vision;
    const std::string md = argv[1], fx = argv[2];
    nlohmann::json g;
    std::ifstream(fx + "/grid.json") >> g;
    const int gh = g.at("h"), gw = g.at("w");
    const VitConfig cfg = VitConfig::from_model_dir(md);
    auto t0 = std::chrono::steady_clock::now();
    const VitWeights w = load_vit(md + "/vision_weight.q4nx", cfg);
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
