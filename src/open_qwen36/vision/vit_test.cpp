// vit_test: the C++ vision tower against replica_vit.py's fixture (OPEN-VISION-VIT-REF).
//   python open_kernels/model/replica_vit.py --grid 16 16 --no-hf --fixture DIR
//   vit_test <model_dir> DIR
// Prints the correlation and the max error against the numpy reference (which is itself
// checked against transformers), and the forward's wall time.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
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

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: vit_test <model_dir> <fixture_dir>\n"); return 2; }
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
