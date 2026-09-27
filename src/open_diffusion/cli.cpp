// open_diffusion_cli: one FLUX.2 [klein] image from a bundle, every op on the NPU.
//
//   open_diffusion_cli --bundle C:\dev\klein-bundle --size 512 --ids ids.npy --out img.png
//       [--noise noise.npy | --seed N] [--runs N] [--profile]
//
// --ids: the chat-templated prompt's token ids (.npy int64/int32, or a comma list);
//   utilities/dit-chain/klein_tokens.py writes them. --noise: packed initial latents
//   [(size/16)^2, 128] as bf16 bits (.npy uint16), e.g. capture_pipeline_inputs.py's.
#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.hpp"

namespace {

struct Npy {
    std::string descr;
    std::vector<size_t> shape;
    std::vector<char> data;
};

Npy read_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "\x93NUMPY", 6)) throw std::runtime_error(path + ": not a .npy file");
    uint32_t hlen = 0;
    if (magic[6] == 1) {
        uint16_t h;
        f.read(reinterpret_cast<char*>(&h), 2);
        hlen = h;
    } else {
        f.read(reinterpret_cast<char*>(&hlen), 4);
    }
    std::string hdr(hlen, '\0');
    f.read(hdr.data(), hlen);
    Npy n;
    auto d = hdr.find("'descr':");
    auto q1 = hdr.find('\'', d + 8), q2 = hdr.find('\'', q1 + 1);
    n.descr = hdr.substr(q1 + 1, q2 - q1 - 1);
    if (hdr.find("'fortran_order': True") != std::string::npos)
        throw std::runtime_error(path + ": fortran order is not supported");
    auto s = hdr.find('(', hdr.find("'shape':")), e = hdr.find(')', s);
    std::stringstream ss(hdr.substr(s + 1, e - s - 1));
    for (std::string tok; std::getline(ss, tok, ',');)
        if (tok.find_first_not_of(' ') != std::string::npos) n.shape.push_back(std::stoull(tok));
    std::vector<char> rest((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    n.data = std::move(rest);
    return n;
}

std::vector<int64_t> read_ids(const std::string& arg) {
    std::vector<int64_t> ids;
    if (arg.size() > 4 && arg.substr(arg.size() - 4) == ".npy") {
        Npy n = read_npy(arg);
        if (n.descr == "<i8") {
            ids.resize(n.data.size() / 8);
            std::memcpy(ids.data(), n.data.data(), ids.size() * 8);
        } else if (n.descr == "<i4") {
            std::vector<int32_t> v(n.data.size() / 4);
            std::memcpy(v.data(), n.data.data(), v.size() * 4);
            ids.assign(v.begin(), v.end());
        } else {
            throw std::runtime_error(arg + ": token ids must be int64 or int32");
        }
    } else {
        std::stringstream ss(arg);
        for (std::string tok; std::getline(ss, tok, ',');) ids.push_back(std::stoll(tok));
    }
    return ids;
}

double process_cpu_s() {
    FILETIME c, e, k, u;
    GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
    auto s = [](FILETIME t) {
        return (static_cast<double>(t.dwHighDateTime) * 4294967296.0 + t.dwLowDateTime) * 1e-7;
    };
    return s(k) + s(u);
}

int usage() {
    std::fprintf(stderr,
                 "usage: open_diffusion_cli --bundle DIR --size 512|1024 --ids FILE.npy|a,b,c --out IMG.png\n"
                 "                          [--noise FILE.npy | --seed N] [--runs N] [--profile]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    std::map<std::string, std::string> a;
    bool profile = false;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--profile") { profile = true; continue; }
        if (k.rfind("--", 0) != 0 || i + 1 >= argc) return usage();
        a[k.substr(2)] = argv[++i];
    }
    if (!a.count("bundle") || !a.count("ids") || !a.count("out")) return usage();
    try {
        int size = a.count("size") ? std::stoi(a["size"]) : 512;
        int runs = a.count("runs") ? std::stoi(a["runs"]) : 1;
        auto t0 = GetTickCount64();
        open_diffusion::Engine eng(a["bundle"], size);
        std::printf("loaded %dx%d in %.1f s\n", size, size, (GetTickCount64() - t0) / 1e3);

        std::vector<uint16_t> noise;
        if (a.count("noise")) {
            Npy n = read_npy(a["noise"]);
            if (n.descr != "<u2") throw std::runtime_error("--noise must be uint16 (bf16 bits)");
            noise.resize(n.data.size() / 2);
            std::memcpy(noise.data(), n.data.data(), noise.size() * 2);
        } else {
            noise = eng.seeded_noise(a.count("seed") ? std::stoull(a["seed"]) : 0);
        }
        auto ids = read_ids(a["ids"]);
        for (int r = 0; r < runs; ++r) {
            double c0 = process_cpu_s();
            eng.set_tokens(ids);
            eng.set_noise(noise);
            auto t = eng.run(profile);
            double cpu = process_cpu_s() - c0;
            auto img = eng.rgb();
            open_diffusion::write_png_rgb(a["out"], img.data(), size, size);
            std::printf("[run %d] %.2f s on the NPU (", r, t.total_s);
            for (size_t i = 0; i < t.phases.size(); ++i)
                std::printf("%s%s %.2f", i ? ", " : "", t.phases[i].first.c_str(), t.phases[i].second);
            std::printf("); host CPU %.2f s -> %s\n", cpu, a["out"].c_str());
            if (profile) {
                auto ops = eng.ops();
                std::map<std::string, double> by_set;
                for (size_t i = 0; i < t.op_ms.size(); ++i) by_set[std::get<0>(ops[i])] += t.op_ms[i];
                std::printf("  per set (ms):");
                for (auto& [s, ms] : by_set) std::printf(" %s %.0f", s.c_str(), ms);
                std::printf("\n");
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "open_diffusion_cli: %s\n", e.what());
        return 1;
    }
    return 0;
}
