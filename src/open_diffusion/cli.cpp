// open_diffusion_cli: one FLUX.2 [klein] image from a bundle, every op on the NPU.
//
//   open_diffusion_cli --model DIR [--kernels DIR] --size 512 --ids ids.npy --out img.png|.jpg
//       [--noise noise.npy | --seed N] [--steps N] [--runs N] [--profile] [--ref FILE]
//
// --model: q4nx-build --open-diffusion's output. --kernels: an installed kernel set
//   (export_dit_kernels.py --install); default OFLM_DIFFUSION_KERNELS_DIR, else
//   <model>/open_kernels.
// --ids: the chat-templated prompt's token ids (.npy int64/int32, or a comma list);
//   utilities/dit-chain/klein_tokens.py writes them. --noise: packed initial latents
//   [(size/16)^2, 128] as bf16 bits (.npy uint16), e.g. capture_pipeline_inputs.py's.
// --ref: an edit's reference -- a PNG or JPEG (reference.hpp prepares it: orientation,
//   centre crop, resize to --size), or a prepared [size, size, 3] uint8 .npy
//   (capture_edit_goldens.py's ref_<i>.npy).
#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.hpp"
#include "reference.hpp"

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
                 "usage: open_diffusion_cli --model DIR [--kernels DIR] --size 512|1024\n"
                 "                          --ids FILE.npy|a,b,c --out IMG.png|.jpg\n"
                 "                          [--noise FILE.npy | --seed N] [--steps N] [--runs N] [--profile]\n"
                 "                          [--ref IMAGE.png|.jpg|PREPARED.npy]\n");
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
    if (!a.count("model") || !a.count("ids") || !a.count("out")) return usage();
    try {
        int size = a.count("size") ? std::stoi(a["size"]) : 512;
        int runs = a.count("runs") ? std::stoi(a["runs"]) : 1;
        int steps = a.count("steps") ? std::stoi(a["steps"]) : 0;
        std::string format = open_diffusion::format_for_path(a["out"]);
        if (format.empty()) throw std::runtime_error("--out must end in .png, .jpg or .jpeg");
        std::string how, kernels = a.count("kernels") ? a["kernels"] : "";
        if (kernels.empty()) {
            const char* env = std::getenv("OFLM_DIFFUSION_KERNELS_DIR");
            kernels = open_diffusion::find_kernels(a["model"], env ? env : "", {}, &how);
            if (kernels.empty()) throw std::runtime_error("no kernel set found: pass --kernels");
        }
        std::vector<uint8_t> ref;
        if (a.count("ref")) {
            const std::string& rp = a["ref"];
            if (rp.size() > 4 && rp.substr(rp.size() - 4) == ".npy") {
                Npy n = read_npy(rp);
                if (n.descr != "|u1" || n.shape != std::vector<size_t>{static_cast<size_t>(size), static_cast<size_t>(size), 3})
                    throw std::runtime_error("--ref .npy must be uint8 [size, size, 3]");
                ref.assign(n.data.begin(), n.data.end());
            } else {
                std::ifstream f(rp, std::ios::binary);
                if (!f) throw std::runtime_error("cannot open " + rp);
                std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                auto prepared = open_diffusion::prepare_reference(bytes.data(), bytes.size(), size);
                std::printf("%s\n", prepared.describe().c_str());
                ref = std::move(prepared.rgb);
            }
        }
        auto t0 = GetTickCount64();
        std::optional<open_diffusion::Engine> engine;
        if (ref.empty()) {
            engine.emplace(a["model"], kernels, size);   // the size known up front: see engine.hpp
            if (steps) engine->select(size, steps);
        } else {
            engine.emplace(a["model"], kernels);
            engine->select(size, steps, true);
        }
        open_diffusion::Engine& eng = *engine;
        std::printf("loaded %dx%d%s, %d steps, in %.1f s\n", size, size, ref.empty() ? "" : " (edit)", eng.steps(),
                    (GetTickCount64() - t0) / 1e3);

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
            if (!ref.empty()) eng.set_reference(ref);
            auto t = eng.run(profile);
            double cpu = process_cpu_s() - c0;
            auto img = eng.encode(format);
            std::ofstream f(a["out"], std::ios::binary);
            if (!f.write(reinterpret_cast<const char*>(img.data()), static_cast<std::streamsize>(img.size())))
                throw std::runtime_error("cannot write " + a["out"]);
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
                std::map<std::string, std::pair<int, double>> by_stream;
                for (size_t i = 0; i < t.op_ms.size(); ++i) {
                    auto& e = by_stream[std::get<0>(ops[i]) + ":" + std::get<1>(ops[i])];
                    ++e.first;
                    e.second += t.op_ms[i];
                }
                std::vector<std::pair<std::string, std::pair<int, double>>> rows(by_stream.begin(), by_stream.end());
                std::sort(rows.begin(), rows.end(), [](auto& x, auto& y) { return x.second.second > y.second.second; });
                std::printf("  per stream (ms, calls):\n");
                for (auto& [s, e] : rows) std::printf("    %-28s %8.1f %5d\n", s.c_str(), e.second, e.first);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "open_diffusion_cli: %s\n", e.what());
        return 1;
    }
    return 0;
}
