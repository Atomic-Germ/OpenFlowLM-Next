// elf_probe: the Granite bfp16 prefill streams on two hardware contexts (xclbins) against one full ELF
// (open_kernels/compose_prefill_elf.py). Times the layer pattern both ways, and each switch alone.
//   elf_probe <kernel dir> [--reps N] [--prio 0x180] [--tier af512] [--mode all|alone|layer]
// Buffers hold whatever the allocation gives (zeros): only time and hangs are measured.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_kernel.h"

namespace fs = std::filesystem;
using json = nlohmann::json;
using clk = std::chrono::steady_clock;

static std::vector<char> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    std::vector<char> d(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(d.data(), static_cast<std::streamsize>(d.size()));
    return d;
}

struct Op { std::string set, stream; };

static double med(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[v.size() / 2];
}

int main(int argc, char** argv) try {
    if (argc < 2) { std::fprintf(stderr, "usage: elf_probe <kernel dir> [--reps N] [--prio P] [--tier af512] [--mode m]\n"); return 2; }
    fs::path kdir = argv[1];
    int reps = 9;
    unsigned prio = 0x180, xprio = 0;
    std::string tier = "af512", mode = "all", meta_name = "prefill_elf.json";
    int layers = 40;
    for (int i = 2; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--reps") reps = std::atoi(argv[i + 1]);
        else if (a == "--prio") prio = static_cast<unsigned>(std::strtoul(argv[i + 1], nullptr, 0));
        else if (a == "--xprio") xprio = static_cast<unsigned>(std::strtoul(argv[i + 1], nullptr, 0));
        else if (a == "--tier") tier = argv[i + 1];
        else if (a == "--meta") meta_name = argv[i + 1];
        else if (a == "--mode") mode = argv[i + 1];
        else if (a == "--layers") layers = std::atoi(argv[i + 1]);
    }
    json man = json::parse(read_file(kdir / "manifest.json"));
    json meta = json::parse(read_file(kdir / meta_name));
    xrt::device dev(0);

    // ---- buffers: per set, per argument index, the largest any stream of the set needs
    std::map<std::string, std::vector<xrt::bo>> bos;
    for (auto& [set, names] : meta.at("streams").items()) {
        std::vector<size_t> mx;
        for (auto& n : names) {
            auto b = meta.at("arg_bytes").at(n.get<std::string>()).get<std::vector<size_t>>();
            if (mx.size() < b.size()) mx.resize(b.size(), 0);
            for (size_t i = 0; i < b.size(); ++i) mx[i] = std::max(mx[i], b[i]);
        }
        for (size_t s : mx) {
            xrt::bo bo = xrt::ext::bo(dev, (s + 4095) / 4096 * 4096);
            std::memset(bo.map<void*>(), 0, (s + 4095) / 4096 * 4096);
            bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bos[set].push_back(bo);
        }
    }
    auto nargs = [&](const std::string& st) { return meta.at("nargs").at(st).get<size_t>(); };

    // ---- the xclbin contexts (what the engine runs today)
    struct XK { xrt::hw_context ctx; xrt::kernel k; };
    std::map<std::string, XK> xk;
    for (auto& set : meta.at("sets")) {
        std::string s = set.get<std::string>();
        fs::path p = kdir / man.at("contexts").at(s).get<std::string>();
        xrt::xclbin xcl(p.string());
        auto uuid = dev.register_xclbin(xcl);
        xrt::hw_context ctx = xprio ? xrt::hw_context(dev, uuid, xrt::hw_context::cfg_param_type{{"priority", xprio}}) : xrt::hw_context(dev, uuid);
        xk[s] = XK{ctx, xrt::kernel(ctx, "MLIR_AIE")};
    }
    std::map<std::string, std::pair<xrt::bo, size_t>> insts;
    for (auto& [name, kd] : man.at("kernels").items()) {
        if (!meta.at("nargs").contains(name)) continue;
        auto d = read_file(kdir / kd.at("insts").get<std::string>());
        std::string set = kd.at("context").get<std::string>();
        xrt::bo b(dev, d.size(), xrt::bo::flags::cacheable, xk.at(set).k.group_id(1));
        std::memcpy(b.map<void*>(), d.data(), d.size());
        b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        insts[name] = {b, d.size() / 4};
    }
    auto xrun = [&](const Op& o) {
        xrt::run r(xk.at(o.set).k);
        r.set_arg(0, 3);
        r.set_arg(1, insts.at(o.stream).first);
        r.set_arg(2, static_cast<int>(insts.at(o.stream).second));
        int i = 3;
        for (size_t a = 0; a < nargs(o.stream); ++a) r.set_arg(i++, bos.at(o.set)[a]);
        return r;
    };

    // ---- the ELF context
    auto t0 = clk::now();
    xrt::elf elf((kdir / meta.at("elf").get<std::string>()).string());
    xrt::hw_context ectx(dev, elf, {{"priority", prio}}, xrt::hw_context::access_mode::shared);
    std::map<std::string, xrt::ext::kernel> ek;
    auto ekernel = [&](const std::string& n) -> xrt::ext::kernel& {
        auto it = ek.find(n);
        if (it == ek.end()) it = ek.emplace(n, xrt::ext::kernel(ectx, n)).first;
        return it->second;
    };
    for (auto& [set, c] : meta.at("cfg").items())
        for (auto& n : c) ekernel(n.get<std::string>());
    double create_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    std::printf("ELF %s: context + cfg kernels %.0f ms\n", meta.at("elf").get<std::string>().c_str(), create_ms);
    auto erun_mk = [&](const Op& o) {
        xrt::run r(ekernel(o.set + ":" + o.stream));
        for (size_t a = 0; a < nargs(o.stream); ++a) r.set_arg(a, bos.at(o.set)[a]);
        return r;
    };
    int flip = 0;
    auto cfg_run = [&](const std::string& set) {
        auto c = meta.at("cfg").at(set);
        std::string n = c[flip].get<std::string>();
        flip ^= 1;
        return xrt::run(ekernel(n));
    };

    // ---- sequences. The layer pattern of one 256-token block: qkv, prep, attention, o, glu, down.
    const std::string G = "gemmbfp", A = "af";
    std::vector<Op> layer = {{G, "gemm_bfp16_n4096_k2560"}, {A, "afp"}, {A, tier}, {G, "gemm_bfp16_n3072_k2560"},
                             {G, "gemm_bfp16_n16384_k2560_glu"}, {G, "gemm_bfp16_n3072_k8192_h"}};
    auto repeat = [&](std::vector<Op> v, int n) { std::vector<Op> o; for (int i = 0; i < n; ++i) o.insert(o.end(), v.begin(), v.end()); return o; };

    auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };

    // xclbin, queued: every run started, then all waited (the engine's order within a block, minus its host work)
    auto x_queued = [&](const std::vector<Op>& seq) {
        std::vector<xrt::run> rs;
        auto t = clk::now();
        for (auto& o : seq) { rs.push_back(xrun(o)); rs.back().start(); }
        for (auto& r : rs) r.wait();
        return ms(t, clk::now());
    };
    auto x_waited = [&](const std::vector<Op>& seq) {
        auto t = clk::now();
        for (auto& o : seq) { auto r = xrun(o); r.start(); r.wait(); }
        return ms(t, clk::now());
    };
    // ELF, stretches: one runlist per run of same-set ops, led by that set's configure (as the diffusion engine)
    auto e_stretch = [&](const std::vector<Op>& seq, size_t window) {
        std::deque<xrt::runlist> inflight;
        std::deque<std::vector<xrt::run>> keep;
        auto t = clk::now();
        size_t i = 0;
        while (i < seq.size()) {
            size_t j = i;
            while (j < seq.size() && seq[j].set == seq[i].set) ++j;
            keep.emplace_back();
            auto& rs = keep.back();
            rs.reserve(j - i + 1);
            rs.push_back(cfg_run(seq[i].set));
            for (size_t k = i; k < j; ++k) rs.push_back(erun_mk(seq[k]));
            xrt::runlist l(ectx);
            for (auto& r : rs) l.add(r);
            if (inflight.size() >= window) { inflight.front().wait(); inflight.pop_front(); keep.pop_front(); }
            l.execute();
            inflight.push_back(std::move(l));
            i = j;
        }
        for (auto& l : inflight) l.wait();
        return ms(t, clk::now());
    };
    // ELF, one run at a time, configure only on a set change, waited
    auto e_waited = [&](const std::vector<Op>& seq) {
        auto t = clk::now();
        std::string cur;
        for (auto& o : seq) {
            if (o.set != cur) { auto c = cfg_run(o.set); c.start(); c.wait(); cur = o.set; }
            auto r = erun_mk(o);
            r.start();
            r.wait();
        }
        return ms(t, clk::now());
    };
    // ELF, queued individual runs, configure only on a set change
    auto e_queued = [&](const std::vector<Op>& seq) {
        std::vector<xrt::run> rs;
        std::string cur;
        auto t = clk::now();
        for (auto& o : seq) {
            if (o.set != cur) { rs.push_back(cfg_run(o.set)); rs.back().start(); cur = o.set; }
            rs.push_back(erun_mk(o));
            rs.back().start();
        }
        for (auto& r : rs) r.wait();
        return ms(t, clk::now());
    };

    auto rep = [&](const char* what, auto fn, const std::vector<Op>& seq, int n) {
        std::vector<double> v;
        fn(seq);                                     // warm-up
        for (int i = 0; i < n; ++i) v.push_back(fn(seq));
        std::sort(v.begin(), v.end());
        std::printf("  %-34s %9.3f ms  (min %.3f max %.3f)  per op %.3f\n", what, med(v), v.front(), v.back(), med(v) / seq.size());
        std::fflush(stdout);
        return med(v);
    };

    if (mode == "hammer") {
        // a contender for the contention test: xclbin contexts alternating gemm / attention ops for --reps seconds
        std::vector<Op> ga = {{G, "gemm_bfp16_n3072_k2560"}, {A, "afp"}, {G, "gemm_bfp16_n4096_k2560"}, {A, tier}};
        auto end = clk::now() + std::chrono::seconds(reps);
        long n = 0;
        while (clk::now() < end) { x_waited(ga); n += static_cast<long>(ga.size()); }
        std::printf("hammer: %ld ops in %d s\n", n, reps);
        return 0;
    }
    if (mode == "third") {
        // configure cost of every set in the ELF, alone and in a GEMM / other / GEMM / attention cycle
        std::vector<std::string> sets;
        for (auto& s : meta.at("sets")) sets.push_back(s.get<std::string>());
        auto timed = [&](const std::vector<std::string>& seq, bool queued) {
            std::vector<xrt::run> rs;
            auto t = clk::now();
            for (auto& s : seq) {
                rs.push_back(cfg_run(s));
                rs.back().start();
                if (!queued) rs.back().wait();
            }
            for (auto& r : rs) r.wait();
            return ms(t, clk::now()) / seq.size();
        };
        auto show = [&](const std::string& what, const std::vector<std::string>& seq) {
            for (bool q : {true, false}) {
                std::vector<double> v;
                timed(seq, q);
                for (int i = 0; i < reps; ++i) v.push_back(timed(seq, q));
                std::printf("  %-34s %s per cfg %7.3f ms (min %.3f)\n", what.c_str(), q ? "queued" : "waited", med(v),
                            *std::min_element(v.begin(), v.end()));
            }
            std::fflush(stdout);
        };
        for (auto& s : sets) show("cfg " + s + " repeated", std::vector<std::string>(20, s));
        std::vector<std::string> cyc;
        for (int i = 0; i < 5; ++i)
            for (auto& s : sets) if (s != G) { cyc.push_back(G); cyc.push_back(s); }
        show("cycle G,x,G,y,... over all sets", cyc);
        return 0;
    }
    if (mode == "all" || mode == "switch") {
        std::printf("== switch cost alone (waited unless noted), %d reps of 20\n", reps);
        auto cfg_waited = [&](const std::vector<std::string>& sets) {
            auto t = clk::now();
            for (auto& s : sets) { auto c = cfg_run(s); c.start(); c.wait(); }
            return ms(t, clk::now());
        };
        auto cfg_queued = [&](const std::vector<std::string>& sets) {
            std::vector<xrt::run> rs;
            auto t = clk::now();
            for (auto& s : sets) { rs.push_back(cfg_run(s)); rs.back().start(); }
            for (auto& r : rs) r.wait();
            return ms(t, clk::now());
        };
        auto many = [&](const char* what, auto fn, std::vector<std::string> sets) {
            std::vector<double> v;
            fn(sets);
            for (int i = 0; i < reps; ++i) v.push_back(fn(sets));
            std::printf("  %-34s per cfg %7.3f ms (min %.3f max %.3f)\n", what, med(v) / sets.size(),
                        v.front() / sets.size(), v.back() / sets.size());
            std::fflush(stdout);
        };
        std::vector<std::string> g(20, G), a(20, A), alt;
        for (int i = 0; i < 10; ++i) { alt.push_back(G); alt.push_back(A); }
        many("cfg gemm repeated, waited", cfg_waited, g);
        many("cfg af repeated, waited", cfg_waited, a);
        many("cfg gemm/af alternating, waited", cfg_waited, alt);
        many("cfg gemm repeated, queued", cfg_queued, g);
        many("cfg af repeated, queued", cfg_queued, a);
        many("cfg gemm/af alternating, queued", cfg_queued, alt);
        // xclbin: the same alternation with the cheapest op of each set
        std::vector<Op> gg(20, Op{G, "gemm_bfp16_n3072_k2560"}), aa(20, Op{A, "afp"}), ga;
        for (int i = 0; i < 10; ++i) { ga.push_back(gg[0]); ga.push_back(aa[0]); }
        auto xw = [&](const std::vector<Op>& s) { return x_waited(s) / s.size(); };
        std::vector<double> v1, v2, v3;
        x_waited(ga);
        for (int i = 0; i < reps; ++i) { v1.push_back(xw(gg)); v2.push_back(xw(aa)); v3.push_back(xw(ga)); }
        std::printf("  xclbin o waited %.3f, prep waited %.3f, alternating waited %.3f ms/op -> per switch %.3f ms\n",
                    med(v1), med(v2), med(v3), med(v3) - (med(v1) + med(v2)) / 2);
        std::vector<double> e1, e2, e3;
        auto ew = [&](const std::vector<Op>& s) { return e_waited(s) / s.size(); };
        auto e_always = [&](const std::vector<Op>& s) {
            auto t = clk::now();
            for (auto& o : s) { auto c = cfg_run(o.set); c.start(); c.wait(); auto r = erun_mk(o); r.start(); r.wait(); }
            return ms(t, clk::now()) / s.size();
        };
        ew(ga);
        for (int i = 0; i < reps; ++i) { e1.push_back(ew(gg)); e2.push_back(ew(aa)); e3.push_back(e_always(ga)); }
        std::printf("  ELF o waited %.3f, prep waited %.3f, cfg+op each time (alternating) %.3f ms/op -> per switch %.3f ms\n",
                    med(e1), med(e2), med(e3), med(e3) - (med(e1) + med(e2)) / 2);
    }
    if (mode == "all" || mode == "alone") {
        std::printf("== each stream alone, 20 queued back to back (xclbin) / after one configure (ELF, queued)\n");
        for (auto& [set, names] : meta.at("streams").items())
            for (auto& n : names) {
                std::string s = n.get<std::string>();
                if (set == A && s != "afp" && s != tier) continue;
                Op o{set, s};
                std::vector<Op> seq(20, o);
                std::printf("%s:%s\n", set.c_str(), s.c_str());
                rep("xclbin queued", x_queued, seq, reps);
                rep("xclbin waited", x_waited, seq, reps);
                rep("ELF queued (1 cfg)", e_queued, seq, reps);
                rep("ELF waited (1 cfg)", e_waited, seq, reps);
            }
    }
    if (mode == "all" || mode == "layer") {
        std::printf("== layer pattern x%d (qkv, prep, attn %s, o, glu, down), %d reps\n", layers, tier.c_str(), reps);
        auto seq = repeat(layer, layers);
        double xq = rep("xclbin queued", x_queued, seq, reps);
        double xw = rep("xclbin waited", x_waited, seq, reps);
        double eq = rep("ELF queued, configure on change", e_queued, seq, reps);
        double ew = rep("ELF waited, configure on change", e_waited, seq, reps);
        double es1 = rep("ELF stretches, window 1", [&](auto& s) { return e_stretch(s, 1); }, seq, reps);
        double es4 = rep("ELF stretches, window 4", [&](auto& s) { return e_stretch(s, 4); }, seq, reps);
        double es32 = rep("ELF stretches, window 32", [&](auto& s) { return e_stretch(s, 32); }, seq, reps);
        std::printf("  switches per pattern: %d to attention + %d back; ELF queued - xclbin queued per switch %.3f ms\n",
                    layers, layers, (eq - xq) / (2 * layers));
        (void)xw; (void)ew; (void)es1; (void)es4; (void)es32;
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "elf_probe: %s\n", e.what());
    return 1;
}
