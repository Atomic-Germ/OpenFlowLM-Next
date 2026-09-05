// run_kernel: drive IRON / mlir-aie kernels on the NPU from a small .cfg
// program — the same 6-line language phlegm's driver speaks, so every design's
// make_test.py / compare.py pair works unchanged against this host.
//
//   device                                  open the NPU
//   xclbin  <name> <final.xclbin>           register + hw_context
//   kernelx <name> <xclbin> <insts.bin>     classic flow: xrt::kernel("MLIR_AIE") + instr BO
//   kernel  <name> <xclbin> <insts.elf>     ELF flow: xrt::elf -> module -> ext::kernel
//   buf     <name> <bytes> [init-file]      device buffer (zeroed, or from file)
//   load    <buf> <file>                    overwrite a buffer from a file
//   run     <kernel> <buf> [<buf> ...]      opcode 3, buffers at args 3.. ; wait
//   dump    <buf> <file> [bytes [offset]]   read back to a file
//   copy    <dst> <dst_off> <src> <src_off> <bytes>
//   moeroute  <kernel> <rout-buf>           MoE expert fills -> the router's 8 experts
//   moeroute2 <kernel> <buf> <idx-offset>   ditto, pool-layout placeholder fills
//   attnpos <kernel> <pos>                  KV window / new-row / RoPE record for this token
//
// Relative paths resolve against the .cfg's directory. `#` starts a comment.
// Every `run` prints its ERT state and wall time (start -> wait), which is the
// number the benchmarks quote. Exit 1 on the first failure unless
// HARNESS_KEEP_GOING=1. A run that does not complete within HARNESS_TIMEOUT_MS
// (default 60000; 0 blocks forever) is reported as a failure rather than
// wedging the process — a kernel that hangs the array is a normal outcome when
// a design or an instruction patch is wrong.
//
// Run args follow the mlir-aie convention npu_matmul.cpp already uses: arg 0 =
// opcode 3, arg 1 = instruction BO, arg 2 = instruction word count, buffers
// from arg 3. The firmware rejects runs with too many buffer args (phlegm saw
// aborts at 9 and 14; 6 is known good) — keep designs at <= 8.
//
// `moeroute`/`moeroute2`/`attnpos` rewrite words of a `kernelx` instruction
// stream between runs — how a decode step feeds one shared program per layer
// type with this token's experts and cache position instead of rebuilding it.
// An mlir-aie instruction stream is a word sequence of ops; the ones that
// matter here are op 1 (a BD blockwrite: 8 registers from +4) and op 0x81 (a
// DDR patch: register at +6, host buffer arg index at +8, byte offset into
// that buffer at +10). Every weight fill and every cache transfer is one 0x81,
// so patching its offset word re-points the DMA without recompiling.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_kernel.h"
#include "xrt/experimental/xrt_module.h"
#include "xrt/experimental/xrt_xclbin.h"

#include "stream_patch.hpp"

namespace fs = std::filesystem;

namespace {

constexpr int kOpcode = 3;
constexpr size_t kBoAlign = 1u << 20;  // XDNA wants 1 MB-aligned buffer sizes

size_t padup(size_t n) { return (n + kBoAlign - 1) / kBoAlign * kBoAlign; }

using stream_patch::AttnPatch;
using stream_patch::MoePatch;

std::vector<uint8_t> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot read " + p.string());
    std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> v(static_cast<size_t>(n));
    if (n > 0 && !f.read(reinterpret_cast<char*>(v.data()), n))
        throw std::runtime_error("short read " + p.string());
    return v;
}

void write_file(const fs::path& p, const uint8_t* d, size_t n) {
    std::ofstream f(p, std::ios::binary);
    if (!f || !f.write(reinterpret_cast<const char*>(d), static_cast<std::streamsize>(n)))
        throw std::runtime_error("cannot write " + p.string());
}

struct Kernel {
    // classic (xclbin + insts.bin)
    std::unique_ptr<xrt::kernel> classic;
    std::unique_ptr<xrt::bo> instr;
    size_t nwords = 0;
    // ELF
    std::unique_ptr<xrt::elf> elf;
    std::unique_ptr<xrt::module> mod;
    std::unique_ptr<xrt::ext::kernel> ext;
    // classic only: the instruction stream as loaded, and the patch tables
    // derived from it once (scanning 100k+ words per token would show up).
    std::vector<uint32_t> words;
    std::vector<MoePatch> moe, moe2;
    std::vector<AttnPatch> attn;
    bool moe_built = false, moe2_built = false, attn_built = false;

    uint32_t* instr_words() { return instr->map<uint32_t*>(); }
};

struct Buf {
    xrt::bo bo;
    size_t size = 0;  // requested bytes (the BO itself is padded)
};

struct Host {
    fs::path base;
    std::unique_ptr<xrt::device> dev;
    std::map<std::string, xrt::hw_context> ctxs;
    std::map<std::string, Kernel> kernels;
    std::map<std::string, Buf> bufs;
    int runs = 0;
    bool keep_going = false;
    unsigned timeout_ms = 60000;

    fs::path resolve(const std::string& p) const {
        fs::path q(p);
        return q.is_absolute() ? q : base / q;
    }
    xrt::device& device() {
        if (!dev) throw std::runtime_error("no `device` line before use");
        return *dev;
    }
    xrt::hw_context& ctx(const std::string& n) {
        auto it = ctxs.find(n);
        if (it == ctxs.end()) throw std::runtime_error("no xclbin " + n);
        return it->second;
    }
    Kernel& kernel(const std::string& n) {
        auto it = kernels.find(n);
        if (it == kernels.end()) throw std::runtime_error("no kernel " + n);
        return it->second;
    }
    Buf& buf(const std::string& n) {
        auto it = bufs.find(n);
        if (it == bufs.end()) throw std::runtime_error("no buf " + n);
        return it->second;
    }

    static std::string need(std::istringstream& it, const char* what) {
        std::string s;
        if (!(it >> s)) throw std::runtime_error(std::string("missing ") + what);
        return s;
    }
    static size_t num(const std::string& s, const char* what) {
        try {
            return static_cast<size_t>(std::stoull(s));
        } catch (...) {
            throw std::runtime_error(std::string("bad ") + what + ": " + s);
        }
    }

    /// Read the router's int32 idx[8] out of a buffer.
    std::vector<uint32_t> read_route(const std::string& bufname, size_t off) {
        Buf& b = buf(bufname);
        if (off + 32 > padup(b.size)) throw std::runtime_error("route idx offset out of range");
        b.bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::vector<uint32_t> idx(8);
        std::memcpy(idx.data(), b.bo.map<uint8_t*>() + off, 32);
        for (uint32_t e : idx)
            if (e >= 256) throw std::runtime_error("route: expert index " + std::to_string(e) + " out of range");
        return idx;
    }

    // The patch tables are pure functions of the instruction words
    // (stream_patch.hpp); build each once per kernel.
    const std::vector<MoePatch>& moe_table(Kernel& k, const std::string& kn) {
        if (!k.moe_built) { k.moe = stream_patch::moe_table(k.words, kn); k.moe_built = true; }
        return k.moe;
    }
    const std::vector<MoePatch>& moe2_table(Kernel& k, const std::string& kn) {
        if (!k.moe2_built) { k.moe2 = stream_patch::moe2_table(k.words, kn); k.moe2_built = true; }
        return k.moe2;
    }
    const std::vector<AttnPatch>& attn_table(Kernel& k, const std::string& kn) {
        if (!k.attn_built) { k.attn = stream_patch::attn_table(k.words, kn); k.attn_built = true; }
        return k.attn;
    }

    // Returns false when a run failed and we are not keeping going.
    bool exec(const std::string& line) {
        std::istringstream it(line);
        std::string cmd;
        if (!(it >> cmd) || cmd[0] == '#') return true;

        if (cmd == "device") {
            dev = std::make_unique<xrt::device>(0u);
            std::printf("device: %s\n", dev->get_info<xrt::info::device::name>().c_str());
        } else if (cmd == "xclbin") {
            auto name = need(it, "xclbin name");
            auto path = resolve(need(it, "xclbin path"));
            xrt::xclbin xcl(path.string());
            auto uuid = device().register_xclbin(xcl);
            ctxs.emplace(name, xrt::hw_context(device(), uuid));
            std::printf("xclbin %s\n", name.c_str());
        } else if (cmd == "kernelx") {
            auto name = need(it, "kernelx name");
            auto xn = need(it, "kernelx xclbin");
            auto instp = resolve(need(it, "kernelx insts.bin"));
            Kernel k;
            k.classic = std::make_unique<xrt::kernel>(ctx(xn), "MLIR_AIE");
            auto insts = read_file(instp);
            if (insts.size() % 4) throw std::runtime_error("insts.bin not word-sized");
            k.nwords = insts.size() / 4;
            k.instr = std::make_unique<xrt::bo>(device(), insts.size(), xrt::bo::flags::cacheable,
                                                k.classic->group_id(1));
            std::memcpy(k.instr->map<void*>(), insts.data(), insts.size());
            k.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            k.words.resize(k.nwords);
            std::memcpy(k.words.data(), insts.data(), insts.size());
            std::printf("kernelx %s (%s, %zu words)\n", name.c_str(), instp.string().c_str(), k.nwords);
            kernels[name] = std::move(k);
        } else if (cmd == "kernel") {
            auto name = need(it, "kernel name");
            auto xn = need(it, "kernel xclbin");
            auto elfp = resolve(need(it, "kernel insts.elf"));
            Kernel k;
            k.elf = std::make_unique<xrt::elf>(elfp.string());
            k.mod = std::make_unique<xrt::module>(*k.elf);
            k.ext = std::make_unique<xrt::ext::kernel>(ctx(xn), *k.mod, "MLIR_AIE");
            std::printf("kernel %s (%s)\n", name.c_str(), elfp.string().c_str());
            kernels[name] = std::move(k);
        } else if (cmd == "buf") {
            auto name = need(it, "buf name");
            size_t size = num(need(it, "buf size"), "buf size");
            std::string initf;
            it >> initf;
            Buf b{xrt::ext::bo(device(), padup(size)), size};
            auto* m = b.bo.map<uint8_t*>();
            std::memset(m, 0, padup(size));
            if (!initf.empty()) {
                auto d = read_file(resolve(initf));
                if (d.size() > size)
                    throw std::runtime_error("buf " + name + ": init file larger than buffer");
                std::memcpy(m, d.data(), d.size());
            }
            b.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bufs.erase(name);
            bufs.emplace(name, std::move(b));
        } else if (cmd == "load") {
            auto name = need(it, "load buf");
            auto d = read_file(resolve(need(it, "load file")));
            Buf& b = buf(name);
            size_t n = d.size() < b.size ? d.size() : b.size;
            std::memcpy(b.bo.map<uint8_t*>(), d.data(), n);
            b.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        } else if (cmd == "run") {
            auto kn = need(it, "run kernel");
            std::vector<std::string> names;
            for (std::string s; it >> s;) names.push_back(s);
            if (names.empty()) throw std::runtime_error("run: needs at least one buffer");
            Kernel& k = kernel(kn);
            xrt::run r = k.classic ? xrt::run(*k.classic) : xrt::run(*k.ext);
            r.set_arg(0, kOpcode);
            if (k.classic) {
                r.set_arg(1, *k.instr);
                r.set_arg(2, static_cast<int>(k.nwords));
            } else {
                r.set_arg(1, 0);
                r.set_arg(2, 0);
            }
            for (size_t i = 0; i < names.size(); ++i) r.set_arg(static_cast<int>(3 + i), buf(names[i]).bo);
            auto t0 = std::chrono::steady_clock::now();
            r.start();
            auto st = timeout_ms ? r.wait(std::chrono::milliseconds(timeout_ms)) : r.wait();
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            ++runs;
            std::printf("run %s [%zu bufs] -> state %d (%.3f ms)\n", kn.c_str(), names.size(),
                        static_cast<int>(st), ms);
            if (st != ERT_CMD_STATE_COMPLETED) {
                std::printf("run %s FAILED (state %d)%s\n", kn.c_str(), static_cast<int>(st),
                            keep_going ? "; continuing (HARNESS_KEEP_GOING)" : "");
                return keep_going;
            }
        } else if (cmd == "dump") {
            auto name = need(it, "dump buf");
            auto outp = resolve(need(it, "dump file"));
            std::string s;
            size_t size = (it >> s) ? num(s, "dump size") : 0;
            size_t off = (it >> s) ? num(s, "dump offset") : 0;
            Buf& b = buf(name);
            if (size == 0) size = b.size;
            if (off + size > padup(b.size)) throw std::runtime_error("dump " + name + ": out of range");
            b.bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            write_file(outp, b.bo.map<uint8_t*>() + off, size);
        } else if (cmd == "copy") {
            auto dst = need(it, "copy dst");
            size_t doff = num(need(it, "copy dst_off"), "copy dst_off");
            auto src = need(it, "copy src");
            size_t soff = num(need(it, "copy src_off"), "copy src_off");
            size_t n = num(need(it, "copy nbytes"), "copy nbytes");
            Buf& s = buf(src);
            Buf& d = buf(dst);
            if (soff + n > padup(s.size) || doff + n > padup(d.size))
                throw std::runtime_error("copy: out of range");
            s.bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            std::memcpy(d.bo.map<uint8_t*>() + doff, s.bo.map<uint8_t*>() + soff, n);
            d.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        } else if (cmd == "moeroute" || cmd == "moeroute2") {
            bool v2 = cmd == "moeroute2";
            auto kn = need(it, "route kernel");
            auto rb = need(it, "route buf");
            // moeroute reads the router kernel's own output (int32 idx[8] at
            // byte 1024); moeroute2 takes the offset, since the fused layer
            // writes the router record into its activation buffer.
            size_t ioff = 1024;
            if (v2) ioff = num(need(it, "route idx offset"), "route idx offset");
            auto t0 = std::chrono::steady_clock::now();
            auto idx = read_route(rb, ioff);
            Kernel& k = kernel(kn);
            uint32_t* iw = k.instr_words();
            if (v2) stream_patch::moe2_apply(iw, moe2_table(k, kn), idx.data());
            else stream_patch::moe_apply(iw, moe_table(k, kn), idx.data());
            k.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("%s %s idx [%u %u %u %u %u %u %u %u] (%.3f ms)\n", cmd.c_str(), kn.c_str(), idx[0],
                        idx[1], idx[2], idx[3], idx[4], idx[5], idx[6], idx[7], ms);
        } else if (cmd == "attnpos") {
            // attnpos <kernel> <pos>: this token's cache position in the (shared)
            // ax0 stream — the window fill reads rows [0, max(pos, 1)), the new
            // row lands at row pos, and the RoPE record is ptab row pos. Three
            // words and one instruction-BO sync per token.
            auto kn = need(it, "attnpos kernel");
            size_t pos = num(need(it, "attnpos pos"), "attnpos pos");
            // The capacity is whatever the KV / ptab buffers were sized to (the
            // kernel only sees runtime-patched offsets); the ptab buffer is the
            // one declared in this program, so bound by it.
            if (Buf* pt = bufs.count("ptab") ? &buf("ptab") : nullptr; pt && (pos + 1) * stream_patch::kAttnPtabRow > pt->size)
                throw std::runtime_error("attnpos: pos " + std::to_string(pos) + " beyond the ptab buffer");
            auto t0 = std::chrono::steady_clock::now();
            Kernel& k = kernel(kn);
            stream_patch::attn_apply(k.instr_words(), attn_table(k, kn), pos);
            k.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("attnpos %s pos %zu (%.3f ms)\n", kn.c_str(), pos, ms);
        } else {
            throw std::runtime_error("unknown directive: " + cmd);
        }
        return true;
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: run_kernel <program.cfg>\n");
        return 2;
    }
    fs::path cfg = fs::absolute(argv[1]);
    std::ifstream f(cfg);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", cfg.string().c_str());
        return 2;
    }
    Host h;
    h.base = cfg.parent_path();
    if (const char* kg = std::getenv("HARNESS_KEEP_GOING")) h.keep_going = std::strcmp(kg, "1") == 0;
    if (const char* tm = std::getenv("HARNESS_TIMEOUT_MS")) h.timeout_ms = std::strtoul(tm, nullptr, 10);

    std::string line;
    int lineno = 0;
    try {
        while (std::getline(f, line)) {
            ++lineno;
            if (!h.exec(line)) {
                std::printf("RUN FAILED at line %d\n", lineno);
                return 1;
            }
        }
    } catch (const std::exception& e) {
        std::printf("ERROR line %d: %s\n  %s\n", lineno, e.what(), line.c_str());
        return 1;
    }
    std::printf("DONE runs=%d\n", h.runs);
    return 0;
}
