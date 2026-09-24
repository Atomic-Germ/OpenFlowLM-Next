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
//   fdload  <buf> <fd> [bytes [offset]]     overwrite a buffer from an inherited fd (a pipe):
//                                           read exactly `bytes` (default: the buffer's size)
//                                           straight into the BO -- weights packed at load
//                                           time by the driver, no file (model/lax_pack.py)
//   run     <kernel> <buf> [<buf> ...]      opcode 3, buffers at args 3.. ; wait
//   dump    <buf> <file> [bytes [offset]]   read back to a file
//   copy    <dst> <dst_off> <src> <src_off> <bytes>
//   moeroute  <kernel> <rout-buf>           MoE expert fills -> the router's 8 experts
//   moeroute2 <kernel> <buf> <idx-offset>   ditto, pool-layout placeholder fills
//   attnpos <kernel> <pos>                  KV window / new-row / RoPE record for this token
//   feed <dst> <table> <id|last>            dst (f32 row) <- the table's bf16 row for a token id
//   greedy <logits> <n>                     argmax over the first n f32 logits -> `last`, printed
//   tick                                    wall ms since the previous tick
//   stopat <id>                             end the program here if `greedy` last picked <id>
//   poolbase <dst> <off> <src>              write src's device address (+0x80000000) into dst
//   ondvctrl <dst> <pool> <i0,..,i7>        fill dst with the on-device router's control
//                                           words for `pool` and those top-8 indices (the
//                                           same generator the router core runs)
//   runlist <name>                          empty runlist; runlist_add <name> <kernel>
//                                           <buf...> appends a run, runlist_exec <name>
//                                           executes the whole list ONCE and prints ms
//
// The NPU is shared: a run failure of the form "qds_device::wait() unexpected command state"
// is retried HARNESS_RETRY_CONTENTION times (default 0, 5 s apart) because that error also
// reproduces on known-good kernels while another process holds accel0.
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

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

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
#include "ondv_ctrl.h"   // designs/router: the on-device router's control-word generator

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
    std::string ctxname;   // whose hw_context a runlist of this kernel must use
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

// A runlist is bound to ONE hw_context, i.e. one xclbin UUID -- that is why the 35B's
// two layer types have to live in one xclbin before 40 layers can be one submit.
struct RunList {
    std::string ctxname;
    std::unique_ptr<xrt::runlist> rl;
    std::deque<xrt::run> runs;    // xrt::runlist::add does not take ownership, and a
                                  // vector's push_back reallocation would invalidate the
                                  // references the list holds (runlist_exec then reports
                                  // ERT_CMD_STATE_NEW); deque keeps them stable
};

struct Host {
    fs::path base;
    std::unique_ptr<xrt::device> dev;
    std::map<std::string, xrt::hw_context> ctxs;
    std::map<std::string, Kernel> kernels;
    std::map<std::string, Buf> bufs;
    std::map<std::string, RunList> runlists;
    int runs = 0;
    size_t last_token = 0;                              // `greedy`'s pick, `feed ... last`
    bool stopped = false;                               // `stopat` hit: end the program
    std::chrono::steady_clock::time_point tick_t = std::chrono::steady_clock::now();
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
        // Only the first topk slots are expert indices (moe_apply); bound them by
        // the routed-expert count the `moegeom` directive set.
        for (unsigned s = 0; s < mg.topk && s < idx.size(); ++s)
            if (idx[s] >= mg.experts)
                throw std::runtime_error("route: expert index " + std::to_string(idx[s]) + " out of range (" +
                                         std::to_string(mg.experts) + " experts)");
        return idx;
    }

    // The patch tables are pure functions of the instruction words
    // (stream_patch.hpp); build each once per kernel.
    // The pool / cache geometry the tables are decoded against: the 27B's by
    // default, set by the `attngeom` / `moegeom` directives (a manifest's values).
    stream_patch::AttnGeometry ag;
    stream_patch::MoeGeometry mg;

    const std::vector<MoePatch>& moe_table(Kernel& k, const std::string& kn) {
        if (!k.moe_built) { k.moe = stream_patch::moe_table(k.words, kn, mg); k.moe_built = true; }
        return k.moe;
    }
    const std::vector<MoePatch>& moe2_table(Kernel& k, const std::string& kn) {
        if (!k.moe2_built) { k.moe2 = stream_patch::moe2_table(k.words, kn, mg); k.moe2_built = true; }
        return k.moe2;
    }
    const std::vector<AttnPatch>& attn_table(Kernel& k, const std::string& kn) {
        if (!k.attn_built) { k.attn = stream_patch::attn_table(k.words, kn, ag); k.attn_built = true; }
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
            k.ctxname = xn;
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
            k.ctxname = xn;
            std::printf("kernel %s (%s)\n", name.c_str(), elfp.string().c_str());
            kernels[name] = std::move(k);
        } else if (cmd == "buf") {
            auto name = need(it, "buf name");
            size_t size = num(need(it, "buf size"), "buf size");
            std::string initf;
            it >> initf;
            Buf b{xrt::ext::bo(device(), padup(size)), size};
            std::printf("buf %s @ 0x%llx (%zu B)\n", name.c_str(),
                        static_cast<unsigned long long>(b.bo.address()), size);
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
        } else if (cmd == "fdload") {
            auto name = need(it, "fdload buf");
            int fd = static_cast<int>(num(need(it, "fdload fd"), "fdload fd"));
            Buf& b = buf(name);
            std::string sb, so;
            size_t n = (it >> sb) ? num(sb, "fdload bytes") : b.size;
            size_t off = (it >> so) ? num(so, "fdload offset") : 0;
            if (off + n > b.size) throw std::runtime_error("fdload " + name + ": range past the buffer");
            auto* m = b.bo.map<uint8_t*>() + off;
            for (size_t got = 0; got < n;) {
                ssize_t r = ::read(fd, m + got, n - got);
                if (r < 0 && errno == EINTR) continue;
                if (r <= 0)
                    throw std::runtime_error("fdload " + name + ": " + (r ? std::strerror(errno) : "EOF") +
                                             " after " + std::to_string(got) + " of " + std::to_string(n) + " B");
                got += static_cast<size_t>(r);
            }
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
        } else if (cmd == "poolbase") {
            // write a buffer's device address (XRT's DDR view = bo.address() + 0x8000_0000)
            // into another buffer: the fused kernel's router forms the retarget addresses
            // against the pool BO, and that address is only known once the BO exists.
            auto dst = need(it, "poolbase dst");
            size_t off = num(need(it, "poolbase offset"), "poolbase offset");
            auto src = need(it, "poolbase src");
            // Optional per-column shim-queue mask (bit c = column c on MM2S ch1). It rides
            // the low 8 bits of the pool address's low word -- the part of the cfg element
            // the emitter core provably receives -- because the pool BO is >= 1 MB aligned
            // (bits 0..19 zero) and cfg[2..9] does not reach the core (see
            // designs/router/ondv_ctrl_col.cc). Needed by the merged `lax` design, whose
            // single emitter core must retarget different channels per layer type.
            uint32_t qmask = 0;
            if (std::string qs; it >> qs)
                if (!qs.empty() && qs[0] != '#')
                    qmask = static_cast<uint32_t>(std::stoull(qs, nullptr, 0)) & 0xFFu;
            uint64_t addr = buf(src).bo.address() + 0x80000000ull;
            Buf& b = buf(dst);
            if (off + 8 > b.size) throw std::runtime_error("poolbase: dst buffer too small");
            uint32_t lo = static_cast<uint32_t>(addr) | qmask, hi = static_cast<uint32_t>(addr >> 32);
            auto* m = b.bo.map<uint8_t*>();
            std::memcpy(m + off, &lo, 4);
            std::memcpy(m + off + 4, &hi, 4);
            b.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            std::printf("poolbase %s+%zu <- %s @ 0x%llx (qmask 0x%02x)\n", dst.c_str(), off,
                        src.c_str(), static_cast<unsigned long long>(addr), qmask);
        } else if (cmd == "feed") {
            // feed <dst> <table> <id|last>: a token's embedding. The table buffer holds bf16 rows
            // of dst's width (dst is the f32 residual stream, `xres`).
            auto dst = need(it, "feed dst");
            auto tab = need(it, "feed table");
            auto ids = need(it, "feed id");
            size_t id = ids == "last" ? last_token : num(ids, "feed id");
            Buf& d = buf(dst);
            Buf& t = buf(tab);
            size_t w = d.size / 4;
            if ((id + 1) * w * 2 > t.size) throw std::runtime_error("feed: token id past the table");
            const uint16_t* row = t.bo.map<const uint16_t*>() + id * w;
            auto* o = d.bo.map<float*>();
            for (size_t i = 0; i < w; ++i) {
                uint32_t b = static_cast<uint32_t>(row[i]) << 16;
                std::memcpy(o + i, &b, 4);
            }
            d.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        } else if (cmd == "greedy") {
            auto lb = need(it, "greedy logits");
            size_t n = num(need(it, "greedy n"), "greedy n");
            Buf& b = buf(lb);
            if (n * 4 > b.size) throw std::runtime_error("greedy: n past the logits buffer");
            b.bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            const float* v = b.bo.map<const float*>();
            size_t best = 0;
            for (size_t i = 1; i < n; ++i)
                if (v[i] > v[best]) best = i;
            last_token = best;
            std::printf("greedy %zu\n", best);
            std::fflush(stdout);                        // a wrapper streams these as they come
        } else if (cmd == "tick") {
            auto now = std::chrono::steady_clock::now();
            std::printf("tick %.3f ms\n", std::chrono::duration<double, std::milli>(now - tick_t).count());
            tick_t = now;
            std::fflush(stdout);
        } else if (cmd == "stopat") {
            if (last_token == num(need(it, "stopat id"), "stopat id")) {
                stopped = true;
                std::printf("stop %zu\n", last_token);
            }
        } else if (cmd == "ondvctrl") {
            // Fill a buffer with the on-device router's control words using the SAME
            // generator the router core runs, for a given pool BO and top-8 index list.
            // This isolates the packet path (control BDs -> TileControl -> retarget +
            // enqueue) from the router core's ability to emit the words.
            auto dst = need(it, "ondvctrl dst");
            auto pl = need(it, "ondvctrl pool");
            std::string idxs;
            it >> idxs;
            int32_t idx[8];
            int k = 0;
            for (size_t pp = 0; pp < idxs.size() && k < 8;) {
                size_t q = idxs.find(',', pp);
                if (q == std::string::npos) q = idxs.size();
                idx[k++] = static_cast<int32_t>(std::strtol(idxs.substr(pp, q - pp).c_str(), nullptr, 0));
                pp = q + 1;
            }
            if (k != 8) throw std::runtime_error("ondvctrl: need 8 comma-separated indices");
            uint64_t addr = buf(pl).bo.address() + 0x80000000ull;
            Buf& b = buf(dst);
            if (b.size < 8u * 8u * 15u * 4u) throw std::runtime_error("ondvctrl: dst buffer too small");
            ondv_ctrl_impl(idx, static_cast<uint32_t>(addr), static_cast<uint32_t>(addr >> 32),
                           reinterpret_cast<int32_t*>(b.bo.map<uint8_t*>()));
            b.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            std::printf("ondvctrl %s <- pool %s @ 0x%llx, idx %s\n", dst.c_str(), pl.c_str(),
                        static_cast<unsigned long long>(addr), idxs.c_str());
        } else if (cmd == "runlist") {
            auto name = need(it, "runlist name");
            runlists.erase(name);
            runlists[name] = RunList{};
            std::printf("runlist %s (empty; its context is the first kernel added)\n", name.c_str());
        } else if (cmd == "runlist_add") {
            auto name = need(it, "runlist_add name");
            auto kn = need(it, "runlist_add kernel");
            std::vector<std::string> names;
            for (std::string s; it >> s;) names.push_back(s);
            if (names.empty()) throw std::runtime_error("runlist_add: needs at least one buffer");
            Kernel& k = kernel(kn);
            auto found = runlists.find(name);
            if (found == runlists.end()) throw std::runtime_error("runlist_add: no such runlist " + name);
            RunList& r = found->second;
            if (r.rl == nullptr) {
                r.ctxname = k.ctxname;
                r.rl = std::make_unique<xrt::runlist>(ctx(k.ctxname));
            } else if (r.ctxname != k.ctxname) {
                throw std::runtime_error("runlist_add: " + kn + " is in xclbin " + k.ctxname +
                                         " but runlist " + name + " is in " + r.ctxname +
                                         " -- a runlist is bound to ONE hw_context");
            }
            xrt::run run = k.classic ? xrt::run(*k.classic) : xrt::run(*k.ext);
            run.set_arg(0, kOpcode);
            if (k.classic) {
                run.set_arg(1, *k.instr);
                run.set_arg(2, static_cast<int>(k.nwords));
            } else {
                run.set_arg(1, 0);
                run.set_arg(2, 0);
            }
            for (size_t i = 0; i < names.size(); ++i) run.set_arg(static_cast<int>(3 + i), buf(names[i]).bo);
            r.runs.push_back(std::move(run));
            r.rl->add(r.runs.back());
        } else if (cmd == "runlist_exec") {
            auto name = need(it, "runlist_exec name");
            auto found = runlists.find(name);
            if (found == runlists.end() || found->second.rl == nullptr)
                throw std::runtime_error("runlist_exec: no such non-empty runlist " + name);
            RunList& r = found->second;
            auto t0 = std::chrono::steady_clock::now();
            r.rl->execute();
            r.rl->wait();   // throws on the first failing command in the list
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            ++runs;
            // Reaching here means the whole list completed: xrt::runlist::wait() throws on
            // any failing command. The individual run::state() is NOT updated by a runlist
            // (only the last command of the chain is polled), so it is reported for
            // information and is NOT the verdict -- counting it made every successful
            // runlist look "incomplete".
            int reporting = 0;
            for (auto& run : r.runs)
                if (run.state() == ERT_CMD_STATE_COMPLETED) ++reporting;
            // ONE execute for the whole list: that is the `one xrt::runlist submit` the
            // objective asks for, and %.3f ms is the per-token number.
            std::printf("runlist_exec %s [%zu runs, %d reporting completed] -> ok (%.3f ms)\n",
                        name.c_str(), r.runs.size(), reporting, ms);
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
            if (v2) stream_patch::moe2_apply(iw, moe2_table(k, kn), idx.data(), mg);
            else stream_patch::moe_apply(iw, moe_table(k, kn), idx.data(), mg);
            k.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("%s %s idx [%u %u %u %u %u %u %u %u] (%.3f ms)\n", cmd.c_str(), kn.c_str(), idx[0],
                        idx[1], idx[2], idx[3], idx[4], idx[5], idx[6], idx[7], ms);
        } else if (cmd == "attngeom") {
            // attngeom <kv_row> <ptab_row>: the KV cache row and position-record sizes of the
            // streams that follow (manifest.json layout.kv_row / ptab_row); default the 27B's.
            ag.kv_row = num(need(it, "attngeom kv_row"), "attngeom kv_row");
            ag.ptab_row = num(need(it, "attngeom ptab_row"), "attngeom ptab_row");
            std::string w;
            ag.window = (it >> w) ? num(w, "attngeom window") : 0;
            std::printf("attngeom kv_row %llu ptab_row %llu window %llu\n", (unsigned long long)ag.kv_row,
                        (unsigned long long)ag.ptab_row, (unsigned long long)ag.window);
        } else if (cmd == "moegeom") {
            // moegeom <experts> <topk> <stripe> <up_bytes> <down_core> <pool_down> <share_up> <share_gate> <share_down>
            mg.experts = (unsigned)num(need(it, "moegeom experts"), "moegeom");
            mg.topk = (unsigned)num(need(it, "moegeom topk"), "moegeom");
            mg.stripe = num(need(it, "moegeom stripe"), "moegeom");
            mg.up_bytes = num(need(it, "moegeom up_bytes"), "moegeom");
            mg.down_core = num(need(it, "moegeom down_core"), "moegeom");
            mg.pool_down = num(need(it, "moegeom pool_down"), "moegeom");
            mg.share_up = num(need(it, "moegeom share_up"), "moegeom");
            mg.share_gate = num(need(it, "moegeom share_gate"), "moegeom");
            mg.share_down = num(need(it, "moegeom share_down"), "moegeom");
            std::printf("moegeom set\n");
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
            if (Buf* pt = bufs.count("ptab") ? &buf("ptab") : nullptr; pt && (pos + 1) * ag.ptab_row > pt->size)
                throw std::runtime_error("attnpos: pos " + std::to_string(pos) + " beyond the ptab buffer");
            auto t0 = std::chrono::steady_clock::now();
            Kernel& k = kernel(kn);
            stream_patch::attn_apply(k.instr_words(), attn_table(k, kn), pos, ag);
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
        std::fprintf(stderr, "usage: run_kernel <program.cfg | ->\n");
        return 2;
    }
    // `-`: read the program from stdin as it is written, so a driver (model/lax_chat.py) can
    // keep the device state across prompts and decide each next line from the previous output
    const bool from_stdin = std::strcmp(argv[1], "-") == 0;
    fs::path cfg = from_stdin ? fs::current_path() / "stdin" : fs::absolute(argv[1]);
    std::ifstream file;
    if (!from_stdin) {
        file.open(cfg);
        if (!file) {
            std::fprintf(stderr, "cannot open %s\n", cfg.string().c_str());
            return 2;
        }
    }
    std::istream& f = from_stdin ? static_cast<std::istream&>(std::cin) : file;
    Host h;
    const char* rc_env = std::getenv("HARNESS_RETRY_CONTENTION");
    const int retry_contention = rc_env ? std::atoi(rc_env) : 0;
    h.base = cfg.parent_path();
    if (const char* kg = std::getenv("HARNESS_KEEP_GOING")) h.keep_going = std::strcmp(kg, "1") == 0;
    if (const char* tm = std::getenv("HARNESS_TIMEOUT_MS")) h.timeout_ms = std::strtoul(tm, nullptr, 10);

    std::string line;
    int lineno = 0;
    try {
        while (!h.stopped && std::getline(f, line)) {
            ++lineno;
            // The NPU is shared. Contention surfaces as xrt's "qds_device::wait()
            // unexpected command state" -- measured on the KNOWN-GOOD shipped lx0 kernel
            // while a peer's job held accel0 -- so a run failure of that shape is retried
            // (HARNESS_RETRY_CONTENTION, default 0) rather than reported as a design fault.
            int attempts = 1 + retry_contention;
            for (;;) {
                try {
                    if (h.exec(line)) break;
                    std::printf("RUN FAILED at line %d\n", lineno);
                    return 1;
                } catch (const std::exception& e) {
                    const std::string what = e.what();
                    if (--attempts > 0 && what.find("unexpected command state") != std::string::npos) {
                        std::printf("contention (\"%s\"); retrying line %d (%d attempt(s) left)\n",
                                    what.c_str(), lineno, attempts);
                        std::this_thread::sleep_for(std::chrono::seconds(5));
                        continue;
                    }
                    std::printf("ERROR line %d: %s\n  %s\n", lineno, e.what(), line.c_str());
                    return 1;
                }
            }
        }
    } catch (const std::exception& e) {
        std::printf("ERROR line %d: %s\n  %s\n", lineno, e.what(), line.c_str());
        return 1;
    }
    std::printf("DONE runs=%d\n", h.runs);
    return 0;
}
