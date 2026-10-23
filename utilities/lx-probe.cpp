// Standalone NPU dispatch of one layer_x / dense-stream kernel context.
//
// The engine (src/open_qwen36) pins five DDR buffers per layer and submits ONE
// xrt run per program step; this probe does exactly that for a single set of
// file fixtures, so a wedging instruction stream hangs here without the
// engine's model-load machinery in the picture:
//
//     source utilities/env.sh
//     g++ -O2 -std=c++20 -o lx-probe utilities/lx-probe.cpp \
//         -I$XILINX_XRT/include -L$XILINX_XRT/lib -lxrt_coreutil
//     ./lx-probe lx_f32/final.xclbin lx_f32/insts.bin \
//         pool0.f32.bin xres.bin consts0.f32.bin - act_out.f32.bin \
//         157286400 16384 13197312 2211840 253952 30
//
// usage: lx-probe <xclbin> <insts.bin> <pool> <xres> <consts> <state> <act_out>
//                 <pool_bytes> <xres_bytes> <consts_bytes> <state_bytes>
//                 <act_bytes> <wait_s>
//
// Generic mode (any run argument count -- ax, the dense tail, future sets):
//
//     lx-probe -g <xclbin> <insts.bin> <out_path> <wait_s> <in_bytes...>
//
// `in_bytes` are the byte sizes of every buffer in the stream's run-argument
// order; every buffer starts as zeros and the LAST one is copied out to
// <out_path> on completion (hang tests never need live data).
//
// The five run arguments are given in the manifest program's stream order
// (pool xres consts state act; act is only written out, it starts as zeros).
// The submission mirrors Core::start_run exactly: instr BO at kernel
// group_id(1), opcode 3 at arg 0, word count at arg 2, the buffers at 3..7.
// An input file that is missing or "-" starts as zeros: the DMA choreography
// does not need valid weights to wedge or to complete, so a hang test runs
// fine with everything "-". Exit 0 and an act copy on completion; 124 when
// the wait times out.
#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

// only ships on Linux; on Windows the phlegm shim loads the same streams.
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_ext.h>

namespace {

constexpr int kOpcode = 3;  // Core::kOpcode in src/open_qwen36/core.cpp

// One dispatch of the instruction stream against files: `paths[i]` is the
// fixture for run argument 3+i (empty = zeros); the LAST buffer is dumped to
// `out_path` on completion. `sizes` are the buffers' byte sizes in the same
// order, minus the trailing act... -- as many as the stream's run args.

std::vector<uint8_t> read_all(const std::string& path, size_t bytes) {
    std::vector<uint8_t> v(bytes, 0);
    if (path.empty() || path == "-") return v;
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "lx-probe: cannot open %s (starting zeros)\n", path.c_str());
        return v;
    }
    size_t got = 0;
    while (got < bytes) {
        ssize_t n = ::read(fd, v.data() + got, bytes - got);
        if (n < 0) {
            if (errno != EINTR) {
                std::fprintf(stderr, "lx-probe: read error on %s\n", path.c_str());
                break;
            }
        } else if (n == 0) {
            std::fprintf(stderr, "lx-probe: EOF: %s shorter than %zu B\n", path.c_str(), bytes);
            break;
        } else {
            got += size_t(n);
        }
    }
    ::close(fd);
    return v;
}

int probe(const char* xcl, const char* insts_path, const char* out_path, double wait_s,
          const std::vector<std::string>& paths, const std::vector<size_t>& sizes_all) {
    try {
        xrt::device dev(0);
        xrt::xclbin xclbin{std::string(xcl)};
        auto uuid = dev.register_xclbin(xclbin);
        xrt::hw_context ctx(dev, uuid);
        xrt::kernel k(ctx, "MLIR_AIE");

        std::vector<uint32_t> words;
        {
            std::ifstream fi(insts_path, std::ios::binary);
            if (!fi) {
                std::fprintf(stderr, "lx-probe: cannot open %s\n", insts_path);
                return 2;
            }
            std::vector<char> raw((std::istreambuf_iterator<char>(fi)), std::istreambuf_iterator<char>());
            std::string bad = raw.empty() || raw.size() % 4 ? " is not word-sized" : "";
            if (!bad.empty()) {
                std::fprintf(stderr, "lx-probe: %s%s\n", insts_path, bad.c_str());
                return 2;
            }
            words.resize(raw.size() / 4);
            std::memcpy(words.data(), raw.data(), raw.size());

            // Core::load_kernel: for each context, the instr BO is cacheable and sits at group 1.
            xrt::bo instr(dev, raw.size(), xrt::bo::flags::cacheable, k.group_id(1));
            std::memcpy(instr.map<uint8_t*>(), raw.data(), raw.size());
            instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            static xrt::bo keep_instr_alive = instr;

            const size_t n = sizes_all.size();
            // paths.empty() = the generic mode: every buffer starts as zeros and
            // the LAST one is copied out. Otherwise paths[i] is the fixture for
            // run argument 3+i and the last buffer is write-out only.
            const size_t n_in = paths.empty() ? 0 : paths.size();
            // Core::alloc: one ext::bo per run argument, zero-filled, host-loaded, synced in.
            std::vector<std::optional<xrt::ext::bo>> bufs(n);
            for (size_t i = 0; i < n; ++i) {
                bufs[i].emplace(dev, sizes_all[i]);
                auto* m = bufs[i]->map<uint8_t*>();
                std::memset(m, 0, sizes_all[i]);
                if (i < n_in) {
                    auto v = read_all(paths[i].c_str(), sizes_all[i]);
                    std::memcpy(m, v.data(), v.size());
                }
                bufs[i]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            }

            xrt::run r(k);
            r.set_arg(0, kOpcode);                              // Core::start_run: opcode, instr, nwords
            r.set_arg(1, instr);
            r.set_arg(2, static_cast<int>(words.size()));
            for (size_t i = 0; i < n; ++i) r.set_arg(3 + i, *bufs[i]);
            r.start();

            // Core::wait_run: a blocking wait with a timeout, then report the state.
            const auto st = r.wait(std::chrono::milliseconds(long(wait_s * 1000)));
            std::printf("lx-probe: run state %d (%s)\n", int(st),
                        st == ERT_CMD_STATE_COMPLETED ? "completed"
                        : st == ERT_CMD_STATE_TIMEOUT ? "timeout"
                        : st == ERT_CMD_STATE_ERROR ? "error" : "other");
            if (st != ERT_CMD_STATE_COMPLETED) return 124;

            bufs[n - 1]->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            std::ofstream fo(out_path, std::ios::binary);
            fo.write(reinterpret_cast<const char*>(bufs[n - 1]->map<uint8_t*>()), std::streamsize(sizes_all[n - 1]));
            std::printf("lx-probe: act -> %s (%zu B)\n", out_path, sizes_all[n - 1]);
            return 0;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "lx-probe: %s\n", e.what());
        return 1;
    }
}
}  // namespace

int main(int argc, char** argv) {
    // Generic mode: lx-probe -g <xclbin> <insts.bin> <out_path> <wait_s> <in_bytes...>
    bool generic = argc > 3 && std::string(argv[1]) == "-g";
    if (generic) {
        if (argc < 6) {
            std::fprintf(stderr, "usage: lx-probe -g <xclbin> <insts.bin> <out_path> <wait_s> <in_bytes...>\n");
            return 2;
        }
        std::vector<std::string> rest{argv + 2, argv + argc};   // xclbin insts out wait sizes...
        std::vector<size_t> sizes;
        for (size_t i = 4; i < rest.size(); ++i) sizes.push_back(std::stoul(rest[i]));
        return probe(rest[0].c_str(), rest[1].c_str(), rest[2].c_str(), std::stod(rest[3]), {}, sizes);
    }
    if (argc < 14) {
        std::fprintf(stderr,
                     "usage: lx-probe <xclbin> <insts.bin> <pool> <xres> <consts> <state> <act_out>\n"
                     "              <pool_bytes> <xres_bytes> <consts_bytes> <state_bytes>\n"
                     "              <act_bytes> <wait_s>\n"
                     "       lx-probe -g <xclbin> <insts.bin> <out_path> <wait_s> <in_bytes...>\n");
        return 2;
    }
    const char* xcl = argv[1];
    const char* insts_path = argv[2];
    std::vector<std::string> paths{argv[3], argv[4], argv[5], argv[6]};  // pool xres consts state
    const std::vector<size_t> sizes = {std::stoul(argv[8]), std::stoul(argv[9]), std::stoul(argv[10]),
                                       std::stoul(argv[11]), std::stoul(argv[12])};  // ... act
    const char* act_out = argv[7];
    const double wait_s = std::stod(argv[13]);
    return probe(xcl, insts_path, act_out, wait_s, paths, sizes);
}
