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

std::vector<uint8_t> read_all(const std::string& path, size_t bytes) {
    std::vector<uint8_t> v(bytes, 0);
    if (path == "-" || path.empty()) return v;
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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 14) {
        std::fprintf(stderr,
                     "usage: lx-probe <xclbin> <insts.bin> <pool> <xres> <consts> <state> <act_out>\n"
                     "              <pool_bytes> <xres_bytes> <consts_bytes> <state_bytes>\n"
                     "              <act_bytes> <wait_s>\n");
        return 2;
    }
    const char* xcl = argv[1];
    const char* insts_path = argv[2];
    const char* in4[] = {argv[3], argv[4], argv[5], argv[6]};  // pool xres consts state
    const size_t sizes[5] = {std::stoul(argv[8]), std::stoul(argv[9]), std::stoul(argv[10]),
                             std::stoul(argv[11]), std::stoul(argv[12])};  // pool xres consts state act
    const char* act_out = argv[7];
    const double wait_s = std::stod(argv[13]);

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

            // Core::alloc: one ext::bo per run argument, zero-filled, host-loaded, synced in.
            std::optional<xrt::ext::bo> bufs[5];
            for (int i = 0; i < 5; ++i) {
                bufs[i].emplace(dev, sizes[i]);
                auto* m = bufs[i]->map<uint8_t*>();
                std::memset(m, 0, sizes[i]);
                if (i < 4) {
                    auto v = read_all(in4[i], sizes[i]);
                    std::memcpy(m, v.data(), v.size());
                }
                bufs[i]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            }

            xrt::run r(k);
            r.set_arg(0, kOpcode);                              // Core::start_run: opcode, instr, nwords
            r.set_arg(1, instr);
            r.set_arg(2, static_cast<int>(words.size()));
            for (int i = 0; i < 5; ++i) r.set_arg(3 + i, *bufs[i]);
            r.start();

            // Core::wait_run: a blocking wait with a timeout, then report the state.
            const auto st = r.wait(std::chrono::milliseconds(long(wait_s * 1000)));
            std::printf("lx-probe: run state %d (%s)\n", int(st),
                        st == ERT_CMD_STATE_COMPLETED ? "completed"
                        : st == ERT_CMD_STATE_TIMEOUT ? "timeout"
                        : st == ERT_CMD_STATE_ERROR ? "error" : "other");
            if (st != ERT_CMD_STATE_COMPLETED) return 124;

            bufs[4]->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            std::ofstream fo(act_out, std::ios::binary);
            fo.write(reinterpret_cast<const char*>(bufs[4]->map<uint8_t*>()), std::streamsize(sizes[4]));
            std::printf("lx-probe: act -> %s (%zu B)\n", act_out, sizes[4]);
            return 0;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "lx-probe: %s\n", e.what());
        return 1;
    }
}
