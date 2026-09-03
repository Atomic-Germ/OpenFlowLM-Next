/// \file npu_matmul.cpp
/// \brief Implementation of the NPU2 BF16 matmul backend.
/// \date 2026-08-30
///
/// Loads one xrt::device, one xrt::kernel and three buffers (A/B/C) per
/// compiled (K, N, padded-M) shape, then dispatches opcode-3 kernel runs
/// exactly like the validated mlir-aie host harness.
#include "npu_utils/npu_utils_matmul.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <unordered_map>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

namespace open_embedding {

namespace {

constexpr const char* kKernelName = "MLIR_AIE";
constexpr unsigned kOpcode = 3;  // ERT "start CU sequence" command.

bool read_file(const std::string& path, std::vector<char>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    std::streamsize n = f.tellg();
    f.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(n));
    if (n > 0) f.read(out.data(), n);
    return static_cast<bool>(f) || n == 0;
}

bool read_insts(const std::string& path, std::vector<uint32_t>& out) {
    std::vector<char> bytes;
    if (!read_file(path, bytes) || bytes.size() % 4 != 0) return false;
    out.resize(bytes.size() / 4);
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return true;
}

struct Shape {
    int m_pad = 0;
    std::vector<uint32_t> insts;
    std::unique_ptr<xrt::hw_context> context;
    std::unique_ptr<xrt::kernel> kernel;
    std::unique_ptr<xrt::bo> bo_instr;
    std::unique_ptr<xrt::bo> bo_a;
    std::unique_ptr<xrt::bo> bo_b;
    std::unique_ptr<xrt::bo> bo_c;
};

std::string shape_key(int K, int N) {
    return std::to_string(K) + "x" + std::to_string(N);
}

}  // namespace

struct NpuMatmul::Impl {
    std::mutex mu_;
    std::atomic<bool> enabled_{false};
    std::unique_ptr<xrt::device> device_;
    // (K, N) -> compiled shapes, ascending by padded M.
    std::unordered_map<std::string, std::vector<Shape>> shapes_;
};

NpuMatmul::NpuMatmul() : impl_(std::make_unique<Impl>()) {}

NpuMatmul::~NpuMatmul() = default;

bool NpuMatmul::init(const std::string& asset_dir, const std::string& device_id) {
    if (asset_dir.empty()) return false;

    // Locate every compiled shape artifact in the directory.
    struct Candidate {
        int m_pad, K, N;
        std::string xclbin, insts;
    };
    std::vector<Candidate> candidates;
    try {
        namespace fs = std::filesystem;
        for (const auto& entry : fs::directory_iterator(asset_dir)) {
            const std::string name = entry.path().filename().string();
            // Names look like "m512_768x768.xclbin".
            if (name.size() < 12 || name.substr(name.size() - 6) != "xclbin") continue;
            if (name[0] != 'm') continue;
            int m_pad = 0, K = 0, N = 0;
            if (std::sscanf(name.c_str(), "m%d_%dx%d.xclbin", &m_pad, &K, &N) != 3)
                continue;
            if (m_pad <= 0 || K <= 0 || N <= 0) continue;
            std::string base = entry.path().string();
            base.resize(base.size() - 7);  // drop ".xclbin"
            candidates.push_back(
                {m_pad, K, N, base + ".xclbin", base + ".insts"});
        }
    } catch (...) {
        return false;
    }
    if (candidates.empty()) {
        std::fprintf(stderr, "open_embedding: no NPU matmul artifacts in %s\n",
                     asset_dir.c_str());
        return false;
    }

    // Prefer an explicit device id, then fall back to automatic selection.
    std::string want = device_id.empty() ? "0000:c2:00.1" : device_id;
    try {
        impl_->device_ = std::make_unique<xrt::device>(want);
    } catch (...) {
        try {
            impl_->device_ = std::make_unique<xrt::device>(0);
        } catch (...) {
            std::fprintf(stderr,
                         "open_embedding: cannot open NPU device %s; CPU fallback\n",
                         want.c_str());
            return false;
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  if (a.m_pad != b.m_pad) return a.m_pad < b.m_pad;
                  if (a.K != b.K) return a.K < b.K;
                  return a.N < b.N;
              });

    for (const auto& cand : candidates) {
        try {
            Shape s;
            s.m_pad = cand.m_pad;
            if (!read_insts(cand.insts, s.insts)) {
                std::fprintf(stderr, "open_embedding: cannot read %s\n",
                             cand.insts.c_str());
                continue;
            }
            const auto xcl = xrt::xclbin(cand.xclbin);
            impl_->device_->register_xclbin(xcl);
            s.context = std::make_unique<xrt::hw_context>(*impl_->device_,
                                                          xcl.get_uuid());
            s.kernel =
                std::make_unique<xrt::kernel>(*s.context, kKernelName);
            s.bo_instr = std::make_unique<xrt::bo>(
                *impl_->device_, s.insts.size() * sizeof(uint32_t),
                XCL_BO_FLAGS_CACHEABLE, s.kernel->group_id(1));
            s.bo_a = std::make_unique<xrt::bo>(
                *impl_->device_, (size_t)s.m_pad * cand.K * sizeof(uint16_t),
                XRT_BO_FLAGS_HOST_ONLY, s.kernel->group_id(3));
            s.bo_b = std::make_unique<xrt::bo>(
                *impl_->device_, (size_t)cand.K * cand.N * sizeof(uint16_t),
                XRT_BO_FLAGS_HOST_ONLY, s.kernel->group_id(4));
            // Output buffer is now FP32 (bf16_f32 kernel)
            s.bo_c = std::make_unique<xrt::bo>(
                *impl_->device_, (size_t)s.m_pad * cand.N * sizeof(float),
                XRT_BO_FLAGS_HOST_ONLY, s.kernel->group_id(5));

            std::memcpy(s.bo_instr->map<void*>(), s.insts.data(),
                        s.insts.size() * sizeof(uint32_t));
            s.bo_instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);

            impl_->shapes_[shape_key(cand.K, cand.N)].push_back(std::move(s));
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                         "open_embedding: failed to load NPU shape %d %dx%d: %s\n",
                         cand.m_pad, cand.K, cand.N, e.what());
        }
    }

    if (impl_->shapes_.empty()) {
        std::fprintf(stderr, "open_embedding: no usable NPU matmul shapes; "
                             "CPU fallback\n");
        return false;
    }
    impl_->enabled_ = true;
    std::fprintf(stderr, "open_embedding: NPU matmul enabled (%zu shapes, %s)\n",
                 impl_->shapes_.size(), want.c_str());
    return true;
}

bool NpuMatmul::enabled() const { return impl_->enabled_.load(); }

int NpuMatmul::m_pad_for(int K, int N, int M) const {
    if (M <= 0) return 0;
    std::lock_guard<std::mutex> lock(impl_->mu_);
    auto it = impl_->shapes_.find(shape_key(K, N));
    if (it == impl_->shapes_.end()) return 0;
    for (const Shape& s : it->second)
        if (s.m_pad >= M) return s.m_pad;
    return 0;
}

bool NpuMatmul::matmul_bf16(int M, int K, int N, const uint16_t* a,
                            const uint16_t* b, float* c) {
    if (!impl_->enabled_.load()) return false;
    std::lock_guard<std::mutex> lock(impl_->mu_);
    auto it = impl_->shapes_.find(shape_key(K, N));
    if (it == impl_->shapes_.end() || M <= 0) return false;
    Shape* chosen = nullptr;
    for (Shape& s : it->second) {
        if (s.m_pad == M) {
            chosen = &s;
            break;
        }
    }
    if (!chosen) return false;
    Shape& s = *chosen;

    try {
        uint16_t* am = s.bo_a->map<uint16_t*>();
        uint16_t* bm = s.bo_b->map<uint16_t*>();
        float* cm = s.bo_c->map<float*>();
        std::memcpy(am, a, (size_t)M * K * sizeof(uint16_t));
        std::memcpy(bm, b, (size_t)K * N * sizeof(uint16_t));
        // Output buffer must be synced to device before launch (zero-padded)
        std::memset(cm, 0, (size_t)M * N * sizeof(float));
        s.bo_a->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s.bo_b->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s.bo_c->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto run = (*s.kernel)(kOpcode, *s.bo_instr,
                               static_cast<uint32_t>(s.insts.size()), *s.bo_a,
                               *s.bo_b, *s.bo_c);
        run.wait();
        s.bo_c->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::memcpy(c, cm, (size_t)M * N * sizeof(float));
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "open_embedding: NPU matmul dispatch failed: %s\n",
                     e.what());
        return false;
    }
}

}  // namespace open_embedding