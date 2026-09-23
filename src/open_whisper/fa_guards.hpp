//===- fa_guards.hpp -----------------------------------------*- C++ -*-===//
//
// open_whisper -- OW_ATTN mode parsing/resolution and fa.json's geometry
// guard. Deliberately its own header with NO dependency on npu_device.hpp
// (unlike fa_attention.hpp): guards_test.cpp links guards.cpp (which
// implements these) alone, with no XRT lib and no device, so a misparsed
// OW_ATTN value or a mismatched fa.json is reachable without hardware --
// the same discipline as kernels.hpp's check_stream_shape /
// KernelSet::read_b_layout (see guards.cpp).
//
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <string>

namespace ow {

enum class AttnMode { Auto, Host, Npu };

// OW_ATTN parsing. `env_value` is exactly what std::getenv("OW_ATTN") would
// return (nullptr or a C string) -- passed in rather than read here so this
// is testable without touching the real environment. nullptr/empty is Auto
// (the default: use the NPU FlashAttention kernel when one is found next to
// the GEMM kernel set, else host). Any value other than "auto"/"host"/"npu"
// throws -- a misspelling must not silently resolve to Auto while the
// operator believes they pinned a mode (the same discipline as the old
// host-vs-anything-but-"npu" parser this replaces).
AttnMode parse_attn_mode(const char *env_value);
const char *to_string(AttnMode mode);

// mode=Auto: npu iff `fa_kernel_present`, silently -- "silently" only in the
// sense that it is not an error either way; the caller (encoder.cpp) always
// prints which one was chosen and why, never leaving the choice unlogged.
// mode=Npu: npu, REFUSING (throwing, naming `fa_dir`) if `fa_kernel_present`
// is false -- npu without a kernel must never fall back to host.
// mode=Host: never npu, regardless of `fa_kernel_present`.
bool resolve_use_npu_attn(AttnMode mode, bool fa_kernel_present, const std::string &fa_dir);

// The subset of fa.json this engine reads and checks against its own fixed
// geometry (Whisper-large-v3-turbo's encoder attention: H=20, dk=dv=64,
// lq=lk=1536 padded, valid_len=1500 real rows -- task 0180 Part 5/6).
// `mlir_aie_version`/`peano_version` are read and kept only for the log line
// (T39/trap 7c: a binary must name what built it) -- not checked, because
// this guard does not care which toolchain built the kernel (mlir-aie or
// MLIR-AIR): fa.json records whatever the build actually was.
struct FaKernelInfo {
  int64_t heads = 0, dk = 0, dv = 0, lq = 0, lk = 0, valid_len = 0;
  bool fp32_state = false;
  bool emulate_bfp16 = false;
  std::string mlir_aie_version, peano_version;
};

// Parses `<fa_dir>/fa.json`. Every field of FaKernelInfo above is REQUIRED:
// design.json's own read_b_layout has the same rule and the same reason --
// the caller checks this against its own compiled-in geometry, so an absent
// field defaulted here would be a guess checked against itself. Throws,
// naming the path and the missing/malformed field, on anything short of a
// complete record.
FaKernelInfo read_fa_kernel_info(const std::string &fa_dir);

// Refuses (throws, naming `where`, the mismatched field, and both values) if
// `info`'s geometry does not match this engine's fixed shape. `emulate_bfp16`
// and `fp32_state` are recorded, not checked here -- they are the kernel's
// numerics, not its shape, and either combination dispatches at the same
// buffer sizes.
void check_fa_geometry(const std::string &where, const FaKernelInfo &info);

}  // namespace ow
