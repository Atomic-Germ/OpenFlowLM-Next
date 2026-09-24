//===- guards.cpp --------------------------------------------*- C++ -*-===//
//
// open_whisper -- the refusals: the geometry a kernel set must have, and the
// dtype a weight tensor must have before it is read as bf16 bits. Deliberately
// its own translation unit, pulling in no XRT and no device, so guards_test.cpp
// can link it alone and every refusal is reachable without hardware.
// SPDX-License-Identifier: MIT
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "nlohmann/json.hpp"

#include "fa_guards.hpp"
#include "kernels.hpp"
#include "open_qwen36/q4nx_file.hpp"

namespace ow {

// This engine's fixed FlashAttention shape (see fa_attention.hpp's own
// header comment): H=20, dk=dv=64 (Geometry::n_heads/head_dim), lq=lk=1536
// (the M every encoder-layer GEMM already shares), valid_len=1500
// (Geometry::max_src_pos). Duplicated as literals rather than pulling in
// weights.hpp's Geometry: this translation unit is deliberately XRT- and
// container-free (guards_test.cpp links it alone), and these four numbers
// are Whisper-large-v3-turbo's architecture, not a build-time constant --
// encoder.cpp's own Geometry:: constants are asserted equal to these by
// fa_attention.cpp's check_shape() at every call.
namespace {
constexpr int64_t kFaHeads = 20, kFaDk = 64, kFaDv = 64, kFaLq = 1536, kFaLk = 1536,
                  kFaValidLen = 1500;
}  // namespace

AttnMode parse_attn_mode(const char *env_value) {
  if (!env_value || !*env_value) return AttnMode::Auto;
  const std::string v(env_value);
  if (v == "auto") return AttnMode::Auto;
  if (v == "host") return AttnMode::Host;
  if (v == "npu") return AttnMode::Npu;
  throw std::runtime_error("OW_ATTN is '" + v + "': expected 'auto', 'host' or 'npu'");
}

const char *to_string(AttnMode mode) {
  switch (mode) {
    case AttnMode::Auto: return "auto";
    case AttnMode::Host: return "host";
    case AttnMode::Npu:  return "npu";
  }
  return "?";
}

bool resolve_use_npu_attn(AttnMode mode, bool fa_kernel_usable, const std::string &fa_dir,
                          const std::string &reason) {
  switch (mode) {
    case AttnMode::Host: return false;
    case AttnMode::Auto: return fa_kernel_usable;
    case AttnMode::Npu:
      if (!fa_kernel_usable)
        throw std::runtime_error(
            "OW_ATTN=npu: no usable FlashAttention kernel at " + fa_dir + " (" +
            (reason.empty()
                 ? "need air.xclbin, air.insts.bin and fa.json -- set OW_FA_DIR, or "
                   "build one into the kernel set's fa/ subdirectory"
                 : reason) +
            ")");
      return true;
  }
  return false;
}

bool fa_kernel_usable(const std::string &fa_dir, std::string &reason) {
  // File presence only (fa.json's own content is not read here): a directory
  // missing any of the three files is simply "no kernel", the same message
  // fa_kernel_present() (fa_attention.cpp) already gives for that case.
  auto file_present = [](const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
  };
  if (!file_present(fa_dir + "/air.xclbin") || !file_present(fa_dir + "/air.insts.bin") ||
      !file_present(fa_dir + "/fa.json")) {
    reason = "no FlashAttention kernel at " + fa_dir;
    return false;
  }
  // All three files exist -- now the part fa_kernel_present() skipped: does
  // fa.json actually parse, and does it name this engine's own geometry? Both
  // read_fa_kernel_info and check_fa_geometry throw with a specific, actionable
  // message (the malformed/missing field, or the mismatched field and both
  // values); caught here so a stale/wrong-shaped fa/ resolves to "not usable",
  // not to an exception the Auto path has no chance to catch.
  try {
    const FaKernelInfo info = read_fa_kernel_info(fa_dir);
    check_fa_geometry("FA kernel at " + fa_dir, info);
  } catch (const std::exception &e) {
    reason = e.what();
    return false;
  }
  return true;
}

FaKernelInfo read_fa_kernel_info(const std::string &fa_dir) {
  const std::string path = fa_dir + "/fa.json";
  std::ifstream fs(path, std::ios::binary);
  if (!fs) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << fs.rdbuf();
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(ss.str());
  } catch (const nlohmann::json::exception &e) {
    throw std::runtime_error(path + ": invalid JSON: " + e.what());
  }

  auto need_int = [&](const char *key) -> int64_t {
    if (!j.contains(key) || !j[key].is_number_integer())
      throw std::runtime_error(path + ": '" + key +
                               "' is missing -- the build must record what it actually built, "
                               "not have the engine assume it");
    return j[key].get<int64_t>();
  };
  auto need_bool = [&](const char *key) -> bool {
    if (!j.contains(key) || !j[key].is_boolean())
      throw std::runtime_error(path + ": '" + key + "' is missing or not a boolean");
    return j[key].get<bool>();
  };
  auto need_str = [&](const char *key) -> std::string {
    if (!j.contains(key) || !j[key].is_string())
      throw std::runtime_error(path + ": '" + key + "' is missing or not a string");
    return j[key].get<std::string>();
  };

  FaKernelInfo info;
  info.heads = need_int("heads");
  info.dk = need_int("dk");
  info.dv = need_int("dv");
  info.lq = need_int("lq");
  info.lk = need_int("lk");
  info.valid_len = need_int("valid_len");
  info.fp32_state = need_bool("fp32_state");
  info.emulate_bfp16 = need_bool("emulate_bfp16");
  info.mlir_aie_version = need_str("mlir_aie_version");
  info.peano_version = need_str("peano_version");
  return info;
}

void check_fa_geometry(const std::string &where, const FaKernelInfo &info) {
  auto want = [&](const char *field, int64_t have, int64_t expect) {
    if (have != expect)
      throw std::runtime_error(where + ": fa.json's '" + field + "' is " +
                               std::to_string(have) + ", but this engine is built for " +
                               std::to_string(expect) +
                               " -- refusing to dispatch a FlashAttention kernel of a "
                               "different geometry");
  };
  want("heads", info.heads, kFaHeads);
  want("dk", info.dk, kFaDk);
  want("dv", info.dv, kFaDv);
  want("lq", info.lq, kFaLq);
  want("lk", info.lk, kFaLk);
  want("valid_len", info.valid_len, kFaValidLen);
}

const char *op_name(Op op) {
  switch (op) {
    case Op::Conv1: return "conv1";
    case Op::Conv2: return "conv2";
    case Op::Qkv:   return "qkv";
    case Op::O:     return "o";
    case Op::Fc1:   return "fc1";
    case Op::Fc2:   return "fc2";
    case Op::Xkv:   return "xkv";
    default: return "?";
  }
}

StreamShape expected_shape(Op op) {
  StreamShape s;
  switch (op) {
    case Op::Conv1: s.M = 3072; s.K =  384; s.N =  1280; break;   // im2col, 3 x 128 mel taps
    case Op::Conv2: s.M = 1536; s.K = 3840; s.N =  1280; break;   // im2col, stride 2
    case Op::Qkv:   s.M = 1536; s.K = 1280; s.N =  3840; break;   // Q|K|V fused
    case Op::O:     s.M = 1536; s.K = 1280; s.N =  1280; break;
    case Op::Fc1:   s.M = 1536; s.K = 1280; s.N =  5120; break;
    case Op::Fc2:   s.M = 1536; s.K = 5120; s.N =  1280; break;
    case Op::Xkv:   s.M = 1536; s.K = 1280; s.N = 10240; break;   // 4 decoder layers' K|V
    default: break;
  }
  return s;
}

void check_stream_shape(const std::string &where, Op op, int64_t M, int64_t K, int64_t N) {
  const StreamShape w = expected_shape(op);
  if (M != w.M || K != w.K || N != w.N)
    throw std::runtime_error(
        where + ": stream '" + op_name(op) + "' is " + std::to_string(M) + "x" +
        std::to_string(K) + "x" + std::to_string(N) + ", but this engine is built for " +
        std::to_string(w.M) + "x" + std::to_string(w.K) + "x" + std::to_string(w.N) +
        " -- refusing to dispatch against a kernel set of a different geometry");
}

// The B layout the weights must be tiled with, read from design.json. Every one of the
// four fields is REQUIRED: the caller tiles its weights with what this returns and the
// KernelSet constructor then compares design.json against those same values, so a
// defaulted field would be a guess checked against itself. (It is not one today -- the
// constructor's own reader defaults to -1, so an absent field mismatches and is refused --
// but that is one edit away from being true, and a tiling tuple nobody wrote down is not
// a tuple.)
KernelSet::BLayout KernelSet::read_b_layout(const std::string &kernels_dir) {
  const std::string path = kernels_dir + "/design.json";
  std::ifstream fs(path, std::ios::binary);
  if (!fs) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << fs.rdbuf();
  const nlohmann::json design_js = nlohmann::json::parse(ss.str());
  if (!design_js.contains("b_layout"))
    throw std::runtime_error(path + ": no b_layout");
  const auto &bl = design_js["b_layout"];
  auto need = [&](const char *key) -> int64_t {
    if (!bl.contains(key) || !bl[key].is_number_integer())
      throw std::runtime_error(std::string(path) + ": b_layout['" + key +
                               "'] is missing -- the tiling tuple must be recorded, not assumed");
    const int64_t v = bl[key].get<int64_t>();
    if (v <= 0)
      throw std::runtime_error(std::string(path) + ": b_layout['" + key + "'] is " +
                               std::to_string(v) + ", expected a positive value");
    return v;
  };
  BLayout out;
  out.tile_k = need("tile_k");
  out.tile_n = need("tile_n");
  out.mac_s = need("mac_s");
  out.mac_t = need("mac_t");
  return out;
}

void require_bf16(const open_qwen36::Q4nxFile &f, const std::string &name) {
  if (!f.has(name))
    throw std::runtime_error("model.open.safetensors: missing tensor '" + name + "'");
  const std::string &dtype = f.meta(name).dtype;
  if (dtype != "BF16")
    throw std::runtime_error("model.open.safetensors: '" + name + "' is " + dtype +
                             ", expected BF16");
}

}  // namespace ow
