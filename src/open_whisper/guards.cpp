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

#include "kernels.hpp"
#include "open_qwen36/q4nx_file.hpp"

namespace ow {

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
