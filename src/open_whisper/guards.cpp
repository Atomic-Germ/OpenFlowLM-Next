//===- guards.cpp --------------------------------------------*- C++ -*-===//
//
// open_whisper -- the refusals: the geometry a kernel set must have, and the
// dtype a weight tensor must have before it is read as bf16 bits. Deliberately
// its own translation unit, pulling in no XRT and no device, so guards_test.cpp
// can link it alone and every refusal is reachable without hardware.
// SPDX-License-Identifier: MIT
#include <stdexcept>
#include <string>

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

void require_bf16(const open_qwen36::Q4nxFile &f, const std::string &name) {
  if (!f.has(name))
    throw std::runtime_error("model.open.safetensors: missing tensor '" + name + "'");
  const std::string &dtype = f.meta(name).dtype;
  if (dtype != "BF16")
    throw std::runtime_error("model.open.safetensors: '" + name + "' is " + dtype +
                             ", expected BF16");
}

}  // namespace ow
