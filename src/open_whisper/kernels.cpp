//===- kernels.cpp -------------------------------------------*- C++ -*-===//
// open_whisper -- see kernels.hpp. SPDX-License-Identifier: MIT
#include "kernels.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace ow {
namespace {

std::string read_file(const std::string &path) {
  std::ifstream fs(path, std::ios::binary);
  if (!fs) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << fs.rdbuf();
  return ss.str();
}

bool exists(const std::string &path) {
  std::ifstream fs(path);
  return static_cast<bool>(fs);
}

}  // namespace

std::string KernelSet::resolve_dir(const std::string &kernels_dir_hint,
                                   const std::string &model_dir) {
  if (!kernels_dir_hint.empty()) return kernels_dir_hint;
  if (const char *env = std::getenv("OFLM_WHISPER_KERNELS_DIR"))
    if (*env) return env;
  const std::string alongside = model_dir + "/open_kernels";
  if (exists(alongside + "/whisper_kernels.json")) return alongside;
  throw std::runtime_error(
      "no whisper kernel set found: pass --kernels DIR, set "
      "OFLM_WHISPER_KERNELS_DIR, or place one at " + alongside);
}

KernelSet::BLayout KernelSet::read_b_layout(const std::string &kernels_dir) {
  const nlohmann::json design_js = nlohmann::json::parse(read_file(kernels_dir + "/design.json"));
  if (!design_js.contains("b_layout"))
    throw std::runtime_error(kernels_dir + "/design.json: no b_layout");
  const auto &bl = design_js["b_layout"];
  BLayout out;
  out.tile_k = bl.value("tile_k", int64_t{0});
  out.tile_n = bl.value("tile_n", int64_t{0});
  out.mac_s = bl.value("mac_s", int64_t{8});
  out.mac_t = bl.value("mac_t", int64_t{8});
  if (out.tile_k <= 0 || out.tile_n <= 0)
    throw std::runtime_error(kernels_dir + "/design.json: b_layout tile_k/tile_n missing");
  return out;
}

KernelSet::KernelSet(npue::npu::Device &dev, const std::string &kernels_dir_hint,
                     const std::string &model_dir, int64_t weights_tile_k,
                     int64_t weights_tile_n, int64_t weights_mac_s,
                     int64_t weights_mac_t) {
  dir_ = resolve_dir(kernels_dir_hint, model_dir);

  const nlohmann::json marker = nlohmann::json::parse(read_file(dir_ + "/whisper_kernels.json"));
  const std::string format = marker.value("format", std::string());
  if (format != "oflm-open-whisper-kernels-v1")
    throw std::runtime_error(dir_ + "/whisper_kernels.json: format is '" + format +
                             "', expected 'oflm-open-whisper-kernels-v1'");
  if (!marker.value("complete", false))
    throw std::runtime_error(dir_ + "/whisper_kernels.json: 'complete' is not true -- "
                             "this kernel set was not finished exporting");

  // hf_config_check: every key must agree with model_dir/config.json, by
  // value, as text -- a kernel set built for a different checkpoint (fewer
  // layers, a different head count, ...) would otherwise dispatch happily
  // and return a plausible, wrong forward pass.
  const nlohmann::json cfg = nlohmann::json::parse(read_file(model_dir + "/config.json"));
  if (!marker.contains("hf_config_check") || !marker["hf_config_check"].is_object())
    throw std::runtime_error(dir_ + "/whisper_kernels.json: no hf_config_check object");
  for (const auto &kv : marker["hf_config_check"].items()) {
    const std::string &key = kv.key();
    if (!cfg.contains(key))
      throw std::runtime_error(dir_ + "/whisper_kernels.json: hf_config_check names '" +
                               key + "', which " + model_dir + "/config.json does not have");
    const std::string want = kv.value().dump();
    const std::string got = cfg[key].dump();
    if (want != got)
      throw std::runtime_error(dir_ + "/whisper_kernels.json: hf_config_check['" + key +
                               "'] = " + want + " but " + model_dir + "/config.json has " +
                               got + " -- this kernel set does not match this model");
  }

  const nlohmann::json design_js = nlohmann::json::parse(read_file(dir_ + "/design.json"));
  if (!design_js.contains("b_layout"))
    throw std::runtime_error(dir_ + "/design.json: no b_layout");
  const auto &bl = design_js["b_layout"];
  auto want_eq = [&](const char *key, int64_t want) {
    const int64_t got = bl.value(key, int64_t{-1});
    if (got != want)
      throw std::runtime_error(dir_ + "/design.json: b_layout['" + key + "'] = " +
                               std::to_string(got) + ", but the weights were tiled with " +
                               std::to_string(want) + " -- refusing to dispatch a design "
                               "whose B layout the loaded weights do not match");
  };
  want_eq("tile_k", weights_tile_k);
  want_eq("tile_n", weights_tile_n);
  want_eq("mac_s", weights_mac_s);
  want_eq("mac_t", weights_mac_t);
  const std::string order = bl.value("order", std::string());
  if (order.rfind("k,n", 0) != 0)
    throw std::runtime_error(dir_ + "/design.json: b_layout['order'] = '" + order +
                             "', expected it to begin 'k,n' (kb outer, nb inner) -- "
                             "the weights were tiled with that order and no other");

  design_ = std::make_unique<npue::npu::Design>(dev, dir_);

  // Slot 0 is whatever insts.bin the Design constructor loaded -- the kernel
  // set's own export always makes that conv1's stream (its file is
  // byte-identical to insts_conv1.bin). Every other op is loaded here, IN
  // THE ORDER design.json's own "streams" array lists them, and the slot
  // load_instr() returns is checked against design.json's declared slot: a
  // whisper_kernels.json/design.json drift would otherwise bind an op to the
  // wrong instruction stream silently.
  if (!design_js.contains("streams") || !design_js["streams"].is_array())
    throw std::runtime_error(dir_ + "/design.json: no streams array");
  for (const auto &s : design_js["streams"]) {
    const std::string op = s.value("op", std::string());
    Op which;
    if (op == "conv1") which = Op::Conv1;
    else if (op == "conv2") which = Op::Conv2;
    else if (op == "qkv") which = Op::Qkv;
    else if (op == "o") which = Op::O;
    else if (op == "fc1") which = Op::Fc1;
    else if (op == "fc2") which = Op::Fc2;
    else if (op == "xkv") which = Op::Xkv;
    else throw std::runtime_error(dir_ + "/design.json: unknown stream op '" + op + "'");

    StreamShape &sh = shapes_[static_cast<size_t>(which)];
    sh.M = s.value("M", int64_t{0});
    sh.K = s.value("K", int64_t{0});
    sh.N = s.value("N", int64_t{0});
    check_stream_shape(dir_ + "/design.json", which, sh.M, sh.K, sh.N);
    const int64_t declared_slot = s.value("slot", int64_t{-1});

    if (declared_slot == 0) {
      sh.instr_slot = 0;   // the design's own insts.bin
    } else {
      const std::string file = s.value("file", std::string());
      if (file.empty())
        throw std::runtime_error(dir_ + "/design.json: stream '" + op + "' has no file");
      const size_t got_slot = design_->load_instr(dir_ + "/" + file);
      if (static_cast<int64_t>(got_slot) != declared_slot)
        throw std::runtime_error(dir_ + "/design.json: stream '" + op +
                                 "' declares slot " + std::to_string(declared_slot) +
                                 " but loading its file in the recorded order gave slot " +
                                 std::to_string(got_slot));
      sh.instr_slot = got_slot;
    }
  }
  for (size_t i = 0; i < shapes_.size(); ++i)
    if (shapes_[i].M == 0)
      throw std::runtime_error(dir_ + "/design.json: stream '" +
                               op_name(static_cast<Op>(i)) + "' never appeared in streams[]");

  std::printf("  kernels    %s (mlir-aie %s, peano %s)\n", dir_.c_str(),
             design_->info().mlir_aie_version.c_str(),
             design_->info().peano_version.c_str());
}

size_t KernelSet::stage_b(const uint16_t *tiled, size_t elems) {
  return design_->stage(1, tiled, elems * sizeof(uint16_t));
}

const float *KernelSet::run(Op op, const uint16_t *a_bf16, size_t b_slot, double *t_in,
                            double *t_disp, double *t_out) {
  const StreamShape &sh = shapes_[static_cast<size_t>(op)];
  const size_t a_bytes = static_cast<size_t>(sh.M) * static_cast<size_t>(sh.K) * 2;
  const size_t c_bytes = static_cast<size_t>(sh.M) * static_cast<size_t>(sh.N) * 4;

  auto now = []() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  };
  double t0 = now();
  std::memcpy(design_->host_ptr(0), a_bf16, a_bytes);
  design_->sync_to_device(0, a_bytes);
  if (t_in) *t_in += now() - t0;

  t0 = now();
  design_->bind_instr(sh.instr_slot);
  design_->bind(0, 0);
  design_->bind(1, b_slot);
  design_->bind(2, 0);
  design_->dispatch_only();
  if (t_disp) *t_disp += now() - t0;

  // OW_DOUBLE_CHECK=1: dispatch the same bound buffers a second time and
  // compare the two results. A real encode's dispatches differ from the
  // stress loop's only in that A changes every time, so this catches a bad
  // dispatch in the act rather than inferring it from a wrong transcript.
  static const bool double_check = [] {
    const char *e = std::getenv("OW_DOUBLE_CHECK");
    return e && *e && *e != '0';
  }();
  if (double_check) {
    std::vector<float> c1(c_bytes / 4);
    design_->sync_from_device(2, c_bytes);
    std::memcpy(c1.data(), design_->host_ptr(2), c_bytes);
    design_->dispatch_only();
    design_->sync_from_device(2, c_bytes);
    const float *c2 = static_cast<const float *>(design_->host_ptr(2));
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < c1.size(); ++i)
      if (c1[i] != c2[i]) { if (!bad) first = i; ++bad; }
    if (bad)
      std::printf("  [double-check] %s: %zu of %zu C elements differ between two "
                  "dispatches of the SAME input; first at row %zu col %zu (%.6f vs %.6f)\n",
                  op_name(op), bad, c1.size(), first / static_cast<size_t>(sh.N),
                  first % static_cast<size_t>(sh.N), (double)c1[first], (double)c2[first]);
  }

  t0 = now();
  design_->sync_from_device(2, c_bytes);
  if (t_out) *t_out += now() - t0;

  // OW_VERIFY_A=1: read A back off the device and compare it with what we
  // handed over. The array is bit-deterministic (512 dispatches cycling four
  // streams and 32 weight slots agree byte for byte), and the corruption that
  // shows up in a real encode is a handful of WRONG ROWS -- so the question
  // this answers is whether the row the core computed from is the row we
  // uploaded. Diagnostic only: it costs a full A read-back per dispatch.
  static const bool verify_a = [] {
    const char *e = std::getenv("OW_VERIFY_A");
    return e && *e && *e != '0';
  }();
  if (verify_a) {
    std::vector<uint16_t> back(a_bytes / 2);
    design_->sync_from_device(0, a_bytes);
    std::memcpy(back.data(), design_->host_ptr(0), a_bytes);
    size_t bad_rows = 0, first_bad = 0, first_elem = 0;
    for (int64_t r = 0; r < sh.M; ++r) {
      const size_t off = static_cast<size_t>(r) * static_cast<size_t>(sh.K);
      if (std::memcmp(back.data() + off, a_bf16 + off,
                      static_cast<size_t>(sh.K) * 2) != 0) {
        if (bad_rows == 0) {
          first_bad = static_cast<size_t>(r);
          for (int64_t c = 0; c < sh.K; ++c)
            if (back[off + c] != a_bf16[off + c]) { first_elem = static_cast<size_t>(c); break; }
        }
        ++bad_rows;
      }
    }
    if (bad_rows)
      std::printf("  [verify A] %s: %zu of %lld rows read back changed; first row %zu col %zu\n",
                  op_name(op), bad_rows, (long long)sh.M, first_bad, first_elem);
  }

  return static_cast<const float *>(design_->host_ptr(2));
}

}  // namespace ow
