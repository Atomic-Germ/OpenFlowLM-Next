//===- guards_test.cpp ----------------------------------------*- C++ -*-===//
//
// The open Whisper engine's REFUSALS, tested without a device and without the
// 1.6 GB container: a kernel set whose recorded geometry is not the one this
// engine dispatches for, and a weight tensor whose dtype is not the one the
// loader is about to read it as. Both are silent failures if they are not
// refused -- the first transfers the wrong number of bytes, the second returns
// finite, plausible, wrong logits.
//
//   out\guards_test.exe          (no arguments, no NPU, no model)
//
// SPDX-License-Identifier: MIT
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "kernels.hpp"
#include "open_qwen36/q4nx_file.hpp"

namespace {

int failures = 0;

void check(bool ok, const std::string &what) {
  std::printf("  %-62s %s\n", what.c_str(), ok ? "ok" : "FAIL");
  if (!ok) ++failures;
}

// Runs `fn` and reports whether it threw, and whether the message names `needle`
// (an error nobody can act on is barely better than no error).
template <typename F>
void expect_throw(F fn, const std::string &needle, const std::string &what) {
  try {
    fn();
  } catch (const std::exception &e) {
    const bool named = std::string(e.what()).find(needle) != std::string::npos;
    check(named, what + " (names '" + needle + "')");
    return;
  }
  check(false, what + " -- did not throw");
}

template <typename F>
void expect_ok(F fn, const std::string &what) {
  try {
    fn();
    check(true, what);
  } catch (const std::exception &e) {
    std::printf("    threw: %s\n", e.what());
    check(false, what);
  }
}

// A minimal safetensors file: 8-byte header length, JSON header, then the data.
// Same layout q4nx-build writes, so Q4nxFile reads it unchanged.
void write_safetensors(const std::string &path, const std::string &name, const std::string &dtype,
                       const std::vector<size_t> &shape, const std::vector<uint8_t> &data) {
  std::string shape_s;
  for (size_t i = 0; i < shape.size(); ++i) shape_s += (i ? "," : "") + std::to_string(shape[i]);
  std::string header = "{\"" + name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":[" + shape_s +
                       "],\"data_offsets\":[0," + std::to_string(data.size()) + "]}}";
  header.append((8 - header.size() % 8) % 8, ' ');
  const uint64_t n = header.size();
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char *>(&n), 8);
  f.write(header.data(), static_cast<std::streamsize>(header.size()));
  f.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
}

void test_stream_shapes() {
  std::printf("-- kernel set geometry --\n");
  // Every stream's own recorded shape is accepted...
  for (size_t i = 0; i < static_cast<size_t>(ow::Op::Count); ++i) {
    const ow::Op op = static_cast<ow::Op>(i);
    const ow::StreamShape w = ow::expected_shape(op);
    expect_ok([&] { ow::check_stream_shape("design.json", op, w.M, w.K, w.N); },
              std::string("accepts ") + ow::op_name(op) + "'s own shape");
  }
  // ... and one wrong dimension is not, in any position. K is the dangerous one:
  // run() takes the A transfer size from it, and the caller's buffer is sized from
  // the geometry the engine was built for.
  const ow::StreamShape q = ow::expected_shape(ow::Op::Qkv);
  expect_throw([&] { ow::check_stream_shape("design.json", ow::Op::Qkv, q.M, q.K * 2, q.N); },
               "qkv", "refuses qkv with twice the K");
  expect_throw([&] { ow::check_stream_shape("design.json", ow::Op::Qkv, q.M + 256, q.K, q.N); },
               "built for", "refuses qkv with a larger M");
  expect_throw([&] { ow::check_stream_shape("design.json", ow::Op::Fc1, q.M, q.K, q.N); },
               "fc1", "refuses fc1 carrying qkv's shape");
  // A set built for a different Whisper (large-v3's 32 decoder layers would make xkv
  // 8x wider) is the realistic version of the same mistake.
  const ow::StreamShape x = ow::expected_shape(ow::Op::Xkv);
  expect_throw([&] { ow::check_stream_shape("design.json", ow::Op::Xkv, x.M, x.K, x.N * 8); },
               "refusing to dispatch", "refuses xkv built for another decoder depth");
}

void test_weight_dtype(const std::string &tmp_dir) {
  std::printf("-- container dtypes --\n");
  const std::string bf16_path = tmp_dir + "/guards_bf16.safetensors";
  const std::string f16_path = tmp_dir + "/guards_f16.safetensors";
  const std::vector<uint8_t> two_by_two(2 * 2 * 2, 0x11);   // [2,2], two bytes per element

  write_safetensors(bf16_path, "decoder.layers.0.fc1.weight", "BF16", {2, 2}, two_by_two);
  write_safetensors(f16_path, "decoder.layers.0.fc1.weight", "F16", {2, 2}, two_by_two);

  // The F16 file is the one that matters: same shape, same byte count, different dtype.
  // Before the dtype check it was read as bf16 and produced finite, wrong numbers.
  open_qwen36::Q4nxFile bf(bf16_path), f16(f16_path);
  check(bf.meta("decoder.layers.0.fc1.weight").dtype == "BF16", "a BF16 tensor reads as BF16");
  check(f16.meta("decoder.layers.0.fc1.weight").dtype == "F16",
        "an F16 tensor of the same shape and size reads as F16");
  expect_throw([&] { (void)f16.bf16("decoder.layers.0.fc1.weight"); }, "not BF16",
               "Q4nxFile::bf16() refuses the F16 tensor");
  // require_bf16() is the guard the decoder's RAW loader calls -- the path that keeps
  // the bits and therefore never reaches Q4nxFile::bf16()'s own check.
  expect_ok([&] { ow::require_bf16(bf, "decoder.layers.0.fc1.weight"); },
            "require_bf16() accepts the BF16 tensor");
  expect_throw([&] { ow::require_bf16(f16, "decoder.layers.0.fc1.weight"); }, "expected BF16",
               "require_bf16() refuses the F16 tensor of identical shape and size");
  expect_throw([&] { ow::require_bf16(bf, "decoder.layers.9.fc1.weight"); }, "missing tensor",
               "require_bf16() refuses a tensor that is not there");

  std::remove(bf16_path.c_str());
  std::remove(f16_path.c_str());
}

// design.json carrying one b_layout object, so read_b_layout() can be pointed at a
// directory holding exactly the field set under test.
void write_design(const std::string &dir, const std::string &b_layout) {
  std::ofstream f(dir + "/design.json", std::ios::binary);
  f << "{\"name\":\"whisper_gemm\",\"b_layout\":" << b_layout << "}";
}

void test_b_layout(const std::string &tmp_dir) {
  std::printf("-- b_layout tuple --\n");
  const std::string full =
      "{\"kind\":\"block_panel\",\"tile_k\":64,\"tile_n\":32,\"order\":\"k,n,kt,nt\","
      "\"inner\":\"s,t\",\"mac_s\":8,\"mac_t\":8,\"dtype\":\"BF16\"}";
  write_design(tmp_dir, full);
  expect_ok([&] {
    const ow::KernelSet::BLayout b = ow::KernelSet::read_b_layout(tmp_dir);
    if (b.tile_k != 64 || b.tile_n != 32 || b.mac_s != 8 || b.mac_t != 8)
      throw std::runtime_error("read back the wrong tuple");
  }, "reads the shipped tuple (64, 32, 8, 8)");

  // Each field in turn, absent. The weights are tiled with whatever this returns, and
  // the KernelSet constructor then checks design.json against those same values -- so a
  // defaulted field would be a guess compared with itself.
  for (const char *key : {"tile_k", "tile_n", "mac_s", "mac_t"}) {
    std::string one = full;
    const std::string needle = std::string("\"") + key + "\":";
    const size_t at = one.find(needle);
    const size_t end = one.find(',', at);
    one.erase(at, end - at + 1);
    write_design(tmp_dir, one);
    expect_throw([&] { (void)ow::KernelSet::read_b_layout(tmp_dir); }, key,
                 std::string("refuses a b_layout with no ") + key);
  }

  std::string zero = full;
  const size_t at = zero.find("\"mac_t\":8");
  zero.replace(at, std::string("\"mac_t\":8").size(), "\"mac_t\":0");
  write_design(tmp_dir, zero);
  expect_throw([&] { (void)ow::KernelSet::read_b_layout(tmp_dir); }, "positive",
               "refuses mac_t = 0");

  write_design(tmp_dir, "{}");
  expect_throw([&] { (void)ow::KernelSet::read_b_layout(tmp_dir); }, "missing",
               "refuses an empty b_layout");
  std::remove((tmp_dir + "/design.json").c_str());
}

}  // namespace

int main(int argc, char **argv) {
  const std::string tmp_dir = argc > 1 ? argv[1] : ".";
  std::printf("== open_whisper guards ==\n");
  try {
    test_stream_shapes();
    test_b_layout(tmp_dir);
    test_weight_dtype(tmp_dir);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "guards_test: unexpected exception: %s\n", e.what());
    return 1;
  }
  std::printf("%s\n", failures ? "FAILED" : "all guards hold");
  return failures ? 1 : 0;
}
