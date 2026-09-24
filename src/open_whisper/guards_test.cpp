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
#include <sstream>
#include <string>
#include <vector>

#include "fa_guards.hpp"
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

// OW_ATTN parsing (task 0180 defaults: auto/npu/host) and its resolution
// against whether a FlashAttention kernel is actually present -- entirely
// without touching the filesystem or a device, per parse_attn_mode() and
// resolve_use_npu_attn()'s own contract (fa_guards.hpp).
void test_attn_mode() {
  std::printf("-- OW_ATTN mode --\n");
  check(ow::parse_attn_mode(nullptr) == ow::AttnMode::Auto, "unset -> auto");
  check(ow::parse_attn_mode("") == ow::AttnMode::Auto, "empty -> auto");
  check(ow::parse_attn_mode("auto") == ow::AttnMode::Auto, "'auto' -> auto");
  check(ow::parse_attn_mode("host") == ow::AttnMode::Host, "'host' -> host");
  check(ow::parse_attn_mode("npu") == ow::AttnMode::Npu, "'npu' -> npu");
  expect_throw([&] { (void)ow::parse_attn_mode("npuu"); }, "OW_ATTN",
               "refuses a misspelt 'npuu' rather than reading it as host");
  expect_throw([&] { (void)ow::parse_attn_mode("Auto"); }, "OW_ATTN",
               "refuses 'Auto' (case-sensitive, not read as auto)");

  // auto: npu iff a kernel is present, silently either way (no throw).
  check(ow::resolve_use_npu_attn(ow::AttnMode::Auto, true, "dir") == true,
        "auto + kernel present -> npu");
  check(ow::resolve_use_npu_attn(ow::AttnMode::Auto, false, "dir") == false,
        "auto + no kernel -> host");
  // host: never npu, regardless of what is on disk.
  check(ow::resolve_use_npu_attn(ow::AttnMode::Host, true, "dir") == false,
        "host + kernel present -> still host");
  // npu: requires the kernel; THIS is the guard the task asked for by name --
  // "npu without a kernel set refuses".
  check(ow::resolve_use_npu_attn(ow::AttnMode::Npu, true, "dir") == true,
        "npu + kernel present -> npu");
  expect_throw([&] { (void)ow::resolve_use_npu_attn(ow::AttnMode::Npu, false, "/some/fa/dir"); },
               "/some/fa/dir", "npu + no kernel present -> refuses, naming the directory");
}

// fa.json's required fields and the geometry guard -- read_fa_kernel_info()
// refuses an incomplete file (every field is required, the same discipline
// as design.json's read_b_layout), and check_fa_geometry() refuses a
// complete-but-wrong-shaped one.
void write_fa_json(const std::string &dir, const std::string &body) {
  std::ofstream f(dir + "/fa.json", std::ios::binary);
  f << body;
}

// Builds the JSON object field by field so a field can be OMITTED (rather
// than string-surgered out of a fixed template, which is fragile exactly at
// the last field -- no trailing comma to remove). `omit`, when non-null,
// names the one field to leave out entirely.
std::string fa_json(int64_t heads = 20, int64_t dk = 64, int64_t dv = 64, int64_t lq = 1536,
                    int64_t lk = 1536, int64_t valid_len = 1500, const char *omit = nullptr) {
  std::vector<std::pair<std::string, std::string>> fields = {
      {"heads", std::to_string(heads)},      {"dk", std::to_string(dk)},
      {"dv", std::to_string(dv)},            {"lq", std::to_string(lq)},
      {"lk", std::to_string(lk)},            {"valid_len", std::to_string(valid_len)},
      {"fp32_state", "true"},                {"emulate_bfp16", "false"},
      {"mlir_aie_version", "\"1.4.3.dev55\""},
      {"peano_version", "\"22.0.0.2026092301\""},
  };
  std::ostringstream ss;
  ss << "{";
  bool first = true;
  for (const auto &kv : fields) {
    if (omit && kv.first == omit) continue;
    if (!first) ss << ",";
    first = false;
    ss << "\"" << kv.first << "\":" << kv.second;
  }
  ss << "}";
  return ss.str();
}

void test_fa_kernel_info(const std::string &tmp_dir) {
  std::printf("-- fa.json --\n");
  expect_throw([&] { (void)ow::read_fa_kernel_info(tmp_dir + "/no-such-fa-dir"); },
               "cannot open", "refuses a directory with no fa.json at all");

  write_fa_json(tmp_dir, fa_json());
  expect_ok([&] {
    const ow::FaKernelInfo info = ow::read_fa_kernel_info(tmp_dir);
    if (info.heads != 20 || info.dk != 64 || info.dv != 64 || info.lq != 1536 ||
        info.lk != 1536 || info.valid_len != 1500 || !info.fp32_state || info.emulate_bfp16 ||
        info.mlir_aie_version != "1.4.3.dev55")
      throw std::runtime_error("read back the wrong record");
  }, "reads a complete, correctly-shaped fa.json");
  expect_ok([&] { ow::check_fa_geometry("fa.json", ow::read_fa_kernel_info(tmp_dir)); },
            "check_fa_geometry accepts the engine's own shape (20/64/64/1536/1536/1500)");

  for (const char *key : {"heads", "dk", "dv", "lq", "lk", "valid_len", "fp32_state",
                          "emulate_bfp16", "mlir_aie_version", "peano_version"}) {
    write_fa_json(tmp_dir, fa_json(20, 64, 64, 1536, 1536, 1500, key));
    expect_throw([&] { (void)ow::read_fa_kernel_info(tmp_dir); }, key,
                 std::string("refuses fa.json with no '") + key + "'");
  }

  // A kernel built for a different shape (this repo's own T44/T60-style
  // family, or a stale build predating a geometry change) must be refused,
  // not dispatched against -- the same class of guard as check_stream_shape.
  write_fa_json(tmp_dir, fa_json(/*heads=*/16));
  expect_throw([&] { ow::check_fa_geometry("fa.json", ow::read_fa_kernel_info(tmp_dir)); },
               "heads", "refuses a kernel built for 16 heads instead of 20");
  write_fa_json(tmp_dir, fa_json(20, 64, 64, 1536, 1536, /*valid_len=*/448));
  expect_throw([&] { ow::check_fa_geometry("fa.json", ow::read_fa_kernel_info(tmp_dir)); },
               "valid_len", "refuses a kernel built for valid_len 448 (the decoder's, not "
                            "the encoder's 1500)");

  write_fa_json(tmp_dir, "{}");
  expect_throw([&] { (void)ow::read_fa_kernel_info(tmp_dir); }, "heads",
               "refuses an empty fa.json");
  std::remove((tmp_dir + "/fa.json").c_str());
}

// fa_kernel_usable() and its effect through resolve_use_npu_attn(): OW_ATTN=auto
// must not throw on a stale/mismatched fa/ -- it must resolve to host, naming why
// -- and OW_ATTN=npu must still refuse it (PR #111 review: auto previously only
// checked that the three files existed, so a malformed or wrong-geometry fa.json
// made FaAttention's constructor throw instead of falling back).
void test_fa_kernel_usable(const std::string &tmp_dir) {
  std::printf("-- fa_kernel_usable (auto/npu probe) --\n");
  const std::string xclbin = tmp_dir + "/air.xclbin", insts = tmp_dir + "/air.insts.bin";
  auto touch = [](const std::string &p) { std::ofstream(p, std::ios::binary) << "x"; };
  auto rm_all = [&] {
    std::remove(xclbin.c_str());
    std::remove(insts.c_str());
    std::remove((tmp_dir + "/fa.json").c_str());
  };
  rm_all();

  // No files at all: not usable, reason names "no FlashAttention kernel".
  {
    std::string reason;
    check(!ow::fa_kernel_usable(tmp_dir, reason), "no files -> not usable");
    check(reason.find("no FlashAttention kernel") != std::string::npos,
          "  reason names 'no FlashAttention kernel' (got: " + reason + ")");
  }

  touch(xclbin);
  touch(insts);

  // Files present but fa.json malformed (not valid JSON): fa_kernel_present()
  // alone would have said "present"; fa_kernel_usable() must not.
  {
    write_fa_json(tmp_dir, "{ this is not json");
    std::string reason;
    check(!ow::fa_kernel_usable(tmp_dir, reason), "malformed fa.json -> not usable");
    check(reason.find("invalid JSON") != std::string::npos,
          "  reason names 'invalid JSON' (got: " + reason + ")");
    check(ow::resolve_use_npu_attn(ow::AttnMode::Auto, false, tmp_dir, reason) == false,
          "  auto + malformed fa.json -> host (no throw)");
    expect_throw([&] { (void)ow::resolve_use_npu_attn(ow::AttnMode::Npu, false, tmp_dir, reason); },
                 "invalid JSON", "  npu + malformed fa.json -> refuses, naming why");
  }

  // Files present, fa.json valid JSON, but the wrong geometry (16 heads, not
  // 20): auto falls back to host; npu refuses, naming the mismatched field.
  {
    write_fa_json(tmp_dir, fa_json(/*heads=*/16));
    std::string reason;
    const bool usable = ow::fa_kernel_usable(tmp_dir, reason);
    check(!usable, "wrong geometry (16 heads) -> not usable");
    check(reason.find("heads") != std::string::npos,
          "  reason names the mismatched field 'heads' (got: " + reason + ")");
    check(ow::resolve_use_npu_attn(ow::AttnMode::Auto, usable, tmp_dir, reason) == false,
          "  auto + wrong geometry -> host (no throw)");
    expect_throw([&] { (void)ow::resolve_use_npu_attn(ow::AttnMode::Npu, usable, tmp_dir, reason); },
                 "heads", "  npu + wrong geometry -> refuses, naming 'heads'");
  }

  // A complete, correctly-shaped fa.json IS usable -- the positive control,
  // proving the two negatives above are about the content, not the plumbing.
  {
    write_fa_json(tmp_dir, fa_json());
    std::string reason;
    check(ow::fa_kernel_usable(tmp_dir, reason), "correct fa.json -> usable");
    check(ow::resolve_use_npu_attn(ow::AttnMode::Auto, true, tmp_dir, reason) == true,
          "  auto + usable kernel -> npu");
  }

  rm_all();
}

}  // namespace

int main(int argc, char **argv) {
  const std::string tmp_dir = argc > 1 ? argv[1] : ".";
  std::printf("== open_whisper guards ==\n");
  try {
    test_stream_shapes();
    test_b_layout(tmp_dir);
    test_weight_dtype(tmp_dir);
    test_attn_mode();
    test_fa_kernel_info(tmp_dir);
    test_fa_kernel_usable(tmp_dir);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "guards_test: unexpected exception: %s\n", e.what());
    return 1;
  }
  std::printf("%s\n", failures ? "FAILED" : "all guards hold");
  return failures ? 1 : 0;
}
