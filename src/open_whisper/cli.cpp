//===- cli.cpp ------------------------------------------------*- C++ -*-===//
//
// open_whisper_cli -- phase 2b's gate. Runs the NPU encoder over a golden
// clip and reports, layer by layer, its float64 cosine and relative error
// against transformers' own float64 forward pass -- chained (this encoder's
// own previous output feeds the next layer) and, with --forced, teacher-
// forced (the golden hidden state feeds each layer independently, isolating
// one layer's error from 32 layers of accumulation).
//
//   open_whisper_cli --model DIR --kernels DIR --golden FILE.safetensors [--forced]
//
// SPDX-License-Identifier: MIT
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "encoder.hpp"
#include "open_qwen36/q4nx_file.hpp"

namespace {

double cosine(const float *a, const float *b, size_t n) {
  double dot = 0, na = 0, nb = 0;
  for (size_t i = 0; i < n; ++i) {
    const double av = a[i], bv = b[i];
    dot += av * bv;
    na += av * av;
    nb += bv * bv;
  }
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

double rel_err(const float *a, const float *b, size_t n) {
  double num = 0, den = 0;
  for (size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    num += d * d;
    den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return std::sqrt(num) / std::sqrt(den);
}

bool any_nan(const float *a, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (std::isnan(a[i]) || std::isinf(a[i])) return true;
  return false;
}

struct Args {
  std::string model, kernels, golden, dump;
  long long stress = 0;
  bool forced = false;
};

bool parse_args(int argc, char **argv, Args &a) {
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(s + " needs a value");
      return argv[++i];
    };
    if (s == "--model") a.model = next();
    else if (s == "--kernels") a.kernels = next();
    else if (s == "--golden") a.golden = next();
    else if (s == "--forced") a.forced = true;
    else if (s == "--dump") a.dump = next();
    else if (s == "--stress") a.stress = std::stoll(next());
    else { std::fprintf(stderr, "unknown argument: %s\n", s.c_str()); return false; }
  }
  if (a.model.empty() || a.golden.empty()) {
    std::fprintf(stderr,
                 "usage: open_whisper_cli --model DIR --kernels DIR --golden "
                 "FILE.safetensors [--forced]\n");
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  Args args;
  if (!parse_args(argc, argv, args)) return 2;

  bool saw_nan = false;
  double enc_out_cos = -2.0;

  try {
    std::printf("== open_whisper_cli ==\n  model      %s\n  golden     %s\n",
               args.model.c_str(), args.golden.c_str());

    open_qwen36::Q4nxFile golden(args.golden);
    auto has = [&](const std::string &n) { return golden.has(n); };
    auto get = [&](const std::string &n) { return golden.f32(n); };

    if (!has("mel")) throw std::runtime_error(args.golden + ": no 'mel' tensor");
    const auto &mel_meta = golden.meta("mel");
    if (mel_meta.shape.size() != 2 || mel_meta.shape[0] != 128 || mel_meta.shape[1] != 3000)
      throw std::runtime_error(args.golden + ": 'mel' is not [128,3000]");
    std::vector<float> mel = get("mel");

    ow::Encoder enc(args.model, args.kernels);

    struct Row { std::string name; double cos = -2, rel = -1; size_t n = 0; };
    std::vector<Row> rows;

    // Chained pass: the encoder's own output feeds the next stage, exactly
    // as it would in production. The hook compares each stage against its
    // golden tensor as soon as the encoder produces it.
    auto hook = [&](const std::string &name, const float *data, int64_t r, int64_t c) {
      if (!has(name)) return;   // golden may not carry every optional tensor (e.g. cross.*)
      const size_t n = static_cast<size_t>(r) * static_cast<size_t>(c);
      std::vector<float> g = get(name);
      if (g.size() != n) {
        std::printf("  %-16s SIZE MISMATCH got %zu golden %zu\n", name.c_str(), n, g.size());
        return;
      }
      Row row{name, cosine(data, g.data(), n), rel_err(data, g.data(), n), n};
      if (any_nan(data, n)) { saw_nan = true; std::printf("  %-16s contains NaN/Inf\n", name.c_str()); }
      std::printf("  %-16s cos %.8f  rel %.3e  (%zu rows x %lld)\n", name.c_str(), row.cos,
                 row.rel, n / static_cast<size_t>(c), (long long)c);
      if (!args.dump.empty()) {
        // Raw f32, row-major, for an off-line comparison against the numpy
        // replica fed the SAME input (which is the only way to tell a wrong
        // layer from a wrong input).
        const std::string path = args.dump + "/" + name + ".f32";
        if (FILE *f = std::fopen(path.c_str(), "wb")) {
          std::fwrite(data, sizeof(float), n, f);
          std::fclose(f);
        }
      }
      if (name == "enc.out") enc_out_cos = row.cos;
      rows.push_back(row);
    };

    if (args.stress > 0) {
      std::printf("-- stress: %lld identical qkv dispatches, B slot cycling --\n", args.stress);
      return enc.stress_qkv(args.stress, 32) == 0 ? 0 : 1;
    }

    std::printf("-- chained (this encoder's own state feeds the next layer) --\n");
    enc.encode(mel.data(), hook);

    if (args.forced) {
      std::printf("-- teacher-forced (golden enc.hidden.<i> feeds layer i alone) --\n");
      for (int64_t i = 0; i < 32; ++i) {
        const std::string in_name = "enc.hidden." + std::to_string(i);
        const std::string out_name = "enc.hidden." + std::to_string(i + 1);
        if (!has(in_name) || !has(out_name)) continue;
        std::vector<float> in = get(in_name);
        std::vector<float> out(1500 * 1280);
        if (in.size() != static_cast<size_t>(1500 * 1280)) {
          std::printf("  %-16s golden input is %zu floats, expected %d -- skipping\n",
                     in_name.c_str(), in.size(), 1500 * 1280);
          continue;
        }
        enc.run_layer_from(i, in.data(), out.data());
        std::vector<float> g = get(out_name);
        const double c = cosine(out.data(), g.data(), out.size());
        const double r = rel_err(out.data(), g.data(), out.size());
        if (any_nan(out.data(), out.size())) saw_nan = true;
        std::printf("  L%02lld forced   cos %.8f  rel %.3e\n", (long long)i, c, r);
      }
    }

    std::printf("-- host stage timers (host wall clock; NOT an NPU performance claim) --\n");
    const auto &t = enc.timers;
    std::printf("  im2col       %8.1f ms\n", t.im2col * 1e3);
    std::printf("  bf16 round   %8.1f ms\n", t.bf16 * 1e3);
    std::printf("  layer_norm   %8.1f ms\n", t.layer_norm * 1e3);
    std::printf("  gelu (+bias) %8.1f ms\n", t.gelu * 1e3);
    std::printf("  bias add     %8.1f ms\n", t.bias * 1e3);
    std::printf("  residual add %8.1f ms\n", t.residual * 1e3);
    std::printf("  attention    %8.1f ms\n", t.attention * 1e3);
    std::printf("  npu in-sync  %8.1f ms  (host wall clock: memcpy + sync_to_device)\n",
               t.npu_in * 1e3);
    std::printf("  npu dispatch %8.1f ms  (host wall clock: submit+wait, dominated by "
               "hardware -- see docs on trace-based NPU numbers)\n",
               t.npu_dispatch * 1e3);
    std::printf("  npu out-sync %8.1f ms  (host wall clock: sync_from_device)\n",
               t.npu_out * 1e3);
    std::printf("  TOTAL        %8.1f ms  (host wall clock, end to end)\n", t.total * 1e3);

  } catch (const std::exception &e) {
    std::fprintf(stderr, "open_whisper_cli: FAILED: %s\n", e.what());
    return 1;
  }

  if (saw_nan) { std::fprintf(stderr, "open_whisper_cli: NaN/Inf in the output -- FAIL\n"); return 1; }
  if (enc_out_cos < 0.99) {
    std::fprintf(stderr, "open_whisper_cli: enc.out cosine %.8f < 0.99 -- FAIL\n", enc_out_cos);
    return 1;
  }
  std::printf("open_whisper_cli: PASS (enc.out cosine %.8f)\n", enc_out_cos);
  return 0;
}
