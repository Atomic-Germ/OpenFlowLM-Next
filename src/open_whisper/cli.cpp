//===- cli.cpp ------------------------------------------------*- C++ -*-===//
//
// open_whisper_cli -- phase 2b's gate (the NPU encoder) plus, from phase 3,
// the host decoder's own gate. Runs the NPU encoder over a golden clip and
// reports, layer by layer, its float64 cosine and relative error against
// transformers' own float64 forward pass -- chained (this encoder's own
// previous output feeds the next layer) and, with --forced, teacher-forced
// (the golden hidden state feeds each layer independently, isolating one
// layer's error from 32 layers of accumulation).
//
// --decode hf|host runs the phase-3 host decoder on that protocol's golden
// token sequence: teacher-forced (argmax agreement and logits cosine against
// the golden, over the free-running region only -- the forced prefix is a
// prompt, not a prediction) and free-running greedy from the golden prefix
// (compared token for token against the golden path). See open_kernels/model/
// whisper_goldens.py and whisper_decode_check.py for the reference protocol
// this mirrors in C++ with a KV cache instead of a from-scratch pass per step.
//
//   open_whisper_cli --model DIR --kernels DIR --golden FILE.safetensors ^
//       [--forced] [--decode hf|host]
//
// SPDX-License-Identifier: MIT
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <fstream>
#include <sstream>

#include "nlohmann/json.hpp"

#include "decoder.hpp"
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
  std::string model, kernels, golden, dump, decode, baseline;
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
    else if (s == "--decode") a.decode = next();
    else if (s == "--baseline") a.baseline = next();
    else { std::fprintf(stderr, "unknown argument: %s\n", s.c_str()); return false; }
  }
  if (a.decode != "" && a.decode != "hf" && a.decode != "host") {
    std::fprintf(stderr, "--decode must be 'hf' or 'host', got '%s'\n", a.decode.c_str());
    return false;
  }
  if (a.model.empty() || a.golden.empty()) {
    std::fprintf(stderr,
                 "usage: open_whisper_cli --model DIR --kernels DIR --golden "
                 "FILE.safetensors [--forced] [--decode hf|host]\n");
    return false;
  }
  return true;
}

// `${proto}.tokens` is int32 (whisper_goldens.py writes np.int32); read raw
// rather than through Q4nxFile::f32()/bf16(), which only know those two
// dtypes.
std::vector<int32_t> read_tokens(const open_qwen36::Q4nxFile &golden, const std::string &name) {
  const auto &m = golden.meta(name);
  if (m.shape.size() != 1) throw std::runtime_error(name + ": not a 1-D tensor");
  size_t nbytes = 0;
  const uint8_t *raw = golden.raw(name, &nbytes);
  std::vector<int32_t> out(m.shape[0]);
  if (m.dtype == "I32") {
    if (nbytes != out.size() * 4) throw std::runtime_error(name + ": I32 byte count mismatch");
    std::memcpy(out.data(), raw, nbytes);
  } else if (m.dtype == "I64") {
    if (nbytes != out.size() * 8) throw std::runtime_error(name + ": I64 byte count mismatch");
    std::vector<int64_t> tmp(out.size());
    std::memcpy(tmp.data(), raw, nbytes);
    for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<int32_t>(tmp[i]);
  } else {
    throw std::runtime_error(name + ": dtype is '" + m.dtype + "', expected an integer token id tensor");
  }
  return out;
}

// The phase-3 decoder gate for one protocol. Mirrors open_kernels/model/
// whisper_decode_check.py: teacher-forced argmax agreement over the
// free-running region (the forced prefix is a prompt, not a prediction -- see
// whisper_goldens.py's own `first_free`), then a from-scratch greedy
// free-run from the golden prefix compared token for token. Returns whether
// the gate passed. `model_dir` is reopened as its own Decoder rather than
// sharing anything with `enc` -- the decoder never touches the NPU kernel set
// at all.
bool run_decode_gate(ow::Encoder &enc, const std::string &model_dir, const open_qwen36::Q4nxFile &golden,
                     const std::string &proto, const std::vector<int32_t> &baseline,
                     int64_t baseline_agree) {
  const std::string tok_name = proto + ".tokens", lg_name = proto + ".logits";
  if (!golden.has(tok_name) || !golden.has(lg_name)) {
    std::printf("-- decode %s SKIPPED: golden has no '%s'/'%s' --\n", proto.c_str(), tok_name.c_str(),
               lg_name.c_str());
    return true;
  }
  const std::vector<int32_t> tokens = read_tokens(golden, tok_name);
  const int64_t T = static_cast<int64_t>(tokens.size());
  const int64_t V = ow::DecoderGeometry::vocab, VP = ow::DecoderGeometry::vocab_padded;
  const auto &lm = golden.meta(lg_name);
  if (lm.shape.size() != 2 || static_cast<int64_t>(lm.shape[0]) != T || static_cast<int64_t>(lm.shape[1]) != V)
    throw std::runtime_error(lg_name + ": shape does not match " + tok_name + " x vocab " + std::to_string(V));
  const std::vector<float> glogits = golden.f32(lg_name);
  const int64_t first_free = (proto == "hf") ? 4 : 3;
  const int32_t EOT = 50257;

  std::printf("== decode %s (%lld golden tokens, first_free=%lld) ==\n", proto.c_str(), (long long)T,
             (long long)first_free);

  ow::Decoder dec(model_dir);
  dec.set_encoder_output(enc.xkv().data());
  dec.clear_context();

  // Teacher-forced: feed every golden token through the KV cache, one at a
  // time, and keep every step's logits -- this is the KV-cache path;
  // whisper_goldens.py's own assertion (and whisper_decode_check.py) is what
  // establishes that it agrees with an uncached full-context pass.
  std::vector<std::vector<float>> logits(static_cast<size_t>(T), std::vector<float>(static_cast<size_t>(VP)));
  for (int64_t i = 0; i < T; ++i) dec.step(tokens[static_cast<size_t>(i)], logits[static_cast<size_t>(i)].data());

  bool pad_ok = true;
  for (int64_t i = 0; i < T && pad_ok; ++i)
    for (int64_t j = V; j < VP; ++j) {
      const float x = logits[static_cast<size_t>(i)][static_cast<size_t>(j)];
      if (!(std::isinf(x) && x < 0)) pad_ok = false;
    }
  std::printf("  vocab pad [%lld,%lld) is -inf on every step: %s\n", (long long)V, (long long)VP,
             pad_ok ? "PASS" : "FAIL");

  int64_t agree = 0, total = 0;
  double cos_sum = 0.0, cos_min = 2.0;
  for (int64_t i = first_free - 1; i < T - 1; ++i) {
    const float *lg = logits[static_cast<size_t>(i)].data();
    const float *gg = glogits.data() + static_cast<size_t>(i) * static_cast<size_t>(V);
    int64_t am = 0;
    float best = lg[0];
    for (int64_t j = 1; j < V; ++j)
      if (lg[j] > best) { best = lg[j]; am = j; }
    ++total;
    if (am == tokens[static_cast<size_t>(i + 1)]) ++agree;
    const double c = cosine(lg, gg, static_cast<size_t>(V));
    cos_sum += c;
    if (c < cos_min) cos_min = c;
  }
  // Agreement with float64 is the bar; where the bf16 datapath itself changes
  // a token, the recorded baseline's own count is (see the free-run note below).
  const bool argmax_ok = total > 0 && (agree == total || (baseline_agree >= 0 && agree == baseline_agree));
  std::printf("  teacher-forced argmax agreement, steps [%lld,%lld] predicting the next golden token: "
             "%lld/%lld\n", (long long)(first_free - 1), (long long)(T - 2), (long long)agree, (long long)total);
  std::printf("  logits cosine vs golden, same range: mean %.8f  min %.8f\n",
             total ? cos_sum / total : -2.0, total ? cos_min : -2.0);

  // Free-running greedy from the golden prefix, plain argmax over the real
  // vocab (the padded tail is -inf and could never win anyway).
  dec.clear_context();
  std::vector<int32_t> ids(tokens.begin(), tokens.begin() + first_free);
  std::vector<float> last(static_cast<size_t>(VP));
  for (int64_t i = 0; i < first_free; ++i) dec.step(ids[static_cast<size_t>(i)], last.data());
  while (static_cast<int64_t>(ids.size()) < 440) {
    int64_t am = 0;
    float best = last[0];
    for (int64_t j = 1; j < V; ++j)
      if (last[j] > best) { best = last[j]; am = j; }
    ids.push_back(static_cast<int32_t>(am));
    if (am == EOT) break;
    dec.step(static_cast<int32_t>(am), last.data());
  }
  bool same = ids.size() == tokens.size();
  int64_t first_div = -1;
  for (size_t i = 0; i < std::min(ids.size(), tokens.size()); ++i)
    if (ids[i] != tokens[i]) { first_div = static_cast<int64_t>(i); same = false; break; }
  if (same) {
    std::printf("  free-run: %zu tokens, MATCHES the golden path exactly\n", ids.size());
  } else if (first_div >= 0) {
    std::printf("  free-run: DIVERGES from float64 at index %lld (got %d, golden %d) -- %zu vs %zu "
               "tokens total\n", (long long)first_div, ids[static_cast<size_t>(first_div)],
               tokens[static_cast<size_t>(first_div)], ids.size(), tokens.size());
    // A difference from float64 is not automatically this engine's error: the
    // bf16 datapath itself changes a token on one clip of the golden set, and
    // that path is recorded (--baseline). Matching it exactly is the real
    // statement -- two independent implementations of the same datapath, the
    // numpy replica and this one, taking the same turn at the same token.
    // Without this the gate would be permanently red on that clip, and a check
    // nobody can satisfy is a check its reader learns to skip (NpuEmbeddings
    // T64).
    if (!baseline.empty()) {
      if (baseline == ids)
        std::printf("  free-run: MATCHES the recorded bf16 datapath baseline exactly (%zu tokens) -- "
                   "the divergence above is the datapath's, not this engine's\n", ids.size());
      else
        std::printf("  free-run: and does NOT match the bf16 baseline either (%zu vs %zu tokens) -- "
                   "this is this engine's own divergence\n", ids.size(), baseline.size());
    }
  } else {
    std::printf("  free-run: agrees up to the shorter length, %zu vs %zu tokens (one is a prefix of the "
               "other)\n", ids.size(), tokens.size());
  }

  const auto &t = dec.timers;
  std::printf("  decode timers, teacher-forced + free-run combined (host wall clock; no NPU dispatch "
             "in this class at all)\n");
  std::printf("    embed        %8.1f ms\n", t.embed * 1e3);
  std::printf("    layer_norm   %8.1f ms\n", t.layer_norm * 1e3);
  std::printf("    linear       %8.1f ms\n", t.linear * 1e3);
  std::printf("    attention    %8.1f ms\n", t.attention * 1e3);
  std::printf("    gelu         %8.1f ms\n", t.gelu * 1e3);
  std::printf("    TOTAL        %8.1f ms over %lld steps (%.3f ms/token, %.2f tok/s)\n", t.total * 1e3,
             (long long)t.steps, t.steps ? t.total * 1e3 / static_cast<double>(t.steps) : 0.0,
             t.total > 0 ? static_cast<double>(t.steps) / t.total : 0.0);

  return pad_ok && argmax_ok && (same || (!baseline.empty() && baseline == ids));
}

}  // namespace

int main(int argc, char **argv) {
  Args args;
  if (!parse_args(argc, argv, args)) return 2;

  bool saw_nan = false;
  bool decode_ok = true;
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
    // The summary below describes THIS encode. run_layer_from() below goes
    // through run_layer() and would add the --forced passes to the stage and
    // NPU buckets but not to `total`, so take the timers before it runs.
    ow::Timers t = enc.timers;

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

    if (!args.decode.empty()) {
      // The bf16 datapath's own token path for this clip and protocol, if it
      // was recorded (open_kernels/model/whisper_decode_check.py
      // --write-baseline). Absent is fine: the gate then measures against
      // float64 alone.
      std::vector<int32_t> baseline;
      int64_t baseline_agree = -1;
      if (!args.baseline.empty()) {
        std::ifstream bf(args.baseline);
        if (!bf) throw std::runtime_error("cannot open " + args.baseline);
        std::stringstream bs;
        bs << bf.rdbuf();
        const nlohmann::json bj = nlohmann::json::parse(bs.str());
        std::string stem = args.golden;
        const size_t slash = stem.find_last_of("/\\");
        if (slash != std::string::npos) stem = stem.substr(slash + 1);
        const size_t dot = stem.find_last_of('.');
        if (dot != std::string::npos) stem = stem.substr(0, dot);
        const auto &clips = bj.at("clips");
        if (clips.contains(stem) && clips[stem].contains(args.decode)) {
          const auto &e = clips[stem][args.decode];
          baseline = e.at("tokens").get<std::vector<int32_t>>();
          if (e.contains("forced_argmax")) baseline_agree = e["forced_argmax"][0].get<int64_t>();
          std::printf("  baseline   %s [%s]: %zu tokens, matches float64: %s\n", stem.c_str(),
                     args.decode.c_str(), baseline.size(),
                     e.value("matches_fp64", false) ? "yes" : "no");
        } else {
          std::printf("  baseline   %s has no entry for %s [%s]\n", args.baseline.c_str(),
                     stem.c_str(), args.decode.c_str());
        }
      }
      decode_ok = run_decode_gate(enc, args.model, golden, args.decode, baseline, baseline_agree);
    }

    std::printf("-- host stage timers (host wall clock; NOT an NPU performance claim) --\n");
    for (size_t o = 0; o < static_cast<size_t>(ow::Op::Count); ++o)
      t.npu_dispatch += t.npu_disp_op[o];
    std::printf("  im2col       %8.1f ms\n", t.im2col * 1e3);
    std::printf("  bf16 round   %8.1f ms\n", t.bf16 * 1e3);
    std::printf("  layer_norm   %8.1f ms\n", t.layer_norm * 1e3);
    std::printf("  gelu (+bias) %8.1f ms\n", t.gelu * 1e3);
    std::printf("  bias add     %8.1f ms\n", t.bias * 1e3);
    std::printf("  residual add %8.1f ms\n", t.residual * 1e3);
    std::printf("  attention    %8.1f ms\n", t.attention * 1e3);
    // OW_ATTN_PHASES=1 splits that into the two GEMMs a kernel set could take and
    // the softmax that stays on the host whichever way they go. Summed across
    // threads, so these are CPU-seconds and total more than the wall time above.
    {
      const auto &ph = t.attn_phases;
      const double tot = ph.scores + ph.softmax + ph.values;
      if (tot > 0) {
        std::printf("    Q.K^T      %8.1f ms  (%.1f%% of attention CPU time)\n",
                   ph.scores * 1e3, 100.0 * ph.scores / tot);
        std::printf("    softmax    %8.1f ms  (%.1f%%)  -- stays on the host\n",
                   ph.softmax * 1e3, 100.0 * ph.softmax / tot);
        std::printf("    P.V        %8.1f ms  (%.1f%%)\n",
                   ph.values * 1e3, 100.0 * ph.values / tot);
      }
    }
    std::printf("  npu in-sync  %8.1f ms  (host wall clock: memcpy + sync_to_device)\n",
               t.npu_in * 1e3);
    std::printf("  npu dispatch %8.1f ms  (host wall clock: submit+wait, dominated by hardware)\n",
               t.npu_dispatch * 1e3);
    // Per stream, against the DRAM traffic the design actually moves.
    // gemm_pretiled.py's fill loop streams A once and C once, but B ONCE PER
    // ROW BLOCK -- b_reuse is off because it does not build at 8 columns
    // (T48 / tasks-0046: the mem tile has A(1) + B(1) + C(4 rows) = 6 of 6
    // channels, and the C join is what spends them). With
    // n_row_blocks = M / (m * n_aie_rows) = M / 256:
    //     bytes = M*K*2  +  n_row_blocks * K*N*2  +  M*N*4
    // The array's measured shim roof is ~45.5 GB/s (NpuEmbeddings T45), so
    // the GB/s column says how close each shape runs to the memory system.
    double tot_bytes = 0;
    for (size_t o = 0; o < static_cast<size_t>(ow::Op::Count); ++o) {
      const ow::Op op = static_cast<ow::Op>(o);
      const ow::StreamShape sh = ow::expected_shape(op);
      const bool per_layer = !(op == ow::Op::Conv1 || op == ow::Op::Conv2 ||
                               op == ow::Op::Xkv);
      const double calls = per_layer ? 32.0 : 1.0;
      const double nrb = static_cast<double>(sh.M) / 256.0;
      const double bytes = calls * (static_cast<double>(sh.M) * sh.K * 2.0 +
                                    nrb * static_cast<double>(sh.K) * sh.N * 2.0 +
                                    static_cast<double>(sh.M) * sh.N * 4.0);
      tot_bytes += bytes;
      const double ms = t.npu_disp_op[o] * 1e3;
      std::printf("    %-6s %8.1f ms  %4.0f calls  %7.2f GB  %6.1f GB/s\n",
                 ow::op_name(op), ms, calls, bytes / 1e9,
                 ms > 0 ? bytes / 1e9 / (ms / 1e3) : 0.0);
    }
    std::printf("    %-6s %8.1f ms              %7.2f GB  %6.1f GB/s  (shim roof ~45.5)\n",
               "TOTAL", t.npu_dispatch * 1e3, tot_bytes / 1e9,
               t.npu_dispatch > 0 ? tot_bytes / 1e9 / t.npu_dispatch : 0.0);
    std::printf("  npu out-sync %8.1f ms  (host wall clock: sync_from_device)\n",
               t.npu_out * 1e3);
    std::printf("  golden cmp   %8.1f ms  (GATE ONLY: float64 comparison of 42 stage "
               "tensors; not part of an encode)\n",
               t.hook * 1e3);
    const double enc_only = t.total - t.hook;
    std::printf("  ENCODE       %8.1f ms  (host wall clock, total minus the gate)\n",
               enc_only * 1e3);
    const double named = t.im2col + t.bf16 + t.layer_norm + t.gelu + t.bias + t.residual +
                        t.attention + t.npu_in + t.npu_dispatch + t.npu_out;
    std::printf("  unattributed %8.1f ms  (%.1f%% of ENCODE -- named buckets sum to %.1f ms)\n",
               (enc_only - named) * 1e3,
               enc_only > 0 ? 100.0 * (enc_only - named) / enc_only : 0.0, named * 1e3);
    std::printf("  TOTAL        %8.1f ms  (host wall clock, end to end, gate included)\n",
               t.total * 1e3);

  } catch (const std::exception &e) {
    std::fprintf(stderr, "open_whisper_cli: FAILED: %s\n", e.what());
    return 1;
  }

  if (saw_nan) { std::fprintf(stderr, "open_whisper_cli: NaN/Inf in the output -- FAIL\n"); return 1; }
  if (enc_out_cos < 0.99) {
    std::fprintf(stderr, "open_whisper_cli: enc.out cosine %.8f < 0.99 -- FAIL\n", enc_out_cos);
    return 1;
  }
  if (!decode_ok) {
    std::fprintf(stderr, "open_whisper_cli: decode gate FAILED (see --decode output above)\n");
    return 1;
  }
  std::printf("open_whisper_cli: PASS (enc.out cosine %.8f)\n", enc_out_cos);
  return 0;
}
