//===- npue_encoder.hpp --------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the NPU encoders, lifted out of main.cpp.
// SPDX-License-Identifier: MIT
//
// WHY THIS FILE EXISTS. Everything here used to live in main.cpp's anonymous
// namespace, next to the CLI, the HTTP server, the model hub and the probe
// modes. That is fine for a program and impossible for a library: another host
// application cannot link `main()`. tasks/0156.
//
// WHY THE DEFINITIONS ARE IN THE HEADER, and not split into the .cpp where
// they would normally go. Every `Encoder` method is implicitly inline inside
// one translation unit today, and MSVC /O2 inlines the per-layer helpers
// (`i8w`, `lap`, `use_tier`, `par`, `par_rows`) straight into `gemm`/`run`.
// Out-lining them costs real throughput and NOT ONE GATE IN THIS REPOSITORY
// CAN SEE IT -- the golden gate, the cross-lane bitwise check and the semantic
// gate all measure the numbers, none measures the time. tasks/0156 recorded a
// `--bench` baseline before the move precisely so that this could be checked
// rather than assumed. So: structs keep their methods in-class, and free
// functions are `inline`.
//
// THE GEOMETRY IS STILL PROCESS-WIDE. ~21 values, read at ~150 sites.
// Threading a struct through those sites would be a large diff across 43
// `#if defined(__AVX2__)` blocks that the gates only cover end to end, and it
// would buy multi-instance -- which nothing wants and which a lease can refuse
// far more cheaply. The scalars are `extern` here and defined in
// npue_encoder.cpp; the container-typed ones are function-local statics behind
// const accessors, so there is no dynamic static initialisation left for a
// host application's own statics to race (tasks/0156 step 2).
//
// detail::apply_model_shape() is the only writer, and ShapeLease is its only
// entrance -- a second live model would silently reinterpret the first one's
// weights, so it REFUSES rather than overwriting.
//
// NOT INCLUDED, ON PURPOSE: http.hpp (it pulls in winsock, and nothing between
// the tokenizer and pick_artifacts touches it), hub.hpp, and <windows.h> --
// `cpu_seconds()` is the encode path's only Win32 call and it stayed behind in
// main.cpp, where its only callers (`--bench`) live.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "gemma_encode.hpp"
#include "gemma_kernels.hpp"
#include "json_min.hpp"
#include "npu_device.hpp"
#include "npue.hpp"
#include "tokenizer.hpp"
#include "tokenizer_xlmr.hpp"

namespace npue {
namespace enc {

// Model geometry, READ FROM THE CONTAINER at startup rather than compiled in.
//
// This runtime serves several models -- MiniLM-L6 (6 layers, mean pooling),
// bge-small (12 layers, CLS pooling), bge-large (24 layers, hidden 1024,
// head_dim 64) -- and their depth, width and pooling all differ. As
// constexpr, `g_layers = 6` would have run a 12-layer model for six layers and
// returned a plausible wrong vector with exit code 0.
//
// They are file scope and mutable because they are read at ~150 sites;
// threading a struct through all of them would be a large diff for no
// behavioural gain. detail::apply_model_shape() is the ONLY writer, it runs
// once under a ShapeLease before any Encoder exists, and every value starts at
// 0 so that a missed initialisation divides by zero or allocates nothing
// rather than quietly using a stale MiniLM number.
// head_dim / 8, bounded so the attention kernels can hold the vectors on the
// stack. 16 covers head_dim up to 128; every BERT-family encoder we target is
// 32 or 64.
constexpr int kMaxHeadVecs = 16;

// Whether a ShapeLease is outstanding. Function-local for the same
// initialisation-order reason as the container accessors below;
// atomic because two threads racing to load a model is precisely
// the case the lease has to catch rather than one it may assume
// away.
namespace detail {
inline std::atomic<bool> &shape_held() {
  static std::atomic<bool> held{false};
  return held;
}
}  // namespace detail

extern int64_t g_seq, g_hidden, g_heads, g_head_dim;
extern int64_t g_ffn, g_layers, g_max_positions;
extern bool g_cls_pool, g_l2_normalize;

// TRUNCATION POLICY. Default: refuse. See npue::InputTooLong for why this is
// worth a flag rather than a constant.
//
// The default is a BEHAVIOUR CHANGE and deliberately so -- it is the whole
// point of the change. It cannot move any measured number, though: an input
// that fitted before still fits and still produces the bit-identical vector,
// and the only inputs whose behaviour changes are the ones that were being
// silently answered wrong. --allow-truncation restores the old behaviour
// exactly, and says so on stderr each time it fires.
extern bool g_allow_truncation;
// Set once truncation has actually been permitted and taken, so the warning
// is printed once per run rather than once per row (a 2,048-input request
// would otherwise emit 2,048 identical lines).
extern std::atomic<bool> g_truncation_warned;

// One place, so the message is the same wherever the cut is detected.
// `index` is the caller's own input index -- a global row number, not an
// offset into whatever tier the runtime happened to batch it into, because
// the caller cannot see tiers.
inline void check_truncation(bool truncated, int32_t n_tokens_full, size_t index,
                      int64_t limit) {
  if (!truncated) return;
  if (!g_allow_truncation) throw npue::InputTooLong(index, n_tokens_full, limit);
  if (!g_truncation_warned.exchange(true))
    std::fprintf(stderr,
                 "  WARNING  --allow-truncation: input %zu is %d tokens and "
                 "was CUT to %lld. Its vector is a vector of the first %lld "
                 "tokens, not of the text. Further cuts this run are not "
                 "reported.\n",
                 index, static_cast<int>(n_tokens_full), (long long)limit,
                 (long long)limit);
}
// The model's name and the repository it was packed from.
//
// ACCESSORS, NOT GLOBALS, and the reason is the same one that put the rest of
// this file into a header: it is on its way into another host application.
// A namespace-scope std::string has a dynamic initialiser under C++17, so a
// static object in the HOST's translation units could read it before it is
// constructed -- an ordering nothing in either repository controls. A
// function-local static is constructed on first use, which removes the
// category rather than documenting it, and the thread-safe-init guarantee is
// C++11's.
//
// The readers are const because the writers are the point. Every one of these
// used to carry a comment saying the shape setter was its only writer -- and
// the comment was already FALSE for this one, which main() assigned directly.
// Now the mutable handle lives in `detail` and the claim is checkable rather
// than repeated.
namespace detail {
inline std::string &mut_model_name() { static std::string v; return v; }
inline std::string &mut_source_repo() { static std::string v; return v; }
}  // namespace detail
inline const std::string &model_name() { return detail::mut_model_name(); }
inline const std::string &source_repo() { return detail::mut_source_repo(); }
// model_name is the ONE piece of this state that does not come from the
// container: it is the .npue path's stem, so the caller that resolved the
// path is the only thing that knows it. Hence a public setter here and
// nowhere else.
inline void set_model_name(std::string name) {
  detail::mut_model_name() = std::move(name);
}

// T61-1 (tasks/0152). How much of an NPU dispatch runs under the cross-lane
// mutex.
//
// FALSE (the default) is the narrow lock: only `bind_instr` + `bind` + the
// submit are shared state, so a lane uploads its own operand, waits for its
// own command and reads its own result with the lock RELEASED, and one lane's
// command can already be queued behind another's. The hw_context executes
// commands in order, so the array still serialises -- what disappears is the
// host round trip between one command finishing and the next being built.
//
// TRUE restores the pre-0152 behaviour exactly (lock held across sync-to,
// dispatch, sync-from) so the change is an A/B in ONE binary rather than a
// comparison between two builds. tasks/0044's fail-open list is mostly
// "two builds that were supposed to differ in one thing".
extern bool g_wide_lock;

// arch=2 (nomic-embed-text-v1.5, tasks/0069-0070): RoPE on Q/K inside the
// fused qkv buffer, and a gated SwiGLU FFN in place of plain GELU. Both are
// false/0 for every arch=0 (BERT) container -- apply_model_shape() is the only
// writer, same discipline as every other g_* geometry field above.
extern bool g_rope, g_gated_ffn;
extern double g_rope_theta;
// arch=3 (gte-multilingual-base, tasks/0134-0136): the RoPE frequency set IS
// the model -- inv_freq_i = 160000^(-i/32) / 8^(1/32), which is NOT
// expressible as any single theta (the NTK correction is a constant factor,
// not a power law; deriving from rope_theta alone is wrong by 1.9e-02 relfro
// at layer 0, measured in tasks/0134). Read from the container's
// "rope_inv_freq" config array; EMPTY for every other arch, in which case
// g_rope_theta is the source. apply_model_shape() is the only writer.
namespace detail {
inline std::vector<float> &mut_rope_inv_freq() {
  static std::vector<float> v;
  return v;
}
}  // namespace detail
inline const std::vector<float> &rope_inv_freq() {
  return detail::mut_rope_inv_freq();
}
// Which activation the gated FFN applies to its gate half. SiLU used to be
// hardcoded while the container's "activation" key was write-only (T33's
// latent key, made load-bearing by tasks/0135): arch=2 says "silu", arch=3
// says "gelu" -- torch's default EXACT erf GELU, NOT gelu8's polynomial and
// NOT Gemma's tanh approximation. An unknown value REFUSES at load.
// apply_model_shape() is the only writer; irrelevant when g_gated_ffn is false.
enum class GatedAct { Silu, GeluErf };
extern GatedAct g_gated_act;

// Exact erf GELU: 0.5*x*(1+erf(x/sqrt(2))), computed in double like the
// numpy oracle (reference/encoder_gte.py's gelu_exact* both erf in float64)
// and rounded once at the end. Scalar on purpose: correctness first, and the
// bfp16 datapath noise (~2e-04) is three decades above the double-vs-float
// difference this choice removes from the comparison.
inline float gelu_erf_exact(float x) {
  const double xd = static_cast<double>(x);
  return static_cast<float>(
      0.5 * xd * (1.0 + std::erf(xd * 0.70710678118654752440)));
}

// nomic's task-prefix table (tasks/0071): name -> literal prefix text, e.g.
// "search_document" -> "search_document: ". Empty for every container that
// carries no "prompts" key -- the four BERT models have no prefix concept at
// all, and `prompts().empty()` is the single source of truth for that,
// rather than a second bool that could drift from it. apply_model_shape() is
// the only writer, cleared unconditionally on every call for the same
// "no container can leak state into the next" reason as g_rope/g_gated_ffn.
namespace detail {
inline std::map<std::string, std::string> &mut_prompts() {
  static std::map<std::string, std::string> v;
  return v;
}
inline std::string &mut_prompt_default() {
  static std::string v;
  return v;
}
}  // namespace detail
inline const std::map<std::string, std::string> &prompts() {
  return detail::mut_prompts();
}
inline const std::string &prompt_default() {
  return detail::mut_prompt_default();
}

// The container's prompt names, sorted, and a formatter for them. ONE source,
// used by every refusal that has to list them -- the CLI's, the endpoint's and
// /health's -- so the three cannot drift into disagreeing about what this
// model offers.
inline std::vector<std::string> prompt_names_sorted() {
  std::vector<std::string> names;
  names.reserve(prompts().size());
  for (const auto &kv : prompts()) names.push_back(kv.first);
  std::sort(names.begin(), names.end());
  return names;
}

inline std::string join_names(const std::vector<std::string> &names) {
  std::string list;
  for (size_t i = 0; i < names.size(); ++i) list += (i ? ", " : "") + names[i];
  return list;
}

// Batch is NOT a constant: it is read back from the loaded design's M, so the
// runtime cannot disagree with the xclbin it was handed. Every GEMM in the
// encoder is over all tokens of all sequences at once, so batching is purely a
// larger M -- and it is the lever that survives tasks/0024, because the 49
// design switches per encode cost the same no matter how many sequences that
// encode carries.
// A config key added after the container format already shipped has to be
// readable from containers that predate it. This repo's rule is that a missing
// key THROWS rather than defaulting -- right for geometry, which must never be
// guessed -- so the back-compat default is stated explicitly, here, at the one
// place it applies, rather than by weakening config_string(). Same shape as the
// pre-0036 `tokenizer.vocab` fallback in load_tokenizer().
//
// `config_string` returns the raw JSON scalar text, so a JSON `true` arrives as
// the four characters "true".
// Which container architectures this build can actually EXECUTE, as opposed to
// merely load. Kept in one place so the `list` table and the dispatch-time
// refusal in apply_model_shape() cannot drift apart: a table that says "ready"
// while dispatch throws is its own kind of lie, just a politer one.
//
// A packed container and a matching design are NOT sufficient. arch=2 (nomic)
// deliberately reuses BERT's tensor names and shapes so the packer and the NPU
// dispatch path work unchanged -- which means the BERT encoder will read it
// happily and compute the wrong model. tasks/0069.
inline bool encoder_implemented(const std::string &arch) {
  return arch == "bert_abs_gelu_postln" ||      // Encoder, NPU GEMM path
         arch == "nomic_bert_rope_swiglu" ||    // Encoder, NPU GEMM path (0070)
         // GemmaNpuEncoder on the array, or the host-only npue::GemmaEncoder
         // when the container is not pre-tiled. Which one runs is decided from
         // the container's `gemm_layout`, in run_gemma_mode() (tasks/0074).
         arch == "gemma3_mqa_rope_geglu" ||
         // gte-multilingual-base (0.5.0): BERT tensor names ON PURPOSE, so
         // the packer and the NPU dispatch path serve it unchanged; the
         // encoder deltas -- RoPE from rope_inv_freq, exact-erf GELU on the
         // gate half, real biases, XLM-R Unigram tokenizer -- are all
         // data-driven off the container (tasks/0134-0136).
         arch == "gte_new_rope_geglu";
}

inline bool config_flag(const npue::File &f, const char *key, bool fallback) {
  try {
    return f.config_string(key) == "true";
  } catch (const std::exception &) {
    return fallback;   // container predates the key: arch 0 and 1 are ungated
  }
}
// The vocabulary lives inside the .npue as of 0036, so a deployed model is
// ONE file. A model packed before that still works: fall back to the loose
// vocab.txt and say so, rather than failing on a file that is merely older.
// One tokenizer interface for the BERT-family encode path, two tokenizer
// families behind it (tasks/0136). The facade lives HERE rather than giving
// XlmrTokenizer a WordPiece-shaped encode(), because the max_len /
// padding / truncation semantics are this runtime's policy (0110's
// refuse-on-overflow contract runs on the Encoded fields), not a property
// of the Unigram algorithm -- tokenizer_xlmr.cpp stays the line-for-line
// port of its Python reference, diffable function by function. Chosen over
// branching at the call sites because the encode() calls sit inside
// EmbedService::chunk() and the --tokenize loop, and a branch at each
// would be the drift-prone shape encoder_implemented() exists to prevent.
struct AnyTokenizer {
  std::unique_ptr<npue::Tokenizer> wordpiece;
  std::unique_ptr<npue::XlmrTokenizer> xlmr;

  size_t vocab_size() const {
    return wordpiece ? wordpiece->vocab_size() : xlmr->vocab_size();
  }

  npue::Encoded encode(const std::string &text, int max_len) const {
    if (wordpiece) return wordpiece->encode(text, max_len);
    // XLM-R Unigram. XlmrTokenizer::encode() returns the FULL <s>...</s>
    // sequence, unpadded and untruncated (its header: an input that does
    // not fit is the caller's error to raise, not the tokenizer's to
    // hide). This adds the WordPiece path's exact max_len semantics on
    // top: truncation keeps <s> + the first (max_len - 2) pieces + </s>,
    // which is HuggingFace's longest_first truncation under the
    // "<s> A </s>" post-processor, so --tokenize stays diffable against
    // AutoTokenizer. n_tokens_full/truncated feed check_truncation()
    // unchanged -- 0110's refuse-on-overflow applies to arch=3 exactly as
    // to arch=0/2.
    std::vector<int32_t> full = xlmr->encode(text);
    npue::Encoded e;
    e.n_tokens_full = static_cast<int32_t>(full.size());
    e.truncated = e.n_tokens_full > max_len;
    if (e.truncated) {
      full.resize(static_cast<size_t>(max_len));
      full.back() = xlmr->eos_id;
    }
    e.n_tokens = static_cast<int32_t>(full.size());
    e.input_ids = std::move(full);
    e.input_ids.resize(static_cast<size_t>(max_len), xlmr->pad_id);
    e.attention_mask.assign(static_cast<size_t>(max_len), 0);
    for (int32_t s = 0; s < e.n_tokens; ++s) e.attention_mask[s] = 1;
    e.token_type_ids.assign(static_cast<size_t>(max_len), 0);
    return e;
  }
};

inline AnyTokenizer load_tokenizer(npue::File &model,
                            const std::string &model_path) {
  // arch=3: the XLMRTOK1 Unigram blob, stored whole in the container
  // (tasks/0135) and consumed in place. The ARCH decides, not the absence
  // of a tokenizer.vocab key -- absence already means something else below.
  std::string arch;
  try {
    arch = model.config_string("arch");
  } catch (const std::exception &) {
    // Pre-arch container: WordPiece, like everything else that old.
  }
  if (arch == "gte_new_rope_geglu") {
    auto v = model.raw("tokenizer.xlmr_table");
    AnyTokenizer t;
    t.xlmr = std::make_unique<npue::XlmrTokenizer>(
        npue::XlmrTokenizer::from_table_bytes(
            reinterpret_cast<const char *>(v.data), v.bytes));
    // The facade's padding and the sequence template lean on XLM-R's
    // specials -- check the blob rather than assume it.
    if (t.xlmr->bos_id != 0 || t.xlmr->eos_id != 2 || t.xlmr->pad_id != 1)
      throw std::runtime_error(
          "tokenizer.xlmr_table specials are not XLM-R's <s>=0, </s>=2, "
          "<pad>=1 -- refusing rather than padding with the wrong id");
    return t;
  }
  try {
    auto v = model.raw("tokenizer.vocab");
    AnyTokenizer t;
    t.wordpiece = std::make_unique<npue::Tokenizer>(
        npue::Tokenizer::from_vocab_bytes(
            reinterpret_cast<const char *>(v.data), v.bytes));
    return t;
  } catch (const std::exception &) {
    // Pre-0036 container: the loose checkpoint directory beside it, derived
    // from the container's own name rather than assumed to be MiniLM's.
    const std::string p =
        std::filesystem::path(model_path).replace_extension().string() +
        "/vocab.txt";
    std::printf("  tokenizer  .npue has no vocabulary; using %s\n", p.c_str());
    AnyTokenizer t;
    t.wordpiece = std::make_unique<npue::Tokenizer>(
        npue::Tokenizer::from_vocab_file(p));
    return t;
  }
}

// One entry of gemm_rtp's `streams` array: which instruction-stream slot
// runs which op at which batch tier. Parsed here rather than in npu_device
// because it is encoder policy, not device mechanics.
struct StreamEntry {
  std::string op, file;
  int64_t batch = 0, slot = 0, M = 0, K = 0, N = 0;
};

inline std::vector<StreamEntry> parse_streams(const std::string &json) {
  std::vector<StreamEntry> out;
  size_t i = json.find("\"streams\"");
  if (i == std::string::npos) return out;
  i = json.find('[', i);
  if (i == std::string::npos) return out;
  const size_t end = json.find(']', i);
  auto str_field = [&](size_t from, size_t to, const char *key) {
    const std::string k = std::string("\"") + key + "\"";
    size_t a = json.find(k, from);
    if (a == std::string::npos || a > to) return std::string();
    a = json.find('"', json.find(':', a) + 1) + 1;
    return json.substr(a, json.find('"', a) - a);
  };
  auto int_field = [&](size_t from, size_t to, const char *key) -> int64_t {
    const std::string k = std::string("\"") + key + "\"";
    size_t a = json.find(k, from);
    if (a == std::string::npos || a > to) return 0;
    return std::stoll(json.substr(json.find(':', a) + 1));
  };
  size_t p = i;
  while (true) {
    const size_t ob = json.find('{', p);
    if (ob == std::string::npos || ob > end) break;
    const size_t cb = json.find('}', ob);
    StreamEntry e;
    e.op = str_field(ob, cb, "op");
    e.file = str_field(ob, cb, "file");
    e.batch = int_field(ob, cb, "batch");
    e.slot = int_field(ob, cb, "slot");
    e.M = int_field(ob, cb, "M");
    e.K = int_field(ob, cb, "K");
    e.N = int_field(ob, cb, "N");
    if (!e.op.empty()) out.push_back(e);
    p = cb + 1;
  }
  return out;
}

// Pool [take, seq, hidden] hidden states into [take, hidden], then optionally
// L2 normalise. ONE implementation: there were three, and they disagreed --
// the golden path accumulated in float while the other two used double, so a
// comment claiming they matched was wrong by a rounding.
//
// `am` is the 1/0 attention mask, [rows, seq].
inline void pool_rows(const float *h, const float *am, int64_t take, float *out) {
  std::vector<double> acc(static_cast<size_t>(g_hidden));
  for (int64_t b = 0; b < take; ++b) {
    const float *amb = am + b * g_seq;
    const float *hb = h + b * g_seq * g_hidden;

    if (g_cls_pool) {
      // The [CLS] token is position 0 by construction (tokenizer.cpp emits it
      // first). If it is masked the sequence is empty, and returning zeros
      // would be a silently plausible answer.
      if (amb[0] == 0.f)
        throw std::runtime_error("CLS pooling on a sequence whose first "
                                 "token is masked");
      for (int64_t c = 0; c < g_hidden; ++c) acc[c] = hb[c];
    } else {
      float denom = 0.f;
      for (int64_t s = 0; s < g_seq; ++s) denom += amb[s];
      denom = std::max(denom, 1e-9f);
      std::fill(acc.begin(), acc.end(), 0.0);
      for (int64_t s = 0; s < g_seq; ++s) {
        const float m = amb[s];
        if (m == 0.f) continue;
        const float *hr = hb + s * g_hidden;
        for (int64_t c = 0; c < g_hidden; ++c) acc[c] += hr[c] * m;
      }
      for (int64_t c = 0; c < g_hidden; ++c) acc[c] /= denom;
    }

    float *o = out + b * g_hidden;
    if (g_l2_normalize) {
      double nrm = 0.0;
      for (int64_t c = 0; c < g_hidden; ++c) nrm += acc[c] * acc[c];
      nrm = std::sqrt(std::max(nrm, 1e-24));
      for (int64_t c = 0; c < g_hidden; ++c)
        o[c] = static_cast<float>(acc[c] / nrm);
    } else {
      for (int64_t c = 0; c < g_hidden; ++c) o[c] = static_cast<float>(acc[c]);
    }
  }
}
// Populate the geometry from the container. Every value is REQUIRED: a
// missing key throws from npue::File rather than defaulting, because a
// default here is indistinguishable from a correct value and this project has
// shipped six bugs of exactly that shape.
namespace detail {
inline void apply_model_shape(npue::File &m) {
  // FAIL CLOSED ON AN ARCHITECTURE THIS BINARY DOES NOT IMPLEMENT.
  //
  // Encoder::run() started as a pure BERT forward pass: absolute positions, a
  // plain GELU FFN, no rotary anything. arch=2 (nomic) deliberately reuses
  // BERT's tensor names and shapes -- that is what makes the packer and NPU
  // dispatch path free -- so a container this build does NOT implement would
  // otherwise be read happily and run through the wrong math, returning
  // embeddings that look entirely reasonable. Nothing downstream could tell.
  //
  // The arch=1 (Gemma) containers are dispatched away before they reach here,
  // but that check lives at three call sites and works by naming the ONE arch
  // it diverts; anything it does not recognise falls through to this path. So
  // the guard has to be here, stated as a whitelist of what is IMPLEMENTED
  // rather than a blacklist of what is not. tasks/0069, extended for arch=2
  // in tasks/0070 -- encoder_implemented() is the single source both this
  // refusal and the `list`/`serve` tables read, so they cannot drift.
  const std::string arch = m.config_string("arch");
  // SINGLE-SOURCED as of tasks/0136: this refusal used to restate the
  // whitelist as inline string literals while the comment above claimed
  // encoder_implemented() was "the single source" -- so a new arch could
  // land in one list and still be refused by the other. It calls the real
  // list now. The gemma diversion stays as-is: arch=1 IS implemented, but by
  // run_gemma_mode(), not by this BERT-family path -- a gemma container
  // reaching here means that diversion was bypassed, and running it through
  // the setup below would be the exact fail-open this guard exists to stop.
  if (!encoder_implemented(arch) || arch == "gemma3_mqa_rope_geglu")
    throw std::runtime_error(
        "container architecture '" + arch + "' has no encoder in this build. "
        "The NPU GEMM designs for it may well be present -- the tensor names "
        "and shapes are shared with BERT on purpose -- but running it through "
        "the BERT encoder would silently return embeddings for the wrong "
        "model. Refusing.");

  g_layers = m.config_int("num_layers");
  g_hidden = m.config_int("hidden");
  g_heads = m.config_int("num_heads");
  g_head_dim = m.config_int("head_dim");
  g_ffn = m.config_int("intermediate");
  detail::mut_source_repo() = m.config_string("source_repo");
  // NOT g_seq: `max_seq_len` is how many position embeddings were packed,
  // which is 256 while the designs are compiled for 64. The sequence length
  // belongs to the design and is set by set_design_seq().
  g_max_positions = m.config_int("max_seq_len");

  // arch=2: RoPE on Q/K (never V) inside the fused qkv buffer, and a gated
  // SwiGLU FFN. Both default false/0 -- deterministic every call, so a BERT
  // container after a nomic one in the same process (there is none today,
  // but nothing enforces that) cannot inherit stale state.
  g_gated_ffn = config_flag(m, "gated_ffn", false);
  g_rope = false;
  g_rope_theta = 0.0;
  detail::mut_rope_inv_freq().clear();
  g_gated_act = GatedAct::Silu;
  if (arch == "nomic_bert_rope_swiglu") {
    const std::string pet = m.config_string("position_embedding_type");
    if (pet != "rope")
      throw std::runtime_error(
          "container arch is nomic_bert_rope_swiglu but "
          "position_embedding_type is '" + pet + "', expected 'rope' -- "
          "refusing rather than guessing how position is encoded");
    if (!g_gated_ffn)
      throw std::runtime_error(
          "container arch is nomic_bert_rope_swiglu but gated_ffn is not "
          "true -- refusing rather than running a plain (ungated) FFN over "
          "a fused fc11|fc12 weight");
    // swiglu_halves pins which half of the fused ffn_up gets SiLU. READ IT,
    // do not trust the constant -- tools/pack_npue.py writes this exact
    // string today, but a packer that silently changed the fusion order
    // would otherwise compute out = silu(fc11(x)) * fc12(x), the wrong
    // candidate tasks/0068 Q2 measured at rel_fro 4.022e+00 (2.5e7x worse).
    const std::string halves = m.config_string("swiglu_halves");
    if (halves != "fc11_up|fc12_gate")
      throw std::runtime_error(
          "unrecognised swiglu_halves ordering '" + halves + "' -- expected "
          "'fc11_up|fc12_gate'; refusing rather than guessing which half of "
          "the fused ffn_up gets SiLU");
    g_rope_theta = m.config_double("rope_theta");
    if (g_rope_theta <= 0.0)
      throw std::runtime_error(
          "nomic_bert_rope_swiglu container has a non-positive rope_theta");
    g_rope = true;
  }

  // arch=3 (tasks/0136): nomic's shape with three deltas, every one read
  // from the container rather than assumed -- exact-erf GELU on the gate
  // half (the "activation" key, below), real biases (the bias slots are
  // added unconditionally, so nothing here changes), and a RoPE frequency
  // set that is DATA, because no single theta can express it (tasks/0134).
  if (arch == "gte_new_rope_geglu") {
    const std::string pet = m.config_string("position_embedding_type");
    if (pet != "rope")
      throw std::runtime_error(
          "container arch is gte_new_rope_geglu but position_embedding_type "
          "is '" + pet + "', expected 'rope' -- refusing rather than "
          "guessing how position is encoded");
    if (!g_gated_ffn)
      throw std::runtime_error(
          "container arch is gte_new_rope_geglu but gated_ffn is not true -- "
          "refusing rather than running a plain (ungated) FFN over a fused "
          "up|gate weight");
    // Same job as arch=2's swiglu_halves assert: pin which half of the fused
    // ffn_up is the gate. The key is descriptive prose after the marker, so
    // match the marker prefix, not the whole string.
    const std::string halves = m.config_string("glu_halves");
    if (halves.rfind("up_first|gate_second", 0) != 0)
      throw std::runtime_error(
          "unrecognised glu_halves ordering '" + halves + "' -- expected it "
          "to begin 'up_first|gate_second'; refusing rather than guessing "
          "which half of the fused ffn_up gets the activation");
    // rope_inv_freq IS the model (tasks/0134): inv_freq_i =
    // 160000^(-i/32) / 8^(1/32). A container without it REFUSES -- falling
    // back to deriving from rope_theta is measured wrong by 1.9e-02 relfro
    // at layer 0, and silently so.
    std::string raw_freq;
    try {
      raw_freq = m.config_string("rope_inv_freq");
    } catch (const std::exception &) {
      throw std::runtime_error(
          "gte_new_rope_geglu container carries no 'rope_inv_freq' -- the "
          "frequency set is not derivable from rope_theta (wrong by 1.9e-02 "
          "relfro at layer 0, tasks/0134), so refusing rather than falling "
          "back. Repack with tools/pack_npue.py");
    }
    const npue::json::Value v = npue::json::parse(raw_freq);
    for (const auto &e : v.as_array())
      detail::mut_rope_inv_freq().push_back(
          static_cast<float>(e.as_number()));
    if (static_cast<int64_t>(rope_inv_freq().size()) != g_head_dim / 2)
      throw std::runtime_error(
          "rope_inv_freq has " + std::to_string(rope_inv_freq().size()) +
          " entries, expected head_dim/2 = " +
          std::to_string(g_head_dim / 2));
    g_rope = true;
  }

  // The gated activation is DATA (tasks/0135 made the write-only key
  // load-bearing). Missing on an arch=2 container means "silu" -- packed
  // nomic containers may predate the read -- but an arch=3 container
  // without it is malformed, and an unknown value refuses on either arch.
  if (g_gated_ffn && arch != "gemma3_mqa_rope_geglu") {
    std::string act;
    try {
      act = m.config_string("activation");
    } catch (const std::exception &) {
      if (arch == "gte_new_rope_geglu")
        throw std::runtime_error(
            "gte_new_rope_geglu container carries no 'activation' key -- "
            "refusing rather than guessing which activation the gate half "
            "gets");
      act = "silu";
    }
    if (act == "silu")
      g_gated_act = GatedAct::Silu;
    else if (act == "gelu")
      g_gated_act = GatedAct::GeluErf;
    else
      throw std::runtime_error(
          "unknown gated-FFN activation '" + act + "' -- this build "
          "implements 'silu' (SiLU, arch=2) and 'gelu' (exact erf GELU, "
          "arch=3); refusing rather than substituting one");
  }

  // The task-prefix table (tasks/0071). Optional: the four BERT models'
  // containers carry no "prompts" key at all, and that has to leave
  // prompts() genuinely empty -- not throw -- so resolve_prefix() below can
  // use emptiness as "this model has no prefix concept" without a second
  // flag that could drift from it. config_string() throws on a missing key,
  // so the absence check is a try/catch, same shape as config_flag() above.
  detail::mut_prompts().clear();
  detail::mut_prompt_default().clear();
  {
    std::string raw;
    try {
      raw = m.config_string("prompts");
    } catch (const std::exception &) {
      // No "prompts" key -- this container has no task-prefix concept.
      // prompts() stays empty, which IS the "no prefix" signal.
    }
    if (!raw.empty()) {
      const npue::json::Value v = npue::json::parse(raw);
      for (const auto &kv : v.as_object())
        detail::mut_prompts()[kv.first] = kv.second.as_string();
      if (prompts().empty())
        throw std::runtime_error(
            "container has a 'prompts' key but it parsed to zero entries -- "
            "refusing rather than silently running with no prefix");
      // prompt_default is REQUIRED once prompts exists: a container that
      // advertises a prefix table but names no default is malformed, not
      // merely prefix-less.
      detail::mut_prompt_default() = m.config_string("prompt_default");
      if (prompts().find(prompt_default()) == prompts().end())
        throw std::runtime_error(
            "prompt_default '" + prompt_default() + "' is not a key in "
            "this container's own prompts table");
    }
  }

  // Pooling is data. sentence-transformers ships the answer in
  // 1_Pooling/config.json and the packer copies it here; a container that
  // predates that carries "mean", which is what MiniLM wants anyway.
  const std::string pool = m.config_string("pooling");
  if (pool == "cls") g_cls_pool = true;
  else if (pool == "mean") g_cls_pool = false;
  else throw std::runtime_error("unknown pooling mode '" + pool +
                                "' in the .npue -- expected mean or cls");

  // SO IS NORMALISATION, and it was a literal that should have been data
  // (T33). `pack_npue.py` has always written `l2_normalize` into every
  // container and this runtime always ignored it, which stayed harmless only
  // because every model so far wanted `true`.
  //
  // Reading it changes NOTHING today -- every shipped container says true, and
  // the default here is still true for a container that predates the key. What
  // it changes is that the assumption is now stated by the data rather than
  // asserted by the binary, so deciding nomic's case becomes a repack instead
  // of a code change. The decision itself is still open: nomic's
  // sentence-transformers pipeline has no `Normalize` module and returns
  // vectors of norm ~20.9, and 0073 measured Banking77 at 83.77 unnormalised
  // against 79.23 normalised -- 4.5 points on a logistic-regression task,
  // invisible to cosine. That is a question about which geometry downstream
  // code wants, not a precision question, and it is the user's to answer.
  g_l2_normalize = config_flag(m, "l2_normalize", true);

  if (g_layers <= 0 || g_hidden <= 0 || g_heads <= 0 || g_max_positions <= 0)
    throw std::runtime_error("the .npue reports a non-positive shape");
  if (g_head_dim * g_heads != g_hidden)
    throw std::runtime_error("head_dim * heads != hidden in the .npue");
  if (g_head_dim % 8 || g_head_dim / 8 > kMaxHeadVecs)
    throw std::runtime_error(
        "head_dim " + std::to_string(g_head_dim) + " must be a multiple of 8 "
        "and at most " + std::to_string(kMaxHeadVecs * 8) +
        " for the host attention kernels");
  // The host AVX2 paths step 8 floats with no scalar tail.
  if (g_hidden % 8)
    throw std::runtime_error("this runtime requires hidden to be a multiple "
                             "of 8");
}
}  // namespace detail

// THE SHAPE IS A LEASE, NOT A SETTER (tasks/0156, T63).
//
// The geometry above is process-wide, and a second model loaded on top of the
// first would overwrite hidden/heads/layers while the first Encoder kept
// running. THAT FAILURE IS NOT A CRASH. The second model's shape is a
// perfectly valid shape; the first model's weights would simply be read with
// the wrong strides, and what comes back is a correctly sized, correctly
// normed, deterministic vector with a success status. It is the same
// fail-open family as traps 6b/6c/7c/7d, one level further out.
//
// So the shape can only be set by taking a lease on it, and a second lease
// REFUSES. Two properties matter here and neither is decoration:
//
//  * The lease is the ONLY entrance. apply_model_shape() is in `detail` and
//    nothing outside this file calls it -- main() used to call the setter
//    directly, so a guard placed inside a later facade would have left a
//    second door standing open. Putting the guard at the setter is what makes
//    it exhaustive.
//  * It RELEASES on destruction. The obvious alternative -- a one-shot "the
//    shape has been set" flag -- would break the host we are heading for:
//    OpenFlowLM-Next's get_auto_embedding_model() constructs a fresh backend
//    per model switch, so open -> destroy -> open is its NORMAL path, and a
//    latch would refuse the second model for no reason.
//
// A holder must therefore outlive every Encoder built against it. In a class
// that owns both, declare the lease FIRST so it is destroyed LAST.
class ShapeLease {
public:
  explicit ShapeLease(npue::File &m) {
    bool expected = false;
    if (!detail::shape_held().compare_exchange_strong(expected, true))
      throw std::runtime_error(
          "a model shape is already loaded in this process. The encoder's "
          "geometry (hidden, heads, layers, seq) is process-wide, so a second "
          "model would silently reinterpret the first one's weights and "
          "return plausible embeddings for neither. Release the first "
          "ShapeLease, or run the second model in its own process.");
    try {
      detail::apply_model_shape(m);
    } catch (...) {
      // A container that fails validation leaves the geometry half-written,
      // which is exactly the stale state the lease exists to prevent. Hand
      // the lease back so the caller can try another container.
      detail::shape_held().store(false);
      throw;
    }
  }
  ~ShapeLease() { detail::shape_held().store(false); }
  ShapeLease(const ShapeLease &) = delete;
  ShapeLease &operator=(const ShapeLease &) = delete;
};

// The sequence length comes from the design, and the container has to be able
// to feed it. Two independent sources that must agree in one direction: a
// design asking for more positions than were packed would index past the
// position table.
inline void set_design_seq(int64_t seq) {
  if (seq <= 0 || seq % 8)
    throw std::runtime_error("design seq " + std::to_string(seq) +
                             " must be positive and a multiple of 8");
  if (seq > g_max_positions)
    throw std::runtime_error(
        "design seq " + std::to_string(seq) + " exceeds the " +
        std::to_string(g_max_positions) + " position embeddings in the .npue");
  g_seq = seq;
}
// fp32 -> bf16, round-to-nearest-even. The rounding tools/npue.py uses when
// packing; truncation would bias every value toward zero.
inline uint16_t to_bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, sizeof u);
  return static_cast<uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}
inline float from_bf16(uint16_t h) {
  uint32_t u = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

// The vectorised forms below are BIT-IDENTICAL to the scalar ones above, which
// is the only reason they are safe to swap in: every integer op used has the
// same semantics on uint32 as on __m256i lanes, and after the >> 16 the values
// are in [0, 65535] so packus never actually saturates. The scalar tail keeps
// the two paths agreeing on any n.
//
// 13.8 M elements per encode go through these (tasks/0024), which is why they
// are worth writing out.
#if defined(__AVX2__)
#include <immintrin.h>

inline void bf16_fill(void *dst, const float *src, size_t n) {
  auto *d = static_cast<uint16_t *>(dst);
  const __m256i k7fff = _mm256_set1_epi32(0x7FFF);
  const __m256i kone = _mm256_set1_epi32(1);
  auto rne = [&](__m256i u) {
    __m256i odd = _mm256_and_si256(_mm256_srli_epi32(u, 16), kone);
    return _mm256_srli_epi32(
        _mm256_add_epi32(u, _mm256_add_epi32(k7fff, odd)), 16);
  };
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m256i a = rne(_mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(src + i)));
    __m256i b = rne(_mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(src + i + 8)));
    // packus interleaves the two 128-bit lanes; 0xD8 puts them back in order.
    __m256i p = _mm256_permute4x64_epi64(_mm256_packus_epi32(a, b), 0xD8);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(d + i), p);
  }
  for (; i < n; ++i) d[i] = to_bf16(src[i]);
}

inline void bf16_read(float *dst, const void *src, size_t n) {
  const auto *s = static_cast<const uint16_t *>(src);
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(s + i));
    __m256i u = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
    _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(u));
  }
  for (; i < n; ++i) dst[i] = from_bf16(s[i]);
}
#else
inline void bf16_fill(void *dst, const float *src, size_t n) {
  auto *d = static_cast<uint16_t *>(dst);
  for (size_t i = 0; i < n; ++i) d[i] = to_bf16(src[i]);
}
inline void bf16_read(float *dst, const void *src, size_t n) {
  const auto *s = static_cast<const uint16_t *>(src);
  for (size_t i = 0; i < n; ++i) dst[i] = from_bf16(s[i]);
}
#endif

inline double now_s() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

// One dispatch of the unified GEMM design, with the lock scope g_wide_lock
// selects. Shared by the BERT and the Gemma encoders, which had two
// byte-identical copies of this block.
//
// Returns the C pointer -- THIS lane's own slot, which no other lane binds, so
// it stays valid after the lock is dropped (the comment that used to sit at
// the old unlock point; it is now load-bearing for the wait as well).
inline const float *npu_dispatch(npu::Design &d, std::mutex *npu_mu, bool set_instr,
                          size_t islot, size_t slot_a, size_t wslot,
                          size_t slot_c, size_t a_elems, size_t c_bytes,
                          double &t0, double &t_in, double &t_disp,
                          double &t_out) {
  // The encoders' own `lap`, which uses no member state -- charge the elapsed
  // time to a bucket and return the new mark.
  auto lap = [](double from, double &bucket) {
    double t = now_s();
    bucket += t - from;
    return t;
  };
  if (g_wide_lock) {
    std::unique_lock<std::mutex> lk;
    if (npu_mu) lk = std::unique_lock<std::mutex>(*npu_mu);
    if (set_instr) d.bind_instr(islot);
    d.bind(0, slot_a);
    d.bind(1, wslot);
    d.bind(2, slot_c);
    d.sync_to_device(0, a_elems * d.info().a_elem_bytes);
    t0 = lap(t0, t_in);
    d.dispatch_only();
    t0 = lap(t0, t_disp);
    d.sync_from_device(2, c_bytes);
    t0 = lap(t0, t_out);
    return static_cast<const float *>(d.slot_ptr(2, slot_c));
  }
  // The operand crosses the bus addressed by SLOT, so no bind is involved and
  // no lock is needed.
  d.sync_slot_to_device(0, slot_a, a_elems * d.info().a_elem_bytes);
  t0 = lap(t0, t_in);
  npu::Dispatch h;
  {
    std::unique_lock<std::mutex> lk;
    if (npu_mu) lk = std::unique_lock<std::mutex>(*npu_mu);
    if (set_instr) d.bind_instr(islot);
    d.bind(0, slot_a);
    d.bind(1, wslot);
    d.bind(2, slot_c);
    // submit() reads the bind registers and captures the buffer objects into
    // the run; after it returns, another lane may rebind freely.
    h = d.submit();
  }
  d.wait(h);
  t0 = lap(t0, t_disp);
  d.sync_slot_from_device(2, slot_c, c_bytes);
  t0 = lap(t0, t_out);
  return static_cast<const float *>(d.slot_ptr(2, slot_c));
}

// ---------------------------------------------------------------------------
// int8 quantisation, shared by BOTH encoders.
//
// These were inline in Encoder::gemm() until tasks/0081 gave arch=1 an int8
// path too. GemmaNpuEncoder is a separate encoder (RMSNorm x4, MQA, per-layer
// RoPE, GeGLU), and copying ~120 lines of hand-vectorised quantisation into it
// would have created exactly the kind of duplicate that drifts: the reciprocal
// hoist below was a 63 ms fix found once (tasks/0080), and a second copy would
// not have it.
// ---------------------------------------------------------------------------

// A -> int8, per row, with the SmoothQuant divisor folded into the same pass
// (tasks/0078). `ias` is 1/asmooth, reciprocated ONCE by the caller: dividing
// by asmooth[j] in both the max pass and the quantise pass was two divisions
// per element and cost 74 ms of the 126 ms the array had saved.
//
// The divisor is NOT folded into the preceding norm. BERT is post-LN, so the
// norm's output feeds the residual as well as this GEMM (tasks/0078 4a); on
// Gemma the same holds for a different reason -- `pre_feedforward_layernorm`'s
// output is consumed by the GeGLU pair only, but `input_layernorm`'s feeds the
// residual, and one code path is worth more than one folded multiply.
template <typename ParRows>
void quantise_a_int8(const float *a, int64_t rows, int64_t K, const float *ias,
                     int8_t *q_base, float *a_scale, ParRows par_rows) {
  par_rows(rows, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const float *x = a + r * K;
      int8_t *q = q_base + r * K;
      float mx = 0.f;
      int64_t j = 0;
#if defined(__AVX2__)
      const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
      __m256 acc = _mm256_setzero_ps();
      for (; j + 8 <= K; j += 8)
        acc = _mm256_max_ps(acc, _mm256_and_ps(
            _mm256_mul_ps(_mm256_loadu_ps(x + j), _mm256_loadu_ps(ias + j)),
            absmask));
      __m128 h = _mm_max_ps(_mm256_castps256_ps128(acc),
                            _mm256_extractf128_ps(acc, 1));
      h = _mm_max_ps(h, _mm_movehl_ps(h, h));
      h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 1));
      mx = _mm_cvtss_f32(h);
#endif
      for (; j < K; ++j) {
        const float v = std::fabs(x[j] * ias[j]);
        if (v > mx) mx = v;
      }
      const float sc = mx > 0.f ? mx / 127.0f : 1.0f;
      a_scale[static_cast<size_t>(r)] = sc;
      const float inv = 1.0f / sc;
      j = 0;
#if defined(__AVX2__)
      const __m256 invv = _mm256_set1_ps(inv);
      const __m256 hi = _mm256_set1_ps(127.0f);
      const __m256 lo = _mm256_set1_ps(-127.0f);
      for (; j + 8 <= K; j += 8) {
        __m256 v = _mm256_mul_ps(
            _mm256_mul_ps(_mm256_loadu_ps(x + j), _mm256_loadu_ps(ias + j)),
            invv);
        v = _mm256_min_ps(_mm256_max_ps(v, lo), hi);
        // cvtps_epi32 rounds per MXCSR, i.e. nearest-even by default -- the
        // same rule tools/pack_npue.py's np.rint uses on the weights, so the
        // two halves of the product round the same way.
        __m256i i32 = _mm256_cvtps_epi32(v);
        __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(i32),
                                      _mm256_extracti128_si256(i32, 1));
        _mm_storel_epi64(reinterpret_cast<__m128i *>(q + j),
                         _mm_packs_epi16(p16, p16));
      }
#endif
      for (; j < K; ++j) {
        float v = std::nearbyintf(x[j] * ias[j] * inv);
        if (v > 127.f) v = 127.f;
        if (v < -127.f) v = -127.f;
        q[j] = static_cast<int8_t>(v);
      }
    }
  });
}

// GELU, the same degree-8 minimax polynomial gelu_cpu() uses, so the fused and
// unfused paths are bit-identical rather than merely close.
#if defined(__AVX2__)
inline __m256 gelu8(__m256 v) {
  const __m256 u = _mm256_min_ps(
      _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v), _mm256_set1_ps(4.0f));
  __m256 pl = _mm256_fmadd_ps(_mm256_set1_ps(-7.2340282171e-05f), u,
                              _mm256_set1_ps(1.8179518005e-03f));
  pl = _mm256_fmadd_ps(pl, u, _mm256_set1_ps(-1.7707383379e-02f));
  pl = _mm256_fmadd_ps(pl, u, _mm256_set1_ps(8.4577147641e-02f));
  pl = _mm256_fmadd_ps(pl, u, _mm256_set1_ps(-1.9228671834e-01f));
  pl = _mm256_fmadd_ps(pl, u, _mm256_set1_ps(9.8431124458e-02f));
  pl = _mm256_fmadd_ps(pl, u, _mm256_set1_ps(3.6137852062e-01f));
  pl = _mm256_fmadd_ps(pl, u, _mm256_set1_ps(-4.9454128936e-01f));
  pl = _mm256_fmadd_ps(pl, u, _mm256_set1_ps(-1.3007010117e-04f));
  return _mm256_add_ps(_mm256_max_ps(v, _mm256_setzero_ps()), pl);
}
#endif
inline float gelu8(float v) {
  const float u = std::min(std::fabs(v), 4.0f);
  float pl = -7.2340282171e-05f;
  pl = pl * u + 1.8179518005e-03f;
  pl = pl * u + -1.7707383379e-02f;
  pl = pl * u + 8.4577147641e-02f;
  pl = pl * u + -1.9228671834e-01f;
  pl = pl * u + 9.8431124458e-02f;
  pl = pl * u + 3.6137852062e-01f;
  pl = pl * u + -4.9454128936e-01f;
  pl = pl * u + -1.3007010117e-04f;
  return std::max(v, 0.0f) + pl;
}

// THE ffn_up EPILOGUE IN ONE PASS (tasks/0081 T37).
//
// The unfused chain walks the widest tensor in the model six times: the
// dequantiser writes fp32 `up`, GELU reads and writes it, and the next GEMM's
// quantiser reads it again and writes int8. At bge-large's batch 128 that
// tensor is 134 MB, so the chain is ~636 MB per layer and ~15 GB per encode --
// and tasks/0081 section 3 measured the host at 69.6% of the encode, nearly
// all of it memory traffic rather than arithmetic.
//
// Fused it is ~100 MB per layer: read C, and write ffn_down's int8 operand.
// Nothing else is materialised. The row (16 KB at most) stays in L1 across the
// three sub-passes, so the absmax GELU's output needs costs a cache hit rather
// than a DRAM sweep -- which is the whole reason a per-row activation scale is
// affordable at all (tasks/0079 measured the static alternative at 57x worse).
//
// Bit-identical to the unfused path by construction: same polynomial, same
// rounding, same order.
// `act` transforms the dequantised row IN PLACE and leaves `out_n` values at
// row[0, out_n) -- N for a plain GELU, N/2 for a gated FFN that combines two
// halves into one. It is a parameter rather than a branch so the caller keeps
// ownership of the exact intrinsics, which is what makes the fused and
// unfused paths bit-identical rather than merely close.
template <typename ParRows, typename Act>
void dequant_act_quant(const void *c, size_t c_bytes, int64_t rows, int64_t N,
                       int64_t out_n, Act act,
                       const float *sa_up, const float *wscale,
                       const float *bias, const float *ias_next,
                       int8_t *dst, float *sa_next, ParRows par_rows) {
  par_rows(rows, [&](int64_t r0, int64_t r1) {
    std::vector<float> row(static_cast<size_t>(N));
    for (int64_t r = r0; r < r1; ++r) {
      const float sa = sa_up[static_cast<size_t>(r)];
      float *v = row.data();
      int64_t j = 0;
      // 1. dequantise + bias + GELU, into L1. THE INTRINSICS MUST MATCH the
      // unfused path's exactly -- `_mm256_fmadd_ps` rounds once where
      // `a*b + c` rounds twice, so a scalar rewrite of the same formula is
      // NOT the same number. Measured: 1.161e-03 unfused against 1.180e-03
      // for a scalar fused pass. Both pass the gate, but a fused path that
      // silently changes the result is a fused path nobody can A/B.
#if defined(__AVX2__)
      const __m256 sav = _mm256_set1_ps(sa);
      if (c_bytes == 2) {
        const uint16_t *cr = static_cast<const uint16_t *>(c) + r * N;
        for (; j + 16 <= N; j += 16) {
          __m256i raw = _mm256_stream_load_si256(
              reinterpret_cast<const __m256i *>(cr + j));
          __m256 lo = _mm256_castsi256_ps(_mm256_slli_epi32(
              _mm256_cvtepu16_epi32(_mm256_castsi256_si128(raw)), 16));
          __m256 hi = _mm256_castsi256_ps(_mm256_slli_epi32(
              _mm256_cvtepu16_epi32(_mm256_extracti128_si256(raw, 1)), 16));
          _mm256_storeu_ps(v + j, _mm256_fmadd_ps(
              _mm256_mul_ps(lo, sav), _mm256_loadu_ps(wscale + j),
              _mm256_loadu_ps(bias + j)));
          _mm256_storeu_ps(v + j + 8, _mm256_fmadd_ps(
              _mm256_mul_ps(hi, sav), _mm256_loadu_ps(wscale + j + 8),
              _mm256_loadu_ps(bias + j + 8)));
        }
      } else {
        const int32_t *cr = static_cast<const int32_t *>(c) + r * N;
        for (; j + 8 <= N; j += 8) {
          __m256 cf = _mm256_cvtepi32_ps(_mm256_stream_load_si256(
              reinterpret_cast<const __m256i *>(cr + j)));
          _mm256_storeu_ps(v + j, _mm256_fmadd_ps(
              _mm256_mul_ps(cf, sav), _mm256_loadu_ps(wscale + j),
              _mm256_loadu_ps(bias + j)));
        }
      }
#endif
      for (; j < N; ++j) {
        const float cf = c_bytes == 2
            ? from_bf16(static_cast<const uint16_t *>(c)[r * N + j])
            : static_cast<float>(static_cast<const int32_t *>(c)[r * N + j]);
        v[j] = cf * sa * wscale[j] + bias[j];
      }
      // 2. the activation, in place, narrowing to out_n.
      act(v, N);
      // 3. row absmax of the smoothed value, and 4. quantise. Both over a row
      // that is now hot in L1, which is what makes this worth doing at all.
      float mx = 0.f;
      j = 0;
#if defined(__AVX2__)
      {
        const __m256 absmask =
            _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        __m256 acc = _mm256_setzero_ps();
        for (; j + 8 <= out_n; j += 8)
          acc = _mm256_max_ps(acc, _mm256_and_ps(
              _mm256_mul_ps(_mm256_loadu_ps(v + j),
                            _mm256_loadu_ps(ias_next + j)), absmask));
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(acc),
                              _mm256_extractf128_ps(acc, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 1));
        mx = _mm_cvtss_f32(h);
      }
#endif
      for (; j < out_n; ++j) {
        const float a = std::fabs(v[j] * ias_next[j]);
        if (a > mx) mx = a;
      }
      const float sc = mx > 0.f ? mx / 127.0f : 1.0f;
      sa_next[static_cast<size_t>(r)] = sc;
      const float inv = 1.0f / sc;
      int8_t *q = dst + r * out_n;
      j = 0;
#if defined(__AVX2__)
      {
        const __m256 invv = _mm256_set1_ps(inv);
        const __m256 vhi = _mm256_set1_ps(127.0f);
        const __m256 vlo = _mm256_set1_ps(-127.0f);
        for (; j + 8 <= out_n; j += 8) {
          __m256 t = _mm256_mul_ps(
              _mm256_mul_ps(_mm256_loadu_ps(v + j),
                            _mm256_loadu_ps(ias_next + j)), invv);
          t = _mm256_min_ps(_mm256_max_ps(t, vlo), vhi);
          __m256i i32 = _mm256_cvtps_epi32(t);
          __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(i32),
                                        _mm256_extracti128_si256(i32, 1));
          _mm_storel_epi64(reinterpret_cast<__m128i *>(q + j),
                           _mm_packs_epi16(p16, p16));
        }
      }
#endif
      for (; j < out_n; ++j) {
        float t = std::nearbyintf(v[j] * ias_next[j] * inv);
        if (t > 127.f) t = 127.f;
        if (t < -127.f) t = -127.f;
        q[j] = static_cast<int8_t>(t);
      }
    }
  });
}

// C -> fp32: y = acc * sa[row] * wscale[col] + bias[col]. A rank-1
// outer-product scaling folded into the pass that already reads C and adds the
// bias. `c_bytes` selects the transport width the design chose: 4 = int32
// accumulator straight out, 2 = narrowed to bf16 on the core (tasks/0080).
template <typename ParRows>
void dequantise_c(const void *c, size_t c_bytes, int64_t rows, int64_t N,
                  const float *a_scale, const float *wscale, const float *bias,
                  float *out, ParRows par_rows, bool sim_bf16 = false) {
  if (c_bytes == 2) {
    const uint16_t *cb = static_cast<const uint16_t *>(c);
    par_rows(rows, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const uint16_t *cr = cb + r * N;
        float *o = out + r * N;
        const float sa = a_scale[static_cast<size_t>(r)];
        int64_t j = 0;
#if defined(__AVX2__)
        const __m256 sav = _mm256_set1_ps(sa);
        for (; j + 16 <= N; j += 16) {
          // One 32-byte streaming load carries 16 bf16 against 8 int32 -- the
          // same instruction count for twice the elements, which is the whole
          // point of narrowing C.
          __m256i raw = _mm256_stream_load_si256(
              reinterpret_cast<const __m256i *>(cr + j));
          __m256 lo = _mm256_castsi256_ps(_mm256_slli_epi32(
              _mm256_cvtepu16_epi32(_mm256_castsi256_si128(raw)), 16));
          __m256 hi = _mm256_castsi256_ps(_mm256_slli_epi32(
              _mm256_cvtepu16_epi32(_mm256_extracti128_si256(raw, 1)), 16));
          _mm256_storeu_ps(o + j,
              _mm256_fmadd_ps(_mm256_mul_ps(lo, sav),
                              _mm256_loadu_ps(wscale + j),
                              _mm256_loadu_ps(bias + j)));
          _mm256_storeu_ps(o + j + 8,
              _mm256_fmadd_ps(_mm256_mul_ps(hi, sav),
                              _mm256_loadu_ps(wscale + j + 8),
                              _mm256_loadu_ps(bias + j + 8)));
        }
#endif
        for (; j < N; ++j)
          o[j] = from_bf16(cr[j]) * sa * wscale[j] + bias[j];
      }
    });
    return;
  }
  const int32_t *ci = static_cast<const int32_t *>(c);
  par_rows(rows, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const int32_t *cr = ci + r * N;
      float *o = out + r * N;
      const float sa = a_scale[static_cast<size_t>(r)];
      int64_t j = 0;
#if defined(__AVX2__)
      // Streaming loads: C is a write-combined XRT host bo and ordinary loads
      // from it stall per line (tasks/0024).
      const __m256 sav = _mm256_set1_ps(sa);
      for (; j + 8 <= N; j += 8) {
        __m256 cf = _mm256_cvtepi32_ps(_mm256_stream_load_si256(
            reinterpret_cast<const __m256i *>(cr + j)));
        if (sim_bf16) {
          // --sim-c-bf16: round exactly as a narrowed-C design would, to price
          // one before building it (tasks/0080). RNE to bf16 = add half an ulp
          // plus the tie-break bit, then truncate.
          __m256i u = _mm256_castps_si256(cf);
          u = _mm256_add_epi32(
              u, _mm256_add_epi32(
                     _mm256_set1_epi32(0x7FFF),
                     _mm256_and_si256(_mm256_srli_epi32(u, 16),
                                      _mm256_set1_epi32(1))));
          cf = _mm256_castsi256_ps(
              _mm256_and_si256(u, _mm256_set1_epi32(int(0xFFFF0000u))));
        }
        _mm256_storeu_ps(o + j,
            _mm256_fmadd_ps(_mm256_mul_ps(cf, sav),
                            _mm256_loadu_ps(wscale + j),
                            _mm256_loadu_ps(bias + j)));
      }
#endif
      for (; j < N; ++j) {
        float cf = static_cast<float>(cr[j]);
        if (sim_bf16) {
          uint32_t u;
          std::memcpy(&u, &cf, 4);
          u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
          std::memcpy(&cf, &u, 4);
        }
        o[j] = cf * sa * wscale[j] + bias[j];
      }
    }
  });
}

// A persistent pool, because attention is called 12 times per encode and
// spawning threads each time would cost more than it saves.
//
// The calling thread takes chunk 0 and participates, so `n` threads means
// n-1 spawned. Work is partitioned by (batch, head) pair, and every pair writes
// a disjoint slice of `scores` and `ctx`, so there is no sharing to guard.
class Pool {
public:
  explicit Pool(int n) : n_(n < 1 ? 1 : n) {
    for (int i = 1; i < n_; ++i)
      workers_.emplace_back([this, i] {
        int seen = 0;
        for (;;) {
          std::function<void(int, int)> f;
          {
            std::unique_lock<std::mutex> lk(m_);
            cv_work_.wait(lk, [&] { return quit_ || gen_ != seen; });
            if (quit_) return;
            seen = gen_;
            f = fn_;
          }
          f(i, n_);
          {
            std::lock_guard<std::mutex> lk(m_);
            if (--remaining_ == 0) cv_done_.notify_one();
          }
        }
      });
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> lk(m_);
      quit_ = true;
    }
    cv_work_.notify_all();
    for (auto &t : workers_) t.join();
  }
  Pool(const Pool &) = delete;
  Pool &operator=(const Pool &) = delete;

  int size() const { return n_; }

  void run(const std::function<void(int, int)> &f) {
    if (n_ == 1) { f(0, 1); return; }
    {
      std::lock_guard<std::mutex> lk(m_);
      fn_ = f;
      remaining_ = n_ - 1;
      ++gen_;
    }
    cv_work_.notify_all();
    f(0, n_);
    std::unique_lock<std::mutex> lk(m_);
    cv_done_.wait(lk, [&] { return remaining_ == 0; });
  }

private:
  int n_;
  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_work_, cv_done_;
  std::function<void(int, int)> fn_;
  int gen_ = 0, remaining_ = 0;
  bool quit_ = false;
};

#if defined(__AVX2__)
inline float hsum256(__m256 v) {
  __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v),
                         _mm256_extractf128_ps(v, 1));
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}
#endif
// The whole encoder. Designs are constructed once by the caller and reused --
// that is the point of the exercise.
struct Encoder {
  npue::File &model;
  npu::Design &qkv, &attn_out, &ffn_up, &ffn_down, &gelu, &layernorm, &softmax;


  // Staged once: the mask, in the form softmax consumes.
  std::vector<float> add_mask;   // [batch, g_seq]

  // Unified gemm_rtp mode (tasks/0032): all four GEMM refs above point at ONE
  // design; each op is an instruction-stream slot bound before dispatch. The
  // slot order is the export contract of tools/export_gemm_rtp.py:
  // qkv=0 (insts.bin), attn_out=1, ffn_up=2, ffn_down=3 (load_instr order).
  bool unified = false;
  size_t is_qkv = 0, is_ao = 0, is_fu = 0, is_fd = 0;

  // Batch tiers (0037): one xclbin carries a stream per (op, batch), so the
  // encoder can size a request instead of padding it to the largest design.
  // `tiers` is ascending; `use_tier` picks the smallest that fits and points
  // is_* at its slots.
  std::vector<int64_t> tiers;
  std::vector<std::array<size_t, 4>> tier_slots;   // qkv, attn_out, ffn_up, ffn_down

  int64_t use_tier(int64_t want) {
    if (tiers.empty()) return batch;
    size_t pick = tiers.size() - 1;
    for (size_t i = 0; i < tiers.size(); ++i)
      if (tiers[i] >= want) { pick = i; break; }
    batch = tiers[pick];
    rows = batch * g_seq;
    is_qkv = tier_slots[pick][0];
    is_ao = tier_slots[pick][1];
    is_fu = tier_slots[pick][2];
    is_fd = tier_slots[pick][3];
    return batch;
  }

  // Two-encode pipelining (tasks/0033): two Encoder instances share the ONE
  // unified design; each owns its A and C slots, and every NPU interaction
  // (bind + sync + dispatch) happens under this mutex. The array serializes
  // dispatches anyway (note 0004) -- the lock only makes explicit what the
  // hardware enforces -- while each pipeline's HOST work overlaps the other
  // pipeline's NPU work.
  std::mutex *npu_mu = nullptr;
  size_t slot_a = 0, slot_c = 0;

  int64_t batch = 0, rows = 0;   // rows = batch * g_seq, from the design's M
  Pool *pool = nullptr;

  // Chunk a flat range over the pool. Chunks are 64-element aligned so the AVX2
  // conversions never see a split vector and every worker takes the fast path.
  template <typename F> void par(size_t n, F &&f) const {
    if (pool == nullptr || pool->size() == 1 || n < 65536) { f(size_t(0), n); return; }
    pool->run([&](int w, int nw) {
      const size_t chunk = ((n / nw) + 63) & ~size_t(63);
      const size_t lo = std::min(n, chunk * size_t(w));
      const size_t hi = std::min(n, lo + chunk);
      if (lo < hi) f(lo, hi);
    });
  }

  // Row-parallel, for passes where a flat byte range would split a row --
  // int8 quantisation takes a per-row maximum, so a chunk boundary inside a
  // row would give two different scales to one row's halves (tasks/0078).
  template <typename F> void par_rows(int64_t n, F &&f) const {
    if (pool == nullptr || pool->size() == 1) { f(int64_t(0), n); return; }
    pool->run([&](int w, int nw) {
      const int64_t chunk = (n + nw - 1) / nw;
      const int64_t lo = std::min<int64_t>(n, chunk * w);
      const int64_t hi = std::min<int64_t>(n, lo + chunk);
      if (lo < hi) f(lo, hi);
    });
  }

  // Scratch reused across layers. `residual = x` used to allocate and copy
  // 12.6 MB per layer at batch 128 -- 75 MB per encode of pure copying.
  std::vector<float> residual;
  // Scratch, sized once. These used to be six fresh vectors per run() -- ~90
  // MB of allocate-and-touch per encode at batch 128, and ~280 MB at
  // bge-large's width. resize() after the first call is a no-op.
  std::vector<float> qkvbuf, ctx, proj, up, down, scores;
  // arch=2 only: swiglu_cpu()'s output, [rows][g_ffn] -- a separate buffer
  // from `up` because an in-place version of this compaction is NOT safe
  // under threading (see swiglu_cpu's own comment). Empty/unused for every
  // arch=0 container.
  std::vector<float> gated;

  // arch=2 RoPE tables, [g_seq, g_head_dim] each, built ONCE per Encoder (not
  // per layer, not per call) the first time apply_rope_qkv() runs. Every
  // layer shares the same table -- unlike Gemma, nomic has a single rope_theta
  // for the whole model, not a per-layer local/global split (gemma_kernels.hpp
  // trap 2b). Using npue::gemma_rope_tables() here even though nothing about
  // it is Gemma-specific: it is plain NeoX RoPE table construction, and
  // duplicating it for a second arch would be the actual mistake.
  std::vector<float> rope_cos, rope_sin;
  bool rope_ready = false;

  // Device-resident weights, one slot per layer per design, plus the bias
  // pointers straight into the mapped file. Filled by stage_all().
  std::vector<size_t> s_qkv, s_ao, s_fu, s_fd;
  std::vector<const float *> b_qkv, b_ao, b_fu, b_fd;
  // int8 only (tasks/0078): per-output-channel weight scales and the
  // per-input-channel SmoothQuant divisor, straight out of the mapped .npue.
  // Empty for every bf16 container, and the int8 path is selected by the
  // DESIGN's a_dtype, so a mismatch is caught by stage_all() rather than by
  // dereferencing an empty vector.
  std::vector<const float *> ws_qkv, ws_ao, ws_fu, ws_fd;
  std::vector<const float *> as_qkv, as_ao, as_fu, as_fd;
  std::vector<size_t> s_ln;      // 0 = embeddings, then ln1/ln2 per layer
  // Host-side views of the same parameters, straight into the mapped .npue.
  // tasks/0031 measured a LayerNorm call at 725 us of kernel inside ~3 ms of
  // switch+conversion; a threaded fp32 AVX2 LayerNorm on the host costs
  // ~0.5 ms and removes 13 design switches outright. --host-ln selects it.
  std::vector<const float *> h_gamma, h_beta;
  bool host_ln = false;
  bool host_sm = false;
  bool host_gelu = false;
  // Measure-before-build (tasks/0080). Under int8 the GEMM is bandwidth-bound
  // again (0010's model, which 0048 superseded for bf16), and C is 61% of the
  // traffic on three of four shapes -- so narrowing C is the top lever. Doing
  // it needs NO extra core input, because int32 -> bf16 is a pure format
  // conversion: the host still applies sa[i]*wscale[j]. What it costs is
  // accuracy, and that is knowable without writing a kernel. This flag rounds
  // the accumulator exactly as such a design would (int32 -> fp32 -> bf16,
  // round-to-nearest-even, i.e. conv_even per trap 2b) and changes nothing
  // else, so the 1-cos gate prices the design before it exists.
  bool sim_c_bf16 = false;
  // T37 (tasks/0081): fuse the ffn_up epilogue -- dequantise, GELU and the
  // next GEMM's quantisation in ONE pass, so the widest tensor in the model is
  // never materialised in fp32. On by default where it applies; --no-fuse-ffn
  // turns it off, which is how the two paths are compared.
  // T37-BF16 (tasks/0108): the same flag also gates the bf16/bfp16 analogue
  // (narrow-to-bf16 instead of quantise, no scale) -- one flag, two datapaths,
  // so --no-fuse-ffn A/Bs whichever one the loaded container actually uses.
  bool fuse_ffn_epilogue = true;
  double t_hostln = 0.0, t_hostsm = 0.0, t_hostgelu = 0.0;


  // Where the time goes. A single number for the whole encode says "slow";
  // this says which half to fix.
  double t_npu = 0.0;      // memcpy + sync + dispatch, i.e. everything a
                           // fused design would subsume
  double t_attn = 0.0;     // the per-head QK^T and A.V loops on the host
  // Split, because they are different problems: QK^T ends in a horizontal
  // reduction and holds Q in registers (compute), A.V accumulates per output
  // element and streams scores + V (memory). tasks/0086 widened A.V to 512
  // bits for exactly zero, which only makes sense if it is the memory half --
  // and that is a claim this counter can check instead of infer.
  double t_qk = 0.0, t_av = 0.0;
  int n_dispatch = 0;

  // Splitting t_npu further, because removing 21 MB of memcpy and vectorising
  // 13.8 M conversions bought only 9%: the cost is not where it was assumed to
  // be, and one aggregate number cannot say where it is instead (tasks/0024).
  double t_conv = 0.0;     // fp32 <-> bf16 both directions
  double t_in = 0.0;       // sync_to_device
  double t_disp = 0.0;     // kernel(...) + wait
  double t_out = 0.0;      // sync_from_device
  double t_bias = 0.0;     // reading the result buffer, adding bias

  void reset_timers() {
    // t_qk/t_av belong here too. Left out, they accumulated the warm-up encode
    // as well as the benched ones and summed to 1.5x their own parent t_attn --
    // a new counter that is not reset is a counter measuring a different window
    // from everything printed beside it (tasks/0086).
    t_qk = t_av = 0.0;
    t_npu = t_attn = 0.0;
    t_hostln = t_hostsm = t_hostgelu = 0.0;
    t_conv = t_in = t_disp = t_out = t_bias = 0.0;
    n_dispatch = 0;
  }

  // Move every weight onto the device once. Returns the bytes staged, so the
  // banner can state the cost of the trade rather than hiding it.
  size_t stage_all() {
    size_t bytes = 0;
    const bool i8 = qkv.info().a_elem_bytes == 1;
    auto one = [&](npu::Design &d, const std::string &name,
                   std::vector<size_t> &slots,
                   std::vector<const float *> &bias,
                   std::vector<const float *> *wsc = nullptr,
                   std::vector<const float *> *asm_ = nullptr) {
      // The design says what layout it needs; the .npue says what it holds.
      // Refuse unless both spoke AND they agree.
      //
      // tasks/0022 shipped pre-tiled weights into a row-major design: right
      // sizes, wrong order, rel_fro 1.186 -- "a buffer-size check catches a
      // wrong size, never a wrong layout". The layout hash that catches it has
      // been in the file since M4 and was never read on this side.
      const std::string &want = d.info().b_layout_hash;
      const std::string &got = model.info(name).layout_hash;
      if (want.empty())
        throw std::runtime_error(d.info().name + "/design.json has no "
                                 "b_layout_hash -- re-export with "
                                 "tools/export_xclbin.py");
      if (got.empty())
        throw std::runtime_error(name + ": .npue tensor carries no "
                                 "layout_hash -- repack with "
                                 "tools/pack_npue.py");
      if (want != got)
        throw std::runtime_error(
            name + ": layout mismatch -- design " + d.info().name +
            " wants " + want.substr(0, 16) + "..., file has " +
            got.substr(0, 16) + "... The bytes would be the right size and the "
            "wrong order.");
      auto w = model.raw(name);
      slots.push_back(d.stage(1, w.data, w.bytes));
      bias.push_back(model.raw(name + ".bias").as<float>());
      bytes += w.bytes;
      if (i8 && wsc) {
        // Refuse rather than dereference: an int8 DESIGN with a bf16
        // CONTAINER would otherwise fail here with a bare "no such tensor",
        // and the layout-hash check above has already told the user which
        // half is wrong -- so this only fires if a container carries tiled
        // int8 bytes without the scales that give them meaning.
        wsc->push_back(model.raw(name + ".wscale").as<float>());
        asm_->push_back(model.raw(name + ".asmooth").as<float>());
      }
    };
    for (int64_t L = 0; L < g_layers; ++L) {
      const std::string p = "layer." + std::to_string(L) + ".";
      one(qkv, p + "qkv", s_qkv, b_qkv, &ws_qkv, &as_qkv);
      one(attn_out, p + "attn_out", s_ao, b_ao, &ws_ao, &as_ao);
      one(ffn_up, p + "ffn_up", s_fu, b_fu, &ws_fu, &as_fu);
      one(ffn_down, p + "ffn_down", s_fd, b_fd, &ws_fd, &as_fd);
    }

    // gamma and beta share one buffer: a core tile has two input DMA channels
    // and the activations need one of them (tasks/0020).
    std::vector<float> gb(2 * g_hidden);
    auto ln_one = [&](const std::string &g, const std::string &b) {
      std::memcpy(gb.data(), model.raw(g).data, g_hidden * sizeof(float));
      std::memcpy(gb.data() + g_hidden, model.raw(b).data,
                  g_hidden * sizeof(float));
      s_ln.push_back(layernorm.stage(1, gb.data(), gb.size() * sizeof(float)));
      bytes += gb.size() * sizeof(float);
      h_gamma.push_back(model.raw(g).as<float>());
      h_beta.push_back(model.raw(b).as<float>());
    };
    if (host_ln) {
      // No device staging: only the host pointers and the site numbering.
      auto ln_host = [&](const std::string &g, const std::string &b) {
        s_ln.push_back(s_ln.size() + 1);
        h_gamma.push_back(model.raw(g).as<float>());
        h_beta.push_back(model.raw(b).as<float>());
      };
      ln_host("embeddings.ln.weight", "embeddings.ln.bias");
      for (int64_t L = 0; L < g_layers; ++L) {
        const std::string p = "layer." + std::to_string(L) + ".";
        ln_host(p + "ln1.weight", p + "ln1.bias");
        ln_host(p + "ln2.weight", p + "ln2.bias");
      }
      return bytes;
    }
    ln_one("embeddings.ln.weight", "embeddings.ln.bias");
    for (int64_t L = 0; L < g_layers; ++L) {
      const std::string p = "layer." + std::to_string(L) + ".";
      ln_one(p + "ln1.weight", p + "ln1.bias");
      ln_one(p + "ln2.weight", p + "ln2.bias");
    }
    return bytes;
  }

  // `lap` charges the elapsed time to a bucket and returns the new mark, so
  // each stage is attributed without a timer call being able to drift.
  double lap(double t0, double &bucket) {
    double t = now_s();
    bucket += t - t0;
    return t;
  }

  // bf16 in, bf16 out, one input buffer -- GELU and softmax.
  void eltwise(npu::Design &d, float *x, size_t n) {
    double t0 = now_s();
    par(n, [&](size_t lo, size_t hi) {
      bf16_fill(static_cast<uint16_t *>(d.host_ptr(0)) + lo, x + lo, hi - lo);
    });
    t0 = lap(t0, t_conv);
    d.sync_to_device(0);
    t0 = lap(t0, t_in);
    d.dispatch_only();
    t0 = lap(t0, t_disp);
    d.sync_from_device(1);
    t0 = lap(t0, t_out);
    par(n, [&](size_t lo, size_t hi) {
      bf16_read(x + lo, static_cast<const uint16_t *>(d.host_ptr(1)) + lo,
                hi - lo);
    });
    lap(t0, t_conv);
    ++n_dispatch;
  }

  // fp32 two-pass LayerNorm on the host, parallelized over rows: the same
  // two-pass mean/variance formula as the NPU kernel and the M3 oracle, in
  // fp32 throughout -- if anything MORE accurate than the bf16 round trip it
  // replaces. The golden check decides.
  void layer_norm_cpu(std::vector<float> &x, size_t site) {
    double t0 = now_s();
    const float *g = h_gamma[site], *b = h_beta[site];
    const int64_t n_rows = static_cast<int64_t>(x.size()) / g_hidden;
    pool->run([&](int w, int nw) {
      const int64_t chunk = (n_rows + nw - 1) / nw;
      const int64_t lo = std::min<int64_t>(n_rows, chunk * w);
      const int64_t hi = std::min<int64_t>(n_rows, lo + chunk);
      for (int64_t r = lo; r < hi; ++r) {
        float *row = x.data() + r * g_hidden;
#if defined(__AVX2__)
        __m256 s = _mm256_setzero_ps();
        for (int64_t j = 0; j < g_hidden; j += 8)
          s = _mm256_add_ps(s, _mm256_loadu_ps(row + j));
        const float mean = hsum256(s) / g_hidden;
        const __m256 mv = _mm256_set1_ps(mean);
        __m256 v = _mm256_setzero_ps();
        for (int64_t j = 0; j < g_hidden; j += 8) {
          __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
          v = _mm256_fmadd_ps(d, d, v);
        }
        const float var = hsum256(v) / g_hidden;
        const __m256 is = _mm256_set1_ps(1.0f / std::sqrt(var + 1e-12f));
        for (int64_t j = 0; j < g_hidden; j += 8) {
          __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
          __m256 y = _mm256_fmadd_ps(_mm256_mul_ps(d, is),
                                     _mm256_loadu_ps(g + j),
                                     _mm256_loadu_ps(b + j));
          _mm256_storeu_ps(row + j, y);
        }
#else
        double sm = 0.0;
        for (int64_t j = 0; j < g_hidden; ++j) sm += row[j];
        const float mean = static_cast<float>(sm / g_hidden);
        double sv = 0.0;
        for (int64_t j = 0; j < g_hidden; ++j) {
          const float d = row[j] - mean;
          sv += static_cast<double>(d) * d;
        }
        const float var = static_cast<float>(sv / g_hidden);
        const float is = 1.0f / std::sqrt(var + 1e-12f);
        for (int64_t j = 0; j < g_hidden; ++j)
          row[j] = (row[j] - mean) * is * g[j] + b[j];
#endif
      }
    });
    t_hostln += now_s() - t0;
  }

#if defined(__AVX2__)
  static inline __m256 exp2_avx2(__m256 x) {
    const __m256 c0 = _mm256_set1_ps(1.5483275463e-05f);
    const __m256 c1 = _mm256_set1_ps(1.5669833174e-04f);
    const __m256 c2 = _mm256_set1_ps(1.3331825236e-03f);
    const __m256 c3 = _mm256_set1_ps(9.6164605538e-03f);
    const __m256 c4 = _mm256_set1_ps(5.5504156855e-02f);
    const __m256 c5 = _mm256_set1_ps(2.4022684109e-01f);
    const __m256 c6 = _mm256_set1_ps(6.9314717694e-01f);
    const __m256 c7 = _mm256_set1_ps(9.9999998955e-01f);
    __m256i k = _mm256_cvttps_epi32(x);
    __m256 f = _mm256_sub_ps(x, _mm256_cvtepi32_ps(k));
    __m256 pl = _mm256_fmadd_ps(c0, f, c1);
    pl = _mm256_fmadd_ps(pl, f, c2);
    pl = _mm256_fmadd_ps(pl, f, c3);
    pl = _mm256_fmadd_ps(pl, f, c4);
    pl = _mm256_fmadd_ps(pl, f, c5);
    pl = _mm256_fmadd_ps(pl, f, c6);
    pl = _mm256_fmadd_ps(pl, f, c7);
    __m256i bits = _mm256_slli_epi32(
        _mm256_add_epi32(k, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(pl, _mm256_castsi256_ps(bits));
  }
#endif

  // fp32 softmax over rows of g_seq on the host. Same structure as the NPU
  // kernel (max-subtract, exp2 with the -120 argument floor, one reciprocal),
  // but fp32 end to end -- like host LayerNorm, it removes dispatches AND
  // beats the bf16 path on accuracy.
  // The padding mask, as an explicit pass. Only the NPU-softmax branch needs
  // this: softmax_cpu folds the same addition into its per-row prologue, where
  // the row is already in L1 and it costs nothing, while an aie softmax kernel
  // has no second operand to take it from.
  void add_additive_mask(std::vector<float> &scores) {
    const int64_t rows_per_seq = g_heads * g_seq;
    const int64_t n_rows = static_cast<int64_t>(scores.size()) / g_seq;
    pool->run([&](int w, int nw) {
      for (int64_t r = w; r < n_rows; r += nw) {
        float *row = scores.data() + r * g_seq;
        const float *mk = add_mask.data() + (r / rows_per_seq) * g_seq;
        for (int64_t j = 0; j < g_seq; ++j) row[j] += mk[j];
      }
    });
  }

  void softmax_cpu(std::vector<float> &scores) {
    double t0 = now_s();
    const int64_t n_rows = static_cast<int64_t>(scores.size()) / g_seq;
    pool->run([&](int w, int nw) {
      const int64_t chunk = (n_rows + nw - 1) / nw;
      const int64_t lo = std::min<int64_t>(n_rows, chunk * w);
      const int64_t hi = std::min<int64_t>(n_rows, lo + chunk);
      const int64_t rows_per_seq = g_heads * g_seq;
      for (int64_t r = lo; r < hi; ++r) {
        float *row = scores.data() + r * g_seq;
        // The additive padding mask, folded in here rather than in qk(): the
        // row is already resident, so this is free, and it leaves qk() as the
        // pure matmul an array kernel could run. Same single float addition
        // qk() used to do, so the result is unchanged to the bit.
        const float *mk = add_mask.data() + (r / rows_per_seq) * g_seq;
        for (int64_t j = 0; j < g_seq; ++j) row[j] += mk[j];
#if defined(__AVX2__)
        __m256 mx = _mm256_loadu_ps(row);
        for (int64_t j = 8; j < g_seq; j += 8)
          mx = _mm256_max_ps(mx, _mm256_loadu_ps(row + j));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(mx),
                               _mm256_extractf128_ps(mx, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
        const __m256 mv = _mm256_set1_ps(_mm_cvtss_f32(m4));
        const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
        const __m256 argfloor = _mm256_set1_ps(-120.0f);
        __m256 sum = _mm256_setzero_ps();
        for (int64_t j = 0; j < g_seq; j += 8) {
          __m256 a = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(row + j), mv),
                                   log2e);
          __m256 e = exp2_avx2(_mm256_max_ps(a, argfloor));
          _mm256_storeu_ps(row + j, e);
          sum = _mm256_add_ps(sum, e);
        }
        const __m256 inv = _mm256_set1_ps(1.0f / hsum256(sum));
        for (int64_t j = 0; j < g_seq; j += 8)
          _mm256_storeu_ps(row + j,
                           _mm256_mul_ps(_mm256_loadu_ps(row + j), inv));
#else
        float m = row[0];
        for (int64_t j = 1; j < g_seq; ++j) m = std::max(m, row[j]);
        float sum = 0.f;
        for (int64_t j = 0; j < g_seq; ++j) {
          row[j] = std::exp(row[j] - m);
          sum += row[j];
        }
        const float inv = 1.0f / sum;
        for (int64_t j = 0; j < g_seq; ++j) row[j] *= inv;
#endif
      }
    });
    t_hostsm += now_s() - t0;
  }

  void gelu_cpu(std::vector<float> &x) {
    double t0 = now_s();
    par(x.size(), [&](size_t lo, size_t hi) {
      size_t i = lo;
#if defined(__AVX2__)
      const __m256 vR = _mm256_set1_ps(4.0f);
      const __m256 vz = _mm256_setzero_ps();
      const __m256 sign = _mm256_set1_ps(-0.0f);
      const __m256 c0 = _mm256_set1_ps(-7.2340282171e-05f);
      const __m256 c1 = _mm256_set1_ps(1.8179518005e-03f);
      const __m256 c2 = _mm256_set1_ps(-1.7707383379e-02f);
      const __m256 c3 = _mm256_set1_ps(8.4577147641e-02f);
      const __m256 c4 = _mm256_set1_ps(-1.9228671834e-01f);
      const __m256 c5 = _mm256_set1_ps(9.8431124458e-02f);
      const __m256 c6 = _mm256_set1_ps(3.6137852062e-01f);
      const __m256 c7 = _mm256_set1_ps(-4.9454128936e-01f);
      const __m256 c8 = _mm256_set1_ps(-1.3007010117e-04f);
      for (; i + 8 <= hi; i += 8) {
        __m256 v = _mm256_loadu_ps(x.data() + i);
        __m256 u = _mm256_min_ps(_mm256_andnot_ps(sign, v), vR);
        __m256 pl = _mm256_fmadd_ps(c0, u, c1);
        pl = _mm256_fmadd_ps(pl, u, c2);
        pl = _mm256_fmadd_ps(pl, u, c3);
        pl = _mm256_fmadd_ps(pl, u, c4);
        pl = _mm256_fmadd_ps(pl, u, c5);
        pl = _mm256_fmadd_ps(pl, u, c6);
        pl = _mm256_fmadd_ps(pl, u, c7);
        pl = _mm256_fmadd_ps(pl, u, c8);
        _mm256_storeu_ps(x.data() + i,
                         _mm256_add_ps(_mm256_max_ps(v, vz), pl));
      }
#endif
      for (; i < hi; ++i) {
        const float v = x[i];
        const float u = std::min(std::fabs(v), 4.0f);
        float pl = -7.2340282171e-05f;
        pl = pl * u + 1.8179518005e-03f;
        pl = pl * u + -1.7707383379e-02f;
        pl = pl * u + 8.4577147641e-02f;
        pl = pl * u + -1.9228671834e-01f;
        pl = pl * u + 9.8431124458e-02f;
        pl = pl * u + 3.6137852062e-01f;
        pl = pl * u + -4.9454128936e-01f;
        pl = pl * u + -1.3007010117e-04f;
        x[i] = std::max(v, 0.0f) + pl;
      }
    });
    t_hostgelu += now_s() - t0;
  }

  // arch=2 gated FFN activation: [rows][2*inter] -> [rows][inter].
  //   out[r][j] = lo[r][j] * silu(hi[r][j])
  // lo = cols [0, inter) (fc11, the untouched up-path), hi = cols
  // [inter, 2*inter) (fc12, the SiLU gate) -- pinned by the container's
  // swiglu_halves == "fc11_up|fc12_gate", asserted once in apply_model_shape()
  // rather than trusted here.
  //
  // silu(x) = x / (1 + exp(-x)); exp(-x) = exp2(-x*log2e), reusing
  // exp2_avx2 instead of adding a second exponential. The argument floor
  // mirrors softmax_cpu's -120: when x (the gate, hi[j]) is large positive,
  // -x*log2e is large negative, and unfloored that corrupts exp2_avx2's
  // internal int32 conversion (cvttps2dq's "indefinite integer" case)
  // instead of cleanly underflowing toward 0 -- the same failure mode
  // softmax's own masked (very negative) rows hit without that floor.
  //
  // OUT OF PLACE, into a caller-supplied buffer -- NOT the in-place scheme
  // this function originally shipped with. The "safe in-place, compacting
  // forward" proof this comment used to carry was WRONG, and it was wrong in
  // exactly the way the task that wrote it demanded be checked for
  // ("VERIFY this claim numerically... rather than trusting the algebra") --
  // caught by that numerical check, on real hardware, tasks/0070:
  //   Row r WRITES [r*inter, (r+1)*inter) and READS [r*2*inter,(r+1)*2*inter).
  //   The original proof showed no row r' > r can have its READ range
  //   clobbered by row r's WRITE -- true, but it never checked r' < r. Row
  //   r's write range and row r' = floor(r/2)'s READ range overlap for
  //   EVERY r >= 1 (write=[r*inter,(r+1)*inter), read=[2r'*inter,(2r'+2)*inter),
  //   and r=2r' or r=2r'+1 both fall inside that read range by construction).
  //   Sequentially this is harmless (r' < r is always processed first in an
  //   ascending loop, so its read completes before r's write). Threaded, it
  //   is not: pool->run() hands CONTIGUOUS chunks to independent threads with
  //   no ordering between them, so whenever r and floor(r/2) land in
  //   different chunks (e.g. r'=63 in one thread's chunk, r=127 in another's,
  //   with no happens-before edge), row 127's write can race row 63's read.
  //   Measured effect: rows immediately after such a boundary came back
  //   catastrophically wrong (rel err up to 1.23, i.e. wrong sign / wrong
  //   magnitude, not bf16 noise) while every other row matched the oracle to
  //   ~1e-3 -- found via reference/encoder_nomic.py's own L0.fc11/fc12/gated
  //   taps, which isolated the corruption to exactly this function on a
  //   4-sentence batch after `--embed` (the golden gate never caught it
  //   because it tiles ONE 4-sentence batch 32x, so every "different" row is
  //   actually identical content and a wrong-row read is indistinguishable
  //   from a right-row read).
  void swiglu_cpu(const std::vector<float> &x, std::vector<float> &out) {
    double t0 = now_s();
    const int64_t inter = g_ffn;
    const int64_t n_rows = rows;
    pool->run([&](int w, int nw) {
      const int64_t chunk = (n_rows + nw - 1) / nw;
      const int64_t lo_r = std::min<int64_t>(n_rows, chunk * w);
      const int64_t hi_r = std::min<int64_t>(n_rows, lo_r + chunk);
      for (int64_t r = lo_r; r < hi_r; ++r) {
        const float *lo = x.data() + r * 2 * inter;
        const float *hi = lo + inter;
        float *dst = out.data() + r * inter;
        // arch=3 (tasks/0136): exact-erf GELU on the gate half, same halves
        // order. The SiLU arm below is byte-for-byte what arch=2 always ran.
        if (g_gated_act == GatedAct::GeluErf) {
          for (int64_t j = 0; j < inter; ++j)
            dst[j] = lo[j] * gelu_erf_exact(hi[j]);
          continue;
        }
        int64_t j = 0;
#if defined(__AVX2__)
        const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
        const __m256 argfloor = _mm256_set1_ps(-120.0f);
        const __m256 one = _mm256_set1_ps(1.0f);
        for (; j + 8 <= inter; j += 8) {
          __m256 xv = _mm256_loadu_ps(hi + j);
          __m256 a = _mm256_max_ps(
              _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), xv), log2e),
              argfloor);
          __m256 e = exp2_avx2(a);
          __m256 s = _mm256_div_ps(xv, _mm256_add_ps(one, e));
          __m256 loV = _mm256_loadu_ps(lo + j);
          _mm256_storeu_ps(dst + j, _mm256_mul_ps(loV, s));
        }
#endif
        for (; j < inter; ++j) {
          const float xv = hi[j];
          float a = -xv * 1.4426950408889634f;
          if (a < -120.0f) a = -120.0f;
          const float e = std::exp2(a);
          const float s = xv / (1.0f + e);
          dst[j] = lo[j] * s;
        }
      }
    });
    // Reuses gelu_cpu's bucket -- it is the same "FFN activation, on the
    // host, in place of an NPU dispatch" cost this timer already names.
    t_hostgelu += now_s() - t0;
  }

  void layer_norm(std::vector<float> &x, size_t slot) {
    if (host_ln) { layer_norm_cpu(x, slot - 1); return; }
    double t0 = now_s();
    layernorm.bind(1, slot);
    par(x.size(), [&](size_t lo, size_t hi) {
      bf16_fill(static_cast<uint16_t *>(layernorm.host_ptr(0)) + lo,
                x.data() + lo, hi - lo);
    });
    t0 = lap(t0, t_conv);
    layernorm.sync_to_device(0);
    t0 = lap(t0, t_in);
    layernorm.dispatch_only();
    t0 = lap(t0, t_disp);
    layernorm.sync_from_device(2);
    t0 = lap(t0, t_out);
    par(x.size(), [&](size_t lo, size_t hi) {
      bf16_read(x.data() + lo,
                static_cast<const uint16_t *>(layernorm.host_ptr(2)) + lo,
                hi - lo);
    });
    lap(t0, t_conv);
    ++n_dispatch;
  }

  // Per-row activation scales for the int8 path, [rows]. Filled by gemm()'s
  // quantisation pass and consumed by its dequantisation pass in the same
  // call, so it never has to be threaded anywhere.
  std::vector<float> a_scale;
  // ...except when the ffn_up epilogue is fused (T37), where one pass produces
  // BOTH this GEMM's dequantised result and the next one's quantised operand,
  // so the two sets of row scales are live at once.
  std::vector<float> a_scale_next;
  std::vector<float> inv_smooth_next;
  const float *inv_smooth_next_src = nullptr;
  // 1/asmooth for the GEMM currently being run, rebuilt only when the pointer
  // changes (i.e. per op, not per layer-iteration, and never per row).
  std::vector<float> inv_smooth;
  const float *inv_smooth_src = nullptr;

  // Set when gemm() should not write `out` at all, but instead run the
  // activation and quantise straight into the NEXT GEMM's operand.
  struct FusedNext {
    const float *asmooth;   // the next GEMM's per-input-channel divisor
    int8_t *dst;            // the next GEMM's device A slot
    float *scale;           // filled with the next GEMM's per-row scales
    bool gated;             // SwiGLU narrows 2*inter -> inter; GELU does not
  };

  // T37-BF16 (tasks/0108): as FusedNext, for the bf16/bfp16 datapath. No
  // quantisation on this path, so no smoothing divisor and no per-row scale
  // -- the only thing that survives is the narrow-to-bf16 write, straight
  // into the next GEMM's device A slot.
  struct FusedNextBf16 {
    uint16_t *dst;           // the next GEMM's device A slot (bf16)
    bool gated;              // SwiGLU/GeGLU narrows 2*inter -> inter
  };

  // nullptr for a bf16 container, where these vectors are empty and the int8
  // arms of gemm() never run.
  static const float *i8w(const std::vector<const float *> &v, int64_t L) {
    return v.empty() ? nullptr : v[static_cast<size_t>(L)];
  }

  // T37-BF16 (tasks/0108): as dequant_act_quant (T37, tasks/0082), for the
  // bf16/bfp16 datapath. STEP 1 of this task measured the three passes this
  // collapses -- ffn_up's C-readback+bias (into `up`), the activation in
  // place, and ffn_down's A-conversion (out of `up`) -- at 17.7-28.3% of
  // wall clock across the shipped catalogue; `up` is never materialised.
  // No quantisation step exists on this path, so unlike dequant_act_quant
  // there is no per-row scale and no smoothing divisor to carry.
  //
  // BIT-IDENTICAL to the unfused path BY CONSTRUCTION, not by review: the
  // C-readback+bias arm below is copied verbatim from gemm()'s own unfused
  // bf16-C and fp32-C branches (the ones just below this function), the
  // activation arms are copied verbatim from the int8 fused epilogue's own
  // lambda (itself verified bit-identical to gelu_cpu/swiglu_cpu, tasks/0082
  // sec 2), and the final narrowing calls the SAME bf16_fill() the unfused
  // A-conversion calls. Calling bf16_fill per row rather than once over the
  // whole buffer changes nothing: it is round-to-nearest-even, purely
  // elementwise -- every output bit depends only on its own input float, not
  // on its neighbours or its position in the array.
  void dequant_act_bf16(const void *c, size_t c_bytes, int64_t N,
                        int64_t out_n, bool gated, const float *bias,
                        uint16_t *dst) {
    par_rows(rows, [&](int64_t r0, int64_t r1) {
      std::vector<float> row(static_cast<size_t>(N));
      for (int64_t r = r0; r < r1; ++r) {
        float *v = row.data();
        int64_t j = 0;
#if defined(__AVX2__)
        if (c_bytes == 2) {
          const uint16_t *cr = static_cast<const uint16_t *>(c) + r * N;
          for (; j + 16 <= N; j += 16) {
            __m256i raw = _mm256_stream_load_si256(
                reinterpret_cast<const __m256i *>(cr + j));
            __m256i lo = _mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm256_castsi256_si128(raw)), 16);
            __m256i hi = _mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm256_extracti128_si256(raw, 1)), 16);
            _mm256_storeu_ps(v + j, _mm256_add_ps(_mm256_castsi256_ps(lo),
                                                  _mm256_loadu_ps(bias + j)));
            _mm256_storeu_ps(v + j + 8, _mm256_add_ps(_mm256_castsi256_ps(hi),
                                                  _mm256_loadu_ps(bias + j + 8)));
          }
        } else {
          const float *cr = static_cast<const float *>(c) + r * N;
          for (; j + 8 <= N; j += 8) {
            __m256i raw = _mm256_stream_load_si256(
                reinterpret_cast<const __m256i *>(cr + j));
            _mm256_storeu_ps(v + j, _mm256_add_ps(_mm256_castsi256_ps(raw),
                                                  _mm256_loadu_ps(bias + j)));
          }
        }
#endif
        for (; j < N; ++j) {
          const float cf = c_bytes == 2
              ? from_bf16(static_cast<const uint16_t *>(c)[r * N + j])
              : static_cast<const float *>(c)[r * N + j];
          v[j] = cf + bias[j];
        }
        // Activation, in place -- COPIED VERBATIM from the int8 fused
        // epilogue's own lambda just below, not re-derived.
        if (!gated) {
          int64_t k = 0;
#if defined(__AVX2__)
          for (; k + 8 <= N; k += 8)
            _mm256_storeu_ps(v + k, gelu8(_mm256_loadu_ps(v + k)));
#endif
          for (; k < N; ++k) v[k] = gelu8(v[k]);
        } else if (g_gated_act == GatedAct::GeluErf) {
          // arch=3 (tasks/0136): exact-erf GELU on the gate half. The SiLU
          // arm below is byte-for-byte what arch=2 always ran.
          const int64_t inter = N / 2;
          const float *hi = v + inter;
          for (int64_t k = 0; k < inter; ++k)
            v[k] = v[k] * gelu_erf_exact(hi[k]);
        } else {
          const int64_t inter = N / 2;
          const float *hi = v + inter;
          int64_t k = 0;
#if defined(__AVX2__)
          const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
          const __m256 argfloor = _mm256_set1_ps(-120.0f);
          const __m256 one = _mm256_set1_ps(1.0f);
          for (; k + 8 <= inter; k += 8) {
            __m256 xv = _mm256_loadu_ps(hi + k);
            __m256 a = _mm256_max_ps(
                _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), xv), log2e),
                argfloor);
            __m256 e = exp2_avx2(a);
            __m256 sg = _mm256_div_ps(xv, _mm256_add_ps(one, e));
            _mm256_storeu_ps(v + k, _mm256_mul_ps(_mm256_loadu_ps(v + k), sg));
          }
#endif
          for (; k < inter; ++k) {
            const float xv = hi[k];
            float a = -xv * 1.4426950408889634f;
            if (a < -120.0f) a = -120.0f;
            v[k] = v[k] * (xv / (1.0f + std::exp2f(a)));
          }
        }
        bf16_fill(dst + r * out_n, v, static_cast<size_t>(out_n));
      }
    });
  }

  void gemm(npu::Design &d, size_t islot, const std::vector<float> &a,
            size_t wslot, const float *bias, std::vector<float> &out,
            int64_t N, const float *wscale = nullptr,
            const float *asmooth = nullptr,
            FusedNext *fuse = nullptr, bool a_ready = false,
            FusedNextBf16 *fuse_bf16 = nullptr) {
    const bool i8 = d.info().a_elem_bytes == 1;
    if (i8 && (wscale == nullptr || asmooth == nullptr))
      throw std::runtime_error(
          "int8 design but this encoder has no quantisation scales -- the "
          "container is bf16, or a pipeline lane was constructed without "
          "copying ws_*/as_* from lane 0");
    double t0 = now_s();
    if (i8 && a_ready) {
      // A was written straight into the device slot by the PREVIOUS gemm's
      // fused epilogue (T37), and a_scale already holds its row scales. There
      // is nothing to convert: the whole point is that this tensor is never
      // materialised in fp32 at all.
    } else if (i8) {
      // QUANTISE A, per row, with the SmoothQuant divisor folded into the same
      // pass (tasks/0078). Two reads of each row would cost a second sweep of
      // 12.6 MB at batch 128; one pass computes max|x/s| and the second writes
      // the rounded quotient.
      //
      // The smoothing divisor is NOT folded into LayerNorm: BERT is post-LN,
      // so each LayerNorm's output feeds the residual as well as this GEMM,
      // and scaling gamma/beta would scale the residual too (tasks/0078 4a).
      const int64_t K = static_cast<int64_t>(a.size()) / rows;
      a_scale.resize(static_cast<size_t>(rows));
      // RECIPROCATE THE SMOOTHING VECTOR ONCE PER GEMM, not twice per element.
      // The first version divided by asmooth[j] in both the max pass and the
      // quantise pass -- two divisions per element, and `conv` went 36.2 ms
      // (bf16) to 109.9 ms, eating 74 ms of the 126 ms the array had saved.
      // K floats of setup replaces 2*rows*K divisions.
      if (inv_smooth.size() != static_cast<size_t>(K) ||
          inv_smooth_src != asmooth) {
        inv_smooth.resize(static_cast<size_t>(K));
        for (int64_t j = 0; j < K; ++j) inv_smooth[j] = 1.0f / asmooth[j];
        inv_smooth_src = asmooth;
      }
      const float *ias = inv_smooth.data();
      auto *abuf = static_cast<int8_t *>(d.slot_ptr(0, slot_a));
      quantise_a_int8(a.data(), rows, K, ias, abuf, a_scale.data(),
                      [&](int64_t n, auto f) { par_rows(n, f); });
      t0 = lap(t0, t_conv);
    } else if (a_ready) {
      // T37-BF16 (tasks/0108): A was written straight into the device slot,
      // already narrowed to bf16, by the PREVIOUS gemm's fused epilogue --
      // the bf16 analogue of the i8-and-a_ready arm above. Nothing to do.
    } else {
    auto *abuf = static_cast<uint16_t *>(d.slot_ptr(0, slot_a));
    par(a.size(), [&](size_t lo, size_t hi) {
      bf16_fill(abuf + lo, a.data() + lo, hi - lo);
    });
    t0 = lap(t0, t_conv);
    }
    // NPUE-M9 (tasks/0045): with --c-bf16 the design narrows C on the core
    // after the fp32 K reduction, so the drain moves half the bytes. The size
    // comes from the design, never from an assumption about the datatype.
    //
    // The pointer survives the unlock -- it is THIS pipeline's own bo; the
    // other pipeline binds its own slots and never touches this memory. That
    // was true before T61-1 and it is what makes the narrow lock legal.
    const float *c = npu_dispatch(
        d, npu_mu, unified, islot, slot_a, wslot, slot_c, a.size(),
        static_cast<size_t>(rows) * N * d.info().c_elem_bytes,
        t0, t_in, t_disp, t_out);
    // The bias add reads the result buffer directly. It used to be a memcpy
    // out followed by a second pass over the same 21 MB; this is one pass.
    //
    // STREAMING loads in both arms: the C buffer is an XRT host bo, and
    // ordinary loads from it measured ~80 ms per encode (~2 GB/s) -- the
    // signature of uncached/write-combined memory, where each load stalls the
    // core. movntdqa reads a whole WC line per transaction. Alignment holds:
    // the bo map is page-aligned and N is a multiple of 16.
    if (i8 && fuse) {
      // FUSED EPILOGUE (T37): dequantise, apply GELU, and quantise into the
      // next GEMM's operand in ONE pass over the widest tensor in the model.
      // `out` is deliberately never written -- see dequant_gelu_quant.
      const int64_t Kn = fuse->gated ? N / 2 : N;
      if (inv_smooth_next.size() != static_cast<size_t>(Kn) ||
          inv_smooth_next_src != fuse->asmooth) {
        inv_smooth_next.resize(static_cast<size_t>(Kn));
        for (int64_t j = 0; j < Kn; ++j)
          inv_smooth_next[j] = 1.0f / fuse->asmooth[j];
        inv_smooth_next_src = fuse->asmooth;
      }
      const int64_t out_n = fuse->gated ? N / 2 : N;
      dequant_act_quant(
          c, d.info().c_elem_bytes, rows, N, out_n,
          [gated = fuse->gated](float *v, int64_t n) {
            if (!gated) {
              int64_t j = 0;
#if defined(__AVX2__)
              for (; j + 8 <= n; j += 8)
                _mm256_storeu_ps(v + j, gelu8(_mm256_loadu_ps(v + j)));
#endif
              for (; j < n; ++j) v[j] = gelu8(v[j]);
              return;
            }
            // arch=3 (tasks/0136): no int8 gte container exists yet
            // (pack_npue refuses --int8 for arch=3), but if one arrives this
            // arm must not silently run SiLU over a GELU model.
            if (g_gated_act == GatedAct::GeluErf) {
              const int64_t inter = n / 2;
              const float *hi = v + inter;
              for (int64_t j = 0; j < inter; ++j)
                v[j] = v[j] * gelu_erf_exact(hi[j]);
              return;
            }
            // SwiGLU, narrowing 2*inter -> inter in place. Identical
            // intrinsics to swiglu_cpu, including its -120 argument floor.
            const int64_t inter = n / 2;
            const float *hi = v + inter;
            int64_t j = 0;
#if defined(__AVX2__)
            const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
            const __m256 argfloor = _mm256_set1_ps(-120.0f);
            const __m256 one = _mm256_set1_ps(1.0f);
            for (; j + 8 <= inter; j += 8) {
              __m256 xv = _mm256_loadu_ps(hi + j);
              __m256 a = _mm256_max_ps(
                  _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), xv), log2e),
                  argfloor);
              __m256 e = Encoder::exp2_avx2(a);
              __m256 sg = _mm256_div_ps(xv, _mm256_add_ps(one, e));
              _mm256_storeu_ps(v + j, _mm256_mul_ps(_mm256_loadu_ps(v + j), sg));
            }
#endif
            for (; j < inter; ++j) {
              const float xv = hi[j];
              float a = -xv * 1.4426950408889634f;
              if (a < -120.0f) a = -120.0f;
              v[j] = v[j] * (xv / (1.0f + std::exp2f(a)));
            }
          },
          a_scale.data(), wscale, bias, inv_smooth_next.data(), fuse->dst,
          fuse->scale, [&](int64_t n, auto f) { par_rows(n, f); });
    } else if (i8) {
      // DEQUANTISE: y = int32_acc * sa[row] * wscale[col] + bias[col], folded
      // into the pass that already reads C and adds the bias -- one extra
      // multiply per output element, no extra sweep of the 679 MB tasks/0044
      // measured this readback at. The helper picks the transport width from
      // the design, so a narrowed-C set (tasks/0080) needs nothing here.
      dequantise_c(c, d.info().c_elem_bytes, rows, N, a_scale.data(), wscale,
                   bias, out.data(),
                   [&](int64_t n, auto f) { par_rows(n, f); }, sim_c_bf16);
    } else if (fuse_bf16) {
      // T37-BF16 (tasks/0108): `out` is deliberately never written -- see
      // dequant_act_bf16 above.
      const int64_t out_n = fuse_bf16->gated ? N / 2 : N;
      dequant_act_bf16(c, d.info().c_elem_bytes, N, out_n, fuse_bf16->gated,
                       bias, fuse_bf16->dst);
    } else if (d.info().c_elem_bytes == 2) {
      const uint16_t *cb16 = reinterpret_cast<const uint16_t *>(c);
      par(size_t(rows), [&](size_t r0, size_t r1) {
        for (size_t r = r0; r < r1; ++r) {
          const uint16_t *cr = cb16 + r * N;
          float *o = out.data() + r * N;
          int64_t j = 0;
#if defined(__AVX2__)
          // One 32-byte streaming load carries 16 bf16, against 8 fp32 --
          // which is the whole point: same instruction count, half the traffic.
          for (; j + 16 <= N; j += 16) {
            __m256i raw = _mm256_stream_load_si256(
                reinterpret_cast<const __m256i *>(cr + j));
            __m256i lo = _mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm256_castsi256_si128(raw)), 16);
            __m256i hi = _mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm256_extracti128_si256(raw, 1)), 16);
            _mm256_storeu_ps(o + j,
                             _mm256_add_ps(_mm256_castsi256_ps(lo),
                                           _mm256_loadu_ps(bias + j)));
            _mm256_storeu_ps(o + j + 8,
                             _mm256_add_ps(_mm256_castsi256_ps(hi),
                                           _mm256_loadu_ps(bias + j + 8)));
          }
#endif
          for (; j < N; ++j) o[j] = from_bf16(cr[j]) + bias[j];
        }
      });
    } else {
      par(size_t(rows), [&](size_t r0, size_t r1) {
        for (size_t r = r0; r < r1; ++r) {
          const float *cr = c + r * N;
          float *o = out.data() + r * N;
          int64_t j = 0;
#if defined(__AVX2__)
          for (; j + 8 <= N; j += 8) {
            __m256i raw = _mm256_stream_load_si256(
                reinterpret_cast<const __m256i *>(cr + j));
            _mm256_storeu_ps(o + j, _mm256_add_ps(_mm256_castsi256_ps(raw),
                                                  _mm256_loadu_ps(bias + j)));
          }
#endif
          for (; j < N; ++j) o[j] = cr[j] + bias[j];
        }
      });
    }
    lap(t0, t_bias);
    ++n_dispatch;
  }

  // THE OTHER MULTI-PASS CHAIN (T37, tasks/0082 section 5).
  //
  //   add_into(x, y)      read y, read residual, write x
  //   layer_norm(x)       read x, write x
  //   memcpy(residual, x) read x, write residual
  //   quantise for next   read x, write int8
  //
  // Eight streaming passes over a rows x hidden tensor, TWICE per layer --
  // 33.5 MB at bge-large's batch 128. Every one of them touches the same row,
  // and a row is 4 KB, so all four kernels can share one L1-resident copy:
  // read y and residual once, write x, residual and the int8 operand once.
  //
  // `dst` may be null (the last layer's LN feeds pooling, not a GEMM), in
  // which case this is add + norm + residual with no quantisation.
  //
  // BIT-IDENTICAL to the four kernels it replaces, and that is not automatic:
  // it holds only because the intrinsics and the accumulation ORDER match
  // theirs exactly. tasks/0082 measured a scalar rewrite of the same algebra
  // landing on a different number, because fmadd rounds once where a*b+c
  // rounds twice.
  void add_norm_quant(std::vector<float> &x, const std::vector<float> &y,
                      size_t site, const float *ias_next, int8_t *dst,
                      float *scale_next) {
    double t0 = now_s();
    const float *g = h_gamma[site], *b = h_beta[site];
    const int64_t H = g_hidden;
    par_rows(rows, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        float *row = x.data() + r * H;
        const float *yr = y.data() + r * H;
        float *res = residual.data() + r * H;
        int64_t j = 0;
#if defined(__AVX2__)
        for (; j + 8 <= H; j += 8)                       // == add_into
          _mm256_storeu_ps(row + j, _mm256_add_ps(_mm256_loadu_ps(yr + j),
                                                  _mm256_loadu_ps(res + j)));
#endif
        for (; j < H; ++j) row[j] = yr[j] + res[j];
#if defined(__AVX2__)
        __m256 s = _mm256_setzero_ps();                  // == layer_norm_cpu
        for (j = 0; j + 8 <= H; j += 8)
          s = _mm256_add_ps(s, _mm256_loadu_ps(row + j));
        const float mean = hsum256(s) / H;
        const __m256 mv = _mm256_set1_ps(mean);
        __m256 v = _mm256_setzero_ps();
        for (j = 0; j + 8 <= H; j += 8) {
          __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
          v = _mm256_fmadd_ps(d, d, v);
        }
        const float var = hsum256(v) / H;
        const __m256 is = _mm256_set1_ps(1.0f / std::sqrt(var + 1e-12f));
        for (j = 0; j + 8 <= H; j += 8) {
          __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
          __m256 yv = _mm256_fmadd_ps(_mm256_mul_ps(d, is),
                                      _mm256_loadu_ps(g + j),
                                      _mm256_loadu_ps(b + j));
          _mm256_storeu_ps(row + j, yv);
          _mm256_storeu_ps(res + j, yv);                 // == memcpy residual
        }
#else
        double sm = 0.0;
        for (j = 0; j < H; ++j) sm += row[j];
        const float mean = static_cast<float>(sm / H);
        double vs = 0.0;
        for (j = 0; j < H; ++j) vs += double(row[j] - mean) * (row[j] - mean);
        const float is = 1.0f / std::sqrt(static_cast<float>(vs / H) + 1e-12f);
        for (j = 0; j < H; ++j) {
          row[j] = (row[j] - mean) * is * g[j] + b[j];
          res[j] = row[j];
        }
#endif
        if (!dst) continue;
        float mx = 0.f;                                  // == quantise_a_int8
        j = 0;
#if defined(__AVX2__)
        {
          const __m256 absmask =
              _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
          __m256 acc = _mm256_setzero_ps();
          for (; j + 8 <= H; j += 8)
            acc = _mm256_max_ps(acc, _mm256_and_ps(
                _mm256_mul_ps(_mm256_loadu_ps(row + j),
                              _mm256_loadu_ps(ias_next + j)), absmask));
          __m128 h = _mm_max_ps(_mm256_castps256_ps128(acc),
                                _mm256_extractf128_ps(acc, 1));
          h = _mm_max_ps(h, _mm_movehl_ps(h, h));
          h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 1));
          mx = _mm_cvtss_f32(h);
        }
#endif
        for (; j < H; ++j) {
          const float a = std::fabs(row[j] * ias_next[j]);
          if (a > mx) mx = a;
        }
        const float sc = mx > 0.f ? mx / 127.0f : 1.0f;
        scale_next[static_cast<size_t>(r)] = sc;
        const float inv = 1.0f / sc;
        int8_t *q = dst + r * H;
        j = 0;
#if defined(__AVX2__)
        {
          const __m256 invv = _mm256_set1_ps(inv);
          const __m256 vhi = _mm256_set1_ps(127.0f);
          const __m256 vlo = _mm256_set1_ps(-127.0f);
          for (; j + 8 <= H; j += 8) {
            __m256 t = _mm256_mul_ps(
                _mm256_mul_ps(_mm256_loadu_ps(row + j),
                              _mm256_loadu_ps(ias_next + j)), invv);
            t = _mm256_min_ps(_mm256_max_ps(t, vlo), vhi);
            __m256i i32 = _mm256_cvtps_epi32(t);
            __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(i32),
                                          _mm256_extracti128_si256(i32, 1));
            _mm_storel_epi64(reinterpret_cast<__m128i *>(q + j),
                             _mm_packs_epi16(p16, p16));
          }
        }
#endif
        for (; j < H; ++j) {
          float t = std::nearbyintf(row[j] * ias_next[j] * inv);
          if (t > 127.f) t = 127.f;
          if (t < -127.f) t = -127.f;
          q[j] = static_cast<int8_t>(t);
        }
      }
    });
    t_hostln += now_s() - t0;
  }

  // T37-BF16 (tasks/0108): as add_norm_quant, for the bf16/bfp16 datapath --
  // add + LayerNorm + residual copy + the NEXT gemm's A-conversion, one
  // L1-resident pass instead of the unfused chain's four streaming ones
  // (add_into, layer_norm_cpu, memcpy(residual), and the next gemm()'s own
  // bf16_fill). No quantisation on this path, so this is a strict subset of
  // add_norm_quant's work -- the add/LN/residual arithmetic below is copied
  // verbatim from it, with the int8 absmax+quantise tail replaced by a
  // single bf16_fill call. `dst` may be null (the last layer's LN feeds
  // pooling, not a GEMM), in which case this is add + norm + residual only,
  // exactly like the unfused path's own unconditional residual write.
  void add_norm_bf16(std::vector<float> &x, const std::vector<float> &y,
                     size_t site, uint16_t *dst) {
    double t0 = now_s();
    const float *g = h_gamma[site], *b = h_beta[site];
    const int64_t H = g_hidden;
    par_rows(rows, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        float *row = x.data() + r * H;
        const float *yr = y.data() + r * H;
        float *res = residual.data() + r * H;
        int64_t j = 0;
#if defined(__AVX2__)
        for (; j + 8 <= H; j += 8)                       // == add_into
          _mm256_storeu_ps(row + j, _mm256_add_ps(_mm256_loadu_ps(yr + j),
                                                  _mm256_loadu_ps(res + j)));
#endif
        for (; j < H; ++j) row[j] = yr[j] + res[j];
#if defined(__AVX2__)
        __m256 s = _mm256_setzero_ps();                  // == layer_norm_cpu
        for (j = 0; j + 8 <= H; j += 8)
          s = _mm256_add_ps(s, _mm256_loadu_ps(row + j));
        const float mean = hsum256(s) / H;
        const __m256 mv = _mm256_set1_ps(mean);
        __m256 v = _mm256_setzero_ps();
        for (j = 0; j + 8 <= H; j += 8) {
          __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
          v = _mm256_fmadd_ps(d, d, v);
        }
        const float var = hsum256(v) / H;
        const __m256 is = _mm256_set1_ps(1.0f / std::sqrt(var + 1e-12f));
        for (j = 0; j + 8 <= H; j += 8) {
          __m256 d = _mm256_sub_ps(_mm256_loadu_ps(row + j), mv);
          __m256 yv = _mm256_fmadd_ps(_mm256_mul_ps(d, is),
                                      _mm256_loadu_ps(g + j),
                                      _mm256_loadu_ps(b + j));
          _mm256_storeu_ps(row + j, yv);
          _mm256_storeu_ps(res + j, yv);                 // == memcpy residual
        }
#else
        double sm = 0.0;
        for (j = 0; j < H; ++j) sm += row[j];
        const float mean = static_cast<float>(sm / H);
        double vs = 0.0;
        for (j = 0; j < H; ++j) vs += double(row[j] - mean) * (row[j] - mean);
        const float is = 1.0f / std::sqrt(static_cast<float>(vs / H) + 1e-12f);
        for (j = 0; j < H; ++j) {
          row[j] = (row[j] - mean) * is * g[j] + b[j];
          res[j] = row[j];
        }
#endif
        if (dst) bf16_fill(dst + r * H, row, static_cast<size_t>(H));
      }
    });
    t_hostln += now_s() - t0;
  }

  // x += y, elementwise. The residual adds move 12.6 MB per layer at batch 128.
  void add_into(std::vector<float> &x, const std::vector<float> &y) {
    par(x.size(), [&](size_t lo, size_t hi) {
      size_t i = lo;
#if defined(__AVX2__)
      for (; i + 8 <= hi; i += 8)
        _mm256_storeu_ps(x.data() + i,
                         _mm256_add_ps(_mm256_loadu_ps(y.data() + i),
                                       _mm256_loadu_ps(residual.data() + i)));
#endif
      for (; i < hi; ++i) x[i] = y[i] + residual[i];
    });
  }

  // scores[b,h,i,j] = dot(Q[b,i,h], K[b,j,h]) + mask[b,j]
  // scores[b,h,i,j] = Q[b,i,h] . K[b,j,h]. NO mask: this is the operation an
  // array kernel would perform, and the mask is a property of the batch rather
  // than of the matmul. add_additive_mask() below applies it.
  // NV is head_dim/8 as a COMPILE-TIME constant where we have one, so the
  // inner loop unrolls and qv[]/acc[] stay in registers. NV == 0 keeps the
  // fully generic path for a width we have not met yet.
  template <int NV>
  void qk_impl(const std::vector<float> &qkvbuf, std::vector<float> &scores) {
    const int64_t pairs = batch * g_heads;
    // __restrict, because these are members now: the compiler could prove two
    // fresh local allocations did not overlap and cannot prove it for two
    // fields of the same object, and without the proof every store to dst[j]
    // re-issues the loads. Measured at 2x on this loop.
    const float *__restrict qkv_p = qkvbuf.data();
    float *__restrict sc_p = scores.data();
    pool->run([&](int w, int nw) {
      for (int64_t p = w; p < pairs; p += nw) {
        const int64_t b = p / g_heads, h = p % g_heads;
        for (int64_t i = 0; i < g_seq; ++i) {
          const float *q = &qkv_p[(b * g_seq + i) * 3 * g_hidden + h * g_head_dim];
          float *dst = &sc_p[(p * g_seq + i) * g_seq];
          // head_dim / 8 vectors, held across the j loop. head_dim is 32 for
          // MiniLM and bge-small and 64 for bge-large; kMaxHeadVecs bounds the
          // stack array and apply_model_shape() refuses anything larger.
#if defined(__AVX512F__)
          // QK^T AT 512 BITS. Unlike A.V (which gained nothing, tasks/0086),
          // this one pays -- and not because the arithmetic is wider. The win
          // is that `_mm512_reduce_add_ps` replaces `hsum256`'s four-
          // instruction shuffle chain, and QK^T does one reduction per (i, j)
          // pair. Microbenchmarked at bge-large's geometry: 1.33x on the inner
          // loop, against 1.00x for the same widening applied to A.V.
          //
          // NOT BIT-IDENTICAL, and it cannot be: summing head_dim floats as
          // lanes of 16 associates differently from lanes of 8. That is a
          // legitimate reassociation, not an error, but it means the byte
          // comparison the fusions used does not apply here -- this is gated on
          // 1-cos instead.
          //
          // head_dim is a multiple of 8 but not necessarily of 16, so the
          // 512-bit part takes what it can and a 256-bit tail finishes.
          const int64_t nv = NV ? NV : g_head_dim / 8;
          const int64_t nz = nv / 2;
          const bool zt = (nv & 1) != 0;
          __m512 zq[(NV ? NV : kMaxHeadVecs) / 2 + 1];
          __m256 yq;
          for (int64_t v = 0; v < nz; ++v) zq[v] = _mm512_loadu_ps(q + v * 16);
          if (zt) yq = _mm256_loadu_ps(q + nz * 16);
#else
          __m256 qv[NV ? NV : kMaxHeadVecs];
          const int64_t nv = NV ? NV : g_head_dim / 8;
          for (int64_t v = 0; v < nv; ++v) qv[v] = _mm256_loadu_ps(q + v * 8);
#endif
          for (int64_t j = 0; j < g_seq; ++j) {
            const float *k = &qkv_p[(b * g_seq + j) * 3 * g_hidden + g_hidden +
                                    h * g_head_dim];
#if defined(__AVX512F__)
            __m512 zs = _mm512_mul_ps(zq[0], _mm512_loadu_ps(k));
            for (int64_t v = 1; v < nz; ++v)
              zs = _mm512_fmadd_ps(zq[v], _mm512_loadu_ps(k + v * 16), zs);
            float acc = _mm512_reduce_add_ps(zs);
            if (zt) acc += hsum256(_mm256_mul_ps(yq,
                                                 _mm256_loadu_ps(k + nz * 16)));
            dst[j] = acc;
#elif defined(__AVX2__)
            // Accumulate in the same order the unrolled version did, so the
            // floating-point result is unchanged for head_dim 32.
            __m256 s = _mm256_mul_ps(qv[0], _mm256_loadu_ps(k));
            for (int64_t v = 1; v < nv; ++v)
              s = _mm256_fmadd_ps(qv[v], _mm256_loadu_ps(k + v * 8), s);
            dst[j] = hsum256(s);
#else
            float s = 0.f;
            for (int64_t d = 0; d < g_head_dim; ++d) s += q[d] * k[d];
            dst[j] = s;
#endif
          }
        }
      }
    });
  }

  // Dispatch on the width the container reported. head_dim 32 is MiniLM and
  // bge-small, 64 is bge-large; anything else still works, just generically.
  void qk(const std::vector<float> &qkvbuf, std::vector<float> &scores) {
    switch (g_head_dim) {
      case 32: qk_impl<4>(qkvbuf, scores); break;
      case 64: qk_impl<8>(qkvbuf, scores); break;
      default: qk_impl<0>(qkvbuf, scores); break;
    }
  }

  // ctx[b,i,h] = sum_j scores[b,h,i,j] * V[b,j,h]
  template <int NV>
  void av_impl(const std::vector<float> &scores,
               const std::vector<float> &qkvbuf, std::vector<float> &ctx) {
    const int64_t pairs = batch * g_heads;
    const float *__restrict sc_p = scores.data();
    const float *__restrict qkv_p = qkvbuf.data();
    float *__restrict ctx_p = ctx.data();
    pool->run([&](int w, int nw) {
      for (int64_t p = w; p < pairs; p += nw) {
        const int64_t b = p / g_heads, h = p % g_heads;
        for (int64_t i = 0; i < g_seq; ++i) {
          const float *a = &sc_p[(p * g_seq + i) * g_seq];
          float *o = &ctx_p[(b * g_seq + i) * g_hidden + h * g_head_dim];
#if defined(__AVX2__)
#if defined(__AVX512F__)
          // A.V AT 512 BITS IS BIT-IDENTICAL TO THE 256-BIT FORM, and that is
          // not luck -- it is why this half was done first (tasks/0086).
          //
          // Each accumulator lane owns ONE output element and sums over j in
          // the same order either way; widening only changes how many lanes
          // ride in a register, never which numbers are added or when. So the
          // byte-comparison harness applies here exactly as it did to the
          // fusions.
          //
          // QK^T is the opposite case and is NOT converted: it ends in a
          // horizontal reduction (`hsum256`), and reducing head_dim floats as
          // 2 lanes of 16 sums them in a different ORDER than 4 lanes of 8.
          // That is a reassociation, so it would change the result -- small,
          // legitimate, and no longer checkable by byte comparison. Left for a
          // measurement that uses the 1-cos gate instead.
          //
          // head_dim is a multiple of 8 (apply_model_shape refuses otherwise) but
          // not necessarily of 16, so the 512-bit path takes the multiple-of-16
          // part and a 256-bit tail finishes it.
          const int64_t nv = NV ? NV : g_head_dim / 8;
          const int64_t nz = nv / 2;                  // 512-bit accumulators
          __m512 zacc[(NV ? NV : kMaxHeadVecs) / 2 + 1];
          __m256 yacc;
          for (int64_t v = 0; v < nz; ++v) zacc[v] = _mm512_setzero_ps();
          const bool tail = (nv & 1) != 0;
          if (tail) yacc = _mm256_setzero_ps();
          for (int64_t j = 0; j < g_seq; ++j) {
            const float *v = &qkv_p[(b * g_seq + j) * 3 * g_hidden +
                                    2 * g_hidden + h * g_head_dim];
            const __m512 zaj = _mm512_set1_ps(a[j]);
            for (int64_t k = 0; k < nz; ++k)
              zacc[k] = _mm512_fmadd_ps(zaj, _mm512_loadu_ps(v + k * 16),
                                        zacc[k]);
            if (tail)
              yacc = _mm256_fmadd_ps(_mm256_set1_ps(a[j]),
                                     _mm256_loadu_ps(v + nz * 16), yacc);
          }
          for (int64_t v = 0; v < nz; ++v)
            _mm512_storeu_ps(o + v * 16, zacc[v]);
          if (tail) _mm256_storeu_ps(o + nz * 16, yacc);
#else
          __m256 acc[NV ? NV : kMaxHeadVecs];
          const int64_t nv = NV ? NV : g_head_dim / 8;
          for (int64_t v = 0; v < nv; ++v) acc[v] = _mm256_setzero_ps();
          for (int64_t j = 0; j < g_seq; ++j) {
            const float *v = &qkv_p[(b * g_seq + j) * 3 * g_hidden +
                                    2 * g_hidden + h * g_head_dim];
            const __m256 aj = _mm256_set1_ps(a[j]);
            for (int64_t k = 0; k < nv; ++k)
              acc[k] = _mm256_fmadd_ps(aj, _mm256_loadu_ps(v + k * 8), acc[k]);
          }
          for (int64_t v = 0; v < nv; ++v)
            _mm256_storeu_ps(o + v * 8, acc[v]);
#endif
#else
          for (int64_t d = 0; d < g_head_dim; ++d) o[d] = 0.f;
          for (int64_t j = 0; j < g_seq; ++j) {
            const float *v = &qkv_p[(b * g_seq + j) * 3 * g_hidden +
                                    2 * g_hidden + h * g_head_dim];
            for (int64_t d = 0; d < g_head_dim; ++d) o[d] += a[j] * v[d];
          }
#endif
        }
      }
    });
  }

  void av(const std::vector<float> &scores, const std::vector<float> &qkvbuf,
          std::vector<float> &ctx) {
    switch (g_head_dim) {
      case 32: av_impl<4>(scores, qkvbuf, ctx); break;
      case 64: av_impl<8>(scores, qkvbuf, ctx); break;
      default: av_impl<0>(scores, qkvbuf, ctx); break;
    }
  }

  // arch=2: rotate Q and K IN PLACE inside the fused qkv buffer -- Q at
  // column offset 0, K at `hidden`, V at `2*hidden` -- rather than repacking
  // into [B,H,S,D] the way gemma_encode.cpp does. Repacking would be a
  // ~200 MB shuffle per layer at production batch; rotating the 64 floats of
  // each head in place needs no extra buffer at all.
  //
  // For each (b, s) row and each head h, at `row_base + q_off + h*head_dim`
  // (`half = head_dim/2`):
  //   for d in [0, half):
  //     x1 = v[d]; x2 = v[d + half]
  //     v[d]        = x1*cos[s][d] - x2*sin[s][d]
  //     v[d + half] = x2*cos[s][d] + x1*sin[s][d]
  // Both halves read cos[s][d]/sin[s][d] for the SAME d -- that is what
  // NeoX's concat(freqs, freqs) means (gemma_kernels.hpp's own
  // gemma_rope_tables() already duplicates the table this way), so only the
  // first `half` columns of the table are ever read here. Applied to Q
  // (offset 0) and K (offset g_hidden). NEVER V.
  void apply_rope_qkv(std::vector<float> &qkv) {
    if (!rope_ready) {
      // Built ONCE per Encoder: g_seq/g_head_dim/g_rope_theta are fixed for
      // the whole design (set_design_seq() runs once, before any Encoder
      // exists), so every layer of every call shares this table.
      rope_cos.resize(static_cast<size_t>(g_seq * g_head_dim));
      rope_sin.resize(static_cast<size_t>(g_seq * g_head_dim));
      if (!rope_inv_freq().empty()) {
        // arch=3: the frequencies come from the container (tasks/0134 --
        // not derivable from any single theta). Same NeoX
        // concat(freqs, freqs) table layout gemma_rope_tables() emits, and
        // the same double-angle, round-once-at-the-end arithmetic; the
        // rotation below only ever reads the first half of each row.
        const int64_t half = g_head_dim / 2;
        for (int64_t s = 0; s < g_seq; ++s) {
          float *cs = rope_cos.data() + s * g_head_dim;
          float *sn = rope_sin.data() + s * g_head_dim;
          for (int64_t j = 0; j < half; ++j) {
            const double ang =
                static_cast<double>(s) *
                static_cast<double>(rope_inv_freq()[static_cast<size_t>(j)]);
            const float c = static_cast<float>(std::cos(ang));
            const float si = static_cast<float>(std::sin(ang));
            cs[j] = c;
            cs[half + j] = c;
            sn[j] = si;
            sn[half + j] = si;
          }
        }
      } else {
        npue::gemma_rope_tables(g_seq, g_head_dim, g_rope_theta,
                                rope_cos.data(), rope_sin.data());
      }
      rope_ready = true;
    }
    const int64_t half = g_head_dim / 2;
    const int64_t row_stride = 3 * g_hidden;
    const int64_t n_rows = batch * g_seq;
    float *__restrict p = qkv.data();
    const float *__restrict cos_p = rope_cos.data();
    const float *__restrict sin_p = rope_sin.data();

    auto rotate_pair = [half](float *v, const float *cs, const float *sn) {
      int64_t d = 0;
#if defined(__AVX2__)
      for (; d + 8 <= half; d += 8) {
        __m256 x1 = _mm256_loadu_ps(v + d);
        __m256 x2 = _mm256_loadu_ps(v + d + half);
        __m256 c = _mm256_loadu_ps(cs + d);
        __m256 s = _mm256_loadu_ps(sn + d);
        __m256 o1 = _mm256_sub_ps(_mm256_mul_ps(x1, c), _mm256_mul_ps(x2, s));
        __m256 o2 = _mm256_add_ps(_mm256_mul_ps(x2, c), _mm256_mul_ps(x1, s));
        _mm256_storeu_ps(v + d, o1);
        _mm256_storeu_ps(v + d + half, o2);
      }
#endif
      for (; d < half; ++d) {
        const float x1 = v[d], x2 = v[d + half];
        v[d] = x1 * cs[d] - x2 * sn[d];
        v[d + half] = x2 * cs[d] + x1 * sn[d];
      }
    };

    pool->run([&](int w, int nw) {
      for (int64_t row = w; row < n_rows; row += nw) {
        const int64_t s = row % g_seq;    // [b][s] row order -> s = row % seq
        const float *cs = cos_p + s * g_head_dim;
        const float *sn = sin_p + s * g_head_dim;
        float *row_base = p + row * row_stride;
        for (int64_t h = 0; h < g_heads; ++h) {
          rotate_pair(row_base + h * g_head_dim, cs, sn);              // Q
          rotate_pair(row_base + g_hidden + h * g_head_dim, cs, sn);   // K
        }
      }
    });
  }

  std::vector<float> run(const std::vector<float> &emb_in) {
    std::vector<float> x = emb_in;
    layer_norm(x, s_ln[0]);

    qkvbuf.resize(rows * 3 * g_hidden);
    ctx.resize(rows * g_hidden);
    proj.resize(rows * g_hidden);
    up.resize(rows * (g_gated_ffn ? 2 : 1) * g_ffn);
    if (g_gated_ffn) gated.resize(rows * g_ffn);
    down.resize(rows * g_hidden);
    scores.resize(batch * g_heads * g_seq * g_seq);

    residual.resize(x.size());
    // Set by the PREVIOUS iteration's fused LayerNorm, which already wrote
    // this layer's qkv operand and its row scales into the device slot.
    bool qkv_a_ready = false;
    for (int64_t L = 0; L < g_layers; ++L) {
      if (!qkv_a_ready)
        std::memcpy(residual.data(), x.data(), x.size() * sizeof(float));

      gemm(qkv, is_qkv, x, s_qkv[L], b_qkv[L], qkvbuf, 3 * g_hidden,
           i8w(ws_qkv, L), i8w(as_qkv, L), nullptr, /*a_ready=*/qkv_a_ready);
      qkv_a_ready = false;
      // arch=2: rotate Q and K in place, strictly after the GEMM (RoPE is a
      // per-position rotation of the projected q/k, not of the input) and
      // strictly before qk() reads them. No-op (false) for every arch=0
      // container.
      if (g_rope) apply_rope_qkv(qkvbuf);

      double ta = now_s();
      // QK^T per head, on the host: [64,32]x[32,64] does not tile (head_dim 32
      // fails the whole-array design's M % (m*4) == 0).
      // 1/sqrt(head_dim) is already folded into Q by the .npue.
      //
      // head_dim is 32 = four AVX2 vectors, and the 32 floats of one head ARE
      // contiguous even though consecutive rows are 3*hidden apart. So the dot
      // product vectorises without any repacking.
      qk(qkvbuf, scores);
      t_attn += now_s() - ta; t_qk += now_s() - ta;

      if (host_sm) {
        softmax_cpu(scores);  // applies add_mask itself
      } else {
        add_additive_mask(scores);
        eltwise(softmax, scores.data(), scores.size());
      }

      ta = now_s();
      // A.V. Each (b, h, i) owns its own 32 output floats, so this accumulates
      // in registers and stores once -- no zero-fill of ctx needed, and no
      // sharing between threads.
      av(scores, qkvbuf, ctx);
      t_attn += now_s() - ta; t_av += now_s() - ta;

      gemm(attn_out, is_ao, ctx, s_ao[L], b_ao[L], proj, g_hidden,
           i8w(ws_ao, L), i8w(as_ao, L));
      // T37 site 1: add + LayerNorm + residual copy + ffn_up's quantisation,
      // one L1-resident pass instead of four streaming ones.
      const bool fuse_ln = fuse_ffn_epilogue && host_ln &&
                           ffn_up.info().a_elem_bytes == 1;
      // T37-BF16 (tasks/0108): the bf16/bfp16 analogue -- same fused pass,
      // narrowing straight to bf16 instead of quantising (no scale, no
      // smoothing divisor).
      const bool fuse_ln_bf16 = fuse_ffn_epilogue && host_ln &&
                                ffn_up.info().a_elem_bytes == 2;
      if (fuse_ln) {
        const float *asf = i8w(as_fu, L);
        if (inv_smooth.size() != static_cast<size_t>(g_hidden) ||
            inv_smooth_src != asf) {
          inv_smooth.resize(static_cast<size_t>(g_hidden));
          for (int64_t j = 0; j < g_hidden; ++j) inv_smooth[j] = 1.0f / asf[j];
          inv_smooth_src = asf;
        }
        a_scale.resize(static_cast<size_t>(rows));
        // `s_ln[...]` is a design SLOT; h_gamma/h_beta are indexed by
        // site, and layer_norm() converts with `slot - 1`. Passing the
        // slot straight through segfaulted on the last layer.
        add_norm_quant(x, proj, s_ln[1 + 2 * L] - 1, inv_smooth.data(),
                       static_cast<int8_t *>(ffn_up.slot_ptr(0, slot_a)),
                       a_scale.data());
      } else if (fuse_ln_bf16) {
        add_norm_bf16(x, proj, s_ln[1 + 2 * L] - 1,
                     static_cast<uint16_t *>(ffn_up.slot_ptr(0, slot_a)));
      } else {
        add_into(x, proj);
        layer_norm(x, s_ln[1 + 2 * L]);
        std::memcpy(residual.data(), x.data(), x.size() * sizeof(float));
      }
      // T37: when the whole ffn_up -> GELU -> ffn_down chain is int8 and GELU
      // is on the host, the epilogue is fused and `up` is never materialised.
      // Gated FFNs fuse too: SwiGLU narrows 2*inter -> inter inside the same
      // L1-resident row, which is if anything a bigger saving because their
      // ffn_up output is twice as wide.
      FusedNext fn{i8w(as_fd, L),
                   static_cast<int8_t *>(ffn_down.slot_ptr(0, slot_a)),
                   nullptr, g_gated_ffn};
      const bool fuse_ffn = fuse_ffn_epilogue && host_gelu &&
                            ffn_up.info().a_elem_bytes == 1 &&
                            ffn_down.info().a_elem_bytes == 1;
      if (fuse_ffn) {
        a_scale_next.resize(static_cast<size_t>(rows));
        fn.scale = a_scale_next.data();
      }
      // T37-BF16 (tasks/0108): the bf16/bfp16 analogue of `fn`/`fuse_ffn`
      // above -- STEP 1 measured this chain (ffn_up's C-readback+bias, the
      // activation, ffn_down's A-convert) at 17.7-28.3% of wall clock across
      // the shipped catalogue before this fusion existed.
      FusedNextBf16 fn_bf16{
          static_cast<uint16_t *>(ffn_down.slot_ptr(0, slot_a)), g_gated_ffn};
      const bool fuse_ffn_bf16 = fuse_ffn_epilogue && host_gelu &&
                                 ffn_up.info().a_elem_bytes == 2 &&
                                 ffn_down.info().a_elem_bytes == 2;
      gemm(ffn_up, is_fu, x, s_fu[L], b_fu[L], up,
           g_gated_ffn ? 2 * g_ffn : g_ffn, i8w(ws_fu, L), i8w(as_fu, L),
           fuse_ffn ? &fn : nullptr, /*a_ready=*/fuse_ln || fuse_ln_bf16,
           fuse_ffn_bf16 ? &fn_bf16 : nullptr);

      // arch=2: SwiGLU (fc11 * silu(fc12), fused as one [hidden, 2*inter]
      // ffn_up) in place of plain GELU over a [hidden, inter] ffn_up --
      // writes into the separate `gated` buffer (see swiglu_cpu's own
      // comment for why NOT in place). Every arch=0 container has
      // g_gated_ffn == false and takes the untouched branch below.
      if (fuse_ffn) {
        // T37: the activation's output was never written. The ffn_up epilogue
        // already produced ffn_down's int8 operand in the device slot and its
        // row scales, so this dispatch has nothing to convert.
        a_scale.swap(a_scale_next);
        gemm(ffn_down, is_fd, g_gated_ffn ? gated : up, s_fd[L], b_fd[L], down,
             g_hidden, i8w(ws_fd, L), i8w(as_fd, L), nullptr, /*a_ready=*/true);
      } else if (fuse_ffn_bf16) {
        // T37-BF16: `up`/`gated` was never written -- the ffn_up epilogue
        // already produced ffn_down's bf16 operand in the device slot.
        gemm(ffn_down, is_fd, g_gated_ffn ? gated : up, s_fd[L], b_fd[L], down,
             g_hidden, i8w(ws_fd, L), i8w(as_fd, L), nullptr, /*a_ready=*/true);
      } else if (g_gated_ffn) {
        swiglu_cpu(up, gated);
        gemm(ffn_down, is_fd, gated, s_fd[L], b_fd[L], down, g_hidden,
                 i8w(ws_fd, L), i8w(as_fd, L));
      } else if (false) {
        // T37: `up` was never written. The ffn_up epilogue already produced
        // ffn_down's int8 operand in the device slot and its row scales, so
        // this dispatch has nothing to convert.
        a_scale.swap(a_scale_next);
        gemm(ffn_down, is_fd, up, s_fd[L], b_fd[L], down, g_hidden,
             i8w(ws_fd, L), i8w(as_fd, L), nullptr, /*a_ready=*/true);
      } else {
        if (host_gelu)
          gelu_cpu(up);
        else
          eltwise(gelu, up.data(), up.size());
        gemm(ffn_down, is_fd, up, s_fd[L], b_fd[L], down, g_hidden,
                 i8w(ws_fd, L), i8w(as_fd, L));
      }
      // T37 site 2: the same fusion at the layer's second LayerNorm. Its
      // consumer is the NEXT layer's qkv, so the loop's own
      // `memcpy(residual, x)` is what this replaces -- and on the last layer
      // there is no next GEMM, only pooling, so `dst` is null there.
      const bool last = (L + 1 == g_layers);
      if (fuse_ffn_epilogue && host_ln && qkv.info().a_elem_bytes == 1) {
        const float *asq = last ? nullptr : i8w(as_qkv, L + 1);
        if (asq && (inv_smooth.size() != static_cast<size_t>(g_hidden) ||
                    inv_smooth_src != asq)) {
          inv_smooth.resize(static_cast<size_t>(g_hidden));
          for (int64_t j = 0; j < g_hidden; ++j) inv_smooth[j] = 1.0f / asq[j];
          inv_smooth_src = asq;
        }
        if (asq) a_scale.resize(static_cast<size_t>(rows));
        add_norm_quant(x, down, s_ln[2 + 2 * L] - 1,
                       asq ? inv_smooth.data() : nullptr,
                       asq ? static_cast<int8_t *>(qkv.slot_ptr(0, slot_a))
                           : nullptr,
                       asq ? a_scale.data() : nullptr);
        qkv_a_ready = !last;
      } else if (fuse_ffn_epilogue && host_ln && qkv.info().a_elem_bytes == 2) {
        // T37-BF16 (tasks/0108): as the int8 branch above, narrowing to bf16
        // instead of quantising -- no scale to compute.
        add_norm_bf16(x, down, s_ln[2 + 2 * L] - 1,
                      last ? nullptr
                           : static_cast<uint16_t *>(qkv.slot_ptr(0, slot_a)));
        qkv_a_ready = !last;
      } else {
        add_into(x, down);
        layer_norm(x, s_ln[2 + 2 * L]);
      }
    }
    return x;
  }
};

// ===========================================================================
// EmbeddingGemma-300M on the array (arch=1) -- tasks/0074.
// ===========================================================================
//
// WHY A SECOND ENCODER AND NOT A BRANCH IN Encoder::run().
//
// arch=2 (nomic) could be a branch because it IS a BERT block with two things
// swapped: RoPE instead of an absolute position table, and a gated FFN instead
// of a plain one. Gemma is not. Its block is a four-RMSNorm sandwich
// (pre-norm AND post-norm around both sub-layers), its attention carries
// q_norm/k_norm between the projection and RoPE, its RoPE base changes per
// layer, it scales the embedding by sqrt(hidden), and it ends in two post-pool
// Dense heads. Threading all of that through a function that serves five
// shipped models as `if (arch == ...)` would put the shipped models one typo
// away from a silent wrong answer, which is the failure this project keeps
// finding. So this is separate code that happens to reuse the same MACHINERY:
// Pool, the bf16 conversions, npu::Design staging, and the same
// bind/sync/dispatch/bias sequence Encoder::gemm() uses.
//
// The host-only npue::GemmaEncoder (tasks/0064, verified to 1-cos 5.496e-13
// against reference/encoder_gemma.py) is NOT replaced by this. It is the
// discriminating control: the same container geometry, the same tokenizer,
// every GEMM in double precision on the CPU. Any disagreement between the two
// beyond the bf16 floor is a bug in THIS file.
//
// WHAT RUNS WHERE. Four GEMMs per layer go to the array -- 97.7% of the
// model's MACs (tasks/0074 sec 4). Attention's QK^T and A.V stay on the host:
// at head_dim 256 and seq 64 their N is 64, which fails the design's
// `N % (n * n_aie_cols) == 0` outright, and F3 prices the whole of attention
// at 2.3% of MACs here. RMSNorm, RoPE and GeGLU stay on the host on
// tasks/0032's measured precedent that a host eltwise pass beats an NPU
// dispatch at these widths.
struct GemmaNpuEncoder {
  npue::File &model;
  npu::Design &d;
  Pool *pool = nullptr;
  npue::GemmaTokenizer tok;

  // Geometry, read from the container. Nothing here is a literal: this file
  // has no idea that hidden is 768 or that there are 24 layers.
  int64_t hidden = 0, heads = 0, kv_heads = 0, head_dim = 0, inter = 0,
          layers = 0, dense_hidden = 0, qkv_n = 0, swp = 6;
  int64_t q_off = 0, k_off = 0, v_off = 0, kv_w = 0;
  double eps = 1e-6, rope_theta = 0.0, rope_theta_local = 0.0,
         attn_scale = 1.0;

  int64_t seq = 0, batch = 0, rows = 0;
  std::vector<int64_t> tiers;
  std::vector<std::array<size_t, 4>> tier_slots;
  size_t is_qkv = 0, is_ao = 0, is_fu = 0, is_fd = 0;
  size_t slot_a = 0, slot_c = 0;
  // Lanes (tasks/0033's mechanism, applied to this arch): several encoders
  // share the ONE design, each owning its A and C slots, with every NPU
  // interaction under one mutex. The array serialises dispatches anyway, so
  // the lock only makes explicit what the hardware enforces -- what overlaps
  // is one lane's HOST work with another's array work. It matters more here
  // than on any BERT model: the first measurement of this path put the array
  // at 48% of wall clock and the host at 47%, which is as close to the ideal
  // case for overlap as this project has met.
  std::mutex *npu_mu = nullptr;

  std::vector<size_t> s_qkv, s_ao, s_fu, s_fd;
  std::vector<const float *> b_qkv, b_ao, b_fu, b_fd;
  // int8 (tasks/0081). Per-output-channel weight scales and per-input-channel
  // SmoothQuant divisors, one pointer per layer per op, filled by stage_all()
  // and empty on a bf16 container.
  std::vector<const float *> ws_qkv, ws_ao, ws_fu, ws_fd;
  std::vector<const float *> as_qkv, as_ao, as_fu, as_fd;
  // Per-row activation scales for the current GEMM, [rows].
  std::vector<float> a_scale;
  // 1/asmooth for the GEMM being run, rebuilt only when the pointer changes --
  // K floats of setup against 2*rows*K divisions (tasks/0080).
  std::vector<float> inv_smooth;
  const float *inv_smooth_src = nullptr;
  std::vector<float> a_scale_next, inv_smooth_next;
  const float *inv_smooth_next_src = nullptr;
  bool fuse_ffn_epilogue = true;

  struct LayerHost {
    const float *q_norm, *k_norm, *ln_in, *ln_pa, *ln_pf, *ln_pof;
  };
  std::vector<LayerHost> lh;
  const float *w_embed = nullptr, *w_norm = nullptr;
  const float *w_dense2 = nullptr, *w_dense3 = nullptr;

  // Scratch, sized once per batch and reused across layers and calls.
  std::vector<float> x, hbuf, qkvbuf, ctx, proj, upbuf, gatedbuf, down,
      scores, add_mask, cos_g, sin_g, cos_l, sin_l;
  std::vector<int32_t> ids;
  std::vector<uint8_t> mask;

  double t_conv = 0, t_in = 0, t_disp = 0, t_out = 0, t_bias = 0,
         t_norm = 0, t_attn = 0, t_rope = 0, t_geglu = 0, t_tok = 0;
  int n_dispatch = 0;
  void reset_timers() {
    t_conv = t_in = t_disp = t_out = t_bias = 0;
    t_norm = t_attn = t_rope = t_geglu = t_tok = 0;
    n_dispatch = 0;
  }

  GemmaNpuEncoder(npue::File &m, npu::Design &design, Pool &p)
      : model(m), d(design), pool(&p) {
    const std::string arch = m.config_string("arch");
    if (arch != "gemma3_mqa_rope_geglu")
      throw std::runtime_error("GemmaNpuEncoder given arch '" + arch + "'");
    // The container must say it holds PRE-TILED operands. A host-only
    // container carries the same tensor VALUES in the same file under
    // different names and a different layout; reading one as the other is
    // tasks/0022's rel_fro 1.186 all over again.
    const std::string layout = m.config_string("gemm_layout");
    if (layout != "pretiled_bf16")
      throw std::runtime_error(
          "this container's gemm_layout is '" + layout + "', not "
          "'pretiled_bf16' -- it holds host-side row-major F32 operands and "
          "has no tiled weights for the array. Repack it with "
          "tools/pack_npue.py (the NPU layout is now the default).");

    hidden = m.config_int("hidden");
    heads = m.config_int("num_heads");
    kv_heads = m.config_int("num_key_value_heads");
    head_dim = m.config_int("head_dim");
    inter = m.config_int("intermediate");
    layers = m.config_int("num_layers");
    dense_hidden = m.config_int("dense_hidden");
    swp = m.config_int("sliding_window_pattern");
    eps = m.config_double("rms_norm_eps");
    rope_theta = m.config_double("rope_theta");
    rope_theta_local = m.config_double("rope_local_base_freq");
    attn_scale = std::pow(m.config_double("query_pre_attn_scalar"), -0.5);
    qkv_n = m.config_int("qkv_n");
    kv_w = kv_heads * head_dim;

    if (kv_heads != 1)
      throw std::runtime_error(
          "this encoder's attention loop assumes num_key_value_heads == 1 "
          "(EmbeddingGemma-300M); a GQA checkpoint needs the K/V reuse "
          "generalised first");
    if (head_dim * heads != hidden)
      throw std::runtime_error("head_dim * num_heads != hidden");
    if (head_dim % 2)
      throw std::runtime_error("odd head_dim -- RoPE cannot half-split it");
    if (m.config_string("geglu_halves") != "gate|up")
      throw std::runtime_error(
          "unrecognised geglu_halves '" + m.config_string("geglu_halves") +
          "' -- expected 'gate|up'; refusing rather than guessing which half "
          "of the fused ffn_up gets the GELU. tasks/0068 Q2 measured the "
          "swapped variant on the sibling architecture at rel_fro 4.022e+00.");

    // Q/K/V offsets are READ, never derived. `3*hidden` is the BERT answer and
    // it is wrong here twice over: MQA makes K and V narrower than Q, and the
    // operand is zero-padded past them (tasks/0074).
    q_off = 0;
    k_off = hidden;
    v_off = hidden + kv_w;
    if (qkv_n < v_off + kv_w)
      throw std::runtime_error("qkv_n is too small to hold Q|K|V");

    w_embed = m.raw("embed_tokens.weight").as<float>();
    w_norm = m.raw("norm.weight").as<float>();
    w_dense2 = m.raw("dense2.weight").as<float>();
    w_dense3 = m.raw("dense3.weight").as<float>();
    lh.resize(static_cast<size_t>(layers));
    for (int64_t L = 0; L < layers; ++L) {
      const std::string p = "layer." + std::to_string(L) + ".";
      LayerHost &l = lh[static_cast<size_t>(L)];
      l.q_norm = m.raw(p + "q_norm.weight").as<float>();
      l.k_norm = m.raw(p + "k_norm.weight").as<float>();
      l.ln_in = m.raw(p + "input_layernorm.weight").as<float>();
      l.ln_pa = m.raw(p + "post_attention_layernorm.weight").as<float>();
      l.ln_pf = m.raw(p + "pre_feedforward_layernorm.weight").as<float>();
      l.ln_pof = m.raw(p + "post_feedforward_layernorm.weight").as<float>();
    }
    auto tv = m.raw("tokenizer.gemma_table");
    tok = npue::GemmaTokenizer::from_table_bytes(
        reinterpret_cast<const char *>(tv.data), tv.bytes);
  }

  template <typename F> void par(size_t n, F &&f) const {
    if (pool == nullptr || pool->size() == 1 || n < 65536) {
      f(size_t(0), n);
      return;
    }
    pool->run([&](int w, int nw) {
      const size_t chunk = ((n / nw) + 63) & ~size_t(63);
      const size_t lo = std::min(n, chunk * size_t(w));
      const size_t hi = std::min(n, lo + chunk);
      if (lo < hi) f(lo, hi);
    });
  }
  // Row-parallel, for the strided passes where a flat byte range would split
  // a row.
  template <typename F> void par_rows(int64_t n, F &&f) const {
    if (pool == nullptr || pool->size() == 1) { f(int64_t(0), n); return; }
    pool->run([&](int w, int nw) {
      const int64_t chunk = (n + nw - 1) / nw;
      const int64_t lo = std::min<int64_t>(n, chunk * w);
      const int64_t hi = std::min<int64_t>(n, lo + chunk);
      if (lo < hi) f(lo, hi);
    });
  }

  double lap(double t0, double &bucket) {
    const double t = now_s();
    bucket += t - t0;
    return t;
  }

  size_t stage_all() {
    size_t bytes = 0;
    const bool i8 = d.info().a_elem_bytes == 1;
    auto one = [&](const std::string &name, std::vector<size_t> &slots,
                   std::vector<const float *> &bias,
                   std::vector<const float *> *wsc,
                   std::vector<const float *> *asm_) {
      const std::string &want = d.info().b_layout_hash;
      const std::string &got = model.info(name).layout_hash;
      if (want.empty() || got.empty() || want != got)
        throw std::runtime_error(
            name + ": B layout mismatch -- design wants " +
            (want.empty() ? std::string("(nothing stated)") : want.substr(0, 16)) +
            ", container has " +
            (got.empty() ? std::string("(nothing stated)") : got.substr(0, 16)) +
            ". The bytes would be the right size and the wrong order.");
      auto w = model.raw(name);
      slots.push_back(d.stage(1, w.data, w.bytes));
      // Gemma has no biases anywhere; the packer zero-fills them so this
      // dispatch path stays byte-for-byte the BERT one (tasks/0074 sec 5).
      bias.push_back(model.raw(name + ".bias").as<float>());
      if (i8) {
        // Refuse rather than dereference: an int8 DESIGN with a bf16
        // CONTAINER passes the layout check only if the container is also
        // int8, but a container packed by an older packer would carry i8
        // bytes without the scales that give them meaning.
        wsc->push_back(model.raw(name + ".wscale").as<float>());
        asm_->push_back(model.raw(name + ".asmooth").as<float>());
      }
      bytes += w.bytes;
    };
    for (int64_t L = 0; L < layers; ++L) {
      const std::string p = "layer." + std::to_string(L) + ".";
      one(p + "qkv", s_qkv, b_qkv, &ws_qkv, &as_qkv);
      one(p + "attn_out", s_ao, b_ao, &ws_ao, &as_ao);
      one(p + "ffn_up", s_fu, b_fu, &ws_fu, &as_fu);
      one(p + "ffn_down", s_fd, b_fd, &ws_fd, &as_fd);
    }
    return bytes;
  }

  int64_t use_tier(int64_t want) {
    if (tiers.empty()) return batch;
    size_t pick = tiers.size() - 1;
    for (size_t i = 0; i < tiers.size(); ++i)
      if (tiers[i] >= want) { pick = i; break; }
    batch = tiers[pick];
    rows = batch * seq;
    is_qkv = tier_slots[pick][0];
    is_ao = tier_slots[pick][1];
    is_fu = tier_slots[pick][2];
    is_fd = tier_slots[pick][3];
    return batch;
  }

  // Identical in shape to Encoder::gemm() -- convert A to bf16 into this
  // pipeline's own slot, bind, sync, dispatch, then read C back with a
  // streaming load and add the (zero) bias.
  // The scale vectors are empty on a bf16 container, so this yields nullptr
  // and gemm()'s own check decides whether that is legal for the design.
  static const float *at(const std::vector<const float *> &v, int64_t L) {
    return L < static_cast<int64_t>(v.size()) ? v[static_cast<size_t>(L)]
                                              : nullptr;
  }

  // T37 (tasks/0082): as Encoder::FusedNext, for arch=1's GeGLU. `gated` is
  // always true here -- Gemma has no un-gated FFN.
  struct FusedNext {
    const float *asmooth;
    int8_t *dst;
    float *scale;
  };

  // T37-BF16 (tasks/0108): as Encoder::FusedNextBf16, for arch=1's GeGLU.
  struct FusedNextBf16 {
    uint16_t *dst;
  };

  // T37-BF16 (tasks/0108): as Encoder::dequant_act_bf16, for arch=1's GeGLU
  // (gelu_pytorch_tanh(gate) * up, NOT the same function as BERT's GELU --
  // see geglu()'s own comment). Bit-identical to the unfused
  // gemm(...)+geglu() pair by construction: the C-readback+bias arm is
  // copied verbatim from this gemm()'s own unfused branches below, and the
  // activation arm is copied verbatim from the int8 fused epilogue's lambda
  // above (itself bit-identical to geglu(), including the scalar tail's
  // double-precision rounding).
  void dequant_act_bf16(const void *c, size_t c_bytes, int64_t N,
                        int64_t out_n, const float *bias, uint16_t *dst) {
    par_rows(rows, [&](int64_t r0, int64_t r1) {
      std::vector<float> row(static_cast<size_t>(N));
      for (int64_t r = r0; r < r1; ++r) {
        float *v = row.data();
        int64_t j = 0;
#if defined(__AVX2__)
        if (c_bytes == 2) {
          const uint16_t *cr = static_cast<const uint16_t *>(c) + r * N;
          for (; j + 16 <= N; j += 16) {
            __m256i raw = _mm256_stream_load_si256(
                reinterpret_cast<const __m256i *>(cr + j));
            __m256i lo = _mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm256_castsi256_si128(raw)), 16);
            __m256i hi = _mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm256_extracti128_si256(raw, 1)), 16);
            _mm256_storeu_ps(v + j, _mm256_add_ps(_mm256_castsi256_ps(lo),
                                                  _mm256_loadu_ps(bias + j)));
            _mm256_storeu_ps(v + j + 8, _mm256_add_ps(_mm256_castsi256_ps(hi),
                                                  _mm256_loadu_ps(bias + j + 8)));
          }
        } else {
          const float *cr = static_cast<const float *>(c) + r * N;
          for (; j + 8 <= N; j += 8) {
            __m256i raw = _mm256_stream_load_si256(
                reinterpret_cast<const __m256i *>(cr + j));
            _mm256_storeu_ps(v + j, _mm256_add_ps(_mm256_castsi256_ps(raw),
                                                  _mm256_loadu_ps(bias + j)));
          }
        }
#endif
        for (; j < N; ++j) {
          const float cf = c_bytes == 2
              ? from_bf16(static_cast<const uint16_t *>(c)[r * N + j])
              : static_cast<const float *>(c)[r * N + j];
          v[j] = cf + bias[j];
        }
        // GeGLU, in place, narrowing to out_n -- COPIED VERBATIM from the
        // int8 fused epilogue's lambda / geglu() above.
        const int64_t inter = N / 2;
        const float *u = v + inter;
        int64_t k = 0;
#if defined(__AVX2__)
        const __m256 c_half = _mm256_set1_ps(0.5f);
        const __m256 c_one = _mm256_set1_ps(1.0f);
        const __m256 c_sq = _mm256_set1_ps(0.7978845608028654f);
        const __m256 c_k = _mm256_set1_ps(0.044715f);
        const __m256 c_2log2e = _mm256_set1_ps(2.885390081777927f);
        const __m256 c_lim = _mm256_set1_ps(15.0f);
        for (; k + 8 <= inter; k += 8) {
          __m256 xv = _mm256_loadu_ps(v + k);
          __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(xv, xv), xv);
          __m256 y = _mm256_mul_ps(c_sq, _mm256_fmadd_ps(c_k, x3, xv));
          y = _mm256_min_ps(
              _mm256_max_ps(y, _mm256_sub_ps(_mm256_setzero_ps(), c_lim)),
              c_lim);
          __m256 e = Encoder::exp2_avx2(_mm256_mul_ps(y, c_2log2e));
          __m256 th = _mm256_div_ps(_mm256_sub_ps(e, c_one),
                                    _mm256_add_ps(e, c_one));
          __m256 act = _mm256_mul_ps(_mm256_mul_ps(c_half, xv),
                                     _mm256_add_ps(c_one, th));
          _mm256_storeu_ps(v + k, _mm256_mul_ps(act, _mm256_loadu_ps(u + k)));
        }
#endif
        for (; k < inter; ++k) {
          const double xv = v[k];
          const double y =
              0.7978845608028654 * (xv + 0.044715 * xv * xv * xv);
          const float act =
              static_cast<float>(0.5 * xv * (1.0 + std::tanh(y)));
          v[k] = static_cast<float>(static_cast<double>(act) *
                                    static_cast<double>(u[k]));
        }
        bf16_fill(dst + r * out_n, v, static_cast<size_t>(out_n));
      }
    });
  }

  void gemm(size_t islot, const float *a, size_t a_len, size_t wslot,
            const float *bias, std::vector<float> &out, int64_t N,
            const float *wscale = nullptr, const float *asmooth = nullptr,
            FusedNext *fuse = nullptr, bool a_ready = false,
            FusedNextBf16 *fuse_bf16 = nullptr) {
    const bool i8 = d.info().a_elem_bytes == 1;
    if (i8 && (wscale == nullptr || asmooth == nullptr))
      throw std::runtime_error(
          "int8 design but this encoder has no quantisation scales -- the "
          "container is bf16, or it was packed before tools/pack_npue.py "
          "--int8 supported arch=1");
    double t0 = now_s();
    if (i8 && a_ready) {
      // A is already in the device slot, written by the previous GEMM's fused
      // epilogue, and a_scale already holds its row scales.
    } else if (i8) {
      const int64_t K = static_cast<int64_t>(a_len) / rows;
      a_scale.resize(static_cast<size_t>(rows));
      if (inv_smooth.size() != static_cast<size_t>(K) ||
          inv_smooth_src != asmooth) {
        inv_smooth.resize(static_cast<size_t>(K));
        for (int64_t j = 0; j < K; ++j) inv_smooth[j] = 1.0f / asmooth[j];
        inv_smooth_src = asmooth;
      }
      quantise_a_int8(a, rows, K, inv_smooth.data(),
                      static_cast<int8_t *>(d.slot_ptr(0, slot_a)),
                      a_scale.data(),
                      [&](int64_t n, auto f) { par_rows(n, f); });
      t0 = lap(t0, t_conv);
    } else if (a_ready) {
      // T37-BF16 (tasks/0108): A was written straight into the device slot,
      // already narrowed to bf16, by the PREVIOUS gemm's fused epilogue.
    } else {
    auto *abuf = static_cast<uint16_t *>(d.slot_ptr(0, slot_a));
    par(a_len, [&](size_t lo, size_t hi) {
      bf16_fill(abuf + lo, a + lo, hi - lo);
    });
    t0 = lap(t0, t_conv);
    }
    const float *c = npu_dispatch(
        d, npu_mu, /*set_instr=*/true, islot, slot_a, wslot, slot_c, a_len,
        static_cast<size_t>(rows) * N * d.info().c_elem_bytes,
        t0, t_in, t_disp, t_out);
    if (i8 && fuse) {
      // FUSED GeGLU EPILOGUE (T37): dequantise, gate, and quantise ffn_down's
      // operand in one L1-resident pass. `out` is never written.
      const int64_t out_n = N / 2;
      if (inv_smooth_next.size() != static_cast<size_t>(out_n) ||
          inv_smooth_next_src != fuse->asmooth) {
        inv_smooth_next.resize(static_cast<size_t>(out_n));
        for (int64_t j = 0; j < out_n; ++j)
          inv_smooth_next[j] = 1.0f / fuse->asmooth[j];
        inv_smooth_next_src = fuse->asmooth;
      }
      dequant_act_quant(
          c, d.info().c_elem_bytes, rows, N, out_n,
          [](float *v, int64_t n) {
            // Identical intrinsics to geglu(), including its +-15 clamp before
            // exp2 -- a scalar rewrite of the same algebra is a different
            // number (tasks/0082 section 2).
            const int64_t inter = n / 2;
            const float *u = v + inter;
            int64_t j = 0;
#if defined(__AVX2__)
            const __m256 c_half = _mm256_set1_ps(0.5f);
            const __m256 c_one = _mm256_set1_ps(1.0f);
            const __m256 c_sq = _mm256_set1_ps(0.7978845608028654f);
            const __m256 c_k = _mm256_set1_ps(0.044715f);
            const __m256 c_2log2e = _mm256_set1_ps(2.885390081777927f);
            const __m256 c_lim = _mm256_set1_ps(15.0f);
            for (; j + 8 <= inter; j += 8) {
              __m256 xv = _mm256_loadu_ps(v + j);
              __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(xv, xv), xv);
              __m256 y = _mm256_mul_ps(c_sq, _mm256_fmadd_ps(c_k, x3, xv));
              y = _mm256_min_ps(
                  _mm256_max_ps(y, _mm256_sub_ps(_mm256_setzero_ps(), c_lim)),
                  c_lim);
              __m256 e = Encoder::exp2_avx2(_mm256_mul_ps(y, c_2log2e));
              __m256 th = _mm256_div_ps(_mm256_sub_ps(e, c_one),
                                        _mm256_add_ps(e, c_one));
              __m256 act = _mm256_mul_ps(_mm256_mul_ps(c_half, xv),
                                         _mm256_add_ps(c_one, th));
              _mm256_storeu_ps(v + j,
                               _mm256_mul_ps(act, _mm256_loadu_ps(u + j)));
            }
#endif
            for (; j < inter; ++j) {
              const double xv = v[j];
              const double y =
                  0.7978845608028654 * (xv + 0.044715 * xv * xv * xv);
              const float act =
                  static_cast<float>(0.5 * xv * (1.0 + std::tanh(y)));
              v[j] = static_cast<float>(static_cast<double>(act) *
                                        static_cast<double>(u[j]));
            }
          },
          a_scale.data(), wscale, bias, inv_smooth_next.data(), fuse->dst,
          fuse->scale, [&](int64_t n, auto f) { par_rows(n, f); });
    } else if (i8) {
      // Same rank-1 dequantisation the BERT encoder uses, from the same
      // helper -- the transport width comes from the design, so a narrowed-C
      // int8 set (tasks/0080) needs nothing extra here.
      dequantise_c(c, d.info().c_elem_bytes, rows, N, a_scale.data(), wscale,
                   bias, out.data(),
                   [&](int64_t n, auto f) { par_rows(n, f); });
    } else if (fuse_bf16) {
      // T37-BF16 (tasks/0108): `out` is deliberately never written -- see
      // dequant_act_bf16 above.
      dequant_act_bf16(c, d.info().c_elem_bytes, N, N / 2, bias,
                       fuse_bf16->dst);
    } else if (d.info().c_elem_bytes == 2) {
      const uint16_t *cb16 = reinterpret_cast<const uint16_t *>(c);
      par_rows(rows, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
          const uint16_t *cr = cb16 + r * N;
          float *o = out.data() + r * N;
          for (int64_t j = 0; j < N; ++j) o[j] = from_bf16(cr[j]) + bias[j];
        }
      });
    } else {
      par_rows(rows, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
          const float *cr = c + r * N;
          float *o = out.data() + r * N;
          int64_t j = 0;
#if defined(__AVX2__)
          // Streaming loads: C is an XRT write-combined host bo and ordinary
          // loads from it stall per line (tasks/0024). N is a multiple of 48
          // here, so the 8-wide tail is handled by the scalar loop.
          for (; j + 8 <= N; j += 8)
            _mm256_storeu_ps(o + j,
                             _mm256_add_ps(_mm256_castsi256_ps(
                                               _mm256_stream_load_si256(
                                                   reinterpret_cast<const __m256i *>(cr + j))),
                                           _mm256_loadu_ps(bias + j)));
#endif
          for (; j < N; ++j) o[j] = cr[j] + bias[j];
        }
      });
    }
    lap(t0, t_bias);
    ++n_dispatch;
  }

  // Gemma3RMSNorm over `dim` contiguous elements per row, with independent
  // input and output row strides so it can work on a slice of the fused qkv
  // buffer in place. `out = x * rsqrt(mean(x^2) + eps) * (1 + w)` -- the
  // `1 +` is Gemma's and omitting it is the single easiest way to produce a
  // plausible wrong answer here (gemma_kernels.hpp's own warning).
  //
  // fp32 accumulation, not the double reduction npue::rms_norm_cpu uses. That
  // function exists to match reference/encoder_gemma.py bit-for-bit and is
  // still what the host-only control runs; this path already rounds every
  // GEMM through bf16, so a double-precision reduction here would buy nothing
  // measurable and costs a factor on 96 norm sites per encode. The end-to-end
  // check against the control is what decides whether that is true.
  void rms_norm(const float *xin, int64_t in_stride, float *xout,
                int64_t out_stride, int64_t n_rows, int64_t dim,
                const float *w) {
    const double t0 = now_s();
    const float e = static_cast<float>(eps);
    par_rows(n_rows, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const float *in = xin + r * in_stride;
        float *o = xout + r * out_stride;
        int64_t j = 0;
        float ss;
#if defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        for (; j + 8 <= dim; j += 8) {
          __m256 v = _mm256_loadu_ps(in + j);
          acc = _mm256_fmadd_ps(v, v, acc);
        }
        ss = hsum256(acc);
#else
        ss = 0.f;
#endif
        for (; j < dim; ++j) ss += in[j] * in[j];
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(dim) + e);
        j = 0;
#if defined(__AVX2__)
        const __m256 iv = _mm256_set1_ps(inv);
        const __m256 one = _mm256_set1_ps(1.0f);
        for (; j + 8 <= dim; j += 8)
          _mm256_storeu_ps(o + j,
                           _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(in + j), iv),
                                         _mm256_add_ps(one, _mm256_loadu_ps(w + j))));
#endif
        for (; j < dim; ++j) o[j] = in[j] * inv * (1.0f + w[j]);
      }
    });
    t_norm += now_s() - t0;
  }

  // RoPE on Q (every head) and K (the single KV head) in place, inside the
  // fused qkv buffer. Position is `row % seq`, which holds because rows are
  // laid out [batch][seq]. NeoX convention (concat(freqs,freqs), rotate-half),
  // matching gemma_rope_tables().
  void apply_rope(std::vector<float> &qkv, const float *cs_t, const float *sn_t) {
    const double t0 = now_s();
    const int64_t half = head_dim / 2;
    float *__restrict p = qkv.data();
    auto rot = [half](float *v, const float *cs, const float *sn) {
      int64_t dd = 0;
#if defined(__AVX2__)
      for (; dd + 8 <= half; dd += 8) {
        __m256 x1 = _mm256_loadu_ps(v + dd);
        __m256 x2 = _mm256_loadu_ps(v + dd + half);
        __m256 c = _mm256_loadu_ps(cs + dd);
        __m256 s = _mm256_loadu_ps(sn + dd);
        _mm256_storeu_ps(v + dd,
                         _mm256_sub_ps(_mm256_mul_ps(x1, c), _mm256_mul_ps(x2, s)));
        _mm256_storeu_ps(v + dd + half,
                         _mm256_add_ps(_mm256_mul_ps(x2, c), _mm256_mul_ps(x1, s)));
      }
#endif
      for (; dd < half; ++dd) {
        const float a = v[dd], b = v[dd + half];
        v[dd] = a * cs[dd] - b * sn[dd];
        v[dd + half] = b * cs[dd] + a * sn[dd];
      }
    };
    par_rows(rows, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const int64_t s = r % seq;
        const float *cs = cs_t + s * head_dim;
        const float *sn = sn_t + s * head_dim;
        float *base = p + r * qkv_n;
        for (int64_t hh = 0; hh < heads; ++hh)
          rot(base + q_off + hh * head_dim, cs, sn);
        rot(base + k_off, cs, sn);           // kv_heads == 1
      }
    });
    t_rope += now_s() - t0;
  }

  // out = gelu_pytorch_tanh(gate) * up, over the fused [gate | up] ffn_up
  // buffer. Gemma's activation is the tanh approximation, NOT the exact-erf
  // GELU the BERT path uses -- a different function, kept deliberately
  // separate (gemma_kernels.hpp).
  void geglu(const std::vector<float> &fused, std::vector<float> &out) {
    const double t0 = now_s();
    par_rows(rows, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const float *g = fused.data() + r * 2 * inter;
        const float *u = g + inter;
        float *o = out.data() + r * inter;
        int64_t j = 0;
#if defined(__AVX2__)
        const __m256 c_half = _mm256_set1_ps(0.5f);
        const __m256 c_one = _mm256_set1_ps(1.0f);
        const __m256 c_sq = _mm256_set1_ps(0.7978845608028654f);  // sqrt(2/pi)
        const __m256 c_k = _mm256_set1_ps(0.044715f);
        // 2*log2(e): tanh(y) = (2^(2y*log2e) - 1) / (2^(2y*log2e) + 1)
        const __m256 c_2log2e = _mm256_set1_ps(2.885390081777927f);
        // Clamp before exp2 so a large activation saturates tanh instead of
        // overflowing to inf and producing (inf-1)/(inf+1) = NaN. |y| >= 15
        // is tanh = +-1 to well inside fp32 already.
        const __m256 c_lim = _mm256_set1_ps(15.0f);
        for (; j + 8 <= inter; j += 8) {
          __m256 xv = _mm256_loadu_ps(g + j);
          __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(xv, xv), xv);
          __m256 y = _mm256_mul_ps(c_sq, _mm256_fmadd_ps(c_k, x3, xv));
          y = _mm256_min_ps(_mm256_max_ps(y, _mm256_sub_ps(_mm256_setzero_ps(), c_lim)),
                            c_lim);
          __m256 e = Encoder::exp2_avx2(_mm256_mul_ps(y, c_2log2e));
          __m256 th = _mm256_div_ps(_mm256_sub_ps(e, c_one),
                                    _mm256_add_ps(e, c_one));
          __m256 act = _mm256_mul_ps(_mm256_mul_ps(c_half, xv),
                                     _mm256_add_ps(c_one, th));
          _mm256_storeu_ps(o + j, _mm256_mul_ps(act, _mm256_loadu_ps(u + j)));
        }
#endif
        // Scalar tail, in the reference's own two-stage rounding (round `act`
        // to fp32, THEN promote and multiply by `up`). It never runs at this
        // model's intermediate width -- 1152 is a multiple of 8 -- and is kept
        // matching the reference rather than matching the vector body above,
        // so a future width with a tail lands on the more accurate form.
        for (; j < inter; ++j) {
          const double xv = g[j];
          const double y = 0.7978845608028654 * (xv + 0.044715 * xv * xv * xv);
          const float act = static_cast<float>(0.5 * xv * (1.0 + std::tanh(y)));
          o[j] = static_cast<float>(static_cast<double>(act) *
                                    static_cast<double>(u[j]));
        }
      }
    });
    t_geglu += now_s() - t0;
  }

  // MQA attention on the host. Every one of the `heads` query heads attends to
  // the SAME single K/V head -- mathematically identical to repeat_kv() but
  // without materialising the repeat.
  void attention(const std::vector<float> &qkv, std::vector<float> &out) {
    const double t0 = now_s();
    const int64_t pairs = batch * heads;
    par_rows(pairs, [&](int64_t p0, int64_t p1) {
      std::vector<float> row(static_cast<size_t>(seq));
      for (int64_t pi = p0; pi < p1; ++pi) {
        const int64_t b = pi / heads, hh = pi % heads;
        const float *base = qkv.data() + b * seq * qkv_n;
        const float *mk = add_mask.data() + b * seq;
        for (int64_t i = 0; i < seq; ++i) {
          const float *qi = base + i * qkv_n + q_off + hh * head_dim;
          float mx = -3.4e38f;
          for (int64_t j = 0; j < seq; ++j) {
            const float *kj = base + j * qkv_n + k_off;
            int64_t dd = 0;
            float acc;
#if defined(__AVX2__)
            __m256 a = _mm256_setzero_ps();
            for (; dd + 8 <= head_dim; dd += 8)
              a = _mm256_fmadd_ps(_mm256_loadu_ps(qi + dd),
                                  _mm256_loadu_ps(kj + dd), a);
            acc = hsum256(a);
#else
            acc = 0.f;
#endif
            for (; dd < head_dim; ++dd) acc += qi[dd] * kj[dd];
            const float sv =
                acc * static_cast<float>(attn_scale) + mk[j];
            row[static_cast<size_t>(j)] = sv;
            mx = std::max(mx, sv);
          }
          float sum = 0.f;
          for (int64_t j = 0; j < seq; ++j) {
            const float e = std::exp(row[static_cast<size_t>(j)] - mx);
            row[static_cast<size_t>(j)] = e;
            sum += e;
          }
          const float inv = 1.0f / sum;
          float *o = out.data() + (b * seq + i) * hidden + hh * head_dim;
          std::memset(o, 0, sizeof(float) * static_cast<size_t>(head_dim));
          for (int64_t j = 0; j < seq; ++j) {
            const float w = row[static_cast<size_t>(j)] * inv;
            const float *vj = base + j * qkv_n + v_off;
            int64_t dd = 0;
#if defined(__AVX2__)
            const __m256 wv = _mm256_set1_ps(w);
            for (; dd + 8 <= head_dim; dd += 8)
              _mm256_storeu_ps(o + dd,
                               _mm256_fmadd_ps(wv, _mm256_loadu_ps(vj + dd),
                                               _mm256_loadu_ps(o + dd)));
#endif
            for (; dd < head_dim; ++dd) o[dd] += w * vj[dd];
          }
        }
      }
    });
    t_attn += now_s() - t0;
  }

  // A plain threaded fp32 host GEMM, for the two post-pool Dense heads only.
  // They run once per SEQUENCE, not once per token (tasks/0074 sec 4).
  void gemm_host(const float *a, int64_t M, int64_t K, const float *b,
                 int64_t N, float *c) const {
    par_rows(M, [&](int64_t r0, int64_t r1) {
      for (int64_t i = r0; i < r1; ++i) {
        const float *ar = a + i * K;
        float *cr = c + i * N;
        std::memset(cr, 0, sizeof(float) * static_cast<size_t>(N));
        for (int64_t k = 0; k < K; ++k) {
          const float av = ar[k];
          if (av == 0.f) continue;
          const float *br = b + k * N;
          int64_t j = 0;
#if defined(__AVX2__)
          const __m256 avv = _mm256_set1_ps(av);
          for (; j + 8 <= N; j += 8)
            _mm256_storeu_ps(cr + j, _mm256_fmadd_ps(avv, _mm256_loadu_ps(br + j),
                                                     _mm256_loadu_ps(cr + j)));
#endif
          for (; j < N; ++j) cr[j] += av * br[j];
        }
      }
    });
  }

  void ensure_tables() {
    if (!cos_g.empty()) return;
    cos_g.resize(static_cast<size_t>(seq * head_dim));
    sin_g.resize(cos_g.size());
    cos_l.resize(cos_g.size());
    sin_l.resize(cos_g.size());
    npue::gemma_rope_tables(seq, head_dim, rope_theta, cos_g.data(), sin_g.data());
    npue::gemma_rope_tables(seq, head_dim, rope_theta_local, cos_l.data(),
                            sin_l.data());
  }

  // Encode `texts` (padded/tiled by the caller to exactly `batch` entries).
  // Returns [batch][hidden], L2-normalized.
  // `index_base` and `n_real` exist only so a truncation error can name the
  // CALLER's input. This function is handed a group that the caller has
  // padded up to the tier by repeating its last real text, so rows at
  // `b >= n_real` are duplicates whose index does not exist upstream --
  // checking them would report a row number the caller cannot look up. A
  // duplicate that truncates is a copy of a real row that also truncates, and
  // the real one is checked first, so nothing escapes by being skipped here.
  // `tokens`, when given, accumulates the REAL texts' token counts -- the same
  // thing the BERT path's chunk() reports and the same field `usage.
  // prompt_tokens` needs (tasks/0115). Padding repeats are excluded, which is
  // what `n_real` already distinguishes for the truncation check.
  std::vector<float> encode_batch(const std::vector<std::string> &texts,
                                  const std::string &prefix,
                                  size_t index_base = 0,
                                  size_t n_real = static_cast<size_t>(-1),
                                  int64_t *tokens = nullptr) {
    if (static_cast<int64_t>(texts.size()) != batch)
      throw std::runtime_error("encode_batch given " +
                               std::to_string(texts.size()) +
                               " texts, tier is " + std::to_string(batch));
    ensure_tables();
    double t0 = now_s();
    ids.assign(static_cast<size_t>(rows), 0);
    mask.assign(static_cast<size_t>(rows), 0);
    for (int64_t b = 0; b < batch; ++b) {
      const npue::GemmaEncoded en =
          tok.encode(texts[static_cast<size_t>(b)], static_cast<int>(seq), prefix);
      if (static_cast<size_t>(b) < n_real) {
        check_truncation(en.truncated, en.n_tokens_full,
                         index_base + static_cast<size_t>(b), seq);
        if (tokens) *tokens += en.n_tokens;
      }
      for (int64_t s = 0; s < seq; ++s) {
        ids[static_cast<size_t>(b * seq + s)] = en.input_ids[static_cast<size_t>(s)];
        mask[static_cast<size_t>(b * seq + s)] =
            static_cast<uint8_t>(en.attention_mask[static_cast<size_t>(s)]);
      }
    }
    t_tok += now_s() - t0;

    x.assign(static_cast<size_t>(rows * hidden), 0.f);
    hbuf.resize(x.size());
    qkvbuf.resize(static_cast<size_t>(rows * qkv_n));
    ctx.resize(x.size());
    proj.resize(x.size());
    upbuf.resize(static_cast<size_t>(rows * 2 * inter));
    gatedbuf.resize(static_cast<size_t>(rows * inter));
    down.resize(x.size());
    add_mask.resize(static_cast<size_t>(rows));

    const float MASK_FILL = -3.4028235e38f;
    for (int64_t r = 0; r < rows; ++r)
      add_mask[static_cast<size_t>(r)] = mask[static_cast<size_t>(r)] ? 0.f : MASK_FILL;

    // embed: x = W[id] * sqrt(hidden)
    const float escale = static_cast<float>(std::sqrt(static_cast<double>(hidden)));
    par_rows(rows, [&](int64_t r0, int64_t r1) {
      for (int64_t r = r0; r < r1; ++r) {
        const float *wv = w_embed + static_cast<size_t>(ids[static_cast<size_t>(r)]) * hidden;
        float *dst = x.data() + r * hidden;
        for (int64_t c = 0; c < hidden; ++c) dst[c] = wv[c] * escale;
      }
    });

    for (int64_t L = 0; L < layers; ++L) {
      const LayerHost &l = lh[static_cast<size_t>(L)];
      const bool full = npue::gemma_is_full_attention_layer(L, swp);
      const float *cs_t = full ? cos_g.data() : cos_l.data();
      const float *sn_t = full ? sin_g.data() : sin_l.data();

      rms_norm(x.data(), hidden, hbuf.data(), hidden, rows, hidden, l.ln_in);
      gemm(is_qkv, hbuf.data(), hbuf.size(), s_qkv[L], b_qkv[L], qkvbuf,
           qkv_n, at(ws_qkv, L), at(as_qkv, L));

      // q_norm / k_norm: RMSNorm over head_dim, PER HEAD, strictly between the
      // projection and RoPE. Each (row, head) slice of head_dim floats is
      // contiguous inside the fused buffer even though consecutive rows are
      // qkv_n apart, so a strided call needs no repacking.
      for (int64_t hh = 0; hh < heads; ++hh)
        rms_norm(qkvbuf.data() + q_off + hh * head_dim, qkv_n,
                 qkvbuf.data() + q_off + hh * head_dim, qkv_n, rows, head_dim,
                 l.q_norm);
      rms_norm(qkvbuf.data() + k_off, qkv_n, qkvbuf.data() + k_off, qkv_n,
               rows, head_dim, l.k_norm);

      apply_rope(qkvbuf, cs_t, sn_t);
      attention(qkvbuf, ctx);

      gemm(is_ao, ctx.data(), ctx.size(), s_ao[L], b_ao[L], proj,
           hidden, at(ws_ao, L), at(as_ao, L));
      rms_norm(proj.data(), hidden, proj.data(), hidden, rows, hidden, l.ln_pa);
      par(x.size(), [&](size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; ++i) x[i] += proj[i];
      });

      rms_norm(x.data(), hidden, hbuf.data(), hidden, rows, hidden, l.ln_pf);
      const bool fuse_ffn = fuse_ffn_epilogue &&
                            d.info().a_elem_bytes == 1 && at(as_fd, L);
      FusedNext fn{at(as_fd, L), static_cast<int8_t *>(d.slot_ptr(0, slot_a)),
                   nullptr};
      if (fuse_ffn) {
        a_scale_next.resize(static_cast<size_t>(rows));
        fn.scale = a_scale_next.data();
      }
      // T37-BF16 (tasks/0108): the bf16/bfp16 analogue of `fn`/`fuse_ffn`
      // above -- no quantisation scale on this path, so no `at(as_fd, L)`
      // gate is needed.
      const bool fuse_ffn_bf16 = fuse_ffn_epilogue && d.info().a_elem_bytes == 2;
      FusedNextBf16 fn_bf16{static_cast<uint16_t *>(d.slot_ptr(0, slot_a))};
      gemm(is_fu, hbuf.data(), hbuf.size(), s_fu[L], b_fu[L], upbuf,
           2 * inter, at(ws_fu, L), at(as_fu, L), fuse_ffn ? &fn : nullptr,
           /*a_ready=*/false, fuse_ffn_bf16 ? &fn_bf16 : nullptr);
      if (fuse_ffn || fuse_ffn_bf16) {
        if (fuse_ffn) a_scale.swap(a_scale_next);
      } else {
        geglu(upbuf, gatedbuf);
      }
      gemm(is_fd, gatedbuf.data(), gatedbuf.size(), s_fd[L], b_fd[L], down,
           hidden, at(ws_fd, L), at(as_fd, L), nullptr,
           /*a_ready=*/fuse_ffn || fuse_ffn_bf16);
      rms_norm(down.data(), hidden, down.data(), hidden, rows, hidden, l.ln_pof);
      par(x.size(), [&](size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; ++i) x[i] += down[i];
      });
    }

    rms_norm(x.data(), hidden, x.data(), hidden, rows, hidden, w_norm);

    // masked mean pool, include_prompt=true
    std::vector<float> pooled(static_cast<size_t>(batch * hidden), 0.f);
    for (int64_t b = 0; b < batch; ++b) {
      double denom = 0.0;
      float *o = pooled.data() + b * hidden;
      for (int64_t s = 0; s < seq; ++s) {
        if (!mask[static_cast<size_t>(b * seq + s)]) continue;
        denom += 1.0;
        const float *row = x.data() + (b * seq + s) * hidden;
        for (int64_t c = 0; c < hidden; ++c) o[c] += row[c];
      }
      const float inv = static_cast<float>(1.0 / std::max(denom, 1e-9));
      for (int64_t c = 0; c < hidden; ++c) o[c] *= inv;
    }

    std::vector<float> d2(static_cast<size_t>(batch * dense_hidden));
    gemm_host(pooled.data(), batch, hidden, w_dense2, dense_hidden, d2.data());
    std::vector<float> out(static_cast<size_t>(batch * hidden));
    gemm_host(d2.data(), batch, dense_hidden, w_dense3, hidden, out.data());

    for (int64_t b = 0; b < batch; ++b) {
      float *o = out.data() + b * hidden;
      double nrm = 0.0;
      for (int64_t c = 0; c < hidden; ++c) nrm += static_cast<double>(o[c]) * o[c];
      const float inv = static_cast<float>(1.0 / std::max(std::sqrt(nrm), 1e-12));
      for (int64_t c = 0; c < hidden; ++c) o[c] *= inv;
    }
    return out;
  }
};
// Does this design set serve THIS model? Every op's (K, N) must match what the
// model's geometry implies -- not merely `hidden` appearing as some "K".
//
// The old predicate asked only the latter, and its own comment named precisely
// the danger it was failing to catch: "a design built for another width has the
// same filenames and loads fine -- it would simply compute the wrong thing."
// That was sound only because every model shipped so far has ffn == 4*hidden,
// which makes `hidden` determine all four shapes. It stops being sound the
// moment two models share a K set and differ in an N.
//
// nomic-embed-text-v1.5 is the first: its K set {768, 3072} is IDENTICAL to
// bge-base's, while its gated ffn_up is N=6144 against bge-base's N=3072. The
// old check accepts bge-base's design for nomic, and the runtime then
// dispatches a stream built for HALF the output width -- no error, no warning,
// the gate half silently lost. tasks/0069, thread T31.
//
// Matched against the `streams` array, which every design.json has carried
// since 0032, so this works unchanged on design sets exported long before the
// geometry keys existed -- no re-export needed to close the hole.
inline bool design_fits(const std::string &design_dir, int64_t hidden,
                 int64_t intermediate, bool gated_ffn, int64_t qkv_n = 0,
                 const std::string &want_layout = "",
                 const std::string &want_datapath = "") {
  if (hidden <= 0 || intermediate <= 0) return false;
  std::ifstream f(design_dir + "/gemm_rtp/design.json");
  if (!f) return false;
  std::stringstream b;
  b << f.rdbuf();
  const std::string js = b.str();
  const std::vector<StreamEntry> streams = parse_streams(js);
  if (streams.empty()) return false;

  // GEOMETRY IS NOT ENOUGH ONCE THERE IS MORE THAN ONE DATAPATH.
  // tasks/0080: an int8 container and a bf16 container of the same model have
  // identical (op, K, N) on every stream, so this function accepted a bf16
  // design for an int8 model and the encode died at stage time on the layout
  // hash. Failing closed, but selecting wrongly. The container knows its own
  // B layout -- pass it, and the choice becomes a fact about the DATA rather
  // than about which directory sorts first. Empty means "caller did not say",
  // which keeps every pre-0080 call site behaving exactly as before.
  if (!want_layout.empty()) {
    const size_t k = js.find("\"b_layout_hash\"");
    if (k == std::string::npos) return false;
    const size_t q1 = js.find('"', js.find(':', k) + 1);
    if (q1 == std::string::npos) return false;
    const size_t q2 = js.find('"', q1 + 1);
    if (q2 == std::string::npos) return false;
    if (js.substr(q1 + 1, q2 - q1 - 1) != want_layout) return false;
  }

  // THE MMAC DATAPATH (tasks/0104, T23). bfp16 is a SECOND thing want_layout
  // above cannot catch: it changes MMAC precision, not B's tiling, so a
  // bfp16 design and a plain-bf16 design at the SAME geometry carry the
  // SAME b_layout_hash (gemm_b_layout() only ever sees dtype="BF16"). Without
  // this check, a model adopted for one datapath and NOT the other -- exactly
  // bge-small, which shares MiniLM's hidden-384 geometry and FAILED the
  // bfp16 MTEB gate at -0.5010 (tasks/0103) -- would have two directories
  // fit it, and pick_artifacts()'s alphabetical tie-break would decide which
  // datapath it runs, silently. want_datapath is "bf16" or "bfp16"; empty
  // means "caller did not say" (every call site before this field existed,
  // and any explicit --artifacts override, which still wins over this
  // function entirely).
  if (!want_datapath.empty()) {
    const size_t k = js.find("\"emulate_bfp16\"");
    bool is_bfp16 = false;
    if (k != std::string::npos) {
      size_t p = js.find(':', k) + 1;
      while (p < js.size() && (js[p] == ' ' || js[p] == '\n' || js[p] == '\t'))
        ++p;
      is_bfp16 = js.compare(p, 4, "true") == 0;
    }
    if ((is_bfp16 ? "bfp16" : "bf16") != want_datapath) return false;
  }

  // tasks/0074: qkv's width was `3 * hidden`, which is true exactly when
  // num_key_value_heads == num_attention_heads. EmbeddingGemma-300M has ONE
  // KV head at head_dim 256, so its fused (and tile-padded) qkv is 1536 wide
  // against 3*768 = 2304 -- the check would have rejected its own correct
  // design, and in the other direction it is the T31 fail-open one field to
  // the left. 0 means "this container did not say", which is every BERT and
  // nomic container ever packed, and for those 3*hidden IS the answer.
  struct Want { const char *op; int64_t K, N; };
  const Want want[] = {
      {"qkv",      hidden,       qkv_n > 0 ? qkv_n : 3 * hidden},
      {"attn_out", hidden,       hidden},
      {"ffn_up",   hidden,       gated_ffn ? 2 * intermediate : intermediate},
      {"ffn_down", intermediate, hidden},
  };
  // Every op must be PRESENT, and EVERY occurrence of it must match -- a design
  // carrying one right batch tier and one wrong one does not fit.
  for (const Want &w : want) {
    bool seen = false;
    for (const StreamEntry &s : streams) {
      if (s.op != w.op) continue;
      seen = true;
      if (s.K != w.K || s.N != w.N) return false;
    }
    if (!seen) return false;
  }
  return true;
}

// The design set for a model, when --artifacts is not given.
//
// Three layouts have to work and none is privileged: a single-width release
// (<root>/gemm_rtp), a multi-width release (<root>/<set>/gemm_rtp) and the
// source tree (<root>/runtime/artifacts*/gemm_rtp). Each candidate is tested
// by whether its design actually serves this width, so the answer is a fact
// about the design rather than a naming convention.
inline std::string pick_artifacts(const std::string &root, int64_t hidden,
                           int64_t intermediate, bool gated_ffn,
                           int64_t qkv_n = 0,
                           const std::string &want_layout = "",
                           const std::string &want_datapath = "") {
  namespace fs = std::filesystem;
  if (hidden <= 0 || intermediate <= 0) return "";
  std::error_code ec;
  if (design_fits(root, hidden, intermediate, gated_ffn, qkv_n, want_layout,
                  want_datapath))
    return root;

  // Sorted, so the choice is reproducible rather than filesystem-order
  // dependent -- and never by mtime, which a JIT cache hit does not restamp
  // (CLAUDE.md trap 7c).
  std::vector<std::string> cands;
  for (const fs::path base : {fs::path(root), fs::path(root) / "runtime"})
    for (fs::directory_iterator it(base, ec), end; !ec && it != end;
         it.increment(ec))
      if (it->is_directory(ec)) cands.push_back(it->path().string());
  std::sort(cands.begin(), cands.end());

  // SEVERAL sets can serve one width and NOT be interchangeable in speed.
  // tasks/0080 added an int8 design set that narrows C to bf16; it carries the
  // same b_layout_hash as the int32-C set (C's width is not part of B's
  // layout), so both pass design_fits and the sort silently prefers whichever
  // sorts first -- which happened to be the slower one. A wrong pairing still
  // refuses at stage time on the layout hash, so this is not a correctness
  // hole; it is the "status line reports the intention, not the value" shape
  // that has cost this project time repeatedly (traps 7c, tasks/0042). Name
  // what was chosen and what else fitted, and let the caller pass --artifacts.
  std::vector<std::string> fits;
  for (const auto &c : cands)
    if (design_fits(c, hidden, intermediate, gated_ffn, qkv_n, want_layout,
                    want_datapath))
      fits.push_back(c);
  if (fits.empty()) return "";
  // TIE-BREAK ON EVIDENCE, NOT ON SPELLING. Among sets that fit equally,
  // rank by what their design.json can actually account for, then let
  // alphabetical order settle the rest (stable -- never mtime, trap 7c):
  //
  //  1. records `emulate_bfp16`. Both kinds are genuinely correct here
  //     (absent is treated as bf16 for selection), but picking the
  //     self-describing one turns "UNRECORDED" into a real answer for free.
  //  2. the larger `tg_depth` (T61-2, tasks/0152). A `--tg-depth 2` set is
  //     the SAME design -- same geometry, same datapath, same
  //     b_layout_hash, byte-identical final.xclbin and byte-identical
  //     embeddings -- with a software-pipelined runtime sequence, measured
  //     1.034-1.141x of array time on the seven catalogue models. So when
  //     both are installed, preferring it is free speed for a stated
  //     reason, rather than the alphabetical accident that would otherwise
  //     hand `artifacts_base_bfp16` the win over `artifacts_base_tgp`.
  //
  // This is the same failure the paragraph above describes (tasks/0080's
  // two int8 sets), arriving a second time from a different direction --
  // which is why it is a rank now rather than one more special case.
  auto rank = [](const std::string &p) {
    std::ifstream f(p + "/gemm_rtp/design.json");
    if (!f) return std::pair<int, long long>(0, 0LL);
    std::string js((std::istreambuf_iterator<char>(f)),
                   std::istreambuf_iterator<char>());
    const int described = js.find("\"emulate_bfp16\"") != std::string::npos;
    long long depth = 1;
    const size_t k = js.find("\"tg_depth\"");
    if (k != std::string::npos) {
      const size_t c = js.find(':', k);
      if (c != std::string::npos) depth = std::atoll(js.c_str() + c + 1);
    }
    return std::pair<int, long long>(described, depth);
  };
  std::stable_sort(fits.begin(), fits.end(),
                   [&](const std::string &a, const std::string &b) {
                     return rank(a) > rank(b);
                   });
  if (fits.size() > 1) {
    std::fprintf(stderr,
                 "note: %zu design sets serve hidden %lld; using %s\n",
                 fits.size(), (long long)hidden,
                 fs::path(fits[0]).filename().string().c_str());
    for (size_t i = 1; i < fits.size(); ++i)
      std::fprintf(stderr, "      also fits: %s  (--artifacts to choose)\n",
                   fs::path(fits[i]).filename().string().c_str());
  }
  return fits[0];
}


// A LANE IS A COPY OF LANE 0. This is the whole point of the function.
//
// Both lane loops used to build a FRESH encoder and then assign the ~20
// fields that belong to the design rather than to the lane. That shape has
// failed twice, and both failures were silent under a single-lane test:
//
//  * tasks/0078 -- the int8 scale vectors were not on the list. Lane 0 worked;
//    lanes 1+ dereferenced a null wscale, and the process segfaulted ONLY
//    under --pipeline.
//  * tasks/0037 -- the tier table was not on the list. A lane without it falls
//    back to the pre-0037 flat slot contract (0,1,2,3), which under a
//    16-stream export selects entirely the wrong shapes, measured as
//    `1-cos 1.0` on WHICHEVER CHUNK that lane happened to take. A
//    nondeterministic partial wrong answer: no single-lane gate can see it,
//    and the golden gate averages it away.
//
// Copying inverts the default. Everything the design gave lane 0 -- staged
// weight slots, bias pointers, int8 weight and SmoothQuant scales, the tier
// table, the LayerNorm host pointers, the host/fuse policy flags -- comes
// across because it is already there, and only what genuinely differs per
// lane is named below. A field added to the encoder tomorrow is copied
// without anyone remembering to add a line here, which is exactly the
// property the two failures above lacked.
//
// Safe for both encoders because neither does device work in its constructor:
// Encoder is an aggregate, and GemmaNpuEncoder's constructor only reads
// config, takes mmap pointers and builds a tokenizer -- all deterministic
// from the same container. And lanes are always made AFTER lane 0's
// stage_all() (so the design state exists to copy) and BEFORE any encode (so
// the per-call scratch vectors are still empty).
//
// The three overridden fields are the three that must NOT be shared:
// its own host thread pool, its own A and C buffers on the shared design, and
// the one mutex every lane serialises its NPU interaction on.
template <typename Enc>
std::unique_ptr<Enc> clone_lane(const Enc &lead, Pool &pool, npu::Design &d,
                                std::mutex &npu_mu) {
  auto e = std::make_unique<Enc>(lead);
  e->pool = &pool;
  e->npu_mu = &npu_mu;
  e->slot_a = d.stage_alloc(0, d.info().buffer_bytes[0]);
  e->slot_c = d.stage_alloc(2, d.info().buffer_bytes[2]);
  return e;
}


// ONE CONSTRUCTION PATH (tasks/0156 A3, T63).
//
// Everything between "here is a container and a design directory" and "here is
// an Encoder with its lanes staged and ready" used to live in main(), which
// meant a host application could link npue_embed and then have no way to build
// anything with it. The alternative -- a second copy of this sequence inside a
// facade -- is the one thing the plan forbids: two copies drift, and the gates
// only ever cover the one main() runs. So main() uses this too, and its diff
// for the change is deletions.
//
// The body below is main()'s own text, moved unchanged except for two
// `return 2` that became throws (main's top-level catch already returns 2, so
// the exit code is identical). That the text is unchanged is deliberate: every
// printf, refusal and ordering is load-bearing somewhere -- verify_tail.py and
// verify_embed_e2e.py both SCRAPE the `datapath` line, and tasks/0042's rule
// is that a status line reports the value rather than the intention -- so
// rewriting it while moving it would make a byte-identical output gate
// meaningless.
//
// MEMBER ORDER IS DESTRUCTION ORDER, REVERSED. The encoders hold pointers into
// the pools, the designs and the mutex, so those are declared FIRST and
// therefore destroyed LAST. Pool's destructor joins its threads; an Encoder
// outliving its Pool is a use-after-free at exit, which is the kind of bug
// that only shows on someone else's machine.
//
// A LIVE ShapeLease IS A PRECONDITION, not something this owns. The lease has
// to be taken before this runs, because choosing a design set needs the
// container's geometry -- so the lease is upstream of the design and cannot be
// nested inside it.
struct StackOptions {
  int threads = 1;
  int lanes = 0;                  // 0 or 1 = no pipelining; N = N lanes
  bool host_ln = false, host_sm = false, host_gelu = false;
  bool sim_c_bf16 = false;
  bool fuse_ffn_epilogue = true;
};

struct Stack {
  npue::File &model;
  bool unified = false;
  std::vector<StreamEntry> streams;
  std::unique_ptr<npu::Design> ud, ld_qkv, ld_ao, ld_fu, ld_fd, ld_gelu, ld_ln,
      ld_sm;
  npu::Design *p_qkv = nullptr, *p_ao = nullptr, *p_fu = nullptr,
              *p_fd = nullptr, *p_gelu = nullptr, *p_ln = nullptr,
              *p_sm = nullptr;
  int64_t rows = 0, batch = 0;
  size_t staged = 0;
  int n_lanes = 1;

  // Declared before the encoders; destroyed after them. See the note above.
  std::vector<std::unique_ptr<Pool>> pools;
  std::mutex npu_mutex;

  std::unique_ptr<Encoder> lead;
  std::vector<std::unique_ptr<Encoder>> lanes;
  std::vector<Encoder *> all;

  Stack(const Stack &) = delete;
  Stack &operator=(const Stack &) = delete;

  npu::Design &d_qkv() { return *p_qkv; }
  npu::Design &d_ao() { return *p_ao; }
  npu::Design &d_fu() { return *p_fu; }
  npu::Design &d_fd() { return *p_fd; }
  npu::Design &d_gelu() { return *p_gelu; }
  npu::Design &d_ln() { return *p_ln; }
  npu::Design &d_sm() { return *p_sm; }

  // The additive mask reaches EVERY lane, and that is the whole reason this
  // is a method rather than a public member. Setting it on lane 0 only would
  // leave lanes 1+ masking nothing, and the result would then depend on which
  // lane happened to take which chunk -- tasks/0037's failure shape exactly,
  // which no single-lane gate can see.
  void set_mask(const std::vector<float> &m) {
    for (Encoder *e : all) e->add_mask = m;
  }

  Stack(npu::Device &dev, npue::File &m, const std::string &art,
        const StackOptions &opt)
      : model(m) {
    const int nthreads = opt.threads;
    const int pipeline = opt.lanes;
    bool host_ln = opt.host_ln, host_sm = opt.host_sm,
         host_gelu = opt.host_gelu;
    const bool sim_c_bf16 = opt.sim_c_bf16;
    const bool no_fuse_ffn = !opt.fuse_ffn_epilogue;

    // Unified mode: art/gemm_rtp holds ONE xclbin whose four instruction
    // streams are the four GEMM shapes (tools/export_gemm_rtp.py). Every design
    // reference below binds to that one Design; the eltwise ops are forced onto
    // the host, and the encode runs in a single hw_context -- zero switches.
    const bool unified =
        std::ifstream(art + "/gemm_rtp/design.json").good();
    // `ud` and `ld_*` are MEMBERS of Stack. Declaring them here would
    // shadow them, and the designs would die with this constructor while
    // every Encoder still held a reference into them.
    std::vector<StreamEntry> streams;
    if (unified) {
      ud = std::make_unique<npu::Design>(dev, art + "/gemm_rtp");
      std::ifstream sj(art + "/gemm_rtp/design.json");
      std::stringstream sbuf;
      sbuf << sj.rdbuf();
      streams = parse_streams(sbuf.str());
      if (streams.empty()) {
        // A pre-0037 export: four streams, no tiers, the old flat names.
        ud->load_instr(art + "/gemm_rtp/insts_attn_out.bin");   // 1
        ud->load_instr(art + "/gemm_rtp/insts_ffn_up.bin");     // 2
        ud->load_instr(art + "/gemm_rtp/insts_ffn_down.bin");   // 3
        std::printf("  designs    ONE xclbin, 4 instruction streams, one "
                    "hw_context\n");
      } else {
        // Load in slot order and CHECK it -- a stream bound to the wrong slot
        // would compute a different shape with the right buffer sizes, which
        // is exactly the failure mode this project has hit five times.
        std::sort(streams.begin(), streams.end(),
                  [](const StreamEntry &a, const StreamEntry &b) {
                    return a.slot < b.slot;
                  });
        for (const auto &s : streams) {
          const size_t got = ud->load_instr(art + "/gemm_rtp/" + s.file);
          if (static_cast<int64_t>(got) != s.slot)
            throw std::runtime_error("stream " + s.file + " landed in slot " +
                                     std::to_string(got) + ", design.json says " +
                                     std::to_string(s.slot));
        }
        std::set<int64_t> tset;
        for (const auto &s : streams) tset.insert(s.batch);
        std::printf("  designs    ONE xclbin, %zu streams (%zu batch tiers), "
                    "one hw_context\n", streams.size(), tset.size());
      }
    } else {
      ld_qkv = std::make_unique<npu::Design>(dev, art + "/qkv");
      ld_ao = std::make_unique<npu::Design>(dev, art + "/attn_out");
      ld_fu = std::make_unique<npu::Design>(dev, art + "/ffn_up");
      ld_fd = std::make_unique<npu::Design>(dev, art + "/ffn_down");
      ld_gelu = std::make_unique<npu::Design>(dev, art + "/gelu");
      ld_ln = std::make_unique<npu::Design>(dev, art + "/layernorm");
      ld_sm = std::make_unique<npu::Design>(dev, art + "/softmax");
      std::printf("  designs    7 resident xclbins\n");
    }
    npu::Design &d_qkv = unified ? *ud : *ld_qkv;
    npu::Design &d_ao = unified ? *ud : *ld_ao;
    npu::Design &d_fu = unified ? *ud : *ld_fu;
    npu::Design &d_fd = unified ? *ud : *ld_fd;
    npu::Design &d_gelu = unified ? *ud : *ld_gelu;
    npu::Design &d_ln = unified ? *ud : *ld_ln;
    npu::Design &d_sm = unified ? *ud : *ld_sm;

    // WHICH DATAPATH WAS ACTUALLY SELECTED (tasks/0104), read off the loaded
    // design, never off a flag or the directory name that happened to be
    // picked -- "reports the intention, not the value" is a cost this project
    // has already paid twice (tasks/0042, 0081) for a_dtype/c_dtype; bfp16 gets
    // the same discipline from day one.
    if (!d_qkv.info().datapath_recorded)
      std::printf("  datapath   UNRECORDED (design predates tasks/0104), "
                  "C as %s\n",
                  d_qkv.info().c_elem_bytes == 2 ? "bf16" : "fp32");
    else
      std::printf("  datapath   %s MMAC, C as %s\n",
                  d_qkv.info().emulate_bfp16 ? "bfp16-emulated" : "bf16",
                  d_qkv.info().c_elem_bytes == 2 ? "bf16" : "fp32");

    // WHICH TOOLCHAIN BUILT THIS DESIGN (T39, tasks/0106) -- read off d_qkv,
    // same reasoning as the datapath line above (7-design and unified sets
    // both report their qkv design's provenance).
    if (!d_qkv.info().toolchain_recorded)
      std::printf("  toolchain  UNRECORDED (design predates tasks/0106)\n");
    else
      std::printf("  toolchain  mlir_aie %s, peano %s, mlir-aie HEAD %s\n",
                  d_qkv.info().mlir_aie_version.c_str(),
                  d_qkv.info().peano_version.c_str(),
                  d_qkv.info().mlir_aie_git_head.c_str());
    // WHICH BARRIER SCHEDULE THIS DESIGN'S INSTRUCTION STREAMS CARRY (T61-2,
    // tasks/0152). Two sets differing only here are identical in every other
    // field and in final.xclbin -- trap 7c -- so the schedule is stated.
    if (!d_qkv.info().schedule_recorded)
      std::printf("  sequence   UNRECORDED (design predates tasks/0152)\n");
    else
      std::printf("  sequence   %lld halves in flight, %lld row blocks per barrier\n",
                  (long long)d_qkv.info().tg_depth,
                  (long long)(d_qkv.info().tb_max_n_rows / 2));

    // Batch comes from the design, not from a constant here, so a mismatch is
    // impossible rather than merely unlikely.
    // The design says what sequence length it was built for; the container
    // says how many positions it can feed. set_design_seq checks the second
    // against the first rather than trusting either alone.
    if (d_qkv.info().seq <= 0)
      throw std::runtime_error(
          "this design set records no sequence length -- re-export it with "
          "tools/export_gemm_rtp.py, or add \"seq\": 64 to its design.json if "
          "you know it was built for seq 64");
    set_design_seq(d_qkv.info().seq);

    const int64_t rows = d_qkv.info().M, batch = rows / g_seq;
    if (rows % g_seq || batch < 1)
      throw std::runtime_error("design M=" + std::to_string(rows) +
                               " is not a whole number of seq-" +
                               std::to_string(g_seq) + " sequences");
    std::printf("  shape      batch %lld x seq %lld  (M = %lld)\n",
                (long long)batch, (long long)g_seq, (long long)rows);

    p_qkv = &d_qkv; p_ao = &d_ao; p_fu = &d_fu; p_fd = &d_fd;
    p_gelu = &d_gelu; p_ln = &d_ln; p_sm = &d_sm;
    this->rows = rows;
    this->batch = batch;
    this->unified = unified;
    this->streams = streams;
    // The Encoder needs a mask of the right shape at construction; every mode
    // overwrites it per chunk before dispatching, and the golden path replaces
    // it wholesale through set_mask() above.
    std::vector<float> mask(static_cast<size_t>(rows), 0.f);

    // Pipelining splits the thread budget: each lane gets its own pool, so no
    // lane can stall another's host work.
    if (pipeline > 1) {
      for (int l = 0; l < pipeline; ++l)
        pools.push_back(std::make_unique<Pool>(std::max(1, nthreads / pipeline)));
    } else {
      pools.push_back(std::make_unique<Pool>(nthreads));
    }
    Pool &pool = *pools[0];

    this->lead = std::make_unique<Encoder>(
        Encoder{model, d_qkv, d_ao, d_fu, d_fd, d_gelu, d_ln, d_sm, mask});
    Encoder &enc = *this->lead;
    enc.batch = batch;
    enc.rows = rows;
    enc.pool = &pool;
    if (unified) {
      // The unified artifact has no eltwise designs by construction.
      host_ln = host_sm = host_gelu = true;
      enc.unified = true;
      enc.is_qkv = 0;
      enc.is_ao = 1;
      enc.is_fu = 2;
      enc.is_fd = 3;
      if (!streams.empty()) {
        std::set<int64_t> tset;
        for (const auto &s : streams) tset.insert(s.batch);
        for (int64_t b : tset) {
          std::array<size_t, 4> slots{};
          bool complete = true;
          const char *ops[4] = {"qkv", "attn_out", "ffn_up", "ffn_down"};
          for (int k = 0; k < 4; ++k) {
            auto it = std::find_if(streams.begin(), streams.end(),
                                   [&](const StreamEntry &s) {
                                     return s.batch == b && s.op == ops[k];
                                   });
            if (it == streams.end()) { complete = false; break; }
            slots[k] = static_cast<size_t>(it->slot);
          }
          if (!complete) continue;         // a tier missing an op is not a tier
          enc.tiers.push_back(b);
          enc.tier_slots.push_back(slots);
        }
        enc.use_tier(batch);
        std::printf("  tiers      ");
        for (size_t i = 0; i < enc.tiers.size(); ++i)
          std::printf("%s%lld", i ? ", " : "", (long long)enc.tiers[i]);
        std::printf("  (requests are right-sized, not padded)\n");
      }
    }
    enc.host_ln = host_ln;
    enc.host_sm = host_sm;
    enc.host_gelu = host_gelu;
    enc.sim_c_bf16 = sim_c_bf16;
    enc.fuse_ffn_epilogue = !no_fuse_ffn;
    if (sim_c_bf16) {
      // Say so loudly, and refuse where it would mean nothing -- a status line
      // that reports the intention rather than the value is this project's
      // recurring fail-open (tasks/0042's `tile (64, 32)`).
      if (enc.qkv.info().a_elem_bytes != 1) {
        throw std::runtime_error(
            "--sim-c-bf16 is only meaningful on an int8 design (this one "
            "carries " + std::to_string(enc.qkv.info().a_elem_bytes) +
            "-byte operands)");
      }
      if (enc.qkv.info().c_elem_bytes == 2) {
        throw std::runtime_error(
            "--sim-c-bf16 simulates a narrowed-C design; this design "
            "already narrows C on the core, so the flag would only round a "
            "second time");
      }
      std::printf("  SIMULATION int32 C rounded to bf16 before dequantisation --\n"
                  "             prices a narrowed-C design; NOT a shipped path\n");
    }
    if (host_gelu)
      std::printf("  gelu       on the HOST (fp32) -- %lld fewer NPU dispatches\n",
                  (long long)g_layers);
    if (host_sm)
      std::printf("  softmax    on the HOST (fp32) -- %lld fewer NPU dispatches\n",
                  (long long)g_layers);
    if (host_ln)
      // One before the layer stack plus two per layer.
      std::printf("  layernorm  on the HOST (fp32) -- %lld fewer NPU dispatches\n",
                  (long long)(1 + 2 * g_layers));
    const size_t staged = enc.stage_all();
    // What the allocation mode actually bought, in addresses. Printed
    // because "1 MB padding gives large-page backing" is a mechanism
    // claim, and the alignment is the only visible part of it.
    std::printf("  bo-align   last data buffer aligned to %zu B%s\n",
                npu::last_bo_alignment(),
                npu::last_bo_alignment() >= (1u << 21) ? " (>= 2 MB)" : "");
    std::printf("  weights    %.2f MB staged on the device once, not per call\n",
                staged / 1e6);

    // `lanes`, `pools` and `npu_mutex` are MEMBERS -- see the
    // destruction-order note above.
    if (pipeline > 1) {
      if (!unified)
        throw std::runtime_error(
            "--pipeline requires the unified gemm_rtp artifact");
      enc.npu_mu = &npu_mutex;
      for (int l = 1; l < pipeline; ++l) {
        lanes.push_back(clone_lane(enc, *pools[l], d_qkv, npu_mutex));
        lanes.back()->use_tier(batch);
      }
      for (const auto &lp : lanes) {
        if (lp->tiers != enc.tiers || lp->tier_slots.size() != enc.tier_slots.size())
          throw std::runtime_error(
              "lane stream policy differs from lane 0 -- refusing to run, "
              "because the lanes would compute different things");
      }
      std::printf("  pipeline   %d concurrent encodes of %lld, one NPU mutex, "
                  "%d host threads per lane\n", pipeline, (long long)batch,
                  pools[0]->size());
    }

    this->staged = staged;
    this->n_lanes = pipeline > 1 ? pipeline : 1;
    all.push_back(this->lead.get());
    for (auto &lp : this->lanes) all.push_back(lp.get());
  }
};

// TEXT IN, VECTORS OUT. One service, used by --embed (batch, from a file)
// and --serve (an OpenAI-shaped HTTP endpoint). Sharing it is the point:
// the endpoint cannot drift from the thing the tests measure.
struct EmbedService {
  AnyTokenizer tok;
  const float *w_word, *w_pos, *w_typ;
  Encoder *lead;
  std::vector<Encoder *> all;
  int64_t fallback_batch;

  // Greedy against the tier ladder: 64 texts with tiers {4,16,32,128}
  // becomes 32+32, both exact, instead of one half-padded 128.
  std::vector<std::pair<int64_t, int64_t>> plan(int64_t n) const {
    std::vector<std::pair<int64_t, int64_t>> jobs;
    int64_t base = 0;
    while (base < n) {
      const int64_t left = n - base;
      int64_t take = lead->tiers.empty() ? std::min(fallback_batch, left) : 0;
      for (int64_t tr : lead->tiers)
        if (tr <= left && tr > take) take = tr;
      if (take == 0)
        take = lead->tiers.empty() ? left
                                   : std::min(left, lead->tiers.front());
      jobs.emplace_back(base, take);
      base += take;
    }
    return jobs;
  }

  // `prefix_text` is the literal text to prepend, "" for none. It is an
  // ARGUMENT rather than a member (tasks/0118) because --serve now takes the
  // prompt per request: holding it as state is what made one server able to
  // answer only one kind of query. Prepended to the RAW text before
  // tokenization, the same place tools/verify_embed_e2e.py does it, so the
  // two agree on what "applying a prefix" means.
  void chunk(Encoder &e, const std::vector<std::string> &texts,
             int64_t base, int64_t take, const std::string &prefix_text,
             std::vector<float> &out, int64_t *tokens) const {
    const size_t row_floats = static_cast<size_t>(g_seq) * g_hidden;
    const int64_t bt = e.use_tier(take);
    std::vector<float> buf(static_cast<size_t>(bt) * row_floats, 0.f);
    std::vector<float> cmask(static_cast<size_t>(bt) * g_seq, -1.0e30f);
    std::vector<float> cam(static_cast<size_t>(bt) * g_seq, 0.f);
    int64_t ntok = 0;
    for (int64_t b = 0; b < take; ++b) {
      const auto en = prefix_text.empty()
          ? tok.encode(texts[base + b], static_cast<int>(g_seq))
          : tok.encode(prefix_text + texts[base + b],
                      static_cast<int>(g_seq));
      // `base + b` is the caller's own index. Tiers are an implementation
      // detail of how this runtime batches, and naming a tier-local row
      // would send someone looking at the wrong text.
      check_truncation(en.truncated, en.n_tokens_full,
                       static_cast<size_t>(base + b), g_seq);
      ntok += en.n_tokens;
      for (int64_t s = 0; s < g_seq; ++s) {
        const int32_t id = en.input_ids[s];
        const float m = static_cast<float>(en.attention_mask[s]);
        cam[b * g_seq + s] = m;
        cmask[b * g_seq + s] = m > 0 ? 0.f : -1.0e30f;
        float *dst = buf.data() + (b * g_seq + s) * g_hidden;
        const float *wv = w_word + static_cast<size_t>(id) * g_hidden;
        const float *pv = w_pos + static_cast<size_t>(s) * g_hidden;
        for (int64_t c = 0; c < g_hidden; ++c)
          dst[c] = wv[c] + pv[c] + w_typ[c];
      }
    }
    e.add_mask = cmask;
    auto h = e.run(buf);
    pool_rows(h.data(), cam.data(), take, out.data() + base * g_hidden);
    if (tokens) *tokens += ntok;
  }

  std::vector<float> embed(const std::vector<std::string> &texts,
                           const std::string &prefix_text,
                           int64_t *tokens = nullptr) {
    std::vector<float> out(texts.size() * g_hidden, 0.f);
    const auto jobs = plan(static_cast<int64_t>(texts.size()));
    std::atomic<int64_t> tok_total{0};
    if (all.size() > 1 && jobs.size() > 1) {
      std::atomic<size_t> next{0};
      std::vector<std::thread> ts;
      // chunk() can throw -- npue::InputTooLong on a caller's bad input, or
      // anything e.run() raises on a device error -- and an exception that
      // escapes a std::thread's entry point calls std::terminate. This
      // branch had no handler, which was survivable only for as long as
      // nothing on the path threw. Capture the first, stop handing out work,
      // rethrow on the joining thread.
      std::mutex emu;
      std::exception_ptr first_err;
      std::atomic<bool> stop{false};
      auto worker = [&](Encoder *e) {
        for (size_t j = next++; j < jobs.size(); j = next++) {
          if (stop.load(std::memory_order_relaxed)) return;
          try {
            int64_t nt = 0;
            chunk(*e, texts, jobs[j].first, jobs[j].second, prefix_text,
                  out, &nt);
            tok_total += nt;
          } catch (...) {
            // First one wins. Which job reports first is a thread race, so
            // for a request with several oversized inputs the index named is
            // whichever lane got there -- deliberately not "the lowest",
            // because pretending to a determinism the scheduler does not
            // provide would be the worse lie. The caller has to fix all of
            // them regardless.
            std::lock_guard<std::mutex> lk(emu);
            if (!first_err) first_err = std::current_exception();
            stop.store(true, std::memory_order_relaxed);
            return;
          }
        }
      };
      for (size_t l = 1; l < all.size(); ++l)
        ts.emplace_back([&, l] { worker(all[l]); });
      worker(lead);
      for (auto &th : ts) th.join();
      if (first_err) std::rethrow_exception(first_err);
    } else {
      for (const auto &j : jobs) {
        int64_t nt = 0;
        chunk(*lead, texts, j.first, j.second, prefix_text, out, &nt);
        tok_total += nt;
      }
    }
    if (tokens) *tokens = tok_total.load();
    return out;
  }
};


// THE WHOLE THING, IN ONE OBJECT (tasks/0156 A3, T63).
//
// Everything above is what `npuembed.exe` uses. This is what ANOTHER HOST
// APPLICATION uses: container, shape lease, device, designs, pools, encoder,
// lanes and tokenizer, constructed in the one order that works, with a single
// call that takes text and returns vectors.
//
// It adds no new construction sequence -- it owns a Stack, which is the same
// code main() runs. That is the point of the whole extraction: two copies of
// that sequence would drift and the gates only cover the one main() runs.
//
// FOUR PROPERTIES THAT ARE NOT DECORATION.
//
// 1. MEMBER ORDER IS CONSTRUCTION ORDER. The container is opened first, the
//    shape lease taken on it second (nothing may read the geometry before it
//    is set), the device third, the Stack fourth. Reversed at destruction,
//    which is what keeps the lease alive until every Encoder is gone.
//
// 2. embed() IS NOT REENTRANT, and takes a mutex saying so. Two concurrent
//    calls would hand the same Encoder to two threads and corrupt its scratch
//    vectors. THROUGHPUT DOES NOT COME FROM CONCURRENT CALLS HERE -- it comes
//    from batching inside one call, across `lanes` lanes. A caller that wants
//    more work done should pass more texts, not call from more threads: a
//    single text through the smallest tier is measured 5.8x worse per text
//    (T40, tasks/0113).
//
// 3. THE PROMPT IS REFUSED RATHER THAN GUESSED. A model with a prompts table
//    that is handed no prompt name gets an exception naming the valid ones,
//    and a model with no table that is handed one gets an exception too. This
//    is tasks/0118's contract and it is the most dangerous seam in the whole
//    integration, because a wrongly-prefixed embedding is correctly shaped,
//    correctly normed and deterministic -- nothing downstream can tell it is
//    wrong.
//
// 4. NEVER PUT ONE AT NAMESPACE SCOPE. Pool's destructor joins threads, so a
//    static Embedder would join them after main() returns, under the Windows
//    loader lock. A member of whatever object serves requests is the right
//    home (a unique_ptr member of the host's request handler, say).
struct EmbedderOptions {
  std::string npue_path;        // the .npue container
  std::string artifacts_dir;    // the design set; REQUIRED, never searched for
  int threads = 1;
  int lanes = 1;                // concurrent encode lanes inside one call
  bool fuse_ffn_epilogue = true;
};

class Embedder {
 public:
  explicit Embedder(const EmbedderOptions &o)
      : model_(o.npue_path),
        lease_(model_),
        stack_(dev_, model_, o.artifacts_dir, stack_opts(o)),
        svc_{load_tokenizer(model_, o.npue_path),
             model_.raw("embeddings.word").as<float>(),
             model_.raw("embeddings.position").as<float>(),
             model_.raw("embeddings.token_type").as<float>(),
             stack_.lead.get(), {}, stack_.batch} {
    set_model_name(std::filesystem::path(o.npue_path).stem().string());
    svc_.all = stack_.all;
  }

  // `prompt_name` is a key in the container's own prompts table, or "" for a
  // model that has none. See property 3 above for why neither side of that is
  // allowed to be guessed.
  std::vector<float> embed(const std::vector<std::string> &texts,
                           const std::string &prompt_name = std::string(),
                           int64_t *tokens = nullptr) {
    const std::string prefix = resolve_prompt(prompt_name);
    std::lock_guard<std::mutex> lk(call_mu_);
    return svc_.embed(texts, prefix, tokens);
  }

  int64_t hidden() const { return g_hidden; }
  int64_t seq() const { return g_seq; }
  int64_t batch() const { return stack_.batch; }
  int lanes() const { return stack_.n_lanes; }
  size_t vocab_size() const { return svc_.tok.vocab_size(); }
  std::vector<std::string> prompt_names() const { return prompt_names_sorted(); }
  const std::string &name() const { return model_name(); }
  const std::string &source_repo() const { return npue::enc::source_repo(); }
  // What the DESIGN says it is, for a host that wants to report it. Read off
  // the loaded design rather than inferred from the directory name -- the
  // rule tasks/0042 made and T64 needed.
  std::string datapath() const {
    const auto &i = stack_.p_qkv->info();
    if (!i.datapath_recorded) return "UNRECORDED";
    return std::string(i.emulate_bfp16 ? "bfp16-emulated MMAC" : "bf16 MMAC") +
           (i.c_elem_bytes == 2 ? ", C as bf16" : ", C as fp32");
  }

  Embedder(const Embedder &) = delete;
  Embedder &operator=(const Embedder &) = delete;

 private:
  static StackOptions stack_opts(const EmbedderOptions &o) {
    StackOptions s;
    s.threads = o.threads;
    s.lanes = o.lanes;
    s.fuse_ffn_epilogue = o.fuse_ffn_epilogue;
    return s;
  }

  std::string resolve_prompt(const std::string &name) const {
    const auto &table = prompts();
    if (table.empty()) {
      if (!name.empty())
        throw std::runtime_error(
            "this model has no task prompts, but prompt name '" + name +
            "' was given. Refusing rather than ignoring it: a caller that "
            "thinks it applied a prefix and did not gets a correctly shaped, "
            "correctly normed vector of the wrong thing.");
      return std::string();
    }
    if (name.empty())
      throw std::runtime_error(
          "this model has task prompts and one must be named: pass one of [" +
          join_names(prompt_names_sorted()) +
          "], or \"\" for no prompt at all. Refusing to pick one -- a "
          "wrongly-prefixed embedding is correctly shaped and correctly "
          "normed, so nothing downstream can tell that the answer is wrong.");
    const auto it = table.find(name);
    if (it == table.end())
      throw std::runtime_error(
          "unknown prompt name '" + name + "' -- this model offers [" +
          join_names(prompt_names_sorted()) + "]");
    return it->second;
  }

  // Construction order. Do not reorder: see property 1 above.
  npue::File model_;
  ShapeLease lease_;
  npu::Device dev_;
  Stack stack_;
  EmbedService svc_;
  std::mutex call_mu_;
};

}  // namespace enc
}  // namespace npue
