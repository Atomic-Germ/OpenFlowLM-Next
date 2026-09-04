//===- npue_pack.hpp ----------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- build a .npue from an upstream checkpoint, without Python.
// SPDX-License-Identifier: MIT
//
// The release does not ship the model. These weights belong to
// sentence-transformers/all-MiniLM-L6-v2, and a user is better served by
// fetching them from the canonical source with a checksum to verify than by
// trusting a 66 MB blob in someone's zip file. This is what makes that
// practical: two downloads and one command, no Python, no toolchain.
//
// The result must be BYTE-IDENTICAL to tools/pack_npue.py's output. Two
// implementations of one binary layout is a real risk -- a disagreement would
// mean correctly-sized weights in the wrong order, which no size check
// catches -- so it is verified rather than assumed
// (tools/verify_pack_parity.py).

#pragma once

#include <cstdint>
#include <string>

namespace npue {

// `layout_json` and `layout_hash` describe the pre-tiled B layout and must be
// exactly what the compiled designs expect; the runtime compares the hash
// before it will dispatch, so a mismatch fails loudly instead of producing
// plausible garbage.
// The canonical B-layout descriptor and its hash, built in ONE place.
//
// tools/npue.py's docstring records why: every hand-written copy of this dict
// is a chance for two sides to drift, and the drift is invisible -- the hash
// changes, the bytes do not, and the check meant to catch wrong layouts starts
// reporting a mismatch that is not one. main.cpp used to carry both the JSON
// and a FROZEN hash for tile_n = 48, which made a second tile size
// unexpressible without editing the packer.
//
// `json` preserves tools/npue.py's INSERTION order (the bytes that go in the
// file); `hash` is over the key-SORTED form, which is what npue.py hashes.
struct Layout {
  std::string json;
  std::string hash;
};
Layout gemm_b_layout(int64_t tile_k, int64_t tile_n, int64_t mac_s = 8,
                     int64_t mac_t = 8);

// The one C++ SHA-256. Exposed so the downloader (src/hub.cpp) verifies a
// checkpoint with exactly the implementation that records `source_sha256`
// into the container. Streams the file; safe on the 438 MB checkpoints.
std::string sha256_file(const std::string &path);

void prepare_model(const std::string &safetensors, const std::string &vocab,
                   const std::string &config_json_path,
                   const std::string &pooling,
                   const std::string &source_repo,
                   const std::string &out, const std::string &source_sha,
                   const std::string &layout_json,
                   const std::string &layout_hash,
                   int64_t tile_k, int64_t tile_n, int64_t max_seq,
                   void (*log)(const std::string &) = nullptr);

// arch=1 (EmbeddingGemma / Gemma3 MQA+RoPE+GeGLU) mirror of
// tools/pack_npue.py's pack_gemma(). `model_dir` must hold
// model.safetensors, config.json, 2_Dense/model.safetensors,
// 3_Dense/model.safetensors and (optionally) gemma_tokenizer.bin.
// `source_repo` is resolved by the caller exactly as for the BERT path
// (CHECKPOINT.json or --source-repo), so both packers agree on it.
//
// tasks/0074: the four per-layer GEMM operands are now PRE-TILED bf16 under
// BERT's tensor names, with Q|K|V fused and zero-padded to a legal tile width,
// so the array runs 97.7% of this model's MACs. `host_only` restores what
// tasks/0064-0065 shipped -- plain F32 row-major operands for the CPU-only
// npue::GemmaEncoder -- which is now the correctness CONTROL rather than the
// product. Both packers must keep producing byte-identical output for BOTH
// modes (tools/verify_pack_parity.py); tasks/0065 established that property
// and it is not allowed to lapse.
void prepare_model_gemma(const std::string &model_dir, const std::string &out,
                         const std::string &source_repo,
                         void (*log)(const std::string &) = nullptr,
                         int64_t tile_k = 64, int64_t tile_n = 48,
                         bool host_only = false);

// arch=2 (nomic-embed-text-v1.5 / RoPE + gated SwiGLU) mirror of
// tools/pack_npue.py's pack_nomic() (tasks/0069, tasks/0070, tasks/0071).
// Emits the SAME tensor names and SAME emission order as prepare_model()
// above, so Encoder::run()'s existing NPU dispatch path works unchanged --
// with three departures from BERT: no absolute position table (RoPE
// instead; zero-filled placeholder of the right shape so the unconditional
// "embeddings.position" read stays untouched), no biases anywhere
// (zero-filled placeholders, same rationale), and a gated SwiGLU `ffn_up`
// that fuses fc11 (untouched "up") | fc12 (SiLU "gate") along N -- one GEMM,
// not two. Every GEMM operand IS pre-tiled here (unlike Gemma): nomic has a
// real NPU design, so this packer's output is meant to be dispatched, not
// just loaded.
//
// `pooling` and `source_repo` are resolved by the CALLER exactly as for
// prepare_model() above (same 1_Pooling/config.json and CHECKPOINT.json /
// --source-repo sources), so the BERT-family and nomic packers cannot
// disagree about either.
void prepare_model_nomic(const std::string &model_dir,
                         const std::string &pooling,
                         const std::string &source_repo,
                         const std::string &out,
                         const std::string &layout_json,
                         const std::string &layout_hash,
                         int64_t tile_k, int64_t tile_n, int64_t max_seq,
                         void (*log)(const std::string &) = nullptr);

// arch=3 (gte-multilingual-base / NTK RoPE + gated GeGLU, model_type "new")
// mirror of tools/pack_npue.py's pack_gte() (tasks/0135, tasks/0138). Same
// tensor names and emission order as arch=0/2 -- including the
// `ln.weight -> tokenizer -> ln.bias` interleaving that is load-bearing for
// byte parity -- with pack_gte()'s departures from the nomic shape it
// otherwise mirrors: REAL biases on qkv / attn_out / ffn_down (the Q third
// of the qkv bias scale-folded along with the Q weight block -- exact, RoPE
// is linear), a gated FFN that arrives ALREADY FUSED upstream
// (up_gate_proj, up columns first), exact-erf GELU recorded as
// config["activation"], the NTK inv_freq set carried as DATA
// (config["rope_inv_freq"], 32 float32 values the runtime must read -- a
// consumer deriving frequencies from rope_theta alone is wrong by 1.9e-02
// relfro at layer 0, tasks/0134), and the XLMRTOK1 Unigram tokenizer blob
// stored whole as "tokenizer.xlmr_table". The blob is read from the cached
// models/<dir>/xlmr_tokenizer.bin when present and otherwise generated here
// in C++ (generate_xlmr_tokenizer_table(), byte-identical to the Python
// generator per tasks/0133) and written back to that cache path -- the same
// self-sufficiency prepare_model_gemma() has for its own table.
//
// `pooling` and `source_repo` are resolved by the CALLER exactly as for the
// other packers above. The output must be byte-identical to pack_gte()'s
// for the same inputs -- tools/verify_pack_parity.py's standing gate, held
// for this arch in tasks/0138.
void prepare_model_gte(const std::string &model_dir,
                       const std::string &pooling,
                       const std::string &source_repo,
                       const std::string &out,
                       const std::string &layout_json,
                       const std::string &layout_hash,
                       int64_t tile_k, int64_t tile_n, int64_t max_seq,
                       void (*log)(const std::string &) = nullptr);

// ONE CALL THAT PACKS A CHECKPOINT (tasks/0156, T63).
//
// The four prepare_model_* entry points above each need the right arguments,
// and choosing them is a real decision: which architecture the checkpoint is,
// which tile width is legal for its widths, which pooling its own
// 1_Pooling/config.json declares, which repository the weights came from, and
// which max_seq the container should record. That decision lived in ~200 lines
// inside `--prepare-model` in main.cpp, which means it lived somewhere a host
// application cannot reach.
//
// WHY THAT MATTERED ENOUGH TO MOVE. OpenFlowLM-Next fetches the model author's
// OWN HuggingFace checkpoint and packs the container locally -- nothing is
// re-hosted, so the weights a user gets are the author's bytes with the
// author's hash. The alternative to this function was a second copy of the
// decision over there, and a second copy of a decision that has to agree
// EXACTLY is the one thing the sync design forbids: the moment the two differ,
// this repository's byte-identity gate stops being evidence about the fork's
// containers.
//
// EVERY DISPATCH IS ON THE CHECKPOINT'S OWN config.json, never on a directory
// name. `model_type` is the field; gemma3_text, nomic_bert and "new" route to
// their own packers and everything else is BERT. tools/pack_npue.py's main()
// makes the same decision from the same field, and tools/verify_pack_parity.py
// is the standing gate that they agree byte for byte.
//
// It REFUSES rather than guessing, in three places, and each refusal is a
// statement someone would otherwise have had to make up:
//   * no 1_Pooling/config.json -- there is no way to tell whether the
//     checkpoint pools by mean or by CLS, and the two are different models.
//   * no source_repo and no CHECKPOINT.json -- a container that misattributes
//     its own weights is a licensing statement.
//   * an architecture whose model_type is recognised but whose packer this
//     build does not carry -- not applicable today, and the BERT fallback is
//     deliberately the last branch rather than a catch-all guess.
struct PrepareOptions {
  // The checkpoint directory: model.safetensors, config.json, and (for the
  // BERT family) vocab.txt and 1_Pooling/config.json, as downloaded.
  std::string checkpoint_dir;
  // Where to write. Empty means <checkpoint_dir>/<directory name>.npue --
  // named after the checkpoint, not after a model, which was a literal until
  // a second model made it visible.
  std::string out_path;
  // Which repository the weights came from. Empty means read it out of
  // CHECKPOINT.json's repo_id, and refuse if there is none.
  std::string source_repo;
  // Tile size is A PROPERTY OF THE MODEL, not a constant: the design asserts
  // N % (tile_n * n_cols) == 0, and bge-large's N in {1024, 3072, 4096} makes
  // 48 illegal -- its legal set is {8, 16, 32, 64} and 64 does not fit L1
  // (65,536 B against the 63 KB budget), so it must be 32.
  int64_t tile_k = 64;
  int64_t tile_n = 48;
  // arch=1 escape hatch (tasks/0074). The default is the production geometry,
  // so a cold clone self-produces a container the ARRAY can run -- before that
  // default existed it self-produced a host-only container and quietly ran at
  // 0.2 seq/s.
  bool gemma_host_only = false;
  void (*log)(const std::string &) = nullptr;
};

// Returns the path written. Throws on anything it will not guess.
std::string prepare_model_auto(const PrepareOptions &opt);

}  // namespace npue
