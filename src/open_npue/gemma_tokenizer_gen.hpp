//===- gemma_tokenizer_gen.hpp -------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- C++ port of tools/gen_gemma_tokenizer_table.py.
// SPDX-License-Identifier: MIT
//
// Generates the binary SentencePiece-BPE table
// `runtime/src/tokenizer_gemma.cpp` reads at load time
// (`GEMATOK1` magic, see that Python script's docstring for the exact
// format and the empirically-confirmed tokenizer pipeline: SentencePiece
// **BPE**, not Unigram -- `model.type == "BPE"` in the real
// tokenizer.json, checked here, not assumed).
//
// WHY THIS EXISTS IN C++ AT ALL. `runtime/src/npue_pack.cpp`'s
// `prepare_model_gemma()` used to only ever READ an already-produced
// `gemma_tokenizer.bin` from disk -- nothing in the shipped C++ build could
// generate one. A fresh clone that fetches EmbeddingGemma via
// `npuembeddings.exe embed ...` therefore packed a `.npue` with no
// `tokenizer.gemma_table` tensor, and the first real encode threw. This is
// the C++ side of that gap (tasks/0067); see gemma_tokenizer_gen.cpp for
// the port and json_min.hpp for the JSON parser it needed and this project
// did not have (the existing config.json reader in npue_pack.cpp is
// intentionally too primitive for tokenizer.json's nesting and 262k/515k
// entry counts).
//
// Every validation the Python generator performs is ported faithfully and
// throws std::runtime_error (fail closed, CLAUDE.md) rather than silently
// producing a wrong table if a future checkpoint revision violates an
// assumption this format depends on.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace npue {

// Reads `tokenizer_json_path` (the checkpoint's tokenizer.json) and
// `sbert_config_path` (its config_sentence_transformers.json, for the task
// -prefix table) and returns the GEMATOK1 binary table bytes, in memory --
// byte-identical to tools/gen_gemma_tokenizer_table.py's output for the
// same two input files (verified in tasks/0067's TASK.md).
std::vector<uint8_t> generate_gemma_tokenizer_table(
    const std::string &tokenizer_json_path,
    const std::string &sbert_config_path);

}  // namespace npue
