//===- xlmr_tokenizer_gen.hpp --------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- C++ port of tools/gen_xlmr_tokenizer_table.py.
// SPDX-License-Identifier: MIT
//
// Generates the binary SentencePiece-Unigram table
// `runtime/src/tokenizer_xlmr.cpp` reads at load time (`XLMRTOK1` magic;
// see that Python script's docstring for the exact format and the
// empirically-confirmed tokenizer pipeline: SentencePiece **Unigram**, not
// BPE -- `model.type == "Unigram"` in the real tokenizer.json, checked
// here, not assumed).
//
// WHY THIS EXISTS IN C++ AT ALL. Same reason as gemma_tokenizer_gen.hpp,
// whose lesson this port inherits (tasks/0067): the shipped C++ build must
// be able to pack a fresh checkpoint fetch into a `.npue` without Python in
// the process (CLAUDE.md rule 5) -- a runtime that can only READ an
// already-generated xlmr_tokenizer.bin leaves a fresh clone unable to pack
// gte-multilingual-base at all. The generator therefore exists twice, and
// tasks/0133 holds the two byte-identical over the real tokenizer.json
// (sha256 compare), which is what makes the duplication safe.
//
// Every validation the Python generator performs is ported faithfully, in
// the same order, and throws std::runtime_error (fail closed, CLAUDE.md)
// rather than silently producing a wrong table if a future checkpoint
// revision violates an assumption this format depends on.
//
// f64 NOTE: the scores are parsed by json_min's strtod (correctly-rounded
// decimal-to-double on this toolchain, like Python's float()) and written
// as raw f64 -- 65,856 of the 250,002 log-probs do not survive f32
// (tasks/0127), so any float anywhere on this path would break the
// byte-identity this port is held to.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace npue {

// Reads `tokenizer_json_path` (the checkpoint's tokenizer.json, 17 MB) and
// returns the XLMRTOK1 binary table bytes, in memory -- byte-identical to
// tools/gen_xlmr_tokenizer_table.py's output for the same input file
// (verified in tasks/0133's TASK.md).
std::vector<uint8_t> generate_xlmr_tokenizer_table(
    const std::string &tokenizer_json_path);

}  // namespace npue
