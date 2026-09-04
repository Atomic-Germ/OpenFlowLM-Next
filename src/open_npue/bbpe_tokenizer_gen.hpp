//===- bbpe_tokenizer_gen.hpp --------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- C++ port of tools/gen_bbpe_tokenizer_table.py.
// SPDX-License-Identifier: MIT
//
// Generates the `BBPETOK1` binary that runtime/src/tokenizer_bbpe.cpp reads at
// load time. See that Python script's docstring for the format and for the
// pipeline it was read out of.
//
// WHY THIS EXISTS IN C++ AT ALL. The same reason as gemma_tokenizer_gen.hpp
// (tasks/0067) and xlmr_tokenizer_gen.hpp (tasks/0133): the shipped C++ build
// must be able to pack a freshly fetched checkpoint into a `.npue` with no
// Python in the process (CLAUDE.md rule 5). A runtime that can only READ an
// already-generated table leaves a fresh clone unable to pack the model at
// all.
//
// Every validation the Python generator performs is ported faithfully and in
// the same order, and throws std::runtime_error -- fail closed -- rather than
// producing a table that is quietly wrong for a checkpoint revision that
// violates an assumption. The two are held byte-identical over the real
// tokenizer.json files (tasks/0153), which is what makes the duplication safe.
//
// NOTE what is NOT duplicated: the Unicode character-class and NFC tables.
// Those live in the generated bbpe_unicode_tables.hpp, are model-independent,
// and are produced by tools/gen_bbpe_unicode_tables.py -- which needs
// HuggingFace's own splitter as an oracle and therefore cannot be ported. It
// does not need to be: a fresh clone gets the header from the repository, the
// same way it gets bert_unicode_tables.hpp.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace npue {

// Reads `tokenizer_json_path` and returns the BBPETOK1 table bytes in memory,
// byte-identical to tools/gen_bbpe_tokenizer_table.py's output for the same
// input file.
std::vector<uint8_t> generate_bbpe_tokenizer_table(
    const std::string &tokenizer_json_path);

}  // namespace npue
