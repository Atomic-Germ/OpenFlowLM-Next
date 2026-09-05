//===- json_min.hpp -----------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- a minimal, dependency-free JSON DOM parser.
// SPDX-License-Identifier: MIT
//
// WHY THIS EXISTS. `runtime/src/npue_pack.cpp`'s BERT-family `config.json`
// reader (`cfg_int`/`cfg_str`/`cfg_raw`) gets away with ad-hoc "find this key,
// read the raw text after the colon" scanning because those files are flat,
// small, single-object JSON. EmbeddingGemma's `tokenizer.json` is not: it is
// ~33 MB, deeply nested (`model.vocab` alone is a 262,144-entry object, and
// `model.merges` a ~515k-element array of 2-element arrays), and the C++
// tokenizer-table generator (`gemma_tokenizer_gen.hpp`) needs to walk all of
// it faithfully to reproduce `tools/gen_gemma_tokenizer_table.py`'s output
// byte for byte. That needs a real parser, not more scanning.
//
// CLAUDE.md forbids vendoring a third-party dependency (no nlohmann/json,
// no rapidjson) -- `runtime/CMakeLists.txt` has zero third-party C++ deps
// beyond XRT/ws2_32/winhttp, and this keeps it that way. This file is
// self-contained: no external headers beyond the standard library.
//
// SCOPE. A straightforward recursive-descent DOM parser. It supports every
// JSON construct (`object`, `array`, `string` with all six named escapes
// plus `\/` and `\uXXXX` including surrogate pairs, `number`, `true`,
// `false`, `null`) because the input is untrusted-shape (a checkpoint
// revision could restructure this file) and a partial parser would fail in
// a confusing place rather than a clear one. It does not preserve object key
// order (objects are `std::unordered_map`, since `model.vocab`'s 262k
// entries are looked up by key and by id, never enumerated in file order);
// arrays DO preserve element order (`std::vector`), which is exactly what
// `model.merges`' rank ordering needs.
//
// PERFORMANCE. This runs once, at pack time, never per-request (CLAUDE.md
// rule 5: build-time only). Still, a naive parser that builds strings one
// character at a time would be minutes slow on 262k+515k entries, so the
// string scanner below fast-paths the (overwhelmingly common) run of plain
// bytes between escapes with one `append(ptr, len)` rather than a
// character-at-a-time loop.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace npue {
namespace json {

enum class Type { Null, Bool, Number, String, Array, Object };

// A tagged-union JSON value. Only the member matching `type` is meaningful;
// the others are default-constructed and unused -- simplicity over the few
// bytes a real union would save, and this is a build-time tool, not a
// kernel.
class Value {
public:
  Type type = Type::Null;
  bool bool_v = false;
  double num_v = 0.0;
  std::string str_v;
  std::vector<Value> arr_v;
  std::unordered_map<std::string, Value> obj_v;

  Value() = default;

  bool is_null() const { return type == Type::Null; }
  bool is_object() const { return type == Type::Object; }
  bool is_array() const { return type == Type::Array; }
  bool is_string() const { return type == Type::String; }
  bool is_number() const { return type == Type::Number; }
  bool is_bool() const { return type == Type::Bool; }

  // Throws std::runtime_error with a clear message if the value is not the
  // requested shape -- fail closed, per CLAUDE.md, rather than returning a
  // silently-wrong default.
  const std::string &as_string() const;
  double as_number() const;
  bool as_bool() const;
  const std::vector<Value> &as_array() const;
  const std::unordered_map<std::string, Value> &as_object() const;

  // Object access. `at()` throws if this is not an object or the key is
  // absent; `find()` returns nullptr instead of throwing (mirrors
  // std::map::find's caller-checks-first style, used where a key is
  // legitimately optional, e.g. a checkpoint field the generator only
  // cross-checks when present).
  const Value &at(const std::string &key) const;
  const Value *find(const std::string &key) const;
  bool contains(const std::string &key) const;
};

// Parses `text` as a complete JSON document. Throws std::runtime_error with
// a byte-offset-annotated message on any malformed input, including trailing
// non-whitespace data after the top-level value.
Value parse(const std::string &text);

}  // namespace json
}  // namespace npue
