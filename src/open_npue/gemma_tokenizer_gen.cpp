//===- gemma_tokenizer_gen.cpp -------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- C++ port of tools/gen_gemma_tokenizer_table.py.
// SPDX-License-Identifier: MIT
// See gemma_tokenizer_gen.hpp for why this exists.
//
// Every check below is a direct port of a `raise SystemExit(...)` guard in
// the Python script, kept in the same order and against the same fields, so
// the two can be diffed against each other line by line. Comments repeat
// only what is not obvious from the port itself.

#include "gemma_tokenizer_gen.hpp"

#include "json_min.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace npue {
namespace {

std::string slurp_text(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot open '" + path + "'");
  std::ostringstream ss;
  ss << f.rdbuf();
  if (!f && !f.eof())
    throw std::runtime_error("error reading '" + path + "'");
  return ss.str();
}

// JSON "truthiness", mirroring Python's `if model.get(key):` -- used for the
// continuing_subword_prefix/end_of_word_suffix/byte_fallback checks, which
// the Python script tests by truthiness rather than an explicit type check.
bool truthy(const json::Value *v) {
  if (!v || v->is_null()) return false;
  switch (v->type) {
    case json::Type::Bool: return v->bool_v;
    case json::Type::Number: return v->num_v != 0.0;
    case json::Type::String: return !v->str_v.empty();
    case json::Type::Array: return !v->arr_v.empty();
    case json::Type::Object: return !v->obj_v.empty();
    default: return false;
  }
}

int64_t json_int(const json::Value &v, const std::string &what) {
  if (!v.is_number())
    throw std::runtime_error("expected a number for " + what);
  const double d = v.num_v;
  const int64_t i = static_cast<int64_t>(d);
  if (static_cast<double>(i) != d)
    throw std::runtime_error("expected an integer for " + what);
  return i;
}

void put_u16(std::vector<uint8_t> &out, uint16_t v) {
  out.push_back(static_cast<uint8_t>(v & 0xFF));
  out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}
void put_u32(std::vector<uint8_t> &out, uint32_t v) {
  out.push_back(static_cast<uint8_t>(v & 0xFF));
  out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
  out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
  out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}
// u16-length-prefixed UTF-8 bytes -- the vocab/merge-name/prefix encoding
// used throughout the binary format (struct.pack("<H", len(b)) + b).
void put_str16(std::vector<uint8_t> &out, const std::string &s) {
  if (s.size() > 0xFFFF)
    throw std::runtime_error("string exceeds 65535 bytes: '" +
                             s.substr(0, 50) + "...'");
  put_u16(out, static_cast<uint16_t>(s.size()));
  out.insert(out.end(), s.begin(), s.end());
}

}  // namespace

std::vector<uint8_t> generate_gemma_tokenizer_table(
    const std::string &tokenizer_json_path,
    const std::string &sbert_config_path) {
  const json::Value tok = json::parse(slurp_text(tokenizer_json_path));

  const json::Value &model = tok.at("model");

  {
    const json::Value *type_v = model.find("type");
    if (!type_v || !type_v->is_string() || type_v->str_v != "BPE")
      throw std::runtime_error(
          "expected model.type == 'BPE' -- this generator implements "
          "SentencePiece BPE specifically, not Unigram; re-check " +
          tokenizer_json_path + " before proceeding");
  }
  if (!truthy(model.find("byte_fallback")))
    throw std::runtime_error("expected byte_fallback: true -- this "
                             "generator assumes it (" +
                             tokenizer_json_path + ")");
  {
    const json::Value *dropout = model.find("dropout");
    if (dropout && !dropout->is_null())
      throw std::runtime_error(
          "expected dropout: null (deterministic BPE) -- got a value in " +
          tokenizer_json_path);
  }
  if (truthy(model.find("continuing_subword_prefix")) ||
      truthy(model.find("end_of_word_suffix")))
    throw std::runtime_error(
        "expected no continuing_subword_prefix/end_of_word_suffix -- the "
        "merge-is-plain-concatenation assumption below would be wrong");

  {
    const json::Value *norm = tok.find("normalizer");
    // The exact 3-byte UTF-8 encoding of '▁' (metaspace), written as a
    // byte-escape sequence rather than the literal character -- same
    // convention tokenizer_gemma.cpp's metaspace() uses, so this file does
    // not depend on the source file's encoding being read as UTF-8 by MSVC.
    static const std::string kMetaspace = "\xE2\x96\x81";
    bool ok = false;
    if (norm && norm->is_object()) {
      const json::Value *type = norm->find("type");
      const json::Value *pattern = norm->find("pattern");
      const json::Value *content = norm->find("content");
      if (type && type->is_string() && type->str_v == "Replace" && pattern &&
          pattern->is_object() && pattern->obj_v.size() == 1 &&
          pattern->contains("String") && pattern->at("String").is_string() &&
          pattern->at("String").str_v == " " && content &&
          content->is_string() && content->str_v == kMetaspace) {
        ok = true;
      }
    }
    if (!ok)
      throw std::runtime_error(
          "unexpected normalizer, re-verify pipeline assumptions in " +
          tokenizer_json_path);
  }

  const json::Value &vocab_val = model.at("vocab");
  const auto &vocab_obj = vocab_val.as_object();
  const size_t vocab_size = vocab_obj.size();
  std::vector<const std::string *> id_to_token(vocab_size, nullptr);
  for (const auto &kv : vocab_obj) {
    const int64_t tid = json_int(kv.second, "vocab id for '" + kv.first + "'");
    if (tid < 0 || static_cast<size_t>(tid) >= vocab_size)
      throw std::runtime_error("vocab id " + std::to_string(tid) + " for '" +
                               kv.first + "' out of [0, " +
                               std::to_string(vocab_size) + ")");
    if (id_to_token[static_cast<size_t>(tid)] != nullptr)
      throw std::runtime_error("duplicate vocab id " + std::to_string(tid));
    id_to_token[static_cast<size_t>(tid)] = &kv.first;
  }
  {
    std::vector<size_t> missing_sample;
    size_t total_missing = 0;
    for (size_t i = 0; i < vocab_size; ++i) {
      if (id_to_token[i]) continue;
      ++total_missing;
      if (missing_sample.size() < 5) missing_sample.push_back(i);
    }
    if (total_missing > 0) {
      std::ostringstream os;
      os << "vocab has " << total_missing << " unused ids, e.g. ";
      for (size_t k = 0; k < missing_sample.size(); ++k)
        os << (k ? ", " : "") << missing_sample[k];
      throw std::runtime_error(os.str());
    }
  }

  const json::Value &merges_val = model.at("merges");
  const auto &merges_arr = merges_val.as_array();
  struct Merge { uint32_t a, b, merged; };
  std::vector<Merge> merges;
  merges.reserve(merges_arr.size());
  for (size_t rank = 0; rank < merges_arr.size(); ++rank) {
    const json::Value &m = merges_arr[rank];
    if (!m.is_array() || m.arr_v.size() != 2)
      throw std::runtime_error("merge rank " + std::to_string(rank) +
                               " is not a 2-element [piece, piece] array");
    const std::string &a = m.arr_v[0].as_string();
    const std::string &b = m.arr_v[1].as_string();
    const auto ita = vocab_obj.find(a);
    const auto itb = vocab_obj.find(b);
    if (ita == vocab_obj.end() || itb == vocab_obj.end())
      throw std::runtime_error("merge rank " + std::to_string(rank) +
                               " references unknown piece '" + a + "'/'" +
                               b + "'");
    const std::string merged_str = a + b;
    const auto itm = vocab_obj.find(merged_str);
    if (itm == vocab_obj.end())
      throw std::runtime_error(
          "merge rank " + std::to_string(rank) + " (" + a + "+" + b +
          ") has no vocab entry for '" + merged_str +
          "' -- the plain-concatenation assumption is wrong");
    merges.push_back(
        {static_cast<uint32_t>(json_int(ita->second, "merge id_a")),
         static_cast<uint32_t>(json_int(itb->second, "merge id_b")),
         static_cast<uint32_t>(json_int(itm->second, "merge merged_id"))});
  }

  // Special ids -- cross-checked against the constants documented in
  // gen_gemma_tokenizer_table.py's module header, so a future checkpoint
  // revision that changes them fails loudly instead of silently shipping
  // wrong ids.
  static const char *const kSpecialNames[5] = {"<pad>", "<eos>", "<bos>",
                                               "<unk>", "<mask>"};
  static const int kSpecialWant[5] = {0, 1, 2, 3, 4};
  int64_t special_id[5] = {0, 0, 0, 0, 0};
  for (int k = 0; k < 5; ++k) {
    const auto it = vocab_obj.find(kSpecialNames[k]);
    if (it == vocab_obj.end())
      throw std::runtime_error(std::string("vocab has no entry for ") +
                               kSpecialNames[k]);
    const int64_t got = json_int(it->second, kSpecialNames[k]);
    if (got != kSpecialWant[k])
      throw std::runtime_error(std::string("expected ") + kSpecialNames[k] +
                               "=" + std::to_string(kSpecialWant[k]) +
                               ", checkpoint has " + std::to_string(got));
    special_id[k] = got;
  }

  bool add_bos = false, add_eos = false;
  {
    const json::Value *post = tok.find("post_processor");
    const json::Value *single =
        (post && post->is_object()) ? post->find("single") : nullptr;
    if (single && single->is_array()) {
      for (const auto &step : single->arr_v) {
        if (!step.is_object()) continue;
        const json::Value *st = step.find("SpecialToken");
        if (!st || !st->is_object()) continue;
        const json::Value *id = st->find("id");
        if (!id || !id->is_string()) continue;
        if (id->str_v == "<bos>") add_bos = true;
        if (id->str_v == "<eos>") add_eos = true;
      }
    }
    if (!(add_bos && add_eos))
      throw std::runtime_error(
          "expected post_processor to add both <bos> and <eos>");
  }

  // --- task prefixes ---------------------------------------------------
  const json::Value sbert = json::parse(slurp_text(sbert_config_path));
  const json::Value &prompts_val = sbert.at("prompts");
  const auto &prompts_obj = prompts_val.as_object();

  // This project's chosen default task prefix -- a decision, not a fact
  // from the checkpoint. See gen_gemma_tokenizer_table.py's module
  // docstring for why "document".
  static const std::string kDefaultPrefixName = "document";
  if (prompts_obj.find(kDefaultPrefixName) == prompts_obj.end()) {
    std::vector<std::string> names;
    names.reserve(prompts_obj.size());
    for (const auto &kv : prompts_obj) names.push_back(kv.first);
    std::sort(names.begin(), names.end());
    std::ostringstream os;
    os << "chosen default prefix '" << kDefaultPrefixName
       << "' not in checkpoint's prompts dict: [";
    for (size_t i = 0; i < names.size(); ++i)
      os << (i ? ", " : "") << "'" << names[i] << "'";
    os << "]";
    throw std::runtime_error(os.str());
  }
  std::vector<std::string> prefix_names;
  prefix_names.reserve(prompts_obj.size());
  for (const auto &kv : prompts_obj) prefix_names.push_back(kv.first);
  std::sort(prefix_names.begin(), prefix_names.end());  // deterministic order
  size_t default_prefix_index = 0;
  for (size_t i = 0; i < prefix_names.size(); ++i)
    if (prefix_names[i] == kDefaultPrefixName) {
      default_prefix_index = i;
      break;
    }

  // --- write the binary table ------------------------------------------
  std::vector<uint8_t> out;
  out.reserve(64 + vocab_size * 8 + merges.size() * 12 + 4096);

  static const char kMagic[8] = {'G', 'E', 'M', 'A', 'T', 'O', 'K', '1'};
  out.insert(out.end(), kMagic, kMagic + 8);
  put_u32(out, 1);  // VERSION
  put_u32(out, static_cast<uint32_t>(vocab_size));
  put_u32(out, static_cast<uint32_t>(merges.size()));
  put_u32(out, static_cast<uint32_t>(special_id[0]));  // pad
  put_u32(out, static_cast<uint32_t>(special_id[1]));  // eos
  put_u32(out, static_cast<uint32_t>(special_id[2]));  // bos
  put_u32(out, static_cast<uint32_t>(special_id[3]));  // unk
  put_u32(out, static_cast<uint32_t>(special_id[4]));  // mask
  put_u32(out, add_bos ? 1u : 0u);
  put_u32(out, add_eos ? 1u : 0u);
  put_u32(out, static_cast<uint32_t>(prefix_names.size()));
  put_u32(out, static_cast<uint32_t>(default_prefix_index));

  // vocab: id order, u16-length-prefixed UTF-8 bytes
  for (size_t i = 0; i < vocab_size; ++i) put_str16(out, *id_to_token[i]);

  // merges: rank order, (id_a, id_b, merged_id) as u32 triples
  for (const auto &m : merges) {
    put_u32(out, m.a);
    put_u32(out, m.b);
    put_u32(out, m.merged);
  }

  // task prefixes: name then prefix text, both u16-length-prefixed
  for (const auto &name : prefix_names) {
    put_str16(out, name);
    put_str16(out, prompts_obj.at(name).as_string());
  }

  return out;
}

}  // namespace npue
