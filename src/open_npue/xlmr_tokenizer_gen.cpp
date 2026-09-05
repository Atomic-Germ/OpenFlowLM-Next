//===- xlmr_tokenizer_gen.cpp --------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- C++ port of tools/gen_xlmr_tokenizer_table.py.
// SPDX-License-Identifier: MIT
// See xlmr_tokenizer_gen.hpp for why this exists.
//
// Every check below is a direct port of a `raise SystemExit(...)` guard in
// the Python script, kept in the same order and against the same fields, so
// the two can be diffed against each other line by line (the
// gemma_tokenizer_gen.cpp discipline). Comments repeat only what is not
// obvious from the port itself.

#include "xlmr_tokenizer_gen.hpp"

#include "json_min.hpp"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

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

// JSON "truthiness", mirroring Python's `if meta.get(key):` / `bool(...)` --
// used for the byte_fallback/special/normalized/add_prefix_space/split
// checks, which the Python script tests by truthiness rather than an
// explicit type check.
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

// Strict base64 (RFC 4648 alphabet, '=' padding, whitespace skipped, any
// other character an error -- fail closed where Python's b64decode would
// silently discard).
std::vector<uint8_t> base64_decode(const std::string &s) {
  auto val = [](char c) -> int {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
  };
  std::vector<uint8_t> out;
  out.reserve(s.size() / 4 * 3);
  uint32_t acc = 0;
  int nbits = 0;
  for (char c : s) {
    if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
    const int v = val(c);
    if (v < 0)
      throw std::runtime_error("invalid base64 character in "
                               "precompiled_charsmap");
    acc = (acc << 6) | static_cast<uint32_t>(v);
    nbits += 6;
    if (nbits >= 8) {
      nbits -= 8;
      out.push_back(static_cast<uint8_t>((acc >> nbits) & 0xFF));
    }
  }
  return out;
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
// f64 little-endian, raw bits -- the scores must land byte-identical to
// struct.pack("<d", ...). Same native-LE assumption as every read_u32/
// memcpy in this runtime.
void put_f64(std::vector<uint8_t> &out, double v) {
  uint8_t b[8];
  std::memcpy(b, &v, 8);
  out.insert(out.end(), b, b + 8);
}
// u16-length-prefixed UTF-8 bytes (struct.pack("<H", len(b)) + b).
void put_str16(std::vector<uint8_t> &out, const std::string &s) {
  if (s.size() > 0xFFFF)
    throw std::runtime_error("string exceeds 65535 bytes: '" +
                             s.substr(0, 50) + "...'");
  put_u16(out, static_cast<uint16_t>(s.size()));
  out.insert(out.end(), s.begin(), s.end());
}

// Codepoint count of a valid UTF-8 string (== Python len(str)): lead bytes.
size_t utf8_chars(const std::string &s) {
  size_t n = 0;
  for (char c : s)
    if ((static_cast<unsigned char>(c) & 0xC0) != 0x80) ++n;
  return n;
}

}  // namespace

std::vector<uint8_t> generate_xlmr_tokenizer_table(
    const std::string &tokenizer_json_path) {
  const json::Value tok = json::parse(slurp_text(tokenizer_json_path));

  // --- model: must be Unigram, and shaped the way this generator expects
  const json::Value &model = tok.at("model");
  {
    const json::Value *type_v = model.find("type");
    if (!type_v || !type_v->is_string() || type_v->str_v != "Unigram")
      throw std::runtime_error(
          "expected model.type == 'Unigram' -- this generator implements "
          "SentencePiece Unigram specifically (the BPE generator is "
          "gemma_tokenizer_gen.cpp); re-check " + tokenizer_json_path +
          " before proceeding");
  }
  if (truthy(model.find("byte_fallback")))
    throw std::runtime_error(
        "expected byte_fallback: false -- the Viterbi <unk> path assumes no "
        "byte pieces; a checkpoint with byte_fallback needs new code, not "
        "this table");
  const json::Value *unk_id_v = model.find("unk_id");
  const json::Value &vocab_val = model.at("vocab");
  const auto &vocab_arr = vocab_val.as_array();
  const size_t vocab_size = vocab_arr.size();
  int64_t unk_id = -1;
  if (unk_id_v && unk_id_v->is_number()) {
    const double d = unk_id_v->num_v;
    if (static_cast<double>(static_cast<int64_t>(d)) == d)
      unk_id = static_cast<int64_t>(d);
  }
  if (unk_id < 0 || static_cast<size_t>(unk_id) >= vocab_size)
    throw std::runtime_error("unk_id not an id in [0, " +
                             std::to_string(vocab_size) + ")");

  std::vector<const std::string *> pieces;
  std::vector<double> scores;
  pieces.reserve(vocab_size);
  scores.reserve(vocab_size);
  std::unordered_set<std::string> seen;
  seen.reserve(vocab_size * 2);
  for (const json::Value &row : vocab_arr) {
    if (!(row.is_array() && row.arr_v.size() == 2))
      throw std::runtime_error("vocab row not a [piece, score] pair");
    const std::string &piece = row.arr_v[0].as_string();
    const double score = row.arr_v[1].as_number();
    if (!seen.insert(piece).second)
      throw std::runtime_error(
          "duplicate vocab piece '" + piece + "' -- the C++ hash lookup "
          "would silently keep one of the two scores");
    pieces.push_back(&row.arr_v[0].str_v);
    scores.push_back(score);
  }

  double min_score = scores[0];
  for (double s : scores)
    if (s < min_score) min_score = s;
  const double unk_score = min_score - 10.0;  // tokenizers' kUnkPenalty

  // --- normalizer: Precompiled charsmap, decoded and sanity-checked
  const json::Value *norm = tok.find("normalizer");
  if (!norm || !norm->is_object() || !norm->find("type") ||
      !norm->at("type").is_string() || norm->at("type").str_v != "Precompiled" ||
      !norm->contains("precompiled_charsmap"))
    throw std::runtime_error(
        "expected normalizer type 'Precompiled' with a precompiled_charsmap "
        "-- the charsmap trie below would not apply");
  const std::vector<uint8_t> charsmap =
      base64_decode(norm->at("precompiled_charsmap").as_string());
  if (charsmap.size() < 4)
    throw std::runtime_error(
        "precompiled_charsmap shorter than its own size field");
  uint32_t trie_size;
  std::memcpy(&trie_size, charsmap.data(), 4);
  if (trie_size % 4 != 0 || 4 + static_cast<size_t>(trie_size) > charsmap.size())
    throw std::runtime_error(
        "charsmap trie size " + std::to_string(trie_size) +
        " inconsistent with blob length " + std::to_string(charsmap.size()) +
        " -- the u32-LE-size-then-trie-then-strings layout assumption is "
        "wrong for this checkpoint");
  const uint8_t *trie = charsmap.data() + 4;
  const uint8_t *normalized = charsmap.data() + 4 + trie_size;
  const size_t norm_size = charsmap.size() - 4 - trie_size;
  if (norm_size != 0 && normalized[norm_size - 1] != 0)
    throw std::runtime_error(
        "charsmap normalized-strings blob does not end in NUL -- the "
        "'replacement runs to the next NUL' walk would read past the end");

  // --- pre_tokenizer: exactly [WhitespaceSplit, Metaspace]
  const json::Value *pre = tok.find("pre_tokenizer");
  if (!pre || !pre->is_object() || !pre->find("type") ||
      !pre->at("type").is_string() || pre->at("type").str_v != "Sequence")
    throw std::runtime_error("expected pre_tokenizer Sequence");
  const json::Value *subs = pre->find("pretokenizers");
  auto sub_type = [&](size_t i) -> std::string {
    if (!subs || !subs->is_array() || i >= subs->arr_v.size()) return "";
    const json::Value *t = subs->arr_v[i].find("type");
    return (t && t->is_string()) ? t->str_v : "";
  };
  if (!subs || !subs->is_array() || subs->arr_v.size() != 2 ||
      sub_type(0) != "WhitespaceSplit" || sub_type(1) != "Metaspace")
    throw std::runtime_error(
        "expected pre_tokenizers [WhitespaceSplit, Metaspace] -- the "
        "pre-tokenization in the runtime tokenizer would be wrong");
  const json::Value &meta = subs->arr_v[1];
  // The exact 3-byte UTF-8 encoding of the metaspace codepoint U+2581,
  // written as byte escapes rather than the literal character -- same
  // convention as gemma_tokenizer_gen.cpp, so this file does not depend on
  // the source file's encoding being read as UTF-8 by MSVC.
  static const std::string kMetaspace = "\xE2\x96\x81";
  const json::Value *replacement = meta.find("replacement");
  if (!replacement || !replacement->is_string() ||
      replacement->str_v != kMetaspace)
    throw std::runtime_error("expected Metaspace replacement U+2581");
  std::string scheme_name;
  {
    const json::Value *scheme_v = meta.find("prepend_scheme");
    if (scheme_v && !scheme_v->is_null()) {
      scheme_name = scheme_v->as_string();
    } else {
      // legacy serialization: add_prefix_space bool only
      scheme_name = truthy(meta.find("add_prefix_space")) ? "always" : "never";
    }
  }
  uint32_t prepend_scheme;
  if (scheme_name == "never") prepend_scheme = 0;
  else if (scheme_name == "first") prepend_scheme = 1;
  else if (scheme_name == "always") prepend_scheme = 2;
  else
    throw std::runtime_error("unknown Metaspace prepend_scheme '" +
                             scheme_name + "'");
  if (meta.contains("add_prefix_space") &&
      truthy(meta.find("add_prefix_space")) != (scheme_name != "never"))
    throw std::runtime_error("Metaspace add_prefix_space contradicts "
                             "prepend_scheme '" + scheme_name + "'");
  const bool metaspace_split =
      meta.contains("split") ? truthy(meta.find("split")) : true;

  // --- specials: read from added_tokens, cross-checked against the vocab
  std::unordered_map<std::string, const json::Value *> added;
  {
    const json::Value *added_v = tok.find("added_tokens");
    if (added_v && added_v->is_array())
      for (const json::Value &t : added_v->arr_v)
        added[t.at("content").as_string()] = &t;
  }
  static const char *const kExpectSpecials[5] = {"<s>", "<pad>", "</s>",
                                                 "<unk>", "<mask>"};
  std::unordered_map<std::string, int64_t> ids;
  for (const char *name : kExpectSpecials) {
    const auto it = added.find(name);
    if (it == added.end())
      throw std::runtime_error(std::string("special token '") + name +
                               "' missing from added_tokens");
    if (!truthy(it->second->find("special")))
      throw std::runtime_error(std::string("'") + name +
                               "' present but not marked special");
    if (truthy(it->second->find("normalized")))
      throw std::runtime_error(std::string("'") + name +
                               "' is normalized:true -- unexpected for XLM-R");
    const int64_t tid = json_int(it->second->at("id"),
                                 std::string("added token id for ") + name);
    ids[name] = tid;
    // every special must also be a vocab row at the same id, or the
    // runtime id->piece table would disagree with the added_tokens ids
    if (tid < 0 || static_cast<size_t>(tid) >= vocab_size ||
        *pieces[static_cast<size_t>(tid)] != name)
      throw std::runtime_error(
          std::string("added token '") + name + "' id " +
          std::to_string(tid) + " does not match vocab row " +
          (tid >= 0 && static_cast<size_t>(tid) < vocab_size
               ? "'" + *pieces[static_cast<size_t>(tid)] + "'"
               : std::string("<oob>")));
  }
  if (ids["<unk>"] != unk_id)
    throw std::runtime_error("added <unk> id " +
                             std::to_string(ids["<unk>"]) +
                             " != model unk_id " + std::to_string(unk_id));

  // --- post_processor: single-sequence template must be <s> A </s>
  const json::Value *post = tok.find("post_processor");
  if (!post || !post->is_object() || !post->find("type") ||
      !post->at("type").is_string() ||
      post->at("type").str_v != "TemplateProcessing")
    throw std::runtime_error("expected TemplateProcessing");
  {
    // Python compares `single` against the exact literal structure
    // [{SpecialToken:{id:'<s>',type_id:0}}, {Sequence:{id:'A',type_id:0}},
    //  {SpecialToken:{id:'</s>',type_id:0}}] by dict equality -- extra or
    // missing keys anywhere fail. Ported as explicit shape checks.
    auto is_step = [](const json::Value &step, const char *kind,
                      const char *id) {
      if (!step.is_object() || step.obj_v.size() != 1) return false;
      const json::Value *inner = step.find(kind);
      if (!inner || !inner->is_object() || inner->obj_v.size() != 2)
        return false;
      const json::Value *id_v = inner->find("id");
      const json::Value *tid_v = inner->find("type_id");
      return id_v && id_v->is_string() && id_v->str_v == id && tid_v &&
             tid_v->is_number() && tid_v->num_v == 0.0;
    };
    const json::Value *single = post->find("single");
    if (!single || !single->is_array() || single->arr_v.size() != 3 ||
        !is_step(single->arr_v[0], "SpecialToken", "<s>") ||
        !is_step(single->arr_v[1], "Sequence", "A") ||
        !is_step(single->arr_v[2], "SpecialToken", "</s>"))
      throw std::runtime_error(
          "expected single template [<s>, A, </s>] -- the fixed "
          "add_bos/add_eos flags below would lie");
  }
  {
    const json::Value *post_ids = post->find("special_tokens");
    for (const char *name : {"<s>", "</s>"}) {
      const json::Value *entry =
          (post_ids && post_ids->is_object()) ? post_ids->find(name) : nullptr;
      const json::Value *got =
          (entry && entry->is_object()) ? entry->find("ids") : nullptr;
      if (!got || !got->is_array() || got->arr_v.size() != 1 ||
          !got->arr_v[0].is_number() ||
          got->arr_v[0].num_v != static_cast<double>(ids[name]))
        throw std::runtime_error(std::string("post_processor maps '") + name +
                                 "' to something other than added_tokens' [" +
                                 std::to_string(ids[name]) + "]");
    }
  }
  const uint32_t add_bos = 1, add_eos = 1;

  {
    const json::Value *trunc = tok.find("truncation");
    const json::Value *pad = tok.find("padding");
    if ((trunc && !trunc->is_null()) || (pad && !pad->is_null()))
      throw std::runtime_error(
          "tokenizer.json carries baked-in truncation/padding -- this "
          "project refuses silent truncation (tasks/0110); re-check");
  }

  size_t max_piece_bytes = 0, max_piece_chars = 0;
  for (const std::string *p : pieces) {
    if (p->size() > max_piece_bytes) max_piece_bytes = p->size();
    const size_t chars = utf8_chars(*p);
    if (chars > max_piece_chars) max_piece_chars = chars;
  }
  if (max_piece_bytes > 0xFFFF)
    throw std::runtime_error("a vocab piece exceeds 65535 bytes");

  // --- write the binary table ----------------------------------------
  std::vector<uint8_t> out;
  out.reserve(64 + charsmap.size() + vocab_size * 12);

  static const char kMagic[8] = {'X', 'L', 'M', 'R', 'T', 'O', 'K', '1'};
  out.insert(out.end(), kMagic, kMagic + 8);
  put_u32(out, 1);  // VERSION
  put_u32(out, static_cast<uint32_t>(vocab_size));
  put_u32(out, static_cast<uint32_t>(unk_id));
  put_u32(out, static_cast<uint32_t>(ids["<s>"]));    // bos
  put_u32(out, static_cast<uint32_t>(ids["</s>"]));   // eos
  put_u32(out, static_cast<uint32_t>(ids["<pad>"]));  // pad
  put_u32(out, static_cast<uint32_t>(ids["<mask>"]));  // mask
  put_u32(out, add_bos);
  put_u32(out, add_eos);
  put_u32(out, 0x2581);  // ord(replacement), asserted == U+2581 above
  put_u32(out, prepend_scheme);
  put_u32(out, metaspace_split ? 1u : 0u);
  put_u32(out, static_cast<uint32_t>(max_piece_bytes));
  put_u32(out, static_cast<uint32_t>(max_piece_chars));
  put_f64(out, min_score);
  put_f64(out, unk_score);
  put_u32(out, trie_size);
  put_u32(out, static_cast<uint32_t>(norm_size));
  out.insert(out.end(), trie, trie + trie_size);
  out.insert(out.end(), normalized, normalized + norm_size);
  for (double s : scores) put_f64(out, s);
  for (const std::string *p : pieces) put_str16(out, *p);

  return out;
}

}  // namespace npue
