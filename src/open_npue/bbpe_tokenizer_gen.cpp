//===- bbpe_tokenizer_gen.cpp --------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- C++ port of tools/gen_bbpe_tokenizer_table.py.
// See bbpe_tokenizer_gen.hpp.
// SPDX-License-Identifier: MIT
//
// The Python script is the reference; this file follows it statement for
// statement, including the order of its refusals, so a diff between the two
// stays readable. Where a message differs it is only in wording that names
// C++ rather than Python.

#include "bbpe_tokenizer_gen.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "json_min.hpp"

namespace npue {
namespace {

[[noreturn]] void fail(const std::string &msg) {
  throw std::runtime_error("gen_bbpe: " + msg);
}

void put_u16(std::vector<uint8_t> &o, uint16_t v) {
  o.push_back(static_cast<uint8_t>(v & 0xFF));
  o.push_back(static_cast<uint8_t>(v >> 8));
}
void put_u32(std::vector<uint8_t> &o, uint32_t v) {
  o.push_back(static_cast<uint8_t>(v & 0xFF));
  o.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
  o.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
  o.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}
void put_i32(std::vector<uint8_t> &o, int32_t v) {
  put_u32(o, static_cast<uint32_t>(v));
}
void put_str(std::vector<uint8_t> &o, const std::string &s) {
  if (s.size() > 0xFFFF) fail("a string exceeds 65535 bytes");
  put_u16(o, static_cast<uint16_t>(s.size()));
  o.insert(o.end(), s.begin(), s.end());
}

// GPT-2's byte -> printable-codepoint map, as UTF-8 strings. The same
// construction as the Python `bytes_to_unicode()`.
std::vector<std::string> byte_chars() {
  bool used[256] = {false};
  uint32_t cp[256];
  auto take = [&](int lo, int hi) {
    for (int b = lo; b <= hi; ++b) { cp[b] = static_cast<uint32_t>(b); used[b] = true; }
  };
  take('!', '~');
  take(0xA1, 0xAC);
  take(0xAE, 0xFF);
  int n = 0;
  for (int b = 0; b < 256; ++b)
    if (!used[b]) cp[b] = static_cast<uint32_t>(256 + n++);

  std::vector<std::string> out(256);
  for (int b = 0; b < 256; ++b) {
    std::string s;
    const uint32_t c = cp[b];
    if (c < 0x80) {
      s.push_back(static_cast<char>(c));
    } else if (c < 0x800) {
      s.push_back(static_cast<char>(0xC0 | (c >> 6)));
      s.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    } else {
      s.push_back(static_cast<char>(0xE0 | (c >> 12)));
      s.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
      s.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    }
    out[b] = s;
  }
  return out;
}

}  // namespace

std::vector<uint8_t> generate_bbpe_tokenizer_table(
    const std::string &tokenizer_json_path) {
  std::ifstream f(tokenizer_json_path, std::ios::binary);
  if (!f) fail("cannot open " + tokenizer_json_path);
  std::stringstream ss;
  ss << f.rdbuf();
  const json::Value j = json::parse(ss.str());

  // --- normalizer -------------------------------------------------------
  uint32_t norm = 0;
  if (const json::Value *n = j.find("normalizer")) {
    if (!n->is_null()) {
      if (!n->is_object() || !n->contains("type") ||
          n->at("type").as_string() != "NFC")
        fail("unsupported normalizer -- this generator implements `null` and "
             "`NFC`. Add it to tokenizer_bbpe.cpp AND here, or the runtime "
             "will normalize differently from HuggingFace and every id will "
             "look reasonable and be wrong.");
      norm = 1;
    }
  }

  // --- pre_tokenizer ----------------------------------------------------
  const json::Value *pre = j.find("pre_tokenizer");
  if (pre == nullptr || !pre->is_object() ||
      pre->at("type").as_string() != "ByteLevel")
    fail("pre_tokenizer is not ByteLevel -- this generator implements "
         "ByteLevel only. A `Sequence` or an explicit `Split` pattern "
         "(Llama-3, Qwen-2.5 and tekken use one) is a DIFFERENT regex and "
         "needs its own scanner; refusing rather than pretending the GPT-2 "
         "pattern is universal.");
  if (const json::Value *ur = pre->find("use_regex"))
    if (!ur->as_bool())
      fail("pre_tokenizer has use_regex=false -- the whole text becomes one "
           "word. Supportable, but no checkpoint here needs it and an "
           "untested path is worse than a refusal.");
  uint32_t add_prefix_space = 0;
  if (const json::Value *ap = pre->find("add_prefix_space"))
    add_prefix_space = ap->as_bool() ? 1u : 0u;

  // --- model ------------------------------------------------------------
  const json::Value &m = j.at("model");
  // `model.type` is ABSENT in the pre-0.10 tokenizer.json format
  // (roberta-base still ships one). The Python generator reports that as
  // "model.type is None, expected BPE"; this said "object has no key 'type'",
  // which is the same refusal wearing a parser's clothes. The port is meant
  // to be faithful down to its messages.
  const json::Value *mtype = m.find("type");
  if (mtype == nullptr || !mtype->is_string())
    fail("model.type is None, expected BPE");
  if (mtype->as_string() != "BPE")
    fail("model.type is '" + mtype->as_string() + "', expected BPE");
  for (const char *key : {"continuing_subword_prefix", "end_of_word_suffix"}) {
    const json::Value *v = m.find(key);
    if (v != nullptr && v->is_string() && !v->as_string().empty())
      fail(std::string("model.") + key +
           " is set -- affix-marked BPE is a different segmentation and is "
           "not implemented");
  }
  auto flag = [&](const char *key) {
    const json::Value *v = m.find(key);
    return v != nullptr && v->is_bool() && v->as_bool();
  };
  if (flag("byte_fallback"))
    fail("model.byte_fallback is true -- that belongs to SentencePiece BPE "
         "(see tokenizer_gemma.cpp); a byte-level model needs no fallback "
         "because every byte is already a vocabulary entry");
  if (flag("ignore_merges"))
    fail("model.ignore_merges is true -- a word present in the vocabulary "
         "bypasses the merge loop entirely. Not implemented; it changes the "
         "segmentation of exactly the common words, so ignoring it would be "
         "undetectable in a spot check and wrong in a corpus.");
  if (const json::Value *d = m.find("dropout"))
    if (!d->is_null()) fail("model.dropout is set -- BPE-dropout is stochastic");
  if (const json::Value *u = m.find("unk_token"))
    if (!u->is_null())
      fail("model.unk_token is set -- a byte-level model has no unknown "
           "pieces; this generator does not implement one");

  // --- vocabulary -------------------------------------------------------
  const auto &vocab_obj = m.at("vocab").as_object();
  std::unordered_map<std::string, int32_t> vocab;
  vocab.reserve(vocab_obj.size() * 2);
  int32_t max_id = -1;
  for (const auto &kv : vocab_obj) {
    const int32_t id = static_cast<int32_t>(kv.second.as_number());
    vocab.emplace(kv.first, id);
    max_id = std::max(max_id, id);
  }
  if (static_cast<size_t>(max_id) + 1 != vocab.size())
    fail("vocab ids are not contiguous 0..n-1");
  std::vector<std::string> id_to_token(vocab.size());
  std::vector<bool> seen(vocab.size(), false);
  for (const auto &kv : vocab) {
    if (kv.second < 0 || static_cast<size_t>(kv.second) >= id_to_token.size())
      fail("vocab id out of range");
    if (seen[static_cast<size_t>(kv.second)]) fail("duplicate vocab id");
    seen[static_cast<size_t>(kv.second)] = true;
    id_to_token[static_cast<size_t>(kv.second)] = kv.first;
  }

  // --- merges -----------------------------------------------------------
  //
  // `tokenizers` >= 0.20 writes ["a", "b"] pairs; older files use "a b".
  // Byte-level pieces never contain a space (it is mapped to U+0120), so the
  // split is unambiguous either way, and both forms are read.
  const auto &merges_arr = m.at("merges").as_array();
  std::vector<std::array<uint32_t, 3>> merges;
  merges.reserve(merges_arr.size());
  for (const json::Value &e : merges_arr) {
    std::string a, b;
    if (e.is_array()) {
      const auto &p = e.as_array();
      if (p.size() != 2) fail("a merge entry is not a pair");
      a = p[0].as_string();
      b = p[1].as_string();
    } else {
      const std::string &s = e.as_string();
      const size_t sp = s.find(' ');
      if (sp == std::string::npos) fail("a merge entry has no separator");
      a = s.substr(0, sp);
      b = s.substr(sp + 1);
    }
    auto ia = vocab.find(a), ib = vocab.find(b), im = vocab.find(a + b);
    if (ia == vocab.end() || ib == vocab.end() || im == vocab.end())
      fail("merge '" + a + "'+'" + b +
           "' references a piece not in the vocabulary");
    merges.push_back({static_cast<uint32_t>(ia->second),
                      static_cast<uint32_t>(ib->second),
                      static_cast<uint32_t>(im->second)});
  }

  // --- added tokens -----------------------------------------------------
  struct Added { std::string content; int32_t id; uint32_t flags; };
  std::vector<Added> added;
  if (const json::Value *at = j.find("added_tokens")) {
    for (const json::Value &t : at->as_array()) {
      auto boolean = [&](const char *k) {
        const json::Value *v = t.find(k);
        return v != nullptr && v->is_bool() && v->as_bool();
      };
      if (boolean("single_word"))
        fail("an added token sets single_word -- that constrains the match to "
             "word boundaries and is not implemented");
      if (boolean("rstrip"))
        fail("an added token sets rstrip -- not implemented, because no "
             "checkpoint available here sets it and the branch would ship "
             "unverified (lstrip IS implemented and IS exercised, by [MASK])");
      const uint32_t flags = boolean("lstrip") ? 1u : 0u;
      added.push_back(Added{t.at("content").as_string(),
                            static_cast<int32_t>(t.at("id").as_number()),
                            flags});
    }
  }

  // --- named special ids ------------------------------------------------
  //
  // Searched in the added tokens too, not just model.vocab: granite-4.2-3b
  // keeps `<|padding|>` only in added_tokens.
  std::unordered_map<std::string, int32_t> lookup = vocab;
  for (const Added &a : added) lookup.emplace(a.content, a.id);
  auto named = [&](std::initializer_list<const char *> names) -> int32_t {
    for (const char *n : names) {
      auto it = lookup.find(n);
      if (it != lookup.end()) return it->second;
    }
    return -1;
  };
  const int32_t cls_id = named({"[CLS]", "<s>", "<|endoftext|>"});
  const int32_t sep_id = named({"[SEP]", "</s>"});
  const int32_t pad_id = named({"[PAD]", "<pad>", "<|padding|>"});
  const int32_t unk_id = named({"[UNK]", "<unk>"});
  const int32_t mask_id = named({"[MASK]", "<mask>"});

  // --- post processor ---------------------------------------------------
  std::vector<int32_t> prefix_ids, suffix_ids;
  const json::Value *post = j.find("post_processor");
  if (post != nullptr && !post->is_null()) {
    if (post->at("type").as_string() != "TemplateProcessing")
      fail("post_processor is not TemplateProcessing -- ByteLevel and "
           "RobertaProcessing exist in this family and are NOT the same "
           "wrapping");
    const json::Value *specials = post->find("special_tokens");
    bool seen_sequence = false;
    const json::Value *single = post->find("single");
    if (single == nullptr) fail("post_processor has no `single` template");
    for (const json::Value &item : single->as_array()) {
      if (item.contains("Sequence")) {
        if (seen_sequence) fail("post_processor template has two sequences");
        seen_sequence = true;
        continue;
      }
      if (item.contains("SpecialToken")) {
        const std::string &name = item.at("SpecialToken").at("id").as_string();
        if (specials == nullptr || !specials->contains(name))
          fail("template names special token '" + name +
               "' with no entry in special_tokens");
        const auto &ids = specials->at(name).at("ids").as_array();
        if (ids.size() != 1)
          fail("special token '" + name + "' does not map to exactly one id");
        (seen_sequence ? suffix_ids : prefix_ids)
            .push_back(static_cast<int32_t>(ids[0].as_number()));
        continue;
      }
      fail("unrecognised post_processor template item");
    }
    if (!seen_sequence)
      fail("post_processor template never places the sequence");
  }

  // --- the byte characters must all be present --------------------------
  const std::vector<std::string> bc = byte_chars();
  std::vector<int> unexpected;
  for (int b = 0; b < 256; ++b) {
    if (vocab.count(bc[static_cast<size_t>(b)])) continue;
    // 0xC0, 0xC1 and 0xF5..0xFF cannot occur in valid UTF-8, so a vocabulary
    // may legitimately omit them.
    if (b == 0xC0 || b == 0xC1 || b >= 0xF5) continue;
    unexpected.push_back(b);
  }
  if (!unexpected.empty()) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "0x%02X", unexpected[0]);
    fail(std::to_string(unexpected.size()) +
         " byte characters are absent from the vocabulary (first: " + buf +
         ") -- a byte-level BPE vocabulary must contain every byte that can "
         "occur in UTF-8");
  }

  // --- emit -------------------------------------------------------------
  std::vector<uint8_t> o;
  const char magic[8] = {'B', 'B', 'P', 'E', 'T', 'O', 'K', '1'};
  o.insert(o.end(), magic, magic + 8);
  put_u32(o, 1);                                   // version
  put_u32(o, norm);
  put_u32(o, add_prefix_space);
  put_u32(o, static_cast<uint32_t>(id_to_token.size()));
  put_u32(o, static_cast<uint32_t>(merges.size()));
  put_u32(o, static_cast<uint32_t>(added.size()));
  put_i32(o, cls_id);
  put_i32(o, sep_id);
  put_i32(o, pad_id);
  put_i32(o, unk_id);
  put_i32(o, mask_id);
  put_u32(o, static_cast<uint32_t>(prefix_ids.size()));
  for (int32_t i : prefix_ids) put_i32(o, i);
  put_u32(o, static_cast<uint32_t>(suffix_ids.size()));
  for (int32_t i : suffix_ids) put_i32(o, i);
  for (const std::string &t : id_to_token) put_str(o, t);
  for (const auto &mg : merges) {
    put_u32(o, mg[0]);
    put_u32(o, mg[1]);
    put_u32(o, mg[2]);
  }
  for (const Added &a : added) {
    put_str(o, a.content);
    put_i32(o, a.id);
    put_u32(o, a.flags);
  }
  return o;
}

}  // namespace npue
