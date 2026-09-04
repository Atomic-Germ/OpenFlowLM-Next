//===- json_min.cpp -----------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- a minimal, dependency-free JSON DOM parser.
// SPDX-License-Identifier: MIT
// See json_min.hpp for why this exists and its scope.

#include "json_min.hpp"

#include <cstdlib>
#include <sstream>

namespace npue {
namespace json {

const std::string &Value::as_string() const {
  if (type != Type::String)
    throw std::runtime_error("json::Value: expected string");
  return str_v;
}
double Value::as_number() const {
  if (type != Type::Number)
    throw std::runtime_error("json::Value: expected number");
  return num_v;
}
bool Value::as_bool() const {
  if (type != Type::Bool)
    throw std::runtime_error("json::Value: expected bool");
  return bool_v;
}
const std::vector<Value> &Value::as_array() const {
  if (type != Type::Array)
    throw std::runtime_error("json::Value: expected array");
  return arr_v;
}
const std::unordered_map<std::string, Value> &Value::as_object() const {
  if (type != Type::Object)
    throw std::runtime_error("json::Value: expected object");
  return obj_v;
}
const Value &Value::at(const std::string &key) const {
  const auto &o = as_object();
  auto it = o.find(key);
  if (it == o.end())
    throw std::runtime_error("json::Value: object has no key '" + key + "'");
  return it->second;
}
const Value *Value::find(const std::string &key) const {
  if (type != Type::Object) return nullptr;
  auto it = obj_v.find(key);
  return it == obj_v.end() ? nullptr : &it->second;
}
bool Value::contains(const std::string &key) const {
  return find(key) != nullptr;
}

namespace {

// Encodes one Unicode codepoint as UTF-8 into `out`. Used both for a plain
// \uXXXX escape and for a combined UTF-16 surrogate pair.
void append_utf8(std::string &out, uint32_t cp) {
  if (cp <= 0x7F) {
    out.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

class Parser {
public:
  explicit Parser(const std::string &s) : s_(s), n_(s.size()) {}

  Value parse_document() {
    skip_ws();
    Value v = parse_value();
    skip_ws();
    if (i_ != n_) throw err("trailing data after top-level JSON value");
    return v;
  }

private:
  const std::string &s_;
  size_t i_ = 0;
  size_t n_;

  std::runtime_error err(const std::string &msg) const {
    std::ostringstream os;
    os << "json parse error at byte " << i_ << ": " << msg;
    return std::runtime_error(os.str());
  }

  char peek() const {
    if (i_ >= n_) throw err("unexpected end of input");
    return s_[i_];
  }

  void skip_ws() {
    while (i_ < n_) {
      const char c = s_[i_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
        ++i_;
      else
        break;
    }
  }

  void expect(char c) {
    if (i_ >= n_ || s_[i_] != c) {
      std::ostringstream os;
      os << "expected '" << c << "'";
      throw err(os.str());
    }
    ++i_;
  }

  bool consume_literal(const char *lit, size_t len) {
    if (i_ + len > n_) return false;
    if (s_.compare(i_, len, lit, len) != 0) return false;
    i_ += len;
    return true;
  }

  Value parse_value() {
    skip_ws();
    if (i_ >= n_) throw err("unexpected end of input, expected a value");
    const char c = s_[i_];
    switch (c) {
      case '{': return parse_object();
      case '[': return parse_array();
      case '"': return parse_string_value();
      case 't':
        if (consume_literal("true", 4)) {
          Value v; v.type = Type::Bool; v.bool_v = true; return v;
        }
        throw err("invalid literal, expected 'true'");
      case 'f':
        if (consume_literal("false", 5)) {
          Value v; v.type = Type::Bool; v.bool_v = false; return v;
        }
        throw err("invalid literal, expected 'false'");
      case 'n':
        if (consume_literal("null", 4)) {
          Value v; v.type = Type::Null; return v;
        }
        throw err("invalid literal, expected 'null'");
      default:
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        throw err("unexpected character starting a value");
    }
  }

  Value parse_object() {
    expect('{');
    Value v;
    v.type = Type::Object;
    skip_ws();
    if (i_ < n_ && s_[i_] == '}') { ++i_; return v; }
    for (;;) {
      skip_ws();
      if (i_ >= n_ || s_[i_] != '"')
        throw err("expected a string key in object");
      std::string key = parse_raw_string();
      skip_ws();
      expect(':');
      Value val = parse_value();
      v.obj_v.emplace(std::move(key), std::move(val));
      skip_ws();
      if (i_ >= n_) throw err("unterminated object");
      if (s_[i_] == ',') { ++i_; continue; }
      if (s_[i_] == '}') { ++i_; break; }
      throw err("expected ',' or '}' in object");
    }
    return v;
  }

  Value parse_array() {
    expect('[');
    Value v;
    v.type = Type::Array;
    skip_ws();
    if (i_ < n_ && s_[i_] == ']') { ++i_; return v; }
    for (;;) {
      Value elem = parse_value();
      v.arr_v.push_back(std::move(elem));
      skip_ws();
      if (i_ >= n_) throw err("unterminated array");
      if (s_[i_] == ',') { ++i_; continue; }
      if (s_[i_] == ']') { ++i_; break; }
      throw err("expected ',' or ']' in array");
    }
    return v;
  }

  Value parse_string_value() {
    Value v;
    v.type = Type::String;
    v.str_v = parse_raw_string();
    return v;
  }

  // Parses a JSON string starting at the opening '"' (which must be the
  // current character) and returns its decoded UTF-8 content. The common
  // case -- a run of plain bytes with no escape -- is copied in one
  // `append(ptr, len)` rather than character-by-character, since
  // tokenizer.json has ~262k vocab strings and ~515k merge-pair strings.
  std::string parse_raw_string() {
    expect('"');
    std::string out;
    const char *base = s_.data();
    size_t run_start = i_;
    while (true) {
      if (i_ >= n_) throw err("unterminated string");
      const unsigned char c = static_cast<unsigned char>(s_[i_]);
      if (c == '"') {
        out.append(base + run_start, i_ - run_start);
        ++i_;
        return out;
      }
      if (c == '\\') {
        out.append(base + run_start, i_ - run_start);
        ++i_;
        if (i_ >= n_) throw err("unterminated escape sequence");
        const char esc = s_[i_];
        switch (esc) {
          case '"': out.push_back('"'); ++i_; break;
          case '\\': out.push_back('\\'); ++i_; break;
          case '/': out.push_back('/'); ++i_; break;
          case 'b': out.push_back('\b'); ++i_; break;
          case 'f': out.push_back('\f'); ++i_; break;
          case 'n': out.push_back('\n'); ++i_; break;
          case 'r': out.push_back('\r'); ++i_; break;
          case 't': out.push_back('\t'); ++i_; break;
          case 'u': {
            ++i_;
            uint32_t cp = parse_hex4();
            if (cp >= 0xD800 && cp <= 0xDBFF) {
              // High surrogate -- must be followed by a low surrogate to
              // form one astral codepoint (e.g. an emoji or, more relevant
              // here, an obscure Cuneiform codepoint the byte_fallback path
              // exercises).
              if (i_ + 1 >= n_ || s_[i_] != '\\' || s_[i_ + 1] != 'u')
                throw err("unpaired UTF-16 high surrogate in \\u escape");
              i_ += 2;
              uint32_t lo = parse_hex4();
              if (lo < 0xDC00 || lo > 0xDFFF)
                throw err("invalid UTF-16 low surrogate in \\u escape");
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
              throw err("unpaired UTF-16 low surrogate in \\u escape");
            }
            append_utf8(out, cp);
            break;
          }
          default:
            throw err("invalid escape character");
        }
        run_start = i_;
        continue;
      }
      if (c < 0x20) throw err("unescaped control character in string");
      ++i_;
    }
  }

  uint32_t parse_hex4() {
    if (i_ + 4 > n_) throw err("truncated \\u escape");
    uint32_t v = 0;
    for (int k = 0; k < 4; ++k) {
      const char c = s_[i_ + k];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
      else throw err("invalid hex digit in \\u escape");
    }
    i_ += 4;
    return v;
  }

  Value parse_number() {
    const size_t start = i_;
    if (i_ < n_ && s_[i_] == '-') ++i_;
    if (i_ >= n_ || s_[i_] < '0' || s_[i_] > '9')
      throw err("invalid number");
    while (i_ < n_ && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
    if (i_ < n_ && s_[i_] == '.') {
      ++i_;
      if (i_ >= n_ || s_[i_] < '0' || s_[i_] > '9')
        throw err("invalid number: digits must follow '.'");
      while (i_ < n_ && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
    }
    if (i_ < n_ && (s_[i_] == 'e' || s_[i_] == 'E')) {
      ++i_;
      if (i_ < n_ && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
      if (i_ >= n_ || s_[i_] < '0' || s_[i_] > '9')
        throw err("invalid number: digits must follow exponent sign");
      while (i_ < n_ && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
    }
    Value v;
    v.type = Type::Number;
    v.num_v = std::strtod(s_.c_str() + start, nullptr);
    return v;
  }
};

}  // namespace

Value parse(const std::string &text) {
  Parser p(text);
  return p.parse_document();
}

}  // namespace json
}  // namespace npue
