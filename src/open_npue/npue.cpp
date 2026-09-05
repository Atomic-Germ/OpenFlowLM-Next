//===- npue.cpp ---------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- .npue reader. See npue.hpp and docs/04-model/npue-format.md.
// SPDX-License-Identifier: MIT
//
// The JSON directory is parsed with a small hand-written scanner rather than a
// dependency. It is not a general JSON parser and does not pretend to be: the
// directory is written by tools/npue.py with `separators=(",",":")`, so its
// shape is known exactly. A general parser would be a third-party dependency in
// a runtime whose selling point is a lean native binary -- the same reasoning
// that rules out vendoring a tokenizer (docs/00-overview).
//
// It fails loudly on anything it does not recognise.

#include "npue.hpp"

#include <cstring>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
// The container is mmapped, never read: a 400 MB .npue is faulted in by page
// as the encoder touches it, and the pages are shared between processes. Both
// platforms do that; only the spelling differs, and it is confined to the two
// functions below so nothing else in this file knows which one it is on.
#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace npue {
namespace {

#pragma pack(push, 1)
struct FileHeader {
  char magic[4];        // "NPUE"
  uint32_t version;     // 1
  uint32_t arch;        // 0 = BERT_ABS_GELU_POSTLN
  uint32_t flags;       // bit0 = pre-tiled
  uint64_t json_offset;
  uint64_t json_length;
  uint64_t data_offset; // 4096-aligned
  uint64_t data_length;
  uint8_t reserved[16]; // 16, not 24 -- see the spec's correction note
};
#pragma pack(pop)
static_assert(sizeof(FileHeader) == 64, "the .npue header is exactly 64 bytes");

// -- a scanner for exactly the JSON tools/npue.py emits -------------------

size_t skip_ws(const std::string &s, size_t i) {
  while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' ||
                          s[i] == '\r'))
    ++i;
  return i;
}

// Reads a JSON string starting at the opening quote. No escape handling: tensor
// names and dtype tags are [A-Za-z0-9_.], and the writer controls them.
std::string read_string(const std::string &s, size_t &i) {
  if (s[i] != '"') throw std::runtime_error(".npue: expected a JSON string");
  size_t start = ++i;
  while (i < s.size() && s[i] != '"') {
    if (s[i] == '\\')
      throw std::runtime_error(".npue: escapes are not supported in the "
                               "directory; the writer never emits them");
    ++i;
  }
  std::string out = s.substr(start, i - start);
  ++i;
  return out;
}

// Reads a scalar (number, bool, null) or a string, returning its literal text.
std::string read_scalar(const std::string &s, size_t &i) {
  i = skip_ws(s, i);
  if (s[i] == '"') return read_string(s, i);
  size_t start = i;
  while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']') ++i;
  return s.substr(start, i - start);
}

// Skips one value of any kind, tracking nesting.
void skip_value(const std::string &s, size_t &i) {
  i = skip_ws(s, i);
  if (s[i] == '"') { read_string(s, i); return; }
  if (s[i] == '{' || s[i] == '[') {
    int depth = 0;
    do {
      if (s[i] == '{' || s[i] == '[') ++depth;
      else if (s[i] == '}' || s[i] == ']') --depth;
      else if (s[i] == '"') { read_string(s, i); continue; }
      ++i;
    } while (i < s.size() && depth > 0);
    return;
  }
  read_scalar(s, i);
}

std::vector<int64_t> read_int_array(const std::string &s, size_t &i) {
  std::vector<int64_t> out;
  i = skip_ws(s, i);
  if (s[i] != '[') throw std::runtime_error(".npue: expected an array");
  ++i;
  while (true) {
    i = skip_ws(s, i);
    if (s[i] == ']') { ++i; break; }
    out.push_back(std::stoll(read_scalar(s, i)));
    i = skip_ws(s, i);
    if (s[i] == ',') ++i;
  }
  return out;
}

}  // namespace

File::File(const std::string &path) {
#if defined(_WIN32)
  std::wstring wpath(path.begin(), path.end());
  handle_file_ = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ,
                             nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                             nullptr);
  if (handle_file_ == INVALID_HANDLE_VALUE)
    throw std::runtime_error("cannot open " + path);

  LARGE_INTEGER sz{};
  GetFileSizeEx(handle_file_, &sz);
  size_ = static_cast<size_t>(sz.QuadPart);

  handle_map_ = CreateFileMappingW(handle_file_, nullptr, PAGE_READONLY, 0, 0,
                                   nullptr);
  if (!handle_map_) throw std::runtime_error("cannot map " + path);
  base_ = static_cast<const uint8_t *>(
      MapViewOfFile(handle_map_, FILE_MAP_READ, 0, 0, 0));
  if (!base_) throw std::runtime_error("cannot view " + path);
#else
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path);
  struct stat st {};
  if (::fstat(fd, &st) != 0) { ::close(fd); throw std::runtime_error("cannot stat " + path); }
  size_ = static_cast<size_t>(st.st_size);
  void *p = size_ ? ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0)
                  : MAP_FAILED;
  // The mapping holds its own reference to the file, so the descriptor is not
  // needed past this point -- which is why handle_file_/handle_map_ stay null
  // on this branch and the destructor has only munmap to do.
  ::close(fd);
  if (p == MAP_FAILED) throw std::runtime_error("cannot map " + path);
  base_ = static_cast<const uint8_t *>(p);
#endif

  if (size_ < sizeof(FileHeader)) throw std::runtime_error(path + ": truncated");
  FileHeader h{};
  std::memcpy(&h, base_, sizeof h);
  if (std::memcmp(h.magic, "NPUE", 4) != 0)
    throw std::runtime_error(path + ": not a .npue file");
  if (h.version != 1)
    throw std::runtime_error(path + ": version " + std::to_string(h.version) +
                             ", expected 1");
  if (h.data_offset % 4096 != 0)
    throw std::runtime_error(path + ": data_offset is not 4096-aligned");
  if (size_ != h.data_offset + h.data_length)
    throw std::runtime_error(path + ": size does not match the header");

  version_ = h.version;
  data_offset_ = h.data_offset;
  data_length_ = h.data_length;

  std::string js(reinterpret_cast<const char *>(base_ + h.json_offset),
                 h.json_length);

  // The directory is {"config":{...},"tensors":[{...},...]}.
  size_t i = skip_ws(js, 0);
  if (js[i] != '{') throw std::runtime_error(".npue: directory is not an object");
  ++i;
  while (true) {
    i = skip_ws(js, i);
    if (js[i] == '}') break;
    std::string key = read_string(js, i);
    i = skip_ws(js, i);
    if (js[i] != ':') throw std::runtime_error(".npue: expected ':'");
    ++i;

    if (key == "config") {
      i = skip_ws(js, i);
      ++i;  // '{'
      while (true) {
        i = skip_ws(js, i);
        if (js[i] == '}') { ++i; break; }
        std::string ck = read_string(js, i);
        i = skip_ws(js, i);
        ++i;  // ':'
        i = skip_ws(js, i);
        if (js[i] == '{' || js[i] == '[') {
          size_t start = i;
          skip_value(js, i);
          config_[ck] = js.substr(start, i - start);
        } else {
          config_[ck] = read_scalar(js, i);
        }
        i = skip_ws(js, i);
        if (js[i] == ',') ++i;
      }
    } else if (key == "tensors") {
      i = skip_ws(js, i);
      ++i;  // '['
      while (true) {
        i = skip_ws(js, i);
        if (js[i] == ']') { ++i; break; }
        ++i;  // '{'
        TensorInfo t;
        while (true) {
          i = skip_ws(js, i);
          if (js[i] == '}') { ++i; break; }
          std::string tk = read_string(js, i);
          i = skip_ws(js, i);
          ++i;  // ':'
          if (tk == "name") t.name = read_scalar(js, i);
          else if (tk == "role") t.role = read_scalar(js, i);
          else if (tk == "dtype") t.dtype = read_scalar(js, i);
          else if (tk == "logical_shape") t.logical_shape = read_int_array(js, i);
          else if (tk == "padded_shape") t.padded_shape = read_int_array(js, i);
          else if (tk == "offset") t.offset = std::stoull(read_scalar(js, i));
          else if (tk == "nbytes") t.nbytes = std::stoull(read_scalar(js, i));
          else if (tk == "layout_hash") t.layout_hash = read_scalar(js, i);
          else skip_value(js, i);
          i = skip_ws(js, i);
          if (js[i] == ',') ++i;
        }
        if (t.offset + t.nbytes > data_length_)
          throw std::runtime_error(".npue: tensor " + t.name +
                                   " runs past the data segment");
        tensors_[t.name] = t;
        i = skip_ws(js, i);
        if (js[i] == ',') ++i;
      }
    } else {
      skip_value(js, i);
    }
    i = skip_ws(js, i);
    if (js[i] == ',') ++i;
  }
}

File::~File() {
#if defined(_WIN32)
  if (base_) UnmapViewOfFile(base_);
  if (handle_map_) CloseHandle(handle_map_);
  if (handle_file_ && handle_file_ != INVALID_HANDLE_VALUE)
    CloseHandle(handle_file_);
#else
  if (base_) ::munmap(const_cast<uint8_t *>(base_), size_);
#endif
}

const TensorInfo &File::info(const std::string &name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end())
    throw std::runtime_error(".npue: no tensor named " + name);
  return it->second;
}

Span File::raw(const std::string &name) const {
  const TensorInfo &t = info(name);
  return Span{base_ + data_offset_ + t.offset, t.nbytes};
}

int64_t File::config_int(const std::string &key) const {
  return std::stoll(config_string(key));
}

double File::config_double(const std::string &key) const {
  return std::stod(config_string(key));
}

std::string File::config_string(const std::string &key) const {
  auto it = config_.find(key);
  if (it == config_.end())
    throw std::runtime_error(".npue: no config key " + key);
  return it->second;
}

}  // namespace npue
