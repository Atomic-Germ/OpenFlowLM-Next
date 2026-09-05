/// \file q4nx_file.cpp
/// \brief Open reader for FLM's `.q4nx` weight container (see q4nx_file.hpp).
#include "open_qwen36/q4nx_file.hpp"

#include <cstring>
#include <stdexcept>

#include "nlohmann/json.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace open_qwen36 {

Q4nxFile::Q4nxFile(const std::string& path) : path_(path) {
#ifdef _WIN32
    HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) throw std::runtime_error("q4nx: cannot open " + path);
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(f, &sz)) { CloseHandle(f); throw std::runtime_error("q4nx: size of " + path); }
    HANDLE m = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) { CloseHandle(f); throw std::runtime_error("q4nx: cannot map " + path); }
    const void* p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (!p) { CloseHandle(m); CloseHandle(f); throw std::runtime_error("q4nx: cannot map view of " + path); }
    file_ = f;
    mapping_ = m;
    map_ = static_cast<const uint8_t*>(p);
    map_size_ = static_cast<size_t>(sz.QuadPart);
#else
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("q4nx: cannot open " + path);
    struct stat st;
    if (fstat(fd_, &st) != 0) { ::close(fd_); throw std::runtime_error("q4nx: size of " + path); }
    void* p = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd_, 0);
    if (p == MAP_FAILED) { ::close(fd_); throw std::runtime_error("q4nx: cannot map " + path); }
    map_ = static_cast<const uint8_t*>(p);
    map_size_ = static_cast<size_t>(st.st_size);
#endif
    if (map_size_ < 8) throw std::runtime_error("q4nx: " + path + " is not a container");
    uint64_t n = 0;
    std::memcpy(&n, map_, 8);
    if (n == 0 || n > (1u << 28) || 8 + n > map_size_) throw std::runtime_error("q4nx: bad header length in " + path);
    auto hdr = nlohmann::json::parse(map_ + 8, map_ + 8 + n, nullptr, false);
    if (!hdr.is_object()) throw std::runtime_error("q4nx: header of " + path + " is not JSON");
    data_base_ = 8 + static_cast<size_t>(n);
    for (auto it = hdr.begin(); it != hdr.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        const auto& v = it.value();
        TensorMeta t;
        t.dtype = v.value("dtype", "");
        for (const auto& d : v.at("shape")) t.shape.push_back(d.get<size_t>());
        t.start = v.at("data_offsets")[0].get<size_t>();
        t.end = v.at("data_offsets")[1].get<size_t>();
        if (data_base_ + t.end > map_size_) throw std::runtime_error("q4nx: tensor " + it.key() + " past EOF");
        tensors_.emplace(it.key(), std::move(t));
    }
    // 1.0.2 packs q4_1 in 5120 B chunks; 1.0.3 packs Q4_K in 4736 B ones and
    // needs a different dequant. Nothing in the header records the version,
    // so read it off a tensor — and refuse the format this reader can't do.
    for (const auto& [k, t] : tensors_) {
        if (t.dtype == "I8" && k != "lm_head.weight") {
            chunk_bytes_ = t.shape.back();
            if (chunk_bytes_ != 5120)
                throw std::runtime_error("q4nx: " + path + " has " + std::to_string(chunk_bytes_) +
                                         "-byte quant chunks (FLM 1.0.3 / Q4_K?); the open engine reads the "
                                         "1.0.2 q4_1 container only");
            break;
        }
    }
}

Q4nxFile::~Q4nxFile() {
#ifdef _WIN32
    if (map_) UnmapViewOfFile(map_);
    if (mapping_) CloseHandle(static_cast<HANDLE>(mapping_));
    if (file_) CloseHandle(static_cast<HANDLE>(file_));
#else
    if (map_) munmap(const_cast<uint8_t*>(map_), map_size_);
    if (fd_ >= 0) ::close(fd_);
#endif
}

void Q4nxFile::drop_pages() {
#ifdef _WIN32
    UnmapViewOfFile(map_);
    const void* p = MapViewOfFile(static_cast<HANDLE>(mapping_), FILE_MAP_READ, 0, 0, 0);
    if (!p) throw std::runtime_error("q4nx: cannot remap " + path_);
    map_ = static_cast<const uint8_t*>(p);
#else
    madvise(const_cast<uint8_t*>(map_), map_size_, MADV_DONTNEED);
#endif
}

const TensorMeta& Q4nxFile::meta(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("q4nx: no tensor " + name + " in " + path_);
    return it->second;
}

const uint8_t* Q4nxFile::raw(const std::string& name, size_t* nbytes) const {
    const TensorMeta& t = meta(name);
    if (nbytes) *nbytes = t.end - t.start;
    return map_ + data_base_ + t.start;
}

std::vector<float> Q4nxFile::bf16(const std::string& name) const {
    const TensorMeta& t = meta(name);
    if (t.dtype != "BF16") throw std::runtime_error("q4nx: " + name + " is " + t.dtype + ", not BF16");
    size_t n = (t.end - t.start) / 2;
    const uint8_t* p = map_ + data_base_ + t.start;
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        uint16_t u;
        std::memcpy(&u, p + 2 * i, 2);
        out[i] = bf16_to_f32(u);
    }
    return out;
}

std::vector<float> Q4nxFile::f32(const std::string& name) const {
    const TensorMeta& t = meta(name);
    if (t.dtype != "F32") throw std::runtime_error("q4nx: " + name + " is " + t.dtype + ", not F32");
    std::vector<float> out((t.end - t.start) / 4);
    std::memcpy(out.data(), map_ + data_base_ + t.start, out.size() * 4);
    return out;
}

void Q4nxFile::bf16_row(const std::string& name, size_t row, size_t dim, float* out) const {
    const TensorMeta& t = meta(name);
    if (t.dtype != "BF16") throw std::runtime_error("q4nx: " + name + " is " + t.dtype + ", not BF16");
    if ((row + 1) * dim * 2 > t.end - t.start) throw std::runtime_error("q4nx: row " + std::to_string(row) + " past " + name);
    const uint8_t* p = map_ + data_base_ + t.start + row * dim * 2;
    for (size_t i = 0; i < dim; ++i) {
        uint16_t u;
        std::memcpy(&u, p + 2 * i, 2);
        out[i] = bf16_to_f32(u);
    }
}

}  // namespace open_qwen36
