/// \file q4nx_file.hpp
/// \brief Open reader for FLM's `.q4nx` weight container (format 1.0.2, q4_1).
/// \note Replaces the closed q4_npu_eXpress reader on the open Qwen3.6 path.
///
/// The container is a safetensors file: an 8-byte header length, a JSON header
/// of tensor name -> {dtype, shape, data_offsets}, then the data. BF16 / F32
/// tensors are plain row-major. Quantized tensors are packed 8192-value chunks
/// (5120 B q4_1 with bf16 scale + min per 32-block; 8704 B q8 for lm_head),
/// stored in the file's raster order — the NPU pools reorder them (pools.hpp).
///
/// The file is memory-mapped and read on demand: the whole 22 GB is never
/// resident on the host, only what the packers touch while filling device
/// buffers, plus the embedding rows read per token.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace open_qwen36 {

struct TensorMeta {
    std::string dtype;
    std::vector<size_t> shape;
    size_t start = 0, end = 0;  // data_offsets, relative to the data section
};

class Q4nxFile {
public:
    explicit Q4nxFile(const std::string& path);
    ~Q4nxFile();
    Q4nxFile(const Q4nxFile&) = delete;
    Q4nxFile& operator=(const Q4nxFile&) = delete;

    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    const TensorMeta& meta(const std::string& name) const;
    /// Raw bytes of a tensor (a view into the mapping).
    const uint8_t* raw(const std::string& name, size_t* nbytes = nullptr) const;
    /// A BF16 tensor decoded to f32.
    std::vector<float> bf16(const std::string& name) const;
    std::vector<float> f32(const std::string& name) const;
    /// One row of a BF16 [rows, dim] tensor as f32 (the embedding lookup).
    void bf16_row(const std::string& name, size_t row, size_t dim, float* out) const;

    /// Release the mapping's resident pages (unmap + map again). After the
    /// pools are packed, ~22 GB of file pages sit in the working set with no
    /// further use beyond the embedding rows; on a box holding 21 GB of NPU
    /// buffers that is the difference between fitting and paging.
    void drop_pages();

    size_t chunk_bytes() const { return chunk_bytes_; }
    const std::string& path() const { return path_; }

private:
    std::string path_;
    const uint8_t* map_ = nullptr;
    size_t map_size_ = 0;
    size_t data_base_ = 0;
    size_t chunk_bytes_ = 5120;
    std::unordered_map<std::string, TensorMeta> tensors_;
#ifdef _WIN32
    void* file_ = nullptr;
    void* mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

inline float bf16_to_f32(uint16_t u) {
    uint32_t w = static_cast<uint32_t>(u) << 16;
    float f;
    std::memcpy(&f, &w, 4);
    return f;
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);  // round to nearest even
}

}  // namespace open_qwen36
