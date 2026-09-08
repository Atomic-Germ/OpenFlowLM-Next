/// \file gguf_file.hpp
/// \brief Open reader for GGUF weight files (the GGUF-direct path).
///
/// A GGUF (v2/v3) is a header of typed key-value metadata, then one tensor
/// info per tensor (name, dims, type, offset), then the data, block-quantized
/// in llama.cpp's own layouts. Unlike the `.q4nx` container's pool chunks, the
/// blocks are NOT what the NPU kernels stream: the packers (pools.cpp
/// `std_perm_gguf`) permute the codes into the chunk's interleaved order and
/// widen the fp16 block scales EXACTLY to f32 (fp16 -> f32 is lossless; see
/// open_kernels/gguf_pool.py), so the pool never requantizes or narrows.
///
/// Matmul tensors the pack ops accept: Q4_0 / Q4_1 (Q8_0 for the q8 lm_head,
/// MoE families). Anything else (the K-quants, IQ quants) is refused at pack
/// time with a pointer to q4nx-build. The embedding row is dequantized per
/// token on the host; Q4_K and Q6_K are supported there (they are what most
/// quants use for token_embd), mirroring llama.cpp's dequant loops.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "open_qwen36/weight_file.hpp"

namespace open_qwen36 {

class GgufFile final : public WeightFile {
public:
    enum class Type : uint32_t {
        F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
        Q2_K = 10, Q3_K_S = 11, Q3_K_M = 12, Q3_K_L = 13, Q4_K_S = 14, Q4_K_M = 15,
        Q5_K_S = 16, Q5_K_M = 17, Q6_K = 18, IQ2_XXS = 19, IQ2_XS = 20, Q2_K_S = 21,
        Q3_K_XS = 22, IQ3_XXS = 23, Q8_K = 24, IQ4_NL = 28, IQ4_XS = 29, BF16 = 30,
    };

    struct TensorInfo {
        Type type = Type::F32;
        std::vector<uint64_t> dims;      // fastest first, as GGUF stores them
        uint64_t offset = 0;             // relative to the (aligned) data section
        uint64_t nvalues() const { uint64_t n = 1; for (uint64_t d : dims) n *= d; return n; }
    };

    explicit GgufFile(const std::string& path);
    ~GgufFile() override;
    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    bool has(const std::string& name) const override { return tensors_.count(gguf_name(name)) != 0; }
    /// Raw bytes of a tensor (the GGUF's own block layout; a view into the mapping).
    const uint8_t* raw(const std::string& name, size_t* nbytes = nullptr) const override;
    /// One embedding row as f32, dequantized from whatever type stores it.
    void embed_row(const std::string& name, size_t row, size_t dim, float* out) const override;
    /// Release the mapping's resident pages (madvise DONTNEED) after the pools are packed.
    void drop_pages() override;

    const TensorInfo& tensor(const std::string& name) const;
    /// HF-style name -> GGUF name (the manifest speaks the q4nx/HF convention;
    /// a GGUF speaks llama.cpp's blk.N.attn_q one). Unknown names pass through.
    static std::string gguf_name(const std::string& name);
    /// GGUF metadata: string / u64 accessors (the config derivation's source).
    bool kv_has(const std::string& key) const { return meta_.count(key) != 0; }
    std::string kv_str(const std::string& key) const;
    uint64_t kv_u64(const std::string& key) const;
    double kv_f64(const std::string& key) const;

    static const char* type_name(Type t);

private:
    std::string path_;
    const uint8_t* map_ = nullptr;
    size_t map_size_ = 0;
    size_t data_base_ = 0;
    std::map<std::string, TensorInfo> tensors_;
    std::map<std::string, std::string> meta_;       // everything as strings (the derivation's view)
#ifdef _WIN32
    void* file_ = nullptr;
    void* mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

}  // namespace open_qwen36
