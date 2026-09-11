/// \file weight_file.hpp
/// \brief The weight container behind the packers and the step loop: a `.q4nx`
///        container (Q4nxFile) or a GGUF (GgufFile).
///
/// The pack ops need the raw bytes of a tensor by name; the step loop needs
/// one embedding row per token as f32 (the row is stored quantized in a GGUF).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace open_qwen36 {

/// fp16 -> f32, exact (widen the mantissa, add the exponent bias difference).
inline float fp16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {                                   // subnormal: value = man * 2^-24, normalize
            int k = -1;
            for (uint32_t v = man; v; v >>= 1) ++k;
            bits = sign | (static_cast<uint32_t>(k + 103) << 23) | ((man << (23 - k)) & 0x7FFFFF);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000 | (man << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

/// f32 -> fp16, round-to-nearest-even. Used by the host requant of K-quant /
/// Q8_0 matmul tensors into the pool's fp16-scale q4 layout (pools.cpp).
/// Scales are normal-range values; subnormals and NaNs are not exercised.
inline uint16_t f32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man = x & 0x7FFFFF;
    if (exp <= 0) return static_cast<uint16_t>(sign);                    // flush to zero
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);         // inf
    man += 0x0FFFu + ((man >> 13) & 1);                                  // RNE
    if (man & 0x800000u) {
        man = 0;
        if (++exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((man >> 13) & 0x3FF));
}

class WeightFile {
public:
    virtual ~WeightFile() = default;
    virtual bool has(const std::string& name) const = 0;
    /// Raw bytes of a tensor (a view into the mapping; the container's own layout).
    virtual const uint8_t* raw(const std::string& name, size_t* nbytes = nullptr) const = 0;
    /// One embedding row as f32 (dequantized on the fly if the container stores it quantized).
    virtual void embed_row(const std::string& name, size_t row, size_t dim, float* out) const = 0;
    /// Release the mapping's resident pages after the pools are packed.
    virtual void drop_pages() = 0;
};

}  // namespace open_qwen36
