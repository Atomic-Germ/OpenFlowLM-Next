// Host decode control only: deterministic selection and BF16 row lookup.
// Neural operators (including final RMSNorm and LM head) stay on the NPU.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace decode_control {
inline uint32_t greedy(const float* logits, size_t count) {
    if (!count || count > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("greedy: invalid vocabulary size");
    uint32_t best = 0;
    for (size_t i=0; i<count; ++i) {
        if (!std::isfinite(logits[i])) throw std::runtime_error("greedy: nonfinite logit");
        if (logits[i] > logits[best]) best = static_cast<uint32_t>(i);
    }
    return best; // First (lowest) index wins exact ties, including signed zero.
}

inline std::vector<float> embedding(const std::filesystem::path& file, uint32_t token,
                                    size_t rows, size_t width) {
    const auto limit = static_cast<size_t>(std::numeric_limits<std::streamoff>::max());
    if (!rows || !width || token>=rows || width>limit/2 || rows>limit/(width*2))
        throw std::runtime_error("embed: invalid token or geometry");
    if (std::filesystem::file_size(file)!=rows*width*2)
        throw std::runtime_error("embed: file size does not match BF16 matrix");
    std::ifstream f(file,std::ios::binary);
    f.seekg(static_cast<std::streamoff>(token*width*2));
    std::vector<uint16_t> raw(width);
    if (!f.read(reinterpret_cast<char*>(raw.data()),static_cast<std::streamsize>(width*2)))
        throw std::runtime_error("embed: short row read");
    std::vector<float> result(width);
    for (size_t i=0; i<width; ++i) {
        uint32_t bits = static_cast<uint32_t>(raw[i]) << 16;
        std::memcpy(&result[i],&bits,sizeof(bits));
        if (!std::isfinite(result[i])) throw std::runtime_error("embed: nonfinite value");
    }
    return result;
}
} // namespace decode_control
