// Private diagnostic: combine independently paired-input high/low weight results.
#pragma once
#include <cstddef>
#include <vector>
namespace open_qwen36::weighted_gemm {
struct Workspace {
    std::vector<float> output;
    template<class High, class Low>
    const float* run(size_t size, High high, Low low) {
        const float* h = high();
        output.assign(h, h + size);
        const float* l = low();
        for (size_t i = 0; i < size; ++i) output[i] += l[i];
        return output.data();
    }
};
}
