// open_diffusion schedule: the per-step-count host inputs, for step counts other than the
// bundle's. klein_pipeline.py's sigmas / timestep_features / dt_params. The sigmas and dts
// are float32 as numpy computes them, bit for bit. The features' exp / cos / sin are taken
// in double and rounded: numpy's float32 exp is not correctly rounded (53 of the 128
// frequencies are an ulp off), so an exact match would mean copying its SIMD kernel. The
// open_diffusion_schedule CTest holds this file to the bundle's own tf_<R>.bin / dt_<R>.bin:
// the dts exactly, the features within 2^-8 (5 of 131,584 words differ, by one rounding).
//
// Setup, like the noise: 256 cosines and one float per step, written once per step count.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace open_diffusion {
namespace schedule {

constexpr int kTfDim = 256;                  // Timesteps(256, flip_sin_to_cos)
constexpr int kTfRows = 512;                 // the modulation GEMM's M: steps padded to 512
constexpr int kEl = 3072;                    // dit_ew's row element
constexpr int kDtSlot = 4;                   // vectors per step: slack, dt, slack, pad

inline uint16_t bf16_bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7FFF + ((u >> 16) & 1);           // round to nearest even (ml_dtypes' bfloat16)
    return static_cast<uint16_t>(u >> 16);
}

inline float bf16_round(float x) {
    uint32_t u = static_cast<uint32_t>(bf16_bits(x)) << 16;
    float y;
    std::memcpy(&y, &u, 4);
    return y;
}

// diffusers' compute_empirical_mu (FLUX.2), in double as Python computes it.
inline double empirical_mu(int image_seq_len, int num_steps) {
    const double a1 = 8.73809524e-05, b1 = 1.89833333;
    const double a2 = 0.00016927, b2 = 0.45666666;
    if (image_seq_len > 4300) return a2 * image_seq_len + b2;
    double m_200 = a2 * image_seq_len + b2;
    double m_10 = a1 * image_seq_len + b1;
    double a = (m_200 - m_10) / 190.0;
    return a * num_steps + (m_200 - 200.0 * a);
}

// FlowMatchEulerDiscreteScheduler's sigmas, terminal 0 appended: steps + 1 values.
inline std::vector<float> sigmas(int image_tokens, int steps) {
    std::vector<float> out;
    const double start = 1.0, stop = 1.0 / steps;
    const double step = steps > 1 ? (stop - start) / (steps - 1) : 0.0;
    const float em = static_cast<float>(std::exp(empirical_mu(image_tokens, steps)));
    for (int i = 0; i < steps; ++i) {
        // np.linspace: start + i * step, the last one exactly stop; then float32
        float s = static_cast<float>(i == steps - 1 ? stop : start + i * step);
        float d = 1.0f / s - 1.0f;
        out.push_back(em / (em + d));
    }
    out.push_back(0.0f);
    return out;
}

// TF: [512, 256] bf16 bits + 512 slack; row s = step s's timestep features.
inline std::vector<uint16_t> timestep_features(const std::vector<float>& sig, int steps) {
    const int half = kTfDim / 2;
    std::vector<float> freqs(half);
    const float k = static_cast<float>(-std::log(10000.0));
    for (int i = 0; i < half; ++i)
        freqs[i] = static_cast<float>(std::exp(static_cast<double>(k * static_cast<float>(i) / static_cast<float>(half))));
    std::vector<uint16_t> tf(static_cast<size_t>(kTfRows) * kTfDim + 512, 0);
    for (int s = 0; s < steps; ++s) {
        // the timestep as the transformer sees it: bf16(bf16(bf16(1000 sigma) / 1000) * 1000)
        float t = bf16_round(sig[s] * 1000.0f);
        t = bf16_round(t / 1000.0f);
        t = bf16_round(t * 1000.0f);
        for (int i = 0; i < half; ++i) {
            double ang = static_cast<double>(t * freqs[i]);     // the float32 product, as numpy's
            tf[static_cast<size_t>(s) * kTfDim + i] = bf16_bits(static_cast<float>(std::cos(ang)));
            tf[static_cast<size_t>(s) * kTfDim + half + i] = bf16_bits(static_cast<float>(std::sin(ang)));
        }
    }
    return tf;
}

// DT: per step a dit_ew parameter run holding dt = sigma_{s+1} - sigma_s as fp32 (two bf16
// words at vector 1 of the step's slot).
inline std::vector<uint16_t> dt_params(const std::vector<float>& sig, int steps) {
    std::vector<uint16_t> p(static_cast<size_t>(steps) * kDtSlot * kEl, 0);
    for (int s = 0; s < steps; ++s) {
        float dt = sig[s + 1] - sig[s];
        std::memcpy(&p[static_cast<size_t>(s * kDtSlot + 1) * kEl], &dt, 4);
    }
    return p;
}

}  // namespace schedule
}  // namespace open_diffusion
