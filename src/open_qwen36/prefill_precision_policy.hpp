// Paired prefill currently supports the validated wide dense Qwen3.5 geometry.
#pragma once
#include "open_qwen36/manifest.hpp"
#include <stdexcept>

namespace open_qwen36::prefill_precision_policy {
inline void require(bool valid) {
    if (!valid) throw std::runtime_error("paired prefill requires validated Qwen3.8 projection roles/geometry");
}
inline bool matches(const Step& s, const std::string& kernel,
                    const std::vector<std::string>& args) {
    return s.op == "run" && s.kernel == kernel && s.args == args && !s.split;
}
inline void validate(const GemmBlockProgram& g, size_t hidden) {
    require(hidden == 5120 && g.t == 256 && g.ff == 17408 && g.norm_groups == 1 &&
            g.program.size() == 2 && g.ffn_program.size() == 2 && !g.out_split &&
            g.shared_program.empty() && g.moe_kernel.empty() && !g.moe_batch.present());
    require(matches(g.ffn_program[0], "gemm_n34816_k5120", {"gffn_ug_w", "gemm_x_k5120", "gemm_y_n34816"}));
    require(matches(g.ffn_program[1], "gemm_n5120_k17408", {"gffn_down_w", "gemm_x_k17408", "gemm_y_n5120"}));
    if (g.kind == "linear") {
        require(g.qkv_dim == 10240 && g.vw == 6144 && g.key_heads == 16 &&
                g.value_heads == 48 && g.head_dim == 128 && g.conv_kernel == 4 && g.conv_f32);
        require(matches(g.program[0], "gemm_n16384_k5120", {"gqkvz_w", "gemm_x_k5120", "gemm_y_n16384"}));
        require(matches(g.program[1], "gemm_n5120_k6144", {"gout_w", "gemm_x_k6144", "gemm_y_n5120"}));
    } else {
        require(g.kind == "full" && g.qw == 6144 && g.kvw == 1024 && g.nh == 24 &&
                g.kvh == 4 && g.hd == 256 && g.rot == 64);
        require(matches(g.program[0], "gemm_n14336_k5120", {"gqkvg_w", "gemm_x_k5120", "gemm_y_n14336"}));
        require(matches(g.program[1], "gemm_n5120_k6144", {"go_w", "gemm_x_k6144", "gemm_y_n5120"}));
    }
}
inline void validate(const Manifest& m, const GemmBlockProgram& g) {
    if (g.low_kernels.empty()) return;
    require(m.version >= 4 && m.family == "qwen35" && m.attn_kv_pair && m.linear_conv_f32);
    validate(g, m.hidden);
    require(g.eps == 1e-6 && g.act == "silu" && !g.sandwich && g.low_kernels.size() == 4);
    for (const auto* program : {&g.program, &g.ffn_program})
        for (const auto& step : *program) {
            const auto link = g.low_kernels.find(step.kernel);
            // A same-ABI low kernel for another projection can hang or produce
            // plausible wrong values. Bind the companion to this exact shape/role.
            require(link != g.low_kernels.end() && link->second == step.kernel + "_low");
            const auto low = m.kernels.find(link->second);
            require(low != m.kernels.end());
            const auto& high = m.kernels.at(step.kernel);
            require(low->second.weight_component == "bf16_residual" && low->second.patch.empty() &&
                    high.weight_component.empty() && low->second.context != high.context &&
                    low->second.insts != high.insts);
        }
}
}
