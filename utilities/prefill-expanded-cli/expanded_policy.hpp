// Private experiment: extend only the validated Qwen3.8 projection roles.
#pragma once
#include "policy.hpp"

namespace open_qwen36::expanded {
inline void validate(const GemmBlockProgram& g, size_t hidden) {
    combined::validate(g, hidden);
    combined::require(g.eps == 1e-6 && g.act == "silu" && !g.sandwich);
}
inline void validate_manifest(const Manifest& m) {
    combined::require(m.hidden == 5120 && m.attn_kv_pair && m.linear_conv_f32);
    for (const auto& item : m.layer_types) validate(item.second.gemm_block, m.hidden);
}
inline std::string low_kernel(const GemmBlockProgram& g, const Step& s,
                              size_t t, size_t k, size_t n) {
    // Reuse exact argument/shape validation, including rejection of undeclared roles.
    const auto selected = combined::low_kernel(g, s, t, k, n);
    if (!selected.empty()) return selected;
    if (combined::matches(s, g.ffn_program.at(1).kernel, g.ffn_program.at(1).args))
        return "private_down_low";
    if (g.kind == "full" && combined::matches(s, g.program.at(0).kernel, g.program.at(0).args))
        return "private_attention_low";
    if (g.kind == "full" && combined::matches(s, g.program.at(1).kernel, g.program.at(1).args))
        return "private_out_low";
    throw std::runtime_error("undeclared expanded prefill projection");
}
}
