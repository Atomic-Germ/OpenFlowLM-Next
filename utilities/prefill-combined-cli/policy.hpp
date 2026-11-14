// Private experiment: exact role matching, never select precision by shape alone.
#pragma once
#include "open_qwen36/manifest.hpp"
#include <stdexcept>

namespace open_qwen36::combined {
inline void require(bool valid) {
    if (!valid) throw std::runtime_error("combined prefill requires validated Qwen3.8 projection roles/geometry");
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
inline std::string low_kernel(const GemmBlockProgram& g, const Step& s,
                              size_t t, size_t k, size_t n) {
    require(t == 256 && !s.split);
    // validate() runs once at construction. Calls must still match a declared
    // role, including args: linear and full attention share an output shape.
    if (matches(s, g.ffn_program.at(0).kernel, g.ffn_program.at(0).args)) {
        require(k == 5120 && n == 34816);
        return "private_ffn_low";
    }
    if (matches(s, g.ffn_program.at(1).kernel, g.ffn_program.at(1).args)) {
        require(k == 17408 && n == 5120);
        return {};
    }
    if (matches(s, g.program.at(0).kernel, g.program.at(0).args)) {
        require(k == 5120 && n == (g.kind == "linear" ? 16384 : 14336));
        return g.kind == "linear" ? "private_entry_low" : "";
    }
    if (matches(s, g.program.at(1).kernel, g.program.at(1).args)) {
        require(k == 6144 && n == 5120);
        return g.kind == "linear" ? "private_out_low" : "";
    }
    throw std::runtime_error("undeclared combined prefill projection");
}
}
