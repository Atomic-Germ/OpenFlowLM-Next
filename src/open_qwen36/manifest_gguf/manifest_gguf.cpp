/// \file manifest_gguf.cpp
/// \brief GGUF-direct twin manifest implementation (isolated module).
///
/// See manifest_gguf.hpp for the interface declaration.
#include "manifest_gguf.hpp"

#include <fstream>
#include <stdexcept>

namespace open_qwen36 {

std::unique_ptr<Manifest> load_gguf_twin(const nlohmann::json& j, const std::string& where) {
    // Minimal parsing: only the fields needed for the GGUF-direct twin.
    // The full Manifest::parse is called from the engine when the twin is loaded.
    auto m = std::make_unique<Manifest>();
    
    // Extract basic fields that the GGUF twin needs
    if (j.contains("manifest_version")) m->version = j["manifest_version"].get<int>();
    if (j.contains("family")) m->family = j["family"].get<std::string>();
    if (j.contains("spec_hash")) m->spec_hash = j["spec_hash"].get<std::string>();
    if (j.contains("build_key")) m->build_key = j["build_key"].get<std::string>();
    if (j.contains("max_ctx_default")) m->max_ctx_default = j["max_ctx_default"].get<size_t>();
    
    // Layout fields
    if (j.contains("layout")) {
        const auto& lay = j["layout"];
        if (lay.contains("hidden")) m->hidden = lay["hidden"].get<size_t>();
        if (lay.contains("vocab")) m->vocab = lay["vocab"].get<size_t>();
        if (lay.contains("real_vocab")) m->real_vocab = lay["real_vocab"].get<size_t>();
        if (lay.contains("chunk_bytes")) m->chunk_bytes = lay["chunk_bytes"].get<size_t>();
        if (lay.contains("pool_bytes")) m->pool_bytes = lay["pool_bytes"].get<size_t>();
        if (lay.contains("lmhead_pool_bytes")) m->lmhead_pool_bytes = lay["lmhead_pool_bytes"].get<size_t>();
        if (lay.contains("kv_row")) m->kv_row = lay["kv_row"].get<size_t>();
        if (lay.contains("ptab_row")) m->ptab_row = lay["ptab_row"].get<size_t>();
        if (lay.contains("rotary_dim")) m->rotary_dim = lay["rotary_dim"].get<size_t>();
        if (lay.contains("rope_theta")) m->rope_theta = lay["rope_theta"].get<double>();
        if (lay.contains("rope_inv_freq")) m->rope_inv_freq = lay["rope_inv_freq"].get<std::vector<double>>();
        if (lay.contains("rout_idx_off")) m->rout_idx_off = lay["rout_idx_off"].get<size_t>();
    }
    
    // Layer types and kernels (minimal)
    if (j.contains("layers")) m->layers = j["layers"].get<std::vector<std::string>>();
    if (j.contains("layer_types")) {
        const auto& lt = j["layer_types"];
        // Parse layer types minimally
        for (auto it = lt.begin(); it != lt.end(); ++it) {
            // Store layer type name and basic info
            m->layer_types[it.key()] = {};
        }
    }
    
    if (j.contains("contexts")) {
        for (auto it = j["contexts"].begin(); it != j["contexts"].end(); ++it) {
            m->contexts[it.key()] = it.value().get<std::string>();
        }
    }
    
    if (j.contains("kernels")) {
        const auto& krn = j["kernels"];
        for (auto it = krn.begin(); it != krn.end(); ++it) {
            // Minimal kernel descent parsing
            auto& desc = m->kernels[it.key()];
            if (it.value().contains("context")) desc.context = it.value()["context"].get<std::string>();
            if (it.value().contains("insts")) desc.insts = it.value()["insts"].get<std::string>();
        }
    }
    
    if (j.contains("tail")) {
        // Minimal tail parsing
    }
    
    if (j.contains("globals")) {
        for (auto it = j["globals"].begin(); it != j["globals"].end(); ++it) {
            m->globals[it.key()] = it.value().get<uint64_t>();
        }
    }
    
    if (j.contains("per_row_globals")) {
        for (auto it = j["per_row_globals"].begin(); it != j["per_row_globals"].end(); ++it) {
            // Minimal parsing
        }
    }
    
    if (j.contains("embed_tensor")) m->embed_tensor = j["embed_tensor"].get<std::string>();
    if (j.contains("norm_tensor")) m->norm_tensor = j["norm_tensor"].get<std::string>();
    if (j.contains("embed_scale")) m->embed_scale = j["embed_scale"].get<double>();
    if (j.contains("norm_bytes")) m->norm_bytes = j["norm_bytes"].get<size_t>();
    if (j.contains("lmhead_ops")) {
        // Minimal lmhead_ops parsing
        const auto& lo = j["lmhead_ops"];
        for (auto it = lo.begin(); it != lo.end(); ++it) {
            m->lmhead_ops.push_back({it.key(), it.value()});
        }
    }
    
    return m;
}

ModelSpec make_gguf_twin_spec(const ModelSpec& base_spec, const std::string& quant) {
    // Create a new spec with the quantization changed to the GGUF twin format.
    // This mirrors the logic in export_qwen36_kernels.py:
    //   m["gguf"] = manifest(replace(spec, quant="q4_1_f32"), max_ctx, key)
    ModelSpec result = base_spec;
    result.quant = quant;
    return result;
}

std::string emit_gguf_twin(const ModelSpec& spec, int max_ctx, const std::string& key) {
    // Emit a GGUF-direct twin manifest JSON fragment.
    // This mirrors the logic in export_qwen36_kernels.py:
    //   if gguf:
    //       m["gguf"] = manifest(replace(spec, quant="q4_1_f32"), max_ctx, key)
    
    // Build the base manifest and then replace the quant
    auto m = manifest(spec, max_ctx, key);
    
    // Pop the gguf key if present (we're creating the twin, not the parent)
    auto gguf = m.pop("gguf", nullptr);
    
    // Change quant to q4_1_f32 for the twin
    if (gguf) {
        // The twin uses q4_1_f32 quantization
        m["gguf"] = manifest(make_gguf_twin_spec(spec, "q4_1_f32"), max_ctx, key);
    }
    
    return dumps(m);
}

bool gguf_quant_allowed(GgufType type) {
    // Q4_0, Q4_1, Q8_0 are supported directly through the pool path.
    // Q4_K, Q6_K, Q8_K require q4nx-build conversion.
    switch (type) {
        case GgufType::Q4_0:
        case GgufType::Q4_1:
        case GgufType::Q8_0:
            return true;
        case GgufType::Q4_K:
        case GgufType::Q6_K:
        case GgufType::Q8_K:
        default:
            return false;
    }
}

size_t gguf_tensor_chunk_bytes(GgufType type) {
    // Return the chunk size for a GGUF tensor type, matching the pool layout.
    switch (type) {
        case GgufType::Q4_0:
        case GgufType::Q4_1:
            return 6144;  // 5120 + 1024 for min plane (Q4_1) or nibble offset (Q4_0)
        case GgufType::Q8_0:
            return 9216;  // q8 lm_head pool chunk
        default:
            return 0;  // unsupported / requires q4nx-build
    }
}

} // namespace open_qwen36