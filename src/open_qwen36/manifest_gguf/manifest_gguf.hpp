/// \file manifest_gguf.hpp
/// \brief GGUF-direct manifest module (isolated).
///
/// Provides the GGUF-direct twin manifest logic that is embedded within the
/// main Manifest struct. The Manifest has a `std::unique_ptr<Manifest> gguf`
/// field that holds a complete manifest for the f32-scale twin used when
/// loading .gguf model files.
///
/// This module is isolated from the main manifest code to allow the GGUF-direct
/// path to be built/tested independently.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace open_qwen36 {

/// Load a GGUF-direct twin manifest from JSON.
/// This is the same Manifest::parse logic but operating on a GGUF-specific
/// JSON fragment (with quant="q4_1_f32").
///
/// \param j      The JSON object (must have "manifest_version", "family", "spec",
///               "spec_hash", "build_key", "max_ctx_default", "hf_config_check",
///               "hf_config_defaults", "layout", "layers", "contexts", "kernels",
///               "layer_types", "tail", "globals", "per_row_globals", "embed_tensor",
///               "norm_tensor", "embed_scale", "lmhead_ops", "norm_bytes")
/// \param where  Debug label for error messages
/// \return A unique_ptr to the parsed Manifest, or throws on failure
std::unique_ptr<Manifest> load_gguf_twin(const nlohmann::json& j, const std::string& where);

// Create a GGUF-direct twin spec from a base spec, changing the quantization.
/// \param base_spec  The original ModelSpec
/// \param quant      The quantization type for the twin (e.g. "q4_1_f32")
/// \return A new ModelSpec with the changed quantization
ModelSpec make_gguf_twin_spec(const ModelSpec& base_spec, const std::string& quant);

// Emit a GGUF-direct twin manifest JSON fragment.
// This mirrors the logic in export_qwen36_kernels.py:
///   gguf = prog.pop("gguf", None)
//   if gguf:
//       m["gguf"] = manifest(replace(spec, quant="q4_1_f32"), max_ctx, key)
std::string emit_gguf_twin(const ModelSpec& spec, int max_ctx = 4096,
                           const std::string& key = "");

// Validate that a GGUF tensor's quant type is supported by the pool pack ops.
// Refuses K-quants (Q4_K, Q6_K, Q8_K) with a pointer to q4nx-build conversion.
bool gguf_quant_allowed(GgufType type);

// Get the chunk bytes for a GGUF tensor type, for pool packing.
size_t gguf_tensor_chunk_bytes(GgufType type);

} // namespace open_qwen36