/// \file npue_embedding.hpp
/// \brief NpueEmbedding: AutoEmbeddingModel backed by the NpuEmbeddings runtime.
/// \date 2026-09-03
/// \version 0.2.0
/// \note A SECOND embedding backend, additive to open_embedding. It does not
///       replace OpenGemma_Embedding and does not serve embed-gemma:300m.
///
/// SPDX-License-Identifier: MIT
///
/// WHAT THIS ADDS. open_embedding runs a CPU forward pass with a per-projection
/// bf16 matmul offloaded to the NPU, which is general: any new model shape
/// works as soon as an xclbin of that shape exists. This backend is the other
/// trade. A whole encoder layer is four GEMM dispatches over ONE resident
/// xclbin in ONE hw_context, with pre-tiled weights staged once and batch tiers
/// so a request is right-sized rather than padded. It needs a compiled design
/// per geometry, so it is less general and much faster -- and four design sets
/// cover seven models, because the sets are keyed by GEOMETRY and not by
/// fine-tune name. Generality by default, a fast path where a design exists.
///
/// The engine is a synced copy under src/open_npue/ -- see its SYNCED.md.
/// Edit it upstream, not here.
///
/// ---------------------------------------------------------------------------
/// WHY THIS HEADER IS A PIMPL, AND WHY THAT IS NOT STYLE.
///
/// The engine is header-defined: `Encoder`, `Stack`, `EmbedService` and every
/// host kernel they inline are `inline` functions in npue_encoder.hpp, and
/// roughly half of an encode is AVX2 intrinsics behind `#if defined(__AVX2__)`
/// with a correct scalar `#else`.
///
/// The first version of this header included npue_encoder.hpp directly. That
/// compiles, links, runs and RETURNS SLIGHTLY WRONG NUMBERS -- measured, on
/// bge-base, at 1-cos 1.04e-04 against the same engine in its own binary on
/// the same text, same container and same design set, byte for byte.
///
/// The mechanism: src/CMakeLists.txt gives `/arch:AVX2` to the open_npue
/// sources, because without it those kernels are 2.1-2.6x slower. It does not
/// give it to rest_handler.cpp -- that is the whole point of a per-source
/// flag. So every inline function from the engine instantiated in
/// rest_handler.cpp was compiled with __AVX2__ UNDEFINED, i.e. down the scalar
/// path, while the same functions in the open_npue objects were compiled with
/// it defined. Two different definitions of the same inline function is an ODR
/// violation, and the linker resolves it by silently keeping one COMDAT copy
/// per function -- a mixture, chosen by link order.
///
/// The scalar and AVX2 paths are both correct and reduce in different orders,
/// so the mixture is correct-looking and wrong: a correctly shaped, correctly
/// normed, deterministic vector that no downstream check can flag.
///
/// A PIMPL fixes it by construction. npue_encoder.hpp is now included by
/// EXACTLY ONE translation unit -- npue_embedding.cpp, which is in
/// OPEN_NPUE_SOURCES and therefore carries the same flags as the rest of the
/// engine. Nothing else in this tree can instantiate an engine function at a
/// different ISA level, because nothing else can see one.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "auto_embedding_model.hpp"

class NpueEmbedding : public AutoEmbeddingModel {
public:
    NpueEmbedding(flm_rt::device* npu_device_inst, std::string tag);
    ~NpueEmbedding() override;

    /// \brief Load a model directory.
    ///
    /// `model_path` is the directory ModelDownloader populated: the model
    /// author's OWN HuggingFace checkpoint, plus a packed `.npue` container.
    /// Nothing is re-hosted -- the weights a user gets are the author's bytes
    /// with the author's hash.
    void load_model(std::string model_path, json model_info,
                    bool enable_preemption = false) override;

    std::vector<float> embed(std::string& text,
                             embedding_task_type_t task_type) override;

    /// \brief Embed several texts in one call.
    ///
    /// NOT an override -- AutoEmbeddingModel::embed() takes one text, and
    /// widening the base class is a separate change that deserves to be judged
    /// on its own. It matters here because THIS ENGINE'S THROUGHPUT IS IN THE
    /// BATCH: it encodes `batch` sequences per dispatch over a resident
    /// xclbin, so a single text pays for a whole tier. Upstream measures a
    /// single text at 5.8x worse per text than a full batch. The REST handler
    /// already parses the whole `input` array before discarding it in a loop,
    /// so adopting this is one line there once the base class allows it.
    std::vector<float> embed_batch(const std::vector<std::string>& texts,
                                   embedding_task_type_t task_type,
                                   int64_t* tokens = nullptr);

    int64_t hidden() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
