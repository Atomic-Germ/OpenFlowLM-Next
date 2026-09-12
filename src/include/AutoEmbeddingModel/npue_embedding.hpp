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
    NpueEmbedding(oflm_rt::device* npu_device_inst, std::string tag);
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

    /// The container's own prompt table, read at load, and EMPTY is the common case.
    /// Of the packers in npue_pack.cpp only two write one: prepare_model_gemma()
    /// copies the checkpoint's own table, and prepare_model_nomic() synthesises one
    /// (marked prompts_source, because it is not from the checkpoint). The BERT
    /// family and gte-multilingual-base write none, so their prompt_names() is empty
    /// and /v1/embeddings does not require the field. An earlier version of this
    /// comment claimed gte was non-empty; it is not, and nothing had checked.
    std::vector<std::string> prompt_names() const override;

    /// Container-driven: this backend prefixes exactly when the container
    /// declares a prompt table. The BERT family and gte declare none.
    bool supports_task_prompts() const override { return !prompt_names().empty(); }

    /// \brief Embed several texts in one call.
    ///
    /// An override now. It used to be a non-virtual extra, because widening
    /// AutoEmbeddingModel deserved to be judged on its own -- it has been, and
    /// the base class carries a default that loops over embed(). This is the
    /// override that makes the batch real: THIS ENGINE'S THROUGHPUT IS IN THE
    /// BATCH, since it encodes a whole tier of sequences per dispatch over a
    /// resident xclbin, so a single text pays for the tier either way.
    /// Measured on bge-base: sixteen texts cost 405 ms looped against 70 ms
    /// batched. `oflm bench-embed` reports that ratio per batch size, which is
    /// what turned it from a README note into a number.
    std::vector<float> embed_batch(const std::vector<std::string>& texts,
                                   embedding_task_type_t task_type,
                                   int64_t* tokens = nullptr) override;

    int64_t hidden() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
