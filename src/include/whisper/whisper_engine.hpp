/// \file whisper_engine.hpp
/// \brief The contract between the Whisper host (audio, mel, token loop) and an engine
/// \note Whisper is not a causal_lm: it encodes a 30 s window once, then decodes one token
///       at a time while attending to that window. The host in modeling_whisper.cpp needs
///       exactly the calls below and nothing else, so this is the whole seam. The closed
///       `whisper_npu` is a concrete class from a prebuilt library and cannot derive from
///       this without changing its ABI; it is wrapped instead (whisper_engine_closed.cpp).
#pragma once
#include <memory>
#include <string>
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "typedef.hpp"

class whisper_engine {
public:
    virtual ~whisper_engine() = default;

    /// \brief Encode one window and make it the cross-attention source for every later
    ///        decode_audio() until the next call.
    /// \param mel_feature [128][3000] log-mel, row-major, as Whisper::_preprocess_audio writes it
    virtual void encode_audio(buffer<bf16>& mel_feature) = 0;

    /// \brief Feed one token, return the logits for the next one.
    /// \return vocab_size logits as Whisper_Config pads it (a multiple of 32); the pad tail
    ///         must never win a sample
    virtual buffer<bf16> decode_audio(int last_id) = 0;

    /// \brief Reset the decoder's self-attention state. The encoded window survives.
    virtual void clear_context() = 0;

    virtual int get_current_context_length() = 0;

    /// \brief One line naming the engine and what selected it, for the load log.
    virtual std::string describe() const = 0;
};

/// \brief Build the engine for a Whisper model directory.
///
/// OFLM_WHISPER_ENGINE=open|closed forces one. UNSET, the model directory decides: the
/// open engine when it holds `model.open.safetensors` AND a kernel set resolves, the
/// closed engine when it holds `model.q4nx`, and an error naming both when neither is
/// there. A request for an engine this build or this directory cannot provide is an error
/// that names what is missing, never a quiet fallback -- both engines return a transcript,
/// so a fallback would be invisible. The selector logs which rule fired
/// (whisper_engine_select.cpp).
std::unique_ptr<whisper_engine> make_whisper_engine(const std::string& model_path,
                                                    Whisper_Config& config,
                                                    oflm_rt::device* device,
                                                    bool enable_preemption);
