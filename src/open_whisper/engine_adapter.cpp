//===- engine_adapter.cpp -------------------------------------*- C++ -*-===//
// open_whisper -- see engine_adapter.hpp. SPDX-License-Identifier: MIT
//
#include "engine_adapter.hpp"

#include <stdexcept>
#include <vector>

#include "decoder.hpp"
#include "encoder.hpp"
#include "kernels.hpp"

namespace open_whisper {
namespace {
// Whisper's own mel geometry: 128 channels x 3000 frames (30 s at a 10 ms
// hop). Not exposed by ow::Weights::Geometry (that struct is about the
// encoder's internal d_model/n_mel tiling, not the raw feature shape), so it
// is named here once, matching whisper_engine.hpp's own doc comment on
// encode_audio()'s `mel_feature` and modeling_whisper.cpp's `_preprocess_audio`.
constexpr int64_t kMelChannels = 128;
constexpr int64_t kMelFrames = 3000;
}  // namespace

OpenWhisperEngine::OpenWhisperEngine(const std::string &model_dir,
                                     const std::string &kernels_dir_hint,
                                     int64_t vocab_padded)
    : kernels_dir_(ow::KernelSet::resolve_dir(kernels_dir_hint, model_dir)),
      vocab_padded_(vocab_padded),
      encoder_(std::make_unique<ow::Encoder>(model_dir, kernels_dir_hint)),
      decoder_(std::make_unique<ow::Decoder>(model_dir)) {
  // Both sides pad "51866 real tokens" up to a multiple of 32, independently:
  // Whisper_Config::from_pretrained does it from config.json's vocab_size,
  // DecoderGeometry::vocab_padded is a compile-time constant this decoder
  // was written against. They agree today (51872) because both start from
  // the same 51866 -- but nothing enforces that at the type level, and a
  // silent mismatch here would truncate or overrun decode_audio()'s returned
  // buffer<bf16>, which is exactly this project's "fails open" class: a
  // wrong-shaped, plausible-looking transcript with no error anywhere.
  if (vocab_padded_ != ow::DecoderGeometry::vocab_padded) {
    throw std::runtime_error(
        "open whisper engine: Whisper_Config's padded vocab width (" +
        std::to_string(vocab_padded_) +
        ") does not match this decoder's own (" +
        std::to_string(ow::DecoderGeometry::vocab_padded) +
        "); this build's open decoder is whisper-large-v3-turbo only");
  }
}

OpenWhisperEngine::~OpenWhisperEngine() = default;

void OpenWhisperEngine::encode_audio(buffer<bf16> &mel_feature) {
  const size_t n = static_cast<size_t>(kMelChannels * kMelFrames);
  if (mel_feature.size() != n) {
    throw std::runtime_error(
        "open whisper engine: encode_audio expected " + std::to_string(n) +
        " mel values (128x3000), got " + std::to_string(mel_feature.size()));
  }
  // bf16 -> fp32: bfloat16_t's operator float() is exact widening (zero-extend
  // the mantissa), the same conversion ow::from_bf16 performs -- so this does
  // not introduce a second rounding rule on top of whatever
  // Whisper::_preprocess_audio already rounded the mel to.
  std::vector<float> mel_f32(n);
  for (size_t i = 0; i < n; ++i) mel_f32[i] = static_cast<float>(mel_feature[i]);

  encoder_->encode(mel_f32.data());
  // The encoder's own xkv() buffer is owned by encoder_ (kept alive for as
  // long as this OpenWhisperEngine is), which is exactly what Decoder::
  // set_encoder_output()'s contract requires -- it does not copy.
  decoder_->set_encoder_output(encoder_->xkv().data());
}

buffer<bf16> OpenWhisperEngine::decode_audio(int last_id) {
  std::vector<float> logits_f32(static_cast<size_t>(ow::DecoderGeometry::vocab_padded));
  decoder_->step(last_id, logits_f32.data());

  buffer<bf16> out(static_cast<size_t>(vocab_padded_));
  // fp32 -> bf16, round-to-nearest-even: bfloat16_t's explicit float
  // constructor implements exactly that (see biovault_bfloat16.h), the same
  // rule as ow::to_bf16. The -inf tail (indices [vocab, vocab_padded)) rounds
  // to -inf's own bf16 bit pattern, which still compares below every finite
  // logit, so the pad still cannot win an argmax or a sample downstream.
  for (int64_t i = 0; i < vocab_padded_; ++i)
    out[static_cast<size_t>(i)] = bf16(logits_f32[static_cast<size_t>(i)]);
  return out;
}

void OpenWhisperEngine::clear_context() { decoder_->clear_context(); }

int OpenWhisperEngine::get_current_context_length() {
  return static_cast<int>(decoder_->position());
}

std::string OpenWhisperEngine::describe() const {
  return "open (" + kernels_dir_ + ")";
}

}  // namespace open_whisper
