//===- engine_adapter.hpp --------------------------------*- C++ -*-===//
//
// open_whisper -- a whisper_engine (src/include/whisper/whisper_engine.hpp)
// over ow::Encoder + ow::Decoder. Phase 3b, issue #72: this is the seam that
// lets `oflm serve --asr 1` transcribe with the open encoder/decoder built in
// phases 2b/3, alongside (never in place of) the closed whisper_npu engine.
//
// Only forward-declares ow::Encoder/ow::Decoder, deliberately: this header is
// included from src/common/whisper/whisper_engine_select.cpp, a translation
// unit that must NOT need open_npue's npu_device.hpp on its include path (the
// per-source INCLUDE_DIRECTORIES that gives open_whisper's own .cpp files
// that path, in src/CMakeLists.txt, does not extend to this file's callers).
// Full types are pulled in only by engine_adapter.cpp, which has that path.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "whisper/whisper_engine.hpp"

namespace ow {
class Encoder;
class Decoder;
}  // namespace ow

namespace open_whisper {

// Owns one Encoder + one Decoder for the lifetime of a loaded ASR model. Not
// reentrant -- the same discipline as every other engine behind whisper_engine.
class OpenWhisperEngine final : public whisper_engine {
public:
  // `kernels_dir_hint` is passed straight to ow::KernelSet::resolve_dir --
  // empty means "look at OFLM_WHISPER_KERNELS_DIR, else <model_dir>/open_kernels".
  // `vocab_padded` is Whisper_Config's own padded vocab width
  // (Whisper_Config::from_pretrained rounds vocab_size up to a multiple of
  // 32); the constructor refuses if it disagrees with the decoder's own
  // DecoderGeometry::vocab_padded rather than assuming the two are the same
  // build's numbers.
  OpenWhisperEngine(const std::string &model_dir, const std::string &kernels_dir_hint,
                    int64_t vocab_padded);
  ~OpenWhisperEngine() override;

  void encode_audio(buffer<bf16> &mel_feature) override;
  buffer<bf16> decode_audio(int last_id) override;
  void clear_context() override;
  int get_current_context_length() override;
  std::string describe() const override;
  bool is_open() const override { return true; }
  std::string config_summary() const override;

private:
  std::string kernels_dir_;
  int64_t vocab_padded_ = 0;
  std::unique_ptr<ow::Encoder> encoder_;
  std::unique_ptr<ow::Decoder> decoder_;
};

}  // namespace open_whisper
