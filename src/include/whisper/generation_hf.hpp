/// \file generation_hf.hpp
/// \brief The `hf` Whisper decoding protocol: a faithful, scoped port of
///        transformers' WhisperForConditionalGeneration.generate() for greedy
///        decoding (num_beams=1, do_sample=False), used by Whisper::generate()
///        when OFLM_WHISPER_PROTOCOL=hf.
/// \note This header is deliberately free of XRT/buffer<bf16>/tokenizer
///       dependencies -- every function here works on plain std::vector<float>
///       logits and std::vector<int> token id sequences, real (unpadded)
///       vocab_size as an explicit bound. That is what lets
///       src/open_whisper/generation_hf_test.cpp replay HF-generated test
///       vectors against this code with no NPU, no device, no XRT link.
/// \note Ported from transformers 5.15.0
///       (models/whisper/generation_whisper.py,
///       generation/logits_process.py) -- see generation_hf.cpp for the exact
///       lines each function mirrors. The legacy protocol
///       (Whisper::_generate_legacy in modeling_whisper.cpp) is unchanged and
///       stays the default; this is the OFLM_WHISPER_PROTOCOL=hf path.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace whisper_hf {

/// \brief The subset of generation_config.json this port reads.
/// \note Every field is read as-is from the file; nothing here silently
///       substitutes a default HF's Python side might apply for a field the
///       file omits (CLAUDE.md rule: a number without a traceable artifact is
///       not a result -- here, "a behaviour without a source field" is the
///       same failure one layer up). `load()` throws naming the missing
///       field and the path.
struct GenerationConfig {
    int decoder_start_token_id = -1;  ///< <|startoftranscript|>, 50258 for turbo
    int eos_token_id = -1;            ///< 50257
    int no_timestamps_token_id = -1;  ///< 50364
    int max_length = -1;              ///< total sequence cap (prompt + generated), 448
    bool has_max_initial_timestamp_index = false;
    int max_initial_timestamp_index = 0;  ///< valid only if has_max_initial_timestamp_index

    std::unordered_map<std::string, int> lang_to_id;  ///< "<|en|>" -> 50259, ...
    std::unordered_map<std::string, int> task_to_id;  ///< "transcribe" -> 50360, "translate" -> 50359

    std::vector<int> suppress_tokens;        ///< applied every step (may be empty)
    std::vector<int> begin_suppress_tokens;  ///< applied only at begin_index (may be empty)

    /// \brief Load from `<model_dir>/generation_config.json`.
    /// \throws std::runtime_error if the file is missing, is not valid JSON, or a
    ///         required field (decoder_start_token_id, eos_token_id,
    ///         no_timestamps_token_id, max_length, lang_to_id, task_to_id) is absent.
    ///         suppress_tokens/begin_suppress_tokens/max_initial_timestamp_index are
    ///         optional (HF: `generation_config.suppress_tokens is not None` / `getattr`).
    static GenerationConfig load(const std::string& model_dir);

    /// \brief The language-token ids from lang_to_id, in ASCENDING id order.
    /// \note HF builds a boolean mask over the whole vocab from `.values()` (dict
    ///       insertion order is irrelevant to a mask); ascending order here is only so
    ///       the id set is deterministic for the caller and for testing, not because
    ///       order matters to detect_language's argmax.
    std::vector<int> lang_ids() const;

    int timestamp_begin() const { return no_timestamps_token_id + 1; }
};

/// \brief argmax over logits[0, vocab_size) only.
/// \note vocab_size must be the REAL (unpadded) vocabulary size -- the engine pads
///       logits to a multiple of 32 and "the pad tail must never win a sample"
///       (whisper_engine.hpp). Matches torch.argmax's first-index-wins tie break.
int argmax(const std::vector<float>& logits, int vocab_size);

/// \brief Port of `WhisperGenerationMixin.detect_language`'s scoring step: mask every
///        id NOT in `lang_ids` to -inf, then argmax. `sot_logits` is the logits
///        returned by feeding exactly `[decoder_start_token_id]` to the decoder (one
///        token of context, matching HF's `decoder_input_ids = [[decoder_start_token_id]]`
///        with `use_cache=False`).
int detect_language(const std::vector<float>& sot_logits, const std::vector<int>& lang_ids, int vocab_size);

/// \brief Port of `SuppressTokensLogitsProcessor.__call__`: unconditional, every step.
void apply_suppress_tokens(std::vector<float>& logits, const std::vector<int>& suppress_tokens, int vocab_size);

/// \brief Port of `SuppressTokensAtBeginLogitsProcessor.__call__`: only when the
///        current decoder position equals begin_index, i.e. `at_begin_index` is true
///        for the very first free-generation step of a window and false after.
void apply_suppress_tokens_at_begin(std::vector<float>& logits, const std::vector<int>& begin_suppress_tokens,
                                     bool at_begin_index, int vocab_size);

/// \brief Port of `WhisperTimeStampLogitsProcessor.__call__`, specialised to batch
///        size 1 (this engine never batches decode_audio calls).
/// \note HF's processor is stateless per call except for `begin_index`, which is fixed
///       for one window (condition_on_prev_tokens=False: no prompt carries across
///       windows, so begin_index is the same constant -- 3 with timestamps, 4 without
///       -- for every step of a window). `generated` is `input_ids[:, begin_index:]`
///       -- the tokens produced so far in THIS window, not including the one about to
///       be chosen.
class WhisperTimestampProcessor {
public:
    WhisperTimestampProcessor(int no_timestamps_token_id, int eos_token_id, bool has_max_initial_timestamp_index,
                               int max_initial_timestamp_index);

    /// \brief Mutates `logits` in place. `generated.empty()` is HF's
    ///        `input_ids.shape[1] == begin_index` (the very first free step).
    void apply(std::vector<float>& logits, const std::vector<int>& generated, int vocab_size) const;

    int timestamp_begin() const { return timestamp_begin_; }

private:
    int no_timestamps_token_id_;
    int timestamp_begin_;
    int eos_token_id_;
    bool has_max_initial_timestamp_index_;
    int max_initial_timestamp_index_;
};

/// \brief Port of `WhisperGenerationMixin._retrieve_segment`'s seek arithmetic,
///        specialised to batch size 1 and to a caller that already knows the window's
///        duration in seconds and does not need per-segment start/end times (the host
///        streams decoded text token by token as it generates -- see
///        Whisper::_generate_hf in modeling_whisper.cpp -- so only the amount to
///        advance the seek pointer by is needed here, not the segment list itself).
/// \param generated the full token sequence produced for this window (same
///        "since begin_index" sequence WhisperTimestampProcessor saw), AFTER
///        generation for the window has finished (EOS or max_length).
/// \param timestamp_begin GenerationConfig::timestamp_begin().
/// \param window_seconds the audio duration actually fed to the encoder for this
///        window (<=30; the last window of a clip may be shorter).
/// \param time_precision seconds per timestamp-token step; 0.02 for every released
///        Whisper checkpoint (the <|0.00|>, <|0.02|>, ... token ladder).
/// \return seconds to advance the seek pointer by. This is a BIT-FAITHFUL port of
///         `_retrieve_segment`'s arithmetic (no floor, no clamp): it CAN return exactly
///         0.0 when the last unmatched timestamp pair opens at <|0.00|> (verified
///         against transformers 5.15.0 directly -- see
///         testdata/segment_offset_cases.json's "immediate_double_timestamp" case,
///         where real HF's own `_retrieve_segment` also returns a zero-frame offset).
///         HF's batched seek loop tolerates a zero offset because OTHER items in the
///         batch still make progress and a stalled item is bounded by `max_length`;
///         this engine's driver is a single sequential window, so its caller
///         (Whisper::_generate_hf in modeling_whisper.cpp) applies its OWN documented
///         floor after calling this -- deliberately kept out of this function so the
///         function itself stays a faithful, independently-testable port.
float compute_segment_offset_seconds(const std::vector<int>& generated, int timestamp_begin, float window_seconds,
                                      float time_precision = 0.02f);

}  // namespace whisper_hf
