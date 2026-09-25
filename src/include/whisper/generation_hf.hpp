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
///       lines each function mirrors. This is the OFLM_WHISPER_PROTOCOL=hf
///       path: the default for the open engine. The closed engine defaults to
///       the unchanged legacy loop (Whisper::_generate_legacy in
///       modeling_whisper.cpp); an explicit OFLM_WHISPER_PROTOCOL overrides
///       either (Whisper::_init_decode_protocol).
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

    /// \brief Refuse (throw) if any id this port will ever hand to the model's real
    ///        vocabulary is out of `[0, vocab_size)` -- PR #111 review (Copilot
    ///        4096191784). Every one of these ids comes from `generation_config.json`
    ///        and is used, unchecked, as a vector index or an argmax bound elsewhere
    ///        in this port (`to_float_vec`, `detect_language`,
    ///        `WhisperTimestampProcessor`, `compute_segment_offset_samples`); a
    ///        checkpoint whose config disagrees with its own `config.json` vocab_size
    ///        (or a corrupted/hand-edited file) would otherwise read or write past
    ///        the logits buffer instead of failing at load. Checks:
    ///        decoder_start_token_id, eos_token_id, no_timestamps_token_id, every
    ///        lang_to_id and task_to_id value, every suppress_tokens /
    ///        begin_suppress_tokens entry, and the timestamp id RANGE
    ///        (`timestamp_begin()` itself, since every id from there to vocab_size-1
    ///        is a valid timestamp token -- there is no separate upper bound to check
    ///        beyond vocab_size, which the caller already knows).
    /// \throws std::runtime_error naming the field and the offending id.
    void validate(int vocab_size) const;

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
///        duration in SAMPLES and does not need per-segment start/end times (the host
///        streams decoded text token by token as it generates -- see
///        Whisper::_generate_hf in modeling_whisper.cpp -- so only the amount to
///        advance the seek pointer by is needed here, not the segment list itself).
///
/// \note Two fixes over the seconds-returning form this replaced (PR #111 review,
///       findings A and F):
///       - **Trailing EOS is stripped internally**, mirroring
///         `if seek_sequence[-1] == generation_config.eos_token_id: seek_sequence =
///         seek_sequence[:-1]` (generation_whisper.py ~L1083-1086), which HF's
///         `generate()` always runs immediately before calling `_retrieve_segment`.
///         `_retrieve_segment` itself never sees the EOS token HF appends when a
///         window ends via EOS rather than max_length, so a caller that hands this
///         function the raw EOS-terminated `generated` (as `Whisper::_generate_hf`
///         does) still gets HF's answer, not a `[..., <ts>, <ts>, EOS]` sequence
///         misread as an unterminated final segment -- the bug that made the NEXT
///         30s window seek backward and re-transcribe the last segment on audio
///         >30s.
///       - **The result is an exact integer sample count, not seconds.** HF's own
///         seek pointer is in mel frames (10ms each): `segment_offset =
///         last_timestamp_pos * input_stride` (`input_stride` = 2, so 20ms per
///         timestamp-token step -> 2 mel frames), then `seek += segment_offset`
///         (generation_whisper.py ~L2048, ~L898). Converting that to audio samples
///         needs one more factor, the mel hop in samples (`FS / 100` = 160 samples
///         per 10ms mel frame at the standard FS=16000), which the caller folds into
///         `samples_per_timestamp_step` so this function does the whole conversion
///         in integers: `last_timestamp_pos * samples_per_timestamp_step`. The old
///         float form (`last_timestamp_pos * time_precision(0.02f)`, seconds,
///         converted back to samples via `int(seconds * FS)` at the call site) could
///         truncate a value that should land on an exact sample boundary, losing up
///         to one sample per window.
/// \param generated the full token sequence produced for this window (same
///        "since begin_index" sequence WhisperTimestampProcessor saw), AFTER
///        generation for the window has finished (EOS or max_length). May end with
///        `eos_token_id`; that trailing token is stripped before the port's own logic
///        runs, exactly as HF strips it before calling `_retrieve_segment`.
/// \param timestamp_begin GenerationConfig::timestamp_begin().
/// \param eos_token_id GenerationConfig::eos_token_id -- identifies the trailing
///        token to strip, matching HF's own comparison.
/// \param window_samples the audio duration actually fed to the encoder for this
///        window, in SAMPLES (<=WINDOW_SAMPLES; the last window of a clip may be
///        shorter). Exact by construction -- this is the same sample count the
///        caller sliced out of the audio buffer, no float round-trip involved.
/// \param samples_per_timestamp_step exact number of audio samples one
///        timestamp-token step (`time_precision` seconds) represents. 320 for every
///        released Whisper checkpoint at the standard FS=16000 (`time_precision`
///        0.02s * FS 16000 = 320, or equivalently `input_stride`(2) *
///        mel_hop_samples(160)) -- derived once by the caller, not recomputed here,
///        so this function performs no floating-point FS conversion of its own.
/// \return SAMPLES to advance the seek pointer by. This is a bit-faithful port of
///         `_retrieve_segment`'s arithmetic (no floor, no clamp): it CAN return
///         exactly 0 when the last unmatched timestamp pair opens at <|0.00|>
///         (verified against transformers 5.15.0 directly -- see
///         testdata/segment_offset_cases.json's "immediate_double_timestamp" case,
///         where real HF's own `_retrieve_segment` also returns a zero-frame offset).
///         HF's batched seek loop tolerates a zero offset because OTHER items in the
///         batch still make progress and a stalled item is bounded by `max_length`;
///         this engine's driver is a single sequential window, so its caller
///         (Whisper::_generate_hf in modeling_whisper.cpp) applies its OWN documented
///         floor after calling this -- deliberately kept out of this function so the
///         function itself stays a faithful, independently-testable port.
int compute_segment_offset_samples(const std::vector<int>& generated, int timestamp_begin, int eos_token_id,
                                    int window_samples, int samples_per_timestamp_step = 320);

}  // namespace whisper_hf
