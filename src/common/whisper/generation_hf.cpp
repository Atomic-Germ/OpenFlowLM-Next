/// \file generation_hf.cpp
/// \brief Implementation of the `hf` Whisper decoding protocol's pure logic --
///        see generation_hf.hpp for what each function ports and why. No XRT,
///        no device, no buffer<bf16>: this file only needs nlohmann/json.hpp
///        (for GenerationConfig::load) and the standard library, which is what
///        lets generation_hf_test.cpp link it standalone.
#include "whisper/generation_hf.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

#include "nlohmann/json.hpp"

namespace whisper_hf {

namespace {
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

[[noreturn]] void fail(const std::string& path, const std::string& what) {
    throw std::runtime_error("generation_config.json (" + path + "): " + what);
}
}  // namespace

GenerationConfig GenerationConfig::load(const std::string& model_dir) {
    const std::string path = model_dir + "/generation_config.json";
    std::ifstream f(path, std::ios::in | std::ios::binary);
    if (!f) {
        fail(path, "file not found");
    }
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& e) {
        fail(path, std::string("invalid JSON: ") + e.what());
    }

    auto require_int = [&](const char* key) -> int {
        if (!j.contains(key) || j[key].is_null()) {
            fail(path, std::string("missing required field '") + key + "'");
        }
        return j[key].get<int>();
    };

    GenerationConfig cfg;
    cfg.decoder_start_token_id = require_int("decoder_start_token_id");
    // eos_token_id is sometimes a single int, sometimes (other Whisper checkpoints) a
    // one-element list; HF's generate() takes generation_config.eos_token_id or
    // generation_config.bos_token_id as a fallback (logits_process.py ~L1976). We only
    // need the scalar case -- large-v3-turbo's generation_config.json has it as a bare
    // int -- and refuse rather than silently pick element 0 of a list we didn't expect.
    if (!j.contains("eos_token_id") || j["eos_token_id"].is_null()) {
        fail(path, "missing required field 'eos_token_id'");
    } else if (j["eos_token_id"].is_array()) {
        if (j["eos_token_id"].empty()) fail(path, "'eos_token_id' is an empty array");
        cfg.eos_token_id = j["eos_token_id"][0].get<int>();
    } else {
        cfg.eos_token_id = j["eos_token_id"].get<int>();
    }
    cfg.no_timestamps_token_id = require_int("no_timestamps_token_id");
    cfg.max_length = require_int("max_length");

    if (j.contains("max_initial_timestamp_index") && !j["max_initial_timestamp_index"].is_null()) {
        cfg.has_max_initial_timestamp_index = true;
        cfg.max_initial_timestamp_index = j["max_initial_timestamp_index"].get<int>();
    }

    if (!j.contains("lang_to_id") || j["lang_to_id"].is_null() || !j["lang_to_id"].is_object()) {
        fail(path, "missing required object field 'lang_to_id'");
    }
    for (auto it = j["lang_to_id"].begin(); it != j["lang_to_id"].end(); ++it) {
        cfg.lang_to_id[it.key()] = it.value().get<int>();
    }

    if (!j.contains("task_to_id") || j["task_to_id"].is_null() || !j["task_to_id"].is_object()) {
        fail(path, "missing required object field 'task_to_id'");
    }
    for (auto it = j["task_to_id"].begin(); it != j["task_to_id"].end(); ++it) {
        cfg.task_to_id[it.key()] = it.value().get<int>();
    }

    if (j.contains("suppress_tokens") && !j["suppress_tokens"].is_null()) {
        for (auto& v : j["suppress_tokens"]) cfg.suppress_tokens.push_back(v.get<int>());
    }
    if (j.contains("begin_suppress_tokens") && !j["begin_suppress_tokens"].is_null()) {
        for (auto& v : j["begin_suppress_tokens"]) cfg.begin_suppress_tokens.push_back(v.get<int>());
    }

    return cfg;
}

void GenerationConfig::validate(int vocab_size) const {
    if (vocab_size <= 0) {
        throw std::runtime_error("GenerationConfig::validate: vocab_size must be positive, got " +
                                  std::to_string(vocab_size));
    }
    auto check = [&](const char* field, int id) {
        if (id < 0 || id >= vocab_size) {
            throw std::runtime_error("generation_config.json: '" + std::string(field) + "' = " +
                                      std::to_string(id) + " is out of range [0, " + std::to_string(vocab_size) +
                                      ") -- the model's real vocab_size (config.json)");
        }
    };
    check("decoder_start_token_id", decoder_start_token_id);
    check("eos_token_id", eos_token_id);
    check("no_timestamps_token_id", no_timestamps_token_id);
    // timestamp_begin() itself is the first valid timestamp id; every id from there
    // to vocab_size-1 is a timestamp token, so checking the boundary covers the
    // whole range (there is nothing above it to check against but vocab_size, which
    // the id-index invariant above already enforces).
    check("no_timestamps_token_id + 1 (timestamp_begin)", timestamp_begin());
    for (const auto& kv : lang_to_id) {
        check(("lang_to_id['" + kv.first + "']").c_str(), kv.second);
    }
    for (const auto& kv : task_to_id) {
        check(("task_to_id['" + kv.first + "']").c_str(), kv.second);
    }
    for (int id : suppress_tokens) {
        check("suppress_tokens[]", id);
    }
    for (int id : begin_suppress_tokens) {
        check("begin_suppress_tokens[]", id);
    }
}

std::vector<int> GenerationConfig::lang_ids() const {
    std::vector<int> ids;
    ids.reserve(lang_to_id.size());
    for (auto& kv : lang_to_id) ids.push_back(kv.second);
    std::sort(ids.begin(), ids.end());
    return ids;
}

int argmax(const std::vector<float>& logits, int vocab_size) {
    if (vocab_size <= 0 || static_cast<size_t>(vocab_size) > logits.size()) {
        throw std::runtime_error("argmax: vocab_size out of range of logits buffer");
    }
    int best = 0;
    float best_v = logits[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return best;
}

int detect_language(const std::vector<float>& sot_logits, const std::vector<int>& lang_ids, int vocab_size) {
    // Port of WhisperGenerationMixin.detect_language (generation_whisper.py ~L1663-1673):
    //   non_lang_mask = ones; non_lang_mask[lang_to_id.values()] = False
    //   logits[non_lang_mask] = -inf
    //   lang_ids = logits.argmax(-1)
    std::vector<float> masked(sot_logits.begin(), sot_logits.begin() + vocab_size);
    std::fill(masked.begin(), masked.end(), kNegInf);
    for (int id : lang_ids) {
        if (id < 0 || id >= vocab_size) throw std::runtime_error("detect_language: lang id out of vocab range");
        masked[id] = sot_logits[id];
    }
    return argmax(masked, vocab_size);
}

void apply_suppress_tokens(std::vector<float>& logits, const std::vector<int>& suppress_tokens, int vocab_size) {
    // SuppressTokensLogitsProcessor.__call__ (logits_process.py ~L1900-1904): unconditional.
    for (int id : suppress_tokens) {
        if (id >= 0 && id < vocab_size) logits[id] = kNegInf;
    }
}

void apply_suppress_tokens_at_begin(std::vector<float>& logits, const std::vector<int>& begin_suppress_tokens,
                                     bool at_begin_index, int vocab_size) {
    // SuppressTokensAtBeginLogitsProcessor.__call__ (logits_process.py ~L1858-1865):
    // only fires when input_ids.shape[-1] == begin_index, i.e. the very first free step.
    if (!at_begin_index) return;
    for (int id : begin_suppress_tokens) {
        if (id >= 0 && id < vocab_size) logits[id] = kNegInf;
    }
}

WhisperTimestampProcessor::WhisperTimestampProcessor(int no_timestamps_token_id, int eos_token_id,
                                                       bool has_max_initial_timestamp_index,
                                                       int max_initial_timestamp_index)
    : no_timestamps_token_id_(no_timestamps_token_id),
      timestamp_begin_(no_timestamps_token_id + 1),
      eos_token_id_(eos_token_id),
      has_max_initial_timestamp_index_(has_max_initial_timestamp_index),
      max_initial_timestamp_index_(max_initial_timestamp_index) {}

void WhisperTimestampProcessor::apply(std::vector<float>& logits, const std::vector<int>& generated,
                                       int vocab_size) const {
    // Port of WhisperTimeStampLogitsProcessor.__call__ (logits_process.py ~L2000-2046),
    // batch size 1. Line references below are relative to that block.
    if (no_timestamps_token_id_ >= 0 && no_timestamps_token_id_ < vocab_size) {
        logits[no_timestamps_token_id_] = kNegInf;  // suppress <|notimestamps|> itself
    }

    const int n = static_cast<int>(generated.size());
    const bool last_was_timestamp = n >= 1 && generated[n - 1] >= timestamp_begin_;
    const bool penultimate_was_timestamp = n < 2 || generated[n - 2] >= timestamp_begin_;

    if (last_was_timestamp) {
        if (penultimate_was_timestamp) {
            // "has to be non-timestamp": forbid every timestamp token.
            for (int i = timestamp_begin_; i < vocab_size; ++i) logits[i] = kNegInf;
        } else {
            // "cannot be normal text tokens": forbid everything below eos_token_id.
            for (int i = 0; i < eos_token_id_ && i < vocab_size; ++i) logits[i] = kNegInf;
        }
    }

    // timestamps shouldn't decrease; forbid timestamp tokens smaller than the last one seen.
    std::vector<int> timestamps;
    for (int t : generated) {
        if (t >= timestamp_begin_) timestamps.push_back(t);
    }
    if (!timestamps.empty()) {
        int timestamp_last;
        if (last_was_timestamp && !penultimate_was_timestamp) {
            timestamp_last = timestamps.back();
        } else {
            timestamp_last = timestamps.back() + 1;  // avoid re-emitting the same timestamp
        }
        for (int i = timestamp_begin_; i < timestamp_last && i < vocab_size; ++i) logits[i] = kNegInf;
    }

    // apply the max_initial_timestamp option -- only at the very first free-generation step.
    if (generated.empty()) {
        for (int i = 0; i < timestamp_begin_ && i < vocab_size; ++i) logits[i] = kNegInf;
        if (has_max_initial_timestamp_index_) {
            const int last_allowed = timestamp_begin_ + max_initial_timestamp_index_;
            for (int i = last_allowed + 1; i < vocab_size; ++i) logits[i] = kNegInf;
        }
    }

    // if sum of probability over timestamps exceeds the best single non-timestamp token,
    // force a timestamp. Computed in double for the log-sum-exp stability HF gets from
    // float32 softmax over a much smaller (post-suppression) support -- but `logits`
    // here has already been rounded to bf16 by whisper_engine::decode_audio (the
    // interface shared with the closed engine, unchanged by this port) before this
    // function ever sees it. Computing the sum in double only avoids adding a SECOND
    // rounding on top of that; it does not, and cannot, undo the first one. So this
    // decision CAN legitimately differ from HF's own (float32 logits, float32 sum)
    // near a probability-mass boundary -- that is a datapath-precision fact (bf16 vs
    // float32 logits), not a bug in this reduction, and not something running it in
    // double either causes or fixes (PR #111 review, finding I; Copilot 4096191944).
    double max_logit = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < vocab_size; ++i) {
        if (logits[i] > static_cast<float>(max_logit)) max_logit = logits[i];
    }
    if (std::isfinite(max_logit)) {
        double sum_exp = 0.0;
        double timestamp_sum_exp = 0.0;
        double max_text_logit = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < vocab_size; ++i) {
            const double e = std::exp(static_cast<double>(logits[i]) - max_logit);
            sum_exp += e;
            if (i >= timestamp_begin_) {
                timestamp_sum_exp += e;
            } else if (logits[i] > static_cast<float>(max_text_logit) || !std::isfinite(max_text_logit)) {
                max_text_logit = logits[i];
            }
        }
        const double log_sum_exp = std::log(sum_exp);
        const double timestamp_logprob =
            timestamp_sum_exp > 0.0 ? std::log(timestamp_sum_exp) - log_sum_exp
                                     : -std::numeric_limits<double>::infinity();
        const double max_text_token_logprob =
            std::isfinite(max_text_logit) ? (max_text_logit - max_logit) - log_sum_exp
                                           : -std::numeric_limits<double>::infinity();
        if (timestamp_logprob > max_text_token_logprob) {
            for (int i = 0; i < timestamp_begin_ && i < vocab_size; ++i) logits[i] = kNegInf;
        }
    }
}

int compute_segment_offset_samples(const std::vector<int>& generated, int timestamp_begin, int eos_token_id,
                                    int window_samples, int samples_per_timestamp_step) {
    // Strip a trailing EOS first (generation_whisper.py ~L1083-1086: `if
    // seek_sequence[-1] == generation_config.eos_token_id: seek_sequence =
    // seek_sequence[:-1]`), which HF's generate() always does before calling
    // _retrieve_segment -- see this function's header (PR #111 review, finding A).
    // A local copy, not an index bound: an EOS in the MIDDLE of `generated` (cannot
    // happen -- the caller breaks its decode loop the step it sees one -- but this
    // keeps the port's scope identical to HF's, which also only ever strips the
    // last element) is left untouched.
    const std::vector<int>* seq = &generated;
    std::vector<int> stripped;
    if (!generated.empty() && generated.back() == eos_token_id) {
        stripped.assign(generated.begin(), generated.end() - 1);
        seq = &stripped;
    }

    // Port of WhisperGenerationMixin._retrieve_segment (generation_whisper.py
    // ~L1993-2074), batch size 1, specialised to the seek offset only (see the header
    // for why the segment list itself is not needed here).
    const int n = static_cast<int>(seq->size());
    std::vector<char> is_ts(n);
    for (int i = 0; i < n; ++i) is_ts[i] = (*seq)[i] >= timestamp_begin ? 1 : 0;

    const bool single_timestamp_ending = n >= 2 && !is_ts[n - 2] && is_ts[n - 1];

    std::vector<int> pair_end_indices;  // index of the SECOND token of each consecutive
                                         // timestamp pair (timestamp_segment_indices, already +1'd)
    for (int i = 0; i + 1 < n; ++i) {
        if (is_ts[i] && is_ts[i + 1]) pair_end_indices.push_back(i + 1);
    }

    if (!pair_end_indices.empty()) {
        std::vector<int> slices = pair_end_indices;
        if (single_timestamp_ending) {
            slices.push_back(n);
        } else {
            slices.back() += 1;
        }
        const int last_slice = slices.back();

        if (single_timestamp_ending) {
            // "single timestamp at the end means no speech after the last timestamp":
            // consume the whole window.
            return window_samples;
        }
        // "otherwise, ignore the unfinished segment and seek to [where it started]".
        const int idx = last_slice - 2;
        if (idx < 0 || idx >= n) {
            // Cannot happen given pair_end_indices is non-empty and the arithmetic above
            // (idx is always the first token of the last matched pair) -- guarded rather
            // than asserted so a future edit that breaks this invariant fails safe.
            return window_samples;
        }
        const int last_timestamp_pos = (*seq)[idx] - timestamp_begin;
        // Bit-faithful: this CAN be 0 (see the header). No floor here on purpose, and
        // no floating point either -- an exact integer sample count.
        return last_timestamp_pos * samples_per_timestamp_step;
    }

    // No consecutive timestamp pair anywhere: HF always consumes the whole window here
    // too (segment_offset = seek_num_frames[prev_idx], generation_whisper.py ~L2072),
    // regardless of whether a single unpaired timestamp token exists.
    return window_samples;
}

}  // namespace whisper_hf
