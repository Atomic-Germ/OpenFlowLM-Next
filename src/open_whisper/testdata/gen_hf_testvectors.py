"""Generate replay test vectors for the `hf` Whisper decoding protocol's C++ port.

Runs the REAL transformers 5.15.0 logits processors (SuppressTokensAtBeginLogitsProcessor,
SuppressTokensLogitsProcessor, WhisperTimeStampLogitsProcessor) and the real
WhisperGenerationMixin._retrieve_segment staticmethod against a small synthetic vocabulary
(so files stay in the KB-MB range instead of ~1.6 GB at the real 51866-token vocab -- the
ported algorithm does not care about absolute vocab size, only the relative id layout, which
this synthetic config mirrors: suppress_tokens scattered through the vocab, begin_suppress
near the very start and at eos, a timestamp region at the tail).

Run from .venv-ref:
    .venv-ref/Scripts/python src/open_whisper/testdata/gen_hf_testvectors.py

Writes, next to this script:
    generation_config.json          -- the synthetic config (same schema the C++
                                        GenerationConfig::load() reads); named exactly
                                        "generation_config.json" because
                                        GenerationConfig::load(dir) always appends that
                                        filename to whatever directory it is given
    greedy_processor_cases.json     -- per-step logits + HF's chosen token, plus each
                                        session's detect_language ground truth
    segment_offset_cases.json       -- _retrieve_segment ground truth on synthetic
                                        generated-token sequences
"""
import json
import os
import random

import torch
from types import SimpleNamespace

from transformers.generation.logits_process import (
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    WhisperTimeStampLogitsProcessor,
)
from transformers.models.whisper.generation_whisper import WhisperGenerationMixin

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Synthetic generation_config, same shape as the real large-v3-turbo one but
# small. Layout (vocab_size=300):
#   0..255   ordinary text tokens
#   256      a "leading space"-like token, mirrors real begin_suppress_tokens[0]=220
#   257      eos_token_id                       (real: 50257)
#   258      startoftranscript                  (real: 50258)  -- NOT in vocab range used
#            as a real token id below 257 boundary checks, kept > eos like the real ids
#   259..278 20 language tokens                 (real: 50259..50358 range, ~100 langs)
#   279      translate task id                  (real: 50359)
#   280      transcribe task id                 (real: 50360)
#   281      (unused, mirrors 50361/50362 gaps)
#   282      (unused)
#   283      no_timestamps_token_id             (real: 50364)
#   284..299 timestamp tokens (16 of them)       (real: 50365.. one per 0.02s step)
VOCAB_SIZE = 300
DECODER_START = 258
EOS = 257
NO_TS = 283
TIMESTAMP_BEGIN = NO_TS + 1  # 284
NUM_TIMESTAMPS = VOCAB_SIZE - TIMESTAMP_BEGIN  # 16
LANG_IDS = list(range(259, 279))  # 20 languages
TASK_TRANSCRIBE = 280
TASK_TRANSLATE = 279
SUPPRESS_TOKENS = [1, 2, 7, 8, 14, 25, 50, 90, 150, 200, 254, 255, 258, 279, 280, 281, 282, 283]
BEGIN_SUPPRESS_TOKENS = [256, 257]
MAX_INITIAL_TIMESTAMP_INDEX = 5

generation_config_test = {
    "decoder_start_token_id": DECODER_START,
    "eos_token_id": EOS,
    "no_timestamps_token_id": NO_TS,
    "max_length": 40,
    "max_initial_timestamp_index": MAX_INITIAL_TIMESTAMP_INDEX,
    "lang_to_id": {f"<|lang{i}|>": lid for i, lid in enumerate(LANG_IDS)},
    "task_to_id": {"transcribe": TASK_TRANSCRIBE, "translate": TASK_TRANSLATE},
    "suppress_tokens": SUPPRESS_TOKENS,
    "begin_suppress_tokens": BEGIN_SUPPRESS_TOKENS,
}
with open(os.path.join(HERE, "generation_config.json"), "w") as f:
    json.dump(generation_config_test, f, indent=2)

# ---------------------------------------------------------------------------
# The HF processor chain, in the exact order _retrieve_logit_processors builds it
# (generation_whisper.py ~L1774-1812): begin_suppress, then suppress_tokens, then
# the timestamp processor last.
gc_ns = SimpleNamespace(
    no_timestamps_token_id=NO_TS,
    eos_token_id=EOS,
    bos_token_id=None,
    max_initial_timestamp_index=MAX_INITIAL_TIMESTAMP_INDEX,
    _detect_timestamp_from_logprob=True,
)

rng = random.Random(20260923)
torch.manual_seed(20260923)


def make_processors(begin_index: int, with_timestamps: bool):
    procs = [
        SuppressTokensAtBeginLogitsProcessor(BEGIN_SUPPRESS_TOKENS, begin_index=begin_index),
        SuppressTokensLogitsProcessor(SUPPRESS_TOKENS),
    ]
    if with_timestamps:
        procs.append(WhisperTimeStampLogitsProcessor(gc_ns, begin_index=begin_index))
    return procs


def run_processors(procs, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    out = scores
    for p in procs:
        out = p(input_ids, out)
    return out


def random_logits(bias_token=None, bias=0.0):
    x = torch.randn(1, VOCAB_SIZE) * 3.0
    if bias_token is not None:
        x[0, bias_token] += bias
    return x


sessions = []
NUM_SESSIONS = 60
MAX_STEPS = 30

for session_id in range(NUM_SESSIONS):
    with_timestamps = session_id % 3 != 0  # mix: most sessions test the timestamp path
    begin_index = 3  # [SOT, lang, task] -- matches the hf protocol's timestamps-on prompt
    if not with_timestamps:
        begin_index = 4  # [SOT, lang, task, notimestamps]

    # detect_language: sot_logits restricted to LANG_IDS, argmax. `hard_language_test`
    # sessions boost a NON-language token above the forced language token, so that an
    # unmasked argmax (OW_BREAK_RULE=detect_language) picks the wrong id for certain --
    # a plain random draw over 300 logits only occasionally produces a stray value above
    # a +8 boost, which made this rule's break invisible across 60 sessions (found by
    # actually running the deliberate-break mode, not assumed).
    sot_logits = random_logits()
    forced_lang = rng.choice(LANG_IDS)
    hard_language_test = session_id % 5 == 0
    sot_logits[0, forced_lang] += 8.0
    if hard_language_test:
        non_lang_token = rng.choice([i for i in range(VOCAB_SIZE) if i not in LANG_IDS])
        sot_logits[0, non_lang_token] += 20.0  # beats the language token unless masked
    masked = sot_logits.clone()
    mask = torch.ones(VOCAB_SIZE, dtype=torch.bool)
    mask[LANG_IDS] = False
    masked[0, mask] = -float("inf")
    detected_lang = int(masked.argmax(-1).item())

    procs = make_processors(begin_index, with_timestamps)

    prefix: list[int] = []
    steps = []
    input_ids = torch.zeros((1, begin_index), dtype=torch.long)  # prompt content doesn't
    # matter to these processors (only its LENGTH does, for begin_index comparisons), so a
    # zero-filled prompt of the right length is a faithful enough stand-in.

    # Every session is adversarial in some way, keyed independently of with_timestamps
    # (session_id % 12 alone correlated with the with_timestamps % 3 split, which meant
    # two modes below never actually fired when with_timestamps was true -- found the
    # same way as the detect_language gap, by running the deliberate-break mode and
    # seeing 0 mismatches where there should have been some).
    adversarial_mode = (session_id * 7 + 3) % 12

    for step in range(MAX_STEPS):
        at_begin = step == 0
        logits = random_logits()

        if with_timestamps and adversarial_mode == 1 and step in (0, 1):
            # force two consecutive timestamps at the very start
            logits[0, TIMESTAMP_BEGIN] += 15.0 if step == 0 else 0.0
            if step == 1:
                logits[0, TIMESTAMP_BEGIN + 1] += 15.0
        elif with_timestamps and adversarial_mode == 2 and step == 0:
            # force the very first timestamp to be exactly <|0.00|>
            logits[0, TIMESTAMP_BEGIN] += 15.0
        elif adversarial_mode == 3 and step == 0:
            # try to force EOS immediately (begin_suppress_tokens=[256,257] must block it
            # at begin_index, in both the with- and without-timestamps prompts).
            logits[0, EOS] += 20.0
        elif adversarial_mode == 5 and step == 0:
            # try to force the other begin_suppress_tokens entry (256, the
            # leading-space-like token) immediately -- begin_suppress must block this one
            # too, independent of EOS.
            logits[0, 256] += 20.0
        elif adversarial_mode == 8 and step == 1:
            # begin_suppress_tokens must NOT apply one step later (begin_index+1) --
            # boost the same tokens just past the begin step to confirm they are free to
            # win there (a processor that suppressed unconditionally, not just at
            # begin_index, would never be caught by the plain OW_BREAK_RULE=begin_suppress
            # run since disabling it can only ever fix false suppressions, not reveal
            # over-suppression -- this checks the port's *scope*, not just its presence).
            logits[0, 256] += 20.0
        elif adversarial_mode == 4 and step == MAX_STEPS - 1:
            # push toward EOS near the end to exercise max_length-adjacent behaviour
            logits[0, EOS] += 6.0
        elif with_timestamps and adversarial_mode == 7 and step in (4, 5, 14, 15, 24, 25):
            # SEVERAL separated timestamp pairs spread across one window (PR #111
            # review, finding 13: "a sequence with several timestamps" using the
            # real prompt layout). Modes 1/2 above only ever force a timestamp in
            # steps 0-1; this fires three independent consecutive-pairs at
            # increasing timestamp ids (steps 4-5, 14-15, 24-25), so the "timestamps
            # shouldn't decrease" / "avoid re-emitting <|0.00|> again" branches of
            # WhisperTimeStampLogitsProcessor.apply() each fire more than once in
            # the same session, and the run between pairs (steps 6-13, 16-23) still
            # has ordinary random logits so ordinary text tokens interleave with
            # the timestamps, matching a real multi-segment transcript's shape.
            pair_base = {4: TIMESTAMP_BEGIN + 1, 5: TIMESTAMP_BEGIN + 1,
                        14: TIMESTAMP_BEGIN + 5, 15: TIMESTAMP_BEGIN + 5,
                        24: TIMESTAMP_BEGIN + 9, 25: TIMESTAMP_BEGIN + 9}[step]
            logits[0, pair_base if step in (4, 14, 24) else pair_base + 1] += 15.0

        full_ids = torch.tensor([prefix], dtype=torch.long) if prefix else torch.zeros((1, 0), dtype=torch.long)
        decoder_input_ids = torch.cat([input_ids, full_ids], dim=-1)

        processed = run_processors(procs, decoder_input_ids, logits.clone())
        token = int(processed.argmax(-1).item())

        steps.append(
            {
                "prefix": list(prefix),
                "at_begin": at_begin,
                "logits": logits[0].tolist(),
                "expected_token": token,
            }
        )
        prefix.append(token)
        if token == EOS:
            break

    sessions.append(
        {
            "session_id": session_id,
            "with_timestamps": with_timestamps,
            "begin_index": begin_index,
            "sot_logits": sot_logits[0].tolist(),
            "detected_lang_id": detected_lang,
            "steps": steps,
        }
    )

with open(os.path.join(HERE, "greedy_processor_cases.json"), "w") as f:
    json.dump({"vocab_size": VOCAB_SIZE, "lang_ids": LANG_IDS, "sessions": sessions}, f)

print(f"wrote {len(sessions)} sessions, {sum(len(s['steps']) for s in sessions)} steps total")

# ---------------------------------------------------------------------------
# _retrieve_segment ground truth. mel hop = 0.01s (100 Hz), input_stride = 2
# (50 Hz encoder / 100 Hz mel), time_precision = 0.02s/token-step -- the real
# Whisper feature-extractor constants, independent of the synthetic vocab above.
# FS = 16000 (Whisper's audio sample rate -- modeling_whisper.hpp's own FS) is
# needed here too, PR #111 review findings A and F: the C++ port
# (compute_segment_offset_samples) now returns an exact SAMPLE count, not
# seconds, and strips a trailing EOS before running _retrieve_segment's own
# logic -- both to be verified here against the REAL transformers function,
# not assumed.
MEL_HOP = 0.01
TIME_PRECISION = 0.02
INPUT_STRIDE = 2
FS = 16000
MEL_HOP_SAMPLES = FS // 100  # 160: samples per 10ms mel frame
SAMPLES_PER_TIMESTAMP_STEP = round(TIME_PRECISION * FS)  # 320 -- must match
# generation_hf.hpp's compute_segment_offset_samples() default argument exactly.
assert SAMPLES_PER_TIMESTAMP_STEP == 320


def retrieve_segment_samples(generated: list[int], timestamp_begin: int, eos_token_id: int, window_samples: int):
    # Finding A: HF's generate() strips a trailing EOS (generation_whisper.py
    # ~L1083-1086) BEFORE ever calling _retrieve_segment -- _retrieve_segment
    # itself has no idea what EOS is. Do the same here, so a case built with a
    # trailing EOS (as the real per-window `generated` this project's
    # modeling_whisper.cpp passes always can be) is checked against what HF
    # actually produces for that window, not against _retrieve_segment's
    # answer on the UNSTRIPPED sequence (which is precisely the bug finding A
    # fixed: an unstripped EOS reads as an unclosed trailing segment).
    stripped = list(generated)
    if stripped and stripped[-1] == eos_token_id:
        stripped = stripped[:-1]

    window_seconds = window_samples / FS
    seek_num_frames = round(window_seconds / MEL_HOP)
    seek_sequence = torch.tensor(stripped, dtype=torch.long)
    seek_outputs = [{}]
    decoder_input_ids = torch.zeros((1, 3), dtype=torch.long)
    segments, segment_offset_frames = WhisperGenerationMixin._retrieve_segment(
        seek_sequence=seek_sequence,
        seek_outputs=seek_outputs,
        time_offset=[0.0],
        timestamp_begin=timestamp_begin,
        seek_num_frames=[seek_num_frames],
        time_precision=TIME_PRECISION,
        time_precision_features=MEL_HOP,
        input_stride=INPUT_STRIDE,
        prev_idx=0,
        idx=0,
        return_token_timestamps=False,
        decoder_input_ids=decoder_input_ids,
    )
    # segment_offset_frames is in MEL FRAMES (10ms units, input_stride already
    # applied by _retrieve_segment itself) -- the single-segment/no-pair
    # branches return seek_num_frames directly, which is exactly
    # window_samples's own mel-frame count, so this conversion is exact there
    # too, not just on the timestamp-pair branch.
    return int(segment_offset_frames) * MEL_HOP_SAMPLES


TB = TIMESTAMP_BEGIN  # reuse the synthetic layout so ids line up with generation_config_test.json
seg_cases = []


def add_case(name, generated, window_seconds, eos_token_id=EOS):
    window_samples = round(window_seconds * FS)
    assert abs(window_samples - window_seconds * FS) < 1e-6, f"{name}: window_seconds does not land on a sample"
    samples = retrieve_segment_samples(generated, TB, eos_token_id, window_samples)
    seg_cases.append(
        {
            "name": name,
            "generated": generated,
            "window_samples": window_samples,
            "expected_samples": samples,
        }
    )


add_case("single_timestamp_ending", [TB, 5, 6, TB + 3], 30.0)
add_case("no_timestamps_at_all", [1, 2, 3, 4, 5], 30.0)
add_case("one_unpaired_timestamp", [TB, 5, 6, 7], 12.5)
add_case("consecutive_pair_then_more_unclosed", [TB, 5, 6, TB + 1, TB + 2, 8, 9], 30.0)
add_case("two_full_segments_then_open_third", [TB, 1, 2, TB + 4, TB + 4, 3, 4, TB + 9, 5, 6], 30.0)
add_case("zero_at_open_of_trailing_segment", [TB, 1, TB, 2, 3], 30.0)  # last_timestamp_pos would be 0
add_case("empty_sequence", [], 7.5)
add_case("single_token_no_timestamp", [3], 30.0)
add_case("immediate_double_timestamp", [TB, TB + 1], 30.0)
add_case("short_final_window", [TB, 1, 2, TB + 2], 4.0)

# PR #111 review, finding A: the SAME cases as above, but every `generated` now ends
# with EOS -- exactly the shape modeling_whisper.cpp's `generated` is in when a window
# ends via EOS rather than max_length (Whisper::_generate_hf pushes the EOS token
# before breaking). Each one is checked against what HF's own generate() would have
# produced for that window (EOS stripped first, see retrieve_segment_samples above),
# which is the regression this project shipped without: before the fix, the C++ port
# read the trailing EOS as an unclosed final segment.
add_case("with_eos_single_timestamp_ending", [TB, 5, 6, TB + 3, EOS], 30.0)
add_case("with_eos_no_timestamps_at_all", [1, 2, 3, 4, 5, EOS], 30.0)
add_case("with_eos_one_unpaired_timestamp", [TB, 5, 6, 7, EOS], 12.5)
add_case("with_eos_consecutive_pair_then_more_unclosed", [TB, 5, 6, TB + 1, TB + 2, 8, 9, EOS], 30.0)
add_case("with_eos_two_full_segments_then_open_third",
        [TB, 1, 2, TB + 4, TB + 4, 3, 4, TB + 9, 5, 6, EOS], 30.0)
add_case("with_eos_zero_at_open_of_trailing_segment", [TB, 1, TB, 2, 3, EOS], 30.0)
add_case("with_eos_only", [EOS], 7.5)
add_case("with_eos_single_token_no_timestamp", [3, EOS], 30.0)
add_case("with_eos_immediate_double_timestamp", [TB, TB + 1, EOS], 30.0)
add_case("with_eos_short_final_window", [TB, 1, 2, TB + 2, EOS], 4.0)

# THE case that demonstrates the actual bug numerically (not just "takes a
# different branch that happens to land on the same answer"), i.e. what "show it
# fails before the fix" means: an EARLIER consecutive timestamp PAIR (so
# pair_end_indices is non-empty -- the function does NOT fall through to "return
# the whole window" regardless) followed by a lone trailing timestamp, then EOS.
#
# Stripped (correct): [TB, 1, 2, TB+4, TB+4, 3, 4, TB+9] -- last two tokens
# [4, TB+9] -> single_timestamp_ending = TRUE -> "no speech after the last
# timestamp" -> returns the FULL window (480000 samples at 30.0s).
#
# Unstripped (the bug): [..., TB+9, EOS] -- last two tokens [TB+9, EOS], and EOS
# is not a timestamp id, so single_timestamp_ending reads FALSE. With
# pair_end_indices=[4] (from the TB+4,TB+4 pair) non-empty, the function takes the
# "ignore the unfinished segment, seek to where it started" branch instead:
# last_timestamp_pos = generated[3] - TB = 4, returning 4*320 = 1280 samples --
# 0.08s into a 30s window. This is EXACTLY the seek-backward-and-re-transcribe
# failure mode finding A describes: on real audio >30s, the NEXT window would
# start 1280 samples in rather than 480000, re-decoding almost the entire
# just-finished window.
add_case("with_eos_pair_then_trailing_single_timestamp", [TB, 1, 2, TB + 4, TB + 4, 3, 4, TB + 9, EOS], 30.0)

with open(os.path.join(HERE, "segment_offset_cases.json"), "w") as f:
    json.dump(
        {
            "timestamp_begin": TB,
            "eos_token_id": EOS,
            "samples_per_timestamp_step": SAMPLES_PER_TIMESTAMP_STEP,
            "cases": seg_cases,
        },
        f,
        indent=2,
    )

print(f"wrote {len(seg_cases)} segment-offset cases")
