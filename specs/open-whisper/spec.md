# The open Whisper engine

Whisper is the last model that runs only on the closed engine
([#72](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/72)). It does not fit
`causal_lm`: it encodes a 30 s window once, then decodes one token at a time while
attending to that window, so it has two halves rather than one. This file is what the open
replacement must do, and what has been measured.

Directory name gives the prefix: `OPEN-WHISPER`.

The model is `openai/whisper-large-v3-turbo` (MIT): d_model 1280, 32 encoder layers of 20
heads x 64 with FFN 5120, 4 decoder layers, 128 mels, 1500 audio frames, 448 text
positions, vocab 51866, pre-LN with biases, exact-erf GELU, learned absolute positions, no
RoPE. Weights ship fp16.

What stays as it is: the FFmpeg decode, the FFTW log-mel, the token loop, language
detection, timestamps and the 30 s chunking in `src/common/whisper/`. They are already
open, and the seam below is the only place the engine is reached.

**The reference is transformers in float64**, not the closed engine. The closed engine is
Q4_1 and its transcripts differ from the model's: a Chinese clip comes back translated into
English, a Korean one as a different sentence, and a quiet window as the well-known Russian
subtitle hallucination. Matching it would mean matching a lossy reference.

## Requirements

### OPEN-WHISPER-REF: the oracle, the replica and the ceiling they establish
**Applies to:** `open_kernels/model/whisper_goldens.py`, `replica_whisper.py`, `whisper_decode_check.py`
**Test category:** offline (torch + transformers; no NPU)
**Acceptance criteria:**
- The goldens are produced by `WhisperForConditionalGeneration` in float64 and carry, per
  clip's first 30 s window: mel, conv stem, all 32 layer inputs, the pre-LN last output,
  `enc.out`, each decoder layer's cross K and V, and the greedy token path with its logits
  under both prompt protocols (transformers', and the host's, which never feeds the
  detected language token).
- Tensors are stored contiguous. `safetensors.numpy` writes a non-contiguous array in
  memory order, silently, and every hidden state of this model is non-contiguous
  (transformers keeps the conv stem's permuted layout through the residual adds).
- `replica_whisper.py --exact` reproduces the goldens to fp32 roundoff, which is what
  makes the decomposition (tap-major im2col, fused QKV with a zero K bias, the fused cross
  K|V operand) a checked claim rather than an intention.
- The bf16-operand replica is the **ceiling** any engine built on the bf16 GEMM can reach.
  Gates are calibrated from it, never set in advance.

**Measured 2026-09-19** (float64 cosine over the 1500 real rows, six clips):
`enc.out` **0.99571-0.99930**, cross K/V min 0.99454-0.99906, per-layer chained
0.99966-0.99999, teacher-forced >= 0.9999980. That reproduces whisper-xdna's independent
finding that 0.999 on `enc.out` is not reachable at 32-layer depth in bf16.
The final LayerNorm amplifies the relative error about 3x (1.5-2.6e-2 before, 3.8-9.3e-2
after).

**And it costs no tokens:** with the exact decoder on the bf16 replica's encoder output,
all six transcripts are token-identical to float64 and 296/296 teacher-forced steps agree.

### OPEN-WHISPER-SEAM: one engine interface, and no silent fallback between engines
**Applies to:** `src/include/whisper/whisper_engine.hpp`, `src/common/whisper/whisper_engine_closed.cpp`
**Test category:** integration (through `oflm serve --asr 1`)
**Acceptance criteria:**
- `Whisper` holds a `whisper_engine`, never a concrete engine. The closed `whisper_npu` is
  a prebuilt class and is wrapped, not derived from.
- The load log names the engine that was selected.
- `OFLM_WHISPER_ENGINE` selects one explicitly; a value this build cannot provide is an
  error naming it, never a fallback to the other engine. Both engines return a transcript,
  so a fallback would be invisible.
- `--asrmodel TAG` chooses the model. A tag that does not resolve to a Whisper model is
  refused **before** anything is downloaded: `model_list::get_model_info()` answers an
  unknown tag with `llama3.2:1b`.

**Measured 2026-09-19:** six clips x three requests through `/v1/audio/transcriptions`,
byte-identical to the pre-seam build (18/18). `OFLM_WHISPER_ENGINE=open` and
`--asrmodel llama3.2:1b` are both refused, exit 1.

### OPEN-WHISPER-CONTAINER: the weights the open engine reads
**Applies to:** `utilities/q4nx-build/q4nx/open_whisper.py`
**Test category:** unit (`utilities/q4nx-build/tests/test_open_whisper.py`)
**Acceptance criteria:**
- Built from the HuggingFace checkpoint, into `model.open.safetensors` -- deliberately not
  `model.q4nx`, so the closed reader is never handed it.
- Encoder GEMM operands are `B = W^T` row-major `[K, N]` bf16, **not** tiled: the tile
  tuple belongs to the kernel set, and a container pre-tiled for one tuple is silently
  wrong for another. The engine tiles at load.
- The fusions the streams expect are done here: `Q|K|V` with a zero K bias (k_proj has
  none), the four decoder layers' cross `K|V` as one operand with bias `0|bv0|...`, and
  the conv stem as tap-major im2col operands.
- Biases, norms and position tables are f32. The decoder keeps transformers' names in
  `[out, in]` bf16, with `embed_tokens` as the output head.
- `tokenizer_config.json` gains `bos_token_id` and `eos_token_id`: the host reads both and
  HuggingFace's file carries neither. They are the only two fields the shipped closed
  container adds, diffed to confirm.
- The output is byte-reproducible, and a geometry other than turbo's is refused.

**Measured 2026-09-19:** 8 unit tests pass, including that the stored im2col operand
computes a direct conv1d at strides 1 and 2; mutating the conv tap order or the cross bias
order each fails the test that names it. The real container is 1,624,143,560 B, 481
tensors.

### OPEN-WHISPER-KERNELS: one xclbin, one instruction stream per encoder shape
**Applies to:** `open_kernels/designs/whisper_gemm/`, `open_kernels/export_whisper_kernels.py`
**Test category:** hardware (`open_kernels/harness/run_kernel`)
**Acceptance criteria:**
- Every encoder matrix product is a stream over the pre-tiled bf16 whole-array GEMM at
  tile (64, 64, 32) on 8 columns, `rtp=True`, `tg_depth=2`: conv1 3072x384x1280, conv2
  1536x3840x1280, qkv 1536x1280x3840, o 1536x1280x1280, fc1 1536x1280x5120, fc2
  1536x5120x1280, xkv 1536x1280x10240 (1500 frames padded to 1536).
- The export **refuses the set** unless every stream's `final.xclbin` is the same static
  configuration. fc1 and xkv drain C one row block at a time and conv2 has K = 3840, so
  they are the streams that could diverge; a divergence means they cannot share one
  hardware context.
- The set records `design.json` (npu::Design's schema, with `b_layout_hash`, `tg_depth`
  and `tb_max_n_rows`), `toolchain.json` and a `whisper_kernels.json` marker carrying the
  geometry it was built for.
- Each stream is within rel_fro 5e-3 of a float64 reference, from the **exported** set
  rather than a private build.

**Measured 2026-09-19** (mlir-aie 1.4.2.dev16+g7e00b57, Peano 21.0.0.2026080301, built
natively on Windows): all seven xclbins identical up to 70-78 bytes in 11-14 short runs
(UUID and metadata); rel_fro 1.4e-07 to 7.1e-07, per-row cosine 1.000000000.

### OPEN-WHISPER-ENCODER: the encoder on the NPU
**Applies to:** `src/open_whisper/`
**Test category:** hardware (`open_whisper_cli --golden`)
**Acceptance criteria:**
- The GEMMs run as instruction streams over one hardware context, with every B staged once
  at load.
- LayerNorm (biased variance, eps 1e-5), bias, exact-erf GELU, the bidirectional attention
  and the softmax run on the host in fp32.
- Padded rows (1500 -> 1536, 3000 -> 3072) are zero and never enter attention as keys.
- Per layer, chained and teacher-forced, the float64 cosine against the golden is at least
  the bf16 replica's for that clip and layer, less a stated margin.
- The kernel set is refused when its marker, its geometry check or its B layout disagrees
  with the container.
- **No host code writes into a buffer the device writes.** A GEMM's C buffer is mapped
  from the device; a host write there leaves dirty cache lines whose write-back lands on
  top of a later dispatch's result.
- Two runs of the same clip give the same numbers.

**Measured 2026-09-20** (idle machine, `open_whisper_cli --golden ... [--forced]`), both
clips at the replica's bf16 ceiling and identical across runs to eight digits:

| | Demos_sample-data_journal (replica) | nvidia (replica) |
|---|---|---|
| conv1 / conv2 | 0.99999785 / 0.99999992 (same) | 0.99999890 / 0.99999995 (same) |
| chained `enc.hidden.1` | 0.99999810 (0.99999810) | 0.99999814 (0.99999814) |
| chained `enc.hidden.32` | 0.99987223 (0.99988110) | 0.99991683 (0.99984733) |
| `enc.out` | **0.99828836** (0.99822134) | **0.99906620** (0.99905335) |
| cross K/V worst | 0.99788008 | 0.99901179 |
| teacher-forced, all 32 layers | >= 0.99999827 | >= 0.99999825 |

Host wall clock, labelled as such and not an NPU figure: 4.4-4.8 s per 30 s window,
attention ~4.3 s of it, NPU submit+wait ~1.6 s.

**The C-buffer rule is in this list because breaking it is silent.** In-place bias on the
C buffer produced single wrong ROWS (one row at cosine 0.78 inside a matrix reading
0.9998), at an unpredictable layer, in whole 64-byte-aligned runs -- and serialising the
host made it disappear without fixing it. Three probes separate the layers and are kept in
the engine: `--stress N` (512 dispatches cycling four streams and 32 weight slots:
identical), `OW_VERIFY_A=1` (the operand read back off the device: always correct) and
`OW_DOUBLE_CHECK=1` (the same dispatch twice: caught it).

### OPEN-WHISPER-CROSSKV: the encoded window survives clear_context
**Applies to:** `src/open_whisper/`
**Test category:** unit + integration
**Acceptance criteria:** `clear_context()` resets the decoder's self-attention state only.
The cross K/V computed by the last `encode_audio()` stay, because the host calls
`clear_context()` **after** encoding and before the first token.

**Status:** not started (needs the decoder).

### OPEN-WHISPER-DECODER, -E2E: the transcript
**Applies to:** `src/open_whisper/decoder.*`, `src/common/whisper/`
**Test category:** hardware, end to end
**Acceptance criteria:**
- The logits have the padded vocabulary width the host's sampler expects, with the pad tail
  never able to win a sample.
- Against the goldens' forced token paths, argmax agrees at every step.
- Free-running greedy under the host's own protocol reproduces the float64 transcript on
  the golden clips. Differences from the **closed** engine are reported, not treated as
  failures.
**Status:** not started.

### OPEN-WHISPER-ENDPOINT: something tests /v1/audio/transcriptions
**Applies to:** `src/server/server.cpp`, `src/server/rest_handler.cpp`, `utilities/oflm-test`
**Test category:** integration (through `oflm serve --asr 1`)
**Acceptance criteria:**
- A clip of clear speech comes back as text containing what is said in it.
- A request with a missing or empty `file`, or a missing `model`, is refused with **400**
  and an OpenAI-shaped error naming the field. This is the client's error; a 5xx here means
  the request reached something that threw instead of being validated
  (`SERVER-REQUEST-VALIDATION`).
- The response names a model.

**Measured 2026-09-20:** the suite exists (`oflm-test`'s transcription task, run as part of
`--audio`) and passes 3/3 against the closed engine. It was written because the audio suite
posts `chat.completions` with `input_audio`, which Whisper refuses as a non-chat model, so
**nothing exercised the endpoint at all**.

It found one defect immediately: the route built its JSON with `parts["file"]`, and
`std::map::operator[]` default-constructs a missing part, so an absent file read as a
present string of the right type and died in the audio decoder as **HTTP 500**. Fixed by
passing on only the parts that are there, and refusing an empty file. Verified both ways:
the pre-fix binary answers 500 (which T2 now judges FAIL), the fixed one answers 400
`missing_required_parameter`.

### OPEN-WHISPER-HOST-PROTOCOL: the language token the host never feeds
**Applies to:** `src/common/whisper/modeling_whisper.cpp`
**Test category:** offline (goldens under both protocols)
**Acceptance criteria:** the decoder prompt is stated and measured, not assumed. Today the
host samples the language after SOT and then feeds `transcribe`, so the model sees
`[SOT, transcribe, ...]` where transformers sends `[SOT, <lang>, transcribe, ...]`.

**Measured 2026-09-19:** on the six golden clips' first windows, in float64, the two
protocols give **identical text on 6/6**; the only difference is one closing timestamp
(`<|3.24|>` vs `<|1.48|>` on a 3.24 s clip). A wrong closing timestamp does move the next
window's start, because the host chunks from the last timestamp. Any change here is a
separate PR and must be measured on both engines.

### OPEN-WHISPER-PERF: what may be called a performance number
**Applies to:** all of the above
**Acceptance criteria:** NPU figures come from hardware traces or static instruction
counts. Harness and engine wall clock is a host-side observation and is labelled as such,
never quoted as array time. End-to-end wall clock is quoted only against a paired run of
the other engine, on a quiet machine.

## Not settled

- **The open container is not published**, so there is no registry entry for it yet: an
  entry pointing at a repository that does not exist is the failure this file exists to
  avoid. It is built locally with `q4nx-build --open-whisper` and selected with
  `--asrmodel` from a user registry.
- **Encoder attention on the array** is a later phase, gated on measuring the host's share.
  Its PV product has N = 64, which does not tile at 32 x 8 columns.
- **The decoder** runs on the host first. Moving it to the NPU needs an `attn.h` variant
  with no RoPE and no cache append, plus a decoder layer design with LayerNorm, biases and
  a non-gated GELU.
