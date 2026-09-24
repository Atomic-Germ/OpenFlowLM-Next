# open_whisper (phase 2b encoder + phase 3 decoder + phase 3b wiring, issue #72)

A C++ Whisper-large-v3-turbo engine. The **encoder** (phase 2b) runs every
matrix product on the NPU through the already-built `whisper_gemm` kernel set
(one xclbin, seven instruction streams, one `hw_context`) and everything else
-- im2col, LayerNorm, GELU, bidirectional attention, bias, residual, bf16
rounding -- on the host in fp32. The **decoder** (phase 3) is 4 layers, entirely
on the host in fp32, with no NPU dispatch at all: it is a KV-cache generation
loop over d_model 1280, 20 heads x 64, FFN 5120, vocab 51866 (tied to
`embed_tokens` -- there is no `lm_head` tensor), reading its cross-attention
K/V straight from the encoder's fixed `Encoder::xkv()`. **Phase 3b wires both
into `oflm.exe`** (`engine_adapter.cpp`, `src/common/whisper/
whisper_engine_select.cpp`), so `oflm serve --asr 1` can transcribe with this
engine instead of the closed `whisper_npu` -- see "Phase 3b" below. No NPU
performance claims (see the note at the bottom -- the decoder has none to make
in the first place, since it never touches the array).

## Files

- `weights.hpp/.cpp` -- loads `model.open.safetensors` (via
  `open_qwen36::Q4nxFile`, which is a plain safetensors reader and needed no
  changes), checks `weights_manifest.json`'s format and `config.json`'s
  geometry, and pre-tiles every `[K,N]` GEMM operand with `tile_b()` (ported
  with attribution from NpuEmbeddings' `npue_pack.cpp`, whose `tile_b` is not
  exported from that translation unit).
- `kernels.hpp/.cpp` -- finds the kernel set (`OFLM_WHISPER_KERNELS_DIR`, else
  `<model_dir>/open_kernels`), checks `whisper_kernels.json`'s format,
  `complete` flag and `hf_config_check` against `config.json`, checks
  `design.json`'s `b_layout` tuple against what `weights.cpp` tiled with, then
  loads the seven instruction streams into one `npue::npu::Design` and drives
  dispatches.
- `host_ops.hpp/.cpp` -- LayerNorm, exact-erf GELU, bias/residual add, im2col,
  bidirectional multi-head attention, and bf16 rounding (the AVX2
  `bf16_fill`/`bf16_read` are ported with attribution from NpuEmbeddings'
  `npue_encoder.hpp`).
- `encoder.hpp/.cpp` -- `class Encoder`: stages all 131 weight buffers once,
  runs the stem + 32 layers + final LayerNorm + cross-KV, with a `StageHook`
  for capturing intermediates and `run_layer_from()` for teacher forcing.
- `decoder.hpp/.cpp` -- `class Decoder` (phase 3): reads the container's
  decoder tensors at their natural `[out, in]` bf16 layout (no NPU tiling --
  the decoder never dispatches), keeps them bf16 and widens on the fly per
  dot product (`linear()`/`dot_bf16()`, ported with attribution from
  `open_qwen36/vision/vit.cpp`'s `linear()`/`widen_avx2`), and runs one
  `step()` per token: embed + position, 4 layers of (causal self-attention
  over a growing KV cache, cross-attention over the encoder's fixed K/V,
  GELU FFN), final LayerNorm, then the tied head (`logits = h . embed_tokens^T`).
  `clear_context()` resets the self-attention cache and position counter only
  -- cross-attention K/V (`set_encoder_output()`) survives it, since it is
  fixed for the whole 30 s window.
- `cli.cpp` -> `open_whisper_cli.exe` -- the gate itself, `--decode hf|host`
  for the decoder.
- `build.cmd` -- standalone MSVC build, modelled on `../open_qwen36/build.cmd`.
- `engine_adapter.hpp/.cpp` (phase 3b) -- `class OpenWhisperEngine`, a
  `whisper_engine` (`src/include/whisper/whisper_engine.hpp`) over one
  `Encoder` + one `Decoder`. `encode_audio()` widens the host's bf16 mel to
  fp32 and calls `Encoder::encode()`, then hands `Decoder::set_encoder_output()`
  the encoder's own `xkv()` buffer (not a copy -- `Encoder` outlives the
  engine). `decode_audio()` calls `Decoder::step()` and narrows its fp32
  logits back to bf16. The constructor checks the `Whisper_Config`-reported
  padded vocab width against `DecoderGeometry::vocab_padded` rather than
  assuming they agree.

## Build

```
set XRT_INCLUDE_DIR=C:/dev/XRT/src/runtime_src/core/include
set XRT_LIB_DIR=C:/dev/xrtNPUfromDLL
build.cmd
```

-> `out\open_whisper_cli.exe`

## Run

```
set PATH=C:\Xilinx\XRT;%PATH%
out\open_whisper_cli.exe --model <model_dir> --kernels <kernel_set_dir> ^
    --golden <clip>.safetensors [--forced] [--decode hf|host]
```

`--model` is an `oflm-open-whisper-v1` container directory (e.g.
`Whisper-V3-Turbo-OpenNPU2`). `--kernels` is a `whisper_gemm` export
directory; if omitted the engine looks for `OFLM_WHISPER_KERNELS_DIR`, then
`<model_dir>/open_kernels`. `--golden` is one of
`open_kernels/model/whisper_goldens.py`'s clips.

The CLI prints, per stage, `cos`/`rel` against the golden float64 forward
pass: `conv1`, `conv2`, `enc.hidden.<1..32>` (chained -- this encoder's own
output feeding the next layer), `enc.out`, `dec.<0..3>.xk`/`.xv` (cross
K/V), then (with `--forced`) each layer run in isolation from the golden
`enc.hidden.<i>`, and finally host-side stage timers (all labelled "host
wall clock", never an NPU performance claim). Exit code is nonzero if
`enc.out`'s cosine is below 0.99 or any output contains NaN/Inf.

`--decode hf|host` additionally runs the phase-3 decoder gate on that
protocol's golden token sequence (`open_kernels/model/whisper_goldens.py`'s
`hf.tokens`/`hf.logits` or `host.tokens`/`host.logits`), mirroring
`open_kernels/model/whisper_decode_check.py`'s protocol in C++ with a KV
cache: teacher-forced argmax agreement and logits cosine over the
free-running region (the forced prefix -- `[SOT, lang, transcribe,
<|0.00|>]` for `hf`, `[SOT, transcribe]` for `host`, per
`whisper_goldens.py` -- is a prompt, not a prediction, so it is excluded),
then a from-scratch greedy free-run from that prefix compared token for
token against the golden path. Also prints whether logits
`[vocab, vocab_padded) = [51866, 51872)` are `-inf` on every step (the host
sampler's padded width; the pad must never win an argmax or a sample), and
decode timers (embed/layer_norm/linear/attention/gelu, ms/token, tok/s --
host wall clock; the decoder never dispatches to the NPU, so there is no
NPU-side split to report). Exit code is nonzero unless argmax agreement is
100% and the free-run matches the golden path exactly, in addition to the
encoder's own PASS condition above.

## The bug this phase actually found: never write into a device-mapped buffer

The first version of this engine added the bias in place in the GEMM's C
buffer (`add_bias(const_cast<float *>(fc1_c), ...)`, and the same for conv1
and conv2). That buffer is **mapped from the device**, and the NPU writes into
it on every dispatch. Writing into it from the host leaves dirty CPU cache
lines on that mapping, and when those lines are written back -- at a moment
nothing in the program controls -- they land on top of what a later dispatch
DMA'd into the same buffer.

What it looked like before the cause was known:

- Two runs of the identical binary on the identical clip gave different
  per-layer cosines, with one large drop at an unpredictable layer.
- Serialising every host loop (`num_threads(1)`) appeared to fix it, so it
  read as a race in MSVC's OpenMP runtime. It was not: serialising only
  changed the timing of the write-back.
- Per-row analysis of the dumped layers showed single WRONG ROWS appearing
  (row 682 at layer 15 in one run), spreading through attention afterwards --
  a few rows out of 1500, which an overall cosine hides.

How it was pinned down, with the three diagnostics that are still in the code
because they are cheap and this class of bug is invisible without them:

- `--stress N`: N dispatches cycling four streams and 32 weight slots, each
  compared against that (layer, op)'s first result. **512/512 identical** --
  so neither the array nor the stream/slot switching is the problem. (The
  harness agrees: five identical `run_kernel` dispatches are byte-identical.)
- `OW_VERIFY_A=1`: reads A back off the device after every dispatch and
  compares it with what was uploaded. **No mismatch ever** -- so the operand
  reaching the core is the operand we sent.
- `OW_DOUBLE_CHECK=1`: dispatches every GEMM twice with the same bound
  buffers and compares. **Caught it**: C differing in whole 64-byte-aligned
  runs (16 floats at a time) between two dispatches of the same input.

The 64-byte granularity is the tell: a cache line, not a DMA block and not
arithmetic. With every C buffer treated as read-only (`gelu_bias()` reads C
and writes elsewhere), two runs agree to every digit, all host loops are
threaded again, and the encode is 4.4 s instead of 20-30 s.

## Gate (2026-09-20, idle machine)

Both clips pass, and every figure sits at `replica_whisper.py`'s bf16 ceiling:

| | Demos_sample-data_journal | nvidia |
|---|---|---|
| conv1 / conv2 | 0.99999785 / 0.99999992 | 0.99999890 / 0.99999995 |
| chained L00 out (`enc.hidden.1`) | 0.99999810 (replica 0.99999810) | 0.99999814 (replica 0.99999814) |
| chained `enc.hidden.32` | 0.99987223 (replica 0.99988110) | 0.99991683 (replica 0.99984733) |
| **`enc.out`** | **0.99828836** (replica 0.99822134) | **0.99906620** (replica 0.99905335) |
| cross K/V worst | 0.99788008 | 0.99901179 |
| teacher-forced, every layer | >= 0.99999827 | >= 0.99999825 |

Two runs of each are identical to all eight printed digits.

## Decode gate (phase 3, 2026-09-20, idle machine)

`--decode hf` and `--decode host`, on the encoder output measured above (so
these numbers already carry the encoder's own bf16-ceiling error). All six
runs (3 clips x 2 protocols) pass: **100% teacher-forced argmax agreement**
and an **exact free-run token match** against the golden path.

| | Recording | Demos_sample-data_journal | nvidia |
|---|---|---|---|
| hf: tokens / argmax agreement | 9 / 5/5 | 32 / 28/28 | 97 / 93/93 |
| hf: logits cosine mean / min | 0.99997547 / 0.99988179 | 0.99997601 / 0.99979626 | 0.99997313 / 0.99957159 |
| hf: free-run | MATCHES (9 tok) | MATCHES (32 tok) | MATCHES (97 tok) |
| host: tokens / argmax agreement | 8 / 5/5 | 31 / 28/28 | 96 / 93/93 |
| host: logits cosine mean / min | 0.99997247 / 0.99986626 | 0.99997768 / 0.99980397 | 0.99997296 / 0.99969309 |
| host: free-run | MATCHES (8 tok) | MATCHES (31 tok) | MATCHES (96 tok) |

`vocab pad [51866,51872)` reads `-inf` on every step in every run. Two runs
of `--decode hf` on `nvidia` agree on every printed cosine, agreement count
and free-run result to all eight digits; only the (labelled) host wall-clock
timers differ between runs, as expected.


## The one clip where bf16 changes a token, and why the gate still passes

`output_voice_clone` under the **host** protocol is the one (clip, protocol) pair of the
twelve where the free-run does not reproduce transformers' float64 path: it diverges at
token index 17, `316` (" A") where float64 says `497` (" R"), and teacher-forced argmax
agreement is 40/41 instead of 41/41.

That is the **datapath**, not this engine. The numpy replica
(`open_kernels/model/replica_whisper.py`, bf16 operands, no NPU) fed to the exact float64
decoder diverges at the **same index, to the same token**, and ends on the same 43-token
path. Two independent implementations of the same bf16 datapath take the same turn.

So the bf16 token path is recorded rather than argued about:

```
python open_kernels/model/whisper_decode_check.py --model-dir <hf snapshot>     --goldens <goldens> --enc-dir <goldens>/replica_bf16_enc --proto both     --write-baseline <goldens>/bf16_token_baseline.json
```

and `--baseline <file>` makes the gate accept a free-run that matches that path exactly,
while still printing the float64 divergence. A gate that can never pass is one its reader
learns to skip.

## Performance

Not measured as a claim. `cli.cpp` prints host-side stage timers labelled
"host wall clock" -- attention dominates (about 4.3 s of a 4.4-4.8 s encode),
followed by NPU dispatch (submit+wait, itself dominated by hardware but
still a host observation, not a hardware trace). No number here is an NPU
performance claim.

The decoder never touches the NPU: `--decode`'s own timers (also host wall
clock) show ~13-14 ms/token, ~70-77 tok/s, split across
embed/layer_norm/linear/attention/gelu -- `linear` (the per-layer projections
plus the 51866-wide tied head) and `attention` (cross-attention over 1500
encoder rows, every layer, every step) dominate. This is a plain,
single-threaded-per-call generation loop with no batching or speculative
decoding; it exists to gate correctness, not to claim a decode rate.

## Phase 3b: wired into `oflm.exe` (2026-09-20)

`OFLM_WHISPER_ENGINE=open|closed` (unset: auto) selects between this engine
and the closed `whisper_npu` in `src/common/whisper/
whisper_engine_select.cpp`, the sole factory `make_whisper_engine()` declared
in `whisper/whisper_engine.hpp`. Auto picks open when the model directory has
BOTH `model.open.safetensors` and a kernel set (`OFLM_WHISPER_KERNELS_DIR`,
else `<model_dir>/open_kernels`, else `<xclbins root>/xclbins/<model name>/open_kernels`); else closed when `model.q4nx` is there;
else it refuses, naming both missing paths. Every branch logs which rule
fired (`modeling_whisper.cpp` also logs `describe()` right after, so a load
prints two lines: which rule chose an engine, then which engine and kernel
set it built). Wrapped in `#ifdef OFLM_USE_OPEN_WHISPER` (set by
`src/CMakeLists.txt` alongside `OFLM_USE_OPEN_QWEN36`, off for `OFLM_USE_HRX`
builds), so an HRX build still compiles with only the closed adapter and
`OFLM_WHISPER_ENGINE=open` refuses by name instead of failing to link.

CMake adds `weights.cpp kernels.cpp host_ops.cpp encoder.cpp decoder.cpp
engine_adapter.cpp` to the `oflm` target (never `cli.cpp` -- that stays
`build.cmd`'s own standalone gate binary and carries its own `main()`), with
a per-source `INCLUDE_DIRECTORIES` pointing at `open_npue/` so `kernels.hpp`'s
bare `#include "npu_device.hpp"` resolves without adding that directory to
the target's global include path (the same "same basename, different file"
hazard `OPEN_NPUE_SOURCES`'s own CMake comment warns about), and `/arch:AVX2
/openmp` (MSVC) / `-mavx2 -mfma -fopenmp` (else), matching `OPEN_NPUE_SOURCES`.

**Kernel set placement.** `whisper_engine_select.cpp`'s `find_open_kernels` searches, in
order, `OFLM_WHISPER_KERNELS_DIR`, `<model_dir>/open_kernels`, then
`<root>/xclbins/<model name>/open_kernels` for every xclbins root -- the same order
as the other open engines (`open_qwen36/engine.cpp`). The last is where
`export_whisper_kernels.py` writes by default, so building this tree is enough; the
server log names the directory and the rule that chose it. Earlier setups placed a
set beside the model instead: Verified against the local `whisper_gemm` export
by making `<model_dir>/open_kernels` an NTFS junction to
`NpuEmbeddings_scratch/whisper-kernels` (`New-Item -ItemType Junction`; no
admin needed, unlike a symlink) -- nothing is copied into the model directory
or the repository.

**Model directory used for verification.** No registry entry exists for the
open container (`Whisper-V3-Turbo-OpenNPU2`), and none was added -- an entry
naming an unpublished HF repo would be worse than none (decided in an earlier
phase). Rather than a new tag, `model.open.safetensors` + `weights_manifest.json`
(hardlink + copy from the open container) and `open_kernels` (junction, above)
were added ALONGSIDE the closed container's own files in the existing
`whisper-v3:turbo` model directory
(`<FLM/OFLM model root>/Whisper-V3-Turbo-NPU2/`), which already had
`model.q4nx`, `config.json`, `tokenizer.json`, `tokenizer_config.json`. This
is the least invasive option that needed zero code or registry changes: the
SAME directory now satisfies `ModelDownloader::is_model_downloaded()`'s
"files" check (untouched, closed container's files) for either engine, and
`OFLM_WHISPER_ENGINE` alone picks which weights load. Confirmed the closed
engine still starts unmodified from the same directory (`OFLM_WHISPER_ENGINE=
closed`).

**End-to-end transcripts, `oflm serve --asr 1`, both engines, one quiet
development machine (not a controlled idle-machine benchmark -- single
`curl` runs, not averaged; all times are HOST WALL CLOCK for the whole HTTP
round trip -- FFmpeg decode, mel, encode, decode loop -- and are NOT an NPU
performance claim):**

| clip | open text | closed text | open / closed wall |
|---|---|---|---|
| `Recording.wav` (zh, 3.2 s) | `这是什么?` | `- This is something.` | 4.20 s / 2.19 s |
| `Demos_sample-data_journal.wav` (ko, 10.1 s) | ` 오늘 일요일에 일어났어요. 아침이 아름다웠` | ` 오늘 하루에 일어나서 고생이 되었습니다. 아침은 아름다` | 5.52 s / 3.51 s |
| `dead-faith-audiobook-example.mp3` (en, 89.1 s) | matches golden `generate_ts0` to a handful of tokens (below) | likewise, independently | 22.22 s / 17.92 s |
| `nvidia.mp3` (en, 110.2 s) | drops a ~5-sentence span the golden and the closed engine both have (below) | closer to golden than open on this clip | 22.19 s / 17.86 s |

The open engine is consistently slower end to end here -- the encoder's own
gate above already says why: attention dominates the encode (~4.3 of 4.4-4.8 s
per 30 s window), on the host, unoptimized relative to the closed engine's
production dispatch path. Nothing here separates encode time from decode
time from FFmpeg/mel time; that split is `cli.cpp`'s job, not the server's.

**Against the float64 golden
(`NpuEmbeddings_scratch/whisper-goldens/meta.json`), per clip:**

- **`Recording.wav`** -- **exact match.** Open: `这是什么?`. Golden `host.text`
  (first, only window): `这是什么?`. Byte-identical.
- **`Demos_sample-data_journal.wav`** -- **diverges after the first sentence,
  and does not finish.** Golden `host.text` (one window, 10.1 s clip):
  ` 오늘 일요일에 일어났어요. 아침을 잘 먹었어요. 그리고 커피를 즐기고 싶었어요.`
  ("...woke up on Sunday. Ate breakfast well. And wanted to enjoy coffee.").
  Open: ` 오늘 일요일에 일어났어요. 아침이 아름다웠` -- agrees on the first
  clause, then diverges to an unrelated, truncated continuation ("the morning
  was beautif-"). **The closed engine, run through the SAME live server
  pipeline, shows the same shape of failure** (` 오늘 하루에 일어나서 고생이
  되었습니다. 아침은 아름다` -- also diverges after the first clause, also
  truncates mid-word on "아름다"). Since both engines break the same way on
  the real FFmpeg-decode-plus-live-mel path while `cli.cpp`'s gate (fed a
  pre-computed golden mel, bypassing FFmpeg/FFTW entirely) passes this exact
  clip at cosine >=0.9999 per layer, the evidence points at something
  upstream of the `whisper_engine` seam -- most likely the live mel
  extraction (`modeling_whisper.cpp`'s FFmpeg decode + FFTW log-mel) differing
  from the Python reference pipeline that produced the golden mel, or a
  30 s-window silence-padding effect on a 10 s clip -- not a phase-3b defect
  in `engine_adapter.cpp` or the decoder. Not chased further here; worth its
  own thread if `oflm-test` starts covering short non-English clips.
- **`dead-faith-audiobook-example.mp3`** -- **effectively matches** the
  golden's own whole-clip greedy free-run (`generate_ts0`, produced by
  `whisper_decode_check.py`'s reference decoder chained across windows the
  same way the live host chunks). Word-for-word the same content throughout
  89 s; the handful of differences are single-token, matching the class of
  drift the encoder gate's own "bf16 changes a token" section documents
  (`seem` vs `seemed`, `striking` vs `strikingly`, final word `identical` vs
  `identifiable`, a missing `?`) -- not a structural divergence.
- **`nvidia.mp3`** -- **a real, structural divergence, not just token drift.**
  The first ~30 s matches the golden's first-window `host.text` closely
  (one extra word: "I got myself **a** two years ago" vs golden's "I got
  myself two years ago"). Past that window, the open transcript SKIPS a span
  present in both the golden's whole-clip `generate_ts0` AND the closed
  engine's own transcript of the same clip through the same live pipeline:
  golden/closed have "...maybe NVIDIA would kind of chip in and do something
  for it. And they said flat out, no, we're not doing any support. And I was
  like, well, we're playing in the same sandbox. Why can't we be nice to each
  other?" between "...working pretty nice." and "...hardware producers think
  about the other stuff as well" -- the open engine's transcript goes
  straight from one to the other, several sentences shorter. This is the one
  finding here that looks like it could be a phase-3b-specific issue (a
  timestamp/chunk-boundary decision made differently from the closed engine
  partway through a long multi-window transcript) rather than a pre-existing
  live-pipeline characteristic, since the closed engine does NOT show the
  same gap on the same clip. Not root-caused in this pass -- flagged here
  with the exact strings so it can be reproduced without re-running the
  server.

**`oflm-test`'s transcription suite: 3/3 PASS against the open engine**
(`dead-faith-audiobook-example.mp3`, T1 content match, T2 no-file refusal,
T3 model field echoed):

```
python -c "import sys; sys.path.insert(0,'.'); from oflm_test.tasks import TranscriptionTask; r=TranscriptionTask('http://127.0.0.1:8090/v1').run(); print(dict(r.verdicts), r.failures)"
{'PASS': 3} []
```

## Speed defaults (2026-09-23, NpuEmbeddings task 0180)

Every variant below passed a 1200-utterance WER gate (LibriSpeech
test-clean/test-other + FLEURS, 9 languages, `NpuEmbeddings/tools/wer/`) --
statistically indistinguishable from, or better than, the exact path -- and
became the DEFAULT. Each is still switchable off by its own env var, and every
one is validated at model load (never lazily on the first request) and
printed to the server log, naming the value in effect and its source
(`default` or the env var). See `NpuEmbeddings/tasks/0180-whisper-fastest/
TASK.md` Parts 9-17 for the WER numbers and paired-test statistics behind
each of these.

| var | values (default in **bold**) | what it does |
|---|---|---|
| `OFLM_WHISPER_PROTOCOL` | **`hf`** (open engine) / `legacy` (closed engine) / explicit `legacy`\|`hf` | HF-faithful greedy decoding (`generation_hf.hpp`) vs. the original per-16-token-watchdog loop. `hf` is only measured on the OPEN engine (WER 16.11% -> 5.50% (open engine, legacy -> hf protocol) on 1200 utterances, sign test p = 1.6e-44) -- unset, the CLOSED engine still defaults to `legacy`. An explicit value overrides for either engine. Validated once in `Whisper::load_model()`, right after the engine loads (`modeling_whisper.cpp`'s `_init_decode_protocol()`), not lazily. |
| `OW_ATTN` | **`auto`** / `host` / `npu` | Bidirectional attention on the NPU (a fused FlashAttention kernel, `fa_attention.hpp`) vs. the host. `auto` uses the kernel when one is found at `<kernels_dir>/fa/` (or `OW_FA_DIR`, which still overrides the search directory) and its `fa.json` matches this engine's geometry (H=20, dk=dv=64, lq=lk=1536, valid_len=1500); `npu` REQUIRES it and refuses if absent or mismatched; `host` never uses it. Always printed which was chosen and why (`encoder.cpp`). `fa.json` records what the kernel build actually was (`heads`/`dk`/`dv`/`lq`/`lk`/`valid_len`/`fp32_state`/`emulate_bfp16`/`mlir_aie_version`/`peano_version`) -- read and checked, never assumed, and it does not matter which toolchain built it. |
| `OW_DEC_XKV` | **`bf16`** / `fp32` | Decoder's gathered cross-attention K/V precision. `fp32` restores the exact path exactly. |
| `OW_DEC_W` | **`int8`** / `bf16` | Decoder linear layers' weight precision (per-output-row symmetric int8). `bf16` restores the exact path exactly. |
| `OW_DEC_HEAD` | **`int8x`** / `int8` / `bf16` | The tied `lm_head` projection's precision (`int8x` recomputes the top-64 logits exactly in bf16 after an int8 sweep). `bf16` restores the exact path exactly. |
| `OW_HOST_FAST` | **`1`** / `0` | Fused, vectorised host ops (AVX2 erf-based GELU, fused LayerNorm+bf16-round, fused bias+residual, fused bias+gather for attention) vs. the original unfused ones. `0` restores exact. |

All six are strict: any value other than the ones listed throws, naming the
env var and the bad value -- never silently read as the default (the
"fails open" class `CLAUDE.md` documents in the sibling `NpuEmbeddings`
repository).

**The GEMM kernel set's datapath (bf16 vs. bf16-via-bfp16-emulation) is a
BUILD-time choice, not a runtime env var** -- it lives in the kernel set
itself (`design.json`'s `emulate_bfp16`, `whisper_kernels.json`'s own copy),
because it is compiled into the xclbin. `open_kernels/export_whisper_kernels.py
--emulate-bfp16` now defaults to **on** (1.71x on the array; WER
indistinguishable from plain bf16 under the `hf` protocol, H4 vs H0);
`--no-emulate-bfp16` builds the plain bf16 datapath instead. The engine
prints which one a loaded kernel set actually is (`kernels.cpp`'s `datapath`
line) -- read from the set, never assumed.

**`fa/` is now produced by the normal kernel-set build, from our own IRON source.**
`open_kernels/export_whisper_kernels.py` builds `<kernels_dir>/fa/` (the
FlashAttention kernel `OW_ATTN` reads above) by default, alongside the seven
GEMM streams, from `open_kernels/designs/whisper_fa/attn_fa.py` -- an IRON
(mlir-aie) port of AMD's MLIR-AIR `attn_npu2.py`/`.cc`
(`kernel_fusion_based`, modified for Whisper), built by the same pinned mlir-aie + Peano
toolchain and verified byte-identical to AMD's own AIR-compiled kernel at
production shape on real Whisper layers (NpuEmbeddings task 0181; see
`open_kernels/designs/whisper_fa/README.md`). `--no-fa` skips it, in which case
`OW_ATTN=auto` falls back to host attention and `OW_ATTN=npu` refuses at load.
There is no longer a separate toolchain (e.g. MLIR-AIR) needed anywhere in the
build of this engine's kernel set.

**The open container builder ships `generation_config.json`.**
`utilities/q4nx-build --open-whisper` now REQUIRES it from the source HF
snapshot (it was optional and silently skipped before, which is how a build
without it reached a test machine) -- the `hf` protocol reads it at load
time and refuses with a clear message if it is absent
(`generation_hf.cpp`'s `GenerationConfig::load`).

**The server log names the whole configuration.** `Whisper::load_model()`
prints, in order: which engine loaded and why, that engine's own
`config_summary()` (the open engine's: `attn=... xkv=... weights=... head=...
host_ops=... gemm_datapath=...`, each with its source), then the decode
protocol and its source.
