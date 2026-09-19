# open_whisper (phase 2b encoder + phase 3 decoder, issue #72)

A C++ Whisper-large-v3-turbo engine. The **encoder** (phase 2b) runs every
matrix product on the NPU through the already-built `whisper_gemm` kernel set
(one xclbin, seven instruction streams, one `hw_context`) and everything else
-- im2col, LayerNorm, GELU, bidirectional attention, bias, residual, bf16
rounding -- on the host in fp32. The **decoder** (phase 3) is 4 layers, entirely
on the host in fp32, with no NPU dispatch at all: it is a KV-cache generation
loop over d_model 1280, 20 heads x 64, FFN 5120, vocab 51866 (tied to
`embed_tokens` -- there is no `lm_head` tensor), reading its cross-attention
K/V straight from the encoder's fixed `Encoder::xkv()`. Not wired into
`oflm.exe` (this is a standalone build, like `../open_qwen36/build.cmd`), no
NPU performance claims (see the note at the bottom -- the decoder has none to
make in the first place, since it never touches the array).

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
