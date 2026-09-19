# open_whisper (phase 2b, issue #72)

A C++ Whisper-large-v3-turbo **encoder** that runs every matrix product on
the NPU through the already-built `whisper_gemm` kernel set (one xclbin,
seven instruction streams, one `hw_context`) and everything else -- im2col,
LayerNorm, GELU, bidirectional attention, bias, residual, bf16 rounding --
on the host in fp32. No decoder, not wired into `oflm.exe` (this is a
standalone build, like `../open_qwen36/build.cmd`), no NPU performance
claims (see the note at the bottom).

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
- `cli.cpp` -> `open_whisper_cli.exe` -- the gate itself.
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
    --golden <clip>.safetensors [--forced]
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

## Performance

Not measured as a claim. `cli.cpp` prints host-side stage timers labelled
"host wall clock" -- attention dominates (about 4.3 s of a 4.4-4.8 s encode),
followed by NPU dispatch (submit+wait, itself dominated by hardware but
still a host observation, not a hardware trace). No number here is an NPU
performance claim.
