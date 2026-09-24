# whisper_fa: the Whisper encoder's bidirectional FlashAttention, as our own IRON source

`attn_fa.py` is an IRON (mlir-aie) port of AMD's MLIR-AIR FlashAttention example
(`Xilinx/mlir-air`, `programming_examples/flash_attention/kernel_fusion_based` at `e91630a`, with Whisper-specific kernel changes,
MIT -- see `PROVENANCE.md`), built by our pinned mlir-aie + Peano toolchain instead of
MLIR-AIR's. It is what `src/open_whisper`'s `fa_attention.cpp` loads at
`<kernels_dir>/fa/` (`OW_ATTN=auto`, the default) instead of running bidirectional
attention on the host.

Fixed shape (`fa_attention.hpp`'s own contract, checked by `fa_guards.hpp` at load,
never assumed): H = 20 heads, dk = dv = 64, lq = lk = 1536 (1500 real frames padded),
valid_len = 1500, fp32 running-sum state, bf16-via-bfp16-emulation matmul.

## Files

- `attn_fa.py` -- the IRON design (`DESIGN`/`SPECIALIZE`, `build_design.py`'s
  convention). `flash_attn`'s body is unmodified from the design that was verified
  byte-identical to AMD's own AIR-compiled kernel at this exact production shape on
  three real Whisper layers (NpuEmbeddings task 0181, round 6) -- only the module
  header and the `DESIGN`/`SPECIALIZE` tail were added for this tree. Its own header
  comment has the full AIR-to-IRON topology mapping; `local_compute_one`/`merge`/
  `make_bot_fn`/`make_mid_fn`/`make_top_fn`'s docstrings record the three bugs found
  porting it (two DMA layout inversions, IRON's 1 KB default core stack against AIR's
  2 KB, and a per-runtime-TaskGroup drain mismatch) -- kept because CLAUDE.md's rule in
  the sibling `NpuEmbeddings` repo is that failures are the valuable part of a record,
  and every one of these was silent (wrong output or a hang with no diagnostic, not a
  compile error).
- `attn_npu2.cc` -- AMD's compute kernels (matmul, softmax, merge arithmetic), vendored
  unmodified at the C++ source level. `attn_fa.py`'s IRON topology calls these exact
  functions; the port changed nothing inside this file. See its own header for what
  changed around it.
- `zero.cc` -- AMD's zero/neg-inf fill helpers, vendored unmodified, from the same
  upstream directory as `attn_npu2.cc` (`#include "zero.cc"`, quote-form: this
  directory, not `mlir-aie`'s own `aie_kernels/aie2p/zero.cc`, which is a different
  implementation).
- `attn_cascade_wrap.cc` -- ours. `#include`s `attn_npu2.cc` verbatim and adds only
  `cascade_get3`/`cascade_put3`, the two-hop cascade-port transfer IRON's
  `CascadeFlow` needs that AIR's compiler otherwise emits inline. See its own header.

## Build it

Built automatically by `export_whisper_kernels.py` (the normal Whisper kernel-set
build -- see `open_kernels/README.md` and `src/open_whisper/README.md`, "Speed
defaults"), which writes it to `<out>/fa/` (`air.xclbin`, `air.insts.bin`, `fa.json`)
alongside the seven GEMM streams. `--no-fa` skips it.

To build just this design directly (e.g. a debugging/smoke-shape build):

```
. C:\dev\mlir-aie\iron_env.ps1
$env:PATH = "C:\Xilinx\XRT;" + $env:PATH
python open_kernels\build_design.py open_kernels\designs\whisper_fa\attn_fa.py <out_dir>
```

`FA_LQ`/`FA_LK`/`FA_LQP`/`FA_LKP`/`FA_DK`/`FA_DV`/`FA_NUM_HEADS`/
`FA_HEADS_PER_UNROLL`/`FA_CASCADE_STAGES`/`FA_VALID_LEN` override the production shape
(defaults above) for a smoke build; `export_whisper_kernels.py` never sets them, so the
shipped kernel set always builds the production shape.

## Test it on hardware

Real Whisper Q/K/V fixtures and the harness that checks a build against AMD's own
AIR-compiled kernel and a float64 SDPA reference (per-layer, plus a negative control)
are `NpuEmbeddings/tasks/0181-fa-iron-port/round6/52_compare_prod_vs_routeA.py` and its
sibling `53_determinism_prod.py` -- not yet ported into this tree's own harness
(`open_kernels/harness/`). See `NpuEmbeddings` task 0181's `TASK.md` Part 7 for the
from-scratch build's hardware results, which reused those scripts pointed at this
design's output, and the whole engine (`open_whisper_cli`) run against it end to end.

## Measured properties (no NPU performance claims -- see the note below)

Byte-identical to AMD's own AIR-compiled kernel at this production shape, on three real
Whisper layers (0, 15, 31), bit-reproducible across repeated dispatches (determinism
checked both sides). Descriptor count: 240 `aie.lock` / 2072 `aie.dma_bd` against AIR's
own 192 / 2040 (like for like -- both counts include the runtime sequence's own
descriptors) -- 25% more locks, 1.6% more BDs, which trap 7b (`NpuEmbeddings/CLAUDE.md`)
prices at roughly 350 us more per context switch into this kernel. Full numbers, the
six rounds of bisection that got here, and the three refuted hypotheses along the way:
`NpuEmbeddings/tasks/0181-fa-iron-port/TASK.md`.

Host-observed device time (backend-reported submit+wait, NOT a hardware trace -- rule 1
of the sibling repo's CLAUDE.md: never an NPU performance claim on its own) was
~2.6-4.0% higher than AIR's own build on the same three layers at the time this was
measured (0181's round 6). No hardware-trace-based comparison has been made.
