# open_kernels/harness — run an IRON kernel from a .cfg

`run_kernel` is the N-buffer opcode-3 runner for the open kernels: it loads an
xclbin, binds up to 8 buffers, submits, waits, and prints the ERT state and
wall time per run. It speaks the same `.cfg` language as phlegm's driver, so
every design's `make_test.py` / `compare.py` pair works unchanged:

```
device
xclbin  G  build/final.xclbin
kernelx k  G build/insts.bin          # or: kernel k G build/insts.elf  (ELF route)
buf w 10485760 w_qkv.bin              # size, optional init file (zeroed otherwise)
buf x 4096     x_qkv.bin
buf y 32768
run k w x y                           # args 3.. ; arg 0 = opcode 3, 1/2 = instr BO / words
dump y y_qkv.bin 32768                # [bytes [offset]]
```

Also `load <buf> <file>`, `copy <dst> <off> <src> <off> <n>`, `#` comments.
Relative paths resolve against the cfg's directory. `HARNESS_KEEP_GOING=1`
continues past a failed run so later dumps show how far the cores got, and
`HARNESS_TIMEOUT_MS` (default 60000, 0 = block) bounds each run so a hung array
is reported instead of wedging the process.

Three directives patch a `kernelx` instruction stream between runs, which is how
one compiled program per layer type serves every layer and every token:

```
moeroute2 <kernel> <buf> <idx-off>   # point the expert fills at the router's top-8
moeroute  <kernel> <rout-buf>        # same, for moe_experts' host-concatenated weights
attnpos   <kernel> <pos>             # KV window length, new-row offset, RoPE record
```

An mlir-aie stream is a sequence of ops; op `0x81` is a DDR patch naming a
register, a buffer arg and a byte offset, so re-pointing a DMA is one word plus
an instruction-BO sync (~0.04 ms per layer). See `../model/` for the decode
program that uses them.

## Build

- **Windows (MSVC):** `build.cmd` → `out\run_kernel.exe`. Needs XRT headers and
  an import lib for `xrt_coreutil.dll`, neither of which ships here: set
  `XRT_INCLUDE_DIR` (a Xilinx/XRT checkout's `src/runtime_src/core/include`)
  and `XRT_LIB_DIR` (the directory holding `xrt_coreutil.lib`, made from the
  system DLL — `src/WinSetup.md`); the script stops with that message if
  either is unset.
- **Linux:** `cmake -S . -B build && cmake --build build` with XRT at
  `$XILINX_XRT` or `/opt/xilinx/xrt`. Or `-DFLM_BUILD_OPEN_KERNELS_HARNESS=ON`
  from `src/`.

## Test a design

```
# WSL, ironvenv (mlir-aie 1.4.2), from open_kernels/:
python build_design.py designs/ln/ln.py
GEMV_N=8192 GEMV_K=2048 GEMV_RS=2 GEMV_CORES=8 \
  python build_design.py designs/gemv_q4/gemv_q4.py designs/gemv_q4/build_qkv
(cd designs/ln && python make_test.py)                       # random fixtures
(cd designs/gemv_q4 && python make_test.py --region qkv)     # synthetic Q4_1 -> pool order

# Windows (NPU), from the design dir:
..\..\harness\out\run_kernel.exe run.cfg && python compare.py
```

Generated cfgs name every path relative to the design directory, so the same
`run.cfg` works from WSL and from Windows. Fixtures that slice weights out of
captured FLM buffers (`router`, `dn_glue`, `dn_post`, `moe_combine`,
`deltanet`, the fused-layer tests) read them from `$OPEN_KERNELS_CAPS` and say
so when it is unset — see `../fixture_paths.py`. The six kernel sets the
engine loads are built by `../export_qwen36_kernels.py`
(`src/open_qwen36/README.md`).

`bench.py <cfg> --driver <exe...> [--driver ...]` runs a cfg through one or more
drivers and prints min / median / max per kernel.

## Results (2026-09-04, Strix, Windows, XRT; mlir-aie 1.4.2 builds)

Every design rebuilt from this tree, run through `run_kernel`, compared against
its fp64 reference. Timing is start→wait per run, median of 18 warm runs.

| design | fixture | check | run_kernel | phlegm driver | phlegm README |
|---|---|---|---|---|---|
| `ln` (RMSNorm+residual, 2048) | random | y maxrel 5.4e-8, xn bit-exact | **0.21 ms** | 0.27 ms | — |
| `gemv_q4` qkv 8192×2048 (10.5 MB) | synthetic Q4_1 → pool | cos 1.0, maxrel 3.3e-6 | **0.92 ms** | 0.89 ms | 0.50–0.55 ms |
| `gemv_q4` qkv | captured FLM pool | cos 1.0, maxrel 1.6e-5 | 0.59 ms (5 runs) | ~1.0 ms | 0.50–0.55 ms |
| `lm_head_q8` full vocab 248320 (540 MB) | captured FLM pool | cos 1.0, maxrel 4.6e-6 | **15.9 ms** | 34 ms (noisy) | 15.6 ms; FLM closed 15.4 ms |
| embedding `m512_768x768` bf16→f32 | random | bit-identical to shipped xclbin | 1.0 ms | — | — |

Synthetic and captured Q4_1 weights give the same kernel behaviour, which is
the point of the synthetic path: no model or captured buffers are needed to
test the kernel. The gemv gap vs phlegm's README number was seen through both
drivers and is a box effect (the machine was under memory pressure that day),
not a host-side one. phlegm's driver exits with an access violation in XRT
teardown after all work is done (known, harmless).

End-to-end context, same day, same box: FLM 1.0.2 on the stock Qwen3.6-35B-A3B
decoded at 0.62 tok/s — but with 3 GB of RAM free and ~100 hard page faults/s;
phlegm measured the same build at ~6.8 tok/s on a quiet box. phlegm's open
kernels do the pruned 27B at ~155 ms/token (6.1 tok/s). Whole-model numbers
from *this* tree wait for the packer + resident driver (Phase B/C).
