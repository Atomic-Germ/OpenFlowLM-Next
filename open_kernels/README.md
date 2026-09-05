> **Vendored from phlegm** (`tools/open-kernels/`, see `PROVENANCE.md`). This file is
> phlegm's status log and trap catalogue, kept verbatim; its paths (`C:/code/phlegm`,
> `C:/caps`, `open-qwen-npu npu`) are phlegm's. In this tree the generators take the
> capture directory from `OPEN_KERNELS_CAPS` and write relative paths
> (`fixture_paths.py`). For building and running the designs from THIS tree use
> `harness/README.md` (`run_kernel`, synthetic fixtures, 1.4.2 pin); for the six
> kernel sets the engine loads, `export_qwen36_kernels.py` (`src/open_qwen36/README.md`).

# open-kernels — our own NPU kernels (IRON / mlir-aie), driven by phlegm

Phase 0a of `.claude/plans/open-kernels-feasibility.md` (2026-09-01): prove we
can build an AIE kernel with the open toolchain and run it on the XDNA2 NPU
through phlegm's own XRT shim, the same way FLM's closed kernels are run.
**Done — ROT13 round-trips byte-exact via both loading paths.**

## Toolchain (WSL builds, Windows runs)

- WSL Ubuntu-24.04: `~/ironenv142` venv with `mlir_aie==1.4.2` (release wheel,
  `-f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.4.2`),
  `llvm-aie` (Peano 21.0.0) from `utils/peano-requirements.txt`, source tree
  `~/mlir-aie` checked out at `v1.4.2` (for `utils/env_setup.sh`).
- The wheel bundles `aiecc`, `aie-opt`, `aie-translate`, `bootgen` but **not**
  `xclbinutil` (needed for the xclbin) nor `aiebu-asm` (needed for the
  `insts.elf` wrap). Both are XRT tools; without sudo in WSL they were built in
  a throwaway `ubuntu:24.04` Docker container from XRT master (2.26.0) and
  installed to `~/xrt-tools/{bin,lib}` (boost .so's alongside). Recipe:
  `scratchpad/build_xclbinutil.sh` of the 2026-09-01 session — `apt` deps,
  `xrtdeps.sh -docker`, `cmake -DXRT_NATIVE_BUILD=yes`, `ninja xclbinutil
  aiebu-asm`.
- Native Windows wheels for `mlir_aie` 1.4.2 and Peano exist (Python 3.11);
  not used yet — same xclbinutil/aiebu gap, and WSL builds are 5 s.

Build shell:
```
source ~/ironenv142/bin/activate
source ~/mlir-aie/utils/env_setup.sh
export PATH=~/xrt-tools/bin:$PATH LD_LIBRARY_PATH=~/xrt-tools/lib
cd /mnt/c/code/phlegm/tools/open-kernels
python build_design.py designs/rot13/rot13.py          # -> designs/rot13/build/{final.xclbin,insts.bin,insts.elf}
```
`build_design.py` pins the device to `npu2` (without it IRON silently targets
NPU1) and calls `DESIGN.specialize(**SPECIALIZE).compile(xclbin_path,
inst_path, elf_path)`; a design module just exposes `DESIGN` and `SPECIALIZE`.

## Running through phlegm

`open-qwen-npu npu <config>` (the decode driver's config language) gained two
directives for open designs:

- `kernelx <name> <xclbin-ctx> <insts.bin>` — classic mlir-aie flow:
  `xrt::kernel(ctx, "MLIR_AIE")` + a cacheable instruction BO bound at args
  1/2 of every run (word count at arg 2).
- `run <kernel> <buf>...` — generic immediate submit, buffers at args 3+.

FLM's flow (`kernel <name> <ctx> <insts.elf>`: `xrt::elf → module →
ext::kernel`) also loads IRON's `insts.elf` unchanged. Example
`designs/rot13/run.cfg`:
```
device
xclbin R .../rot13/build/final.xclbin
kernelx k R .../rot13/build/insts.bin      # or: kernel k R .../build/insts.elf
buf in 1024 .../rot13/in.bin
buf out 1024
run k in out
dump out .../rot13/out1.bin 1024
```
Result 2026-09-01: `run k [2 bufs] -> state 4 (0.602 ms)` classic, `0.447 ms`
ELF; output == ROT13(input) byte-exact for both.

## Designs

- `designs/rot13/` — smoke test. Kernel + dataflow from
  [vegah/LLMNpuTest](https://github.com/vegah/LLMNpuTest) (Apache-2.0,
  `LICENSE.LLMNpuTest`), dispatch half replaced by phlegm's driver. That repo
  is also the trap catalogue to read before writing any kernel here
  (device pin, floor rounding, no fp32 vector multiply on AIE2P, 2-in/2-out
  DMA streams per core, 128-byte shim transfers deliver zeros, ...).

- `designs/expert_fetch/` — phase 0b spike (PASSED 2026-09-02): a shim DMA
  descriptor into a DDR pool retargeted to a runtime-chosen slab by control
  packets bounced through DDR, no host round-trip. `ddr_bounce_fetch.mlir` is
  the proof; the rest is the bisection ladder. See the plan's 0b section.
- `designs/gemv_q4/` — **phase 1**: q4_1 GEMV with in-kernel dequant, consuming
  chunks in the LAYER-POOL order FLM's kernel uses (`pools.rs std_perm`: 64-row
  bands of `K/128` chunks, half = c%2, k-tile = c/2). Tile arithmetic ported from
  vegah's `granite_gemv.h` (same chunk layout). 8 cores, x broadcast, 4 chunks
  per DMA element. Shape via env `GEMV_N/GEMV_K/GEMV_CORES`; `make_test.py
  --region qkv|z|share_up|share_gate|share_down` slices the region out of the
  captured L0 pool, writes an fp64 reference from the same bytes and a
  `run_<region>.cfg`; `compare.py <region>` checks. Results (random bf16 x):

  | region | shape | PASS | steady-state (bf16 kernel → mmul kernel, item 5) |
  |---|---|---|---|
  | qkv | 8192×2048 (10.5 MB) | cos 1.0, maxrel 1.6e-5 | 1.09 → **0.50–0.55 ms** (DMA-only floor 0.45) |
  | share_down | 2048×512 | cos 1.0, maxrel 9.8e-6 | 0.24 → 0.17 ms |
  | share_up | 512×2048 | cos 1.0, maxrel 1.2e-5 | 0.24 → 0.16 ms |

  **Phase 2 item 5 (2026-09-02): the inner product moved to the integer
  matrix unit** (`gemv_q4.h`, plan `open-kernels-phase2-item5-bandwidth.md`).
  vegah's bf16 form was compute-bound at 1.4 GB/s per core — its nibble → bf16
  conversion is 4 accumulator ops per 32 lanes and spilled — while the DMA
  stream feeds ~4 GB/s per core (`GEMV_NULL=1` builds the DMA-only probe).
  Now `mmul<4,8,8,int16,uint8>` consumes each 64 B nibble block as it lies in
  the pool (B = [8 k][8 row pairs]), low and high nibbles masked separately
  (the odd rows' factor 16 folds into d), against the activation
  block-quantised to int16 once per x per core (`gemv_q4_prep_k{K}`: per
  32-wide block, power-of-two scale, exact for bf16 values within 2^7 of the
  block max). Entry points take the table instead of x: `(chunks, tab, y)`,
  `Buffer` of 2.25 K bytes per core, `gemv_q4_prep_k{K}(x, tab)` after each
  x acquire. K is a runtime argument of the one shared tile body (per-K
  template copies overflowed the 16 KB program memory in moe_experts).
  `GEMV_TABDUMP` writes table windows into y for debugging (a `static`
  counter in core code is NOT zero-initialised — .bss is not cleared).

- `designs/lm_head_q8/` — **phase 1**: q8 lm_head GEMV from the captured
  lm_head pool (`C:/caps/m0d/000127.bo`, its own 128-row supertile order:
  32-chunk bands, quarter = c%4, k-tile = c/4). One entry point with a runtime
  `group` argument; 1940 bands split 243/242 over 8 cores with hand-built taps.
  `make_test.py [--bands B]` + `compare.py <tag>`. Full 248320 logits: **PASS
  cos 1.0, maxrel 2.9e-6, 21.4 ms** (540 MB, ~25 GB/s; FLM's closed lm_head:
  15.4 ms). 80-band subset: 0.67 ms at 33 GB/s.

- `designs/deltanet/` — **phase 1**: gated DeltaNet decode step, 32 v-heads,
  S[32,128,128] fp32 in/out + per-head (k, q, v, decay, beta) in, o[32,128]
  out. S does not fit L1, so it streams through each core (one per column,
  4 heads each) in 16-row slices, twice per head: pass 1 forms S^T k, pass 2
  writes S' = decay·S + k⊗delta and forms o = S'^T q/√128. Every fp32 product
  is a bf16 hi/lo split (AIE2P has no fp32 vector multiply). `make_test.py`
  takes S from a real captured boundary state (`C:/caps/pf_t11_full`) and
  random k/q/v/decay/beta; `compare.py` checks S_out and o. **PASS: S_out
  maxrel 6.2e-6, o maxrel 6.3e-6, 0.44 ms** (2 MB state read twice + written
  once).

  Two hardware facts learned here, both silent hangs (ERT state 8) otherwise:
  - **A shim DMA channel's start queue holds 4 BDs.** Queue more fills on one
    channel without awaiting and the extra ones are dropped; the core waits
    forever. Fix: `fill(..., wait=True, group=tg)` per head and `tg.finish()`
    before issuing more than 2 heads (4 BDs) ahead — the sequence in
    `dn_step.py`. Drains are issued first so cores never block on output.
  - **A column shim has 16 BDs total** and IRON packs 4 workers per column by
    default; designs with several fills per core need `Worker(..., tile=Tile(c, 2))`
    to spread one core per column (the verifier catches this one).

- `designs/dn_glue/` — **phase 1**: the linear-attention layer glue around the
  DeltaNet step, one core: alpha/beta projections (2048×32 bf16) → decay =
  exp(A·softplus(·+dt_bias)), beta = sigmoid; depthwise conv1d k=4 over
  [state rows, qkv] + SiLU; per-head L2 norm of q/k; emits the fp32[512]
  per-head records `designs/deltanet` consumes and the shifted conv state.
  Inputs from the captured L0 side pool (`C:/caps/m0d/000119.bo`, repacked
  into our 4 KB-element side blob by `make_test.py`) and the captured decode
  conv state (`C:/caps/m0c/000898.bo`); xn/qkv random. **PASS: state
  bit-exact, k/q/v maxrel ≤ 1.1e-5, decay/beta ≤ 1.4e-7, 0.71 ms.** Uses
  `ironutil.Pipeline` (throttled fills/drains) and fp32 vector exp/reciprocal
  helpers in `dn_glue.h` (`vexp32`, `vrecip32`, `vsigmoid32`).

  Traps met here (each cost a build-run cycle; all silent):
  - **`release(n)` frees the n OLDEST acquired elements.** You cannot hold one
    element (x) across later acquire/release pairs on the same fifo; copy it
    to a `Buffer` first (`glue_copy.cc`). Symptom: garbage that depends on x.
  - **Python-unrolled loops overflow the 16 KB program memory** —
    `XAie_LoadElf failed with XAIE_INVALID_ELF`. Use `range_` and pass indices
    as runtime ints.
  - **The objectfifo lowering allocates depth+1 buffers** when a side acquires
    `depth` at once; the aie-opt `MemoryMap` in the error is the truth.
  - **`aie::tanh` / `aie::invsqrt` / `aie::inv` are LUT approximations
    (~1e-2).** A sigmoid via `(tanh+1)/2` gave 2.6 % error on normalised q/k.
    Build fp32 transcendentals from bf16 MACs: `vexp32` (bit trick + degree-6
    poly, 1e-7) and Newton-refine the hardware `inv`/`invsqrt` seeds.
  - **Stack overflow hangs the core** (state 8): the fp32 exp path spills
    many 128-B vectors; `stack_size=0xD00` was too small, `0x1800` works. The
    stack sits at 0x0 with the fifo buffers right after it.
  - **The build caches too much.** `iron.jit`'s design cache is keyed on the
    Python source + CompileTime args, and aiecc keeps kernel sources/objects
    in `final.prj` and skips recompiling them — header edits produced
    bit-identical results three builds in a row. `build_design.py` now wipes
    `final.prj` before every build.

- `designs/ln/` — layer RMSNorm with fused residual add: y = x + add,
  xn = bf16(rms(y)·w). PASS, xn bit-identical to the fp64→bf16 reference.
- `designs/dn_post/` — DeltaNet post step: og = bf16(rms128(o)·ssm_norm ·
  silu(z)). PASS (4 of 4096 one-ulp bf16 differences).
- `include/vecmath.h` — the fp32-on-bf16-MAC toolkit every kernel above uses:
  `splitN`, `fmulN/faddN/fsubN`, `vexpN` (1e-7), `vrecipN`/`srsqrt`
  (Newton-refined hardware seeds), `vsigmoidN`/`vsiluN`.
- `designs/layer_chain/` — **MILESTONE (2026-09-02): a whole linear-attention
  layer runs on open kernels and matches the CPU replica.** Layer 0 of the
  captured 3LiF decode block (token 248068 at position 11, states from
  `C:/caps/m0c/000898.bo`, weights from the captured L0 pool/pack/side) as a
  host-driven chain of seven dispatches: ln → gemv(qkv) → gemv(z) → glue →
  dn_step → post → gemv(out) → ln(+residual, post-attn norm). Reference:
  `tools/kernel-interp/decode_step.py linear_decode` in fp64.
  `make_chain.py` (WSL, needs the 3LiF model) writes buffers + `run.cfg`;
  `compare_chain.py` checks:

  | output | cos | maxrel | note |
  |---|---|---|---|
  | xn (normed input) | 0.9999986 | 2.4e-3 | bf16 |
  | residual after attention | 0.9999996 | 9.4e-4 | fp32; error = bf16 xn/og rounding, as FLM |
  | MoE input xm | 0.9999975 | 2.7e-3 | bf16 |
  | DeltaNet state S | 1.0000000 | 2.8e-4 | fp32 |
  | conv state | 0.9999996 | 1.9e-3 | bf16 |

  ~15 ms for the chain as run (7 xclbin contexts, cold; the fused single
  dispatch is phase 2). Multiple `xclbin`/`kernelx` contexts in one driver
  config work; the glue's xn slot is filled by `dump xn` + `load side` (host
  round trip, test only).

- `designs/router/` — MoE router: bf16 GEMV 2048×256, fp32 softmax, top-8 on
  the core (positive floats compare as uint32 on the scalar unit), renormalised
  weights. On the layer chain's real MoE input: same 8 experts, same order as
  the fp64 reference; p maxrel 1.6e-5, w 2.9e-6.
- `designs/silu_mul/` — h = bf16(silu(g)·u): bit-exact.
- `designs/moe_combine/` — two designs: `moe_axpy` (acc += w[e]·y_e, one run
  per expert, slot from a small buffer, accumulator ping-pongs between two BOs)
  and `moe_fin` (out = xres + acc + sigmoid(xm·sgw)·shared). maxrel 6.9e-6.
  **A run with 14 buffer arguments is rejected by the firmware (ERT state 6,
  abort, and the driver stops)** — hence two designs; keep runs ≤ ~8 buffers.
- `designs/moe_chain/` — **the MoE block of layer 0 on open kernels, 48
  dispatches over 9 xclbin contexts**: ln → router → per expert (gemv up, gemv
  gate, silu_mul, gemv down, moe_axpy) → shared expert (gemv up/gate, silu_mul,
  gemv down) → moe_fin. Reference: fp64 from the replica's dequantised expert
  weights with bf16 xm and bf16 h exactly as the kernels round. **PASS: router
  idx identical, block output cos 1.0000000 maxrel 2.8e-5** (1.4e-4 vs the
  replica's fp32-activation version). Expert weights are sliced by the host
  from the reference's indices; the on-device fetch (phase 0b) replaces that
  slicing in the fused kernel.

Together with `layer_chain`, **all of layer 0 (attention + MoE) of the
captured decode step now reproduces on open kernels.**

- `designs/attn/` — full-attention decode step, one core: head RMSNorm ×
  effective q/k norm weights, partial RoPE (rotary 64 of 256, half-split,
  cos/sin for the position supplied by the host in a 2 KB meta record),
  online-softmax attention over the cached K/V rows (streamed 1 KB per row,
  two fills per position) plus the new position, sigmoid gate; emits the new
  bf16 cache rows. The meta record is two 1 KB elements, `[qn | kn]` and the
  position record `[pos | nf | cos | sin]` (`layer_x/layout.py ptab()` builds
  the table of them); the core loops over the record's `nf` rows and masks
  rows `t >= pos`. In this standalone design the row fills are still static
  (`ATTN_POS`); `layer_x/ax.py` streams the window as one driver-patched fill.
- `designs/attn_chain/` — **layer 2 (full attention) of the captured decode
  step on open kernels**: ln → gemv q/gate/k/v → attn → gemv o → ln. PASS:
  residual cos 0.9999999 (maxrel 4.0e-5), new cache rows and gated output
  within bf16 rounding, 0.95 ms for the attention kernel at 11 positions.

  Two more traps: **a run with 9 buffer arguments aborts (ERT state 6); 6
  works** — so the GEMV gained `GEMV_YOFF`/`GEMV_YTOT` (write y at an offset
  of a shared output BO: `build_z_hi` puts gate after q, `build_512_hi` puts v
  after k). And **shared kernel bodies must be `inline` (vague linkage) +
  `noinline`**: `static` copies per translation unit overflowed the 16 KB
  program memory (`_XAie_LoadProgMemSection: Overflow of program memory`).

**All three layer types (linear attention, MoE, full attention) of the decode
step now run on open kernels and match the CPU replica.**

- `designs/decode_chain/` — **the whole decode step as one driver config**:
  L0 linear+MoE, L1 linear+MoE, L2 attention+MoE, final norm, lm_head — 164
  dispatches over 19 xclbin contexts, ~all in ~0.3 s of kernel time
  (lm_head 38 ms cold). `make_decode.py` slices every weight from the captured
  pools/packs/sides, predicts each layer's routing with the mirrored math (and
  adopts the NPU's own selection from a previous run if it differs);
  `compare_decode.py` compares logits with FLM's capture (odd vocab rows) and
  the replica. Result (2026-09-02):

  | | corr | top token |
  |---|---|---|
  | open kernels vs CPU replica | **1.00000** (residuals 0.999999 per layer) | same |
  | open kernels vs FLM capture | 0.671 | differs |
  | CPU replica vs FLM capture | 0.671 | differs |

  So the open kernels reproduce the replica's math exactly, and inherit its
  **pre-existing divergence from FLM** (the repo already deposed the CPU model
  as oracle for this reason: 0.57–0.68 vs both NPU paths, which agree at
  0.9996). The divergence is a semantics difference in some op, not
  accumulation; it must be found before the open engine is "correct". The
  decode block in `C:/caps/m0c` is FLM's older many-ops-per-layer flow with
  every op's buffers captured, so it can be bisected op by op against this
  modular chain. (Process exit after 19 contexts segfaults in XRT teardown,
  after all work and dumps are done — harmless, noted.)

  **Diagnosed (2026-09-02): the "CPU-model divergence" is FLM skipping the
  full-attention block.** Bisecting the m0c capture with the replica: the
  normalized layer inputs match at layers 0→1 and 1→2 (0.996–0.9998 per
  token), FLM's own captured q/gate/k/v projections and its CPU-built KV cache
  match ours (0.9994–0.9999, which also pins RoPE to half-split, rotary 64,
  θ=1e7 and the planar q|gate layout), yet layer 2's captured expert inputs
  match our MoE input only at 0.53 — and at **0.995 when the attention
  contribution is set to zero**. Same for the whole step: replica decode
  logits vs FLM's captured logits go from 0.671 to **0.998 with the same top
  token** when layer-2 attention is skipped; the prefill's final hidden from
  0.72 to 0.995. FLM's captured execution of this 3-layer `[L,L,F]`
  (`full_attention_interval=3`) test model contributes nothing from the
  attention block — consistent with the repo's earlier note that FLM
  mis-executes interval-3 models (Josh's pruned 27B is interval-3). So:
  the CPU replica is the faithful (HF) math, the open kernels match it to
  corr 1.00000, and FLM's captures are the wrong oracle for interval-3 models.
  **Control (done): on the base 40-layer interval-4 model the same replica's
  prefill logits match FLM's capture (`C:/caps/pf_t11_full/008566.bo`) at
  corr 0.955 with the same top token (9419)** � FLM does compute attention
  there. The skip is specific to the interval-3 configuration.

- **Josh's pruned Qwen3.6-27B-A2.8B (30 layers, interval 3) on open kernels
  (2026-09-02):** `make_27b.py` builds every layer's pool/pack/side with
  `build_pools.py` from `~/.flm/models/Qwen3.6-27B-A2.8B-open/model.q4nx`,
  slices the kernels' inputs, and runs one decode step at position 0 (zero
  states, empty cache) through all 30 layers + final norm + lm_head as one
  config: **1622 dispatches, logits corr 0.999998 vs the CPU replica, same
  argmax (846) and top-5, every layer's residual ≥ 0.999998** (`compare_27b.py`).
  ~4 s of NPU time host-driven. This is the model FLM mis-executes; the open
  kernels run it correctly.

Phase-1 status and what's next: `.claude/plans/open-kernels-feasibility.md`,
"Phase 1 progress".

## Phase 2 (fast): where the 27B step's 1.24 s went, and the fused MoE

`decode_chain/floor.cfg` (2026-09-02) measured the per-dispatch cost in
isolation, with the driver's new `runlist` + `runx` (a generic `run` queued on
the open runlist) and `copy` (BO→BO through the host) directives:

| pattern | per dispatch |
|---|---|
| gexp (655 KB expert stripe set) alone, same context back to back | 0.23–0.25 ms |
| silu_mul (512 elements) alone | 0.13 ms — the submit/wait floor |
| gexp with the previous dispatch in another xclbin context | **0.65–0.79 ms** |
| silu_mul after a context switch | 0.29–0.37 ms |
| 16 gexp in one runlist | 2.5–2.8 ms (0.16 ms each) |
| 8 silu_mul in one runlist | 0.5–0.6 ms (0.07 ms each) |

So the decode chain's 0.4–0.6 ms average per dispatch is mostly **context
switching** (~0.4–0.5 ms every time consecutive dispatches use different
xclbins, which in the chain is nearly always); runlists halve the floor but
only within one context. Fusion into one xclbin removes both. The 27B step's
1622 dispatches were 61 % routed experts (5 dispatches over 4 contexts per
expert, 1350 dispatches in all), hence MoE first.

- `designs/moe_experts/` — **the 8 routed experts of a MoE block as ONE
  dispatch** (2026-09-02): cores 0–3 stream up band c and gate band c of each
  expert (the gemv_q4 RS=4 entry points unchanged) against xm and emit
  h_c = bf16(silu(g)·u); the four h parts are **joined on a memtile**
  (`of_h.prod().join(...)`) into h[512] and **broadcast to all 8 cores**, which
  each stream two down bands and keep `acc += w[e]·y` for their 256 rows in
  the output element across the 8 experts; one drain at the end. Weights are
  the host-sliced experts concatenated per expert `[up | gate | down]`
  (15.7 MB); the first element of every core's weight stream is the header
  `[xm bf16 | router output f32[1024]]`, because **a core has only 2 input DMA
  channels** (the build error names it: "requires 4 input/2 output DMA
  channels, but only 2 input/2 output available") — w and h take both.
  `make_test.py` (Windows, from `moe_chain`'s layer-0 vectors) + `compare.py`:
  **PASS cos 1.0000000, maxrel 8.4e-5, 2.24–2.4 ms warm** (3.7–3.9 ms cold)
  for all 8 experts, vs ~25 ms as 40 dispatches. 15.7 MB / 2.3 ms ≈ 7 GB/s;
  the per-expert bubble is the h join/broadcast round trip and the 4 idle
  cores during up/gate — item 5 territory.
  Trap: one `extern "C"` entry point per `.cc` (IRON compiles each
  ExternalFunction's source file separately; two entries in one file link as
  duplicate symbols).

  **In the 27B decode step** (`make_27b.py`: `copy hdr ← xm, rout; run me
  wexp hdr acc` per layer, then the shared expert + `fin` as before):
  **1622 → 452 dispatches, 1239 → 460 ms (2.2 tok/s), logits corr 0.999998,
  same argmax (846) and top-5, all 30 residuals ≥ 0.999997**
  (`compare_27b.py`). The fused MoE is 92 ms of the 460 (3.1 ms/layer
  including the context switch); the rest is the unfused linear-attention
  chain (~190 ms over 10 dispatches/layer), the shared expert (5 dispatches,
  ~100 ms), lm_head 23 ms.

  **Step 1b — the shared expert and the combine in the same dispatch.** The
  shared expert is streamed as a 9th expert: its `[share_up | share_gate |
  share_down]` are the same 3 × 655,360 B, so every core's DMA pattern is
  identical to a routed expert's and only the band law differs (RS=2, 64-row
  bands: `gemv_q4_r2x/r2h.cc` wrap `gemv_q4_pool_group` with a runtime group
  and output offset). The header grows to `[xm | rout | sgw | xres]` (exactly
  20480 B); `moe_hdr` computes the gate `sigmoid(xm·sgw)` on every core and
  keeps its 256 rows of xres; `moe_fin` emits `xres + acc + gate·shared`, so
  the kernel's output is the layer's residual. Unit test vs `moe_chain`'s
  block reference: **PASS cos 1.0000000, maxrel 2.75e-5, 2.49 ms** (the
  45-dispatch chain: ~30 ms). **27B decode step: 302 dispatches, 348 ms
  (2.9 tok/s), logits corr 0.999998, same argmax/top-5**
  (`run_27b_fused.log`). Per layer the MoE block is now `rt` + 4 host copies
  + `me` (3.2 ms); the linear-attention chain is the remaining ~190 ms.

  **Step 2a — experts straight from the resident layer pool, no host slice.**
  The kernel's instruction stream has one DDR-patch op per weight fill (144
  on arg 0: opcode 0x81, 12 words, register at +6, arg index at +8, byte
  offset at +10). Built for the host-concatenated `wexp`, each fill's static
  offset names its (expert slot, core, up/gate/down); the driver's new
  `moeroute <kernel> <rout-buf>` reads the router's 8 indices (int32 at byte
  1024 of its output) and rewrites those 144 offsets as offsets into the
  512 MB layer pool (`pools.rs` layout: stripes `(8e + 2c [+1])·163840`,
  down `335544320 + e·655360 + c·81920`, shared fixed), then syncs the
  instruction BO. **0.04 ms per layer.** `make_27b.py` now binds
  `pool_L{l}.bin` from `l30-build`'s output (verified byte-identical to
  `build_pools.py`'s) and no longer writes `wexp{l}.bin`: **the 27B step
  runs with all 30 pools resident (15 GB of BOs, 20 s to load), 302
  dispatches, logits corr 0.999998, same argmax/top-5**; 350–400 ms depending
  on the box's other load (two llama-servers were running; the slowdown was
  uniform across every kernel). The on-device 0b mechanism stays unused: the
  host patch costs 1.2 ms/token, which is the whole difference.

- `designs/lin_layer/` — **phase 2 item 1, design A: the linear-attention
  layer as three dispatches instead of ten** (2026-09-02). The DeltaNet step
  alone uses the whole shim budget (16 fills + 16 drains), so it stays its own
  context; everything around it regroups by shim column into two multi-worker
  xclbins whose stages hand over through a DDR scratch BO (`act`), each
  dependent fill issued after `dma_wait` on the drain that wrote it:
  - `lin_a` = ln (no residual, `ln_nr.cc`) → qkv | z GEMV (8 cores stream
    qkv's 16 bands then z's 8 per core, x broadcast once) → glue. Args
    `pool xres consts state act vec`; qkv/z are read at their **layer-pool
    offsets**, the conv state is updated **in place**, the glue's xn element
    comes from `act` (no more `dump xn` / `load side`). 10 cores, 12 fills,
    10 drains: `Tile(c, 0)` pins on the runtime handles put ln in shim column
    0, x in 1, the glue's side/out in 2 and act in 3, next to the GEMV streams.
  - `lin_c` = post → out GEMV → ln (+residual). Args `wout o consts act xres
    hdr`; it writes the MoE header record directly (xm at 0, the new residual
    at 12288; sgw is preloaded, the router's slot is one host `copy`).
  - `layout.py` holds the byte layouts (per-layer `consts` = [lnw | glue side
    | nw | postln], `act`, `hdr`); `make_test.py` (Windows) chains `la → dn →
    lc` on layer_chain's vectors with the captured pool as the weight BO;
    `compare.py` reuses layer_chain's references and tolerances.

  **PASS with the unfused chain's numbers** (xn 0.9999986, residual
  0.9999996, xm 0.9999975, S 1.0000000, conv state 0.9999996; qkv bit-exact
  vs the unfused GEMV). Warm: la 2.4–2.9 ms, dn 1.5–1.7, lc 1.8 ≈ **6.2 ms
  per layer** vs ~8.5 ms + a host round trip as 10 dispatches (same log
  conditions: run_27b_fused.log). **27B decode step: 302 → 202 dispatches,
  13 contexts, logits corr 0.999998, same argmax (846) / top-5, all residuals
  ≥ 0.999997** (`run_27b_lin2.log`: 378 ms, but the box was ~10–15 % more
  loaded than for the 348 ms run — `me` 3.6 vs 3.3 ms, `lm` 23 vs 20 — so the
  like-for-like saving is the ~2.3 ms × 20 ≈ 45–65 ms the per-kernel times
  show, not the totals). Remaining per linear layer: 3 context switches
  (~1.5 ms) — design B territory.

- `designs/attn_layer/` — **design A′: the full-attention layer as ONE
  dispatch** (2026-09-02): `attn_l` = ln → q | gate | k | v GEMVs → attn →
  o GEMV → ln (+residual), 10 cores in one xclbin (11 fills, 10 drains). One
  ln core runs both norms (`ln_nr` then `ln`); the 8 GEMV cores stream q (8
  bands) | gate (8) | k (1) | v (1) against xn and then o (4 bands of K=4096)
  against og — two elements of one bf16[4096] x fifo, xn's fill being 8 KB
  from `act[0]` with an unread tail — with both entry-point sets (`p4b16r2`,
  `p4b32r2`) on the core; the attention core is `attn.py`'s verbatim. Five
  weight fills and five y drains per column are throttled by
  `ironutil.Pipeline` (which gained per-endpoint `finish(*eps)` so a stage
  can wait for just the drains it needs). Args `pool xres consts kv act
  hdr`: q/k/v/gate/o at their pool offsets, `consts` = [lnw | postln |
  meta], the new cache rows land in `act` (the host still appends them to
  the cache: plan item 3), the MoE header is written directly. `pos` (cached
  rows) is still a CompileTime parameter: `build_pos11` for the unit test,
  `build_pos0` for the 27B step.
  `make_test.py` / `compare.py` on attn_chain's layer-2 vectors (position
  11, the captured layer-2 pool as the weight BO): **PASS first run — knew /
  vnew 1.0000000, og 0.9999996, residual 0.9999999 (maxrel 4.0e-5), xm
  0.9999984; 2.5 ms warm** vs ~8 ms as 8 dispatches over 6 contexts.

  **27B decode step with both fused layers: 1622 → 132 dispatches, 8
  contexts, 311–315 ms (3.2 tok/s) under the same shared load as the 378 ms
  run, logits corr 0.999998, same argmax (846) / top-5, all 30 residuals
  ≥ 0.999997** (`run_27b_attn2.log`). Per layer now: linear 2.9 (la) + 1.6
  (dn) + 1.9 (lc) + 0.75 (rt) + 3.6 (me) ≈ 10.8 ms; attention 3.0 (al) +
  0.75 + 3.6 ≈ 7.4 ms; lm_head 22 ms. What is left is the MoE dispatch
  (108 ms, 7 GB/s — item 5), the linear layer's three context switches
  (design B, ~30 ms), the router (22 ms: fold into `me` or `lc`), lm_head.
  Trap: two designs in one process each pinning `Tile(c, 0)` shim handles
  is fine; a run with 6 buffer args is fine (9 is not).

- `designs/layer_x/` — **the whole layer in ONE xclbin context** (2026-09-02,
  plan `open-kernels-phase2-whole-layer.md`): `lx` = ln → qkv|z → glue →
  DeltaNet → post → out → ln(+res) → router → MoE for a linear-attention
  layer, `ax` = ln → q|gate|k|v → attn → o → ln(+res) → router → MoE for a
  full-attention layer. Eight main cores (Tile(c, 2)) run every GEMV, the
  DeltaNet step (`dnx.h`: S in 20-row slices through the weight stream, heads
  padded to 140 rows in the state BO and updated in place, S' rows out through
  the 256 B result elements) and the MoE block in one core program over three
  streams (w 10 KB, x 4 KB broadcast, y 256 B); helpers: norm + router, post,
  glue / attention. Two instruction streams per layer on one xclbin (part 0
  through the router, `moeroute2`, part 1 = the MoE): **a context switch does
  NOT preserve the array state — the cores restart** — so a layer must be one
  context, and multi-part sequences only work back to back. Program memory
  (16 KB) forced: one runtime-parameterised GEMV entry per destination type
  (`gemv_q4_pool_group_rt`), one scratch buffer per kernel family (`ms`, `ds`),
  one 9-iteration expert loop, `-Os`, shared (`inline` noinline) transcendentals
  in `vecmath.h`, no scalar float ops anywhere (the soft-float library is 2.6 KB).
  Unit test (`make_test.py`, layer 0): PASS with layer_chain's numbers, S pad
  rows zero, routing = moe_chain's, block output cos 0.9999992; lx0 3.4 ms +
  lx1 0.9 ms warm. **27B decode step (`run_27b_x.cfg`, `make_27b.py
  --whole-layer`): 132 → 62 dispatches, 8 → 4 contexts, 208 → 165 ms
  (6.1 tok/s) on a busier box, logits corr 0.999998, same argmax (846) /
  top-5.** Per layer: linear lx0 4.35 + lx1 1.12, attention ax0 2.34 + ax1
  1.10, lm_head 18.9. Driver: `moeroute2 <kernel> <buf> <idx offset>` (pool-
  layout placeholders), `dump <buf> <file> [size [offset]]`, `NPU_KEEP_GOING=1`
  (continue after a timed-out run so dumps show how far the cores got).

- **Phase 2 item 3 — dynamic KV length (2026-09-03,** plan
  `open-kernels-phase2-dynamic-kv.md`). `ax` no longer takes the position at
  build time: the KV cache is one BO per attention layer of interleaved 2 KB
  rows `[K_t | V_t]` (`MAX_CTX` = 4096 rows, 8 MB), the window `[0, nf)` is
  ONE linear fill (the fifo delivers it as `2 nf` 1 KB elements), the new row
  ONE 2 KB drain to row `pos`, and the RoPE cos/sin come from a shared
  position record table `ptab` (1 KB per position: `[pos | nf | cos | sin]`,
  `nf = max(pos, 1)`; position 0 streams one dummy row that the core masks —
  a zero-length DMA is not an option). The stream is built for position 1
  and the driver's **`attnpos <kernel> <pos>`** rewrites three words per
  token (the window fill's BD length, the drain offset, the record offset:
  the same DDR-patch mechanism `moeroute2` uses, plus the length word of the
  BD blockwrite before the patch op) with one instruction-BO sync — the
  `ax0` stream is shared by all attention layers. Unit test
  (`layer_x/make_test_ax.py`, layer 2 at position 11 from attn_chain's
  inputs, the captured cache re-laid as rows): PASS with attn_layer's
  numbers (og cos 0.9999993, xres 0.9999998, new row 0.9999998), three
  replays bit-identical; ax0 1.7 ms warm. **Trap: the instruction-buffer
  runtime's firmware translates only the first 5 buffer args into the AIE
  address space; DDR patches on args 5+ carry `+0x80000000` in the offset
  word (mlir-aie `kDDRAIEAddrOffset`)** — a driver patch must keep that bit
  (dropping it read the record from the raw host address: wrong cos/sin, a
  hang on the next run). `attn.py` and `attn_layer` moved to the same record
  and still PASS at position 11. **27B, three greedy tokens in one config
  (`make_27b.py --whole-layer --tokens 3`, `compare_27b.py --tokens 3`,
  `run_27b_t3.log`): 186 runs, ~155 ms of NPU time per position (no change),
  argmax = the replica's at every position (846, 198, 3710), position 0 corr
  0.999998, every layer through 22 at 1.000000 at position 1.** The logits
  corr at position 1 is 0.9945: at layer 23 the attention output cancels the
  input (|x| 26 -> residual 2.5), so the delta's bf16-level rounding is 2.7 %
  of the residual and layers 25/26 amplify it — the cache path replayed in
  the replica on the NPU's inputs matches (delta cos 0.99999997); see the
  plan doc. `decode_step.moe_decode(..., top=)` takes a routing override for
  like-for-like MoE checks (8th-slot near-ties).

- **Phase 2 item 5 — GEMV bandwidth (2026-09-02).** With the mmul GEMV
  (above) in every design — `moe_experts`, `lin_a`, `lin_c`, `attn_l`
  rebuilt, unit tests PASS at the previous tolerances: me 2.24–2.5 →
  **1.36–1.57 ms**, la 2.4–2.9 → **1.9–2.0**, lc 1.8 → **1.4–1.5**, al 2.5 →
  **1.3–1.5** — the **27B decode step is 313 → 220 ms (4.5 tok/s) under the
  same load, logits corr 0.999998, same argmax (846) / top-5, all 30
  residuals ≥ 0.999998** (`run_27b_item5.log`). Per kernel: me 108 → 60 ms,
  la 58 → 39, lc 37 → 30, al 30 → 20, dn 32 → 29 (no GEMV), lm_head 22 (q8,
  untouched), router 19. Details and traps in the plan doc.

  **Item 5, second half (same day).** (a) `moe_experts` balanced: all 8
  cores do up/gate (64 rows each: the 64-row half c%2 of stripe c//2 is the
  chunk pairs {4kt + 2(c%2), +1}, a strided shim tap of 8 × 10240 B at
  stride 20480 expressed as three real dims [8, 4, 2560] — the BD's highest
  dim is a repeat count, its length covers only the lowest three, and the
  innermost wrap is < 4096 B — which the RS=2 band law consumes as a plain
  64-row band); odd cores hand their 64 h rows to the even neighbour through
  shared L1 (an AIE2 core reads its west neighbour's memory; the memtile
  join keeps 4 producers, it has 6 DMA inputs). The driver's `moeroute` keeps
  each fill's byte position inside its stripe and accepts 144 or 216 weight
  fills. Unit test PASS (maxrel 2.5e-5), **0.85 ms** for the block (was
  1.36–1.57; 2.24–2.5 originally): 17.7 MB at ~25 GB/s. (b) `lm_head_q8`
  on the same path (`mmul<4,8,8,int16,int8>`, an unzip at step 8 splits each
  128 B into the two 8-row B operands; rows come out in order, no
  permutation): **full 21.4 → 15.6 ms (34.7 GB/s), PASS maxrel 4.6e-6** —
  level with FLM's closed lm_head (15.4). The activation table code is now
  `gemv_tab.h`, shared by both kernels.
  **27B decode step: 313 → 208 ms (4.8 tok/s), same load, logits corr
  0.999998, same argmax / top-5** (`run_27b_item5c.log`): me 1.85 ms/layer
  (the unit test's 0.85 + the context switch and a cold pool per layer), la
  2.0, lc 1.4, dn 1.4, al 1.9, rt 0.6, lm 16.8. Of the 208 ms, ~60 are
  context switches (132 dispatches × ~0.45 ms) and ~18 the router: the
  router fold, design B and a whole-layer context are now worth more than
  anything left in item 5.
