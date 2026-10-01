# Phase 7: speed pass, re-ranked by the one-context profile

2026-09-30. Follows `.claude/plans/handoff-2026-09-30.md` and PR #140 (one context per
resolution). Supersedes the Phase 7 list and its "expected speed" table in
`.claude/plans/image-diffusion-npu-only-plan.md`: its item 1, collapsing contexts, is
done by #140.

## Spec impact

**No new, changed or removed requirements.** Each step still has to pass:
- **OPEN-DIFFUSION-DETERMINISM.** Kernel-set changes go into `export_dit_kernels.py` and
  `klein_pipeline.py`, which the pyxrt runner and the engine share. The engine must keep
  matching `generate.py` byte for byte.
- **OPEN-DIFFUSION-QUALITY.** Any step that changes rounding reruns the 512² study gate:
  LPIPS near 0.107 mean, and the text still legible.
- **OPEN-DIFFUSION-PERF.** New numbers go in the table only when they were measured on a
  quiet machine.

## Profile (2026-09-30, **CPU under load: not a PERF number**)

The machine was not quiet: two unrelated Python jobs used ~20 cores. During the same
session the standalone dit_fa bench ran at 43.6 ms, against 30.5 ms quiet. So this
profile ranks the work but doesn't size it. With `--profile`, every op includes its
configure and ~1 ms of per-op wait. That adds up to 4.1 s at 512² and 12.6 s at 1024²,
against 3.7 s and 12.0 s unprofiled.

`open_diffusion_cli --profile` now prints a per-stream table too (`cli.cpp`, uncommitted).

| 1024², ms per image | total | calls | per call | TFLOPS |
|---|---:|---:|---:|---:|
| gemm `sgl_in` (4608×3072×27648) | 3367 | 80 | 42.1 | 18.6 |
| fa `attn_sgl` (24 × 4608² × 128) | 2959 | 80 | 37.0 | 7.1 |
| gemm `sgl_out` (4608×12288×3072) | 1805 | 80 | 22.6 | 15.4 |
| gemm double-block image (qkv, out, ffin, ffout) | 1147 | 80 | | |
| fa `attn_dbl` | 723 | 20 | 36.2 | 7.2 |
| ew `qk_*` (RMSNorm + RoPE) | 414 | 120 | | |
| ew `res_*` (residual + LN + modulate) | 386 | 160 | | |
| VAE (conv + conv1 + vew + fa + gemm) | 1150 | ~100 | | |
| text encoder (te_*, all sets) | ~370 | ~190 | | |

| share | GEMM | attention | ew | VAE | text encoder |
|---|---:|---:|---:|---:|---:|
| 1024² | 54% | 30% | 7% | 9% | 3% |
| 512² | 63% | 15% | 13% | 9% | 9% |

The configures, about 0.3 s per image at either size, are spread across all of these.

## Ranked steps

Ranked for 1024², the default size. Savings are rough estimates and need the quiet
re-baseline (step 0).

### 0. Quiet re-baseline: DONE 2026-09-30 (logs in `C:\dev\switch-work\quiet-0930`)
- **Engine, quiet:** 3.75-3.78 s at 512², 12.16-12.40 s at 1024². No faster than the
  light-load PERF table, so the table stands; the re-measure is recorded under
  OPEN-DIFFUSION-PERF. The quiet `--profile` matches the loaded one above within ~1%, so
  the ranking holds.
- **Where attention's ~5 ms per call goes** (1024², 36.7 ms in the engine vs 30.85 ms
  standalone):
  - Not the reconfiguration. `utilities/reconfig-probe/cold_probe.py` times ops in the
    installed ELF: `attn_sgl` takes 31.9 ms right after its configure and 31.6 ms
    again. The configure itself is 0.8 ms for fa and 0.4-0.65 ms for the other sets.
  - Not the layout, beyond ~1 ms. `switch_probe.py` puts the op on the pipeline's
    buffers at 32.6 ms.
  - **Not the data, mostly.** On synthetic N(0, σ) inputs, attention slows with the score
    spread (32.4 ms at σ = 1, 38.5 at σ = 4): the lazy rescale. But klein's real Q/K
    rarely trigger it: 0.07-0.96% of (tile, chunk) pairs at TAU 8
    (`utilities/dit-chain/attn_rescales.py`, 6 ops of a 1024² image). On the same op, real
    inputs ran 0.9-1.5 ms slower than N(0, σ) inputs of the same std.
  - **Still unexplained:** ~3 ms per call (~0.3 s per 1024² image). Suspects, in order:
    - `attn_sgl`'s layout: qkv_ld 33792 and the interleaved output, against
      `attn_dbl`'s 9216. In one loaded run, `attn_sgl` took 38 ms and `attn_dbl` 33 ms.
    - what the engine does differently from the probes (runlist submission, QoS
      priority 0x180).

    Time it on a quiet machine.
- **GEMM has no such gap:** `sgl_out` takes 22.1 ms in `switch_probe`, 21.6 ms in
  `cold_probe` and 22.3 ms in the engine.

### 0b. Lazy-rescale threshold TAU 8 → 32: DONE 2026-09-30, a small win
- **Standalone, 24 × 4608² × 128, 20 runs (median):**

  | qk-scale | TAU 8 | TAU 32 |
  |---:|---:|---:|
  | 1 | 30.8 ms | 31.3 ms |
  | 2 | 37.7 ms | 31.2 ms |
  | 3 | 38.2 ms | 31.8 ms |

- **Accuracy against fp64 is unchanged:** rel_fro 2.64e-2 at scale 1, 4.18e-2 in both
  builds at scale 3. `compare.py`'s 3e-2 gate fails at scale 3 for TAU 8 too; it was set
  for scale-1 data.
- **Why it's safe:** P ≤ 2^TAU. bf16 O and fp32 l have the range, and bfp16's shared
  exponent is relative within each block of 8.
- **But klein's real scores rarely rescale** (step 0), so the saving is at most ~0.5 ms
  per attention call, about 0.05 s per 1024² image. It is kept: it's free, and it
  protects prompts with sharper attention.
- **Done:**
  - `FA_TAU` (default 32) in `fa_dit.cc`, `DF_TAU` in `dit_fa.py`, and `tau` in
    `export_dit_kernels.py`'s fa stamp, so a change rebuilds the fa streams.
  - TAU = 32 in `fa_emul.py`.
  - Rebuilt and installed the set.
- **Checks:**
  - 11 open-diffusion spec tests pass, including engine = pyxrt byte for byte.
  - The 512² QUALITY gate: LPIPS 0.109 mean / 0.186 max (spec: 0.107 / 0.191). Text is
    legible; prompt 0's sign reads "OPEN", as it does at TAU 8 and in the bf16 reference.
  - Timing: 1024² 11.86-12.10 s against 12.16-12.40 s, but the load had risen to CPU
    18-34%. Not a PERF number.

### 1. Probe: what bfp16 A would buy dit_gemm: DONE 2026-09-30, NO-GO
`utilities/dit-gemm-bench/bfp16a/` (quiet machine, CPU 10-11%, turbo, median of 20):

| shape | A bf16 (today) | A bfp16 | saved |
|---|---:|---:|---:|
| `sgl_in` 4608×3072×27648 | 40.62 ms (19.3 TFLOPS) | 39.90 ms | 1.8% |
| `sgl_out` 4608×12288×3072 | 17.80 ms (19.5 TFLOPS) | 17.71 ms | 0.5% |

- **Neither A's bytes nor the in-core conversion limit dit_gemm.** In the compiled loop,
  the conversion (`vmul.f` + `vconv`) shares its bundles with the 4 MACs: 5 bundles per
  4 MACs, against 4 with bfp16 A.
- **The limit is the C tile's bf16 round trip.** Every 64 of K (8 loop iterations),
  each 2×2 block of C is loaded (`vlda.conv`) and stored (`vst.conv`) again. That
  prologue and epilogue take about as long as the MACs.
- **The lever would be a longer K walk per round trip.** K_T 128 needs ~84 KB of L1,
  against 64 KB (C 32 KB + A 2 × 8 KB + B 2 × 18 KB). Not pursued.
- **Trap for a bfp16-A kernel:** aie2p has two block-load fifo registers. Four block
  streams (A0, A1, B0, B1, as upstream `mm_bfp.cc` has) make Peano spill their state
  around every pop, 2.6× slower. It needs two streams, with the row pairs of A and the
  column pairs of B interleaved, and interleaving B means a new `pack_b` layout.
- **Step 3 is dropped.**

### 2. `qk` RMSNorm + RoPE as the QKV GEMM's epilogue (2-3 days; about −0.35 s at 1024², −0.18 s at 512²)
- A C tile is 128 tokens × exactly one head, so the RMSNorm is tile-local. RoPE is
  already generated in-core in dit_ew and moves across as is.
- It removes 120 ew ops and one set change per block.
- It changes rounding: the result goes straight from the fp32 accumulator instead of
  through a bf16 round trip. Rerun the QUALITY gate.
- A sure win whatever step 1 shows, so it comes right after the probe.

### 3. bfp16 A from the producers: DROPPED (step 1: 0.5-1.8%)

### 4. Attention kernel (open-ended)
- 3.7 s at 1024² at ~7 TFLOPS. Each +1 TFLOPS on dit_fa is about −0.4 s at 1024², but
  only −0.08 s at 512².
- Settle step 0's unexplained ~3 ms per call first. If `attn_sgl`'s layout is the
  cause, it's the cheapest fix here.
- **Before choosing a change:** run a new ablation (no softmax / no P·V / DMA only).
  Real klein Q/K behave like qk-scale ≈ 1 data: std 1.2-1.5 and rare rescales.

### 5. Shrink the per-set configure writes (2-3 days; about −0.3 s at either size)
- Worth 8% at 512², where the fixed costs weigh most.
- Diff each `cfg_<set>_a/_b` against the others and write only the registers that
  differ.
- The two empty-device reset variants have to stay (the `full-elf-context` skill).
  Rerun `contention_trial.ps1` after the change.

### 6. VAE fusions (1 week; about −0.2-0.3 s at 1024², −0.1 s at 512²)
GroupNorm stats in the conv epilogue; GroupNorm-apply + SiLU in the conv prologue; the
upsample folded into the conv read pattern. This is last because it has the smallest
share.

**Not planned:** `res_ln_mod` as the `sgl_out` epilogue. The LayerNorm spans all
3072 columns and a C tile holds 128, so it is not tile-local.

## Expected result (estimates; re-derive after steps 0 and 1)

| | now (PERF) | after 2, 5, 6 |
|---|---:|---:|
| 1024² | 12.0-12.4 s | ~11.2 s |
| 512² | 3.7-3.8 s | ~3.1 s |

## Order

0, 0b and 1 are done; 3 is dropped. **Parked 2026-09-30: the owner judged the speed
enough.** The remaining steps (2, 4, 5, 6, and step 0's unexplained ~3 ms per attention
call) are each worth 0.1-0.4 s at 1024². Timings need a quiet machine; the builds and correctness checks don't.
