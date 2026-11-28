# Plan: Gemma 4 decode in one hardware context, per-layer inputs on the NPU

**Status:** approved as item E of `.claude/plans/round1-perf-plan.md` (items 3 and 4), 2026-10-19.
Done 2026-10-20: both steps merged into spec.md (OPEN-DECODE-ONE-CONTEXT, OPEN-FAMILY-GEMMA4);
the per-layer input in the layer program is the default.
**Branch:** `perf/gemma4-one-context`, stacked on `feat/gemma4-text`.

**Spec impact:**
- **Modified:** `OPEN-DECODE-ONE-CONTEXT` gains the gemma4 family: the four layer kinds are four
  instruction streams over one image (`designs/dense/gx.py`), one context `dx`.
- **Modified:** `OPEN-FAMILY-GEMMA4`: the per-layer inputs move from "on the host" to the dx program
  (gate GEMV, GELU-tanh, projection GEMV, RMSNorm, residual and layer scalar on the NPU); the
  per-token model projection stays on the host, once per token, multithreaded.
- **Modified:** `OPEN-DECODE-PIPELINE`: Gemma 4 no longer turns submit-ahead off.
- No requirement removed.

## Why

Measured from the installed manifests: E2B and E4B change hardware context 16 times per decode token
(four layer kinds, each its own xclbin) and Gemma 4 12B 18 times. A switch costs about 1-1.5 ms and
forces a wait (cross-context submit-ahead hangs the array). On top of that the host runs the
per-layer inputs between every two layers with an `xres` read-back, which turns submit-ahead off for
the whole walk.

## Program memory (measured, `utilities/core_sizes.py` on the bring-up builds)

| cores | E2B (four kinds) | E4B (four kinds) |
|---|---|---|
| main (row 2) | 9.9-10.7 KB | 9.8-10.8 KB |
| norm helper (0, 3) | 7.4 KB | 7.4 KB |
| attention, own cache | 14.0-14.5 KB | 14.2-14.6 KB |
| attention, shared cache | 13.1-13.3 KB | 13.2-13.3 KB |

The attention cores are 1.8-3.3 KB from full, so no attention core can carry a second kind's program
(head dim, rotation and the shared row are compile-time in attn.h). The main cores and the norm helper
have 5.6 KB and 9 KB free.

## Design, step 1 (one context)

- **The main cores** run one program for all kinds with the counts that differ read from RTP words
  written by each stream (ux.py's technique): q|k|v bands, og elements and K, up/gate bands, h
  elements and K, and each K's group / chunk counts. The arithmetic per band is unchanged.
- **E2B's sliding kinds move from 4 main cores to 8.** Their kv projection is 4 bands, so cores 0-3
  take one k band each and cores 4-7 one v band each; every other stage splits evenly. Every output
  row is computed by the same GEMV from the same table, so this is still bit-exact.
- **Four attention core sets**, one per kind, each compiled exactly as today (`symbol_prefix` keeps
  the four objects apart), on separate tiles. A set is parked on its `ain` while other kinds run.
- **Shim budget:** 16 fills and 16 drains over the array. Four sets of four og fifos would need 16
  drains alone, so each set's og fifos are joined in a memory tile into one drain. Fills: 8 w, lni,
  x, 4 ain = 14. Drains: 8 y, lno, 2 aout, 4 og = 15.
- dx.py's attention cores, norm helper and host sequence move into `designs/dense/dxkit.py`, called by
  dx.py and gx.py alike; dx.py's MLIR must not change for any dense spec (`utilities/mlir_equiv.py`).
- **Recipe:** `gemma4.builds()` builds the four kinds as `gx.py` parts; `programs()` points all four
  kernels at context `dx` (the first part's xclbin). `OPEN_LAYER_ONE_CTX=0` at export rolls back to
  the four dx.py builds, and the flavour is in the build key. The `ln` / `lm` tail keeps its contexts.
- **Gate:** bit-exact logits (max |diff| 0.0) against the four-context set, same CLI, E2B and E4B,
  at a layer prefix crossing every kind and at full depth.

## Design, step 2 (per-layer inputs)

- Pool: `inp_gate` (256 x hidden, Q4_0) and `proj` (hidden x 256, Q4_0) appended per layer,
  `std_perm_gguf`, no re-quantization. Consts: `post_norm` and the layer scalar.
- The host computes `ple_in` (model projection, norm, token embedding) once per token before the walk
  and writes each layer's 256 values into that layer's `act`; nothing runs on the host between layers.
- After the FFN residual, the dx stream runs: gate GEMV (4 bands on cores 0-3), GELU-tanh times
  `ple_in` on the norm helper, projection GEMV (hidden rows over 8 cores), RMSNorm with `post_norm`,
  then `(y + r) * scalar` into `xres`, on the norm helper.
- Submit-ahead comes back on for Gemma 4.
- **Gate:** against the host route, argmax and correlation (bf16 against fp64, not bit-exact);
  `oflm-test --llm` 5 of 5 through `oflm serve`.

## Measurement

Decode ms/token at positions ~1 and ~512, before and after, `--bench-decode` under
`dibs run --measure` if the box is quiet, else alternated A/B runs.

## Tests

- `tests/test_gemma4.py`: the one-context emission (one context, four parts of gx.py, the rollback,
  the flavour in the build key), the RTP words each part writes, the q|k|v split of E2B's sliding
  kinds over 8 cores, and the step-2 pack ops and buffer offsets.
- Hardware claims are manual, with the procedure in spec.md.

## What block prefill (item H) needs from this

Nothing here may block it: the block route uses its own GEMM kernels and host stages, not the dx
image. H needs the per-layer inputs in its host stages (or batched GEMMs for gate / proj), the KV
sharing, and `ple_in` for every token of the block (the model projection is per token already).
