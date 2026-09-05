# Running a stock Qwen3.6-MoE decode step on the open kernels

> The engine that keeps the weights resident and serves the app is
> `src/open_qwen36/` (built on the same packing and the same kernels). This
> directory is the batch harness path and the fp64 reference it is scored by.

This directory is the model-side half of the open path: it takes FLM's own
`.q4nx` weight file, packs each layer's weights into the byte order the open
kernels stream, writes a driver program for `../harness/run_kernel.exe`, and
computes an fp64 CPU reference to score the result against.

Nothing here is closed: the container is parsed by [q4nx.py](q4nx.py), the pool
layouts by [pools.py](pools.py), the math by [replica.py](replica.py). FLM's
engine is never loaded. (The weights are still FLM's file — the GGUF path that
replaces it is a separate piece of work; see the repo plan.)

## Use

```
python open_kernels/model/make_decode.py --layers 8 --tokens 1
open_kernels\harness\out\run_kernel.exe open_kernels\model\out\run_decode.cfg
python open_kernels/model/compare_decode.py --tokens 1
```

The program names the kernels at `../designs/layer_x/build_lx0` … `build_ax1`,
`../designs/ln/build` and `../designs/lm_head_q8/build_full`, which is where
`../export_qwen36_kernels.py` builds them (it also copies them to
`src/xclbins/<model>/open_kernels/` for the engine). The model directory comes
from `FLM_MODEL_DIR` (default `~/.flm/models/Qwen3.6-35B-A3B-NPU2`).

`make_decode.py` writes into `out/` (small per-layer blobs, the reference, the
`.cfg`) and `out/pools/` (the 512 MB-per-layer weight pools and the 542 MB
lm_head pool — budget 512 MB × layers of disk AND of RAM, since every pool is a
resident device buffer for the whole run). `--reuse-pools` keeps them across
runs; `--cfg-only` rewrites just the program.

Re-running `make_decode.py` after a run adopts the NPU router's expert choice
wherever it differed from the reference's own, and says so. That is deliberate:
the 8th of eight routed experts is often a near-tie, and a legitimate tie-break
difference would otherwise show up as a large numerical error. `--strict-routing`
turns the adoption off.

## What runs where

Per layer, two dispatches on one xclbin context:

| | linear-attention layer (`lx`) | full-attention layer (`ax`) |
|---|---|---|
| dispatch 0 | norm → qkv/z → glue → DeltaNet → post → out → norm(+residual) → router | norm → q/gate/k/v → attention → o → norm(+residual) → router |
| driver | `moeroute2` — rewrite the expert fills' DDR offsets to the router's top-8 | same |
| dispatch 1 | the MoE block (8 routed experts + shared expert + combine) | same |

Then, once per token, the final norm and the q8 lm_head. The attention layers
share one instruction stream; `attnpos` rewrites its three position-dependent
words per token (KV window length, new-row offset, RoPE record offset).

A layer must be ONE xclbin context: a context switch restarts the AIE cores, so
the array state a fused layer carries between its two dispatches would be lost.

## Provenance

`pools.py` and `replica.py` are vendored from phlegm's `tools/kernel-interp/`
(`build_pools.py`, `decode_step.py`, `full_forward.py`), where every layout law
and every math element was verified against buffers captured from FLM's own
engine. The port is checked byte-for-byte: our pools, packs, sides and lm_head
pool are identical to phlegm's for both layer types of this model.

The kernels themselves are `../designs/`, and their provenance is
`../PROVENANCE.md`.

## Results (2026-09-04, Strix, Windows + XRT)

Stock `Qwen3.6-35B-A3B-NPU2`, decode from position 0 with zeroed state, scored
against `replica.py` in fp64 on the same weights.

| run | logits corr | argmax | top-5 | layer residuals |
|---|---|---|---|---|
| 3 linear layers, 1 token | 0.999998 | same | identical | 1.000000 |
| 8 layers (attention at 3, 7), 1 token | 0.999998 | same | identical | 1.000000 |
| 8 layers, 3 tokens | 0.999998 / 0.999996 / 0.999991 | same at each position | identical | ≥ 0.999995 |
| **40 layers — the whole model** | **0.999997** | **same (846)** | **identical** | **≥ 0.99992** |

Expert routing happens on device. All 24 of 24 (token, layer) selections in the
3-token run matched the reference; 38 of 40 in the whole-model run, the two
exceptions being 8th-slot ties with probability gaps of 7.5e-5 and 8.4e-5.

Whole model, 82 dispatches, 21 GB of resident pools: **323 ms/token (3.1 tok/s)**
— `lx0` 5.2 ms × 30, `lx1` 2.0 × 30, `ax0` 3.8 × 10, `ax1` 1.7 × 10, lm_head
31 ms, and 4 ms total for the 41 instruction patches. The box was holding 21 GB
of 47.6; phlegm's 30-layer runs on the same hardware imply ~220–275 ms at this
layer count, so treat this as an upper bound rather than the kernels' number.
