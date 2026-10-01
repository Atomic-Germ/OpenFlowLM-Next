---
name: open-attention-boundary
description: Isolate real Qwen38 attention rounding, replay captured Q/K/V/gate and cache, and validate corrected Q4 projections with precise attention.
---

# Real attention boundary

Read `specs/open-engine/plans/qwen38-attention-boundary.md` for acceptance and
limitations. Keep the existing model weights and independent full references.
At cold0/layer3, xn is exact but the original gated attention has27 BF16 og
differences:21 local to attention and6 from projected inputs. The existing
precise attention closes the21 local differences; block-carry Q4 projections
close the six remaining differences on this frame.

Use `utilities/diagnose-wide-attention.py prepare --source <full replay>
--kernel <validated attention> --out <fresh directory>`, then run `replay.cfg`
and `compare --out <directory> --require-exact-og`. Default tag is
`cold-0-layer3`. The strict flag checks the conditional oracle using actual
device projections/cache; `device_vs_reference` separately reports model
reference differences. Do not confuse these two tests.

The utility validates fixture/kernel hashes, source metadata and capture
guards, then replays the same input twice. Its NPZ references use FP32
expansions of BF16 values: NumPy serializes custom BF16 directly as opaque V2.
The conditional reference never replaces the full-model reference tensors.

For the recorded baseline use:
- source `build_residual_precision/full` under `open_kernels/designs/wide_deltanet`;
- ordinary attention `open_kernels/designs/attn/build_wide_257`;
- precise attention `open_kernels/designs/wide_deltanet/build_model_precision/attention`.
The ordinary replay must match its source captures; the precise replay should
have0 local og differences but6 differences from the model on this frame.

Build the Q/K/V/gate projection sequentially with other layer_x builds:

```bash
base=open_kernels/designs/wide_deltanet/build_attention_boundary/carry
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope projection --ffn 17408 \
  --projection-k 5120 --projection-n 14336 --projection-correction \
  --product-correction --block-carry --out "$base"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-dense-projection.py prepare \
  --build-dir "$base/projection_k5120"
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel "$base/projection_k5120/projection.cfg"
ironvenv/bin/python utilities/test-dense-projection.py compare --build-dir "$base/projection_k5120"
```

The14 primitive inputs include five exact cancellation cases. Add
`--projection "$base/projection_k5120"` to the real replay's prepare command.
This optional path requires source xn to match the model oracle. It copies the
original contiguous Q,K,V,gate pool span without repacking, uses the same DMA
slices as the full-model schedule and recomputes the conditional oracle from
the newly captured projections. Both projected inputs and outputs are guarded,
repeat checked and reported separately from the model oracle.

For the full replay use `utilities/replay-wide-model.py` with:
- `--output-projection .../build_ffn_boundary/carry/projection_k6144`;
- `--ffn .../build_down_precision/carry/ffn`;
- `--ln .../build_residual_precision/ln_rne`;
- `--attention .../build_model_precision/attention`;
- `--attention-projection "$base/projection_k5120"`.

Use the original `build_full_model` source and a fresh output directory.
Run `decode.cfg`, `test-wide-full-model.py compare` and
`diagnose-wide-model-rounding.py`. Keep seeds, references and thresholds intact.
Run the open-engine CPU suite; any test calling the layer geometry helper must
restore its `OPEN_KERNELS_UNVALIDATED` environment change with monkeypatch.

Recorded full replay:3564 dispatches complete;18891/18896 slice and4121/4122
decode checks pass. Six numerical checks fail, including worse final residual
and norm errors than the preceding variant. Keep this variant experimental.
The first eight norm boundaries are exact; next is cold0/layer4 xn channel786,
one local norm rounding error (device0.059814453125, reference0.0595703125).
All384 residual sums remain exact. Tokens and reset replay still pass.
