---
name: open-dense-ffn-precision
description: Reproduce the strict synthetic H5120/FF17408 FFN accuracy gate with up/gate traces and precise vector SiLU. Use to debug bf16 rounding amplification, compare the real segmented FFN, or handle tiny-vector cosine correctly.
---

# Dense FFN precision gate

Use the repository ironvenv and the open XRT harness from
`.opencode/skill/open-wide-deltanet-ab/SKILL.md`. The two current programs pass
13 synthetic inputs each. This does not validate a complete model layer.

```bash
base=open_kernels/designs/layer_x/build_segmented_precise
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope ffn --trace --ffn 17408 --out "$base"
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope ffn --ffn 17408 --out "$base"
for mode in ffn_trace ffn; do
  ironvenv/bin/python utilities/test-segmented-dense.py prepare --build-dir "$base/$mode"
  HARNESS_TIMEOUT_MS=30000 open_kernels/harness/out/run_kernel "$base/$mode/segmented.cfg"
  ironvenv/bin/python utilities/test-segmented-dense.py compare --build-dir "$base/$mode"
done
ironvenv/bin/python -m pytest specs/open-engine/tests -q
```

Wait for each build before preparing fixtures; generators share their output
`.cc` directory. NPU execution needs `/dev/accel/accel0` access and must be
sequential. Use ordinary tool escalation if hidden by the sandbox. Artifacts
and traces remain in ignored build directories.

The builder records `buffer_args=3`: pool, activation, diagnostic trace (a dummy
unused BO without `--trace`). The fixture utility also understands the older
two-BO metadata. Do not hand-write harness arguments from an old fixture.

## What fixed the failure

- Original up/gate errors were about5e-6; the old SiLU/product added up to1e-5.
  Subsequent bf16 rounding amplified this to1.123084e-4 in a full FFN output.
- `vecmath_precise.h` uses three bf16 components and six products through the
  third significance level, with separately named polynomial/Newton helpers.
  Only segmented dense SiLU/product selects this path. Existing GEMV math,
  Q4 pool format and legacy activation TUs are unchanged.
- Distinct symbols avoid COMDAT mixing of precise and legacy helpers inside
  one core. The probe hashes include headers as well as generated sources.
- Trace output is a separate FIFO and DMA channel, copying ms up/gate before
  the original activation call. Trace and normal h/output are bit-identical
  for all13 inputs. Do not feed CPU references or traces into the device.
- `metric` computes cosine after scaling each nonzero vector by its maximum.
  An absolute denominator epsilon is wrong for tiny vectors. Zero vectors
  have explicit handling, and max-relative uses the real reference maximum.
  Thresholds remain <1e-4 and cosine >0.9999999; the historical failing output
  still fails under the corrected metric.

Final buffers+stack are59392 B/core without tracing and59904 B with it; text
11712/11872 B respectively. This does not prove whole-layer program memory.
The recipe remains behind `OPEN_KERNELS_UNVALIDATED` until whole-layer
integration and the remaining primitive gates are complete. Mixed Q8 is not
implemented for segmentation. Keep the A7 head-local precision caveat visible.

Full evidence and hashes: `specs/open-engine/plans/dense-ffn-precision.md`.
