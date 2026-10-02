---
name: open-segmented-dense
description: Build and validate the layer_x segmented Q4 dense down projection and synthetic full FFN at hidden 5120 and intermediate 17408. Use for segment-major DMA, local partial sums, and the known full-FFN bf16 rounding accuracy gate.
---

# Segmented dense Q4 FFN

Use repository `ironvenv`, XRT and the open harness documented in
`.opencode/skill/open-wide-deltanet-ab/SKILL.md`. Build from repository root;
never run kernel generators concurrently because their `.cc` output is shared.
Wait for each build to finish before preparing fixtures.

```bash
base=open_kernels/designs/layer_x/build_segmented
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope down --ffn 17408 --out "$base"
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope down --final-only --ffn 17408 --out "$base"
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope ffn --ffn 17408 --out "$base"
for mode in down down_final ffn; do
  ironvenv/bin/python utilities/test-segmented-dense.py prepare --build-dir "$base/$mode"
  HARNESS_TIMEOUT_MS=30000 open_kernels/harness/out/run_kernel "$base/$mode/segmented.cfg"
  ironvenv/bin/python utilities/test-segmented-dense.py compare --build-dir "$base/$mode"
done
ironvenv/bin/python -m pytest specs/open-engine/tests -q
```

Hardware execution needs `/dev/accel/accel0`; use normal tool escalation if the
sandbox hides it. Run dispatches sequentially. Build/fixture/results directories
are ignored. `down` emits snapshots after K8192,16384,17408; `down_final` uses
the production emission order. `ffn` runs actual up/gate/SiLU/down, without norm,
attention or DeltaNet. It uses the real model geometry and synthetic Q4 weights.

The initial full FFN comparison failed at1.123084e-4 against the unchanged1e-4
threshold. This was subsequently resolved by precise vector activation math;
see `.opencode/skill/open-dense-ffn-precision/SKILL.md` for the13-input gate and
up/gate traces. Keep device-h-based diagnostics separate from end-to-end
acceptance. Initial evidence is in `specs/open-engine/plans/segmented-dense-ffn.md`.
Whole-layer/model correctness and catalogue promotion remain pending.

## Implementation constraints

- Segment only when the legacy full table cannot fit even with one weight chunk.
  Preserve the existing12288-wide legacy path and its one-chunk weight elements.
- Segment order is8192+8192+1024 across all output bands. DMA reads slices of
  the original standard Q4 pool, never newly packed segment matrices.
- Prepare each f32 h segment once into the8192-wide reusable table; existing
  `dense_prep_f32` rounds to bf16. Reuse existing Q4 GEMV into ms, then add the
  finished band into ds. ds's DeltaNet lifetime has ended before FFN.
- Reset every band on the first segment of every token. Keep partials local;
  snapshots are diagnostic outputs only, never CPU feedback.
- Drain before filling weights; pace weight DMA with Pipeline(3) and finish
  each segment's input transfers before proceeding. A physical queue holds4 BDs.
- Actual buffers+stack:59392 B/core. ELF text:6976 B diagnostic down,6064 B final
  down,11712 B precise full FFN. These figures do not establish whole-layer text size.
- Mixed Q8 is explicitly unimplemented for this capability. Recipe acceptance
  requires `OPEN_KERNELS_UNVALIDATED=1`; fused wide glue remains separately gated.

The builder supplies the diagnostic environment flags. There are no new model
packing steps and no model artifact produced by this skill. Before final model
packaging, keep q4nx-build synchronized with the eventual validated recipe.
