---
name: open-dense-activation-stream
description: Build and validate layer_x Q4 projection probes with streamed wide xn/xm inputs, notably K5120 and K6144. Use for the dense main-core depth-two FIFO constraint and Track B bring-up; not segmented FFN or full-layer acceptance.
---

# Dense wide activation streaming

`xcommon.prep_stream` prepares one4096-byte bf16 input element and releases it
immediately. `prep_bands` and the FFN's xm preparation use it when more than two
elements are required. The prepared table owns the activation thereafter.
Do not increase the broadcast FIFO depth: it consumes scarce L1 on every core.
The one/two-element paths retain their original acquire/release lifetime.

## Build and compare

Use the repository `ironvenv` (mlir-aie1.4.2 and pinned Peano) and the open XRT
harness from `.opencode/skill/open-wide-deltanet-ab/SKILL.md`. From repository root:

```bash
for width in 5120 6144; do
  ironvenv/bin/python utilities/probe-qwen35-wide.py --scope projection \
    --projection-k "$width" --out open_kernels/designs/layer_x/build_streamed_xn
  probe_dir=open_kernels/designs/layer_x/build_streamed_xn/projection_k${width}
  ironvenv/bin/python utilities/test-dense-projection.py prepare --build-dir "$probe_dir"
  HARNESS_TIMEOUT_MS=30000 open_kernels/harness/out/run_kernel "$probe_dir/projection.cfg"
  ironvenv/bin/python utilities/test-dense-projection.py compare --build-dir "$probe_dir"
done
ironvenv/bin/python -m pytest specs/open-engine/tests -q
```

The builder sets the XRT environment, writes an explicitly synthetic FFN8192
specification, enables diagnostic recipe composition, and regenerates the
production kernel TUs. Do not run two such generators concurrently with
different specs: their `.cc` output directory is shared. Kernel builds and
test data remain under ignored build directories. Hardware dispatch needs
`/dev/accel/accel0` access; escalate the execution tool if the sandbox hides it.

`projection_probe.py` calls the actual `X.prep_bands` and GEMV entries on all
eight cores, with the actual `tab`, `ms`, `ds`, weight/input/output FIFOs and
stack. It keeps two64-row output bands per core (N1024) so band resets are also
checked. The table remains sized for the synthetic8192-wide FFN, making the
resource test representative of main-core scratch rather than a tiny GEMV.
The specialization hash includes source dependencies and `Common`/FFN geometry.

The numerical utility packs synthetic weights with the existing Q4 packer and
computes the reference from those exact pool bytes. It covers random, ones,
zero and impulse inputs at chunk boundaries and the last logical value. Unused
activation padding and output values contain NaNs; outputs have64-byte canaries.
Accept finite results, max-relative <1e-4 and cosine >0.9999999, as in the existing
Q4 comparator. Exact zero outputs are checked as equality. Never truncate output
arrays to mask a short transfer or change the reference scale convention.

## Boundaries and next step

The full synthetic layer probe now passes IR generation beyond the former
`acquire(3)`/depth2 failure. It still fails physical placement of fused glue:
three input DMA channels on a core with two. Reproduce using:

```bash
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope layer \
  --out open_kernels/designs/layer_x/build_streamed_xn
```

The A7 separate AB/conv/recurrence chain is the validated alternative; it still
needs layer-program integration. FFN17408 still requires segmented K8192/8192/1024.
These Q4-only projection probes do not validate that FFN, mixed Q8, normalization,
attention or a model, and do not themselves promote catalogue entries.
See `specs/open-engine/plans/dense-wide-input.md` for measured resource/test data.
