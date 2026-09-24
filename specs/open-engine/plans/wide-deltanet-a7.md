# WideDeltaNet A7: synthetic recurrent chain — 2026-09-24

**A7 passes the repository's inherited synthetic acceptance gates** at both
H5120 and H2560. All48 records, conv state and persistent recurrent state were
compared after every token. A stricter head-local output diagnostic does not
pass for several nearly-zero first-token outputs; that limitation is detailed
below. This is not full-model support or whole-layer integration.

## Implementation and reference

`wide_deltanet/glue.py` consumes the separate AB result followed by conv weights
on one side stream, and qkv/old conv state on the second stream. It reuses the
unchanged `dn_glue/glue_conv.cc`, `glue_emit.cc`, and `dn_glue.h`, compiled with
`DNGLUE_NHEAD=48`. `load_ab.cc` copies48 decay/beta values from the AB output;
it performs no new nonlinear approximation. The ten conv tiles contain four
q/k tiles and six value tiles, emitting48 records. Key mapping remains the
existing derived `h / (value_heads/key_heads)` for all heads, including31/32/47.

`deltanet/dn_step.py` now takes `n_heads` as a compile-time argument, default32,
selected by `DN_HEADS` for the CLI build. It validates head/core divisibility
and activation-buffer bounds and includes source/header hashes in its cache
specialization. The existing per-head `dn_step.h`, `dn_pass1.cc`, `dn_pass2.cc`
and the whole-layer `dnx.h` are unchanged. Forty-eight heads use eight cores,
six heads per core, two passes over each128x128 state matrix. State fills are
paced using the existing two-head window.

Hardware program, per token:

```text
NPU AB -> byte-copy AB into conv side -> NPU conv/records -> NPU dn_step
       -> dump intermediate/final outputs for comparison
       -> byte-copy new conv state and new recurrent state into next inputs
```

No reference tensor appears in `chain.cfg`. The open XRT harness only loads
inputs, dispatches kernels and copies bytes between BOs. There is no CPU
compute fallback or closed kernel/model-library dependency. Synthetic tooling
avoids the captured buffers required by the old `deltanet/make_test.py`.

The independent oracle in `utilities/wide_deltanet_reference.py` uses float64
math over the identical bf16/f32 inputs; conv state rounds to bf16 and recurrent
state rounds to f32 between tokens, matching their physical formats. Unit tests
cover analytic grouping, state shift, recurrence orientation, persistent-state
values, and rejection of bad output shapes/NaNs/tail corruption.

## Hardware checks

Same Strix/XRT/IRON/Peano environment as the
[AB report](wide-deltanet-bringup.md). Use the
[new build skill](../../../.opencode/skill/open-wide-deltanet-chain/SKILL.md).

For each hidden width, two sequences of eight tokens:

- cold: zero conv and recurrent state;
- warm: independent nonzero conv and recurrent state.

Weights/inputs use deterministic seed20260924 and distinct per-head data.
Decay is close to one so persistent state cannot be ignored. Every AB result,
record, new conv state, new recurrent state and output is poisoned with NaNs
before dispatch and followed by a64-byte canary. Inputs persist through
device-produced states, not CPU-produced references. All canaries are intact.

| Check | H5120 | H2560 |
|---|---:|---:|
| Tokens compared | 16 | 16 |
| Successful NPU dispatches | 48 | 48 |
| Acceptance checks | 208/208 | 208/208 |
| Conv state | bit-exact | bit-exact |
| Worst AB projection max-relative | 2.412e-6 | 2.108e-6 |
| Worst q/k/v record max-relative | 1.746e-5 | 1.792e-5 |
| Worst persistent-state max-relative | 3.208e-5 | 2.714e-5 |
| Worst output max-relative | 2.675e-5 | 3.004e-5 |
| Minimum state cosine | 0.999999999913 | 0.999999999918 |
| Minimum output cosine | 0.999999999863 | 0.999999999862 |

Thresholds are the existing `dn_glue/compare.py` and `deltanet/compare.py`
definitions: max(abs(got-ref))/max(abs(ref)) over the **whole field/tensor**
<1e-4; cosine >0.99999 for glue fields and >0.9999999 for state/output; finite
values required. These thresholds and this normalization were not relaxed.
Record padding is zero, shared q/k records are bit-identical within each
derived group, and each record also matches its independently indexed reference.

### Additional head-local diagnostic limitation

The first comparator version also enforced the same numerical threshold after
normalizing independently within each head. It reported failures only for
`cold-0/output`: H5120 heads3–5 and15–17, H2560 heads24–29 and39–41. Their
near-orthogonal q/k makes the zero-state first output very small. For example,
H5120 head3 has q·k≈0.00128784. Localizing the reference to device records shows
both the inherited conv/normalization approximation and split-bf16 recurrence
contribute; this is not an uninitialized tail or lost recurrent state.

Worst **head-local** max-relative errors are1.029e-3 (H5120) and6.639e-4 (H2560).
Every whole-tensor gate already passed on those exact same outputs. The final
comparator uses the original repository normalization for A7 and retains the
stricter failures as `head_diagnostics_passed: false` with per-head metrics.
No weights, seed, kernel math, outputs, numerical tolerances or reference data
were altered to remove these failures. A claim of per-head relative accuracy
<1e-4 would be false. Keep this issue visible in subsequent model-level checks.

## Resources and artifacts

Actual compiler address maps and ELF sizes, not Python storage estimates:

| Design | Cores | Input/output DMA per core | Placed buffers+stack per core | ELF text |
|---|---:|---:|---:|---:|
| conv/records | 1 | 2 / 1 | 59776 B | 7944 B |
| recurrence48 | 8 | 2 / 2 | 43776 B | 4048 B |

Conv placement includes3 side,6 activation and4 output FIFO elements after
lowering, plus qk/v scratch, decay/beta and6144-byte stack. This fits the
61440-byte recipe budget. Recurrence stack is3328 bytes. The new chain uses
separate contexts, not the impossible fused three-input glue topology.

Observed dispatch wall times (not model throughput): AB0.570–1.310ms,
conv0.483–0.678ms, recurrence1.322–1.724ms, including context switching.
Two new wide kernels and a default32 regression kernel were built; no libraries
or model packages were built. Artifacts remain ignored under:

```text
open_kernels/designs/wide_deltanet/build_glue
open_kernels/designs/wide_deltanet/build_step
open_kernels/designs/wide_deltanet/build_step_legacy32
open_kernels/designs/wide_deltanet/build_chain_h5120
open_kernels/designs/wide_deltanet/build_chain_h2560
```

The chain directories retain binary fixtures, intermediate/final outputs,
`chain.cfg`, `run.log`, `compare.log`, `chain-fixture.json` (versions and kernel
hashes) and `chain-results.json` (all acceptance and head-local diagnostics).
Build directories contain compiler logs and `final.prj/input_with_addresses.mlir`.

Xclbin SHA256 for this run:

- glue: `5b4a2f2dd1801a17df9b5ec00b2eaa68caf6519c6617e324d43190214b60601e`
- step48: `cd3336541229891cff5ffd0e247ddd59fdee22a0fcd0cf04197aa7bba18d7177`

## TDD and regressions

Four worker/geometry tests failed before the implementation, then passed.
Three oracle tests failed before the reference module, then passed. Further
checks exercise invalid head counts/buffer bounds and ensure the actual
generated hardware program never loads references or resets state each token.

`ironvenv/bin/python -m pytest specs/open-engine/tests -q`:
**684 passed, 47 skipped**. `test_wide_deltanet_chain.py`: **16 passed**.

The default32-head `dn_step` was rebuilt without `DN_HEADS`, then ran four
consecutive NPU tokens with nonzero initial state: **8/8 state/output checks
passed**, maximum relative1.444e-5, minimum cosine0.999999999963. Reproduce with
`utilities/test-deltanet-step.py`; this generator needs no captured buffers.

## Next milestone

Resume Track B on `implement-qwen38-27b-support`: integrate the separate chain
into the 27B layer program; resolve main-core `acquire(3)` versus depth2 at
H5120 and implement segmented FFN8192/8192/1024. Then validate the remaining
primitive points, a real layer, the 8-layer slice and the full64-layer model.
Model conversion, q4nx registration, packaging and LLM tests remain pending.

The validated synthetic chain does not validate the fused `lx` topology,
dnx's padded state layout or a full model. Catalogue entries and whole-layer
guards remain conservative. Flash-specific architecture work has not started.
