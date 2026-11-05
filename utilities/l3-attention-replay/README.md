# Native attention after L3 projection refinement

This private replay feeds both the original and two-pass corrected L3
projections into the **current production** `xlayer.attn_workers` and
`attn_kernels`. It does not copy or replace attention arithmetic. Each of the
two dispatches executes 240 causal positions, with its own NPU-generated
paired-BF16 KV cache starting at zero. No model weight pool is resident.
The corrected projections still include one explicit **host FP32 addition**
from the separately audited two-pass GEMV experiment.

## Reproduce

Use fresh output directories; the retained build/run below are immutable.
The prerequisite is the successful, separately audited
`utilities/l3-projection-replay/refine.py` experiment. Its hashes bind the
original native L3 captures, raw GEMV outputs, activation tables, and independent
FP64 references. The preparation rechecks the actual FP32 composition.

```bash
source utilities/env.sh
export OPENBLAS_NUM_THREADS=1
ironvenv/bin/python utilities/l3-attention-replay/build.py \
  --spec Models/qwen38-27b/production-post-fp32/kernels-v1/spec.json \
  --out Models/qwen38-27b/post-prefill-drift/l3-attention-build-v2
ironvenv/bin/python utilities/l3-attention-replay/replay.py prepare \
  --source Models/qwen38-27b/post-prefill-drift/l3-refinement-v1 \
  --build Models/qwen38-27b/post-prefill-drift/l3-attention-build-v2 \
  --out Models/qwen38-27b/post-prefill-drift/l3-attention-v1
ironvenv/bin/python utilities/l3-attention-replay/replay.py run \
  --out Models/qwen38-27b/post-prefill-drift/l3-attention-v1 \
  --harness Models/qwen38-27b/prefill-sequential-drift/harness-l0-v1/run_kernel
ironvenv/bin/python utilities/l3-attention-replay/replay.py audit \
  --out Models/qwen38-27b/post-prefill-drift/l3-attention-v1
```

## ABI and controls

- Geometry NH24/KVH4/HD256/NHL4, six attention cores, RB1/VEXP1.
- An input record is 61,440 bytes: production norm metadata, position record,
  FP32 Q, K, V, then gate. Projection bytes are never rounded on the host.
- Cache rows are 8192 bytes: `[K_hi, K_lo, V_hi, V_lo]`. Each position reads
  only the completed earlier rows. Position zero transfers one masked dummy
  row. DMA groups finish before the next position starts.
- Output rows contain 6144 FP32 gated attention values (24,576 bytes).
- The control must reproduce **all 240 original native outputs bytewise** and
  **all 240 original KV prefixes bytewise** before numerical interpretation.
- The cache is zeroed and output poisoned with NaNs before each trajectory.
  Non-finite or incomplete captures, extra files, changed inputs, changed
  binaries and stale reports are rejected.
- The primary reference includes independent unrounded layer-input RMSNorm and
  weight projections. A separate conditioned reference uses the candidate's
  actual projected inputs. Both use independent FP64 causal attention.
- All 5760 heads per comparison retain maxabs <0.02 and cosine >0.9999.
  P125/P205/P239 are reported separately, including regressions that still pass
  these bounds. Passing a tolerance is not synonymous with improving accuracy.

## Build provenance and resources

The builder pins current recipe/design sources, every compiler dependency,
linked ELF files, allocated MLIR, XCLBIN and instruction stream. IRON copies
translation units into `final.prj`; their bytes must match production, and
compiler dependencies must include the actual `attn.h` and `attn_pair.h`.
The historical `attention-kv-replay` design targets an earlier BF16 ABI and
must not be used as the current production control.

Validated build `l3-attention-build-v2`: maximum .text **13,760 / 16,384 B**;
L1 allocation end **59,920 / 65,536 B**; stack unchanged at **6144 B**.
Logical data buffers total **22,609,920 B**. These are isolated-attention
resources, not a full-ax integration budget. The first build's compiler phase
succeeded but its provenance checker rejected IRON's copied source paths;
that attempt was retained and not used for execution.

Full-layer propagation, model acceptance, serving and performance need their
own gates. This replay changes no production kernel, registry, model packing
or installed service.

## Verified result (2026-10-28)

Both 240-position NPU trajectories complete. All **240 outputs and 240 KV
prefixes** reproduce the original full-ax native control bytewise. Candidate
primary and conditioned comparisons each pass **5760/5760 heads**.

| Comparison | Maximum absolute error | Minimum cosine |
|---|---:|---:|
| Control / independent unrounded norm | 2.720021153e-4 | 0.999999784269 |
| Candidate / independent unrounded norm | 5.185089467e-5 | 0.999999994834 |
| Candidate / its actual projected inputs | 5.520919823e-5 | 0.999999998349 |

Selected primary errors (control → candidate):

- P125: 5.937908748e-5 → 1.106952289e-5.
- P205: 8.628521564e-5 → 1.553450229e-5.
- **P239: 2.560080227e-5 → 2.574159746e-5**, a small regression despite passing
  the unchanged bounds. Do not equate the earlier CPU-attention improvement
  with improvement through actual NPU arithmetic at every position.

The separate audit recomputes the controls and all numerical measurements,
checks all four captures, and binds source, compiler, preparation and harness
provenance. The build binds 814 compiler input files. TDD: 10 initial failures,
13 final new tests; focused suite 27 passed; full host suite **2026 passed,
47 skipped**. Logs and evidence are retained under
`Models/qwen38-27b/post-prefill-drift/l3-attention-*`.

Next validate output projection, FFN and complete layer residual with an exact
native control. Preserve the P239 cancellation case before any production
promotion. Global full-model numerical acceptance remains FAIL; no performance
acceptance is inferred from the isolated dispatch timings.
