---
name: open-qwen38-model
description: Download and pack the real Qwen3.8-27B model, restore its 48-head GGUF ordering and reproduce standalone 64-layer validation with the accepted open wide kernels.
---

# Real Qwen38 model bring-up

Read `specs/open-engine/plans/qwen38-full-model.md` for measured status. Packing
or a header preflight does not establish 64-layer hardware acceptance or runtime
support. Keep catalogue guards and the distinction between source model,
converted model and packed standalone probe.

Weights belong under the existing ignored `Models/` directory (case-sensitive).
`utilities/download-qwen38-27b.py` pins the official config/tokenizer revision
and ggml-org Q8_0 GGUF revision, verifies Git/LFS content hashes and records
`sources.json`. Large interrupted downloads retain a resumable `.partial` file.
The Q8 source is28.60 GB; canonical Q4NX tensor data totals19.15 GB. Conversion
uses a temporary spool plus final output on the same filesystem.

```bash
ironvenv/bin/python utilities/download-qwen38-27b.py
python utilities/q4nx-build/convert.py \
  -i Models/qwen38-27b/source/Qwen3.8-27B-Q8_0.gguf \
  -o Models/qwen38-27b/converted -t language \
  -s Models/qwen38-27b/source
```

Use an environment with q4nx-build's dependencies. This host also has converter
dependencies in `/tmp/oflm-review-deps` (`PYTHONPATH` with ironvenv Python).
Set `OMP_NUM_THREADS=4` for conversion on the shared host. The27B converter
streams tensors to a normal safetensors/Q4NX container; global embedding/head
are processed in512-row batches. It refuses an existing output container.
Only GGUF language conversion is implemented for this variant; HF-tensor and
vision conversion explicitly report not implemented.

The source GGUF stores V heads in tiled `[slot,key]` order. Restore grouped
`[key,slot]` order for16 key/48 value heads, including alpha/beta, A/dt, V's QKV
rows, gate, conv and output columns. Q/K occupy4096 rows; V occupies6144. Never
split QKV in half or hard-code a two-to-one ratio. Full attention's Q/gate rows
also need deinterleaving. `utilities/verify-qwen38-head-order.py` compares selected
original HF tensors via bounded HTTP ranges, independently of the packer.
The open recipe consumes48 native BF16 alpha/beta rows; no padded quantized AB
companion is needed. Layer projections use Q4_1, the full vocabulary head Q8.

Reuse accepted artifacts from `open-wide-decode`. No new xclbins are needed.
Build the harness as in that skill, then:

```bash
OPEN_KERNELS_UNVALIDATED=1 ironvenv/bin/python utilities/test-wide-full-model.py preflight
OPEN_KERNELS_UNVALIDATED=1 OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-wide-full-model.py prepare
HARNESS_TIMEOUT_MS=30000 /tmp/oflm-wide-decode-harness/run_kernel \
  open_kernels/designs/wide_deltanet/build_full_model/decode.cfg
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-wide-full-model.py compare
```

Preparation requires a fresh `--out` directory. It uses production packing,
streams one pool/constant BO pair across64 layers and keeps48 recurrent states
and16 KV caches separate. The reference evolves independently over three cold
autoregressive tokens. Device selection must feed the next device embedding;
no reference IDs or activations may be loaded. Reset replays all three steps.
Preserve inherited per-layer gates and full-vocabulary correlation/argmax checks;
report the first diverging tensor instead of relaxing bounds. State-head strict
primitive diagnostics remain distinct from inherited whole-state acceptance.

## Recorded result and diagnosis (2026-09-28)

All3564 dispatches complete and the device/reference sequence agrees:
248045 ->8678 ->198 ->2. All reset captures are byte-identical; minimum full-logit
correlation is0.9999865019. **The full-model gate still FAILS**:18894/18896 slice
and4121/4122 decode checks pass. Preserve the nonzero comparison exit status.
Failures are token1 head11 at layer39 (maxrel0.02287582 vs0.02), final layer63
residual (0.02633356 vs0.005), and final norm (0.01282051 vs0.008).

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/diagnose-wide-model-layer.py --tag cold-1-layer39
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/diagnose-wide-model-layer.py --tag cold-1-layer63
```

The offline diagnostic reads device captures without changing fixtures. Local
operations pass from device inputs; replaying layer63 still carries about0.02575
error against the independent path. Trace accumulation earlier in the chain
before choosing a precision change. Do not redo the expensive download,
conversion or reference preparation when continuing from intact artifacts.
Matching three tokens alone does not establish model support or close B7/B8.

For the subsequent compensated kernels and unchanged-fixture replays, use
`open-wide-model-precision` and its report. None of its measured variants closes
the full-depth gate; the first propagated BF16 change is layer1/channel2931.
