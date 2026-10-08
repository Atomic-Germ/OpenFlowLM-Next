# Qwen3.8-27B validation report

Updated 2026-10-08. This branch strengthens decode/source validation, adds an
offline converter command for existing model metadata, and builds the current
Qwen3.8-27B production recipe. An eight-layer, three-token NPU slice passes the
strict CPU-reference comparison, as does the first token at all 64 layers.
Runtime and kernel arithmetic are unchanged.

## Stage 1: complete decode comparison

- Added manifest-derived fixture metadata with the expected token/layer counts,
  hidden/logit sizes, context capacity, seed, spec/build identifiers and reference SHA256 hashes.
- The comparator requires every declared residual and logit capture, exact
  sizes and finite values. Reference hashes are verified before comparison.
- Enforced residual normalized maximum error < 0.005 alongside the existing
  logit correlation > 0.9999 and equal-argmax requirements. Top-5 remains a
  diagnostic. Zero reference residuals require exact zero outputs.
- Default comparison covers the whole fixture; explicit `--tokens` must match.
  Missing metadata requires reference regeneration. A cfg-only rewrite cannot
  relabel or rehash an existing reference, and interrupted generation invalidates
  the old metadata.
- Recorded and printed independent versus device-assisted reference routing.
  Metadata identifies the fixture; it does not hash all weights/kernel binaries.

TDD: the initial regression suite produced **38 failures and 4 passes** before
implementation. After the fix and generator integration checks, **51 tests
pass**. Cases cover truncated/missing files, missing complete layers, non-finite
values, modified references, residual bounds, zero residuals, token-count
mismatches, argmax/correlation failures and interrupted reference generation.
A further failing regression pinned context capacity for cfg-only rewrites.

## Stage 2: model preflight and production build

Added `q4nx-build --normalize-config -i DIR` for existing dense Qwen3.5-family
containers. It checks Q4NX header/ranges, layer indices and input-norm widths,
rejects conflicting metadata, normalizes through the converter's runtime
helper and preserves canonical `model_type=qwen3_5`. The original config is
backed up before atomic replacement. Weights/tokenizer files are not rewritten;
the command works offline and repeated invocation is a no-op.
Added `utilities/check-open-model.cpp` to run the production C++ manifest/config
check without loading XRT or model weights.

Source validation now fails for missing requested layers and non-finite tensor
comparisons. Finite equal scalar/constant tensors pass; nonconstant tensors
retain correlation > 0.99. TDD started with seven failing normalization tests
and six failing source-check tests. Additional regressions cover malformed
metadata/descriptors, canonical model type and backup preservation: **14
normalization tests and six source-check tests pass**.

Local model directory: `Models/qwen38-27b/Qwen3.8-27B-NPU2`. Weights are
hard-linked from the existing container; metadata assets are separate copies.
The normalized config passes C++ `Manifest::check_model`. It derives 64 layers,
248077 real tokenizer IDs and 248320 padded logits. The final export uses
`--model-dir` so those values, rather than a shape-only spec's vocabulary,
are recorded in its manifest.

Independent source check: text layers **0 and 3**, **25 tensors**, `ALL MATCH`;
minimum correlation **0.996756**. Source is the first safetensors shard of
`Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
This checks representative linear/full-attention layers, not all model weights.

Production export: **41 sets** (four decode, 32 attention products, five GEMMs)
under `Models/qwen38-27b/production-kernels`. All **82 binary hashes**, every
manifest file reference and the model-derived spec/build keys were verified.
Spec hash: `ff08e8129933d27b3edad6830486bf757285805118972ffffb3d86ed92640ab0`.
Build key: `7424a5e3be78958995870b6b37a5551a07a0969de477310bf5d7599b09a0c092`.
Program memory maxima:

| Set | Maximum core .text, bytes | Limit |
|---|---:|---:|
| lx | 15584 | 16384 |
| ax | 14400 | 16384 |
| ln | 3424 | 16384 |
| lm_head_q8 | 2432 | 16384 |
| Attention products | 1984 | 16384 |
| GEMMs | 4784 | 16384 |

Linux toolchain: Python 3.14.4, mlir-aie 1.4.2,
llvm-aie `21.0.0.2026080301+c9c5ecb7`, XRT tools in `/opt/xilinx/xrt/bin`.
The exporter requires a local multiprocessing socket, so it was run outside
the sandbox after the restricted attempt failed at forkserver startup.

Reproduction and source/weight hashes are recorded in
[the production preflight skill](../../../.opencode/skill/open-qwen38-production-preflight/SKILL.md).
The export's `toolchain.json` records binary hashes and build provenance.
Compilation was followed by the hardware validation below.

## Stage 3: eight-layer NPU decode and reset validation

Tested on AMD Ryzen AI 9 365 / Strix XDNA2, native Linux, with the production
export above. The harness and standalone C++ CLI were rebuilt from this branch;
CTest passed 3/3. Before execution, all 82 exported binary hashes were verified,
including equality of the eight decode artifacts in the harness's build
directories with their exported counterparts used by the runtime.

Regenerated the independent CPU fixture with `make_decode.py --layers 8
--tokens 3`. It includes six linear-attention and two full-attention layers,
seed token 248045, context capacity 4096, 5120 residual values per layer and
248320 logits per step. `decode_reference.json` pins all 27 reference captures
and the model-derived spec/build keys. The CPU replica mixes FP64 norm/state/
attention math with FP32 projections and LM-head accumulation; reference
captures are stored as FP32.
Harness execution completed **30/30 dispatches**, and the strict comparator
passed all three logits and all 24 layer residuals:

| Step | Full-logit correlation | Argmax, NPU / CPU | Harness / runtime logits |
|---|---:|---|---|
| 0 | 0.9999991073 | 220 / 220 | Byte-identical |
| 1 | 0.9999988538 | 220 / 220 | Byte-identical |
| 2 | 0.9999990533 | 220 / 220 | Byte-identical |

Worst normalized residual maximum error: **0.001397009**, layer 4 at step 0,
below the unchanged **0.005** bound. Every logit correlation exceeds **0.9999**;
dimensions, finite values and reference hashes also pass. The CPU reference
and runtime consume the same sequence: 248045, 220, 220.

The CLI's `--twice` run reproduced all three generated token IDs. Its second
request's logit dumps match the harness byte for byte. An additional
`--det-step 3 --det-full` run replayed the three input IDs from reset three
times: **0/3 replays differed** from the NPU reference run. Captured activation
buffers, current KV rows, residual/norm buffers and logits match byte for byte;
recurrent state is checked by hash. This tests repeatability, not independent
numerical correctness of the recurrent state or unused KV rows.

Reproduction commands are in the production preflight skill. Local captures,
reference metadata, logs and full-precision metrics are retained under ignored
`Models/qwen38-27b/production-slice/` (`results.json`, `logs/`).
This stage changes documentation only; no arithmetic fix or threshold change
was needed. Full 64-layer accuracy, per-head/state reference comparisons,
prompt coverage, serving and hybrid block-prefill parity remain unverified.
No performance claim: the power-mode request failed, and CPU reference
generation ran concurrently with the runtime checks.

## Stage 4: runtime comparison and full-depth checks

Added `compare_decode.py --runtime-prefix PREFIX` to require the CLI's
`PREFIX_tN.bin` logits to match the harness for every declared fixture step.
It rejects missing files, wrong sizes, partial float bytes, non-finite values,
one-ULP differences and signed-zero differences. CPU-reference residual/logit
gates still apply. A runtime run with more steps than the fixture does not
extend the fixture's coverage.

TDD: **18 new tests failed before implementation**, then all **69 comparator
tests passed**. The complete open-engine suite passes **845 tests, 47 skipped**.
The new option also passed on the retained eight-layer/three-token fixture.

The full-depth first-token fixture (seed 248045) passed the same strict gates:
**64/64 residual comparisons**, **248320 logits**, correlation
**0.9999973866**, and argmax **8678 / 8678**. Worst residual normalized maximum
error is **0.002713315**, layer 57, below **0.005**. Harness execution completed
**66 dispatches**. Its logits and the second runtime request's first-token
logits are byte-identical. All reference hashes, dimensions and finite-value
checks passed; no threshold or kernel arithmetic changes were needed.

The hardware capture ran while CPU reference generation continued, using the
same `build_cfg` function after all input pools were written. Its configuration
was verified byte-identical to `run_decode.cfg` published with the completed
reference contract. The 82 export hashes and eight harness decode artifact
hashes were checked before execution.

On the same Ryzen AI 9 365 / Strix NPU, the full **64-layer** runtime generated
8678, 198, 2 from seed 248045 and reproduced those IDs on a second request.
Three reset replays of inputs 248045, 8678, 198 reported **0/3 differing runs**
for observed buffers and recurrent-state hashes. These are repeatability
checks against an NPU reference run, not CPU state validation.

Two short text requests used the model's `chat_template.jinja` with thinking
disabled and sequential NPU prefill:

| Prompt | Prompt tokens | Answer before EOS | Repeated request |
|---|---:|---|---|
| Reply with only the result: 2 + 2 = ? | 25 | 4 | Identical |
| Name the capital of France. Reply with only the city. | 24 | Paris | Identical |

Both emitted `<|im_end|>` immediately after the answer. The diagnostic CLI
generated eight tokens per request; tokens after the first EOS are excluded
from the answer. Rendered prompts, exact input IDs, the template hash and raw
logs are retained under `Models/qwen38-27b/production-full/`. This is limited
chat smoke coverage, not serving or hybrid block-prefill validation.

Full-precision metrics and the pinned reference metadata are in
`Models/qwen38-27b/production-full/results.json`; the skill documents the run
commands. Only the first full-depth token has an independent CPU comparison.
Full-depth multi-token accuracy, independent state/head/FFN-partial checks,
block-boundary prefill parity, serving and resource/timing acceptance remain
open. The power-mode request still failed, so this is not a benchmark result.

## Validation

| Check | Result |
|---|---|
| `specs/open-engine/tests` | 845 passed, 47 skipped |
| `utilities/q4nx-build/tests` | 113 passed, 42 subtests passed |
| Standalone C++ CLI build | Passed |
| CTest | 3/3 passed |
| Manifest generation from the checked-in spec and local container | Both derive `qwen35`, 64 layers, 41 kernel build sets, 64 AB lanes and down split 8192 + 9216 |

The open-engine suite was rerun for stage 4; converter results are from stage 2
and the C++ build/CTest from stage 3 (those sources are unchanged). Weights were
not reconverted. Skipped tests and successful compilation do not extend the
explicit hardware coverage above.

Stage 1 reproduction:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest specs/open-engine/tests/test_compare_decode.py -q
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest specs/open-engine/tests -q
```

The full suite passes with the new comparator tests included. Existing numerical
references and kernel arithmetic were not changed.

## Findings

### Decode comparison regression fixed

The comparator previously truncated logits to the shorter array, printed
residual errors without enforcing them, and stopped checking layers when
either capture was missing.

The following FP32 captures reproduced a false positive:

```text
ref_logits.bin = [1, 3, 2, 4, 0]
y_logits.bin   = [1, 3, 2, 4]
ref_res0.bin   = [1, 3, 2, 4, 0]
y_res0.bin     = [-1, -3, -2, -4, 0]
```

The previous comparator printed residual correlation `-1` and normalized
maximum error `2`, then reported **PASS** with exit `0`. Stage 1 rejects both
the truncated logits and, independently, the incorrect residual. This fixes
acceptance of invalid captures; it does not demonstrate a numerical change
in real model outputs.

### Local container metadata normalized

The initial nested `text_config` failed C++ `Manifest::check_model` with
`config.json lacks 'head_dim'`. Stage 2 provides a converter command to fix
this metadata layout and verifies the normalized production directory against
the same C++ check. The source directory and weight contents are preserved.

### Block prefill includes host computation

`Core::block_layer_linear` invokes host RMSNorm and `host::deltanet_block`
between NPU GEMMs. Block prefill therefore uses a hybrid CPU/NPU execution
route; it must be distinguished from sequential NPU execution when reporting
coverage and performance. No performance measurements were made in this stage.
