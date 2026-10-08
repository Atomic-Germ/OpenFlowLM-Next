# Qwen3.8-27B validation report

Updated 2026-10-08. This branch strengthens decode/source validation, adds an
offline converter command for existing model metadata, and builds the current
Qwen3.8-27B production recipe. Runtime and kernel arithmetic are unchanged.

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
**No inference on the newly built kernels is claimed at this stage.** The next
hardware gate is the eight-layer/three-token slice with pinned CPU references.

## Validation

| Check | Result |
|---|---|
| `specs/open-engine/tests` | 827 passed, 47 skipped |
| `utilities/q4nx-build/tests` | 113 passed, 42 subtests passed |
| Standalone C++ CLI build | Passed |
| CTest | 3/3 passed |
| Manifest generation from the checked-in spec and local container | Both derive `qwen35`, 64 layers, 41 kernel build sets, 64 AB lanes and down split 8192 + 9216 |

The Python suites were rerun for stage 2. The C++ build/CTest results are from
the initial review; C++ code is unchanged. Weights were not reconverted and
full-model NPU inference was not performed. Skipped tests and successful
compilation do not constitute hardware validation.

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
