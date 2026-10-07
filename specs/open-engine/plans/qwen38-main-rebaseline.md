# Qwen3.8-27B validation report

2026-10-07. This branch strengthens decode validation and documents checks of
the current Qwen3.8-27B implementation. Runtime, kernels and converter code
are unchanged.

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

## Validation

| Check | Result |
|---|---|
| `specs/open-engine/tests` | 821 passed, 47 skipped |
| `utilities/q4nx-build/tests` | 99 passed, 42 subtests passed |
| Standalone C++ CLI build | Passed |
| CTest | 3/3 passed |
| Manifest generation from the checked-in spec and local container | Both derive `qwen35`, 64 layers, 41 kernel build sets, 64 AB lanes and down split 8192 + 9216 |

The converter and C++ results above were obtained during the code review;
those components were not modified by this stage. No xclbins were built,
weights reconverted or full-model NPU inference performed during this stage.
Skipped tests do not constitute hardware validation.

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

### Local container metadata requires normalization

The local container's nested `text_config` fails C++ `Manifest::check_model`
with `config.json lacks 'head_dim'`. A temporary copy processed by the current
converter's `inject_oflm_keys` passes the same check. Original model files
were not modified. This checks metadata compatibility, not tensor values.

### Block prefill includes host computation

`Core::block_layer_linear` invokes host RMSNorm and `host::deltanet_block`
between NPU GEMMs. Block prefill therefore uses a hybrid CPU/NPU execution
route; it must be distinguished from sequential NPU execution when reporting
coverage and performance. No performance measurements were made in this stage.
