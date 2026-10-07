# Qwen3.8-27B validation report

2026-10-07. This branch adds a validation report for the current Qwen3.8-27B
implementation. The changes are documentation only; runtime, kernels and
converter code are unchanged.

## Validation

| Check | Result |
|---|---|
| `specs/open-engine/tests` | 770 passed, 47 skipped |
| `utilities/q4nx-build/tests` | 99 passed, 42 subtests passed |
| Standalone C++ CLI build | Passed |
| CTest | 3/3 passed |
| Manifest generation from the checked-in spec and local container | Both derive `qwen35`, 64 layers, 41 kernel build sets, 64 AB lanes and down split 8192 + 9216 |

These results were obtained during the code review. No xclbins were built,
weights reconverted or full-model NPU inference performed during this stage.
Skipped tests do not constitute hardware validation.

## Findings

### Decode comparison accepts incomplete or incorrect captures

`open_kernels/model/compare_decode.py` truncates logits to the shorter array,
prints residual errors without enforcing them, and stops checking layers when
either capture is missing. Top-5 is printed but not enforced.

The following FP32 captures reproduce a false positive with `--tokens 1`:

```text
ref_logits.bin = [1, 3, 2, 4, 0]
y_logits.bin   = [1, 3, 2, 4]
ref_res0.bin   = [1, 3, 2, 4, 0]
y_res0.bin     = [-1, -3, -2, -4, 0]
```

The comparator prints residual correlation `-1` and normalized maximum error
`2`, then reports **PASS** and exits `0`. This is a validation defect; it does
not demonstrate an error in real model outputs. No comparator fix is included
in this branch.

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
