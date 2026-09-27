# Synthetic wide autoregressive decode

**The synthetic B7 multi-token decode gate passes (2026-09-27):4570 inherited
slice checks and579 decode checks across684 open NPU dispatches.**

This B7 step extends the accepted eight-layer slice with final RMSNorm, the
full248320-row Q8 LM head, greedy token selection and embedding feedback.
All weights are synthetic. The 64-layer packed model and production runtime
integration remain separate PR4 gates.

## Implementation and TDD

Six new tests first failed because native token control, composition and the
logits validator did not exist. Native tests compile the actual C++ header with
undefined/bounds sanitizers: negative logits, first-index tie breaking, nonfinite
logit rejection, BF16 row addressing/widening, invalid token, geometry/overflow
and file-size rejection. Python tests verify the device-head/selection ordering,
feedback directives, invalid dimensions and the complete-logit acceptance gate.
After implementation the full CPU suite passes: **783 passed,47 skipped**.

The open harness adds `greedy` and `embed` directives. The latter seeks one row
in the BF16 embedding file and widens its representation to FP32. Both validate
requested BO lengths, not padded allocation capacity. Token selection and row
lookup are host control; all neural operators stay on the NPU. Existing harness
directives are unchanged. No xclbin or host library is rebuilt/distributed.

`test-wide-slice.py` exposes its existing allocation/capture helpers for reuse.
Their generated commands match all32 previously recorded layer evaluations
byte-for-byte. The decode probe reuses every slice comparison, including state
isolation, padding, canaries, independent per-layer outputs and KV row checks.

## Fixture and acceptance

- Layer weights/constants: unchanged distinct production-packed fixtures from
  `wide_deltanet/build_slice` (seed38429 and layer index).
- LM weights/kernel: accepted K5120/N248320/8-core Q8 fixture (seed2963).
- Embeddings: all248320x5120 BF16 rows, independent seed38430; final norm weights
  and warm initial state continue the same random stream.
- Input seed: token248045. Three cold steps at0/1/2, three warm steps at254/255/256,
  then full state reset and replay of all three cold autoregressive steps.
- Every token runs74 layer dispatches, one final LN and one LM head:684 total.
- All six primary token inputs and next IDs are independently computed by the
  FP64 reference. The cfg loads a seed ID only at sequence start/reset. Reference
  outputs/IDs are never loaded for device computation.
- The complete logits vector must be finite, have exactly the expected length,
  correlate>.9999 with reference and select the same argmax, preserving the
  existing `model/compare_decode.py` criteria without its length truncation.
- Supplementary LM comparison uses the actual device-normalized input, requiring
  maxrel<1e-4 and cosine>.9999999. Final norm uses the inherited8e-3 bound.
- Reset replay compares every captured input, output, scratch, state, norm,
  logit vector and token ID byte-for-byte for all three cold steps.

Preparation hashes all fixtures and kernels, refuses existing output directories,
and validates the accepted prerequisite artifacts. Large packed files are linked
and remain hash-checked; the embedding file adds2542796800 bytes on disk.
Device buffers include the1.35 GB head weights; the embedding table remains a
file and only its selected row is read. Declared BO storage totals1776101004
bytes before alignment/XRT overhead. Eleven contexts execute sequentially.

## Hardware results

| Sequence | Positions | Input seed and three selected tokens |
|---|---|---|
| Cold | 0,1,2 | 248045 →208103 →83896 →182358 |
| Warm | 254,255,256 | 248045 →228952 →61499 →228952 |
| Reset/repeat | 0,1,2 | Same cold sequence, all captures byte-identical |

Both primary sequences independently match the CPU reference. The smallest
reference top-two logit margin is2.83560; no tie or tolerance tuning was needed.

| Metric over the six primary tokens | Worst result | Required |
|---|---:|---:|
| End-to-end logits correlation | 0.9999999228297815 | >0.9999 |
| End-to-end logits normalized max error | 0.00039995896 | Diagnostic |
| Selected token | 6/6 identical | All identical |
| Final normalized vector max error | 0.002358491 | <0.008 |
| Conditional LM-head max error | 4.4079191e-6 | <1e-4 |
| Conditional LM-head cosine | 0.9999999999886854 | >0.9999999 |
| Last-layer residual max error | 4.0217049e-5 | <0.005 |
| Whole DeltaNet state max error | 0.002216794 | <0.02 |

Every inherited slice acceptance check passes, including per-head attention
gates, state/activation copies, padding, untouched cache rows and canaries.
The stricter recurrent-state head-local1e-4 diagnostics pass10/36 layer-token
tensors. They remain visible in `slice-results.json`; this result does not
establish strict primitive equivalence. No bounds, seeds or arithmetic were
changed to make the decode gate pass.

This closes synthetic eight-layer autoregressive validation. The next B7 gate
is a64-layer full model, followed by production runtime integration. This probe
does not validate real model quality, runtime instruction patching, packed-model
distribution, prefill, performance or catalogue promotion.

## Reproduction

Use [open-wide-decode](../../../.opencode/skill/open-wide-decode/SKILL.md), including
the harness build against matching XRT headers. Artifacts are in
`open_kernels/designs/wide_deltanet/build_decode`; preparation, hardware,
comparison and CPU-suite logs are `/tmp/wide-decode-{prepare,hardware,compare,unit}.log`.

Host: Ryzen AI9 365/Strix, XRT2.26.0, matching headers from the local XRT build,
GCC15.2.0, CMake Release harness. Harness SHA256:
`1d580f3de43eb17f77973dcc6b4b79102d3ff3f307ffd1ac33bdac31d4168130`.

Evidence SHA256:

- `slice-fixture.json`: `12ff6a830a494dfc1515579cd4bc17d658c193f71df100957a399a4c687676bd`
- `slice-results.json`: `60d089ba58a21fe0bf4d111d984797a4acfc6ae3bb4769b69f223747b576ab6d`
- `decode-results.json`: `474954dfecaf953d02e589b50a8b7666f355eb972bcf62f279f1022f61e3e9ef`
