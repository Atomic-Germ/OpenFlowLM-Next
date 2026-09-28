# Qwen38 full-depth precision experiments

Date:2026-09-28. Continuation of the failing
[real64-layer gate](qwen38-full-model.md). Primitive improvements are measured
separately from full-model acceptance. No runtime/catalogue promotion.
**Full-model acceptance remains FAIL.** The best measured variant in this
stage is corrected output projection only: two failures remain instead of three.

## Changes and resource bounds

All new arithmetic is opt-in. Existing model weights, pool formats, reference
tensors, BF16 interfaces, seed token and acceptance thresholds are unchanged.

| Standalone kernel | Change | Data + stack/core | Text/core |
|---|---|---:|---:|
| K6144/N5120 output projection | Existing residual Q4 correction, reclaim unused recurrent ds | 62976 B | 3792 B |
| H5120/FF17408 FFN | Correct all GEMVs, K4096 down segments | 63552 B | 13840 B |
| Q24/KV4 attention | Precise Q/K RMSNorm, RoPE and output gating | 51728 B | 14944 B |
| N5120 residual RMSNorm | Compensated square accumulation across both halves | 57600 B | 3872 B |

Output projection's table grows to27008 bytes. Its unused ds buffer shrinks
from1280 to32 floats; this is an isolated projection allocation, not permission
to shrink a fused layer's recurrent scratch. The K5120 corrected probe retains
its existing buffers and behavior.

The compensated FFN uses four4096-wide segments and a1024 tail. This bounds the
corrected table by up/gate K5120 (22592 bytes). The generic segment helper now
accepts a1024-aligned limit; its default stays8192. Segment-major DMA still
covers every original Q4 pool chunk once and accumulates in the same local ds
slots. No model repacking is required.

Attention changes are limited to the standalone precise probe. The online
softmax, two-component query storage and KV ABI remain unchanged. Wide LN adds
32 compensation floats to its statistics buffer; every new invocation resets
them at the first half and retains them across the second half.

## Validation and independent replay

Resource/segment/replay tests were added before implementation and failed for
the absent interfaces. The hardware RED case remains the original full-model
capture. Tests cover exact DMA coverage at both8192 and4096, all output bands,
two-token accumulation reset, memory bounds, incompatible attention DMA lengths,
LN epsilon, changed artifacts, and isolated replay destinations.

`utilities/replay-wide-model.py` verifies original fixture/kernel hashes and
each replacement primitive's passing results, geometry and artifact hashes.
It links only immutable input/reference files into a fresh destination and
rewrites kernel bindings; old captures and comparison results are never copied.
The original `test-wide-full-model.py compare` remains the acceptance authority.

Primitive results:

- Output projection:9/9 inputs, worst maxrel1.7348139161e-7.
- Corrected FFN:26/26 checks over13 original inputs; worst maxrel2.8614466311e-6
  versus1e-4. Trace mode additionally passes52/52 checks; h and output captures
  are byte-identical to normal mode on all13 inputs. Pool and seed are unchanged.
- Precise attention: all23 cases pass, including persistent cache and reset.
- Default Q16/KV4 attention regression: all23 cases pass; all47 captured files
  are byte-identical to the prior accepted `attn/build_regression_16_4` run.
- Compensated LN: all12 cases pass; every normalized BF16 output is exact
  against its independent synthetic reference.

## Full-model ablations

Each completed row runs3564 NPU dispatches with three autoregressive tokens
and full reset replay. Token sequence248045 ->8678 ->198 ->2 remains unchanged.
Failures are retained, not hidden by local/conditional comparisons.

| Variant | Token1 layer63 maxrel | Failed inherited checks |
|---|---:|---:|
| Original | 0.02633355729 | 3 |
| Corrected output only (`full_out`) | 0.01981752039 | 2 |
| Output + FFN (`full_ffn`) | 0.03512687048 | 8 |
| Output + FFN + attention (`full_attention`) | 0.03283389240 | 4 |
| Output + FFN + attention + compensated LN (`full_ln`) | 0.02359762341 | 10 |
| Output + attention, original FFN/LN (`full_out_attention`) | 0.03154386626 | 3 |

The output-only variant closes the original layer39/head11 failure. Adding more
accurate primitives does not monotonically reduce whole-model error: BF16
rounding changes subsequent trajectories. The corrected output/FFN reduces
cold token0/layer0 error from1.32105e-5 to1.01619e-7. At cold token0/layer3,
conditional complete-layer error improves from4.60576e-7 to9.21151e-8 with
precise attention. Neither observation proves full-depth acceptance.

All five variants preserve identical tokens and byte-exact reset replay, but
none closes the numerical gate. Do not enable every precision switch merely
because each passes its primitive test. CPU regression:803 passed,47 skipped.

## First propagated rounding change

`utilities/diagnose-wide-model-rounding.py` distinguishes the device's local
norm error (against FP64 RMSNorm of the actual device input) from input already
different from the independent reference. It checks hashes and canaries and
never modifies the acceptance trajectory.

Across384 entry/post-norm boundaries, `full_ln` has845814 BF16 differences from
the independent trajectory, only22 differences attributable to local norm math,
and845806 in the CPU norm replay of device inputs. These categories overlap;
their counts should not be added. Baseline local norm differences are also22.
Compensating norm accumulation therefore does not resolve the dominant source.

The first changed BF16 norm is cold token0/layer1/channel2931. The device
residual is-0.0008849045261740685, reference-0.0008848998695611954, a difference
of-4.6566128731e-9. It rounds to-0.00160980224609375 after normalization instead
of-0.0016021728515625. Replaying RMSNorm on the device input reproduces the device
bits exactly. Layer0 output projection error at this channel is6.5192580223e-9
and FFN output error is-1.1175870895e-8. At the next entry norm19 channels differ;
at the next post norm140 differ, all reproduced by CPU normalization of the
changed inputs. This is measurable propagation across rounding boundaries,
not evidence that RMSNorm itself has a large local error.

Next investigation: rounding/cancellation inside the corrected Q4 block
products and reductions producing this first residual. Retain independent
full-depth acceptance; matching tokens or local conditional outputs is not a
substitute for it.

## Reproduction

Use [open-wide-model-precision](../../../.opencode/skill/open-wide-model-precision/SKILL.md).
Builds, primitive fixtures, independent replays and their JSON results are under
`open_kernels/designs/wide_deltanet/build_model_precision/`. Persistent build logs
live beside each artifact; full attention/LN hardware and comparison logs live
inside their replay directories. The open harness is now built in
`open_kernels/harness/out` so reboot cleanup of `/tmp` does not remove it.
