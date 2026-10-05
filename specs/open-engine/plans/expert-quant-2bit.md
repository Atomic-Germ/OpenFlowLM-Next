# 2-bit and ternary routed experts for the MoE family

2026-10-05. Status: **waiting on PR #161.** Split out of the original spec-decode plan (see
[`spec-decode.md`](spec-decode.md)).

## Decided

- **The engine supports many weight formats** (maintainer, 2026-10-05). t2 from #161 is in,
  and more formats are expected after it. A new format is normal work, not a special case.
- Applies to the **routed experts' gate and up** projections of Qwen3.6-35B-A3B and its
  variants. down_proj stays q4_1. Everything else stays at today's precision.
- Two formats: **ternary** (t2, from #161) and **2-bit affine** (q2, new here).

## Why

Routed experts are the part of decode that grows when speculative decoding checks many
tokens at once: 16 tokens touch up to 102 experts per layer instead of 8. Smaller experts
make every verify pass cheaper. They also help plain decode, where the routed experts are
~26 ms of a ~140 ms step.

Both formats use 2 bits a weight plus a scale per group, so they stream the same bytes, and
routed-expert bytes drop by about 36%. They differ only in how a 2-bit code becomes a
number:

- ternary: `{-1, 0, +1} x scale`
- 2-bit affine: `code x scale + min`, four levels instead of three

## What #161 already provides

PR #161 (Ternary Bonsai 2 27B) adds OPEN-QUANT-T2: ternary weights stored as 2-bit codes, read
by the decode GEMV and the prefill GEMM, plus the converter for PrismML's GGUF. It covers
**dense projections on a dense model**. It does not cover the MoE kernels.

So this plan doesn't design a ternary format. It uses #161's layout, scales and packer, and
extends them to the two MoE expert kernels:

- `mx`, the one-token expert step inside `lx`, used by decode
- `mb_*`, the batched expert kernel, used by prefill and the verify pass

q2 should reuse t2's code layout and differ only in the dequant (one more value per group, the
min), so that one kernel template serves both, built as two kernel sets. They can't share one
binary, because program memory is tight.

## Where the weights come from

- **q2:** `llama-quantize --imatrix`, with gate/up overridden to **Q2_K**. The packer converts
  Q2_K exactly into the native layout, the way OPEN-QUANT-Q4K handles Q4_K.
- **Ternary:** there's no PrismML ternary of the 35B, so we make one. Either llama.cpp's TQ2_0
  (quick, plain rounding) or our own calibrated ternarizer (absmean plus error feedback). See
  the open decision below.
- If ternary quality fails the tool-use check, a short healing fine-tune and a direct pack of
  the healed safetensors. That needs GPU time, decided in
  [`dflash-drafters.md`](dflash-drafters.md).

## Toolchain

- `q4nx-build --expert-quant {t2,q2}`
- `t2` / `q2` roles for `expert_gate_up` in the recipe's quant map, covered by `spec_hash`
- new points enter the validated set only after a hardware comparison, as today

## Checks

- **Unit:** the Q2_K -> native conversion is exact (like `test_quant_q4k.py`); the t2 expert
  packing round-trips.
- **Hardware:** logits against a fake-quant reference that uses the same dequant; then
  `oflm-test --llm --tools`.

## Requirements

| ID | change |
|---|---|
| OPEN-QUANT-T2 | **modified** (it arrives with #161): applies to the `expert_gate_up` role and `mx` / `mb_*` as well |
| OPEN-QUANT-Q2 | **new**: 2-bit affine experts from Q2_K |

## Open decision

**Ternary source:** start with TQ2_0 (fast, plain rounding) or write our own calibrated
ternarizer from the start?
