# Drafters: the pruned 27B, fine-tunes, and DFlash 2

2026-10-05. Status: **not started; needs decisions.** Split out of the original spec-decode plan
(see [`spec-decode.md`](spec-decode.md)). Most of this is training and model work rather than
engine code, and it only makes sense once speculative decoding works end to end (spec-decode
PR 5).

## Models and their drafters

| model | kernels | drafter |
|---|---|---|
| Qwen3.6-35B-A3B (stock) | `qwen36moe` in q4_1, t2, q2 | z-lab DFlash v1, as is |
| 40-layer fine-tunes (Ornith 1.0/1.5, Darwin-36B-Opus, Grug, BigBang, Aquila-mini) | same | z-lab v1; fine-tune only where acceptance drops |
| Josh's pruned Qwen3.6-27B-A2.8B (30 layers) | same kernels, its own manifest | its own: remap v1's taps, then fine-tune |

Each expert format is its own container per model. One drafter serves all formats of a model,
with an optional fine-tune per format.

## PR 7: the pruned 27B on the open engine

So far it has only run through the `make_27b.py` test harness. Its layers have the same shapes
as the 35B's, so it should use the **same kernel sets** with its own manifest. That means a
`q4nx-build` builder, an `oflm-add` entry and a skill file (AGENTS.md). It's worth doing on its
own, and it's needed before its drafter can be trained.

## PR 8: drafter fine-tunes

Use SpecForge's DFlash trainer in offline mode: the target's features are computed ahead of
time, so only the small drafter needs to fit on the training GPU.

- **Quantized targets.** Ternary and 2-bit experts shift the hidden states the drafter reads.
  Dump features from a fake-quant model that uses our exact dequant, on the target's own
  outputs, and fine-tune starting from z-lab's weights. Cheap, because it starts from a working
  drafter.
- **The pruned 27B.** v1 reads layers 1, 6, 11, 16, 22, 27, 32, 37 of 40. Map each one to the
  layer it became after pruning (the nearest survivor where it was removed), keep the fuse
  layer's column order, then fine-tune. Taps near cut points will need the most retraining.
- **Variants.** Fine-tune only where acceptance on that model's own chat outputs falls well
  below the stock model's.

## Later: DFlash 2

DFlash 2 drafts 8 tokens instead of 16, and adds a two-tap convolution and a selector that picks
one consistent path from the top-16 candidates per position. Shorter blocks mean the verify pass
touches about half the experts (57 vs 102 per layer), and that matters most on this hardware.

- **First, try v1 at block 8.** It costs nothing, and it shows whether v2's gain comes from its
  architecture or just from the shorter block.
- **If v2 is still worth it:** copy v1's layers into the v2 structure at the same width, start
  the new convolution as a pass-through so the model begins exactly as v1, add a new selector,
  and train it. The catch is that **there's no public v2 trainer**. We'd have to write the
  training loop from the paper and the model code.
- Transplanting Qwen3.8-27B's own v2 drafter isn't worth it: it's 5120 wide against 2048, so
  every matrix would need to be projected.

## Decisions needed

1. **Pruning map:** which 30 of the 40 layers the 27B kept. Needed for the tap remap.
2. **Variant list:** the 40-layer fine-tunes above, or the top-N on Hugging Face by downloads?
3. **Training compute:** this box has no training GPU. What cloud budget is there for fine-tunes,
   a possible v2 trainer, and a possible ternary healing run?
