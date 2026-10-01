---
name: imatrix-ffn-pruning
description: Pack a dense model whose FFN is too wide for a main core's L1, by --prune-ffn K with an importance matrix. Use when a build dies on "a N-wide activation table does not leave room for the streams in a core's L1", when preparing a Qwen3.8-27B or any wide dense Qwen3.5, or when deciding whether to requant, repack, or give up on a width.
---

# Imatrix FFN pruning

A dense FFN too wide for a core's L1 **cannot be exported at all** — not badly,
not slowly, not at all. Qwen3.8-27B's 17408 puts a 39168 B activation table
against a 61440 B budget that must also hold the x elements and both scratch
buffers, and `recipes/qwen36moe.py` `per_call` refuses. Narrowing the FFN is
the fix. Which neurons survive is the whole question.

```
oflm pack / q4nx-build --prune-ffn 12288 [--imatrix PATH] [-i <gguf>] [-s Qwen/Qwen3.8-27B]
```

## Why the imatrix, and why refusing without one is right

Measured on the 27B's imatrix, all 64 layers, K = 12288 of 17408:

| | activation mass retained |
|---|---|
| top-K by importance | **88.5%** mean, 83.2% worst layer |
| first K (a tail chop) | 70.6% by construction |

Eighteen points, for free. **So the pack refuses when no imatrix is given** —
the alternative is the tail chop, and a silently-unpruned pack would produce a
20 GB container that cannot run, which is worse than any of the three bad
outcomes.

**But this is not a quantisation choice, and the README says so.** Q4_K keeps
every weight at ~4.5 bits; this deletes 29% of the FFN outright. A pruned
container is not interchangeable with a denser one at any bit width, and
comparing the two as if it were would be dishonest. The banner carries the
retained fraction for that reason.

## The three things that fail silently

Each of these produces a container that converts cleanly and computes a
different network than the log claims. All three are commented in the code.

**One index set per layer, from `ffn_down` alone.** `ffn_up`/`ffn_gate` and
`ffn_down` share the intermediate axis; apply `ffn_down`'s ranking to all
three. Scoring each tensor's own columns separately is wrong and undetectable.

**The axis is found by LENGTH, never by a row/column flag.** ggml is
column-major, so `gguf.dequantize` returns the **transpose** of the shape the
GGUF header declares:

```
blk.0.ffn_down.weight   declared [17408, 5120]  ->  dequantized (5120, 17408)
blk.0.ffn_up.weight     declared [5120, 17408]  ->  dequantized (17408, 5120)
```

Opposite to each other, and opposite to the header. `gather_ffn(w, idx,
ffn_width=...)` picks the axis by which dimension equals the FF width. A wrong
axis is **not an error** when K < FF — it gathers the wrong neurons silently.
I shipped this bug once.

**Every layer pruned, or none.** `config.json` declares ONE
`intermediate_size` and the kernel recipe builds for it, so a layer left at
17408 is read as 12288 at runtime. A layer past the imatrix's range is a hard
error naming it, not a warning.

## The MTP block

Qwen3.5 exports its multi-token-prediction head as one **extra transformer
block**. Qwen3.8-27B is `block_count` 65 with `num_hidden_layers` 64 and
`mtp_num_hidden_layers` 1: `blk.64` holds the four `.nextn.*` tensors and is
the only block that does.

Identified by `.nextn.*` being **present** in a block, not by index arithmetic
against the layer count, so it stays right for a model that ships no MTP or
ships two. It is dropped **only in a pruned pack** — an unpruned pack keeps
every tensor it always has, and a 65-layer container is a separate question.

`num_hidden_layers` is then **measured from the converted tensors**, not
adjusted from the source's number. `generate_config_from_gguf` maps
`block_count` straight across (65), while a source config may already exclude
it (64) — subtracting from the latter labels a 64-layer container **63**, and
the engine walks that field.

## Requant and quant format

Requant is genuinely free: `q4_0` / `q4_1` / `q4_k` / `q8_0` are all choices,
and the container follows the spec. Two format facts, both found the hard way:

- **The activation table is `2.25 * K`** — built from *dequantized* values, so
  it is byte-identical across all four formats. Requant cannot relieve an L1
  overage; only `intermediate_size` can, and that is the model's, not a quant
  choice.
- **A Q4_K source converts.** `unpack_q4_k` returns float32 scales (`S*s_j`
  factored, 17 bits to stay exact), `unpack_q8_0` returns float16, and
  `_pack_q8nx` reinterprets scale bytes *as* float16. The cast belongs at the
  pack boundary. Without it, any tensor with a Q4_K source and a Q8_0 target
  — `ssm_alpha_proj` / `ssm_beta_proj` — dies in a bare assert, so a Q4_K /
  Q4_K_M pack could not be converted at all. Both Q8_0 and Q4_K_M sources now
  pack.

## Where the imatrix comes from

`--imatrix` wins, then `OFLM_IMATRIX` (so `oflm pack` can name it in the
environment), then a sidecar beside the model file. A sidecar is matched by
**content** — a GGUF whose tensors are `*.in_sum2` — not by filename, because
llama.cpp's naming varies. `llama-server ... -hf <repo>:Q4_K_M` will have
produced one.

## Verify a pack before trusting it

```python
# the manifest at the head of model.q4nx; no need to read the whole file
raw = open('model.q4nx','rb').read(1 << 22); i = raw.find(b'{')
# ... scan for the balanced object, json.loads, then:
#   all 3*layers FFN shapes consistent, and layers == config num_hidden_layers,
#   and intermediate_size == the --prune-ffn K
```

All 192 FFN tensors at 12288 read as `[160, 48, 5120]` and `[384, 20, 5120]`
(48 and 20 are `12288/256` and `5120/256`). If you see 544 or 68, the prune did
not reach the tensor and the container is the unpruned one.

## Rules

- **Use `-s` to name the source repo.** A pack with no source falls back to a
  GGUF-derived config that has no `linear_num_value_heads` /
  `full_attention_interval` and a `head_dim` taken from a rope dimension (64
  where the model is 256). It now refuses rather than writing 16 GB that dies
  at load; if you see that error, pass `-s Qwen/Qwen3.8-27B`.
- The output directory is tagged `-imx<K>` when the name came from the source
  card, so a pruned model is distinguishable from the unpruned one by `ls`.
- A model can be packed at several K. 12288 is the good default; 11776 exists
  because a narrower FFN buys L1 headroom. Compare retained mass from the pack
  log rather than assuming.
- **A pruned 27B does not run on closed kernels on this build** — the loader's
  `proj_weights` buffer predates 3:1 value:key geometry. See
  `open-qwen36-kernels` for the version-gate detail and `of_lni` for the open
  path's remaining wall.
