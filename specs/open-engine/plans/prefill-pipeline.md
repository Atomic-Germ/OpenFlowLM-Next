# Prefill pipeline: keep the NPU busy while the host works

2026-10-07. Status: **plan, for review.** Builds on #172 (q8 split) and #177 (OMP wait policy).

## Why

On this HX 370 the Qwen3.6-35B-A3B prefills at 112-139 tok/s on OpenFlowLM. llama.cpp with
ggml-xdna reaches 196-259 tok/s on the same model, and hipfire 167-206
(`.claude/plans/qwen36-35b-a3b-engine-benchmarks.md`). Neither engine is near the hardware's
compute limit. At about 6 GFLOP a token, 120 tok/s is 0.7 TFLOPS. The gap is our pipeline.

A 1000-token prefill on the 8-bit set takes 8.35 s:

| Part | ms |
|---|---|
| Projection GEMMs (NPU) | 3,182 |
| Expert pass (NPU dispatches plus prep, patch, readback) | 2,400 |
| Host: DeltaNet conv + rule | 1,042 |
| Host: transposes, residual / norm / router, attention softmax, rest | ~1,720 |

**Every prefill dispatch is synchronous.** `run()` is `start_run` then `wait_run`
(core.cpp:815-827), so the host and the NPU never work at the same time. The spec already says
so: "nothing overlapping" (OPEN-PREFILL-BATCH, the 2026-09-21 result).

## What the code allows today

From a read of the layer-major path (`step_gemm_prompt`, core.cpp:2799-2838), per 256-token block:

- **Linear layer** (`block_layer_linear`, 2931-3002): prenorm, qkv|z GEMM, transpose, DeltaNet
  (host), out GEMM, residual + norm + router (host), shared expert up|gate GEMM, silu (host),
  down GEMM, gated add (host). Four dispatches, all in context `gemm`, so no context change
  inside the layer.
- **Full layer** (`block_layer_full`, 3154-3234): prenorm, q|k|v|gate GEMM, `attention_prep`
  (host), then four `ag` dispatches with a host softmax between, o GEMM, then the same tail and
  shared expert. That's **two context changes per block** (gemm→ag→gemm), about 2.5 ms each.
- **Expert pass**, once per layer for the whole prompt (`moe_block`, 3242-3350), in context `mb`.

The dependencies across blocks of one layer are narrow:

- **Linear:** only DeltaNet chains. Phase 1 needs the last 3 qkv rows of the previous block,
  through the conv state. Phase 2 needs S after the previous block. Prenorm and the qkv|z GEMM
  of block b+1 need nothing from block b, and everything after DeltaNet(b) feeds only this
  layer's expert pass.
- **Full:** only the KV rows chain, and `attention_prep` writes those before the attention
  products read them. Block b+1's projection needs nothing from block b.

What stands in the way of overlap:

- **One instance of every buffer.** `gemm_x_k*` / `gemm_y_n*` BOs, the `bs_` scratch, `gout_`,
  `sg_*`, `split_y_` are all reused every block.
- **Ordering between outstanding runs is only empirical, within one context.** Queuing across a
  context change hangs the array, and an instruction stream may not be patched while a run on it
  is outstanding (OPEN-DECODE-PIPELINE).
- **A busy CPU slows the NPU** through the shared power budget, so overlap won't recover all of
  the host time. That's why each stage has a measurement gate.

## Stages

Each stage is one PR, and each has a gate it must pass before the next starts.

### 1. Overlap within a layer: stage-major instead of block-major

Inside one layer, run each step over all of the prompt's blocks before the next step, instead
of each block through all the steps:

- **Linear layer:**
  1. Submit qkv|z GEMM(b+1).
  2. On the host, transpose and DeltaNet block b, in block order. The S chain is unchanged.
  3. Wait for GEMM(b+1).
  4. Then the out GEMMs, pipelined the same way against residual, norm and router, then the
     shared expert's two GEMMs against their host steps.
- **Full layer:**
  1. All q|k|v|gate GEMMs, pipelined against `attention_prep`.
  2. **All attention for the layer in one visit to `ag`.** That's two context changes per layer
     instead of two per block. It's the "hoist" the spec recorded as not done: 0.66 s at
     2582 tokens.
  3. Then the o GEMMs and the tail, pipelined.

What it needs:

- **Asynchronous submit in prefill:** `start_run` / `wait_run` already exist. At most one run
  outstanding beyond the one being waited on, always in the same context.
- **Two of each GEMM x / y buffer**, ping-pong per block: about 60 MB more for the 8-bit set.
- **Per-block host scratch**, or two of it, so that block b's host step and block b+1's
  transfer don't share memory.
- **The CLFLUSH rule:** the host never writes into a y buffer the NPU will write again
  (OPEN-PREFILL-BATCH, 2026-10-06 result).

**Gate:**

- **Logits** at every position on the 656-token prompt, **byte-identical** to today's
  layer-major. Each block's arithmetic is unchanged; only the order across blocks moves.
  Checked on both the 8-bit and the 4-bit set.
- **Prefill time**, alternated against today at 1k / 2k / 4k / 8k through `oflm bench`.
- **Expected:** a linear layer's time moves from (NPU + host) toward max(NPU, host). From the
  breakdown, that's roughly 8.35 s → 6-6.5 s at 1000 tokens (about 155-165 tok/s), less
  whatever the CPU's power draw takes back.
- If it gains less than 10%, stop and find out why before going on.

### 2. Overlap inside the expert pass

Gather pass i+1's tokens into a second `mb_x` while pass i runs. Scatter pass i's output while
pass i+1 runs, keeping the per-token expert order, which bit-exactness relies on
(core.cpp:2795-2798). A stream may not be patched while it has a run outstanding, so
consecutive passes on the same `mb_s*` stream need two copies of that stream's instruction
buffer, or must wait.

**Gate:** byte-identical logits; prefill time alternated. About 0.3 s of prep, patch and read
is in play at 1000 tokens.

### 3. GEMM efficiency

Measure first: achieved TFLOPS and bandwidth per shape with `--bench-kernel`, against the
NPU's peak and against the same shape at q4_1.

Candidates, in order:

1. **Add the q8 halves on the NPU.** The split GEMM accumulates hi and lo into the same output
   rows, so N rows come back instead of 2N. That halves the readback and removes the host sum.
   Same weights, exact; needs a new kernel variant with two weight streams and one output.
2. **Output in token-major order** ([T, N]), which removes the host transposes (`gemm_tr`,
   ~390 ms at 1000 tokens), if the kernel's output DMA allows it.
3. **A q8-reading GEMM** (a new dequant body in `gemm_q4_prefill`). The q8-block-prefill plan
   left this as its fallback. It halves the projection arithmetic against the split. Only if
   step 1's measurements say the split's arithmetic, not its bytes, is the cost.

**Gate:** per-shape timings before and after; logits within the route's measured spread for any
change that alters arithmetic order, and byte-identical where it doesn't.

### 4. Placement: CPU or NPU per stage, and moving stages

Only once there's a stage worth moving, and after stages 1-2 show what is still on the critical
path. The design:

- **The manifest says what's possible.** A stage can run on the NPU only if this kernel set
  carries its kernel. The engine refuses an NPU placement the set can't serve.
- **The engine owns the default.** Per stage, the NPU where the kernel exists and measurements
  say it wins, recorded in the spec with the measurement.
- **One override.** `OFLM_OPEN_PLACE=<stage>:<cpu|npu|auto>,...` (and `--place` on the CLI)
  replaces a new environment variable per stage. `auto` is a fixed rule on known inputs, mainly
  prompt length (as `OFLM_OPEN_GEMM_BLOCK_MIN` and decode's position threshold already are),
  never on live timing. CPU stages run in fp32 and NPU stages in bf16, so timing-based placement
  would make outputs vary from run to run.
- **Every stage that can move gets its own accuracy gate** against its CPU form. That cost is
  why the set of movable stages stays small.

Candidate stages, cheapest first: norms + router, the attention softmax (between `ag_s` and
`ag_pv`, it would remove a host round trip per head group), then the DeltaNet rule. The
DeltaNet rule is sequential over tokens and suits the NPU worst; decide on it only with stage 1's
numbers.

### Context changes

Item 3 of the review is mostly stage 1's attention hoist. What's left is two changes per layer
(gemm↔mb around the expert pass): about 0.2 s per prefill at any length, about 2.5% at 1k. The
GEMM and expert cores can't share one image (`plans/archive/gemm-context-collapse.md`: L1 and
topology), so that stays.

## Spec impact

- **OPEN-PREFILL-BATCH, modified (stage 1):** the layer-major schedule runs each step across all
  of a layer's blocks, with the next block's NPU work overlapping the host's; a full-attention
  layer visits `ag` once per layer. The criterion "layer-major vs block-major byte-identical with
  `OFLM_OPEN_MOE_BATCH=0`" stays. Added: the stage-major schedule is byte-identical to the
  block-at-a-time layer-major one. `OFLM_OPEN_STAGE_MAJOR=0` keeps the old order for the A/B,
  as `OFLM_OPEN_LAYER_MAJOR=0` does.
- **OPEN-MOE-BATCH, modified (stage 2):** passes overlap; the scatter order is unchanged.
- **New requirement, OPEN-PLACEMENT (stage 4 only):** the manifest/engine/override split above.
  It's written when stage 4 starts, not now.
- Verification stays `manual` (it needs the NPU), with the byte-identical logits as the gate.

## Open questions

- **How much does a busy CPU slow concurrent NPU work?** Stage 1's numbers answer it, and the
  answer sets how far stages 2 and 4 are worth taking.
- **Is the ordering of two outstanding runs in context `gemm` reliable** at the depth stage 1
  needs (one ahead)? Decode already relies on it inside one context. Stage 1 checks it with a
  long soak run before trusting it.
