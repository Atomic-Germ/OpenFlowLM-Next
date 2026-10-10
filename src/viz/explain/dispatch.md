## dispatch:lx0
Title: lx0: linear-attention layer up to the router
Summary: One NPU run: input norm, qkv and z projections, convolution, DeltaNet, gated norm, out projection, residual norm and the MoE router.
Spec: OPEN-MANIFEST, OPEN-FAMILY-QWEN36MOE
Source: open_kernels/designs/layer_x/xlayer.py

`lx0` is the first of a linear-attention layer's two dispatches. It starts from the residual stream (`xres`) and ends when the router has picked this token's experts. In order:
- the norm/router core applies `input_layernorm` to `xres`;
- the eight main cores (row 2) run the `qkv` and `z` projections, 24 bands of 64 rows per core, each core streaming its own weights from `pool`;
- the glue core runs the 4-tap convolution over `qkv`, works out each head's decay and beta, and writes one record per value head;
- the main cores run the DeltaNet update, 4 of the 32 value heads each: a head's 128 × 128 state streams in twice and back out once;
- the post core normalizes the DeltaNet output per head, scales it by `ssm_norm` and gates it with SiLU of `z`;
- the main cores run the out projection (`ssm_out_proj`, stored in `consts`);
- the norm/router core adds the result to the residual, applies `post_attention_layernorm`, streams the 1 MiB router weight and writes the router record into `act`.

At q4_1 it moves about 30 MB: 15.7 MB of `qkv` and `z` weights from `pool`, 6.6 MB from `consts` (mostly the out projection and the router), and the DeltaNet state, read twice (4.6 MB) and written once (2.3 MB). A set that stores these projections at q8 moves twice their bytes.

One instruction stream serves all 30 linear-attention layers. Each run binds that layer's own `pool`, `consts`, `state` and `act`.

## dispatch:lx1
Title: lx1: the MoE block of a linear layer
Summary: The eight main cores run the 8 routed experts and the shared expert, then write the layer's output into xres.
Spec: OPEN-DECODE-PIPELINE, OPEN-QUANT-Q8
Source: open_kernels/designs/layer_x/xcommon.py

`lx1` runs on the eight main cores only. Each core reads a short header (the router record, the shared-expert gate weight and its slice of the residual), then loops over nine slots: the eight routed experts, then the shared expert. Per slot a core computes its 64 rows of the up and gate projections (`gemv_q4_gup`) and multiplies SiLU of the gate by the up value (`moe_silu32`). The full 512-value hidden vector comes back to every core over the broadcast stream, and each core runs its 256 rows of the down projection (`gemv_q4_gdown`) and accumulates (`moe_accfin`). Last, `moe_out` writes the layer's output into `xres`: the residual, plus the routed experts weighted by the router, plus the shared expert scaled by its sigmoid gate.

It moves 18.0 MB, nearly all of it expert weights from `pool`: 1.97 MB per expert (1.31 MB up and gate, 0.66 MB down), nine times. Experts stay at q4_1 in every set of this family (OPEN-QUANT-Q8), so `lx1` is the same size everywhere.

The stream is built with placeholder expert offsets. Between `lx0` and `lx1` the host rewrites them to the experts the router chose (the route stage). The experts on this page are a random illustration; the real ones depend on the token.

## dispatch:ax0
Title: ax0: full-attention layer up to the router
Summary: One NPU run: input norm, q/gate/k/v projections, attention over the KV cache on four cores, o projection, residual norm and the router.
Spec: OPEN-ATTN-CONTEXT, OPEN-DECODE-PIPELINE
Source: open_kernels/designs/layer_x/xlayer.py

`ax0` is the first of a full-attention layer's two dispatches. In order:
- the norm/router core applies `input_layernorm` to `xres`;
- the main cores run the q, gate, k and v projections (18 bands per core). The checkpoint stores q and the output gate as one tensor, `q_proj`, which the pool keeps as two regions;
- four attention cores (row 3, columns 2–5) each own 4 of the 16 query heads. One broadcast stream brings them the q/k norm weights, this position's `ptab` record, q, k, v, the cached K/V rows and the gate. They normalize q and k, apply RoPE, walk the cache in blocks of 4 rows and apply the sigmoid gate. Core 0 also writes this token's K/V row into the cache;
- the main cores run the o projection;
- the norm/router core adds the residual, applies `post_attention_layernorm` and runs the router, as in `lx0`.

At q4_1 it moves about 18 MB: 17.0 MB of projection weights (q and gate 10.5 MB, o 5.2 MB, k and v 0.66 MB each), the 1 MiB router, and 2 KB of KV cache per cached position. A q8 set moves twice the projection bytes.

The stream is built for position 1. Once per step the host patches it for the current position (the attnpos stage), and that one patch serves all ten full-attention layers. Measured on the one-context image, `ax0` takes 0.63 ms at position 1, 1.41 ms at 2048 and 2.15 ms at 4000 (OPEN-ATTN-CONTEXT). The page draws the unpatched stream (see Modelled timing), and its modelled `ax0` is shorter than these measurements.

## dispatch:ax1
Title: ax1: the MoE block of a full-attention layer
Summary: The same expert program as lx1, reading the full-attention layer's buffers; it writes the layer's output into xres.
Spec: OPEN-DECODE-PIPELINE
Source: open_kernels/designs/layer_x/xcommon.py

`ax1` runs the same MoE program on the main cores as `lx1`: a header, eight routed experts, the shared expert, and the layer's output into `xres`. It moves the same 18.0 MB.

It is a separate instruction stream because a full-attention layer keeps the router record, the residual and the expert hidden vector at different offsets in its `act` and `consts`. The host patches it with the router's choice exactly as it patches `lx1`.

## dispatch:ln
Title: ln: the final norm
Summary: One core applies the model's final RMS norm to the residual and writes the bf16 vector the lm head reads.
Spec: OPEN-MANIFEST
Source: open_kernels/designs/ln/ln.py

`ln` runs once per step, after the last layer, on one core and one shim column. Its kernel is the layer norm with a fused residual add: `y = x + add`, then `hn = norm(y) × w`. Here `x` is `xres`, `w` is the final norm weight (`normw`), and `add` is the `zero` buffer, because the last layer has already added its residual. It writes `y` to `xresf` and the bf16 result to `hn`.

It moves 32 KB, the least of any dispatch. On the page most of the time around it is the two context switches: `ln` has its own xclbin, so the NPU switches into it from the layer context and out of it into the lm head's.

## dispatch:lm
Title: lm: the lm head
Summary: Eight cores multiply the final hidden vector by the 248,320-row q8 lm_head matrix and write one fp32 logit per row.
Spec: OPEN-PACK-PLAN
Source: open_kernels/designs/lm_head_q8/lm_head_q8.py

`lm` uses eight cores (columns 0–1, rows 2–5). Each core has its own weight stream from the shim tiles in columns 0–3 (two streams per tile), and all eight receive `hn` once over a broadcast stream from column 4. The matrix is split into 1,940 bands of 128 rows: four cores take 243 bands, four take 242. Each band yields 128 fp32 logits.

The weights are q8, 540 MB in all, read once per step. That makes `lm` the largest single dispatch: one of 82, but about a fifth of the modelled step (13.8 of 70.3 ms on the q4_1 set). It writes 993,280 bytes of logits, which the host reads back.

## dispatch:lx0@ux
Title: lx0 on the one-context image
Summary: lx0 is one of four instruction streams over a single xclbin; two RTP words tell the main cores to run the linear-layer program.
Spec: OPEN-DECODE-ONE-CONTEXT
Source: open_kernels/designs/layer_x/ux.py

In a set built from `ux.py`, the four layer streams `lx0`, `lx1`, `ax0` and `ax1` run over one image, and the manifest points all four at one context, `layer`. The main cores run one program for both layer types. Two numbers differ, so `lx0` writes them into each main core's RTP words at the head of the stream: 24 GEMV bands (16 qkv + 8 z) and 4 DeltaNet heads. A core reads them only after its first activation element arrives, and the stream sends that element after the writes.

The image has 15 cores: the eight main cores, the norm/router core at (0,3), the post core at (1,3), the glue core at (2,4) and four attention cores at (2..5,3). During `lx0` the attention cores sit blocked on their input stream. The glue core is at (2,4) because (2,3) is attention core 0 in this image.

All four streams take the same six buffers, `ptab` included, so the image has one kernel signature; `lx0` never touches `ptab`. The main cores use 16,256 of their 16,384 bytes of program memory.

## dispatch:lx1@ux
Title: lx1 on the one-context image
Summary: The merged image's MoE stream; its core program is byte for byte ax1's, and no context switch separates it from the next layer.
Spec: OPEN-DECODE-ONE-CONTEXT, OPEN-DECODE-PIPELINE
Source: open_kernels/designs/layer_x/ux.py

In `ux.py`, `lx1` and `ax1` run byte for byte the same core program; the two streams differ only in the `act` and `consts` offsets they read. One pass of a main core's program is a whole layer of either type, so after `lx1` every main core waits for the next layer's first input, whichever type that layer is.

Every layer runs in the `layer` context, so the host can always queue the next layer's first dispatch (`lx0` or `ax0`) behind `lx1` before it waits on anything (OPEN-DECODE-PIPELINE). The page draws that as the next dispatch starting the moment `lx1` ends.

## dispatch:ax0@ux
Title: ax0 on the one-context image
Summary: The same main-core program as lx0, told by its RTP words to run 18 bands and no DeltaNet heads; the attention cores do the attention.
Spec: OPEN-DECODE-ONE-CONTEXT, OPEN-ATTN-CONTEXT
Source: open_kernels/designs/layer_x/ux.py

`ax0` writes 18 GEMV bands (8 q + 8 gate + 1 k + 1 v per core) and 0 DeltaNet heads into the main cores' RTP words. With 0 heads the DeltaNet loop runs zero times, so no records and no state cross the streams. The skip is a loop count, not a branch.

The glue core (2,4) and the post core (1,3) stay blocked on their input streams for the whole dispatch. The attention input `ain` comes from shim column 5 and core 0's output `aout` goes to column 6, because in this image column 2 already feeds the glue core's `side` stream and column 1 already drains the post core's `pout`.

## dispatch:ax1@ux
Title: ax1 on the one-context image
Summary: The same core program as lx1 on the same image; the next layer's first dispatch can be queued behind it.
Spec: OPEN-DECODE-ONE-CONTEXT, OPEN-DECODE-PIPELINE
Source: open_kernels/designs/layer_x/ux.py

`ax1` is `ux.py`'s part 3, byte for byte the same core program as `lx1`, with the full-attention layer's offsets in its stream.

Because the following linear layer runs in the same `layer` context, the host queues its `lx0` behind `ax1` (OPEN-DECODE-PIPELINE). In the two-context layout this boundary costs a context switch instead.

## dispatch:lx0@lx
Title: lx0 on the two-context lx image
Summary: Linear-attention layers get their own xclbin (context lx) and full-attention layers a second one (ax).
Spec: OPEN-DECODE-ONE-CONTEXT, OPEN-QUANT-Q8
Source: open_kernels/designs/layer_x/lx.py

`lx.py` builds an image for linear-attention layers only, with 11 cores: the eight main cores, the norm/router core (0,3), the post core (1,3) and the glue core (2,3). It has no attention cores, its kernels take five buffers (no `ptab`), and the band and head counts are fixed at build time instead of read from RTP words.

A set with a q8 projection keeps this layout, because the merged image's main cores have room for one GEMV entry only (OPEN-DECODE-ONE-CONTEXT). The shipped Qwen3.6-35B set is one. Its main cores run `gemv_q8_gy`, and its `lx0` moves about 51 MB instead of 30 MB, because the `qkv`, `z` and out-projection weights are twice the q4_1 size.

## dispatch:lx1@lx
Title: lx1 on the two-context lx image
Summary: The MoE block on the lx image; when the next layer is full attention, the NPU must switch context before ax0 can start.
Spec: OPEN-DECODE-PIPELINE, OPEN-DECODE-ONE-CONTEXT
Source: open_kernels/designs/layer_x/lx.py

The core program and bytes match the one-context set: 18.0 MB, all experts at q4_1. What changes is what follows. The third linear layer of each group of four is followed by a full-attention layer, whose dispatches run in the `ax` context.

The host does not queue `ax0` behind `lx1` across that change: queueing across contexts hung the array in testing (OPEN-DECODE-PIPELINE). It waits for `lx1`, the NPU switches context, and only then `ax0` starts. Between two linear layers it still queues the next `lx0` behind `lx1`. The page charges each switch 0.95 ms; a step of this layout has 22 switches, the one-context layout 3.

## dispatch:ax0@ax
Title: ax0 on the two-context ax image
Summary: Full-attention layers on their own xclbin: 13 cores, attention input from shim column 2, and always a context switch before it.
Spec: OPEN-DECODE-ONE-CONTEXT, OPEN-QUANT-Q8
Source: open_kernels/designs/layer_x/ax.py

`ax.py` builds an image with 13 cores: the eight main cores, the norm/router core (0,3) and the four attention cores (2..5,3). There is no glue or post core. `ain` comes from shim column 2 and `aout` goes to column 1, and the band counts are fixed at build time.

In the shipped Qwen3.6-35B set the q, gate, k, v and o projections are q8 (`gemv_q8_gy`), so `ax0` moves about 35 MB instead of 18 MB. Every `ax0` in this layout follows a switch from the `lx` context, so the host never has it queued ahead.

## dispatch:ax1@ax
Title: ax1 on the two-context ax image
Summary: The MoE block on the ax image; the next layer or the final norm runs in another context, so a switch always follows.
Spec: OPEN-DECODE-ONE-CONTEXT, OPEN-DECODE-PIPELINE
Source: open_kernels/designs/layer_x/ax.py

The core program and bytes match `ax1` in the one-context set (18.0 MB). In this model every full-attention layer is followed by a linear layer or by `ln`, both in other contexts, so a context switch always follows `ax1` and nothing is queued behind it.

Of the step's 22 switches, 20 sit on either side of the ten full-attention layers. Removing those 20 is what the one-context image is for (OPEN-DECODE-ONE-CONTEXT).
