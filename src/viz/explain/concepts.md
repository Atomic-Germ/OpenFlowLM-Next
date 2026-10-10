## concept:array
Title: The NPU array
Summary: An XDNA2 NPU is 8 columns of tiles: a shim tile, a memory tile and four compute tiles each, joined by a stream network.
Source: open_kernels/viz/topology.py

The page draws the AMD XDNA2 (NPU2) array as it is laid out: 8 columns, each with a shim tile in row 0 that talks to DDR, a memory tile in row 1 and four compute tiles in rows 2–5, 32 compute cores in all. Tiles pass data over a switched stream network, and DMA engines move it between memory and the streams.

A kernel set's xclbin fixes which program each core runs and how the streams are routed. In this decode step no dispatch uses the memory tiles. The one-context layer image uses 15 of the 32 cores (the two-context images 11 and 13), the lm head 8 and the final norm 1.

Every connection on the page is an objectfifo, mlir-aie's name for a stream with a fixed element size and a fixed number of buffers (its depth). The producer fills an element and releases it; the consumer acquires it, uses it and releases it, which frees the buffer for the next one. A core that needs an element that has not arrived waits.

## concept:shim
Title: Shim tiles: the array's door to DDR
Summary: Row 0: each shim tile's DMA moves bytes between DDR buffers and the array over two channels in and two out; shims do no compute.
Source: open_kernels/ironutil.py

Nothing in the array reads DDR directly. Every byte comes in or goes out through a shim tile's DMA: per tile, two channels from DDR into the array (fills) and two back (drains), 16 + 16 over the 8 columns. The merged layer image uses 14 fills and 15 drains.

The instruction stream drives the shims. Each transfer is a buffer descriptor: which buffer argument, what offset, how many bytes, in what pattern. A channel queues at most 4 descriptors, so the streams keep at most 3 transfers in flight per channel and wait on the oldest before issuing another. A column's shim has 16 descriptors in all.

## concept:mem
Title: Memory tiles (row 1)
Summary: Row 1 holds one memory tile per column, for staging data between shims and cores; no dispatch in this decode step uses them.
Source: open_kernels/viz/topology.py

A memory tile is a larger local memory with its own DMA channels, sitting between the shim and the compute tiles. Designs use it to split, join or re-tile data on the way to the cores.

The decode kernels shown here do not use it. Every objectfifo in `lx0`, `lx1`, `ax0`, `ax1`, `ln` and `lm` runs straight from a shim tile to the compute tiles and back, so row 1 is idle for the whole step. The prefill kernels (the block GEMMs, the batched expert kernel and the block attention) do route their data through all eight memory tiles.

## concept:core
Title: Compute tiles and their programs
Summary: Rows 2–5: each core runs one program in an endless loop, blocking on its streams, and never sees where a dispatch ends.
Spec: OPEN-DECODE-ONE-CONTEXT
Source: open_kernels/designs/layer_x/xcommon.py

A compute tile is a vector processor with 64 KB of local data memory and 16 KB of program memory. Its program comes from the xclbin and calls kernel functions compiled from C++ (`gemv_q4_gy`, `dnx_row`, `attn_q` and so on); the page shows which one each core is in.

A core's program is a loop that acquires stream elements, calls kernels on them and releases them. It cannot see dispatch boundaries. In the layer image one pass of a main core's program is a whole layer, both dispatches, and it ends waiting for the next layer's first input. A dispatch only decides which data reaches the cores and when.

Program memory is the tight limit. The merged image's main cores use 16,256 of 16,384 bytes (OPEN-DECODE-ONE-CONTEXT), which is why each GEMV shape is one entry point with runtime parameters rather than a separate kernel per stage.

## concept:ddr
Title: DDR: where everything lives
Summary: System memory, shared by the CPU and the NPU, holds every weight, state and scratch buffer; the NPU reaches it only through shim DMA.
Source: src/open_qwen36/core.cpp

The model's weights, its per-layer state and the scratch vectors all sit in DDR as XRT buffer objects. The CPU reads and writes them through its own mapping, the NPU only through the shim tiles. A compute tile holds 64 KB, so weights stream through the cores and are never kept there.

One decode step of the q4_1 Qwen3.6-35B set moves about 2.34 GB between DDR and the array over 82 dispatches; the shipped q8 set moves about 3.14 GB. Most of it is weights: the attention projections, nine experts per layer and the 540 MB lm head. The rest of each layer's 512 MiB pool, mostly the 248 experts this token did not pick, is not read.

The page draws the DDR buffers as a band under the shims. Its durations are modelled from these byte counts (see Modelled timing).

## concept:ctx
Title: Hardware context switch
Summary: Each xclbin runs in its own hardware context; a dispatch from a different xclbin makes the NPU switch, which costs about 0.95 ms.
Spec: OPEN-DECODE-ONE-CONTEXT, OPEN-DECODE-PIPELINE
Source: src/open_qwen36/core.cpp

The engine opens one hardware context per xclbin the manifest names under `contexts`, and each kernel runs in the context the manifest assigns it. When two consecutive dispatches belong to different contexts, the NPU switches the array from one image to the other between them.

OPEN-DECODE-ONE-CONTEXT measured the two-context penalty at about 0.95 ms per affected `ax0` call (on the 27B), and the page charges every switch that much. The one-context set switches 3 times a step: into the layer context (the previous step ended in the lm head's), into `ln` and into `lm`. The two-context set switches 22 times.

A switch also stops submit-ahead. The host queues a dispatch behind the running one only within one context; queueing across a switch hung the array in testing (OPEN-DECODE-PIPELINE).

## concept:modelled
Title: Modelled timing
Summary: The structure on this page comes from the build; every duration is modelled from bytes moved, not measured.
Spec: OPEN-DECODE-ONE-CONTEXT, OPEN-DECODE-PIPELINE
Source: open_kernels/viz/timeline.py

**Real, from the build:** tile placement and what each core runs, every data path with its element size and depth, each dispatch's DMA transfers in issue order with their byte counts, and the dispatch order and route steps from `manifest.json`. The other host stages come from a hand-kept list that mirrors `core.cpp`.

**Modelled:** every duration.
- All DDR transfers share one link at an effective 40.1 GB/s. That figure is fitted so the Qwen3.6-35B one-context set's modelled step equals the 70.3 ms median measured in OPEN-DECODE-ONE-CONTEXT (position 1, 2026-09-22). Other sets reuse it without a fit of their own; the shipped two-context 35B set then models 108 ms, against 105-108 ms measured for a two-context 35B set in OPEN-DECODE-PIPELINE.
- A core call whose data is already in tile memory takes 0.25 µs; the cores' arithmetic is not otherwise modelled.
- A context switch costs 0.95 ms: OPEN-DECODE-ONE-CONTEXT's two-context `ax0` penalty, measured on the 27B. The route gap is 0.045 ms, inside OPEN-DECODE-PIPELINE's measured 0.02–0.07 ms. The other host stages use nominal times.
- Each layer type is simulated once, at position 1, and every layer of that type is drawn with the same timing.

**Illustrative:** the routed experts are a random choice of 8 per layer; the real ones depend on the token. `ax0`'s KV window is sized for position 1 the way the host patches it (`stream_patch::attn_apply`: 3 rows at block size 4), so its attention cores see the rows they expect.

## concept:gpu
Title: GPU: not used
Summary: oflm runs nothing on the GPU; its lane is drawn only to show that it stays idle for the whole step.

Every model computation in this step runs on the NPU. The CPU does only the small host stages between dispatches: the embedding lookup, the two stream patches, the logits readback and sampling.

The integrated GPU does no work for oflm, so its lane stays empty.

## concept:submit_ahead
Title: Submit-ahead across the layer boundary
Summary: The host queues the next layer's first dispatch behind the current layer's last one, so the NPU starts it without waiting on the host.
Spec: OPEN-DECODE-PIPELINE
Source: src/open_qwen36/core.cpp

Run serially, every dispatch is `start(); wait();`, and the array sits idle for a host turnaround each time. Instead `Core::step_impl` starts layer l + 1's first dispatch before it waits on layer l's last one, and the device takes it up the moment the previous command retires.

Two rules limit it. An instruction stream is never patched while a run on it may still be executing, and `lx1` / `ax1` must be patched with the router's choice, which exists only after `lx0` / `ax0` finishes; so the MoE dispatch is never queued ahead. And queueing happens only within one hardware context (`OFLM_OPEN_SUBMIT_AHEAD=1`, the default); level 2, across contexts, hung the array.

Measured on the 35B, it saves about 3 ms of median step time (about 105 against 108 ms at position 1). The ceiling is small because the host gap is only 0.02–0.07 ms per dispatch. The page draws it as the next layer's first dispatch starting the moment the previous one ends.

## buffer:pool
Title: pool: a layer's weights
Summary: One per layer, 512 MiB: all 256 experts, the shared expert and the attention projections, packed at load in the order the cores read them.
Spec: OPEN-PACK-PLAN
Source: src/open_qwen36/pools.cpp

Each of the 40 layers has its own `pool`. At load the engine packs it from the model file by the manifest's pack plan: the routed experts' up and gate projections interleaved in stripes, their down projections, the shared expert, and the layer's attention projections (`qkv` and `z` in a linear layer; q, gate, k, v and o in a full one). Weights sit in the band order the main cores consume, so a dispatch reads each one as a few long transfers.

A decode step reads little of it. A linear layer reads about 33 MB of its 512 MiB at q4_1: the projections and nine experts at 1.97 MB each. The other 248 experts are not touched until a token picks them. The pool is 536,870,912 bytes in the q4_1 set and 540,016,640 in the shipped q8 set.

## buffer:consts
Title: consts: a layer's small weights
Summary: One per layer: norm weights, the router, the shared-expert gate and, in linear layers, the DeltaNet parameters and the out projection.
Spec: OPEN-PACK-PLAN
Source: src/open_qwen36/pools.cpp

A linear-attention layer's `consts` (11.9 MB) holds `input_layernorm`, the alpha and beta projections, `ssm_a`, the `dt` bias, the convolution taps, `ssm_norm`, `post_attention_layernorm`, the router (`moe_router`, 256 × 2,048 bf16, 1 MiB), `shared_expert_gate` and the out projection `ssm_out_proj`. A full-attention layer's (1.06 MB) holds the two layer norms, `q_norm`, `k_norm`, the router and `shared_expert_gate`.

The norm/router core streams the norm weights and the router, the glue core streams the DeltaNet parameters over `side`, and the main cores stream the out projection. `lx1` and `ax1` read the shared-expert gate from here.

## buffer:state
Title: state: DeltaNet state or KV cache
Summary: One per layer: a fixed 2.3 MB recurrent state in linear layers, a KV cache that grows by 2 KB per token in full-attention layers.
Spec: OPEN-ATTN-CONTEXT
Source: src/open_qwen36/core.cpp

In a linear-attention layer `state` is 2,342,912 bytes: the convolution state (the last 3 rows of the 8,192 `qkv` channels, bf16, 48 KB) and one DeltaNet matrix per value head (32 heads, 128 × 128 fp32, padded to 140 rows: 71,680 bytes each). `lx0` updates it in place every token, and its size does not change with the context. `Core::reset` zeroes it.

In a full-attention layer `state` is the KV cache: one 2,048-byte row per position (K then V, 2 KV heads × 256 dims, bf16), allocated for the whole context. `ax0` reads the cached rows and writes the new one.

## buffer:act
Title: act: a layer's scratch vectors
Summary: One per layer: the intermediate vectors that pass between cores within a layer, including the router record.
Source: open_kernels/designs/layer_x/xlayer.py

Cores pass most intermediate results through DDR: a GEMV's bands land in `act`, and the next stage streams them back in. `act` holds the normalized input, the projection outputs, the DeltaNet records and output, the gated output, the attention output, the post-attention residual and norm, the expert hidden vector and the router record. It is 190,464 bytes in a linear layer and 98,304 in a full one.

The router record links a layer's two dispatches. `lx0` or `ax0` writes it (at byte 176,128 in a linear layer, 83,968 in a full one), the host reads the eight expert ids from it, and `lx1` or `ax1` reads the mixing weights from it.

## buffer:xres
Title: xres: the residual stream
Summary: fp32[2,048], shared by all layers: the embedding goes in, each layer reads it and writes its output back, and the final norm reads it.
Spec: OPEN-DECODE-PIPELINE
Source: src/open_qwen36/core.cpp

`xres` is the one buffer every layer writes. The host writes the token's embedding into it. Each layer's first dispatch reads it twice (for the input norm and for the residual add), and the layer's MoE dispatch writes the layer's output back into it. `ln` reads it at the end.

Layer l + 1's first dispatch reads what layer l's last one wrote. When the host queues ahead, only the in-order command queue of one hardware context keeps those two in order, and OPEN-DECODE-PIPELINE's bit-exact gate is the evidence that it does.

## buffer:ptab
Title: ptab: the RoPE position table
Summary: One 1 KB record per position (position, row count, RoPE cos and sin), built at load; ax0 reads one row per step.
Spec: OPEN-ATTN-CONTEXT
Source: src/open_qwen36/pools.cpp

The engine computes `ptab` at load, one row per context position, from the manifest's RoPE frequencies. Each row holds the position, the number of cached rows to walk, and the cos and sin of the 32 RoPE frequencies that rotate 64 of each head's 256 dimensions.

`ax0` streams one row per step to the attention cores, and the attnpos stage points it at the current position. Linear-attention layers never read it.

## buffer:lmpool
Title: lmpool: the lm head weights
Summary: The 248,320 × 2,048 lm_head matrix at q8, about 542 MB, packed at load in the 128-row band order lm reads.
Spec: OPEN-PACK-PLAN
Source: src/open_qwen36/pools.cpp

`lmpool` is a global buffer of 542,113,792 bytes. At load the engine packs `lm_head.weight` into it: 1,940 bands of 128 rows, each band 32 q8 chunks of 8,704 bytes, 540,344,320 bytes in use. The chunks follow the lm head's band order (OPEN-PACK-PLAN), so each core reads all of its bands as one long transfer.

`lm` reads all of it every step. It is the largest single read of the step.

## buffer:hn
Title: hn: the final hidden vector
Summary: bf16[2,048], written by ln and broadcast once to the eight lm head cores.
Source: open_kernels/designs/ln/ln.py

`ln` writes the normalized final hidden state here, 4 KB.

`lm` reads it once over its broadcast stream, and every core uses it for all of its bands.

## buffer:logits
Title: logits: one score per vocabulary entry
Summary: fp32[248,320], written by lm in 128-row bands and read back by the host every step.
Source: src/open_qwen36/core.cpp

Each lm core drains its bands' results here, 512 bytes per band. After `lm` finishes, the host reads the whole buffer (993,280 bytes) back.

Entries past 248,070 come from padding rows of the matrix; the engine sets them to −∞ before sampling.

## buffer:normw
Title: normw: the final norm weight
Summary: bf16[2,048] model.norm.weight, copied from the model file at load and read by ln.
Source: src/open_qwen36/core.cpp

`normw` holds the weight of the model's final RMS norm, 4 KB, copied from the model file when the engine loads.

`ln` streams it in once per step, after the residual and the zero vector.

## buffer:zero
Title: zero: an all-zero vector
Summary: fp32[2,048] of zeros, passed to ln as the value to add, because the final norm has no residual left to add.
Source: open_kernels/designs/ln/ln.py

`ln`'s kernel is the layer norm with a fused residual add: `y = x + add`, then the norm of `y`.

At the end of the model the last layer has already added its residual, so the tail passes this zero buffer as `add`. The engine allocates it at load and never writes it.

## buffer:xresf
Title: xresf: the residual after the last layer
Summary: fp32[2,048] written by ln as xres + 0; nothing later in the decode step reads it.
Source: open_kernels/designs/ln/ln.py

The norm kernel always writes the sum it normalized. With `add` set to zero, that is a copy of `xres`.

The decode step does not use it; it exists because the kernel's output has to go somewhere.

## layer:linear_attention
Title: Linear-attention layer (Gated DeltaNet)
Summary: 30 of the 40 layers: recurrent attention with a fixed-size state per head, then the MoE block; two NPU dispatches.
Spec: OPEN-FAMILY-QWEN36MOE
Source: open_kernels/designs/layer_x/xlayer.py

Qwen3.6-35B-A3B repeats three linear-attention layers and one full-attention layer, ten times. A linear-attention layer replaces softmax attention with Gated DeltaNet: each of its 32 value heads keeps a 128 × 128 state matrix that every token decays and updates. Its cost does not grow with the context.

On the NPU it is two dispatches with one host stage between them:
- `lx0`: input norm, `qkv` and `z` projections, convolution, DeltaNet, gated norm, out projection, residual norm, router;
- route: the host points `lx1` at the chosen experts;
- `lx1`: the eight routed experts and the shared expert; the result is the new residual.

## layer:full_attention
Title: Full-attention layer
Summary: 10 of the 40 layers (every fourth): gated softmax attention over a growing KV cache, then the MoE block; two NPU dispatches.
Spec: OPEN-ATTN-CONTEXT
Source: open_kernels/designs/layer_x/xlayer.py

A full-attention layer has 16 query heads sharing 2 key/value heads, head size 256, RoPE on 64 of those dimensions, RMS norm on q and k, and a sigmoid gate on its output. Its KV cache grows by one 2 KB row per token, so `ax0` is the one dispatch whose cost grows with the context (OPEN-ATTN-CONTEXT).

On the NPU it is two dispatches with one host stage between them:
- `ax0`: input norm, q/gate/k/v projections, attention, o projection, residual norm, router;
- route: the host points `ax1` at the chosen experts;
- `ax1`: the eight routed experts and the shared expert; the result is the new residual.

## fifo:w*
Title: w0–w7: the weight streams
Summary: One stream per main core from its own shim column, carrying weights and other bulk data in 10 KB elements, double-buffered.
Source: open_kernels/designs/layer_x/xcommon.py

In the layer image there is one weight stream per column: `w3`, for example, runs from shim column 3 to main core (3,2). Elements are 10,240 bytes (two 5,120-byte q4_1 chunks, or two q8 half-tiles of the same size) with depth 2, so the shim fills the next element while the core works on the current one. Each stream carries the projection weights, the MoE header, the experts' weights and, in `lx0`, the DeltaNet records and state slices.

In `lm`, `w0`–`w7` come two per shim tile from columns 0–3 and feed the cores in columns 0–1, in 17,408-byte elements (two q8 chunks).

## fifo:x
Title: x: the broadcast activation
Summary: One stream from a single shim to all eight main cores, carrying the activation vector every core multiplies against.
Source: open_kernels/designs/layer_x/xcommon.py

In the layer image `x` runs from shim column 1 to all eight main cores in 4 KB elements, depth 2. It carries the normalized layer input, the gated attention or DeltaNet output (two elements, 4,096 values), the post-attention norm output for the MoE, and, once per expert, the 512-value hidden vector.

In `lm`, `x` runs from shim column 4 with depth 1 and carries `hn` once.

## fifo:y*
Title: y0–y7: the result streams
Summary: One stream per main core back to its shim column; each element is one 64-row band of GEMV results.
Source: open_kernels/designs/layer_x/xcommon.py

In the layer image there is one result stream per column: `y3`, for example, runs from main core (3,2) to shim column 3. Elements are 256 bytes (64 fp32), depth 2. Each stream carries GEMV band results, the updated DeltaNet state rows and outputs in `lx0`, each core's part of the expert hidden vector, and in `lx1` / `ax1` the layer's output into `xres`.

In `lm`, each element is 512 bytes, the 128 logits of one band.

## fifo:lni
Title: lni: into the norm/router core
Summary: Shim (0,0) to core (0,3), 4 KB elements: the residual, norm weights, the attention block's output, then the 1 MiB router weight.
Source: open_kernels/designs/layer_x/xlayer.py

`lni` feeds the norm/router helper at (0,3), depth 5. For the input norm it carries `xres` (two elements) and `input_layernorm`. For the residual step it carries `xres`, `post_attention_layernorm` and the attention block's projected output (two elements).

Then it streams the router weight, 256 elements of 4 KB, which the core multiplies against the normalized vector.

## fifo:lno
Title: lno: out of the norm/router core
Summary: Core (0,3) to shim (0,0), 4 KB elements: the normalized input, the new residual and its norm, then the router record.
Source: open_kernels/designs/layer_x/xlayer.py

`lno` (depth 3) carries the norm/router core's results into `act`: the normalized layer input, then the post-attention residual (two elements) and its normalized form.

Last it carries the router record: the probabilities, the top-8 expert ids and their weights, which the host reads and `lx1` / `ax1` consume.

## fifo:gact
Title: gact: into the glue core
Summary: Shim column 3 to the glue core, 2 KB elements: qkv tiles and the convolution state rows.
Source: open_kernels/designs/layer_x/xlayer.py

`gact` (depth 5) feeds the glue core, one tile of 1,024 channels at a time: two elements of the `qkv` projection output and the three stored convolution rows for those channels.

The glue core runs the 4-tap convolution on them.

## fifo:gout
Title: gout: out of the glue core
Summary: Glue core to shim column 2, 2 KB elements: the shifted convolution state and the per-head DeltaNet records.
Source: open_kernels/designs/layer_x/xlayer.py

`gout` (depth 3) writes back the three updated convolution rows of each tile into `state`.

For the value tiles it also carries the per-head records (k, q, v, decay, beta; 2 KB per head) into `act`, which the main cores read for the DeltaNet update.

## fifo:side
Title: side: the glue core's side channel
Summary: Shim column 2 to the glue core, 4 KB elements: the normalized input and the alpha/beta, decay and convolution parameters.
Source: open_kernels/designs/layer_x/xlayer.py

`side` (depth 2) gives the glue core its own copy of the normalized layer input from `act`. Then, from `consts`, it carries 64 weight tiles of the alpha and beta projections, the small per-head parameters and the convolution taps.

From them the glue core works out each head's decay and beta.

## fifo:pin
Title: pin: into the post core
Summary: Shim column 4 to the post core (1,3), 4 KB elements: the ssm_norm weight, then the DeltaNet output and z, 8 heads at a time.
Source: open_kernels/designs/layer_x/xlayer.py

`pin` (depth 2) first carries the `ssm_norm` weight. Then, for each group of 8 heads, it carries the DeltaNet output and the matching slice of `z`.

The post core normalizes each head, scales it and gates it with SiLU of `z`.

## fifo:pout
Title: pout: out of the post core
Summary: Post core (1,3) to shim column 1, 2 KB elements: the gated output, 8 heads per element.
Source: open_kernels/designs/layer_x/xlayer.py

`pout` (depth 2) writes the gated output into `act`, 8 heads of 128 bf16 values per element.

The main cores then read it back over `x` as the input of the out projection.

## fifo:ain
Title: ain: into the attention cores
Summary: One shim to the four attention cores, 1 KB elements: norm weights, position record, q, k, v, the cached K/V rows and the gate.
Source: open_kernels/designs/layer_x/xlayer.py

`ain` is a broadcast stream, depth 10, from shim column 5 in the one-context image and column 2 in the two-context one. In order it carries the q/k norm weights, this position's `ptab` record, q, k, v, the cached K/V rows (two elements per row) and the output gate.

Every attention core receives every element, and each computes the output only for the 4 heads it owns. The depth lets a core hold a whole block of 4 cached rows at once.

## fifo:aout
Title: aout: out of attention core 0
Summary: Attention core 0 (2,3) to a shim, 1 KB elements: this token's new K/V row into the cache, then core 0's output heads.
Source: open_kernels/designs/layer_x/xlayer.py

`aout` (depth 2) goes to shim column 6 in the one-context image and column 1 in the two-context one. It first writes this token's K row and V row into the KV cache, at the row the attnpos stage chose.

Then it carries attention core 0's gated output for its 4 heads into `act`.

## fifo:og*
Title: og1–og3: attention outputs
Summary: Attention cores 1–3 each drain their gated output heads to the shim in their own column, 1 KB elements.
Source: open_kernels/designs/layer_x/xlayer.py

Each of attention cores 1–3, at (3,3), (4,3) and (5,3), owns 4 of the 16 query heads and writes their gated output through its own stream (depth 2) to the shim below it, into `act`.

Core 0 uses `aout` for the same job. The main cores read the four parts back as the input of the o projection.

## fifo:in
Title: in: into the final norm core
Summary: ln's input stream, shim (0,0) to core (0,2), 4 KB elements: the residual, the zero vector and the final norm weight.
Source: open_kernels/designs/ln/ln.py

`in` (depth 5) carries `xres` (two elements), `zero` (two elements) and `normw` (one element).

The core holds all five at once and runs one call of the norm.

## fifo:out
Title: out: out of the final norm core
Summary: ln's output stream, core (0,2) to shim (0,0), 4 KB elements: the sum into xresf and the normalized vector into hn.
Source: open_kernels/designs/ln/ln.py

`out` (depth 3) carries the sum `xres + zero` (two elements) into `xresf`, then the normalized bf16 vector into `hn`.

`lm` then reads `hn`.

## concept:host
Title: The host CPU
Summary: The CPU runs oflm itself: it queues every dispatch, patches instruction streams between them and turns the logits into a token.
Spec: OPEN-DECODE-PIPELINE, OPEN-DECODE-ONE-CONTEXT
Source: src/open_qwen36/core.cpp

In a decode step on the open kernels the CPU does no layer arithmetic. It copies the token's embedding row into `xres`, patches the `ax0` stream for the position, waits for each layer's first dispatch, reads the router's top-8 and points the MoE stream at those experts, and at the end reads the logits back and samples. Each of those is a host block on the CPU lane.

Between those blocks the CPU waits on the NPU, passively: the engine's host threads run with `OMP_WAIT_POLICY=PASSIVE`, and a run without it is not a valid timing (OPEN-DECODE-PIPELINE). Where two dispatches run in the same hardware context the next one is already queued behind the current one (OPEN-DECODE-PIPELINE), so the array does not wait for the host to submit it.
