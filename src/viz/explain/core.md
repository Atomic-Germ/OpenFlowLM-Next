## core:attn_fin
Title: Attention output: normalise and gate
Summary: Finishes two heads of attention: divides by the softmax sum, multiplies by sigmoid(gate), writes 1 KB of bf16 output.
Spec: OPEN-ATTN-CONTEXT
Source: open_kernels/designs/attn/attn_fin.cc

The last attention step of a full-attention layer, in `ax0`, on the four attention cores (columns 2-5, row 3). Each core owns 4 of the 16 query heads and calls this twice per token, once per pair of heads. Per head it computes `og = (o / l) * sigmoid(gate)`, where `o` is the weighted sum of values in `oacc` and `l` the softmax denominator in `ml`, and stores the result as bf16: one 1 KB og element (2 heads x 256 dims).

The gate comes from the main cores' gate projection. It arrives on the broadcast `ain` stream as two 1 KB elements per og element, for every head; each core reads all of them and computes only its own. Core 0 sends its og out through `aout` (the stream that also carries the new KV cache row); cores 1-3 use `og1`-`og3`. The main cores read the whole og (bf16[4096]) back as the input of the o projection.

`1/l` uses an integer routine (`srecip_rn`, `scalar_fp.h`) in place of the compiler's soft-float divide. The spec lists this among the changes that freed program memory for the 4-row block kernel. Math: `attn_fin_impl` in `attn.h`.

## core:attn_init
Title: Attention: reset the softmax state
Summary: Clears this core's output accumulator and sets each head's running max to -1e30 and running sum to 0 before the cache walk.
Spec: OPEN-ATTN-CONTEXT
Source: open_kernels/designs/attn/attn_init.cc

Called once per token on each attention core (columns 2-5, row 3) in `ax0`, after q, k and v are prepared and before any cached row is scored. It zeroes `oacc` (this core's 4 heads x 256 fp32) and sets up `ml`, the online-softmax state: running max `m = -1e30` and running sum `l = 0` per head. The padding lanes past the core's 4 heads get the same values, so they exponentiate to zero.

It also sets the vector rounding mode (round to nearest even) once for the whole walk. The source notes this used to be written once per position per layer, about 82k times a token. Math: `attn_init_impl` in `attn.h`.

## core:attn_k
Title: Attention: norm and rotate the new key
Summary: Turns one projected key head into cache form: RMSNorm times the k-norm weight, RoPE on the first 64 dims, then bf16.
Spec: OPEN-ATTN-CONTEXT
Source: open_kernels/designs/attn/attn_k.cc

Runs in `ax0` on all four attention cores (columns 2-5, row 3), twice per token: once per KV head (the 35B has 2). The input is one 1 KB `ain` element holding one fp32 key head (256 values) from the main cores' k projection. It applies RMSNorm over the head with the `kn` weight, rotates dims 0-63 with the position's cos/sin (pairs i and i+32), and stores the result as bf16 in `kout`, the K half of the new cache row.

Every attention core keeps the new row in local memory, because the last block of the cache walk (`attn_stepb_new`) reads it from there. Core 0 also copies `kout` and `vout` out through `aout`, and the shim DMA writes them to cache row `pos`. Math: `attn_k_impl` and `norm_rope` in `attn.h`.

## core:attn_meta
Title: Attention: read norm weights and position
Summary: Unpacks the layer's q/k norm weights, this token's RoPE cos/sin and the position counters that drive the cache walk.
Spec: OPEN-ATTN-CONTEXT
Source: open_kernels/designs/attn/attn_meta.cc

The first call on each attention core (columns 2-5, row 3) in `ax0`, once per token. It reads two 1 KB `ain` elements: the layer's meta element (`qn` and `kn`, the bf16[256] q/k RMSNorm weights) and the position record for this token (`pos`, `nf`, then 32 cos and 32 sin values for the 64 rotated dims). It copies them into the local buffers `qn`, `kn` and `cs`.

It fills the parameter block `pb`: `pb[0] = pos` (rows already in the cache), `pb[1] = nf`, two zeroed counters, and `pb[4] = pos / 4`, the number of full 4-row blocks the core takes off the fifo before the last one. The host streams `4 * (pos / 4 + 1) - 1` cached rows to match. Math: `attn_meta_impl` in `attn.h`.

## core:attn_q
Title: Attention: norm, rotate and split the query
Summary: Prepares one query head: RMSNorm times qn, RoPE, then a bf16 hi/lo pair pre-scaled by 1/16 for the score loop.
Spec: OPEN-ATTN-CONTEXT
Source: open_kernels/designs/attn/attn_q.cc

Runs in `ax0` on all four attention cores (columns 2-5, row 3), 16 times per core per token: one call per query head, each on a 1 KB `ain` element holding one fp32 head (256 values). The stream is broadcast, so every core prepares all 16 heads into `qs`; each core then scores only its own 4.

Per head: RMSNorm with the `qn` weight, RoPE on dims 0-63, then each fp32 value is split into two bf16 halves (hi, lo). A score is fp32 q against bf16 K, done as two bf16 MACs. q does not change during the walk, so the split happens once here instead of once per cached row. The 1/sqrt(256) = 1/16 scale is folded into both halves; at head dim 256 that is a power of two, an exponent shift, so the scores are bit-identical.

The host sends q ahead of gate, k and v, so this runs while the main cores are still projecting gate, k and v. Math: `attn_q_impl` in `attn.h`.

## core:attn_stepb
Title: Attention: score a block of 4 cached rows
Summary: Online-softmax attention over 4 KV cache rows at once for this core's 4 heads: scores, one vector exp, rescale, add V.
Spec: OPEN-ATTN-CONTEXT
Source: open_kernels/designs/attn/attn_stepb.cc

The cache walk of a full-attention layer, in `ax0` on the four attention cores (columns 2-5, row 3). Each call takes 8 `ain` elements: 4 cached rows, each a K half and a V half of 1 KB (2 KV heads x 256 bf16). For the core's 4 heads it computes the 16 dot products q.K as one reduction tree, takes the block's max per head, updates the running max and sum with a single 32-lane vector exponential, rescales `oacc` only if the max grew, and adds the weighted V rows into `oacc`.

It runs `pos / 4` times per core per token, so its cost grows with the context. Below position 4 there is no full block and it is not called; it still appears in the core's function list because it is linked into the image.

Blocking 4 rows per call pays for the q loads, the accumulator rescale and the exponential once per block instead of once per row. The spec records `ax0` at position 4000 going from 4.32 ms with one row per call to 2.15 ms with this kernel and its companion changes. Math: `attn_rowb_impl` in `attn.h`.

## core:attn_stepb_new
Title: Attention: last block, with the new row
Summary: The final block of the cache walk: 3 cached rows from the stream plus this token's own K/V row from core memory.
Spec: OPEN-ATTN-CONTEXT
Source: open_kernels/designs/attn/attn_stepb_new.cc

Called once per token on each attention core (columns 2-5, row 3) in `ax0`, after the full blocks. It does what `attn_stepb` does for one 4-row block, but the block is 3 rows from `ain` plus, in the last slot, the new position's k'/v' that `attn_k` and `attn_v` left in `kout` and `vout`. The new row comes from local memory, not from the cache.

Cached plus new rows are padded to a whole number of 4-row blocks. `nv = pos - rows already consumed` says how many of the 3 fifo slots are real; the rest get a score of -1e30, which never wins the max and exponentiates to zero. Below position 4 this is the only scoring call.

It is a separate symbol rather than an argument to `attn_stepb` because IRON type-checks memref arguments per external function, and the scratch rows are a different type from a fifo element. Sending every row through the block kernel, with no single-row kernel built, is what let the 4-row block fit in the core's 16 KB of program memory. Math: `attn.h`.

## core:attn_v
Title: Attention: store the new value row
Summary: Converts one projected value head from fp32 to bf16 for the new KV cache row; values get no norm and no rotation.
Source: open_kernels/designs/attn/attn_v.cc

Runs in `ax0` on all four attention cores (columns 2-5, row 3), twice per token, once per KV head. The input is one 1 KB `ain` element with one fp32 value head (256 values) from the main cores' v projection; the output is its bf16 copy in `vout`, the V half of the new cache row.

As with `attn_k`, every core keeps the row for its last block (`attn_stepb_new`), and core 0 also sends it out through `aout` to cache row `pos`. Math: `attn_v_impl` in `attn.h`.

## core:dnx_delta
Title: DeltaNet: the correction term for one head
Summary: Once per head between the two passes over the state: delta = beta * (v - decay * t), stored as bf16 hi/lo; resets o.
Spec: OPEN-DECODE-ONE-CONTEXT
Source: open_kernels/designs/layer_x/gen_kernels.py

Part of the Gated DeltaNet step that the eight main cores (row 2, one per column) run in `lx0`, between the q|k|v|z projection and the out projection. Each core owns 4 of the 32 value heads. After `dnx_pass1` has formed `t = S^T k` from the old state, this call reads v, decay and beta from the head's record in the `ds` scratch and computes `delta = beta * (v - decay * t)` for the head's 128 columns.

It stores delta and decay as bf16 hi/lo pairs and zeroes the output accumulator `o` for pass 2 (`dnx_row`). The core has no fp32 vector multiply, so each fp32 product is split into bf16 halves and done as MACs. It touches no stream, only `ds`.

The `.cc` is generated by `gen_kernels.py`; the math is `dnx_delta_head` in `layer_x/dnx.h`.

## core:dnx_ofin
Title: DeltaNet: emit one head's output
Summary: Scales half of a head's output o by 1/sqrt(128) and writes it as one 256 B y element; two calls per head.
Spec: OPEN-DECODE-ONE-CONTEXT
Source: open_kernels/designs/layer_x/gen_kernels.py

The last DeltaNet call per head on each main core (row 2) in `lx0`. Pass 2 (`dnx_row`) has accumulated `o = S'^T q` for the head; two calls each take 64 of its 128 values, multiply them by 1/sqrt(128) (as a bf16 hi/lo pair, no scalar float) and write them to one 256 B element of the core's `y` stream.

The host DMA collects them in `act`, and the post core (`post_fn`) reads them as o. Per layer: 2 calls x 4 heads per core, 64 across the array. Math: `dnx_ofin_half` in `layer_x/dnx.h`.

## core:dnx_pass1
Title: DeltaNet pass 1: S^T k over a state slice
Summary: Reads a 20-row slice of a head's 128x128 fp32 state and adds its share of t = S^T k; 7 slices per head.
Spec: OPEN-DECODE-ONE-CONTEXT
Source: open_kernels/designs/layer_x/gen_kernels.py

The first pass of the Gated DeltaNet step on the main cores (row 2) in `lx0`. A head's state S is 128 x 128 fp32 (64 KB), too big for core memory, so it streams through the `w` fifo in 10 KB elements of 20 rows. 7 slices cover 140 rows; the last 12 are zero padding in DDR and add nothing. Each call computes `t[j] += sum_i k[i] * S[i][j]` over its 20 rows. It walks two columns per pass, one from each 64-column half, because one column at a time was bound by the latency of its accumulation chain.

On slice 0 it also splits the head's k and q (from the record `dnx_vcopy` copied in) into bf16 hi/lo tables and zeroes t. Results stay in the `ds` scratch. 7 calls x 4 heads per core per layer, 224 across the array. S streams through a second time for pass 2.

Math: `dnx_pass1_slice` in `layer_x/dnx.h`.

## core:dnx_row
Title: DeltaNet pass 2: update the state
Summary: Writes S' = decay*S + k*delta for a 20-row slice, accumulates o += S'^T q, and streams S' out half a row per call.
Spec: OPEN-DECODE-ONE-CONTEXT
Source: open_kernels/designs/layer_x/gen_kernels.py

The second pass over a head's state on the main cores (row 2) in `lx0`. The slice arrives again through `w` (10 KB, 20 rows). The first call on a slice updates all 20 rows in place, `S'[i] = decay * S[i] + k[i] * delta`, then adds `S'[i] * q[i]` into o. Every call, the first included, copies one half row (64 floats) of S' into a 256 B `y` element, and the shim DMA writes it back over the state in DDR.

That is 40 calls per slice, 280 per head and 1,120 per core per layer, the most frequent call in a linear layer. The 256 B output element (half a row) sets the call count; the arithmetic happens in each slice's first call.

The slice update runs the two 64-column halves side by side so one recurrence chain's latency hides the other's. It computes S' and o in two loops because one loop kept more accumulators live than the core has and spilled them. The source keeps two chains, not four, because four would cost about 580 B more code and the main core has about 370 B of program memory left. Math: `dnx_row_half` and `dnx_slice_update` in `layer_x/dnx.h`.

## core:dnx_vcopy
Title: DeltaNet: load one head's record
Summary: Copies a head's 2 KB record (k, q, v, decay, beta) out of its stream element into local scratch so it can be released.
Spec: OPEN-DECODE-ONE-CONTEXT
Source: open_kernels/designs/layer_x/gen_kernels.py

Starts each DeltaNet head on the main cores (row 2) in `lx0`. The glue core (`glue_emit_fn`) built one fp32[512] record per head: k (128), q (128), v (128), then decay at 384 and beta at 385. It reaches the main core as the first 2 KB of a 10 KB `w` element, and this call copies those 512 floats into the `ds` scratch.

The copy is there because a fifo release frees the oldest element held: the core cannot keep the record element while it acquires and releases the 14 state slices that follow. 4 calls per core per layer, one per head.

In the one-context image the head count is a runtime word written by the instruction stream: 4 in a linear layer and 0 in a full-attention layer, where the DeltaNet loop runs zero times. Math: `layer_x/dnx.h`.

## core:gemv_q4_gdown
Title: MoE down projection (q4_1)
Summary: One expert's down projection for this core's 256 output rows, against the expert hidden h; 8 calls per expert slot.
Source: open_kernels/designs/layer_x/gen_kernels.py

Runs in the MoE half (`lx1` / `ax1`) on the eight main cores (row 2), once per 10 KB `w` element. After `gemv_q4_prep_h` has built the table of the expert's hidden h (512 values), the core computes its 256 rows of the 2048-wide output, `yd = W_down[rows] @ h`, into the `ms` scratch at offset 672.

The kernel picks the weight layout from the slot index: routed experts (slots 0-7) arrive as two 128-row bands of 4 elements, the shared expert (slot 8) as four 64-row bands of 2 elements. Either way it is 8 elements per slot, 72 calls per core per layer. `moe_accfin` then adds `yd` into the block output.

Each element is two 5120-byte q4_1 chunks (32 rows x 256 K). Routed and shared experts share one 9-iteration loop, with the layout and the combine chosen inside the kernels, because the main core's 16 KB of program memory has no room for separate call sites. Tile arithmetic: `designs/gemv_q4/gemv_q4.h`.

## core:gemv_q4_gup
Title: MoE up and gate projections (q4_1)
Summary: Computes this core's 64 rows of an expert's up (band 0) or gate (band 1) projection against xm; 16 calls per expert slot.
Spec: OPEN-QUANT-Q8
Source: open_kernels/designs/layer_x/gen_kernels.py

Runs in the MoE half (`lx1` / `ax1`) on the eight main cores (row 2). An expert's hidden width is 512, split as 64 rows per core. For each expert slot (8 routed, then the shared expert) the core runs 8 calls for its up band and 8 for its gate band (K = 2048, one 10 KB `w` element per call) against the table `gemv_q4_prep_k2048` built from xm. Results go to the `ms` scratch: u at offset 544, g at 608.

Routed experts are stored in 128-row stripes, and the host's DMA pattern pulls this core's 64-row half out of each stripe. The router decides which experts stream: between `lx0` and `lx1` the host patches the eight routed slots' weight addresses from the router's top-8 (`moeroute2`).

144 calls per core per layer, the largest weight stream of the MoE half. On an all-q8 spec (attention, linear and out projections at q8) this one translation unit is compiled with `-Oz` to keep the main core inside 16 KB; the spec records it saving 304 B.

## core:gemv_q4_gy
Title: Projection GEMV band (q4_1)
Summary: One 10 KB slice of a 64-row weight band times the activation table, accumulated into the band's 64-float output element.
Spec: OPEN-DECODE-ONE-CONTEXT
Source: open_kernels/designs/layer_x/gen_kernels.py

The projection kernel of the attention half, on the eight main cores (row 2) in `lx0` and `ax0`. Weights are q4_1 in 5120-byte chunks (32 rows x 256 K: bf16 scales, bf16 mins, 4096 B of nibbles); one call consumes one 10 KB `w` element, 2 chunks. A 64-row band of K = 2048 takes 8 calls, of K = 4096 16 calls. After a band's last call its 64 floats leave as one 256 B `y` element.

Per core per layer:
- `lx0`: 16 q|k|v bands and 8 z bands (K = 2048, against xn), then 4 out bands (K = 4096, against og): 256 calls.
- `ax0`: 8 q, 8 gate, 1 k and 1 v bands (K = 2048), then 4 o bands (K = 4096): 208 calls.

The inner product runs on the integer matrix unit: nibbles as they lie in the chunk against the int16 activation table (`gemv_q4_prep_*`). The band shape is a runtime argument, so one entry point serves every shape, and the row split is applied with shifts because the core has no integer divider. In the one-context image the first stage's band count (24 or 18) is a runtime word. A kernel set with q8 projections runs `gemv_q8_gy` here instead. Math: `designs/gemv_q4/gemv_q4.h`.

## core:gemv_q4_prep_h
Title: MoE: table the expert hidden h
Summary: Rounds the expert's hidden h (512 fp32 values, gathered from all 8 cores) to bf16 and block-quantises it for the down projection.
Source: open_kernels/designs/layer_x/gen_kernels.py

Runs once per expert slot (9 per layer) on each main core (row 2) in `lx1` / `ax1`. Each core produced 64 of the 512 hidden values (`moe_silu32`). The host DMA gathers the eight parts in DDR and broadcasts the whole h back as one 4 KB `x` element. This call converts it fp32 to bf16 to the int16 block table, written at `tab + 4608`, past xm's K = 2048 table, which the next slot's up/gate calls still read.

`gemv_q4_gdown` reads this table. The trip through DDR is how every core gets all of h: each core's 256 down rows need the full hidden vector. Table format: `designs/gemv_q4/gemv_tab.h`.

## core:gemv_q4_prep_k2048
Title: Activation table for K = 2048
Summary: Block-quantises a bf16[2048] activation to int16, one power-of-two scale per 32 values, plus block sums, for the GEMVs.
Source: open_kernels/designs/gemv_q4/gemv_q4.py

The GEMVs do their inner product on the integer matrix unit, so the activation has to be int16 first. This call does that once per activation per core. For each 32-value block, `s = 14 - floor(log2(max|x|))` and `xi = round(x * 2^s)`, plus the block sum as a bf16 hi/lo pair, which q4_1's per-block minimum needs. The table is int16 xi[2048] | int32 s[64] | bf16 sums hi/lo: 4,608 bytes in the core's `tab` buffer. It is exact for every value within 2^7 of its block max.

It runs on the eight main cores (row 2) at the start of `lx0` / `ax0` (on xn, the layer-entry norm output) and of `lx1` / `ax1` (on xm, the MoE input), and on the eight LM-head cores (on hn, the final norm output). The input is one 4 KB `x` element.

`gemv_q4.py` generates the `.cc` at build time (`ensure_prep_entry`); the math is `designs/gemv_q4/gemv_tab.h`.

## core:gemv_q4_prep_k4096_b0n64
Title: Activation table for og, first half
Summary: Block-quantises the first 2048 values (blocks 0-63) of the 4096-wide og into the GEMV table before the out projection.
Source: open_kernels/designs/layer_x/gen_kernels.py

The out projection (linear layers) and the o projection (full-attention layers) take og, a bf16[4096] vector: the DeltaNet post output or the gated attention output. It reaches the main cores (row 2) as two 4 KB `x` elements. This call builds blocks 0-63 of the K = 4096 table from the first one; `gemv_q4_prep_k4096_b64n64` does the second.

It runs once per layer on each main core, in `lx0` after `post_fn` and in `ax0` after `attn_fin`. The quantisation is the same as `gemv_q4_prep_k2048`. The K = 4096 table is 9,216 bytes and fills the whole `tab` buffer. Math: `designs/gemv_q4/gemv_tab.h`.

## core:gemv_q4_prep_k4096_b64n64
Title: Activation table for og, second half
Summary: Block-quantises the second 2048 values (blocks 64-127) of the 4096-wide og, completing the out-projection table.
Source: open_kernels/designs/layer_x/gen_kernels.py

The partner of `gemv_q4_prep_k4096_b0n64`. og (bf16[4096]) arrives as two 4 KB `x` elements; this call reads the second and fills blocks 64-127 of the K = 4096 table. The core acquires both elements together, runs the two preps back to back, then runs its 4 out bands (`gemv_q4_gy`, or `gemv_q8_gy` on a q8 kernel set).

Once per layer on each main core (row 2), in `lx0` and `ax0`. Math: `designs/gemv_q4/gemv_tab.h`.

## core:gemv_q8_gy
Title: Projection GEMV band (q8)
Summary: The q8 twin of gemv_q4_gy: one 10 KB slice of a 64-row band, as 16-row q8 half-tiles, into the band's output element.
Spec: OPEN-QUANT-Q8
Source: open_kernels/designs/layer_x/gen_kernels.py

Used in place of `gemv_q4_gy` when the model stores its attention projections at q8 (the two-context kernel set: `lx0` and `ax0`, on the eight main cores, row 2). The weights stream at q8 with no re-quantisation. A container q8 chunk is 8704 B (32 rows x 256 K), which does not fit the 5120-byte-based `w` element, so the packer splits each chunk into two 16-row half-tiles of 5120 B: 256 B of bf16 scales, 4096 int8 codes and 768 B of padding.

A band is still 64 rows, now four 16-row parts per k-tile, so twice the bytes of q4_1: 16 calls per K = 2048 band and 32 per K = 4096 band. That is 512 calls per core in `lx0` and 416 in `ax0`. The inner product is int8 codes against the same int16 activation table on the integer matrix unit.

The main core has room for only one GEMV entry in the one-context image, so a kernel set with q8 projections is built as the two-context `lx` / `ax` layout. Math: `designs/gemv_q4/gemv_q8.h`.

## core:glue_ab
Title: DeltaNet glue: alpha and beta projections
Summary: Accumulates xn @ Wa, then xn @ Wb (2048 x 32 each), the inputs to each head's decay and beta, one 4 KB tile per call.
Source: open_kernels/designs/dn_glue/glue_ab.cc

Runs on the glue core in `lx0`: column 2, row 4 in the one-context image (column 2, row 3 in the two-context `lx`). Each call takes one 4 KB `side` element holding 64 rows x 32 bf16 of a projection matrix and adds `xn[those 64 rows] @ W` into a 32-lane fp32 accumulator, which resets on tile 0. 32 tiles for alpha, then 32 for beta: 64 calls per layer.

It needs only xn, so the host starts it as soon as the layer-entry norm is in DDR, and it runs while the main cores are still on the q|k|v|z projection. `glue_small_fn` turns the two 32-value results into decay and beta. Math: `glue_ab_tile` in `dn_glue/dn_glue.h`.

## core:glue_conv
Title: DeltaNet glue: causal conv, SiLU, q/k norm
Summary: For one 1024-channel tile: 4-tap causal conv over 3 saved rows plus this token, SiLU, L2 norm of q/k heads, state shift.
Source: open_kernels/designs/dn_glue/glue_conv.cc

The glue core (`lx0`; column 2, row 4 one-context, row 3 two-context) calls this 8 times per layer, once per 1024-channel tile of the 8192 q|k|v channels: tiles 0-1 are q, 2-3 are k, 4-7 are v. Inputs are five 2 KB `gact` elements (this tile's qkv as fp32 in two halves, and the 3 saved conv-state rows as bf16) and two `side` elements with the tile's 4 conv taps.

Per channel it computes `c = silu(w0*s0 + w1*s1 + w2*s2 + w3*x)`. For q and k tiles it then L2-normalises each 128-dim head in place, refining the hardware inverse square root with two Newton steps; v tiles go to a separate buffer for `glue_emit_fn`. It writes the shifted state `[s1, s2, bf16(x)]` as three `gout` elements, and the shim DMA writes them back over the conv state in DDR.

It needs the main cores' q|k|v bands. Each core computes those before its z bands, so the host starts the glue once q|k|v is in DDR while the cores are still on z. Math: `glue_conv_tile` in `dn_glue/dn_glue.h`.

## core:glue_copy_xn
Title: DeltaNet glue: keep a copy of xn
Summary: Copies the layer-entry norm output xn (bf16[2048], 4 KB) from the side stream into the glue core's own buffer.
Source: open_kernels/designs/dn_glue/glue_copy.cc

The glue core's first call in `lx0` (column 2, row 4 one-context; row 3 two-context), once per linear layer. xn arrives as the first 4 KB `side` element, and the alpha/beta weight tiles follow on the same fifo.

A fifo release frees the oldest element held, so xn cannot stay in the fifo while the 64 weight tiles behind it are acquired and released. The core copies it to the local buffer `xnb` and releases the element; every `glue_ab` call reads `xnb`.

## core:glue_emit_fn
Title: DeltaNet glue: build a head's record
Summary: Packs one value head's k, q, v, decay and beta into the fp32[512] record that the main cores' DeltaNet step reads.
Source: open_kernels/designs/dn_glue/glue_emit.cc

Runs on the glue core in `lx0` (column 2, row 4 one-context; row 3 two-context): 8 calls after each v tile's `glue_conv`, 32 per layer, one per value head. The record is fp32[512]: k (128) | q (128) | v (128) | zeros, with decay at 384 and beta at 385. Two value heads share one key head, so head h takes k and q from key head h/2.

Each record leaves as one 2 KB `gout` element. The host DMA collects them in `act`, and each main core later receives its 4 heads' records at the start of each head's DeltaNet stream (`dnx_vcopy`). Math: `glue_emit` in `dn_glue/dn_glue.h`.

## core:glue_small_fn
Title: DeltaNet glue: per-head decay and beta
Summary: Turns the 32 alpha and 32 beta sums into decay = exp(A * softplus(alpha + dt_bias)) and beta = sigmoid(b) per value head.
Source: open_kernels/designs/dn_glue/glue_small.cc

Runs once per linear layer on the glue core (`lx0`; column 2, row 4 one-context, row 3 two-context), right after the 64 `glue_ab` calls. It reads one 4 KB `side` element, `small` = [A f32[32] | dt_bias f32[32]], and computes per value head `decay = exp(A * softplus(alpha + dt_bias))` and `beta = sigmoid(b)` into two 32-float local buffers.

Both values go into every head's record (`glue_emit_fn`). In the state update, decay scales the old state and beta scales the correction. It uses the scalar routines `sexp`, `ssoftplus` and `ssigmoid`; `vecmath.h` describes them as slow software float, meant for a few dozen values per call. Math: `glue_small` in `dn_glue/dn_glue.h`.

## core:lm_head_q8_group
Title: LM head: logits from q8 weights
Summary: Multiplies the final hidden state by two 8704-byte q8 chunks of the 248,320 x 2048 output matrix, into a 128-logit band.
Source: open_kernels/designs/lm_head_q8/lm_head_q8.cc

The last kernel of a decode step, in its own `lm` dispatch after the 40 layers and the final norm. Eight cores (columns 0-1, rows 2-5) each stream their own share of the output matrix: 1,940 bands of 128 vocabulary rows, split 243 / 242 per core. Each call takes one 17 KB `w` element, two q8 chunks of 32 rows x 256 K (512 B of bf16 scales and 8,192 int8 codes each), and accumulates into the band's 128-float `y` element. 16 calls finish a band; 31,040 calls per token across the eight cores.

The input is hn (bf16[2048]), broadcast once on `x` and turned into a table by `gemv_q4_prep_k2048`. The products run on the integer matrix unit (int16 activations x int8 codes), then the per-row, per-32-value scales are applied. Each finished band leaves as 128 logits. About 540 MB of weights stream through this dispatch per token, the largest single weight stream of the step. Math: `lm_head_q8.h`.

## core:ln_fn
Title: Residual add and RMSNorm
Summary: Adds a sublayer's output to the residual in fp32 and RMS-normalises the sum with a weight, giving the next stage's bf16 input.
Source: open_kernels/designs/ln/ln.cc

Computes `y = x + a` over 2048 fp32 values and `xn = bf16(y * rsqrt(mean(y^2) + 1e-6) * w)`. Inputs are five 4 KB elements (x and a as two fp32 halves each, w as bf16[2048]); outputs are y (two halves) and xn.

It runs in two places:
- In `lx0` / `ax0`, on the norm + router core (column 0, row 3), after the out / o projection. x is the layer input and a the attention output; y is the residual after attention (to `act`, read by the MoE), and xn is xm, the post-attention norm output that feeds the router and the experts.
- In the tail `ln` dispatch (its own one-core image) after the last layer, with a = 0: the model's final norm, whose output hn feeds the LM head.

Squares are summed as bf16 hi/lo products in fp32 accumulators, since the core has no fp32 vector multiply. Math: `ln.cc` and `ln.h`.

## core:ln_nr
Title: Layer-entry RMSNorm
Summary: RMS-normalises the layer input (fp32[2048]) with the input-norm weight into bf16 xn, the input of the layer's projections.
Source: open_kernels/designs/lin_layer/ln_nr.cc

The first call of every layer, in `lx0` and `ax0`, on the norm + router core (column 0, row 3). It reads three 4 KB `lni` elements (the residual stream as two fp32 halves and the norm weight as bf16[2048]) and writes one: `xn = bf16(x * rsqrt(mean(x^2) + 1e-6) * w)`.

The host DMA puts xn in `act`, then broadcasts it to the eight main cores for the first projection and, in linear layers, to the glue core for alpha/beta. It is `ln_fn` without the residual add, and its output is bit-identical to `ln_fn` with a zero add.

## core:moe_accfin
Title: MoE: weight and sum expert outputs
Summary: Adds one expert's down output into the block accumulator with its router weight; the shared expert's turn closes the block.
Spec: OPEN-PREFILL-BATCH
Source: open_kernels/designs/layer_x/gen_kernels.py

Called once per expert slot on each main core (row 2) in `lx1` / `ax1`, after the slot's `gemv_q4_gdown`. It works on the core's 256 rows in the `ms` scratch:
- routed slot e (0-7): `acc = (e == 0 ? 0 : acc) + w[e] * yd`, with w the renormalised router weight;
- shared slot (8): `acc = res + acc + sigmoid(xm . sgw) * yd`, the block output: the residual after attention, plus the routed sum, plus the gated shared expert;
- a negative slot closes on `res + acc` alone. The block prefill route (`mx`) uses it when the shared expert ran elsewhere.

`moe_hdr2` loaded w, the shared-expert gate and the residual rows at the start of the MoE. The weighted adds use bf16 hi/lo splits (three MACs) and no scalar float operations, which would pull in the soft-float library. 9 calls per core per layer.

## core:moe_hdr2
Title: MoE: load router weights, gate, residual
Summary: Three calls at the start of the MoE: copy the router's top-8 record, compute the shared-expert gate, copy this core's residual rows.
Source: open_kernels/designs/layer_x/gen_kernels.py

The first MoE calls on each main core (row 2) in `lx1` / `ax1`, three per core, each on one 10 KB `w` element:
- mode 0: the router record (`router_fin`'s output). 32 floats from byte 1024 go to `rw`, so routed weight w[e] sits at `rw[8 + e]`.
- mode 1: the shared-expert gate weight sgw (bf16[2048]). It computes `rw[0] = sigmoid(xm . sgw)` against the xm element.
- mode 2: this core's 256 rows of the residual after attention, copied to `xr`.

Every core does all three for itself, and `moe_accfin` reads the results. The sigmoid is computed on a vector lane because scalar float operations would pull in the soft-float library.

## core:moe_out
Title: MoE: write the layer output
Summary: Copies 64 rows of this core's finished block output into a y element; four calls send the core's 256 rows of the new residual.
Source: open_kernels/designs/layer_x/gen_kernels.py

The last call of a layer on each main core (row 2), in `lx1` / `ax1`, after the shared expert's `moe_accfin`. Call j copies rows 64j to 64j+63 of the `acc` buffer into a 256 B `y` element. The host DMA drains the eight cores' elements straight into `xres`, the residual stream, which is the next layer's input.

4 calls per core per layer, 32 across the array.

## core:moe_silu32
Title: MoE: SiLU(gate) times up
Summary: Computes this core's 64 values of an expert's hidden h = silu(g) * u in fp32 and sends them out as one 256 B y element.
Source: open_kernels/designs/layer_x/gen_kernels.py

Runs once per expert slot (8 routed plus the shared one, 9 per layer) on each main core (row 2) in `lx1` / `ax1`, after the slot's 16 `gemv_q4_gup` calls. It reads u and g from the `ms` scratch and writes `h = silu(g) * u`, with `silu(x) = x * sigmoid(x)`, to a 256 B `y` element.

The eight cores' parts (8 x 64 = 512, the expert width) are gathered in DDR and broadcast back to every core for the down projection (`gemv_q4_prep_h`). The output stays fp32; the bf16 rounding happens in that table prep.

## core:post_copy_nw
Title: DeltaNet post: load the norm weight
Summary: Copies the DeltaNet output-norm weight (bf16[128], the first 256 B of a 4 KB element) into the post core's local buffer.
Source: open_kernels/designs/dn_post/post_copy.cc

The post core's first call in `lx0` (column 1, row 3), once per linear layer, when the main cores have finished the DeltaNet step. The weight arrives as the first `pin` element; the core copies it to the local buffer `nwb` and releases the element.

`post_fn` then reads `nwb` for all 32 heads. The same 128 weights apply to every head.

## core:post_fn
Title: DeltaNet post: norm and output gate
Summary: For 8 heads: RMS-norm each 128-value head of the DeltaNet output, times the norm weight, times silu(z), written as bf16 og.
Source: open_kernels/designs/dn_post/post.cc

Runs on the post core (column 1, row 3) in `lx0`, 4 times per linear layer, between the DeltaNet step and the out projection. Each call takes two 4 KB `pin` elements, 8 heads of the DeltaNet output o (fp32, from `dnx_ofin`) and the same 8 heads of z (fp32, from the main cores' z projection), and writes one 2 KB `pout` element: `og = bf16(o * rsqrt(mean(o^2) + 1e-6) * nw * silu(z))`, with the RMS taken per 128-value head.

The four groups cover the 32 heads (og is bf16[4096]). The host gathers og in `act` and broadcasts it to the main cores, where `gemv_q4_prep_k4096_*` builds the out projection's table. The host issues the out projection's weight stream before this step, so the main cores' weight fifos fill while it runs.

## core:router_acc
Title: Router: expert logits
Summary: Adds 8 input rows of the 2048 x 256 router matrix into the 256 expert logits; 256 calls cover the whole matrix.
Source: open_kernels/designs/router/router.cc

Runs on the norm + router core (column 0, row 3) at the end of `lx0` / `ax0`, after `ln_fn` has produced xm. Each call takes one 4 KB `lni` element, 8 rows x 256 experts of bf16 weights, and adds `x[8rb .. 8rb+7] @ W` into the fp32 accumulator `racc[256]`; the first call zeroes it. 256 calls stream the whole 1 MB router matrix.

These are the last weight reads of the attention half. `router_fin` then turns the logits into the top-8 choice. Math: `router_acc_impl` in `router.h`.

## core:router_copy_x
Title: Router: keep a copy of xm
Summary: Copies xm (bf16[2048]) out of ln_fn's output element into the router's local buffer before that element is released.
Source: open_kernels/designs/router/router_copy.cc

One call per layer on the norm + router core (column 0, row 3) in `lx0` / `ax0`, right after `ln_fn`. xm, the post-attention norm output, is one of `ln_fn`'s three output elements, which leave for DDR (the MoE half reads xm from there).

The router needs xm for all 256 `router_acc` calls, so the core first copies the 2048 bf16 values into the local buffer `rxs`. Math: `router_copy_x_impl` in `router.h`.

## core:router_fin
Title: Router: softmax and top-8 experts
Summary: Softmax over the 256 expert logits, picks the 8 most likely experts and renormalises their weights to sum to 1.
Spec: OPEN-DECODE-PIPELINE
Source: open_kernels/designs/router/router_fin.cc

The last call of `lx0` / `ax0`, on the norm + router core (column 0, row 3). It computes `p = softmax(logits)` with a vector exponential. It then picks the top 8 by comparing the probabilities' bit patterns as unsigned integers (positive floats sort the same way) and divides their weights by their sum. The output is one 4 KB element: p (256 floats), the 8 expert indices at byte 1024 and the 8 weights at byte 1056.

The host DMA writes it to `act`. Before dispatching the MoE half (`lx1` / `ax1`), the host reads the 8 indices and patches that dispatch's routed weight fills to stream exactly those experts (`moeroute2`). The main cores also read the indices and weights directly (`moe_hdr2`). Math: `router_fin_impl` in `router.h`.
