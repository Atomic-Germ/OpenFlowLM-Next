## host:embed
Title: embed: the token's embedding row
Summary: The CPU copies the new token's row of the embedding table into xres as fp32 and syncs it to the device.
Spec: OPEN-EMBED-STRIDE
Source: src/open_qwen36/core.cpp

A decode step starts on the CPU. `Core::step_impl` reads row `token` of `model.embed_tokens.weight` from the model file (2,048 bf16 values), widens it to fp32 in the host view of `xres`, and syncs those 8 KB to the device. The NPU never reads the embedding table; the host looks up the one row the step needs.

For models with M-RoPE (vision input) this stage also writes the position's `ptab` record; the Qwen3.6-35B text path does not. The page shows 20 µs, a nominal time, not a measurement.

## host:attnpos
Title: attnpos: patch ax0 for this position
Summary: Once per step, the CPU rewrites four words of ax0's instruction stream so its KV-cache walk and RoPE record match the current position.
Spec: OPEN-DECODE-PIPELINE, OPEN-ATTN-CONTEXT
Source: open_kernels/harness/stream_patch.hpp

`ax0` is built for position 1. Before the layer walk, `stream_patch::attn_apply` rewrites four words in it: the length of the KV-window fill, where the window starts, the cache row the new K/V row is written to, and which `ptab` row to read. With blocks of 4 rows (manifest `rb` 4) the window is padded to `4 × (pos ÷ 4 + 1) − 1` rows, integer division; the attention cores mask the padding. Then the instruction buffer is synced to the device.

All ten full-attention layers run the same `ax0` stream, so one patch serves the whole step. It happens with nothing outstanding on the NPU, which keeps it safe (OPEN-DECODE-PIPELINE). The page shows 10 µs, nominal, and draws `ax0` with the window this patch gives at position 1.

## host:route
Title: route: point the MoE stream at the experts
Summary: Between a layer's two dispatches, the CPU reads the router's top-8 expert ids and patches lx1 or ax1 to fetch those experts' weights.
Spec: OPEN-DECODE-PIPELINE, OPEN-REQUEST-ISOLATION
Source: src/open_qwen36/core.cpp

When `lx0` or `ax0` finishes, the router record sits in that layer's `act`. `Core::route` reads 32 bytes of it: the eight expert ids. Each id slot was set to a sentinel at the top of the step, so a slot that still holds it means the record has not landed yet, and the host reads again (OPEN-REQUEST-ISOLATION). Then `stream_patch::moe2_apply` rewrites the pool offsets of the routed slots' fills in `lx1` or `ax1`, and the instruction buffer is synced to the device. The host does not read the mixing weights: `lx1` streams them from the same record.

This is the one point in a layer where the NPU waits for the host. The MoE stream can only be patched after `lx0` has finished and after the previous layer's run of the same kernel has finished, because every layer shares one instruction buffer per kernel (OPEN-DECODE-PIPELINE). The measured host gap between dispatches is 0.02–0.07 ms; the page uses 0.045 ms.

## host:readback
Title: readback: copy the logits to the host
Summary: After the lm head finishes, the CPU syncs the 993,280-byte logits buffer from the device and copies it into host memory.
Spec: OPEN-REQUEST-ISOLATION
Source: src/open_qwen36/core.cpp

`Core::step_impl` waits for `lm`, then calls `read_back` on `logits`: a device-to-host sync followed by a memory fence. Every device-to-host read in `core.cpp` goes through it (OPEN-REQUEST-ISOLATION). The 248,320 fp32 logits are then copied into host memory.

`Engine::logits_view` converts them to bf16 for the sampler and sets the 250 padding rows past the tokenizer's 248,070 entries to −∞, so they can never be picked. The page shows 150 µs, nominal.

## host:sample
Title: sample: pick the next token
Summary: The CPU turns the logits into one token id, by argmax or with the request's temperature, top-k, top-p, min-p and penalties.
Spec: TOOLS-REQUEST-PARAMS-RESET
Source: src/common/modules/sampler.cpp

With `top_k` 1 the sampler takes the argmax straight from the bf16 logits. Otherwise it copies them to fp32, applies the repetition, frequency and presence penalties, keeps the top `k`, applies top-p, min-p and temperature, normalizes with a softmax and draws one token. A temperature of 0 keeps only the best candidate.

The settings come from the request, or from the model's defaults when the request leaves them out (TOOLS-REQUEST-PARAMS-RESET). The token id becomes the input of the next step's embed stage. The page shows 250 µs, nominal.
