# Plan: Ternary Bonsai 2 27B on the open kernels (rotated basis, 2-bit weights)

**Status:** in progress (2026-10-02, Opus 5.5). The user asked to skip review and run Phase 1,
Phase 2 and the measurements in one go. Working notes and probe results are in the main
checkout's `.claude/plans/bonsai-2-ternary.md`.

**Builds on:** OPEN-FAMILY-QWEN35's 27B (#148): Qwen3.8-27B is Ternary Bonsai 2's tower.

## Spec impact

**New requirements**
- **OPEN-HADAMARD:** a rotated-basis ternary model runs with every projection input taken
  through H/sqrt(1024) per 1024 block. This happens in the main-core preps
  (`designs/gemv_q4/wht.h`, generated into `dense_prep*.cc`) and in the block route's host
  (`host::hadamard_rows` in `Core::gemm_run`). The attention output also has its ±1 signs
  applied first, from the spec's `hadamard.og_signs`. Every other sign is folded into weights
  by the converter.
  - Verification: `test` for the spec derivation, the generated sign table and the build
    suffixes (`tests/test_ternary_hadamard.py`). `manual` for the NPU transform: greedy
    agreement with PrismML's llama.cpp on the same prompt.
- **OPEN-QUANT-T2:** an all-t2 spec streams 2-bit ternary chunks. A chunk is 32 rows x 256 K
  of codes plus a bf16 scale per (row, 128 K), 2560 B, two per 5 KB w element. The engine
  packs them at load (`pools::t2_perm`) from the container's exact q4_1 copy and refuses any
  chunk that is not exact ternary. The prefill GEMMs keep q4_1 copies of their own
  (`from: pack`).
  - Verification: `test` for the recipe geometry, the pack plan and the 2-bit packing
    against PrismML's decode (`tests/test_ternary_hadamard.py`, `open_kernels/t2_pack.py`).
    `manual` for the kernel: probe 0a (`designs/gemv_t2`) at the DMA floor and bit-exact,
    then the whole model's greedy agreement with Phase 1 and with PrismML.
- **OPEN-CONVERT-PRISM-TERNARY:** q4nx-build reads PrismML's PQ2_0 GGUF (type 142) into an
  exact-ternary q4_1 container. It folds the 5120- and 17408-wide signs into norm gains,
  alpha/beta columns and up rows; un-rotates the embedding (bf16) and the lm head (q8); keeps
  ssm_out's grouped value-head columns; and writes `prism_hadamard` into config.json.
  - Verification: `manual`, the converter's numpy validation against the GGUF reference math
    (in the converter's own report).

**Modified requirements**
- **OPEN-FAMILY-QWEN35** gains a Ternary Bonsai 2 row.
- **OPEN-PREFILL-BATCH** gains the host transform and, for t2, packed GEMM weights.

**No hash moves** for any existing spec: `hadamard` is absent from `to_dict()` when unset,
and `quant_hash()` is unchanged for every q4_1 and q8 spec. The attention-GEMM build
directories carry the quant suffix only for q8 specs, so a rotated or t2 spec shares the
plain 27B's.

## Addendum 2026-10-03: the lm head as t2

PrismML's head is its own ternary PQ2_0 tensor (rotated, untied), so the q8 un-rotation was a
lossy step with 4x the bytes. Spec impact: **modified OPEN-CONVERT-PRISM-TERNARY** (the head is
written as exact ternary q4_1 in the rotated basis, s5120 folded into output_norm,
`prism_hadamard.lm_head = "rotated"`; now `test` + manual), **modified OPEN-QUANT-T2** (a rotated
head runs through `designs/lm_head_t2` from unpadded 2176 B t2 chunks; `t2_perm` takes an
optional pool `chunk_bytes`), **modified OPEN-HADAMARD** (`hadamard.lm_head`). The shared
container keeps its q8 head and its kernel set is unchanged byte for byte; the new head needs the
re-converted container. Working notes: the lmhead worktree's `.claude/plans/lmhead-ternary.md`.

## Addendum 2026-10-05: round 2 speed

Decode 176 -> 162 ms/token and a 512-token prompt 7.3 -> 4.8 s, both measured against this PR's
previous head (87d0c088) in one clean window, 3 interleaved rounds, 129/129 tokens every run.
All of it is in the decode image (`designs/layer_x/dux.py`), the attention kernel, the prefill
GEMM and the host stages; the container is unchanged.

**Decode**
- **Attention:** Bonsai runs RB 4 (four cached rows per kernel call), walks the window in whole
  blocks, and takes #115's integer scalar fp and score tree. **Modified OPEN-ATTN-CONTEXT**
  (Bonsai's rule, the manifest's `rb_win`, its refusals). Not bit-exact against RB 1; the greedy
  tokens are identical. Context slope 17.0 -> 7.9 us per cached token.
- **The tail in the layer context:** the final norm and the lm head become the merged image's third
  stream, so a step uses one hardware context. **Modified OPEN-DECODE-ONE-CONTEXT-DENSE.**
  `OPEN_LAYER_TAIL=0` at export restores the separate images.
- **The prep core:** the down projection's second K piece is transformed on core (7, 4) while the
  main cores run the first. **Modified OPEN-HADAMARD** (where the decode transform runs) and
  **OPEN-DECODE-ONE-CONTEXT-DENSE** (the core list and the shim budget). Logits byte-identical.
- **No spec impact, bit-exact:** the per-128 table written straight-line; one fill reads a DeltaNet
  head's S for both passes; the xn / og fills issued first; the main cores' loop counts as per-core
  data words and the vector exp / reciprocal as loops (main core 15,280 -> 13,648 B, recorded in
  OPEN-DECODE-ONE-CONTEXT-DENSE's size constraint).

**Prefill**
- **bfp16 activations:** the GEMM streams activations as bfp16 blocks and the dequant emits the
  weights' bfp16 blocks (`GQP_XBFP`, manifest `x_bfp`). **Modified OPEN-GEMM-T2.** The host's
  bf16 -> bfp16 conversion matches the hardware's byte for byte (`designs/bfp_cvt`), so the GEMM's
  output and the prefill logits are byte-identical to the previous head.
- **Host stages, no spec impact (bit-identical):** persistent DeltaNet scratch, vectorised conv and
  alpha/beta, a single-pass delta rule, a vector FWHT with register-transposed tiling and SwiGLU
  fused into the down projection's transform, the token-major y read in place, and a threaded
  attention epilogue. Host time 3.2 -> 1.7 s.

**What did not pay:** splitting every prep across the main cores or moving the whole prep to the
idle row-4 cores (the prepared table's DDR round trip costs what the compute saved); a packed
DeltaNet state (the round already sits at its byte floor); memtile-resident prefill activations
(after bfp16 the GEMM is core-bound, not traffic-bound).
