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
