# qwen2.5-14b-instruct-i1-gguf (candidate / set aside)

- **Spec:** qwen2, hidden 5120, 48 dense layers, heads 40/8, head_dim/rotary 128, intermediate 13824, qk_norm false, qkv_bias true.
- **Status:** candidate (not in validated export set). Low priority vs 3.x demand; may revisit if needed.
- **Blocker:** dense `dx` build (`open_kernels/designs/dense/dx.py`, `dense/build_qwen2_h5120`) fails at aiecc link:
  - `ld.lld: error: section '.data' will not fit in region 'data': overflowed by 32 bytes`
  - `aiecc: edge 'elfs_{0}.elf' (key 'main_core_2_3') failed`
- **Context:** core (2,3) is attn core c==0 for this geometry (KVC=0 -> `Tile(2+c,3)`). Qwen3 h=5120 `dx` builds successfully but differs (qk_norm true, no qkv_bias, different FFN/CALL_BYTES/PER_CALL/TAB_BYTES/layout). Qwen2 path includes abias stream/bias handling; likely tight L1/data-region layout for this config. Reducing abias ObjectFifo depth (4->2) did not resolve the 32B overflow.
- **Large-K gemv_q4 note:** streamed-x standalone gate path was implemented/validated for K=13824 (`share_down_k13824_p1` PASS) and is useful independently of full dx export for this spec.
- **Next steps if revisited:** capture per-core link artifacts (ld script/map/elf) for `main_core_2_3`, compare memory map vs qwen3 h=5120 and smaller qwen2 builds, identify `.data`/buffer pressure (abias/attention buffers/layout), and apply a minimal geometry-safe fix before gating/export.
