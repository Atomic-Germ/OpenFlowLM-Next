# Issue #16: long-context attention on every family, batched prefill, vision

Upstream issue: https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/16. The
design and the hardware log live in `.claude/plans/issue-16-prefill-attention-vision.md`
and `.claude/plans/issue-16-hw-results.md` (gitignored); this file is the spec
impact.

## Spec impact

**Modified:** OPEN-ATTN-CONTEXT -- from "an observation" to a requirement: on
every family measured, a decode step's cost stays flat in the context position
(the vector-softmax / split-core / row-block attention kernel), and a family
enters the fast path by measurement (`recipes/attnknobs.py: FAST_ATTENTION`).

**New:**
- OPEN-VISION-VIT-REF -- the vision tower's reference (`model/replica_vit.py`,
  transformers' Qwen3VLVisionModel with the shipped weights) and the host C++
  port that matches it (`src/open_qwen36/vision`).
- OPEN-VISION-EMBED -- the open engine takes an image payload: the tower's
  rows enter as embedding vectors at their M-RoPE positions.

**Not in this change:** batched prefill. PR #39 (vegah) carries it as a
whole-array GEMM under issue #32; the gemm_q4 experiment that measured 1.2
TFLOPS on Qwen3-4B's shapes stays on the fork branch as a reference.

**Extended acceptance:** OPEN-BUILD-CACHE -- `ATTN_FAST` is a probe variable
and every family module exposes `probe_env`.
