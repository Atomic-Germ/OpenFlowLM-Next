# Full-layer Q4 consistent-sum experiment

This private `lx` changes only the direct FP32 activation-table block sum to
use the reconstructed INT16 values already used by the Q4 dot product. It
keeps the production compensated DeltaNet recurrence, geometry, ABI, stack,
weight packing and dispatch schedule. Production recipe selection is unchanged.

`consistent_lx.py` copies the current generated entry points and injects private
GEMV headers. It checks compiler dependency files for the private table/GEMV
headers and unchanged production factored recurrence headers. The separate
resource audit must pass before hardware replay. The control copies the
verified production `lx` binaries.

`consistent_replay.py` forces incoming production residuals, with each layer
starting at zero and carrying its own conv/recurrent state through subsequent
positions. It loads one layer's weights at a time. Outputs and references are
separate from production, with source, model, pool, binary and capture hashes.
Prepare removes stale dumps and reports; score writes every numerical failure.

Acceptance checks:

- Control: complete act/state/residual buffers match production bytewise.
  Numerical results remain visible independently of control identity.
- Candidate: entry norm matches production; QKV matches the validated isolated
  consistent-sum probe bytewise.
- Independent FP64 residual-to-record, conv history and composed DeltaNet
  state/output checks; strict per-head recurrence checks on captured records.
- Z projection, post-recurrence norm/gate and output projection checks. The post
  and output checks are conditioned on captured inputs; Z also uses the
  independent entry norm. Existing BF16 gated-output storage is preserved.
- FP64 FFN norm/up/gate/SiLU and both down partials plus residual closure.
- Whole-layer residual against the existing mixed FP64/FP32 `replica_qwen35`,
  whose causal state starts at zero and never comes from NPU captures.

Thresholds are unchanged: normalized maximum error <0.005 for projections and
residuals; per-head recurrence error <1e-4 and cosine >0.9999999. Passing these
forced-input layer checks does not establish propagation through the model.
Production export integration and fresh full-depth validation are separate.

See the [build/replay runbook](../../.agents/skills/open-qwen38-consistent-full-layer/SKILL.md)
for commands and measured evidence. Keep artifacts under ignored `Models/`.
