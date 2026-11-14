---
name: open-qwen38-refined-upstream
description: Trace refined Qwen3.8 upstream current-input, history and local implementation drift across 64 layers using immutable native captures, exact causal references and downstream controls.
---

# Refined candidate upstream causal scan

Read `utilities/refined-upstream/README.md` for complete commands and scope.
Use after the exact `refined-sublayers/native-v2` replay and `analysis-v1`
audit. These targets pass locally but fail against full-model references.

1. Source `utilities/env.sh`; run `ironvenv/bin/python` with
   `OPENBLAS_NUM_THREADS=1`. This is CPU reference analysis over existing
   audited NPU outputs, not inference fallback or a fresh hardware run.
2. Run `check.py run` against `refined-runtime/native-v1` and downstream
   `refined-sublayers/analysis-v1`, into a fresh versioned Models directory.
   Keep the source and reference metadata unchanged during run/audit.
3. Replay all 64 layers for both roles from position zero through P220;
   inspect P125/P205/P220. Cache only one layer's effective weights.
4. Require 192 byte-exact original reference reconstructions before interpreting
   the 384 role/layer/position cases. Never substitute captured native state
   for independent conv/DeltaNet/KV history. Counterfactuals copy prior state.
5. Require the three downstream target node sets to reproduce their separately
   audited sublayer attribution bytewise. Missing/reordered coverage fails.
6. Run the separate audit to recompute every causal trajectory and NPZ array,
   all measurements, rankings and every control/candidate regression.
7. Keep signed current-input, input-history and implementation-gap vectors.
   Their maxima do not add, and individual effects can cancel. Local gap
   includes accumulated implementation state and oracle arithmetic differences.
   Largest local gap is not automatically the causal source of final failure.

Complete numerical FAIL exits 1 and remains separate from execution failure.
Do not promote the candidate, relax gates, or change a locally passing late
layer based only on amplified residual error. Production export, models,
services, registry and catalogue remain untouched. This scan excludes fresh
NPU boundary validation, all-position acceptance, reset and performance.
+
+`scan.py` is the scanner body that produced the retained evidence; `check.py`
+is the entry point that loads it and adds `retained`. After an unrelated merge,
+use `retained --scan <dir> --out <fresh dir>` to re-verify stored evidence
+without re-running stale compiler inputs or relabelling old builds. It maps the
+recorded hash of the old `check.py` through the byte-identical `scan.py` alias,
+replays the independent reference, and recomputes every signed array and metric.
+It confirms evidence integrity only; it is not a fresh scan or acceptance run.
*** Update File: specs/open-engine/plans/qwen38-main-rebaseline.md
@@
# Qwen3.8-27B validation report

+## Refined candidate upstream drift scan — 2026-10-31
+
+Traced the refined candidate and control across **all 64 layers** at P125/P205/
+P220 using retained native outputs and unchanged independent references. The
+scan completes **384 cases** with **192 byte-exact original reference rows**
+and three byte-exact downstream attribution controls. A separate CPU audit
+reconstructs every causal trajectory, NPZ array, metric and regression.
+
+Every selected local implementation gap passes **192/192** in both roles under
+the existing maxrel gate, while full residuals pass **159/192 control** and
+**156/192 candidate** in this failure-focused subset. Candidate total error
+improves in 86 rows, regresses in 97 and ties in nine. L51/P205 is repaired;
+L55/P125, L55/P220, L60/P125 and L62/P125 become new failures. This is not the
+aggregate 257-position acceptance matrix.
+
+| Candidate case | Current input | Input history | Local gap | Total maxrel |
+|---|---:|---:|---:|---:|
+| L0/P205 | 0 | 0 | 0.0000073863 | 0.0000073863 |
+| L0/P220 | 0 | 0 | 0.0000141479 | 0.0000141479 |
+| L31/P125 | 0.0000224587 | 0.0000538285 | 0.0000018260 | 0.0000657616 |
+| L31/P220 | 0.0002067694 | 0.0000071203 | 0.0000020132 | 0.0002157697 |
+| L49/P205 | 0.0018092996 | 0.0000163031 | 0.0000098344 | 0.0018135651 |
+| L50/P205 | 0.0117169752 | 0.0000294601 | 0.0000377089 | 0.0117765736 |
+
+Signed effect maxima do not add, and effects can cancel. L31/L49 do not
+introduce a large new local error; L31/P125 is primarily history-driven while
+L31/P220 and L49/P205 amplify current-input drift. Small local errors begin at
+L0 (0.0000073863 at P205, 0.0000141479 at P220) and are distributed across
+earlier layers; L6/P125 and L7/P205 are alternate early-source probes. A
+correction needs controlled causal propagation before any downstream
+improvement can be claimed, so the locally passing late amplification layers
+must not be edited first.
+
+Next isolate current-production L0 at P125/P205/P220 and attribute its local
+error across entry/projection/post/FFN boundaries, retaining L6/P125 and
+L7/P205 as alternate probes and L31/P125 as a history-amplification control.
+
+No kernels, xclbins, catalogue entries, packing, production export, installed
+models or services changed. No NPU run was needed; this is CPU reference
+analysis over audited open-NPU captures, not inference fallback. Merged-tree
+host suite: **2090 passed, 47 skipped**; retained CLI CTest **4/4 each**;
+settings registry byte-identical and merged settings CI passes. After the merge,
+`retained-v1` re-verified the stored evidence (complete, 384 cases, 192
+reference rows, 3 controls, numerical FAIL) without relabelling old builds.
+19 contract tests pass. Full-model numerical acceptance, reset/isolation and
+performance remain open. Evidence: `Models/qwen38-27b/refined-upstream/`
+(`scan-v1`, `retained-v1`); runbook `utilities/refined-upstream/README.md`;
+skill `open-qwen38-refined-upstream`.
+
## Refined candidate causal linear sublayers — 2026-10-29
*** End Patch
