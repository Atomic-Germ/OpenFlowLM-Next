---
name: open-qwen38-refined-runtime
description: Validate private integrated ax refinement through all 64 Qwen3.8 layers with fresh shared CLI controls and independent references.
---

# Private refined ax in the shared runtime

Read `utilities/refined-runtime/README.md` for complete commands and scope.
Use after the integrated `open-qwen38-refined-ax` local gate passes. Keep its
original six-buffer ABI and instruction bytes; do not patch host projections.

1. Source `utilities/env.sh`; use `ironvenv/bin/python` and one OpenBLAS thread.
2. Build fresh plain and trace CLIs with `utilities/prefill-production/build.py`.
   Run CTest and retain compiler dependency/source/executable records. The
   trace builder adds only synchronized per-layer captures. On the current
   host compiler and NPU execution use execution-tool escalation.
3. Use `utilities/refined-runtime/check.py prepare` to copy the standard 46-set
   export twice, replacing only `ax/final.xclbin` and `ax/insts.bin` with the
   separately audited control/candidate. Preparation verifies generated
   translation units, compiler inputs, core resources and binary hashes.
4. Keep parent and composite identities separate. Each private bundle has a
   distinct `private-refined-ax:` key and toolchain record binding the parent
   manifest, private build and role. Runtime semantics remain identical. Never
   assign the production recipe key to experimental arithmetic, and never
   rewrite historical reference metadata to claim a current source identity.
5. Run the prepared experiment with exclusive device lock and at least 35 GiB
   available RAM before each process. Four resident processes cover control
   and candidate, each with plain and trace CLIs. Every process starts fresh.
6. Require complete exact historical control identity before the candidate;
   require trace/plain identity for each role. Compare all 64 layers, 257 prompt
   positions and one continuation against immutable independent references.
7. Run the separate audit. Numerical FAIL must remain visible independently
   of execution completion. Preserve regression lists and unchanged gates:
   maxrel <0.005, correlation >0.9999, matching argmax/tokens and prompt alias.

Use fresh directories under Models for any changed pinned source. Existing
output directories are refused. Keep useful tooling under utilities. Do not
alter the installed model, registry, services, standard export or production
recipe. Full-model support/performance is not accepted by this experiment:
hybrid boundary runs, reset/isolation and production throughput remain separate
gates. Diagnostic timing with logit/residual capture is not a benchmark.

## Retained checkpoint

Artifacts are in `Models/qwen38-27b/refined-runtime/`: `plain-v1`, `trace-v1`,
`prepared-v1`, `native-v1`. Four runs and separate audit complete; numerical
acceptance is false (exit 1 is expected). All 324 historical controls and 260
plain/trace outputs per role match. Candidate passes 16,434/16,512 residual
rows and 255/258 logits, with unchanged gates. Worst maxrel 0.0620054922 and
minimum correlation 0.9992857176 still fail. Preserve all 6584 residual
regressions, including 25 newly failing rows. Next isolate L50/P205 candidate
DeltaNet/FFN boundaries; retain L54/P125 and L60/P220 as regression checks.
Host suite 2064 passed, 47 skipped; both CLI builds CTest 4/4.
