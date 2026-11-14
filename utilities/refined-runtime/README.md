# Full-depth private ax refinement

This experiment runs the shared CLI through all 64 layers with 257 prompt
tokens and one continuation. It replaces only the `ax` binaries in private
copies of a compatible standard export. It does not promote the experimental
arithmetic to the production recipe or modify an installed model.

Build and audit the integrated control/candidate with
`utilities/l3-refined-ax/README.md` first. Then build fresh plain and capture
CLIs using `utilities/prefill-production/build.py`; use `--plain` for the first
build and omit it for the second. Keep separate output directories and retain
their CTest/build records. No private host arithmetic is added.

```bash
source utilities/env.sh
export OPENBLAS_NUM_THREADS=1
ironvenv/bin/python utilities/prefill-production/build.py \
  --out Models/qwen38-27b/refined-runtime/plain-v1 --plain --jobs 2 \
  --xrt-include /home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/include \
  --xrt-lib /home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/lib
# Repeat with --out .../trace-v1 and without --plain.

ironvenv/bin/python utilities/refined-runtime/check.py prepare \
  --baseline Models/qwen38-27b/post-prefill-production/boundaries-native-v1 \
  --references Models/qwen38-27b/prefill-full/boundaries/257 \
  --parent Models/qwen38-27b/production-post-fp32/kernels-v1 \
  --control-ax Models/qwen38-27b/post-prefill-drift/l3-refined-ax-control-v2 \
  --candidate-ax Models/qwen38-27b/post-prefill-drift/l3-refined-ax-candidate-v2 \
  --plain-build Models/qwen38-27b/refined-runtime/plain-v1 \
  --trace-build Models/qwen38-27b/refined-runtime/trace-v1 \
  --out Models/qwen38-27b/refined-runtime/prepared-v1
ironvenv/bin/python utilities/refined-runtime/check.py run \
  --prepared Models/qwen38-27b/refined-runtime/prepared-v1 \
  --out Models/qwen38-27b/refined-runtime/native-v1
ironvenv/bin/python utilities/refined-runtime/check.py audit \
  --out Models/qwen38-27b/refined-runtime/native-v1
```

Use fresh versioned directories for every changed source/build. Preparation
pins the parent export, both private builds, copied artifacts, current CLI
sources/compiler dependencies and immutable CPU references. Each composite
bundle receives a `private-refined-ax:` key binding the parent manifest, private
build record and role. It preserves every runtime/packing field and records the
historical parent separately; its identity is not a production recipe key.
Require unchanged ax instruction bytes. Copy files instead of pointing bundle
symlinks outside the bundle root.

The runner acquires the existing exclusive NPU lock and checks device users
and at least 35 GiB available memory before each of four resident processes.
The two fresh control runs precede the candidate. All 324 control captures
(257 prompt logits, two generation logits, scratch and 64 residual arrays)
must reproduce the retained native baseline exactly. Both capture CLIs must
also reproduce their matching plain CLI logits/scratch exactly.

For each role, evaluate all 16,512 residual rows and 258 logits against the
same independent oracle, including the continuation. Keep maxrel <0.005,
correlation >0.9999, matching argmax/tokens and prompt alias checks. Record
improvements, ties and every regression separately. Numerical failure is not
execution failure: a complete report may have `candidate_passed: false`; audit
recomputes the measurements and verdict. Failed identity/provenance is an
execution failure. The command exits nonzero for either failure kind.

This scope excludes hybrid routes, separate 255/256 runs, repeated resident
requests, serving isolation and performance acceptance. `model_acceptance`
remains false even if this experiment's numerical gates pass. Capture/logging
timings are diagnostic only.

## Retained result (2026-10-29)

`plain-v1` / `trace-v1` pass CTest 4/4 and pin 589/590 compiler dependencies.
`prepared-v1` / `native-v1` complete all four runs and independent re-audit.
Control identity: 324/324; plain/trace: 260/260 for each role. All 258 argmax
values and generated tokens match the reference in both roles.

Candidate residuals pass 16,434/16,512 versus control 16,416/16,512. Worst
maxrel falls from 0.0992221863 to 0.0620054922; logits pass 255/258 versus
254/258, minimum correlation rises from 0.9963255072 to 0.9992857176. Both
run and audit deliberately exit 1: execution is complete, numerical acceptance
is false. Do not treat this exit status alone as an incomplete hardware run.

Residual comparison retains 9121 improvements, 6584 regressions and 807 ties;
43 failing rows are repaired and 25 previously passing rows now fail. Candidate
logit failures are P205/P213/P220. First failing depth: L50/P205; earliest
failing position: L54/P125; worst error: L60/P220. Preserve the latter two as
regression controls when localizing candidate DeltaNet/FFN boundaries.
