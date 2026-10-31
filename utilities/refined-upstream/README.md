# Upstream causal drift scan

This follows the exact native/independent audit in `refined-sublayers`.
L50/P205, L54/P125 and L60/P220 locally pass but amplify incoming drift. The
scan traces **all 64 layers** at P125/P205/P220, comparing retained full-depth
control and refined candidate separately. It does not change or execute kernels.

```bash
source utilities/env.sh
export OPENBLAS_NUM_THREADS=1
ironvenv/bin/python utilities/refined-upstream/check.py run \
  --baseline Models/qwen38-27b/refined-runtime/native-v1 \
  --downstream Models/qwen38-27b/refined-sublayers/analysis-v1 \
  --out Models/qwen38-27b/refined-upstream/scan-v1
ironvenv/bin/python utilities/refined-upstream/check.py audit \
  --out Models/qwen38-27b/refined-upstream/scan-v1
ironvenv/bin/python utilities/refined-upstream/check.py retained \
  --scan Models/qwen38-27b/refined-upstream/scan-v1 \
  --out Models/qwen38-27b/refined-upstream/retained-v1
```

Use fresh versioned output directories and one BLAS thread for reproducible
reference arithmetic. Inputs retain their historical provenance: no metadata,
reference values, native outputs or build identities are rewritten. Preparation
verification binds the original native report/audit, private bundle identities,
plain/trace builds, capture inventories, independent references, and audited
downstream replay and attribution.

`scan.py` holds the scanner body that produced `scan-v1`. `check.py` is the
entry point: it loads that body and adds `retained`, which re-verifies the
stored evidence after a source change instead of re-running the stale compiler
inputs. `retained` checks the recorded scanner hash through its historical
source alias, replays the independent reference, and recomputes every signed
array and metric from the retained NPZ files. The old native builds are **not**
relabelled as fresh. `run`/`audit` replay the causal prefix and therefore do
require the original inputs; use `retained` only to confirm evidence after an
unrelated merge.

## Method

Cache only one layer's effective weights, using the same Q4 fallback policy as
the independent reference and production packing. For each layer:

1. Replay the original independent input prefix from zero state through P220.
   All three selected output rows must reconstruct the original stored FP32
   reference **bytewise**, before either native role can be interpreted.
2. Independently replay each role's captured input prefix from zero. Attention
   has separate causal KV; DeltaNet has separate conv/recurrent history. Never
   seed these histories with captured state or truncate the causal prefix.
3. At each selected position, substitute only the current native input into a
   **copy** of the independent pre-position history. The counterfactual must
   not mutate another target's history or either original trajectory.
4. Reuse the existing signed decomposition into reference storage rounding,
   current-input drift, input-history drift and implementation gap. Retain
   propagated drift and total error as well.
5. At the three selected downstream candidate controls, require every oracle,
   counterfactual, native and stored-reference node to reproduce the prior
   sublayer attribution bytewise.

Every role/layer/position retains its arrays in a separate NPZ. There are
384 cases, 192 exact independent reference reconstructions, and three exact
downstream controls. Missing, duplicate, reordered or extra inventory fails
closed. The separate audit reconstructs all causal trajectories and compares
all arrays bytewise, then recomputes all metrics, rankings and regressions.

All output effects use the **same layer-output reference scale**, while the
incoming residual metric uses its input-reference scale. Maxima must not be
added. Effects can cancel, and changing the replacement order changes nonlinear
attribution. A small total error can conceal a larger local error or history
contribution. The implementation gap includes accumulated local state error
and oracle arithmetic differences, not only the current step's arithmetic.
This scan identifies places to investigate; it cannot identify a faulty
projection or prove a proposed kernel change correct without boundary replay.

`complete: true` with exit 1 means completed numerical FAIL, not execution
failure. The existing maxrel <0.005 gate is unchanged. Local gaps passing that
gate do not imply sufficiently accurate full-depth inference. All control to
candidate total-error regressions are retained, including those under tolerance.

## Scope

CPU **reference diagnostics only**, over previously audited open-NPU captures.
No CPU fallback is introduced into inference. No NPU run/build, catalogue
promotion, packing/model conversion, production arithmetic, service or registry
change is made. No TTFT, throughput or performance acceptance claim is made.
Three selected prompt positions do not establish all-position, continuation,
hybrid, reset or serving acceptance. Full-model numerical acceptance remains
blocked until the production-equivalent independent gates pass.

Baseline: commit `353d88e`, **2073 passed, 47 skipped**. Nine new tests were RED
before implementation and pass afterward. Python 3.14.4 and one OpenBLAS thread;
the existing XRT/NPU toolchain is not needed for CPU execution, but remains
bound through retained native build provenance.

## Scan result — 2026-10-29

`scan-v1` completes all 384 cases, with 192 exact original-reference rows and
three exact downstream node sets. Peak cached effective matrices: **1.4258 GiB**.
Both roles' local implementation gaps pass **192/192**, but full residuals pass
only **159/192 control** and **156/192 candidate** in this failure-focused subset.
This is not the aggregate 257-position full-depth acceptance matrix.

Candidate total error improves in 86 rows, regresses in 97 and ties in nine.
L51/P205 is repaired, while L55/P125, L55/P220, L60/P125 and L62/P125 become
new failures. All regressions remain in the machine-readable report. The first
failing depths are L54/P125 and L50/P205/P220 in both roles. The candidate's
largest selected local gap is 0.0001270407 at L63/P220 (control 0.0003050622),
far below its late-layer propagated residual error.

| Candidate case | Current-input effect | Input-history effect | Local implementation gap | Total maxrel |
|---|---:|---:|---:|---:|
| L0/P205 | 0 | 0 | 0.0000073863 | 0.0000073863 |
| L0/P220 | 0 | 0 | 0.0000141479 | 0.0000141479 |
| L31/P125 | 0.0000224587 | 0.0000538285 | 0.0000018260 | 0.0000657616 |
| L31/P220 | 0.0002067694 | 0.0000071203 | 0.0000020132 | 0.0002157697 |
| L49/P205 | 0.0018092996 | 0.0000163031 | 0.0000098344 | 0.0018135651 |
| L50/P205 | 0.0117169752 | 0.0000294601 | 0.0000377089 | 0.0117765736 |

These values reject a large new local error at L31/L49 as the explanation for
their error growth. L31/P125 is primarily history-driven; L31/P220 and L49/P205
primarily amplify current-input drift. Small implementation errors begin at
L0 and are distributed across earlier layers. The candidate's largest selected
pre-L32 local source at P205/P220 is L0; at P125 it is L6 (0.0000052741).
L7/P205 is another early local source (0.0000072617).

Next isolate current-production **L0 at P125/P205/P220**, retaining exact
native outputs and existing independent references, and attribute its small
local error to entry/projection/post/FFN boundaries. Preserve L6/P125 and
L7/P205 as alternate early-source probes and L31/P125 as a history-amplification
control. A local correction needs a controlled causal propagation experiment
before claiming any improvement in downstream logits; rankings alone cannot
establish which source dominates final failure. Do not start by modifying
the locally passing late amplification layers.

Host regression on the merged tree: **2093 passed, 47 skipped**; retained
plain/trace CLI CTest **4/4 each**; settings registry byte-identical and the
merged settings CI check passes. No NPU run/build or production performance
measurement was made.

After the `origin/main` merge, `retained-v1` re-verifies the stored scan with
the historical scanner body mapped through its source alias: **complete, 384
cases, 192 reference rows, 3 downstream controls, numerical FAIL**. It does not
re-run the stale build inputs and does not relabel them. Nineteen contract
tests pass, including retained-verdict, source-mapping and signed-array
rejection tests.

## Verification after the main merge (2026-10-30)

The original scanner is preserved **byte-for-byte** as `scan.py`; `check.py`
now provides its CLI/API and an explicit `retained` verification mode. The
original report and causal audit remain immutable. The historical source entry
for `check.py` must hash exactly to `scan.py`; all other analysis sources must
still match their recorded bytes. This is a narrow source relocation, not a
waiver for changed model, capture, reference or arithmetic files.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/refined-upstream/check.py retained \
  --scan Models/qwen38-27b/refined-upstream/scan-v1 \
  --out Models/qwen38-27b/refined-upstream/retained-after-main-v1
```

This mode validates the bound original causal audit, every pinned input and
384 artifact hashes, the model/oracle identity, native/reference node identity,
192 original reference rows and three downstream controls. It recomputes all
signed effects, input metrics, rankings, regressions and the numerical verdict
from retained arrays. It **does not rerun causal trajectories** or certify old
builds as fresh. The full causal recomputation is established by the unchanged
original audit, whose hash is recorded separately.

The merge changes `dense.py` and `block_host_test.cpp`, both included in old
build provenance. Strict fresh-run validation continues to reject stale build
sources; no metadata or source key is rewritten to bypass that check. Use new
build/preparation directories for a new hardware experiment. The retained
verification is CPU-only and needs no resident model on the NPU.

Additional TDD coverage rejects source-alias tampering, incomplete arrays,
changed signed effects and an incorrect numerical verdict. The historical
nine RED tests, subsequent four provenance cases and three new retained-audit
cases give **16 tests** for this diagnostic tool.
