---
name: open-qwen38-ffn-projection-sweep
description: Build the production-arithmetic FFN probe and validate separate up/gate projections over all Qwen3.8 layers, with byte-exact norm/activation identity and streamed weights.
---

# All-layer FFN up/gate validation

Use with an accepted production-consistent full-depth fixture. See the
[production integration runbook](../open-qwen38-consistent-integration/SKILL.md)
for model/pool provenance, the complete export and resident lifecycle checks.
This probe supplements the composite FFN front/down gates with separate up and
gate measurements so one erroneous projection cannot hide behind their product.

`utilities/ffn-front-replay/probe.py` supports `FRONT_FP32=2`: direct FP32 prep
with `GEMV_Q4_CONSISTENT_SUM=1`, matching the validated production geometry.
Modes 0 and 1 keep the historical BF16/original-sum experiments. New builds
must use `front_build.py`, which selects mode 2, checks the geometry, hashes
recipe/kernel/probe sources and audits actual copied `.cc` files and `.o.d`
production-header dependencies. The JIT key includes the complete source set,
spec and mode. A historical unpinned probe is not accepted for this sweep.

The probe has one norm helper and one GEMV core. It uses the unchanged packed
production up/gate pools and existing norm/GEMV/SiLU functions. It adds separate
projection outputs without modifying production lx/ax or their manifests.
`front_Ln_tT.bin` consists of bands `[up[64], gate[64], h[64]]`; it is not three
contiguous full vectors. `norm_Ln_tT.bin` is FP32[hidden].

## Build and resource audit

Native Ryzen AI 9 365 / Strix XDNA2, Python 3.14, mlir-aie 1.4.2,
Peano 21.0.0.2026080301+c9c5ecb7, XRT `/opt/xilinx/xrt`.
Compiler subprocesses and hardware dispatch need execution-tool escalation on
this host. Do not run source-generating tests while compilation is active.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest -q specs/open-engine/tests
export PATH="$PWD/ironvenv/lib/python3.14/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH"
export XILINX_XRT=/opt/xilinx/xrt
export OFLM_KEEP_FAILED=1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/ffn-front-replay/front_build.py \
  --spec Models/qwen38-27b/production-consistent/kernels/spec.json \
  --out Models/qwen38-27b/ffn-projections/build
ironvenv/bin/python utilities/attention-entry-replay/resources.py \
  --build Models/qwen38-27b/ffn-projections/build \
  --out Models/qwen38-27b/ffn-projections/resources.json
```

Verify linked code <16384 bytes/core and L1 buffers within 65536, and inspect
`build/final.prj/aie.mlir` for the existing 2048-byte stacks. The isolated
probe's fit does not establish whole-layer resource acceptance; retain the
production export's independent audit. Do not ship this diagnostic xclbin.

## Prepare, run and score

Keep replay output separate from the model, export, source fixture, pools and
build. The complete fixture must match the production manifest, include all
model layers and at least two positions, and declare FP32 FFN norm and
consistent Q4 sums. Preparation verifies model/pool hashes against original
provenance and source act captures against the accepted production evidence.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -u utilities/ffn-front-replay/sweep.py \
  --out Models/qwen38-27b/ffn-projections/run \
  --model Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernels Models/qwen38-27b/production-consistent/kernels \
  --fixture Models/qwen38-27b/production-consistent/decode \
  --pools Models/qwen38-27b/production-full/pools \
  --build Models/qwen38-27b/ffn-projections/build \
  --original-inputs Models/qwen38-27b/production-consistent/input-sha256.json \
  --original-captures Models/qwen38-27b/production-consistent/stage-evidence.json
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  /tmp/qwen38-consistent-harness/run_kernel \
  Models/qwen38-27b/ffn-projections/run/run.cfg
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -u utilities/ffn-front-replay/sweep.py \
  --out Models/qwen38-27b/ffn-projections/run --score
```

Rebuild the native harness from `open_kernels/harness` if `/tmp` was cleared;
use the XRT paths documented in the production integration runbook. Run NPU
processes serially. One pool BO is reloaded per layer, with all positions for
that layer executed before the next reload. The scorer loads one pair of
up/gate matrices at a time; full-model NPU residency is unnecessary.

The input record is exactly `hidden*6` bytes: FP32 residual followed by BF16
norm weights. `read_front()` promotes arrays to FP64 for checking; always use
`pack_input()` before writing the NPU record. Writing those FP64 bytes directly
makes the harness consume the wrong fields and produces non-finite norms. The
first rejected run is retained in `rejected-input-layout/`; a RED → GREEN ABI
regression covers the corrected serialization. It is not acceptance evidence.

Preparation clears stale captures. Scoring removes stale success, revalidates
exact input/hash coverage, source identity and replay configuration, then
records every case. Missing or non-finite captures fail. A numerical failure
retains its report and returns nonzero; do not weaken thresholds to obtain PASS.

Require both independent numerical acceptance and production identity:

- Norm and combined `h` must each match production bytewise, including signed
  zero. A close numerical match cannot substitute for this identity gate.
- Separate up, gate and combined activation must pass maxrel <0.005 against
  independent FP64 references using unrounded RMSNorm of the captured residual.
  The norm gate has the same bound. Existing rounded-norm diagnostics never
  override the primary gate or a failed individual projection.

This is replay from captured pre-FFN residuals. Combine it with the matching
full-depth propagation evidence; it does not establish GGUF conversion parity,
block-prefill accuracy, serving isolation or long-context/performance acceptance.
Model tensors, packed weights, production recipes and export remain unchanged.

## Accepted measurement — 2026-10-18

Production key remains
`ac29fbcd21859648ff9d59f79e77f018b99daec0888f77a693e7ff18b2237032`.
A fresh audit of the unchanged production export passes all 41 sets / 82 binary
hashes. The diagnostic probe built in 23.6 seconds: GEMV/norm text 6400/2944 bytes,
L1 buffer end 33024/53248, stacks 2048. Logical NPU buffers total 0.217039 GiB.
Model plus 64 pool hashes and 192 production act captures match original provenance.

The corrected-input run performs 192 NPU dispatches, covering all 64 layers and
three positions. All 192 independent norm/up/gate/activation cases pass. All 192
norm arrays and 192 activation arrays match the production fixture bytewise.
Worst errors at the unchanged 0.005 bound:

| Gate | Maxrel | Layer / position |
|---|---:|---|
| up | 0.0000919118084458 | 45 / 0 |
| gate | 0.000101679045603 | 48 / 0 |
| norm | 0.0000189397633292 | 45 / 0 |
| activation | 0.000251659393294 | 36 / 0 |

Thirteen new TDD regressions cover selection, compiled-source/header drift,
full coverage, independent numerical versus identity failures, streaming,
provenance, output isolation, stale success and exact FP32/BF16 input packing.
Full host suite: 1363 passed, 47 skipped. Production sources and binaries remain
unchanged; no new full-model resident run is claimed in this sweep.

Accepted evidence is `Models/qwen38-27b/ffn-projections/`: `run/score.json`,
`run/audit.json`, `summary.json`, `resources.json`, production audits,
build/hardware/host-test logs and `stage-evidence.json`. The failed FP64-input
fixture remains separately in `rejected-input-layout/` and must not be counted
as passing evidence. Weights and all generated artifacts remain outside Git.

Together with the retained full-depth decode and resident lifecycle checks,
this closes the scoped 64-layer/three-position FFN projection gate. Complete
source/packing coverage, prefill (including 256-token boundaries), serving and
final resource/performance acceptance remain open.
