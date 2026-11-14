---
name: open-qwen38-paired-prefill-production
description: Build the standard 46-set Qwen3.8 paired-prefill export and validate manifest-v4 low-weight companions with the shared runtime.
---

# Standard paired-prefill export

Use the shared runtime skill first. The qwen35 recipe enables paired prefill
only for validated dense H5120/FF17408 geometry at Q4 projections. Manifest v4
maps each high GEMM to its `<high>_low` companion; every low descriptor names
`weight_component: bf16_residual`. Five low streams read the original packed
weights and unchanged GEMM buffers. No model conversion or repacking is needed.

GQP_WEIGHT_LOW=1 specializes the ordinary GEMM design with GQD_WEIGHT_LOW.
It subtracts rounded BF16 dequantization from FP32 dequantization and stores
the residual as BF16. The standard high path remains the default. The export
clears the ambient low-weight knob so it cannot contaminate high builds.

Run host tests before compilation. Never run source-generating tests during
an export. Register this workflow before building; retain failed directories.

```bash
source utilities/env.sh
export OFLM_KEEP_FAILED=1
ironvenv/bin/python open_kernels/export_qwen36_kernels.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --build-root Models/qwen38-27b/prefill-production/build-v3 \
  --out Models/qwen38-27b/prefill-production/kernels-model-v1 -j 2
```

This full export contains 46 sets: the earlier 41 plus five low-weight GEMMs.
Use the exporter and its manifest/toolchain hashes, not hand-edited copies of
historical manifests. Audit actual compiler dependencies and all 32 cores of
the low builds with the retained paired-weight resource checker. Enforce the
original program/L1 limits. Preserve source/build provenance: the exported GQP_WEIGHT_LOW specialization
selects GQD_WEIGHT_LOW in ExternalFunction flags. Check the actual dependency
files against the staged header, and compare the built instruction/core bytes
with retained accepted low kernels. Build logs do not retain the full internal
Peano command line.

Build/runtime acceptance is separate from model acceptance. Compare fresh
shared-runtime captures against retained expanded private hybrid results,
then independent CPU references, both supported schedules and reset. Standard
Core disables stage-major submission for paired routes. Full resident runs
require >=35 GiB available RAM and exclusive NPU access. Sequential long-prefix
precision and FP32 DeltaNet post-output integration remain separate open gates.


## Export and short-logit checkpoint, 2026-10-25

The full export succeeds in build-v3. Preserve failed v1/v2 attempts: Docker-
owned shared staging and locks caused permission errors. `--build-root` now
isolates both (canonical paths keep aliases on one lock); source stages stay
immediately under open_kernels to preserve design-relative imports.

First built kernels-v3 from the checked-in geometry spec. The local model has
real_vocab=248077 versus 248320 in that fixture. Exporting its retained spec to
kernels-model-v1 with the same build-v3 reuses all binaries and creates correct
metadata. `spec_from_model_dir` produces exactly that same spec hash:
ff08e8129933d27b3edad6830486bf757285805118972ffffb3d86ed92640ab0.
Use the model-derived command above for subsequent exports; never edit manifests.

Resource audit: all five low builds cover 32 cores, max .text 5568 bytes,
L1 buffer end 61448, stack 4096. Actual deps.json input hashes match their
compiler inputs and the current dequant header. The existing
`utilities/prefill-paired-weights/build.py::resources` performs the assigned-MLIR
and linked-ELF audit. Results are in prefill-production/resources-v1.json.
Low-identity-v1.json compares the five standard binaries with the retained
expanded candidate: insts.bin exact, final.xclbin equal after masking only the
exporter's known build stamps. All arithmetic/core bytes agree.

Run `utilities/prefill-production/smoke.py` as described in its README. The
unmodified shared cli-v2 and kernels-model-v1 pass both schedules: 44/44 exact
logit files and 42/42 independent CPU logit cases (20 prompt positions and one
continuation per schedule). Smoke-v2 additionally binds reference.json to the
historical preparation artifact hash. It does not rerun historical source
validators after source changes. Available RAM was checked before each request;
no service was started. Full host suite: 1848 passed, 47 skipped.

Retain the model/kernel/reference inputs and JSON reports. This short-logit
smoke does not capture all intermediate layers or prove long-prefix/reset
acceptance. Those checks and FP32 post-output promotion are separate steps.
