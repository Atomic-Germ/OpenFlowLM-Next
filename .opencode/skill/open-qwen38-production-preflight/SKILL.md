---
name: open-qwen38-production-preflight
description: Normalize an existing dense Qwen3.5 Q4NX config, validate Qwen3.8-27B source tensors, build the complete 41-set production recipe on Linux, and verify an eight-layer NPU slice.
---

# Qwen3.8-27B production preflight and Linux build

Use the shared `qwen35` recipe. This workflow validates metadata, selected source
weights, compilation and an eight-layer NPU slice; it does not establish
full-model NPU accuracy.

## Existing model metadata

Keep models and downloaded source files under ignored `Models/`. The canonical
local model directory for this run is
`Models/qwen38-27b/Qwen3.8-27B-NPU2`. It holds a hard link to the verified
`model.q4nx` and copies of the tokenizer/config assets, so changing metadata
leaves the original directory intact without copying 19 GB of weights.
Never convert or overwrite a hard-linked weight file; create a new file for a
new conversion. Normalization only opens weights for reading.

```bash
PYTHONPATH=/tmp/oflm-main-merge-deps OPENBLAS_NUM_THREADS=1 \
  ironvenv/bin/python utilities/q4nx-build/convert.py --normalize-config \
  -i Models/qwen38-27b/Qwen3.8-27B-NPU2
```

The dependency directory above is local to this machine; in a fresh environment
install `utilities/q4nx-build/requirements.txt` in the converter environment.
The command works offline, validates header/ranges/layer count/norm widths,
uses `inject_oflm_keys`, keeps `model_type=qwen3_5`, backs up the original as
`config.json.before-normalize`, and atomically updates config. It is idempotent.
An existing backup is never overwritten. This is only for existing dense
Qwen3.5-family containers; fresh conversion already assembles its own assets.

## Independent source comparison

Pinned source: `Qwen/Qwen3.8-27B`, revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
The source index places text layers 0 and 3 in
`model-00001-of-00018.safetensors` (3,966,730,552 bytes).
Download from that revision into `Models/qwen38-27b/source/`.
SHA256: `ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c`.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python open_kernels/model/container_vs_hf.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --hf-shard Models/qwen38-27b/source/model-00001-of-00018.safetensors \
  --layers 0,3
```

2026-10-08: 25 tensors, `ALL MATCH`, minimum correlation 0.996756. Quantized
projections are compared as the NPU packing reads them, including the Q8-to-Q4
fallback. AB correlations are about 0.999984; conv and DeltaNet scalar parameters
are about 1. Missing requested layers and non-finite comparisons fail.
This covers two representative layers, not every tensor in all 64 layers.

Verified local `model.q4nx` SHA256:
`5f5c282ae81b63df34e5f9ce42c616d73b454691c4b8d253027b2e08a1fe3d84`.

## Complete production export

Toolchain: Python 3.14.4, mlir-aie 1.4.2,
llvm-aie `21.0.0.2026080301+c9c5ecb7`, XRT tools in `/opt/xilinx/xrt/bin`.
The actual Peano path here is Python 3.14, not the older Python 3.12 path.

```bash
source ironvenv/bin/activate
export PATH="$PWD/ironvenv/lib/python3.14/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH"
export XILINX_XRT=/opt/xilinx/xrt
export OFLM_KEEP_FAILED=1
python open_kernels/export_qwen36_kernels.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --out Models/qwen38-27b/production-kernels -j 1
```

Use `--model-dir`: the tokenizer has 248077 real IDs while the padded output
has 248320 lanes. A shape-only checked-in spec has different metadata/hash.
Keep the exported manifest and toolchain hashes together with the binaries.
Check config compatibility through the real C++ runtime parser without XRT:

```bash
c++ -std=c++17 -O2 -Isrc -Isrc/include -Iopen_kernels/harness \
  utilities/check-open-model.cpp src/open_qwen36/manifest.cpp -o /tmp/check-open-model
/tmp/check-open-model Models/qwen38-27b/production-kernels/manifest.json \
  Models/qwen38-27b/Qwen3.8-27B-NPU2/config.json
```

Do not run two exports concurrently in one checkout: generated kernel sources
are shared. On this sandbox, Python 3.14 multiprocessing requires an unsandboxed
build because its forkserver binds a local socket. That failure is an environment
restriction, not a kernel compilation error.

The full export contains four decode sets (`lx`, `ax`, `ln`, `lm_head_q8`),
32 attention-product sets at AG_M1536 and five GEMMs. All 41 sets are required
by the complete manifest. A partial `--only` build is not a complete export.
Use `utilities/core_sizes.py` for retained core ELF program sizes.

Measured maximum .text per set: lx 15584, ax 14400, ln 3424, lm_head_q8 2432,
attention products 1984, GEMMs 4784 bytes; each must remain below 16384.
A successful compile does not validate numerical output or runtime behavior.

The 2026-10-08 final export has 41 sets / 82 verified binary hashes and complete
manifest file coverage. Spec hash is
`sha256:ff08e8129933d27b3edad6830486bf757285805118972ffffb3d86ed92640ab0`;
build key is `sha256:7424a5e3be78958995870b6b37a5551a07a0969de477310bf5d7599b09a0c092`.
The normalized config passes `check-open-model`; the original nested config is
refused with `config.json lacks 'head_dim'`. See
`Models/qwen38-27b/production-preflight.json` for the local audit summary.

## Eight-layer hardware gate (Linux)

Regenerate reference fixtures with the current `make_decode.py`: comparison
requires `decode_reference.json` and enforces residual error as well as logits.
The fixture occupies about 3.2 GB. CPU generation takes several minutes and
publishes the reference metadata only after all steps complete.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python open_kernels/model/make_decode.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --layers 8 --tokens 3 --out Models/qwen38-27b/production-slice
```

The generated harness config uses design **build directories**, whereas the
runtime uses the export. Verify both against the export's hashes before testing:

```bash
ironvenv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('Models/qwen38-27b/production-kernels')
manifest = json.loads((root / 'manifest.json').read_text())
hashes = json.loads((root / 'toolchain.json').read_text())['sha256']
for name, expected in hashes.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
for key in ('lx', 'ax', 'ln', 'lm_head_q8'):
    build = Path('open_kernels/designs') / manifest['builds'][key]['build_dir']
    for name in ('final.xclbin', 'insts.bin'):
        assert hashlib.sha256((build / name).read_bytes()).hexdigest() == hashes[f'{key}/{name}'], (key, name)
print('Export and harness artifacts verified')
PY
```

Build the native harness and CLI. Set `QWEN_XRT_SDK` to the installed XRT SDK;
on this machine it is the driver checkout below (the runtime remains in
`/opt/xilinx/xrt/lib`). These commands rebuild host executables, not xclbins.

```bash
QWEN_XRT_SDK=/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt
cmake -S open_kernels/harness -B open_kernels/harness/out \
  -DXRT_INCLUDE_DIR="$QWEN_XRT_SDK/include" -DXRT_LIB_DIR="$QWEN_XRT_SDK/lib"
cmake --build open_kernels/harness/out -j4
cmake -S src/open_qwen36 -B /tmp/qwen38-main-status-runtime \
  -DXRT_INCLUDE_DIR="$QWEN_XRT_SDK/include" -DXRT_LIB_DIR="$QWEN_XRT_SDK/lib"
cmake --build /tmp/qwen38-main-status-runtime -j4
ctest --test-dir /tmp/qwen38-main-status-runtime --output-on-failure
```

Run NPU jobs serially. XRT device access requires execution outside this
machine's sandbox. Do not rewrite references or loosen thresholds on failure.

```bash
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel Models/qwen38-27b/production-slice/run_decode.cfg
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python open_kernels/model/compare_decode.py \
  --out Models/qwen38-27b/production-slice --tokens 3
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE \
  /tmp/qwen38-main-status-runtime/open_qwen36_cli \
  --model Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernels Models/qwen38-27b/production-kernels \
  --ids 248045 --max-tokens 3 --layers 8 \
  --dump-logits Models/qwen38-27b/production-slice/engine --twice
ironvenv/bin/python - <<'PY'
from pathlib import Path
p = Path('Models/qwen38-27b/production-slice')
for t in range(3):
    suffix = '' if t == 0 else f'_t{t}'
    got = (p / f'engine_t{t}.bin').read_bytes()
    assert len(got) == 248320 * 4
    assert got == (p / f'y_logits{suffix}.bin').read_bytes(), t
print('All three runtime/harness logits are byte-identical')
PY
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE \
  /tmp/qwen38-main-status-runtime/open_qwen36_cli \
  --model Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernels Models/qwen38-27b/production-kernels \
  --ids 248045,220,220 --layers 8 --det-step 3 --det-full
```

2026-10-08 on Ryzen AI 9 365 / Strix XDNA2: 30 harness dispatches completed;
logit correlations 0.9999991073 / 0.9999988538 / 0.9999990533, argmax 220 at each
step. All 24 residual comparisons passed; worst normalized maximum error
0.001397009 (layer 4, step 0), below 0.005. Runtime logits were byte-identical
to the harness, and `--twice` reproduced the token sequence. The CPU layer
replica uses FP64, but CPU LM-head accumulation and saved captures use FP32.

`--twice` checks generated IDs and overwrites dumps with the second request.
`--det-full` matched all observed buffers and recurrent-state hashes in three
reset replays of the input sequence; it compares to an NPU reference run, not
a CPU state reference. Current KV rows are observed, not the unused cache.
No throughput claim: the CLI could not set the power mode, and CPU reference
generation ran concurrently with runtime validation.

Next: full-depth inference and prompt checks, with sequential NPU and hybrid
block-prefill results recorded separately. The slice does not close those
gates or per-head/state numerical validation. Local stage-3 evidence is in
`Models/qwen38-27b/production-slice/results.json` and `logs/`.

Artifacts/logs for this run: `Models/qwen38-27b/production-kernels/`,
`Models/qwen38-27b/production-input-sha256.txt`,
`/tmp/qwen38-production-model-build.log`, `/tmp/qwen38-production-source-compare.log`.
