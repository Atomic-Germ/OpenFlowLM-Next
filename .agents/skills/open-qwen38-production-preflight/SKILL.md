---
name: open-qwen38-production-preflight
description: Normalize an existing dense Qwen3.5 Q4NX config, validate Qwen3.8-27B source tensors, build the complete 41-set production recipe on Linux, and verify slice and full-depth NPU decode.
---

# Qwen3.8-27B production preflight and Linux build

Use the shared `qwen35` recipe. This workflow validates metadata, selected source
weights, compilation, an eight-layer NPU slice and three tokens at full depth.
Independent state/head/FFN-partial, serving and prefill acceptance remain open.

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
For compatibility checks after a recipe update, use `make_decode.py
--kernel-dir <export>`: it verifies full manifest compatibility except the
source build key, checks all required binary hashes, and keeps that export's
original build identity. It does not rebuild or relabel binaries. `--cfg-only`
must select the same export as reference generation. A different layout,
packing plan, geometry or context capacity is rejected.
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
to the harness, and `--twice` reproduced the token sequence. The CPU replica
mixes FP64 norm/state/attention math with FP32 projections and LM-head
accumulation. Saved captures use FP32.

`--twice` checks generated IDs and overwrites dumps with the second request.
`--det-full` matched all observed buffers and recurrent-state hashes in three
reset replays of the input sequence; it compares to an NPU reference run, not
a CPU state reference. Current KV rows are observed, not the unused cache.
No throughput claim: the CLI could not set the power mode, and CPU reference
generation ran concurrently with runtime validation.

Local stage-3 evidence is in
`Models/qwen38-27b/production-slice/results.json` and `logs/`.

## Full-depth first-token gate

Use a separate fixture directory; do not overwrite the eight-layer reference.
The full-depth fixture takes about 17 GB on disk. On this machine, allow about
30 GB of additional memory for a resident 64-layer NPU run. This is a planning
estimate, not a benchmark. The old CPU decoder took roughly 30 minutes per
reference token; contiguous Q4 gathers added on 2026-10-10 remove its main
dequantization bottleneck.
Keep NPU jobs serial and check available memory before starting.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python open_kernels/model/make_decode.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --layers 64 --tokens 1 --out Models/qwen38-27b/production-full
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel Models/qwen38-27b/production-full/run_decode.cfg
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE \
  /tmp/qwen38-main-status-runtime/open_qwen36_cli \
  --model Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernels Models/qwen38-27b/production-kernels \
  --ids 248045 --max-tokens 3 --layers 64 \
  --dump-logits Models/qwen38-27b/production-full/engine --twice
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python open_kernels/model/compare_decode.py \
  --out Models/qwen38-27b/production-full \
  --runtime-prefix Models/qwen38-27b/production-full/engine
```

`--runtime-prefix` verifies every declared token's runtime dump against the
harness byte for byte, alongside the unchanged CPU-reference gates. A one-token
fixture covers only `engine_t0.bin`, even if the runtime generated more steps.
It rejects missing/malformed/non-finite dumps, one-ULP changes and signed-zero
differences. Its 18 regression cases bring the comparator suite to 69 tests.

2026-10-08: the first token at all 64 layers passed, logit correlation
0.9999973866, argmax 8678, worst residual maxrel 0.002713315 (layer 57).
All 64 residual gates and 66 harness dispatches passed; runtime/harness logits
were byte-identical. Full-depth runtime generated 8678, 198, 2 twice, and three
reset replays matched all observed buffers and recurrent-state hashes.
The full open-engine suite passed 845 tests with 47 skipped. Metrics/reference
metadata are retained in `Models/qwen38-27b/production-full/results.json`.

For reset replay, force the actual input sequence from the generation above:
seed 248045 followed by the first two generated IDs (8678 and 198 in this run).

```bash
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE \
  /tmp/qwen38-main-status-runtime/open_qwen36_cli \
  --model Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernels Models/qwen38-27b/production-kernels \
  --ids 248045,8678,198 --layers 64 --det-step 3 --det-full
```

Short chat smoke checks can use `src/open_qwen36/chat.py` with the same `--model`,
`--kernels`, `--layers 64`, `--exe /tmp/qwen38-main-status-runtime/open_qwen36_cli`
and `--max-tokens 8 --twice`. Its no-thinking Qwen prompt matches this model's
template with `enable_thinking=false` for these single user turns. Alternatively,
render the model's `chat_template.jinja` with Jinja2, tokenize with
`tokenizers.Tokenizer` and feed the comma-separated IDs through `--ids-file`.
The local prompt JSON files retain the actual rendered strings, IDs and template
hash. Text is evaluated up to the first `<|im_end|>`: the diagnostic CLI keeps
generating up to `--max-tokens`, including past EOS.

Do not count reset repeatability or a correct short answer as an independent
multi-token CPU comparison. Keep full-depth multi-token, per-head/state,
serving and hybrid block-prefill acceptance separate.

## 2026-10-09 continuation: explicit export, memory gate

Upstream `c2ef0e8` is merged. Native runtime build, CTest 3/3 and the block-host
fixture pass. The model-derived manifest remains equal to the retained
production export except its source build key. All 82 export hashes pass;
working-directory xclbins now have different XRT headers, so the exporter's
strict `--check` rejects those copies. Do not silently use them for a reference
run. `production-kernels-upstream/` is an unverified `--no-build` snapshot,
marked `VALIDATION_FAILED.txt`; it is not a new validated build.

Use the existing `production-kernels/` explicitly for the 64-layer,
three-token compatibility run:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -u open_kernels/model/make_decode.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-kernels \
  --layers 64 --tokens 3 --out Models/qwen38-27b/production-multitoken \
  --pool-dir Models/qwen38-27b/production-full/pools --reuse-pools
```

Pool reuse here is justified by the unchanged container and packing plan;
`--reuse-pools` alone checks sizes, not content provenance. Do not generalize
it to changed weights or packing. The interrupted directory has no completed
reference contract; regenerate it, not `--cfg-only`.

The attempt ran out of available RAM while loading 50/64 layers; no new
full-depth numerical result was produced. Wait for roughly 35–40 GiB available
before attempting the resident NPU run, and finish CPU preparation first when
memory is constrained. The user paused hardware work until memory was freed;
the 2026-10-10 continuation below completed this run. Logs are under
`production-multitoken/logs/`; old stage-3/4 results must not be reported as
reruns on the updated runtime.

Artifacts/logs for this run: `Models/qwen38-27b/production-kernels/`,
`Models/qwen38-27b/production-input-sha256.txt`,
`/tmp/qwen38-production-model-build.log`, `/tmp/qwen38-production-source-compare.log`.

## 2026-10-10: full-depth three-token gate completed

Resumed with about 62 GiB available; finish CPU reference generation before
starting NPU jobs. The preparation command above completed with 195 pinned
captures and 198 dispatches. Use the same verified export for the runtime.
No new kernel/library build was required.

Q4 preparation now uses contiguous gathers with unchanged FP32 values. Check
real-tensor byte identity and CPU dequantization timing with:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/benchmark-q4-decode.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2
```

Layer 0 FFN up projection: 10880 chunks, byte-identical, 7.94 s legacy versus
0.236 s contiguous in one measurement (33.6x). This does not measure NPU
inference. All 65 first-token reference captures match `production-full/`
bytewise. The scalar-layout regression suite passes 8 tests; the complete
open-engine suite passes 870 tests with 47 skipped.

After the CPU preparation above completes:

```bash
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel Models/qwen38-27b/production-multitoken/run_decode.cfg
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE \
  /tmp/qwen38-main-status-runtime/open_qwen36_cli \
  --model Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernels Models/qwen38-27b/production-kernels \
  --ids 248045 --max-tokens 3 --layers 64 \
  --dump-logits Models/qwen38-27b/production-multitoken/engine --twice
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python open_kernels/model/compare_decode.py \
  --out Models/qwen38-27b/production-multitoken --tokens 3 \
  --runtime-prefix Models/qwen38-27b/production-multitoken/engine
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE \
  /tmp/qwen38-main-status-runtime/open_qwen36_cli \
  --model Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernels Models/qwen38-27b/production-kernels \
  --ids 248045,8678,198 --layers 64 --det-step 3 --det-full
```

Results: all 192 residuals pass (worst maxrel 0.004046247 at layer 63 / step 1),
logit correlations 0.9999973866 / 0.9999920389 / 0.9999972735, equal argmax
8678 / 198 / 2 and byte-identical runtime/harness logits at all three steps.
Repeated request reproduced IDs; reset replay reported 0/3 differing runs.
Reset replay still compares with another NPU run, not independent CPU state.
The power-mode request failed, so this is not an NPU benchmark.

Evidence: `production-multitoken/results.json` and `logs/*20261010*` under the
model directory. `prepare-20261010.log` is the cancelled slow baseline;
`prepare-contiguous-20261010.log` is the completed reference generation.
Independent state/head/FFN-partial checks, block-prefill and serving remain open.

## FFN down-partial checks without a full resident model

The current dense recipe retains both down outputs in each `y_actL_tN.bin`:
`A_OUT2` / `A_OUT2B` for linear layers and `AA_OUT2` / `AA_OUT2B` for attention
layers. The two K ranges are [0,8192) and [8192,17408). The captured `h` is FP32.
Use recipe offsets after verifying manifest compatibility; do not interpret an
old act buffer with a different layout.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python open_kernels/model/ffn_partials.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-kernels \
  --out Models/qwen38-27b/production-multitoken --layers 0,3 \
  > Models/qwen38-27b/production-multitoken/ffn-partials-slice.json
```

Pass a comma-separated list of all covered indices to check more layers. The
checker only materializes one layer's down matrix at a time. It verifies every
fixture position for the selected layers, exact capture sizes, finite consumed
values and export/fixture identity. JSON contains capture hashes and per-stage
metrics. Both partials, their sum and the residual must independently satisfy
maxrel < 0.005, including when errors cancel in the sum.

The FP64 reference is conditioned on captured NPU `h` and pre-FFN residual.
This checks down arithmetic/packing and closure only. Up/gate/norm, attention
head and recurrent-state correctness need separate references. Checking saved
captures does not rerun the updated runtime on hardware. No xclbins or libraries
need rebuilding to use this checker.

2026-10-10: checking all 64 layers and three retained positions found one
failing boundary: layer 60, position 2, partial 1 maxrel 0.0073258458. The
partial-0 maximum is 0.0043589308; sum 0.0049578935; closure 0.0022406553.
Reproduce the input-rounding diagnosis (expected nonzero exit, not acceptance):

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python open_kernels/model/ffn_partials.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-kernels \
  --out Models/qwen38-27b/production-multitoken --layers 60 --diagnose-bf16
```

`gemv_q4_prep_f32_rt` rounds the FP32 activation to BF16. The rounded-input
diagnostic reduces the failing partial's error to 0.0000918061. The primary
comparison deliberately retains FP32 `h`, the 0.005 bound and FAIL status.
Do not mark the two-piece FFN gate complete from the diagnostic PASS or the
passing residual/logit gates. Next work is an isolated precision reproduction
and a kernel correction validated against the unchanged primary reference.
