# Paired-weight output projection replay

Build private N5120/K6144/T256 Q4_1 control/low kernels, no B reuse. Reuse the
validated low-weight dequant conversion; preserve packing, stream ABI and FP32
four-product sum order. Build a fresh standard harness after the main merge.
Pin compiler dependencies and audit all 32 cores. Keep existing build directories
and failed evidence intact. Register the associated skill before building.

```sh
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-out-replay/build.py \
  --mode control --out Models/qwen38-27b/prefill-out-replay/builds/control
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-out-replay/build.py \
  --mode low --out Models/qwen38-27b/prefill-out-replay/builds/low
ironvenv/bin/python utilities/prefill-upgate-replay/check.py build \
  --out Models/qwen38-27b/prefill-out-replay/harness \
  --xrt-include /home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/include \
  --xrt-lib /home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/lib
```

The baseline is `prefill-prefix-replay/v2`: complete, valid controls, with L53
residual still failing. Verify historical artifact hashes unchanged. Changed
tracked source files must match the pinned fixture revision `aef4a93` in Git,
and their current bytes must match HEAD. Changed artifacts, untracked files and
external dependencies cannot use this exception. Record both source identities;
never relabel historical execution as current-build acceptance. Recursively
validate referenced evidence records, including binaries and compiler deps.

Prepare original/corrected-entry gated prefixes for L34/P17, L37/P7, L53/P17.
Repack out weights and require source/effective identity plus byte identity to
the prior packed pool. Compute FP64 references from original FP32 gated inputs.
Use the retained independent model pre-FFN residual reference and original
incoming residual; do not replace them with a conditioned reference.

Run 72 serial dispatches with exclusive NPU access: 3 layers × 2 input variants
× 2 placements × 3 kernels (production/control/low) × 2 input planes. Every
production high/low product must match retained outputs individually; new control
products must match production. Check summed output, zero inactive columns and
candidate placement identity. All 20 real rows occupy columns 0–19 or 128–147.
Primary projection maxrel <0.005 and rel_fro <=0.005, with strict improvement
in both metrics. Independent selected residual gate remains mandatory and may
still fail from inherited drift. No full-model residency, production promotion,
corrected FFN/logits or throughput claim. Four products replace two.

```sh
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-out-replay/check.py prepare \
  --out Models/qwen38-27b/prefill-out-replay/prepared-v3 \
  --baseline Models/qwen38-27b/prefill-prefix-replay/v2 \
  --capture Models/qwen38-27b/prefill-prefix-sublayers/run
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 \
ironvenv/bin/python utilities/prefill-out-replay/check.py run \
  --out Models/qwen38-27b/prefill-out-replay/run \
  --prepared Models/qwen38-27b/prefill-out-replay/prepared-v3 \
  --build-root Models/qwen38-27b/prefill-out-replay/builds \
  --harness-build Models/qwen38-27b/prefill-out-replay/harness
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest -q \
  specs/open-engine/tests/test_prefill_out_replay.py
```

Peak logical NPU buffers are 19660800 weight bytes, 3145728 input bytes and
5242880 output bytes (**26.75 MiB**), excluding XRT/CPU reference overhead.
Preparation streams weights per layer. Complete measurements with failed
independent residual gates return exit 1 and retain `complete=true`,
`control_valid=true`, `residual_passed=false`, `model_acceptance=false`.
A runtime/provenance failure instead clears completion/control validity.
The original failed preparation directory is retained; a regression test prevents
inventing a binary path from an advisory CLI hash in a run report. Actual binary
paths/hashes come from pinned build-evidence records.

## Validated checkpoint — 2026-10-22

Both builds pass all 32 core budgets: text max **4784/5568 bytes**, L1 buffer
end **61448 bytes**, 801 pinned compiler dependencies each. Fresh harness:
326 compiler dependencies. Toolchain mlir-aie 1.4.2 / llvm-aie
21.0.0.2026080301+c9c5ecb7. XCLBIN SHA256:

- control: `2006b5280c74c29a1c39fdd693f9bf8fda264aaecf50050c998ee4aced467c3a`
- low: `fe5a582d73e832c78e85d040f880302948b626d6aa4cc4e07417ae183e01fdd2`

Both instruction binaries: `73b82f33a8423e6a41c9770104067c44b716c2b06247de1cac821b0fdcd41c1d`.
Harness SHA256: `2e83a83f4246d3e45d3bbe74c15b561988c39f59f0f25192566061cddc63ff7b`.
Preparation verifies **2075 files / 25 records**; 23 changed tracked sources are
bound to the fixture revision and current HEAD separately. No historical hashes
were changed. `prepared/` and `prepared-v2/` retain failed format-validation
attempts; `prepared-v3/` is the complete preparation.

All **72/72 dispatches** finish, with individual production products byte-exact
to retained dumps, new control products exact, zero inactive columns and exact
candidate placement identity. **240/240 projection checks** pass and strictly
improve both errors. Worst candidate maxrel **1.1842201e-5**, rel_fro
**4.8705972e-6**; worst control maxrel **0.0022730411**.

| Selected residual | Original entry + corrected out | Corrected entry + corrected out |
|---|---:|---:|
| L34/P17 | 0.0007008094 | 0.0006087006 |
| L37/P7 | 0.0010534831 | 0.0008481152 |
| L53/P17 | 0.0067071677 (FAIL) | 0.0066072119 (FAIL) |

Each row is byte-identical across both placements. L53 remains above 0.005,
consistent with inherited drift; this valid measurement returns exit 1.
Fourteen new RED-to-GREEN tests; full suite **1665 passed / 47 skipped**.
No full-model execution or production promotion. Next integrate paired-weight
entry/out alongside up/gate in a private CLI, require a fresh post-merge control,
and evaluate actual cross-layer propagation against independent references.
