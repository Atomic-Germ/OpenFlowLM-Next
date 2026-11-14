---
name: open-norm-bf16-validation
description: Compile and validate fused and grouped RMSNorm BF16 outputs after FP32 arithmetic changes, using Peano and standalone NPU fixtures without model weights.
---

# RMSNorm FP32-to-BF16 output validation

Use after changing `designs/ln/ln.cc`, `ln_xn.cc`, or their vector arithmetic.
Run from the repository root. Keep `mul3bv_32` / `mul3_32` multiplication and
`conv_even` rounding: their result is a **vector**, so BF16 conversion must go
through `accf32(o).to_vector<bfloat16>()`. Calling `o.to_vector` is invalid.
For grouped norms broadcast `inv[g]`, not the `inv` array.

## Compiler regression

```bash
source utilities/env.sh
ironvenv/bin/python -m pytest -q specs/open-engine/tests/test_vecmath_fp32_scalar.py
```

Four tests compile the actual production translation units to AIE2P objects:
fused 2048, streamed 4096, and grouped 2560 with two and four groups. They skip
only when the AIE package or Peano is absent. The remaining tests retain the
three-term precision checks. Python arithmetic tests alone do not check the
C++ API or the grouped branch.

## Build isolated kernel sets

Use a fresh ignored output directory; the commands below use
`Models/ln-bf16-conversion`. Do not run competing exports in the checkout.

```bash
for variant in 2048:1 4096:1 2560:2 2560:4 5120:1; do
  ln_width=${variant%:*}
  ln_groups=${variant#*:}
  LN_N=$ln_width LN_GROUPS=$ln_groups ironvenv/bin/python \
    open_kernels/build_design.py open_kernels/designs/ln/ln.py \
    Models/ln-bf16-conversion/n${ln_width}-g${ln_groups}/build || break
done
```

5120 is the unchanged wide split route (`ln_add2` + `lin_layer/ln_nr`), included
as a control. The other four sets exercise the changed kernels. These standalone
sets do not replace a complete family export or establish whole-model accuracy.

## Generate and run the existing FP64 fixtures

Keep each shape's buffers separately; `make_test.py` otherwise overwrites the
shared design directory. Its existing fixture generator can be redirected:

```bash
ironvenv/bin/python - <<'PY'
import importlib.util, os, shutil, sys
from pathlib import Path
root = Path.cwd()
for n, g in [(2048, 1), (4096, 1), (2560, 2), (2560, 4), (5120, 1)]:
    out = root / 'Models/ln-bf16-conversion' / f'n{n}-g{g}'
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update(LN_N=str(n), LN_GROUPS=str(g), LN_BUILD='build')
    spec = importlib.util.spec_from_file_location('fixture', root / 'open_kernels/designs/ln/make_test.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    fixture.HERE = out
    sys.argv = ['make_test.py', '--runs', '2', '--seed', '0']
    assert fixture.main() == 0
    shutil.copyfile(root / 'open_kernels/designs/ln/compare.py', out / 'compare.py')
PY
for variant in n2048-g1 n4096-g1 n2560-g2 n2560-g4 n5120-g1; do
  ln_out=Models/ln-bf16-conversion/$variant
  flock -w 10 Models/qwen38-27b/.npu.lock \
    open_kernels/harness/out/run_kernel "$ln_out/run.cfg" > "$ln_out/run.log" 2>&1 || break
  ironvenv/bin/python "$ln_out/compare.py" > "$ln_out/compare.log" 2>&1 || break
done
```

Use the existing harness build instructions in `open_kernels/harness/README.md`
if `run_kernel` is absent. NPU access may require execution outside the sandbox.
Check all five logs; do not treat a partially completed loop as a pass.
Keep the existing comparator bounds unchanged: residual maxrel < 1e-6;
BF16 norm maxrel < 8e-3, cosine > 0.999999 and mismatches < N/20.

Validated 2026-11-03: all five builds and NPU comparisons pass, two dispatches
per fixture. BF16 mismatches: 0/2048, 2/4096, 1/2560 (G2), 0/2560 (G4),
2/5120. Minimum cosine 0.99999997. Sources, artifact/input/output hashes and
comparison results are retained in `Models/ln-bf16-conversion/validation.json`.
This is kernel correctness evidence, not a model throughput measurement.

The original all-target Docker build was also rerun successfully after the fix:

```bash
docker run --rm --network none -v "$PWD:/code" -w /code \
  openflowlm-build:ubuntu26 cmake --build build -j 4
```

It exported 19 model specs plus the BERT sets. Host suite: 2385 passed, 47 skipped;
native CTest: 29 passed, 2 skipped. The full family export was built, while NPU
execution in this check was limited to the five standalone norm fixtures.
