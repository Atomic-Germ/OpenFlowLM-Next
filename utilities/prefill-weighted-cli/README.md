# Private paired-weight FFN CLI

Build copied standalone CLIs with paired-input GEMM and synchronized layer
residual capture. Control retains the existing paired-input arithmetic;
candidate adds the validated low-weight kernel only for dense FFN up/gate
N34816/K5120/T256. Reject split weights, mismatched roles and stage-major.
No production source, manifest, model packing or kernel is changed.

The original single-dispatch GEMM body remains intact. Candidate sums the two
input products for each weight part, then adds weight parts in FP32 host memory.
Copy high output before low dispatch, keep workspaces per output global and
never dirty mapped device output. The low kernel uses its own hardware context
and the same weight/input/output BO layout; its absolute paths are compiled
into the private candidate and pinned in build provenance.

Reuse the verified `prefill-paired-weights/v2/low` build. Before native execution,
verify source, staged source, compiler dependencies, CLI/cache and low-kernel
hashes. Both CLIs record layer residuals for each real prompt row and continuation.
The private tracer requires one request, NPU decode, --prefill-logits and
--max-tokens 2. References must remain independent of candidate outputs.

```bash
xrt_sdk=/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt
for variant in control candidate; do
  ironvenv/bin/python utilities/prefill-weighted-cli/build.py \
    --variant "$variant" --out "Models/qwen38-27b/prefill-weighted-cli/v2/$variant" \
    --xrt-include "$xrt_sdk/include" --xrt-lib "$xrt_sdk/lib" || break
done
```

Builds use fresh directories, GCC/CMake Release/OpenMP/XRT, run native CTest
and collect all seven CLI translation-unit dependencies. No new XCLBIN.
Prepare fresh copies of the retained 20-token full-depth independent references,
then run `utilities/prefill-slice/check.py run --out RUN --cli CLI` serially with
`LD_LIBRARY_PATH=/opt/xilinx/xrt/lib`, `OPENBLAS_NUM_THREADS=1`, exclusive NPU access
and at least 35 GiB available RAM. Each process loads all 64 layers.
Require exact control logits against the prior paired-input run, unchanged
sequential logits/residuals, identical hybrid schedules, independent logits and
continuation residual gates. Do not infer long-prefix acceptance from the short
prompt or selected-row replays.

Run and independently re-audit with:

```bash
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 \
ironvenv/bin/python utilities/prefill-weighted-cli/run.py \
  --out Models/qwen38-27b/prefill-weighted-cli/v2 \
  --baseline Models/qwen38-27b/prefill-paired-input/paired/run
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-weighted-cli/audit.py \
  --out Models/qwen38-27b/prefill-weighted-cli/v2 \
  --baseline Models/qwen38-27b/prefill-paired-input/paired/run
```

The runner refuses existing run directories and invalidates aggregate success
before preflight. It copies only reference inputs and runs control first;
candidate execution requires exact control logits. The auditor recomputes all
logit and per-layer residual gates from captured arrays, verifies CLI/build and
low-kernel dependencies, and requires complete capture inventories. Sequential
logits, continuation scratch and all residual files must stay byte-identical;
both hybrid schedules must agree in logits and every layer residual.

The first candidate build failed because XRT has ambiguous constructors for a
string literal. Pass `std::string{path}` explicitly. Keep that failed build and
its log; fresh **v2** control/candidate builds pin the corrected builder.
Both use GCC 15.2.0 and pass CTest **4/4**. Their compiler inventories contain
**585 / 586** dependencies. CLI SHA256:

- Control: `b3f0bae0a8838b21dd8b8726c7424de5f6cb08826f252fe105fb7b24f62ac1b6`
- Candidate: `4757cf20ca63eaa3e4d0cdb81563e0786236fa7e31c49636f2e8d668ad5d16c2`

## Native result (2026-10-21)

All six full-resident requests complete: control/candidate × sequential,
hybrid layer-major and hybrid block-major, with a retained 20-token prompt and
one NPU continuation. Control reproduces **66/66 baseline logits bytewise**.
Candidate sequential logits (22 files), continuation scratch and all 64 layer
residual files are exact (**87/87** checks). Both hybrid schedules agree in
22 logit files and 64 residual files per variant.

| Hybrid metric | Paired-input control | Paired-weight candidate |
|---|---:|---:|
| Minimum logit correlation | 0.9999488108 | 0.9999742814 |
| L63 continuation maxrel | 0.0011956505 | 0.0004262400 |
| Passing layer/position residuals per route | 1299/1344 | 1315/1344 |
| First failing residual (L55/P7) | 0.0088329705 | 0.0062761194 |
| Worst residual (L51/P17) | 0.0183013326 | 0.0138594086 |

Both variants pass **63/63 logit gates**. Sequential residuals pass **1344/1344**
unchanged, worst 0.0048820188. The candidate passes **3974/4032** residual gates
across all routes, versus 3942/4032 for control. However **29 failures per hybrid
route remain** at the original maxrel <0.005 threshold. The aggregate audit
correctly returns **1**, with capture_valid=true, passed=false and
model_acceptance=false. This is numerical failure, not a build/execution error.

A logit/continuation pass does not establish intermediate-layer accuracy.
Capture conditioned sublayers at L55/P7 and L51/P17 next, with byte-exact
baseline checks, before extending to long-prefix runs or production integration.
The retained long-prefix model failures remain unresolved.

TDD: three build/ownership tests plus two audit regressions fail before their
implementations, then pass. Host suite: **1550 passed, 47 skipped**; CTest
**4/4 per v2 build**. Evidence lives under
`Models/qwen38-27b/prefill-weighted-cli/`, including preserved failed-build logs,
v2 build records, reference/capture hashes and v2/comparison.json. No production
arithmetic, kernels, model weights or manifest changed.
