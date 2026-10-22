# Task 28: Parallelize Kernel Export

**Phase:** Build acceleration (unrelated to #188 features, but blocks dev iteration)
**Depends on:** None
**Files to modify:**
- `utilities/export-kernels.py` — parallelize the outer loops
- `src/CMakeLists.txt` — pass `-j` to the export target, add `OFLM_KERNEL_JOBS` option
**Estimated complexity:** Small (~80 lines changed)

## Context

**The problem:** `cmake --build --preset linux-default` runs `export_kernels` (an `ALL` target) on **every** build. It compiles 15 open_kernels specs + 5 BERT families **sequentially** — 20 compilations, 1 CPU at a time, 20-100 minutes.

The script already has a `--jobs` flag, but it only parallelizes *within* each spec (the inner loop of `export_qwen36_kernels.py`). The **outer** loop over specs is sequential.

**The fix:** parallelize the outer loops with `ThreadPoolExecutor` (I/O-bound on subprocess calls, so threads are fine — no GIL issue since `subprocess.run` releases the GIL).

**Note:** This task is build infrastructure, not a #188 feature. It's tracked here because it blocks fast iteration on #188.

## What to do

### 1. Parallelize `export_open_kernels()` in `utilities/export-kernels.py`

Current code (around line 135):

```python
def export_open_kernels(py: Path, specs: list[Path], force: bool, jobs: int) -> int:
    """open_qwen36 + dense families (compile-only, no NPU). Returns 0 on success."""
    failed: list[str] = []
    for spec in specs:
        name = spec.stem
        print(f"-- export {name}", flush=True)
        cmd = [str(py), str(EXPORT), "--spec", str(spec), "--jobs", str(jobs)]
        if force:
            cmd.append("--force")
        if subprocess.run(cmd).returncode != 0:
            failed.append(name)
            print(f"   FAILED: {name}", flush=True)
        else:
            print(f"   ok: {name}", flush=True)
    return len(failed)
```

Replace with:

```python
def export_open_kernels(py: Path, specs: list[Path], force: bool, jobs: int) -> int:
    """open_qwen36 + dense families (compile-only, no NPU). Returns 0 on success.

    Specs are independent, so they run in parallel. The inner --jobs is divided
    among the concurrent specs so the total process count stays near `jobs`.
    """
    failed: list[str] = []

    def build_spec(spec: Path) -> tuple[str, bool]:
        name = spec.stem
        print(f"-- export {name}", flush=True)
        inner_jobs = max(1, jobs // max(1, len(specs)))
        cmd = [str(py), str(EXPORT), "--spec", str(spec), "--jobs", str(inner_jobs)]
        if force:
            cmd.append("--force")
        rc = subprocess.run(cmd).returncode
        return name, rc == 0

    if jobs > 1 and len(specs) > 1:
        with ThreadPoolExecutor(max_workers=min(jobs, len(specs))) as ex:
            futures = {ex.submit(build_spec, spec): spec for spec in specs}
            for fut in as_completed(futures):
                name, ok = fut.result()
                if ok:
                    print(f"   ok: {name}", flush=True)
                else:
                    failed.append(name)
                    print(f"   FAILED: {name}", flush=True)
    else:
        for spec in specs:
            name, ok = build_spec(spec)
            if ok:
                print(f"   ok: {name}", flush=True)
            else:
                failed.append(name)
                print(f"   FAILED: {name}", flush=True)

    return len(failed)
```

### 2. Parallelize `export_bert_sets()`

Current code (around line 152):

```python
def export_bert_sets(py: Path, force: bool) -> int:
    """open_npue BERT design sets (needs pyxrt + NPU). Returns 0 on success."""
    spec = json.loads(BERT_SPEC.read_text(encoding="utf-8"))
    common = spec["common"]
    failed: list[str] = []
    for fam in spec["families"]:
        name = fam["name"]
        out = XCLBINS / name
        if (out / "gemm_rtp" / "design.json").is_file() and not force:
            print(f"-- bert {name}: already built (use --force to rebuild)", flush=True)
            continue
        print(f"-- bert {name} ({', '.join(fam['serves'])})", flush=True)
        cmd = [str(py), str(BERT_EXPORT), *fam["args"], *common, "--out", str(out)]
        if subprocess.run(cmd).returncode != 0:
            failed.append(name)
            print(f"   FAILED: {name}", flush=True)
        else:
            print(f"   ok: {name}", flush=True)

    # The built sets and the spec they were built from must agree.
    rc = subprocess.run([str(py), str(BERT_CHECK), "--xclbins", str(XCLBINS)]).returncode
    if rc != 0:
        failed.append("check_design_sets")
        print("   FAILED: check_design_sets (sets disagree with families.json)", flush=True)
    return len(failed)
```

Replace with:

```python
def export_bert_sets(py: Path, force: bool, jobs: int = 1) -> int:
    """open_npue BERT design sets (needs pyxrt + NPU). Returns 0 on success."""
    spec = json.loads(BERT_SPEC.read_text(encoding="utf-8"))
    common = spec["common"]
    failed: list[str] = []

    def build_family(fam: dict) -> tuple[str, bool]:
        name = fam["name"]
        out = XCLBINS / name
        if (out / "gemm_rtp" / "design.json").is_file() and not force:
            print(f"-- bert {name}: already built (use --force to rebuild)", flush=True)
            return name, True
        print(f"-- bert {name} ({', '.join(fam['serves'])})", flush=True)
        cmd = [str(py), str(BERT_EXPORT), *fam["args"], *common, "--out", str(out)]
        rc = subprocess.run(cmd).returncode
        return name, rc == 0

    families = spec["families"]
    if jobs > 1 and len(families) > 1:
        with ThreadPoolExecutor(max_workers=min(jobs, len(families))) as ex:
            futures = {ex.submit(build_family, fam): fam for fam in families}
            for fut in as_completed(futures):
                name, ok = fut.result()
                if ok:
                    print(f"   ok: {name}", flush=True)
                else:
                    failed.append(name)
                    print(f"   FAILED: {name}", flush=True)
    else:
        for fam in families:
            name, ok = build_family(fam)
            if ok:
                print(f"   ok: {name}", flush=True)
            else:
                failed.append(name)
                print(f"   FAILED: {name}", flush=True)

    # The built sets and the spec they were built from must agree.
    rc = subprocess.run([str(py), str(BERT_CHECK), "--xclbins", str(XCLBINS)]).returncode
    if rc != 0:
        failed.append("check_design_sets")
        print("   FAILED: check_design_sets (sets disagree with families.json)", flush=True)
    return len(failed)
```

### 3. Add the import

At the top of `utilities/export-kernels.py`, after the other imports (around line 53):

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
```

### 4. Update the call site in `main()`

Find the call to `export_bert_sets(...)` in `main()` (around line 230) and add the `jobs` argument:

```python
# OLD:
# rc += export_bert_sets(py, a.force)

# NEW:
rc += export_bert_sets(py, a.force, a.jobs)
```

### 5. Update the `--jobs` help text in `main()`

Find the argparse definition (around line 189):

```python
# OLD:
ap.add_argument("-j", "--jobs", type=int, default=1,
               help="kernel-set builds to run in parallel within each spec (default: 1)")

# NEW:
ap.add_argument("-j", "--jobs", type=int, default=0,
               help="parallel kernel builds (default: 0 = auto-detect CPU count)")
```

And after parsing, resolve 0 to the CPU count:

```python
# After a = ap.parse_args()
if a.jobs <= 0:
    import os
    a.jobs = max(1, os.cpu_count() or 4)
```

### 6. Pass `-j` from the CMake target in `src/CMakeLists.txt`

Find the `export_kernels` target (around line 1349):

```cmake
    add_custom_target(export_kernels ALL
        COMMENT "Exporting open NPU kernel xclbins into src/xclbins"
        COMMAND "${Python3_EXECUTABLE}" "${_export_script}"
                "--specs=${OFLM_KERNEL_SPECS}"
                "-j" "${OFLM_KERNEL_JOBS}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
        VERBATIM)
```

### 7. Add the `OFLM_KERNEL_JOBS` option in `src/CMakeLists.txt`

Add it near the `OFLM_BUILD_KERNELS` option (around line 1348):

```cmake
    # Parallel jobs for the kernel export. 0 = auto-detect CPU count.
    # The export runs one subprocess per kernel spec, so this is what turns a
    # 20-100 minute serial build into a 2-10 minute parallel one.
    include(ProcessorCount)
    ProcessorCount(OFLM_NPROC)
    if(NOT OFLM_NPROC)
        set(OFLM_NPROC 8)
    endif()
    set(OFLM_KERNEL_JOBS "${OFLM_NPROC}" CACHE STRING "Parallel jobs for kernel export (0 = auto)")
```

**IMPORTANT:** The `include(ProcessorCount)` and `set(OFLM_KERNEL_JOBS ...)` must be at the **top level** of `src/CMakeLists.txt` (not inside an `if()` block), because the `export_kernels` target definition may be inside a conditional. Put the option definition BEFORE the target definition.

## Test

```bash
# Verify the script still parses
python3 -c "import ast; ast.parse(open('utilities/export-kernels.py').read())" && echo "SYNTAX OK"

# Verify the CLI accepts -j
python3 utilities/export-kernels.py --help 2>&1 | grep -A1 "jobs"

# Verify auto-detection (dry check via python)
python3 -c "
import os
print('detected jobs:', max(1, os.cpu_count() or 4))
"

# Verify the CMake option is set
cmake --preset linux-default 2>&1 | tail -2
grep OFLM_KERNEL_JOBS build/CMakeCache.txt

# Full verification (takes 2-10 min with parallelism)
time cmake --build --preset linux-default --target export_kernels
```

**Note:** A full kernel export needs the ironvenv + NPU. If the environment lacks them, verify the script logic with a mocked subprocess instead:

```bash
python3 - << 'EOF'
import sys, types
from pathlib import Path
sys.path.insert(0, 'utilities')
import export_kernels

# Mock subprocess.run to count concurrent calls
import threading, time
concurrent = 0
max_concurrent = 0
lock = threading.Lock()
real_run = export_kernels.subprocess.run

def fake_run(cmd, *a, **kw):
    global concurrent, max_concurrent
    with lock:
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
    time.sleep(0.1)
    with lock:
        concurrent -= 1
    return types.SimpleNamespace(returncode=0)

export_kernels.subprocess.run = fake_run
specs = [Path(f"spec{i}.json") for i in range(8)]
failed = export_kernels.export_open_kernels(Path("py"), specs, False, jobs=8)
print("failed:", failed)
print("max concurrent:", max_concurrent)
assert max_concurrent > 1, "FAILED: no parallelism detected"
print("OK: parallel export works")
EOF
```

## Success criteria

- [ ] `utilities/export-kernels.py` imports `ThreadPoolExecutor, as_completed`
- [ ] `export_open_kernels()` runs specs in parallel when `jobs > 1`
- [ ] `export_bert_sets()` runs families in parallel when `jobs > 1`
- [ ] Both functions fall back to sequential when `jobs <= 1` or only 1 item
- [ ] `--jobs` defaults to 0 = auto-detect CPU count
- [ ] `export_bert_sets()` call site passes `a.jobs`
- [ ] CMake `OFLM_KERNEL_JOBS` option added (defaults to CPU count)
- [ ] CMake `export_kernels` target passes `-j ${OFLM_KERNEL_JOBS}`
- [ ] `OFLM_KERNEL_JOBS` is defined BEFORE the `export_kernels` target
- [ ] Script passes syntax check
- [ ] Parallelism verified by the mocked test (max_concurrent > 1)
- [ ] Sequential path still works (jobs=1)
- [ ] `oflm` still builds: `cmake --build --preset linux-default --target oflm`
