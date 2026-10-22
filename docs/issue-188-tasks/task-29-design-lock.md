# Task 29: Per-Design File Lock for Kernel Builds

**Phase:** Build acceleration (fixes the race introduced by Task 28's parallel export)
**Depends on:** Task 28
**Files to modify:** `open_kernels/export_qwen36_kernels.py`
**Estimated complexity:** Small (~60 lines)

## Context

**The bug:** Task 28 parallelized the export across specs. But 50 `build_dir` paths are **shared between specs** (e.g. `lm_head_q4/build_151936` is wanted by `qwen25-3b`, `qwen3-4b`, and `qwen3-8b`). Two specs building the same design concurrently collide:

```
error: unable to rename temporary '.../gemv_q4_prep_rt-301e1315.o.tmp'
   to output file '.../gemv_q4_prep_rt.o': 'No such file or directory'
```

Root cause: `open_kernels/build_design.py:34` does `shutil.rmtree(out / "final.prj")` before building. Two processes in the same `out` dir delete each other's work dir, then race on the `.o` files.

**The fix:** A cross-process **file lock per build_dir**. Independent designs still build in parallel; shared ones serialize.

## What to do

### 1. Add the lock helper to `open_kernels/export_qwen36_kernels.py`

Add the imports (around line 43-55, with the other stdlib imports):

```python
import contextlib
import errno
```

Add the lock machinery right after the `CLEAR` constant (around line 66):

```python
# ————————————————————————————————————————————————————————————————
# Per-design build lock
# ————————————————————————————————————————————————————————————————
# 50 build_dir paths are shared between specs (e.g. lm_head_q4/build_151936 is
# wanted by qwen25-3b, qwen3-4b and qwen3-8b). build_design.py wipes its work
# dir (final.prj) before compiling, so two specs building the same design at
# the same time delete each other's objects and fail with "unable to rename
# temporary ... .o.tmp". The export runs specs in parallel, so the collision is
# the default, not the exception. One lock per build_dir makes the shared
# designs serialize while everything else stays parallel.

LOCKS = DESIGNS / ".locks"


@contextlib.contextmanager
def design_lock(out: Path):
    """Hold an exclusive cross-process lock for one build directory."""
    LOCKS.mkdir(parents=True, exist_ok=True)
    # The build dir is designs/<design>/<build_dir>; flatten it for the lock name.
    key = str(out.relative_to(DESIGNS)).replace("/", "__").replace("\\", "__")
    lock_path = LOCKS / f"{key}.lock"
    fh = open(lock_path, "a+b")
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        fh.close()
```

### 2. Use the lock in `_build_one()`

In `_build_one()` (around line 161-195), wrap the build subprocess in the lock. The current code is:

```python
def _build_one(name: str, sets: dict, spec_file: Path) -> tuple[str, Path]:
    """Build one kernel set. Worker for the parallel build; returns (name, bdir)."""
    src, out, knobs = DESIGNS / sets[name]["design"], DESIGNS / sets[name]["build_dir"], sets[name]["env"]
    env = {k: v for k, v in os.environ.items() if k not in CLEAR}
    env.update(knobs)
    env["OPEN_KERNELS_SPEC"] = str(spec_file)
    t0 = time.time()
    knob_str = " ".join(f"{k}={v}" for k, v in knobs.items())
    print(f"[{name}] {knob_str} python build_design.py {src.relative_to(HERE).as_posix()} "
          f"{out.relative_to(HERE).as_posix()}", flush=True)
    # Passed through as it comes, and watched for the one failure a user cannot act on: aiecc
    # reports a core whose program does not fit only as an invalid ELF.
    p = subprocess.Popen([sys.executable, str(HERE / "build_design.py"), str(src), str(out)], env=env,
                         cwd=str(HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tail, overflow = b"", False
    for chunk in iter(lambda: p.stdout.read1(4096), b""):
        sys.stdout.buffer.write(chunk)
        sys.stdout.flush()
        tail = (tail + chunk)[-4096:]
        overflow = overflow or b"Overflow of program memory" in tail
    if p.wait() != 0:
        if overflow:
            raise RuntimeError(f"[{name}] build FAILED: a core's program is larger than its {PROGRAM_MEMORY} B of "
                               "program memory. This is a bug in the kernels for this model, not in your setup; "
                               "please report it with this log.")
        # Repeat the final compiler output in the exception: the outer exporter
        # continues with other models, and parallel workers can interleave logs.
        detail = tail.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"[{name}] build FAILED ({p.returncode})\n"
                           f"Last build output:\n{detail or '(no output)'}")
    print(f"[{name}] built in {time.time() - t0:.0f}s", flush=True)
    used = fullest_core(out)
    if used:
        print(f"[{name}] program memory: fullest core {used} of {PROGRAM_MEMORY} B ({PROGRAM_MEMORY - used} B free)")
    return name, out
```

Change it to hold the lock around the build:

```python
def _build_one(name: str, sets: dict, spec_file: Path) -> tuple[str, Path]:
    """Build one kernel set. Worker for the parallel build; returns (name, bdir)."""
    src, out, knobs = DESIGNS / sets[name]["design"], DESIGNS / sets[name]["build_dir"], sets[name]["env"]
    env = {k: v for k, v in os.environ.items() if k not in CLEAR}
    env.update(knobs)
    env["OPEN_KERNELS_SPEC"] = str(spec_file)
    t0 = time.time()
    knob_str = " ".join(f"{k}={v}" for k, v in knobs.items())
    print(f"[{name}] {knob_str} python build_design.py {src.relative_to(HERE).as_posix()} "
          f"{out.relative_to(HERE).as_posix()}", flush=True)
    # The lock keeps two specs out of the same build dir at once: the designs
    # are shared, and build_design.py wipes its work dir before compiling.
    with design_lock(out):
        # Passed through as it comes, and watched for the one failure a user cannot act on: aiecc
        # reports a core whose program does not fit only as an invalid ELF.
        p = subprocess.Popen([sys.executable, str(HERE / "build_design.py"), str(src), str(out)], env=env,
                             cwd=str(HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        tail, overflow = b"", False
        for chunk in iter(lambda: p.stdout.read1(4096), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            tail = (tail + chunk)[-4096:]
            overflow = overflow or b"Overflow of program memory" in tail
        if p.wait() != 0:
            if overflow:
                raise RuntimeError(f"[{name}] build FAILED: a core's program is larger than its {PROGRAM_MEMORY} B of "
                                   "program memory. This is a bug in the kernels for this model, not in your setup; "
                                   "please report it with this log.")
            # Repeat the final compiler output in the exception: the outer exporter
            # continues with other models, and parallel workers can interleave logs.
            detail = tail.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"[{name}] build FAILED ({p.returncode})\n"
                               f"Last build output:\n{detail or '(no output)'}")
    print(f"[{name}] built in {time.time() - t0:.0f}s", flush=True)
    used = fullest_core(out)
    if used:
        print(f"[{name}] program memory: fullest core {used} of {PROGRAM_MEMORY} B ({PROGRAM_MEMORY - used} B free)")
    return name, out
```

**The only change is indenting the build block under `with design_lock(out):`.** Everything else stays.

### 3. Make sure the lock dir is gitignored

Check `.gitignore` for `open_kernels/designs/`. The existing rules (around line 48) cover `build*/` dirs and build artifacts, but NOT a `.locks/` directory. Add this line to the root `.gitignore`, right after the `open_kernels/designs/*/build*/` rule:

```
open_kernels/designs/.locks/
```

## Test

```bash
# Syntax
python3 -c "import ast; ast.parse(open('open_kernels/export_qwen36_kernels.py').read())" && echo "SYNTAX OK"

# Lock serializes access to the same build dir
cd open_kernels && python3 - << 'EOF'
import importlib.util, threading, time
from pathlib import Path
spec = importlib.util.spec_from_file_location("ek", "export_qwen36_kernels.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

out = m.DESIGNS / "lm_head_q4" / "build_151936"
concurrent = 0
max_concurrent = 0
lock = threading.Lock()

def worker():
    global concurrent, max_concurrent
    with m.design_lock(out):
        with lock:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
        time.sleep(0.3)
        with lock:
            concurrent -= 1

ts = [threading.Thread(target=worker) for _ in range(4)]
for t in ts: t.start()
for t in ts: t.join()
print("max concurrent in same build_dir:", max_concurrent)
assert max_concurrent == 1, "FAILED: lock did not serialize"
print("OK: design lock serializes same build_dir")

# Different build dirs do NOT block each other
concurrent = 0
max_concurrent = 0
def worker2(d):
    global concurrent, max_concurrent
    with m.design_lock(m.DESIGNS / d):
        with lock:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
        time.sleep(0.3)
        with lock:
            concurrent -= 1
ts = [threading.Thread(target=worker2, args=(f"attn_block/build_s{i}",)) for i in range(4)]
for t in ts: t.start()
for t in ts: t.join()
print("max concurrent across different build_dirs:", max_concurrent)
assert max_concurrent > 1, "FAILED: different designs should not block"
print("OK: different build_dirs stay parallel")
EOF
cd ..

# Full build (this is the real test)
cmake --build --preset linux-default --parallel 16
```

## Success criteria

- [ ] `design_lock()` context manager added to `open_kernels/export_qwen36_kernels.py`
- [ ] `_build_one()` holds the lock around the build subprocess
- [ ] Lock file path is derived from the build_dir (unique per design)
- [ ] POSIX: `fcntl.flock` exclusive lock
- [ ] Windows: `msvcrt.locking` (release build there)
- [ ] Lock released in a `finally` block (no deadlock on failure)
- [ ] Lock dir `open_kernels/designs/.locks/` is gitignored
- [ ] Syntax check passes
- [ ] Same build_dir → max_concurrent == 1 (serialized)
- [ ] Different build_dirs → max_concurrent > 1 (still parallel)
- [ ] Full build succeeds: `cmake --build --preset linux-default --parallel 16`
- [ ] No "unable to rename temporary" errors in the build log
