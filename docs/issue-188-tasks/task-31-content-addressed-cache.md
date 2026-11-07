# Task 31: Content-Addressed Cache for Kernel Designs

**Phase:** Build performance
**Depends on:** Task 29 (design lock), Task 30 (per-spec source isolation)
**Files to modify:** `open_kernels/export_qwen36_kernels.py`
**Estimated complexity:** Medium (~90 lines)

## Context

**The waste (measured):** the last full export ran **340** builds for **130** unique designs — **210 redundant builds (61%)**.

| Design | Rebuilt |
|--------|---------|
| `attn_block/build_s256` | 7× |
| `attn_block/build_pv256` | 7× |
| `attn_block/build_s512` | 7× |
| ... 50 shared build_dirs total | up to 9× |

Average build 16s → **~55 min of CPU time** spent rebuilding byte-identical designs.

**Why it is safe to cache:** the 50 shared `build_dir`s all belong to `attn_block`, `gemm_q4_prefill`, `lm_head_q4`, `ln`. Verified:
- None of these 4 designs has a `gen_kernels.py` (no per-spec sources)
- None of them reads `OPEN_KERNELS_SPEC` (not spec-dependent)
- Their `build_dir` name encodes the compile-time knobs (`_q8` quant variant, `_m1024` ag_m), and the knobs themselves are in `sets[name]["env"]` (`AG_M`, `AG_K`, `AG_N`, `AG_COLS`, …)

So `build_dir` + design sources + knobs ⇒ identical output.

**The fix:** a content key stored next to the build output. If the key matches and the outputs exist, skip the build.

## What to do

### 1. Add the content key function

In `open_kernels/export_qwen36_kernels.py`, next to `design_lock()` (around line 85-112):

```python
# ————————————————————————————————————————————————————————————————
# Content-addressed build cache
# ————————————————————————————————————————————————————————————————
# 50 build_dir paths are shared between specs: attn_block/build_s256 is wanted
# by 7 specs, and the last full export ran 340 builds for 130 unique designs.
# The shared designs (attn_block, gemm_q4_prefill, lm_head_q4, ln) have no
# gen_kernels.py and do not read OPEN_KERNELS_SPEC, so their output is a
# function of (design sources, compile-time knobs) alone -- the build_dir name
# already encodes the spec-dependent parts. Hash those, write the hash next to
# the output, and the next spec to want the design reuses it instead of
# rebuilding an identical xclbin.
#
# The key covers the design's source files, so a design that DOES generate
# per-spec sources (dense, layer_x) changes its key when the spec changes and
# is rebuilt -- which is correct, and is also why Task 30 isolates those
# sources per spec in the first place.

KEY_FILE = ".build_key"


def build_content_key(design: Path, out: Path, env: dict) -> str:
    """Hash of everything that determines the build's output.

    Covers the design's source files (excluding build outputs and caches) and
    the compile-time knobs. Two builds with the same key produce the same
    xclbin, so the second one can reuse the first one's output.
    """
    h = hashlib.sha256()
    h.update(str(out.relative_to(DESIGNS)).encode())
    h.update(b"\0")
    for f in sorted(design.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(design)
        top = rel.parts[0] if rel.parts else ""
        if top.startswith("build") or top in ("__pycache__", ".staging", ".locks"):
            continue
        if f.suffix in (".xclbin", ".bin", ".elf", ".o", ".prj"):
            continue
        h.update(rel.as_posix().encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    for k in sorted(env):
        h.update(f"{k}={env[k]}".encode())
        h.update(b"\0")
    return h.hexdigest()
```

### 2. Use the cache in `_build_one()`

In `_build_one()` (around line 205-250), the build is currently:

```python
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
```

Change it to check the key first, inside the lock (so the check and the skip are
atomic against another spec building the same design):

```python
    with design_lock(out):
        # The cache lives next to the output, so it is covered by the same lock
        # as the build: a spec that finds a current key reuses the xclbin
        # instead of rebuilding an identical one.
        key = build_content_key(src.parent, out, knobs)
        marker = out / KEY_FILE
        have = all((out / f).is_file() for f in FILES)
        if have and marker.is_file() and marker.read_text(encoding="utf-8").strip() == key:
            print(f"[{name}] up to date (build key {key[:19]}), reusing", flush=True)
            return name, out

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
        marker.write_text(key + "\n", encoding="utf-8")
```

**IMPORTANT:** the `return name, out` inside the `with design_lock(out):` block — the context manager releases the lock on exit, including on the early return. That is correct.

### 3. Make sure the key file is gitignored

The key file lives inside the build dirs (`designs/*/build*/.build_key`), and
`open_kernels/designs/*/build*/` is already ignored (root `.gitignore` line 48).
**Verify this** — if the ignore rule is `open_kernels/designs/*/build*/` then
`designs/attn_block/build_s256/.build_key` is covered. If not, add:

```
open_kernels/designs/**/.build_key
```

### 4. Keep `--force` working

`--force` must invalidate the cache. Find where `--force` is handled in
`export_qwen36_kernels.py` (the `a.force` flag) and make sure it also removes
the key marker (or bypasses the cache check entirely). The simplest correct
behavior: when `a.force` is set, delete `out / KEY_FILE` before the build, so
the build always runs.

Note: `export-kernels.py` has its own `--force` that passes through to the
inner script — check the plumbing still works after this change.

## Test

```bash
# Syntax
python3 -c "import ast; ast.parse(open('open_kernels/export_qwen36_kernels.py').read())" && echo "SYNTAX OK"

# The key changes when the knobs change, and not when they don't.
cd open_kernels && python3 - << 'EOF'
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("ek", "export_qwen36_kernels.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

d = m.DESIGNS / "attn_block"
out = d / "build_s256"
k1 = m.build_content_key(d, out, {"AG_M": "2048", "AG_K": "128"})
k2 = m.build_content_key(d, out, {"AG_M": "2048", "AG_K": "128"})
k3 = m.build_content_key(d, out, {"AG_M": "1024", "AG_K": "128"})
print("same knobs  ->", k1[:19], k2[:19])
assert k1 == k2, "FAILED: key not stable"
print("diff knobs  ->", k3[:19])
assert k1 != k3, "FAILED: key ignores the knobs"
print("OK: content key behaves")
EOF
cd ..

# The real test: run the export twice; the second run must skip the builds.
time python3 utilities/export-kernels.py -j 16 2>&1 | grep -c "up to date"
# → should be ~210 (the previously redundant builds)

# And the outputs must be unchanged
python3 utilities/export-kernels.py -j 16 2>&1 | grep -E "FAILED|error" | head
# → nothing

# --force must rebuild
python3 utilities/export-kernels.py -j 16 --force --specs qwen25-3b 2>&1 | grep -c "up to date"
# → 0 for that spec
```

## Success criteria

- [ ] `build_content_key()` added, hashing design sources + knobs
- [ ] `_build_one()` checks the key under the design lock and skips the build when current
- [ ] The key is written after a successful build
- [ ] `--force` invalidates the cache (rebuilds)
- [ ] The key file is gitignored
- [ ] Syntax check passes
- [ ] Key is stable for the same inputs, different for different knobs
- [ ] A second export run skips ~210 builds ("up to date" in the log)
- [ ] The skipped builds' xclbin hashes match what a rebuild produces
- [ ] Full build succeeds: `cmake --build --preset linux-default --parallel 16`
- [ ] All existing tests still pass
