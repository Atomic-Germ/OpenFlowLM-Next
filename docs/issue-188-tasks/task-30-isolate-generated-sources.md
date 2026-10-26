# Task 30: Isolate Per-Spec Generated Kernel Sources

**Phase:** Build correctness (fixes a cross-spec source race exposed by Task 28)
**Depends on:** Task 28
**Files to modify:** `open_kernels/export_qwen36_kernels.py`, possibly `open_kernels/designs/dense/gen_kernels.py`, `open_kernels/designs/layer_x/gen_kernels.py`
**Estimated complexity:** Medium (~80 lines)

## Context

**The bug (correctness, not performance):**

`export_qwen36_kernels.py` calls `gen.generate(F.recipe(spec))` (around line 331-338) which writes generated kernel TUs into the **shared** design directory — `open_kernels/designs/dense/` and `open_kernels/designs/layer_x/`.

The generated content **differs per spec** (measured):

```
gemv_q4_gms.cc: cd8f21be8029 (qwen3-4b)  vs  9a5b2529f9e5 (llama31-8b)
gemv_q4_gy.cc:  6a9ebb1b4969 (qwen3-4b)  vs  f543472fc507 (llama31-8b)
```

`dense` is used by **12 specs**, `layer_x` by **3**. Since Task 28 runs the 15 specs as parallel processes, the timeline is:

1. Spec A: `gen.generate()` writes `designs/dense/gemv_q4_gy.cc` (A's version)
2. Spec A: starts `build_design.py` for `designs/dense/build_qwen3_h2560`, compiles `designs/dense/*.cc`
3. Spec B (parallel): `gen.generate()` **overwrites** `designs/dense/gemv_q4_gy.cc` (B's version)
4. Spec A's aiecc picks up B's source → **A's xclbin has the wrong activation layout**

The build "succeeds" — the kernel is silently wrong. This is a pre-existing latent bug that parallel spec execution makes likely.

**Why the shared-design case is safe:** the 50 shared `build_dir`s all belong to `attn_block`, `gemm_q4_prefill`, `lm_head_q4`, `ln` — none of which read `OPEN_KERNELS_SPEC` and none of which have a `gen_kernels.py`. Only `dense` and `layer_x` generate per-spec sources, and their build dirs are per-spec (0 shared).

## What to do

### Goal

Each spec's build must compile **its own** generated sources, with no other process able to change them mid-build.

### Approach (pick one, verify with the test below)

**Approach A — stage the design dir per spec (recommended):**

1. In `main()` (around line 331-338), before calling `gen.generate()`:
   - Create a per-spec staging directory: `stage_root = DESIGNS / ".staging" / spec.spec_hash()[:19]`
   - Copy the design sources (the `.py`, `.h`, and any other non-build files) from the design dir into the staging dir
2. Call `gen.generate(F.recipe(spec), out=stage)` — `generate()` already takes an `out` parameter (see `designs/dense/gen_kernels.py:190`: `def generate(R, out: Path = HERE) -> int:`)
3. Pass the **staged** design path to `_build_one()` so `build_design.py` compiles from the staging dir

Note `_build_one()` currently derives `src` from `DESIGNS / sets[name]["design"]` (around line 163). It needs to use the staged path instead. Thread the staged path through — either as a new parameter or by computing it from the spec.

**Approach B — lock the design dir across gen+build:**

Hold a per-design-dir lock (reuse `design_lock()` from Task 29) around both `gen.generate()` and the build of every set belonging to that design. This is simpler but serializes the `dense` builds across all 12 specs — slower. Use only if Approach A turns out to be infeasible.

**Approach C — verify-after-build:**

Record a hash of the generated `.cc` files right after `gen.generate()`, and re-verify it after the build completes. If it changed, the build used foreign sources → rebuild (retry once, under a lock). Weaker (a mid-build change can still slip through) — use only as a last resort.

### Important detail

`gen.generate()`'s `out` parameter defaults to `HERE` (the design dir). Both `designs/dense/gen_kernels.py` and `designs/layer_x/gen_kernels.py` must honour the `out` argument. Check that they do — if either one writes to a hardcoded path, fix it to use `out`.

### Also check

Does `build_design.py` resolve any include or source path relative to the **repo root** rather than the design dir? Look at how the design lists the generated `.cc` files for compilation. If a path is hardcoded to the repo, staging the design dir will not be enough — in that case use Approach B.

## Test

```bash
# Syntax
python3 -c "import ast; ast.parse(open('open_kernels/export_qwen36_kernels.py').read())" && echo "SYNTAX OK"

# The race test: two specs generate DIFFERENT sources, both build, both must get their own.
cd open_kernels && python3 - << 'EOF'
import sys, hashlib, importlib.util
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, 'recipes')
from recipes.families import for_spec
from recipes.load import load_spec

def gen_into(spec_path, out_dir):
    """Generate the spec's kernel TUs into out_dir (not the shared design dir)."""
    spec = load_spec(Path(spec_path))
    F = for_spec(spec)
    gspec = importlib.util.spec_from_file_location("gen_kernels", Path(F.GEN_KERNELS))
    g = importlib.util.module_from_spec(gspec); gspec.loader.exec_module(g)
    out_dir.mkdir(parents=True, exist_ok=True)
    g.generate(F.recipe(spec), out=out_dir)
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in out_dir.glob("*.cc")}

import tempfile, shutil
tmp = Path(tempfile.mkdtemp())
a = gen_into('recipes/specs/qwen3-4b.json', tmp / "a")
b = gen_into('recipes/specs/llama31-8b.json', tmp / "b")
print("spec A sources:", a)
print("spec B sources:", b)
assert a != b, "expected the two specs to generate DIFFERENT sources"
assert (tmp / "a" / "gemv_q4_gy.cc").exists(), "A's sources not written to its own dir"
assert (tmp / "b" / "gemv_q4_gy.cc").exists(), "B's sources not written to its own dir"
# A's file must still hold A's content after B generated
assert hashlib.sha256((tmp/"a"/"gemv_q4_gy.cc").read_bytes()).hexdigest()[:12] == a["gemv_q4_gy.cc"]
print("OK: per-spec sources are isolated")
shutil.rmtree(tmp)
EOF
cd ..

# The real test: a full parallel export must produce the SAME xclbin hashes as a
# sequential one, for a spec that uses dense designs.
#   1. Sequential: python3 utilities/export-kernels.py -j 1 --specs qwen3-4b
#   2. Parallel:   python3 utilities/export-kernels.py -j 16 --specs qwen3-4b
#   3. Compare the sha256 of src/xclbins/<model>/open_kernels/dx/final.xclbin
```

## Success criteria

- [ ] `gen.generate()` writes to a per-spec location, not the shared design dir
- [ ] `build_design.py` compiles from the per-spec location
- [ ] `gen_kernels.py` (both dense and layer_x) honours the `out` argument
- [ ] No spec can change another spec's sources mid-build
- [ ] The staging dir is gitignored (`open_kernels/designs/.staging/`)
- [ ] Syntax check passes
- [ ] The isolation test passes (A's sources survive B's generation)
- [ ] Sequential and parallel exports produce **identical** xclbin hashes
- [ ] Full build succeeds: `cmake --build --preset linux-default --parallel 16`
- [ ] No "unable to rename temporary" errors
