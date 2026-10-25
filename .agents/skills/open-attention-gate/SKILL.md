---
name: open-attention-gate
description: Add a new model SIZE to a family that already has kernels -- generate the spec from the model's config.json, compile the family's kernel set, run the attention kernel at the new geometry against an fp64 oracle on a bare NPU, and record the validated point in the catalogue. Use when the catalogue refuses a geometry ("is outside the validated combinations"), when a model has no kernels yet, or when adding a size of a known family.
---

# The attention gate: adding a SIZE to a family

A model size is a geometry. The catalogue (open_kernels/recipes/catalogue.py)
lists the geometries each kernel has actually been RUN at, and refuses the rest
at generation time -- on purpose, because a kernel that was never run is a
kernel whose numbers nobody knows.

The old way to run one: convert the model, install it, decode it, compare
against the fp64 replica. That needs the weights, a container, the engine and
a person, which is why adding a size was hard.

This gate needs none of that. It builds the PRODUCTION attention worker at the
new geometry, runs it against synthetic inputs on an NPU, and scores it
against an fp64 oracle:

```
python utilities/build-new-spec.py --repo Qwen/Qwen3-0.6B
```

That one command:

1. **spec** -- `utilities/gen-kernel-spec.py` reads the model's `config.json`
   from HuggingFace and writes `open_kernels/recipes/specs/<family>-<size>.json`.
   A fine-tune whose README frontmatter names a base model that already has a
   spec LINKS to it: a fine-tune never gets its own kernels.
2. **build** -- `utilities/export-kernels.py --specs <name>` compiles the
   family's kernel set with `OPEN_KERNELS_UNVALIDATED=1`, which is how the
   catalogue's "not validated yet" is overridden for the build under test.
   This proves the design COMPILES at the new geometry.
3. **gate** -- `open_kernels/designs/attn/probe_dense.py` builds the attention
   kernel at that geometry, `make_test.py` writes synthetic inputs and the
   harness program, `open_kernels/harness/run_kernel` runs it and
   `compare.py` scores it against `attn_reference.py` (fp64). The exit status
   is the verdict.
4. **catalogue** -- `utilities/catalogue-add-point.py` writes the validated
   point into the catalogue, with a comment naming where it came from. No
   hand-editing of `catalogue.py`.
5. **rebuild** -- the kernel set is compiled again through the validated path,
   so what ships is a build the catalogue allows.

## The files

| file | what it is |
|---|---|
| `open_kernels/designs/attn/probe_dense.py` | the standalone design: the production worker, synthetic in, results out |
| `open_kernels/designs/attn/probe_support.py` | lifts `_attn` out of `designs/dense/dx.py` (AST; the production source, verbatim) |
| `open_kernels/designs/attn/make_test.py` | the fixture: inputs, the harness program, the reference dumps |
| `open_kernels/designs/attn/compare.py` | the scoring: PASS / FAIL per case |
| `open_kernels/designs/attn/attn_reference.py` | the fp64 oracle (the q/k norm, the rotation, the softmax) |
| `utilities/build-new-spec.py` | the chain above |
| `utilities/catalogue-add-point.py` | the catalogue edit, mechanical and checked |

## The rules

- **The gate measures the shipped kernel.** `probe_support.py` extracts the
  worker body from `designs/dense/dx.py`; there is no second implementation of
  the attention anywhere in the gate. A copy would pass while the shipped
  kernel broke.
- **The compile flags come from the recipe.** `recipes/dense.py:attn_flags()`
  is the ONE list the layer design and the gate both build attn.h with. A
  second copy would drift, and a drifted gate validates a kernel nobody ships.
  (`dx.py` calls it; do not re-inline the list.)
- **The geometry is the spec's, not the gate's.** `probe_dense.py` reads the
  spec and asks `recipes/dense.py:geometry()` for NH / KVH / HD / ROT / the
  core split / RB / VEXP -- the same numbers the export builds with.
- **A point is validated when its fixture passes, and only then.** The gate
  writes the catalogue entry itself, so the catalogue cannot drift from what
  was run.

## What the gate covers, and what it does not

Covers: the attention kernel's own arithmetic at that geometry -- the q/k
norm, the rotation, the GQA mapping, the online softmax, the new cache row,
the core split over heads, the block path (RB 2 / 4).

Does NOT cover: the projections' weights, the packing, the host's window
bookkeeping (which rows are valid is the driver's `attnpos` patch, not the
kernel), the prefill dispatch, or the whole-layer integration. The whole-layer
check against the fp64 replica (`open_kernels/model/`) still needs the model;
that remains the gold standard, and the gate is the cheap half of it.

Not covered by the gate: a family with a q/k/v bias (Qwen2) -- the bias fifo
is not wired into the probe -- and `ATTN_BLOCK_ONLY` (the MoE families). Both
say so and refuse rather than guess.

## The measured example

Qwen3-0.6B (16 heads / 8 kv at head dim 64, a GQA group of 2, rotary 64,
q/k normed, four cores of four heads, RB 4):

- the point the catalogue refused:
  `(64, 16, 8, 64, True, False, False, False)` -- "is outside the validated
  combinations".
- the gate: `probe_dense.py` builds it, four cases (random, the GQA mapping,
  a saturated softmax, zero values) PASS against the fp64 oracle on an NPU.
- the catalogue entry, written by the script:
  `(64, 16, 8, 64, True, False, False, False),  # Qwen3-0.6B: head dim 64, a GQA group of 2`.

The same gate, run against a geometry that was ALREADY in the catalogue
(Qwen3-4B), passes too -- that is the control: it shows the harness agrees
with the kernels that have been shipping.

Qwen3-14B (40 heads / 8 kv at head dim 128, five cores of eight heads, and the
FF and hidden widths that force the split above) is the same story: the gate
validated its attention tuple on the NPU, the split down GEMV and the streaming
norm helper made the design build, and the q4 lm head at K=5120 is the one point
still waiting for a fixture of its own.

## The wide shapes: the split down GEMV and the streaming norm

Two widths do not fit the dense layer's kernels as they were first built:

* an **FF whose activation table does not fit a main core's L1** (Qwen3-14B: FF
  17408 is 39 168 B of table against a 60 KB budget). The down GEMV reduces over
  FF, so it runs in two K pieces (8192 + 9216), each with its own table, and the
  norm helper adds the two partial outputs (`ln_add3`).
* a **hidden whose norm helper cannot hold its five input elements** (HID 5120:
  five 10 KB elements plus the output and the stack do not fit 64 KB). The helper
  streams instead: the residual add goes half by half (`ln_add2`), the sum goes to
  DDR, the norm reads it back.

Both are the MoE / Qwen3.5 tail's mechanisms (`recipes/qwen36moe.py down_split`
and `norm_split`), SHARED with it, not copied: the 27B has been running them since
2026-09-30, and the dense recipe now runs the same ones -- the split bodies live
in `open_kernels/ln_split.py` and the strided piece tap in `ironutil.piece_tap`,
because two copies of that arithmetic would drift.

`DENSE_DOWN_SPLIT=4096+5632` and `DENSE_NORM_SPLIT=1` force the split on a width
that fits whole, so it can be compared against the whole path on a model that is
already tested end to end. They are probe variables: they change the compiled
kernel, so the build key sees them and a probe build cannot ship as a real one.

## Other catalogue ops

The gate covers the `attn` op. The others refuse their own new points:

| op | what validates a new point |
|---|---|
| `gemv_q4` | `open_kernels/designs/gemv_q4/make_test.py` at that K |
| `ln` | `open_kernels/designs/ln/make_test.py` at that width |
| `lm_head_q4` | the q4 head has no fixture of its own: `designs/lm_head_q8/make_test.py` is the model to port (the q4 head is the same kernel with another band split). This is the one point still open for Qwen3-14B |
| `lm_head_q8` | `designs/lm_head_q8/make_test.py` |
| `deltanet`, `short_conv`, `moe`, `router` | family properties, not sizes |

`build-new-spec.py` prints which op refuses the spec and names the fixture
for it, so the remaining work is a fixture run, not an archaeology.
