---
name: kernel-size-and-fifo-diagnostics
description: Measure a design's code size and dataflow when a build fails for reasons the toolchain does not explain. Use when aiecc says only "Overflow of program memory" or "allocated buffers exceeded available memory", when a whole-layer design stops at a validated-set check or an acquire depth, or before changing any depth/fifo constant to see whether the acquire it satisfies is irreducible.
---

# Kernel size and fifo diagnostics

Two tools, both under `utilities/`, both aimed at the same problem: **IRON and
aiecc report failures that say nothing about cause.** A build dies with
`Overflow of program memory` and no shortfall; or
`Number of elements to acquire 3 must be smaller than depth 2` and no fifo.
Neither names what to change. These name it.

They exist because four separate walls in the Qwen3.8-27B work were all
invisible to the normal build output, and three of my first four theories about
them were wrong. In every case the fix was to measure rather than reason.

## 1. `utilities/kernel-size.py` — the per-TU code sizes aiecc deletes

aiecc removes its project directory on the way out, taking the per-core
objects that would say which core was over and by how much. `insts.bin`
survives but is only the cross-core *total*, which for a failing width is often
SMALLER than for one that builds — a small total on a design that overflows is
only explicable as one core carrying more than its share, and nothing in the
output says which.

```
./ironvenv/bin/python utilities/kernel-size.py --spec open_kernels/recipes/specs/qwen35-h2560-L32.json --kernel lx
```

Prints the biggest TUs by `.text`, a total, and the q4 GEMV entry points on
one core. `utilities/_keep_objects.py` is the helper it drives: it wraps
IRON's `compile_external_kernel` to copy each object aside as it is compiled,
*before* aiecc deletes the project. The build is the production one — aiecc is
NOT stubbed, so a failing width still fails.

**What it settled.** The q4 GEMV fold. `gemv_q4_pool_group_rt` is
`static inline`, so `gemv_q4_gy` and `gemv_q4_gms` each carry their own copy of
the band walk. Per-TU text at hidden 2560: `gy` 1360 + `gms` 1360 = 2720
un-folded, against `gyms` 1008 + `gy8` 736 folded. That 1712 B was on the core
that overflowed. The condition had been `bool(Q8)` — on mixed containers — when
the folded entry is correct with or without a q8 body, so every all-q4_1 dense
width was excluded. See `xcommon.FOLD`.

**The trap.** A per-TU table is printed even when the design does NOT build, so
read the header line (`BUILT` / `DID NOT BUILD`) before believing any of it. I
once reported a failing 5120 build as a success on the strength of the table
alone.

## 2. `utilities/why-no-fifo.py` — which fifo, and what its stream looks like

```
./ironvenv/bin/python utilities/why-no-fifo.py --spec <spec.json> --design layer_x/lx.py
./ironvenv/bin/python utilities/why-no-fifo.py --spec <spec.json> --design layer_x/lx.py --trace lni
```

Patches `ObjectFifoHandle.acquire/release` to include the fifo's name and the
design frames that called it, then specializes and compiles the design. Three
ways it has been wrong before it was right, all recorded in the file:

- `acquire` lives on the **handle**. `ObjectFifo` is the device-side object and
  has no `acquire`; patching it is a silent no-op.
- The design must be specialized **in-process**. The patch rebinds a method on
  an imported class, so a child process does not see it — the output is an
  unpatched error that looks like the patch failed.
- If you do edit site-packages to instrument, those files are **untracked** and
  `git checkout` will not revert them. Revert by hand.

**`--trace` is the expensive-to-rediscover part.** It prints every fill,
drain, acquire and release on one fifo in program order, which answers "is
this acquire irreducible?" — the question behind every depth that turns out
not to scale. On hidden 4096, `lni` shows:

```
ACQUIRE lni n=3 / release 3
ACQUIRE lni n=5 / release 5
ACQUIRE lni n=5 / release 5
```

against 8 fills per layer. Two things fall out:

1. **Acquires are emitted BEFORE the fills.** The shim runtime resolves the
   whole program symbolically and matches them afterwards, so an `acquire(5)`
   is satisfied by five fills *anywhere earlier in program order* — not by its
   own stage. Reordering to reduce a depth does not work.
2. **The depth is a cross-layer buffer.** `ln_body` consumes 13 elements per
   layer while the shim issues 8 fills worth 11; the 2-element difference is
   the next layer's `a0 a1` carried over the boundary. A depth that looks
   too large may be doing two jobs at once — here, the norm's live operands
   *and* the pipeline overlap.

## When a design will not build, in order

1. `why-no-fifo.py` with no `--trace` — names the fifo and the call site.
2. If it is a validated-set or L1 check, the recipe already names the set and
   the budget. Compute the budget; do not guess which side is wrong.
3. `kernel-size.py` — if aiecc reached the link stage, the per-TU sizes say
   whether it is a distribution spike or a design that is simply too big.
4. `--trace` on the named fifo before touching any depth constant.

## Rules

- **Read the artifact before believing a theory.** Every wrong conclusion in
  the 27B work came from reasoning about a formula without reading its
  consumer. `glue_ab_tile` (20 lines) settled in one step what three layers of
  inference had gotten wrong.
- A validated set records what HAS BEEN BUILT, not what can be. Adding a width
  and letting the build answer is the intended workflow; the sets are
  `catalogue.py`, and the comment on each says what a new value needs.
- A constant that is only ever *compared* and never used to size anything
  (`shim_fills`) is a recorded observation, not a limit. `Pipeline(3)` throttles
  the stream regardless.
- The `LIMITS` byte budgets were picked against the models that existed when
  they were written. The first model past one fails at allocation rather than
  at a check that explains itself, which is most of why these recipes carry
  explicit budgets and validated sets at all.
- `/tmp` is an 18 GB tmpfs and `/` runs near capacity. Write measurements to
  `build/`, and check `df` before packing a large model.
