# viz: flow-viz, an animated picture of one decode step on the NPU and the host

Prefix `VIZ`. Home repo: openflowlm-next. Covers `open_kernels/viz/` (what the kernel-set export
extracts from each build and models from it), `src/viz/` (the page, its fonts and explainers) and
`src/src/viz_command.hpp` (`oflm viz`). Plan: `specs/viz/plans/flow-viz.md`.

Tests: `python -m pytest specs/viz/tests` (no device, no toolchain: the fixtures are gzipped MLIR,
one compiled instruction stream and extracted topologies of the two Qwen3.6-35B kernel sets, made by
`specs/viz/tests/make_fixtures.py`). The command and the page are documented procedures.

What is real and what is modelled is the one thing every requirement below keeps straight: tiles,
data paths, the DMA task order, byte counts, which core runs which function in which dispatch, and
the dispatch order all come from the compiled set. Durations come from a model -- DDR bytes over
one effective bandwidth -- and the page says so wherever it shows one.

## Requirements

### VIZ-TOPOLOGY: the export writes each kernel's tiles, data paths, core programs and DMA tasks
**Applies to:** openflowlm-next (`open_kernels/viz/topology.py`, `export.py`; the hook in `export_qwen36_kernels.py`)
**Verification:** test

For every kernel dir of a set it writes, the export also writes `topology.json`, read from that
build's `final.prj/aie.mlir` and `input_with_addresses.mlir`: the tiles used, each core's program
(calls, loops with their bounds or the runtime word that sets them, objectfifo acquires and
releases), every objectfifo's endpoints, depth and element size, mem-tile links, and the runtime
sequence's DMA tasks (stream, buffer argument, offset, length) with its starts, awaits and runtime
parameter writes. Logical tiles are mapped to placed ones through the placed file's `loc`
references, so an unplaced design maps as well as a pre-placed one. A build it cannot fully map --
a core with no placed tile, a stream endpoint it cannot place, a task on an undeclared stream -- is
refused with the object named; no partial file is written, and `viz.json` is removed rather than
left stale. A set whose `insts.bin` is not byte-identical to the build's is refused. The refusal
only costs the set its viz: the export prints it and ships the kernels.

**Acceptance criteria:**
- `lx0` (two-context 35B): 11 cores at (0..7, 2) and (0..2, 3); shims in all 8 columns; no mem
  tiles; `w0` from (0, 0) to (0, 2), 10240 B, depth 2; `x` to 8 cores; 263 tasks.
- `lx0` and `lx1`: identical tiles, cores, streams and links; 263 and 330 tasks.
- `lx0`'s task bytes per buffer argument equal the compiled `insts.bin`'s, summed the way
  `stream_patch::ddr_bytes` sums them (31,457,280 B from the pool).
- `lm_head_q8` (every logical tile unplaced): its 8 cores land on (0..1, 2..5); `ln` (placed core,
  unplaced shims) maps too.
- `gemm_q4_prefill` n2048 k512: 32 cores, mem tiles only in row 1, 20 links, each link's inputs
  ending and its outputs starting at its mem tile.
- A placed core with its `loc` removed is refused, naming the core; a task on `@nosuch` is refused.

### VIZ-TIMELINE-DECODE: one decode step as ordered NPU and host events
**Applies to:** openflowlm-next (`open_kernels/viz/timeline.py`)
**Verification:** test

From the manifest and the topologies, the export models one decode step at position 1 into
`viz.json`: the host stages (embed, the `attnpos` patch, each layer's `moeroute2` route, logits
readback, sampling), every dispatch in manifest order, and a context switch wherever consecutive
dispatches name different contexts. Each layer type's dispatches are run as a discrete-event
simulation of the image: shim streams deliver elements through one shared DDR server at the
effective bandwidth, limited by each stream's depth; cores run their programs, blocking on their
acquires; awaits block the instruction stream; cores start each layer fresh. A kernel built with
the `attnpos` patch has its KV window, new-row drain and record fill rewritten for the position
exactly as `stream_patch::attn_apply` does. Every DDR task is labelled with the pack op whose
region holds it (routed experts with an illustrative choice per layer). The bandwidth, 40.1 GB/s,
is fitted once so the one-context 35B set models the 70.3 ms OPEN-DECODE-ONE-CONTEXT measured; a
context switch costs that requirement's 0.95 ms; both are recorded in the file with their source.

**Acceptance criteria:**
- Two-context 35B: 82 dispatches, in manifest order; exactly one `route` host event between each
  layer's two dispatches; 22 context switches; events tile the step with no gap or overlap.
- Every pool, consts and lm-head task names its tensor (none falls back to "layer weights");
  `lx1` routes 8 expert slots and streams `experts down` and the shared expert.
- Every core runs its whole program: per layer type, the calls summed over the template equal the
  calls of one pass of the core programs, with no task forced through.
- One-context 35B: 3 context switches; `lx0` and `ax0` share one image; DeltaNet calls run only in
  linear-attention layers and attention calls only in full-attention ones.
- `attnpos` at rb 4: a 3-row KV window at position 1, 11 rows at position 9, the new row drained
  at row 9.
- The calibration (bandwidth and its source) and the position are in the file.

### VIZ-EXPLAIN-COVERAGE: everything the page can show has an explainer
**Applies to:** openflowlm-next (`src/viz/explain/*.md`, `open_kernels/viz/explain.py`)
**Verification:** test

The page explains what it shows from `src/viz/explain/*.md`: one `## kind:key` section per
dispatch kernel, core function, host stage, stream (globs allowed), DDR buffer, layer type and the
page's own concepts, each with a title, a one-line summary, optional `Spec:` requirement IDs and a
`Source:` path. Every key a `viz.json` can make the page ask for resolves, and every requirement ID
an explainer cites exists in some `specs/*/spec.md`. The export prints the keys still missing; the
page shows "no explainer yet" naming the key, never a blank panel.

**Acceptance criteria:**
- Both 35B fixtures: no unresolved key.
- A topology with one extra core function reports exactly `core:<that function>`.
- No explainer cites an ID that is not a `###` heading of a spec; `OPEN-NO-SUCH-ID` is reported.
- `fifo:w*` resolves `fifo:w3` and not `fifo:x`.

### VIZ-PAGE-FILL: the page's slots are filled once and no payload can close its script tag
**Applies to:** openflowlm-next (`src/viz/viz.html`, `viz_command::render`, `open_kernels/viz/page.py`)
**Verification:** test

The page is one HTML template with four slots -- the fonts, `viz.json`, the explainers and the
model entry -- filled in a single pass in that order, so text inside a payload that looks like a
slot is left alone. Every `</` in a payload is written `<\/`, which JSON reads as `</` and the page
undoes for the explainers, so no payload can end its `<script>` early.

**Acceptance criteria:**
- A payload carrying `</script><script>alert(1)</script>` comes back intact from its script block
  and the page carries no `</script><script>alert`.
- A payload carrying `/*VIZ_EXPLAIN*/` keeps it, and the explainer slot still gets the explainers.
- No slot is left unfilled; the fonts are inlined.

### VIZ-COMMAND: `oflm viz` writes and opens the page for one model
**Applies to:** openflowlm-next (`src/src/viz_command.hpp`, `src/viz/embed.cmake`)
**Verification:** manual

`oflm viz <tag>` looks the tag up strictly (no fallback to another model), finds the model's kernel
set the way the engine does (`--kernels`, `OFLM_OPEN_KERNELS_DIR`, beside the model, then every
xclbins root) without needing its weights, fills the page compiled into the binary, writes it to
`<models dir>/viz/<model name>.html` (or `-o`), and opens it in the default browser (`--no-open`
prints the path only). `oflm viz` alone lists the tags whose set carries a `viz.json`. A model on
the precompiled kernels gets "not implemented" naming its family; a set exported before flow-viz
gets the re-export instruction.

**Verification (manual):** on Windows and on Linux, network off:
1. `oflm viz qwen3.6-moe:35b-a3b` prints the page and kernel-set paths and opens the page.
2. `oflm viz nosuch:1b` is refused naming the tag; it does not fall back to llama3.2:1b.
3. A closed-kernel tag (e.g. `llama3.2:1b`) reports that flow-viz is not implemented for its family.
4. `oflm viz` lists the 35B; `--no-open -o x.html` writes `x.html` and starts no browser.

### VIZ-PLAYER: the page plays the step and explains what it shows
**Applies to:** openflowlm-next (`src/viz/viz.html`)
**Verification:** manual

One playhead drives the layer list, the array (tiles, streams with packets, DDR buffers, host CPU
and iGPU chips), the step's NPU / CPU / iGPU lanes and the current dispatch's stream and core
analyzer. Play, pause, stop, step to the previous or next event, a speed from 2 µs to 50 ms of
modelled time per second, scrubbing and `#t=<µs>` links move it. The explainer panel follows
playback until something is clicked. Durations are labelled modelled.

**Verification (manual):** open the 35B page.
1. Space plays and pauses; ← / → step events; Home rewinds; the speed slider spans its range; dragging
   the step lane scrubs; every view stays on one playhead.
2. Click a layer, a dispatch, a core, a stream, a DDR buffer, a host block and the iGPU chip: each
   opens its explainer with the live facts for that moment.
3. "Modelled timing" opens the calibration and its source.
4. Light and dark themes both read; at phone width the page stacks with no horizontal scroll.
5. The devtools Network tab stays empty.
