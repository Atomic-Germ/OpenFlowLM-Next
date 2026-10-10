# Plan: flow-viz, an animated picture of how a model runs on the NPU and CPU

Spec: `specs/viz/spec.md` (new, prefix `VIZ`). Home repo: openflowlm-next.
Status: Phases 0-4 implemented 2026-10-09 on `feat/flow-viz` (decode, 35B); Phases 5-6 open.

## What was built, and where it departs from the plan below

Done: VIZ-TOPOLOGY, VIZ-TIMELINE-DECODE, VIZ-EXPLAIN-COVERAGE and the new VIZ-PAGE-FILL as
tests (`python -m pytest specs/viz/tests`, plus `viz_render_test` in the CMake build), VIZ-COMMAND
and VIZ-PLAYER as procedures. Run unattended, so the Phase 3a/3b review gates were self-reviews
from headless screenshots (dark, light, 1600 px, 390 px) and a scripted click-through; they still
want a human look.

Decisions taken, each against the alternatives:
1. **Topology parse: text.** It reads all 98 local builds in 2 s (pre-placed, unplaced, mem-tile
   links), and its per-argument task bytes equal the compiled `insts.bin`'s exactly
   (`stream_patch::ddr_bytes`), which a printer change would break loudly. The bindings would have
   tied the test suite to WSL.
2. **Timing: a discrete-event simulation, not phases.** The phase model of the plan stalled
   output drains behind weight streams awaited later (97 forced tasks in `lx0`). The simulation
   (one shared DDR server, stream depths, cores blocking on acquires, awaits blocking the stream)
   runs every core's whole program with no forced task on both 35B sets.
3. **`attnpos` is applied now, not in Phase 5.** The compiled `ax0` stream carries one KV row; at
   rb 4 the attention cores want three, so without `attn_apply`'s rule they stalled and 19 tasks
   were forced. With it, the calls match the core programs exactly.
4. **Calibration:** 40.1 GB/s fitted to the one-context 70.3 ms; the context switch is 0.95 ms,
   which OPEN-DECODE-ONE-CONTEXT measured on the 27B, not the 35B. The same constants put the
   shipped two-context 35B set at 108 ms against 105-108 ms measured.
5. **The stale `lx.py` / `ax.py` docstrings are not fixed here.** `designs/layer_x/*.py` is in the
   qwen36moe build key (OPEN-BUILD-CACHE), so a docstring edit would move every shipped set's key
   and rebuild them all. They go with the next real change to those designs: `lx.py` still
   describes three streams with DeltaNet in its own context; `ax.py` one attention core.
6. **The page is embedded in `oflm`** (CMake `embed.cmake`, a byte array, no MSVC literal limit)
   and written to `<models dir>/viz/<name>.html`; release staging already carries `viz.json` and
   `topology.json` (it ships every ignored file under `src/xclbins`).

## Goal

`oflm viz <tag>` opens a self-contained page in the default browser, on Linux and
Windows, that animates one decode step of that model; one prefill block follows in
Phase 5. It shows which dispatch runs on the NPU, which host stage runs on the CPU,
how each kernel's data moves from DDR through the shim into the tiles, and what each
core runs. Every dispatch, core function and host stage has an explainer.

It is not realtime and carries no model data. The structure is real, taken from the
build: tiles, data paths, DMA order, byte counts, dispatch order. The durations are
modelled from bytes moved until a measured trace replaces them (Phase 6).

## Non-goals (v1)

- Closed-kernel models. They have no manifest and no MLIR, so they get a "not
  implemented" message that names the family (AGENTS.md).
- Diffusion and Whisper. `FLUX.2-klein-4B-NPU2/open_kernels` is
  `diffusion_kernels.json` plus `.elf`, a different format.
- GPU. oflm runs nothing there; the lane is drawn dimmed as "not used".
- A `/viz` route on `oflm serve`. Every handler sends JSON (`server.hpp:96-102`,
  `server.cpp:813`), so the server has no path for an HTML body.

## Where the data comes from

| What the page shows | Source | Real or modelled |
|---|---|---|
| Layers and their types | `manifest.json` `layers` | real |
| Dispatch order per layer, route patches, tail | manifest `layer_types[*].program`, `tail` | real |
| Tiles; which functions each core calls, with loop trip counts; data paths with element size and depth; mem-tile links | `final.prj/aie.mlir` + `input_with_addresses.mlir` of each build | real |
| DMA tasks per dispatch: buffer, offset, length, await points | `aie.runtime_sequence` in `aie.mlir` | real |
| The weight tensor a pool/consts task reads | task offset ∩ the pack plan's op regions (`recipes/pack.py`) | real |
| Routed experts in a decode step | a fixed illustrative set per layer | illustrative, labelled |
| Host stages | a table in the timeline builder, each entry naming its `core.cpp` function | hand-kept (drift risk) |
| Durations | bytes / one bandwidth constant; host steps are fixed constants | modelled (Phase 6: measured) |
| Model header: name, family, size, quant | `model_list.json` `details` | real |
| Explainers | `src/viz/explain/*.md`, drafted from design docstrings, `open_kernels/README.md` "Designs" and the specs | authored |

## Design

### Data pipeline

1. **Topology, at export (Python, WSL).** `export_qwen36_kernels.py` copies each
   kernel's files in its per-kernel loop (`:304-317`), where both the build dir and
   the destination are in hand. That loop calls `open_kernels/viz/topology.py` on
   `<build dir>/final.prj` and writes `<dst>/topology.json`.
   - It also works with `--no-build`, because `bdirs[n]` still points at the build
     dir (`:295`).
   - Nix (`open-kernels.nix` copies `src/xclbins`) and both Windows installers
     (which glob `xclbins\**`) pick the file up with no change.
   - Nix builds in `$TMPDIR` and drops `final.prj`, which is why topology has to be
     written at export and cannot be derived later.
2. **Timeline, at export.** After the loop, `open_kernels/viz/timeline.py` reads
   `manifest.json` and every `topology.json` and writes `viz.json` beside the
   manifest. `viz.json` holds:
   - the topologies, deduped (lx0 and lx1 share one core program and differ only in
     their runtime sequence);
   - the baked timelines: `decode`, one token at a stated position; later `prefill`,
     one 256-token block at a stated L.
3. **`oflm viz <tag>` (C++)**, in `src/src/viz_command.hpp`, modelled on
   `image_command.hpp`.
   - **Tag lookup is strict.** `get_model_info` (`model_list.hpp:58`) silently falls
     back to llama3.2:1b on a miss; viz refuses the tag instead.
   - **Finding the set.** It looks for `<xclbin root>/<model_list name>/open_kernels/viz.json`,
     searching the same roots as `Engine::find_kernels` (`engine.cpp:31-70`). The
     weights don't have to be downloaded.
   - **Writing the page.** It inlines `viz.json`, the model_list entry and the
     explainers into the page, writes `<models dir>/viz/<name>.html`, and opens it
     (`ShellExecuteW` on Windows; `xdg-open` via posix spawn on Linux).
   - **`--no-open`** only prints the path, for headless and WSL use.
   - **`oflm viz` with no tag** lists the tags whose set has a `viz.json`.
   - **Embedding.** The page and explainers are compiled into the binary:
     `file(READ … HEX)` generates a byte array, because MSVC rejects string literals
     past about 64 KB. That means no installer edits and no runtime lookup;
     `find_model_info`'s Windows branch checks only the exe dir.
   - **Escaping.** `</` is escaped in the inlined JSON.

### topology.json (one per kernel dir)

```json
{
  "device": "npu2", "columns": 8,
  "tiles": [{"col": 0, "row": 2, "kind": "core"}],
  "cores": [{"col": 0, "row": 2, "calls": [{"fn": "gemv_q4_gup", "count": 8, "fifos": ["w0", "x"]}]}],
  "fifos": [{"name": "w0", "from": [0, 0], "to": [[0, 2]], "depth": 2, "elem_bytes": 10240}],
  "links": [{"from": ["A_L3L2_0"], "to": ["A_L2L1_0"], "via": [0, 1]}],
  "args": ["pool", "xres", "consts", "state", "act", "ptab"],
  "tasks": [{"fifo": "w0", "arg": 0, "offset": 505282560, "len": 2621440, "await": true}]
}
```

**How the MLIR is parsed.** The parser reads the MLIR as text instead of using the
mlir-aie bindings. Windows Python has no `aie` module, and the pytest suite runs
there.

- **Tile placement.** It takes one of two forms:
  - Pre-placed builds (layer_x, moe_batch) carry their coordinates on the logical
    tile: `aie.logical_tile<CoreTile>(0, 2)`.
  - Unplaced builds (attn_block, gemm_q4_prefill, lm_head_q8) print `(?, ?)`, and
    their placement has to be resolved through `input_with_addresses.mlir`:
    - cores, by `loc(#locN)`, which points back to their `aie.mlir` line;
    - buffers, by `sym_name`;
    - fifo endpoints, by `<fifo>_buff_N` / `<fifo>_cons_buff_N` and
      `shim_dma_allocation`.

    The `#locN` aliases can sit at the top or the bottom of the file. Logical shims
    map many-to-one onto the 8 physical shims.
- **Fifo declarations** may carry `dimensionsToStream [...]` and multi-dimensional
  memrefs. Element bytes = product of the dims × the dtype size.
- **Call counts** multiply through nested `scf.for` loops, using `arith.constant`
  bounds. A runtime bound read from an RTP buffer is resolved from the sequence's
  `aiex.npu.rtp_write`.
- **Tasks** come from `aiex.dma_configure_task_for @fifo { aie.dma_bd(%argN … offset
  len …) }` in textual start order. `aiex.dma_await_task` marks the sync points.
  There is no `dma_memcpy_nd` in these builds.
- **Refusal.** Anything left unmapped (a core with no tile, a fifo endpoint not
  found, zero tasks) is refused by name. A partial file is never written.

### Timeline (in viz.json)

Each event looks like this:
`{lane: npu|cpu, kind: dispatch|host|patch, name, layer, start, dur, bytes: {arg: n}, tasks: [i, j], explain: key}`.

**Decode on qwen36moe.**
1. Host: embedding row into `xres`, then sync.
2. Each layer:
   - `run lx0`;
   - host `moeroute2`: read the top-k, patch lx1;
   - `run lx1`.

   Full-attention layers use ax0 and ax1, with the `attnpos` patch.
3. Tail: `ln`, then `lm`.
4. Host: logits readback, then the sampler.

The 35B has 82 NPU dispatches. The submit-ahead of OPEN-DECODE-PIPELINE is drawn as
the next dispatch queued behind the current one.

**Durations.**
- A dispatch takes its DDR bytes / `bw_gbps`. That is one constant, calibrated so the
  modelled 35B step equals a measured `--bench-decode` step. It is stored in
  `viz.json` with its date and source.
- Host steps use the gaps measured in OPEN-DECODE-PIPELINE (0.02–0.07 ms).
- The page labels every duration "modelled".

**Inside a dispatch.**
- Tasks are spread over the duration in sequence order, weighted by their length.
- A core lights up while a fifo it consumes is carrying a task. It shows the
  function or functions that sit between that fifo's acquire and release.

### The page (`src/viz/viz.html`)

It is one file with inline CSS and JS: no libraries, no network, no build step. When
nothing is inlined, it accepts a dropped `viz.json`, so it can be developed without
building oflm.

- **Drawing is a pure function of the clock.** `frame(t)` derives everything from the
  sorted event list. Pause, stop (t = 0), scrub, speed and rewind then need no extra
  state.
- **Rendering split.** Canvas 2D, scaled for `devicePixelRatio`, draws the array, the
  packets and the lanes. The DOM holds the panels and controls.

**Views**, all on one playhead:
1. **Layer stack** (left). All layers, coloured by type, with the current one
   highlighted. Arrows show `xres` passing down and each layer's weights streaming in
   from the DDR pool.
2. **NPU array** (centre).
   - **Grid.** 8 columns × (shim row, mem row, 4 compute rows), with the DDR buffers
     (pool, xres, consts, state, act, ptab) drawn as a band under the shims.
   - **Packets** travel shim → (mem) → core along the real fifos, coloured by kind:
     weights, activations, state, output.
   - **Cores** show the function they are running. Hovering a packet shows the
     weight tensor it carries.
3. **Device lanes** (bottom). NPU, CPU and GPU (dimmed) across the step, with the
   playhead and scrub bar.
4. **Explainer** (right). Clicking anything shows its title, what it does, why it's
   there, its requirement IDs and its source path.

**Controls:**
- play/pause, stop, and step to the next event;
- speed from 0.1× to 10× (log scale);
- a scrub bar;
- a decode/prefill toggle (prefill is greyed out until Phase 5);
- keyboard: space, ←/→, Home.

**Look.**
- Dark theme first, with a light theme via `prefers-color-scheme`.
- One accent colour per kind of data.
- Eased motion, and no decoration that isn't data.
- A static mock built from the real 35B data is reviewed before any animation work
  (the Phase 3a gate).

### Explainers (`src/viz/explain/{dispatch,core,host}.md`)

**Format.**
- One `## <key>` section per entry. Globs cover tiered names: `gemm_n*_k*`, `mb_s*`,
  `ag_s*`, `ag_pv*`.
- Each section has a one-line summary, a short body, a `Spec:` line with requirement
  IDs and a `Source:` line with a path.
- The page renders a small Markdown subset: paragraphs, bold, code and lists.

**Drafting.**
- Drafted from the design docstrings, the README's "Designs" section and the specs.
- Then checked against `topology.json`, because some docstrings are stale.
  `lx.py:14-18` describes three instruction streams and DeltaNet in its own context.
  The built set has two (lx0, lx1), and lx0's cores call `dnx_*`. That docstring gets
  fixed in the same change.

**Missing entries.** A missing key renders as an explicit "no explainer for X" card,
and the export prints the keys that are missing.

**v1 scope (35B decode):** about 6 dispatch kernels, the 49 distinct core functions
in the built MLIR, and about 8 host stages.

## Requirements (new)

### VIZ-TOPOLOGY: the export writes each kernel's tiles, data paths and DMA tasks
**Applies to:** openflowlm-next
**Verification:** test

For every kernel dir it writes, the kernel-set export writes `topology.json`, extracted
from that build's `final.prj`. It lists:
- the tiles used;
- each core's called functions, with loop trip counts;
- the fifos (endpoints, depth, element bytes) and mem-tile links;
- the runtime DMA tasks, in order.

A build it cannot fully map is refused, naming the unmapped object.

**Acceptance criteria:**
- lx0 fixture: 11 cores at (0..7, 2) and (0..2, 3); 8 shim columns; no mem tiles;
  fifo `w0` from (0, 0) to (0, 2), 10240 B, depth 2; 263 tasks.
- lx0 and lx1 fixtures: identical tiles, cores and fifos; 263 and 330 tasks.
- An unplaced fixture (the smallest of attn_block `pv256` or `lm_head_q8`): every
  logical core resolves to a placed tile; mem tiles are in row 1; links are recorded.
- lx0's task bytes per arg equal the per-arg `ddr_bytes` that `--bench-decode`
  reports for lx0 (recorded once into the fixture).
- The lx0 fixture with one core's `loc` removed is refused, naming that core.

### VIZ-TIMELINE-DECODE: one decode step as ordered NPU and host events
**Applies to:** openflowlm-next
**Verification:** test

From a set's manifest and topologies, the timeline builder writes the decode step into
`viz.json`:
- every dispatch in manifest order, with its DDR bytes per arg and its task range;
- the host stages between dispatches;
- a modelled duration for each event.

**Acceptance criteria:**
- 35B fixture: 82 NPU dispatch events in manifest program order (2 per layer × 40,
  then `ln`, `lm`).
- Each layer has exactly one `moeroute2` host event, between its two dispatches.
- Every pool/consts task names the pack op whose region contains it; none is
  unlabelled.
- Event durations sum to the step total; `bw_gbps` is present with its source.

### VIZ-TIMELINE-PREFILL: one prefill block as ordered NPU and host events
Phase 5; the acceptance criteria get written then.

### VIZ-EXPLAIN-COVERAGE: everything the page can show has an explainer
**Applies to:** openflowlm-next
**Verification:** test

Every dispatch kernel, core function and host-stage key in a shipped `viz.json`
resolves to an explainer section, and every requirement ID on a `Spec:` line exists in
some `specs/*/spec.md`.

**Acceptance criteria:**
- The 35B `viz.json` fixture: zero unresolved keys.
- A fixture with one extra core function: that key is reported unresolved.
- An explainer citing `OPEN-NO-SUCH-ID`: reported.

### VIZ-COMMAND: `oflm viz` opens the page for one model
**Applies to:** openflowlm-next
**Verification:** manual

**Verification (manual):** on Windows and on Linux, with the network off:
1. `oflm viz qwen3.6-moe:35b-a3b` opens the page in the default browser.
2. `oflm viz nosuch:1b` is refused, naming the tag; it does not fall back to
   llama3.2:1b.
3. A closed-kernel tag gives "not implemented", naming the family.
4. `oflm viz` lists the tags that have a viz.
5. `--no-open` only prints the path.

### VIZ-PLAYER: controls, views and explainers
**Applies to:** openflowlm-next
**Verification:** manual

**Verification (manual):** open the 35B page.
1. Play, pause, stop, step, both speed extremes and the scrub bar all keep the three
   views on one playhead.
2. Clicking a layer, a dispatch, a core, a packet and a host stage each opens its
   explainer.
3. Every duration on screen says "modelled".
4. The devtools Network tab stays empty.

**Spec impact elsewhere:** none on `OPEN-*` through Phase 5; the set gains a file the
engine ignores. Phase 6 adds an `OPEN` requirement for the trace.

## Phases

| # | Work | Gate | Estimate |
|---|---|---|---|
| 0 | `specs/viz/spec.md` skeleton from this plan | your review of this plan | — |
| 1 | `topology.py`, export hook, fixtures, tests; re-export `k35q41` with `--no-build` | VIZ-TOPOLOGY green; `topology.json` in every kernel dir | 2–3 d |
| 2 | `timeline.py` (decode), `viz.json`, tests | VIZ-TIMELINE-DECODE green | 1–2 d |
| 3a | Static page: the three views with real 35B data, no motion | your review of screenshots | 2 d |
| 3b | Animation, controls, explainer panel; explainers for 35B decode | VIZ-PLAYER procedure; VIZ-EXPLAIN-COVERAGE green; your review of the explainers | 4–5 d |
| 4 | `oflm viz`, embedding, browser launch | VIZ-COMMAND on Windows and Linux | 1–2 d |
| 5 | Prefill: `gemm_block` route, `attn_block`, `moe_batch`, host-stage table, mem-tile links in the array view | prefill-pipeline stages 1–2 landed; VIZ-TIMELINE-PREFILL green | 3–4 d |
| 6 | Optional measured timing: `TraceRec` keeps `t0`/`t1`, host stages record spans, `OFLM_OPEN_STEP_TRACE=3` writes a trace the page can load in place of the modelled durations | new `OPEN` requirement | 2 d |

Through Phase 4 (35B decode, shipped): about 2–2.5 weeks. With prefill: about 3.

**Other families** get a viz when their set is exported, since topology and timeline
come from the build. New core functions need explainers, and VIZ-EXPLAIN-COVERAGE
flags them. Today only the 35B set is exported in-tree.

## Decisions (recommendation first)

1. **Topology parse:** text (recommended) vs the mlir-aie Python bindings. The
   bindings would survive printer changes, but they exist only in WSL and are untested
   here.
2. **Delivery:** a standalone HTML file that `oflm viz` writes and opens
   (recommended) vs a `/viz` route on `oflm serve`, which needs a non-JSON body path
   in the server first.
3. **Timing:** modelled from bytes in v1, with measured traces as an option later
   (recommended) vs measured only. Measured-only needs Phase 6 first, and an NPU to
   produce every page.
4. **Prefill timing in the schedule:** after prefill-pipeline stages 1–2 land
   (recommended). That route is uncommitted and still changing.

## Risks and open items

- **A toolchain bump changes the MLIR printer.** The extractor then refuses at export,
  naming what it couldn't map; the parser and the fixtures are fixed together.
  `toolchain.json` records the version.
- **The host-stage table drifts from `core.cpp`.** Each entry names its function, and
  Phase 6's trace is the check.
- **Modelled durations get read as measurements.** The label appears in the lane
  header and in the explainer, and the bandwidth constant is shown with its source.
- **ax0's KV tasks are patched by position (`attnpos`).** v1 shows them as built and
  says so. Applying the patch rule in Python is a Phase 5 follow-up.
- **Release staging.** Confirm that `utilities/release/stage-prebuilts.sh` carries
  `topology.json` and `viz.json` and not only the files it requires
  (`final.xclbin`, `insts.bin`).
- **"Beautiful" is subjective.** The Phase 3a review comes before any time is spent on
  animation.
