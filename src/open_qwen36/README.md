# open_qwen36 — Qwen3.6-MoE on open XDNA2 kernels

The open replacement for the closed `qwen3_6_moe_npu` engine. It sits behind
the app's `causal_lm` seam ([engine.hpp](engine.hpp)), so the tokenizer, chat
template, sampler, prompt cache and server — all already open — drive it the
same way they drive the closed DLL. Below the seam it is:

| file | what |
|---|---|
| [q4nx_file.cpp](q4nx_file.cpp) | reads FLM's `.q4nx` container (mmap; the 1.0.2 q4_1 format, refuses others) — replaces `q4_npu_eXpress.dll` on this path |
| [pools.cpp](pools.cpp) | packs each layer's weights into the byte order the kernels stream, straight into resident device buffers; byte-identical to `open_kernels/model/pools.py` |
| [core.cpp](core.cpp) | device, contexts, kernels, 21 GB of resident pools, per-layer state; one `step()` per token |
| [engine.cpp](engine.cpp) | the `causal_lm` adapter: forward / prefill / checkpoint / restore / KV accessors |
| [cli.cpp](cli.cpp), [chat.py](chat.py) | drive the engine without the app (ids in, tokens out; `chat.py` tokenizes) |
| `../xclbins/<model>/open_kernels/` | the six kernels (`lx0 lx1 ax0 ax1 ln lm_head_q8`), **built, not checked in** — see below |

The kernels are `open_kernels/designs/layer_x` (+ `ln`, `lm_head_q8`): one xclbin
context per layer type, two dispatches per layer (everything up to the router;
then the MoE block once the host has re-pointed the expert fills at the
router's top-8), plus the final norm and the q8 lm_head. Per-token instruction
patching (`open_kernels/harness/stream_patch.hpp`) is what lets one compiled
program serve every layer and every position.

## Building the kernels

The compiled kernels are not in the repository (`.gitignore`), the same rule
as the BERT design sets: the source is `open_kernels/designs/`, and one
command produces the six `final.xclbin` + `insts.bin` pairs the engine loads,
in the directory it loads them from:

```
source ~/ironenv142/bin/activate            # mlir-aie 1.4.2 + Peano (ironvenv-requirements.txt)
export PATH=~/xrt-tools/bin:$PATH           # xclbinutil, aiebu-asm (from an XRT build)
python open_kernels/export_qwen36_kernels.py
#   -> src/xclbins/Qwen3.6-35B-A3B-NPU2/open_kernels/{lx0,lx1,ax0,ax1,ln,lm_head_q8}/ + toolchain.json
```

| set | design | knobs | what it is |
|---|---|---|---|
| `lx0` | `layer_x/lx.py` | `LX_PART=0` | linear-attention layer, dispatch 0 (norm → qkv/z → glue → DeltaNet → post → out → norm → router) |
| `lx1` | `layer_x/lx.py` | `LX_PART=1` | its MoE block (same xclbin as `lx0`, second instruction stream) |
| `ax0` | `layer_x/ax.py` | `AX_PART=0` | full-attention layer, dispatch 0 |
| `ax1` | `layer_x/ax.py` | `AX_PART=1` | its MoE block |
| `ln` | `ln/ln.py` | — | final RMSNorm |
| `lm_head_q8` | `lm_head_q8/lm_head_q8.py` | `LMHEAD_N=248320 LMHEAD_CORES=8` | q8 lm_head, full vocab |

About 6 minutes for all six on a Ryzen AI 9 HX 370 (WSL; ~90 s per layer_x set). `--only lx0,lx1`
rebuilds a subset, `--out DIR` redirects (a model directory's `open_kernels/`
and `FLM_OPEN_KERNELS_DIR` are the engine's other two lookup locations).
`toolchain.json` records the mlir-aie and Peano versions, this tree's commit,
and every file's sha256. The distributed package ships the built kernels; a
source checkout builds them. In WSL the kernels are built and on Windows they
run: the export writes into the shared checkout, so nothing needs copying.

**Is the source really the source?** `--check DIR` compares a fresh build with
a previous one. Checked here (2026-09-05) against the binaries this PR
originally shipped, built 2026-09-04 on the same toolchain:

| | result |
|---|---|
| 6 × `insts.bin` (the instruction streams) | **byte-identical** |
| 6 × `final.xclbin` | identical apart from **78–82 bytes each**: the axlf header's unique id, timestamp and UUID, the PDI's UUID in `AIE_PARTITION`, the boot-image header's unique id and the checksum that covers it, and xclbinutil's `XCLBIN_MIRROR_DATA` JSON tail that repeats the header |

Every CDO and every AIE core ELF matched; `--check` masks exactly those stamp
fields (parsing the axlf and partition structs, not by offset guesswork) and
fails on any other byte. A rebuilt `ln` was also run on the NPU through the
harness: `y maxrel 5.4e-8`, `xn` bit-exact, same as the shipped one.

## Selecting it

The app uses the open engine for `qwen3.6-moe` whenever the kernels are
installed for the model (`xclbins/<model name>/open_kernels/`, or
`<model dir>/open_kernels/`, or `FLM_OPEN_KERNELS_DIR`); `FLM_QWEN36_ENGINE=closed`
forces the closed DLL, `=open` fails loudly if the kernels are missing. XRT
builds only (`FLM_USE_HRX=OFF`), like the open embedding NPU backend.

Images still route through the closed engine — the open one refuses an image
payload with a clear error rather than silently ignoring it.

## Standalone

```
src\open_qwen36\build.cmd                                  # MSVC + system XRT -> out\open_qwen36_cli.exe
python src\open_qwen36\chat.py "Explain what an NPU is in two sentences."
out\open_qwen36_cli.exe --model <dir> --kernels <dir> --ids 248045 --max-tokens 3 --layers 8 --dump-logits out\y
```

`cmake -S src/open_qwen36 -B build` builds the same CLI on Linux.

## Results (2026-09-05, Strix, Windows + XRT, stock Qwen3.6-35B-A3B)

Scored against the fp64 reference `open_kernels/model/replica.py` computes from
the same container (the oracle the kernels were accepted against):

| check | result |
|---|---|
| 8 layers, 3 greedy tokens from `[248045]` | logits corr 0.999998 / 0.999996 / 0.999991, same argmax and top-5 at every position — identical to the batch harness |
| two requests on one resident engine | identical token sequences |
| attention window of 6000 rows (past the old 4096 cap) | runs, finite logits |
| **all 40 layers, chat prompt (23 ids), greedy** | *"An NPU is a specialized hardware component designed to accelerate the processing of neural network workloads. It enables AI applications to run more efficiently by offloading complex computations to dedicated silicon."* then `<\|im_end\|>` |

Full model: weights resident in **84 s** (21 GB packed from the mmap'd
container), prefill **124 ms/token** (decode-as-prefill, lm_head skipped),
decode **~140 ms/token ≈ 7 tok/s** (part 0 ≈ 95 ms, MoE ≈ 26 ms, lm_head
12 ms, routing 0.2 ms). The closed engine on the same box, quiet, does ~6.8
tok/s (phlegm's measurement); yesterday's 323 ms/token was the batch harness
on a memory-starved box, not the kernels.

## What is still not closed

- **Batched prefill.** Prompt tokens go through the decode step one at a time.
  Exact, but 124 ms each: a 500-token prompt is a minute. The closed engine
  has batch kernels; nobody has written open ones (phlegm's plan called it
  "weeks, gated on one experiment").
- **Long-context attention cost.** The attention kernel is one core walking the
  cache: ~24 µs per cached row per attention layer, so a 4096-token context
  adds ~1 s per token across the 10 attention layers. Capacity is no longer
  capped (KV buffers are sized from the app's context length); speed at long
  context is a kernel item.
- **Vision.** The model is a VLM; images still need the closed engine.
- **The weight file** is still FLM's `.q4nx`. The GGUF path is a separate piece
  of work; this reader is ~150 lines and will go with it.
- **Memory.** The engine holds 21.6 GB of NPU buffers, and on Windows those
  are managed by the video memory manager and can be evicted under pressure —
  the first server run on a 47 GB box with 0.6 GB free hung a kernel (ERT
  state 8) and the context was dead from then on. Two mitigations are in:
  the container's mapped pages are dropped after packing (they were another
  ~13 GB of working set with no further use), and a failed kernel marks the
  engine so the next request rebuilds it (~90 s) instead of failing forever.
  Budget ~25 GB free for the full model.

## Through the app (2026-09-05)

`flm.exe` built on this box with the vcpkg route (`src/build-windows-vcpkg.cmd`,
notes in `WinSetup.md`). `flm serve qwen3.6-moe:35b-a3b` picks the open engine
(the log says so: *"Qwen3.6-MoE on the open kernels (...)"*) and answers
OpenAI-style chat completions:

| prompt | answer | TTFT | decode |
|---|---|---|---|
| Explain what an NPU is in two sentences. | *An NPU is a specialized hardware component designed to accelerate the processing of neural network workloads. It enables AI applications to run more efficiently by offloading complex computations from general-purpose CPUs.* | 2.2 s / 19 tok | 6.9 tok/s |
| Write a haiku about silicon. | *Silicon is gold, / Chips hum in the server room, / Data flows like light.* | 1.7 s / 15 tok | 7.0 tok/s |
| What is 2+2? Answer briefly. | *2+2=4* | 2.1 s / 18 tok | 5.8 tok/s |
| What is the capital of France? | *The capital of France is **Paris**.* | 1.7 s / 15 tok | 6.3 tok/s |

The same prompt twice in one server session gives the same answer (state reset
through the app's checkpoint/restore path). Prompt cache, sampler, chat
template, tool parsing — all the app's, unchanged.
