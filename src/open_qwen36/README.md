# open_qwen36 — Qwen3.6-MoE, Qwen3 dense, Llama 3, Gemma 3 and HunYuan dense on open XDNA2 kernels

The open replacement for the closed `qwen3_6_moe_npu` engine. It sits behind
the app's `causal_lm` seam ([engine.hpp](engine.hpp)), so the tokenizer, chat
template, sampler, prompt cache and server — all already open — drive it the
same way they drive the closed DLL. Below the seam it is:

| file | what |
|---|---|
| [q4nx_file.cpp](q4nx_file.cpp) | reads FLM's `.q4nx` container (mmap; the 1.0.2 q4_1 format, refuses others) — replaces `q4_npu_eXpress.dll` on this path |
| [manifest.cpp](manifest.cpp) | reads the kernel set's `manifest.json`: layouts, contexts, kernels, per-layer programs, the packing plan, and the model check |
| [pools.cpp](pools.cpp) | interprets the manifest's packing plan: each layer's weights into the byte order the kernels stream, straight into resident device buffers (the same plan `open_kernels/recipes/pack.py` runs in NumPy) |
| [core.cpp](core.cpp) | device, contexts, kernels, 21 GB of resident pools, per-layer state; runs the manifest's program per layer, one `step()` per token |
| [engine.cpp](engine.cpp) | the `causal_lm` adapter: forward / prefill / checkpoint / restore / KV accessors |
| [cli.cpp](cli.cpp), [chat.py](chat.py) | drive the engine without the app (ids in, tokens out; `chat.py` tokenizes) |
| [manifest_test.cpp](manifest_test.cpp) | the OPEN-MANIFEST unit test (no XRT): parses the checked-in fixture, checks the model refusals |
| `../xclbins/<model>/open_kernels/` | `manifest.json` + the six kernels (`lx0 lx1 ax0 ax1 ln lm_head_q8`), **built, not checked in** — see below |

The kernels are `open_kernels/designs/layer_x` (+ `ln`, `lm_head_q8`): one xclbin
context per layer type, two dispatches per layer (everything up to the router;
then the MoE block once the host has re-pointed the expert fills at the
router's top-8), plus the final norm and the q8 lm_head. Per-token instruction
patching (`open_kernels/harness/stream_patch.hpp`) is what lets one compiled
program serve every layer and every position.

**No model constant lives in the C++.** The engine is an interpreter of
`manifest.json`, which `open_kernels/recipes/` writes from the model's
`ModelSpec` (hidden size, heads, experts, ... from `config.json`): the
consts / act / state / pool offsets, the KV row, which xclbin serves which
layer type, the verb sequence per layer (`run lx0` → `moeroute2 lx1` →
`run lx1`; `run ax0` → `moeroute2 ax1` → `run ax1`; then `ln`, `lm`), and the
tensor → pool-offset → chunk-order plan the packer follows. The same recipe
parametrizes the IRON designs, so the numbers in the instruction streams and
the numbers the driver uses have one source. A model whose `config.json`
disagrees with the manifest is refused at startup with the key named
(`specs/open-engine/spec.md`, OPEN-MANIFEST).

## Building the kernels

The compiled kernels are not in the repository (`.gitignore`), the same rule
as the BERT design sets: the source is `open_kernels/designs/` plus the recipe
in `open_kernels/recipes/`, and one command produces the six `final.xclbin` +
`insts.bin` pairs the engine loads and the `manifest.json` it reads, in the
directory it loads them from:

```
source ~/ironenv142/bin/activate            # mlir-aie 1.4.2 + Peano (ironvenv-requirements.txt)
export PATH=~/xrt-tools/bin:$PATH           # xclbinutil, aiebu-asm (from an XRT build)
python open_kernels/export_qwen36_kernels.py [--model-dir ~/.flm/models/Qwen3.6-35B-A3B-NPU2]
#   -> src/xclbins/Qwen3.6-35B-A3B-NPU2/open_kernels/{lx0,lx1,ax0,ax1,ln,lm_head_q8}/
#      + manifest.json + spec.json + toolchain.json
```

`--model-dir` derives the spec from that model's `config.json` (default: the
checked-in `recipes/specs/qwen36-35b-a3b.json`, the same model). The
manifest's `build_key` hashes the recipe, every kernel source the designs
include, the spec and the quant format; an export whose manifest already
carries the key is skipped (`--force` rebuilds). A shipped kernel directory
without a manifest needs one before the engine will take it:
`cd open_kernels && python -m recipes.manifest --model-dir <model> --out <kernel dir>/manifest.json`.

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
and `FLM_OPEN_KERNELS_DIR` are the engine's other two lookup locations; each
must hold a `manifest.json` naming files that exist).
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
harness: `y maxrel 5.4e-8`, `xn` bit-exact, same as the shipped one. The same
check passed again on 2026-09-05 after the designs were rewritten to take
every dimension from the recipe (all six streams byte-identical), and the
8-layer decode through the manifest interpreter scored the same
0.999998 / 0.999996 / 0.999991 logits correlation as before.

## Selecting it

The app uses the open engine for `qwen3.6-moe` whenever the kernels are
installed for the model (`xclbins/<model name>/open_kernels/`, or
`<model dir>/open_kernels/`, or `FLM_OPEN_KERNELS_DIR`); `FLM_QWEN36_ENGINE=closed`
forces the closed DLL, `=open` fails loudly if the kernels are missing. XRT
builds only (`FLM_USE_HRX=OFF`), like the open embedding NPU backend. The
same holds for the `qwen3` dense models (`FLM_QWEN3_ENGINE`), for `llama`
(`FLM_LLAMA_ENGINE`) and for Gemma 3 text models: the engine is the same code,
the manifest is a different recipe's.

## A second family: Qwen3 dense (2026-09-05)

The engine does not know which family it runs. `open_kernels/recipes/qwen3.py`
turns a Qwen3 dense `config.json` (GQA with q/k RMSNorm, full RoPE, no
attention gate, silu-gated FFN, a q4_1 lm_head) into a kernel set of three:
`dx` (the whole layer in one dispatch: `designs/dense/dx.py`, the same 8 main
cores + norm helper + attention helper as the MoE designs), `ln` at the
model's width, `lm_head_q4` (the q4 GEMV with an uneven band split). New
kernel points that came with it, all validated by the layer test: K = 2560 and
9728 GEMVs (activations that are not a whole number of 4 KB elements are
prepared by element index), HD 128 attention with 32/8 heads and full RoPE
(`ATTN_*` macros; the attention core's element is one KV-row half), the
2560-wide norm (`LN_N`), and the position record's sin placed right after its
cos (the fixed offset the 27B used only fit 32 values).

```
python open_kernels/export_qwen36_kernels.py --model-dir ~/.flm/models/Qwen3-4B-NPU2     # WSL
#   -> src/xclbins/Qwen3-4B-NPU2/open_kernels/{dx,ln,lm_head_q4}/ + manifest.json
python src\open_qwen36\chat.py "Explain what an NPU is in two sentences." --model %USERPROFILE%\.flm\models\Qwen3-4B-NPU2 --kernels src\xclbins\Qwen3-4B-NPU2\open_kernels
```

| check (Qwen3-4B, Strix, Windows + XRT) | result |
|---|---|
| 4-layer slice, 2 greedy tokens, harness vs the fp64 replica (`model/replica_dense.py`) | logits corr 0.999997 / 0.999994, same argmax and top-5, every layer's residual corr ≥ 0.999996 |
| the same through the engine (`open_qwen36_cli`, the C++ packer + manifest + attnpos) | identical numbers; request 2 reproduces request 1 |
| all 36 layers, the NPU prompt, greedy | *An NPU, or Neural Processing Unit, is a specialized piece of hardware designed to accelerate AI workloads, particularly those involving machine learning and neural networks. It is optimized for tasks like inference and training of deep learning models, offering improved efficiency and performance compared to general-purpose CPUs or GPUs.* then `<\|im_end\|>` at token 58 |
| speed | prefill 129 ms/token, decode 272 ms/token (3.7 tok/s): every token streams the model's 2.3 GB of q4 weights once, which is the floor of this dataflow on the NPU's DDR bandwidth |

`model/dense_probe.py` compares each stage's DDR bounce (xn, q/k/v, og, out,
res, xm, h, out2) of a dumped `act` buffer with the replica, fed the NPU's own
input, so a stage's error is localized rather than compounded.

## A third family: Llama 3 (2026-09-05)

Llama 3.1 8B runs on the same dense recipe (`recipes/dense.py`, family
`llama3`) with no new kernel: what differs is expressed as spec fields and
compile-time knobs -- no q/k norms (`ATTN_QKNORM=0`), eps 1e-5 (`LN_EPS`),
the llama3 RoPE frequency scaling computed host side
(`ModelSpec.rope_inv_freq`, carried in the manifest, used by both
position-table builders; it reproduces the container's own `rope_freqs.weight`
divisors to bf16 precision). Two sizes needed a different shape of the same
work: the 8 KB norm elements would not fit the norm helper's fused kernel
(five inputs and three outputs at once = its whole memory), so `ln_y` /
`ln_xn` emit one element per call; and the K = 14336 activation table (32 KB)
leaves room for only one 5 KB weight chunk per element (`per_call` in the
recipe, the GEMV entry generated with it).

```
python open_kernels/export_qwen36_kernels.py --model-dir ~/.flm/models/Llama-3.1-8B-NPU2     # WSL
python src\open_qwen36\chat.py "Explain what an NPU is in two sentences." --model %USERPROFILE%\.flm\models\Llama-3.1-8B-NPU2 --kernels src\xclbins\Llama-3.1-8B-NPU2\open_kernels
```

| check (Llama-3.1-8B, Strix, Windows + XRT) | result |
|---|---|
| 4-layer slice, 2 greedy tokens, harness vs the fp64 replica | logits corr 1.000000 / 0.999993, same argmax and top-5, every layer's residual corr ≥ 0.999994 |
| the same through the engine | identical numbers; request 2 reproduces request 1 |
| all 32 layers, the NPU prompt, greedy | *An NPU (Neural Processing Unit) is a specialized electronic component designed to accelerate artificial intelligence (AI) and machine learning (ML) workloads, similar to how a Graphics Processing Unit (GPU) accelerates graphics processing. NPUs are optimized to perform matrix operations and other computations that are common in deep learning and neural network processing, allowing for faster and more efficient AI and ML processing.* then `<\|eot_id\|>` at token 79 |
| speed | prefill 125 ms/token, decode 203 ms/token (4.9 tok/s) for 4.5 GB of q4 weights per token |

## A fourth family: Gemma 3 (2026-09-05) -- the "new op" proof

Gemma 3 4B (text) runs on the dense recipe with the plan's new ops expressed
as knobs, patches and one small kernel: GeGLU-tanh is the generated
activation TU (`dense_act.cc`: x * sigmoid(2z)); the sandwich norms are
`ln_nr32` (a norm without residual, emitting f32 halves) plus the layer
design's sandwich program (t = post_attn_norm(out); res = x + t;
xm = pre_ffn_norm(res); t2 = post_ffn_norm(out2); xres = res + t2); the
sliding window is a per-token `attnpos` patch (the KV fill's offset and
length, the record's row counts) on a second kernel entry (`dx_local`) that
shares the global layers' instruction stream but owns its instruction BO and
window; each layer type has its own position table (local theta 1e4, global
1e6 linearly scaled by 8). The container stores the norms' `1 + w` and the
sqrt(hidden)-scaled embeddings, so those two plan items are not transforms
here (checked against the HF mirror by range requests).

```
python open_kernels/export_qwen36_kernels.py --model-dir ~/.flm/models/Gemma3-4B-NPU2     # WSL
python src\open_qwen36\chat.py "Explain what an NPU is in two sentences." --model %USERPROFILE%\.flm\models\Gemma3-4B-NPU2 --kernels src\xclbins\Gemma3-4B-NPU2\open_kernels
```

| check (Gemma3-4B, Strix, Windows + XRT) | result |
|---|---|
| 6-layer slice (five local, one global), 2 greedy tokens, harness vs the fp64 replica | logits corr 0.999998 / 0.999998, same argmax and top-5, every layer's residual corr 1.000000 (maxrel ≤ 2.5e-4) |
| the same through the engine | identical numbers; request 2 reproduces request 1 |
| a step at position 1103 on the 6-layer slice (the window fill offset at row 80, 1023 rows) | finite logits, 94 ms |
| all 34 layers, the NPU prompt, greedy | *An NPU, or Neural Processing Unit, is a dedicated processor designed to accelerate AI workloads, particularly deep learning tasks. It's optimized for running neural networks much faster and more efficiently than traditional CPUs or GPUs.* then `<end_of_turn>` at token 43 |
| speed | prefill 63 ms/token, decode 96 ms/token (10.5 tok/s) |

Images still route through the closed engine — the open one refuses an image
payload with a clear error rather than silently ignoring it.

## A fifth family: HunYuan dense (2026-09-06)

Hy-MT2-7B (`hunyuan_v1_dense`, Tencent's translation model) is Llama 3.1 8B's
geometry exactly -- 4096 hidden, 32 layers, 32/8 heads at 128, 14336 FFN -- so
the dense recipe composes it from validated kernel points with no new GEMV K,
`ln` width or band split. Three things are new and none of them is a shape:

* the q/k RMSNorm weight multiplies **after** RoPE (`query_layernorm(rope(q))`),
  which is `attn.h`'s `ATTN_QKNORM_POST`. RoPE is orthogonal, so the RMS itself
  is unchanged by the rotation; what moves is the per-dim weight, which does not
  commute with the pair rotation. The order is a family property of the recipe
  (`dense.QKNORM_POST_ROPE`), not a `ModelSpec` field;
* the NTK-alpha RoPE scaling folds into one static base,
  `1e4 * 1000^(128/126)`, exactly as llama.cpp's converter does, so the position
  tables need nothing new;
* the vocabulary (128167) is not a whole number of 64-row head bands, so
  `dense.lm_rows` rounds the head up to 128192 while `hf_config_check` keeps the
  model's own count and `real_vocab` bounds the argmax.

FLM ships no NPU2 container for it: convert `tencent/Hy-MT2-7B-GGUF` with
`utilities/q4nx-build` (`ModelArch.HUNYUAN_DENSE`), which zero-pads the tied
head and applies no rotary permutation (llama.cpp's HunYuan converter adds
none, unlike its Llama one).

```
python utilities/q4nx-build/convert.py -i HY-MT2-7B-Q8_0.gguf -o %USERPROFILE%\.flm\models\Hy-MT2-7B-NPU2 -s tencent/Hy-MT2-7B
python open_kernels/export_qwen36_kernels.py --model-dir ~/.flm/models/Hy-MT2-7B-NPU2     # WSL
python src\open_qwen36\chat.py "Translate the following text into French. Note that you should only output the translated result without any additional explanation: The neural processing unit runs the model directly on the laptop, without sending anything to a server." --model %USERPROFILE%\.flm\models\Hy-MT2-7B-NPU2 --kernels src\xclbins\Hy-MT2-7B-NPU2\open_kernels
```

| check (Hy-MT2-7B, Strix, Windows + XRT) | result |
|---|---|
| 4-layer slice, 2 greedy tokens, harness vs the fp64 replica | logits corr 1.000000 / 0.999996, same argmax and top-5, every layer's residual corr >= 0.999996 (maxrel <= 3.9e-3) |
| the same through the engine | identical numbers; request 2 reproduces request 1; the head's padded rows 128167..128191 come back exactly zero |
| all 32 layers, a French translation instruction, greedy | *L'unite de traitement neuronal execute le modele directement sur l'ordinateur portable, sans envoyer quoi que ce soit a un serveur.* then `<\|eos\|>` at token 33 |
| speed | decode 231 ms/token (4.3 tok/s) -- the 8B's 203 ms for the same per-layer geometry |

Translation quality under q4_1 is a separate question from kernel correctness:
the correlations above prove the kernels, not the quantization. Score real
FLORES pairs before trusting the model for work.

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
