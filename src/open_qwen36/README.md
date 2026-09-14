# open_qwen36 — Qwen3.6-MoE, Qwen3.5 dense, Qwen3 dense, Llama 3, Gemma 3,
# HunYuan dense, IBM Granite and Phi-3 on open XDNA2 kernels

The open replacement for the closed `qwen3_6_moe_npu` engine. It sits behind
the app's `causal_lm` seam ([engine.hpp](engine.hpp)), so the tokenizer, chat
template, sampler, prompt cache and server — all already open — drive it the
same way they drive the closed DLL. Below the seam it is:

| file | what |
|---|---|
| [q4nx_file.cpp](q4nx_file.cpp) | reads OFLM's `.q4nx` container (mmap; q4_1 and q8 chunks, classified per tensor) — replaces `q4_npu_eXpress.dll` on this path |
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
python open_kernels/export_qwen36_kernels.py [--model-dir ~/.oflm/models/Qwen3.6-35B-A3B-NPU2]
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
and `OFLM_OPEN_KERNELS_DIR` are the engine's other two lookup locations; each
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
`<model dir>/open_kernels/`, or `OFLM_OPEN_KERNELS_DIR`); `OFLM_QWEN36_ENGINE=closed`
forces the closed DLL, `=open` fails loudly if the kernels are missing. XRT
builds only (`OFLM_USE_HRX=OFF`), like the open embedding NPU backend. The engine
is the same code for every family; only the manifest is a different recipe's.

One `AutoModel` adapter class per model family makes the choice, all through
`AutoModel::_shared_select_open_engine` (`src/common/AutoModel/automodel.cpp`),
so an architecture's variants behave identically:

| Env var | `model_list.json` families | Adapter classes |
| --- | --- | --- |
| `OFLM_QWEN36_ENGINE` | `qwen3.6-moe` | `Qwen3_6_MOE` |
| `OFLM_QWEN3_ENGINE` | `qwen3`, `qwen3-it`, `qwen3-tk`, `deepseek-r1-0528` | `Qwen3`, `Qwen3_IT`, `Qwen3_TK`, `DeepSeek_r1_0528_8b` |
| `OFLM_LLAMA_ENGINE` | `llama3.1`, `llama3.2`, `deepseek-r1`, `nanbeige4.1` | `Llama3`, `DeepSeek_r1_8b`, `Nanbeige` |
| `OFLM_GEMMA_ENGINE` | `gemma3`, `gemma3-text` | `Gemma3`, `Gemma3_Text_Only` |
| `OFLM_PHI4_ENGINE` | `phi4-mini-it` | `Phi4` |

Kernels are per model directory, so a family entry only means the adapter will
use whatever set is installed for that particular model. Images always go to the
closed engine -- the open one has no vision path.

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
python open_kernels/export_qwen36_kernels.py --model-dir ~/.oflm/models/Qwen3-4B-NPU2     # WSL
#   -> src/xclbins/Qwen3-4B-NPU2/open_kernels/{dx,ln,lm_head_q4}/ + manifest.json
python src\open_qwen36\chat.py "Explain what an NPU is in two sentences." --model %USERPROFILE%\.oflm\models\Qwen3-4B-NPU2 --kernels src\xclbins\Qwen3-4B-NPU2\open_kernels
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
python open_kernels/export_qwen36_kernels.py --model-dir ~/.oflm/models/Llama-3.1-8B-NPU2     # WSL
python src\open_qwen36\chat.py "Explain what an NPU is in two sentences." --model %USERPROFILE%\.oflm\models\Llama-3.1-8B-NPU2 --kernels src\xclbins\Llama-3.1-8B-NPU2\open_kernels
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
python open_kernels/export_qwen36_kernels.py --model-dir ~/.oflm/models/Gemma3-4B-NPU2     # WSL
python src\open_qwen36\chat.py "Explain what an NPU is in two sentences." --model %USERPROFILE%\.oflm\models\Gemma3-4B-NPU2 --kernels src\xclbins\Gemma3-4B-NPU2\open_kernels
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

OFLM ships no NPU2 container for it: convert `tencent/Hy-MT2-7B-GGUF` with
`utilities/q4nx-build` (`ModelArch.HUNYUAN_DENSE`), which zero-pads the tied
head and applies no rotary permutation (llama.cpp's HunYuan converter adds
none, unlike its Llama one).

```
python utilities/q4nx-build/convert.py -i HY-MT2-7B-Q8_0.gguf -o %USERPROFILE%\.oflm\models\Hy-MT2-7B-NPU2 -s tencent/Hy-MT2-7B
python open_kernels/export_qwen36_kernels.py --model-dir ~/.oflm/models/Hy-MT2-7B-NPU2     # WSL
python src\open_qwen36\chat.py "Translate the following text into French. Note that you should only output the translated result without any additional explanation: The neural processing unit runs the model directly on the laptop, without sending anything to a server." --model %USERPROFILE%\.oflm\models\Hy-MT2-7B-NPU2 --kernels src\xclbins\Hy-MT2-7B-NPU2\open_kernels
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

## A sixth family: IBM Granite (2026-09-06)

`granite-4.2-3b` is Llama geometry at **head_dim 64 / hidden 2560** -- the point
every shipped OpenFlowLM design refuses, because head_dim is intrinsic to RoPE
and cannot be padded. Nothing in the design changes for it: `ATTN_HD` / `ATTN_NH`
are compile-time macros and `attn.h` already carried hd 64's
`kScale = 0.125f`. So the family is `spec.py`, `families.py`, a spec JSON and a
test.

What is new is arithmetic that lives outside `ModelSpec`. Granite is Llama plus
four scalar multipliers -- `attention_multiplier` (which replaces the implicit
`hd**-0.5`), `embedding_multiplier`, `residual_multiplier` and `logits_scaling`
-- and the recipe expresses none of them. It does not have to, because **all
four fold exactly into the weights at conversion time**, and folding into an
already-quantized tensor is lossless: a Q4_1 block is `w = code*d + m`, so
scaling by `c` scales `d` and `m` and leaves every 4-bit code untouched. For the
3B the one non-unit factor is `attention_multiplier = 0.015625` at hd 64, giving
`q_proj *= 0.125` -- a power of two, so even the `d`/`m` scaling is an exponent
shift.

After that fold the model uses exactly the `1/sqrt(HD)` `attn.h` hard-codes,
which is the whole reason the dense recipe is legal for it. So the fold is not
optional and the recipe refuses without it, twice: `spec.py` rejects a
container whose `attention_multiplier` is not `head_dim ** -0.5`, and rejects
one that does not state it at all (transformers defaults the key to 1.0, so an
absent key is indistinguishable from an unfolded model, and 1.0 against 0.125 is
a silent factor of eight on every score). `dense.hf_config_check` repeats the
check at engine load, which is what catches a `model.q4nx` swapped under an
already-built kernel set.

OFLM ships no NPU2 container for it. Convert `ibm-granite/granite-4.2-3b-GGUF`
with `utilities/q4nx-build` (`ModelArch.GRANITE`), which applies the four folds
and writes the post-fold multipliers into the deployed `config.json`; install it
with `oflm-add ... --family granite`.

```
python utilities/q4nx-build/convert.py -i granite-4.2-3b-Q8_0.gguf -o %USERPROFILE%\.oflm\models\Granite-4.2-3B-NPU2 -s ibm-granite/granite-4.2-3b
python open_kernels/export_qwen36_kernels.py --model-dir ~/.oflm/models/Granite-4.2-3B-NPU2     # WSL
python src\open_qwen36\chat.py "Explain what an NPU is in two sentences." --model %USERPROFILE%\.oflm\models\Granite-4.2-3B-NPU2 --kernels src\xclbins\Granite-4.2-3B-NPU2\open_kernels
```

| check (Granite-4.2-3B, Strix, Windows + XRT) | result |
|---|---|
| 4-layer slice, 2 greedy tokens, harness vs the fp64 replica | logits corr 0.999998 / 0.999990, same argmax (38457) and top-5, every layer's residual corr 0.999990-0.999999 |
| the same through the engine | identical numbers; request 2 reproduces request 1 |
| all 40 layers, a Norwegian prompt, through the app | coherent, reasoning first, 40/40 layers resident in 11 s |
| speed | see the next section -- for Granite the honest answer needs a context position attached |

**Known rough edge:** the reasoning block is not parsed. Granite carries
`<think>` / `</think>` as real tokens (100274 / 100275) and its metadata sets
`think: true`, but `Granite` implements no `parse_stream_content`, so the chain
of thought prints raw and only the closing tag appears. Cosmetic, and separate
from the kernel path.

## The context term (2026-09-07)

`OPEN-ATTN-CONTEXT` recorded that a decode step's cost is dominated by a term
linear in context position -- 1.5 ms/position on Qwen3-4B, 2.0 on Granite -- so
that a single token at position 2048 cost 3-4 seconds. It also guessed the
mechanism: a fixed per-head cost in the position loop, `attn.h` "not saturating
the vector unit at hd 64".

The direction was right and the mechanism was not. It is **scalar float**, on
the scalar unit, inside a loop that runs `heads x positions` times per layer:
two `sexp()` per head per position for the online softmax, plus one
`* 1/sqrt(HD)` per head, plus a bf16 split and a compare in the accumulation.
The ablation that settles it, same build, same session:

| ablation | decode step @ 2048 |
|---|---:|
| baseline | 185.3 ms |
| drop q's low bf16 half -- **halves the score MACs** | 185.2 ms |
| drop the single `* kScale` beside it | **160.1 ms** |

Halving the arithmetic was free; deleting one scalar operation was 18% of the
context term. What followed from that reading: the exponentials batch over heads
through the vector unit (`ATTN_VEXP`), `1/sqrt(HD)` folds into q as an exponent
shift, the remaining scalars move into the vector phase, the heads split across
five cores (`ATTN_NHL`), the head loops unroll, and four cached rows are
processed per call (`ATTN_RB`) so one 32-lane exponential covers the block.

Measured through `oflm bench` on Granite 4.2 3B, `utilities/bench-configs/bench-1k.json`,
1005 prompt tokens, two builds differing only in these kernels:

| | TTFT | prefill | decode |
|---|---:|---:|---:|
| before | 1216.24 s | 0.826 tok/s | 0.415 tok/s |
| after | **58.93 s** | **17.05 tok/s** | **13.33 tok/s** |
| | 20.6x | 20.6x | 32.1x |

Twenty minutes to first token on a thousand-token prompt, down to under one.
The per-position slope goes 2.289 -> 0.0247 ms, 92x flatter.

**Prefill moves for the same reason decode does**, which is worth stating
because it is not obvious: prefill costs about what a decode step costs *at that
position*, so it is quadratic in the prompt length. On a short prompt that reads
as a flat per-token cost and hides completely.
`utilities/bench-configs/README.md` has the arithmetic and a two-measurement
recipe for predicting it on any model.

Only Granite takes this path today (`recipes/dense.py` sets `VEXP` for it
alone). Every other family compiles what it compiled before, byte for byte:
Qwen3-4B rebuilt on these kernels gives identical `insts.bin` for all three sets
and xclbins differing only in build stamps. The knobs are per-geometry, not
per-family, so another family joins by measuring the same way -- not by
declaring itself.
## A seventh family: Qwen3.5 dense (2026-09-06)

A Qwen3.5 dense layer is a Qwen3.6-MoE layer with the MoE block replaced by a
silu-gated dense FFN. Everything else -- the 3 linear : 1 full layer pattern, the
gated DeltaNet, the attention output gate, head dim 256 with a partial RoPE of 64,
16 linear key heads of 128, conv kernel 4, the q8 head -- is the 35B's, which we
already run. So there is no third recipe: `recipes/qwen35.py` calls
`qwen36moe.layout(..., ffn="dense")` for the whole attention half and adds the dense
recipe's FFN arithmetic on top. This matters because it is the biggest item on the
model census -- 19 published models and about 23.5 k downloads a month, including
the single most-downloaded model on Atomic-Germ's Hugging Face.

Four things are new, and only one of them is a shape:

* **One instruction stream per layer type instead of two.** The MoE needs a part
  split only because the router's output patches the second stream; nothing is
  routed here, so `lx` and `ax` each run the whole layer in one dispatch.
* **5 KB weight elements.** The 9B's down GEMV reads a 12288-wide activation table
  (27 648 B), which leaves no room in a core's 60 KB for two 10 KB weight elements
  beside the x stream and the DeltaNet scratch. `PER_CALL` drops to 1, as Llama 3.1
  8B's does, and the DeltaNet's S slices -- which ride the same weight fifo -- become
  10 rows over 13 slices instead of 20 over 7. That row count was hardwired in
  `dnx.h`; it is now the `DNX_ROWS` macro, default 20, so the shipped kernels
  preprocess identically. (`kPad`, the hi/lo record stride inside `ds`, is a fixed
  160 either way and is deliberately NOT wired to the recipe -- doing so rebuilt every
  `dnx_*` object in the shipped 27B kernels; see the handoff.)
* **8 KB norm elements.** At 4096 hidden the fused `ln_fn` wants the norm core's whole
  memory for its five inputs and three outputs, so the composition uses the split
  `ln_y` / `ln_xn` entries the dense design already introduced for Llama 3.1 8B.
* **q8 weights.** This container stores `ssm_out_proj` as q8 (the 35B stores it q4_1)
  and `ssm_{alpha,beta}_proj` with a bf16 `[heads, hidden]` copy (the 35B stores
  `[hidden, heads]`). Alpha and beta come through a `transpose` op; the out projection
  now has two paths, below. Both packers (NumPy and C++) are checked byte-identical.

```
OPEN_KERNELS_UNVALIDATED=1 python open_kernels/export_qwen36_kernels.py --model-dir ~/.oflm/models/Qwen3.8-Distilled-4B-NPU2   # WSL
python src\open_qwen36\chat.py "Explain what an NPU is in two sentences." --model %USERPROFILE%\.oflm\models\Qwen3.8-Distilled-4B-NPU2 --kernels src\xclbins\Qwen3.8-Distilled-4B-NPU2\open_kernels
```

| check (Qwen3.8-Distilled-4B, HID 2560, Strix, Windows + XRT) | result |
|---|---|
| 8-layer slice (six linear, two full), 3 greedy tokens, harness vs the fp64 replica | logits corr 0.999999 / 0.999992 / 0.999989, same argmax and top-5 at all three, every layer's residual corr >= 0.999991 (maxrel <= 3.1e-3) |
| the same through the engine | **bit-identical** -- max abs difference 0.000e+00 over 248 320 logits at every position; request 2 reproduces request 1; `route 0.00 / part1 0.0` in every step, i.e. one dispatch per layer |
| all 32 layers, the NPU prompt, greedy | *An NPU (Neural Processing Unit) is a specialized hardware accelerator designed to perform artificial intelligence and machine learning workloads with significantly higher efficiency than general-purpose processors...* then `[eos]` at token 85 |
| speed | decode 150 ms/token (6.66 tok/s) |

**All four sizes now run (2026-09-07).** Two fixes got the other three there. The glue core
kept a private bf16[HID] copy of the layer-entry norm output, which fits only up to HID 2816,
so the 9B was 2 560 B over; it now holds ONE 4 KB element and the alpha/beta projection is
re-streamed per half (`glue_ab_e.cc` takes the accumulator reset as an argument), which leaves
1 536 B free at any width. And the 2B / 0.8B, which have 16 linear value heads rather than 32,
hung on the first dispatch for two reasons, not one: `lx.py` ran the value conv tiles as many
times as the key tiles, so the core emitted 32 records where the host drained 16; and
`dn_glue.h` mapped each value head to key head `h / 2`, the right ratio only at 32 heads.
`kNHead` is now the `DNGLUE_NHEAD` knob, passed only when it differs from 32, so the shipped
27B's compile commands and object code do not move.

| size (Strix, Windows + XRT, q4_1 path) | 8-layer slice, 3 tokens | engine | all layers, the NPU prompt |
|---|---|---|---|
| 9B (4096 / 32 / 12288) | corr 0.999999 / 0.999987 / 0.999992, argmax + top-5 match, residual >= 0.999991 | bit-identical | `[eos]` @54, 181 ms/tok (5.53 tok/s) |
| 4B (2560 / 32 / 9216) | corr 0.999999 / 0.999992 / 0.999989 -- the 2026-09-06 numbers reproduced | bit-identical | `[eos]` @60, 138 ms/tok (7.25 tok/s) |
| 2B (2048 / 24 / 6144) | corr 0.999998 / 0.999979 / 0.999980, argmax + top-5 match, residual >= 0.999978 | bit-identical | `[eos]` @63, 69 ms/tok (14.5 tok/s) |
| 0.8B (1024 / 24 / 3584) | corr 0.999982 / 0.999993 / 0.999992, argmax + top-5 match, residual >= 0.999984 | bit-identical | `[eos]` @74, 53 ms/tok (18.7 tok/s) |

Those are the q4_1 numbers: the export ran with `OPEN_KERNELS_FORCE_Q4_1=1`, so the q8
`ssm_out_proj` was re-quantized on the way into the pool. Details and the standalone 16-head
glue compare: `.claude/plans/q35-hw-results.md`.

**Three of the four also run at native q8 (2026-09-07).** A Qwen3.5 container stores
`ssm_out_proj` at q8 while everything else is q4_1, so its main core has to carry both GEMV
bodies -- which did not fit in 16 KB of program memory. It fits now for every size but the
4B: on a spec whose roles mix formats, and only there, the two q4_1 entry points fold into
one whose destination is a runtime argument. Export without the force flag, into a
directory beside the q4_1 one:

```
python open_kernels/export_qwen36_kernels.py --model-dir ~/.oflm/models/Qwen3.8-Distilled-9B-NPU2 \
    --out src/xclbins/Qwen3.8-Distilled-9B-NPU2/open_kernels_q8                                  # WSL
```

| size, native q8 | 8-layer slice vs the replica fed the container's own q8 | engine | all layers | vs q4_1 |
|---|---|---|---|---|
| 9B | corr 0.999999 / 0.999991 / 0.999993, argmax + top-5 match, residual >= 0.999994 | bit-identical | `[eos]` @74, 191 ms/tok (5.25 tok/s) | 181 ms/tok |
| 4B | **no kernels** -- `lx` still overflows program memory | -- | -- | -- |
| 2B | corr 0.999998 / 0.999989 / 0.999986, argmax + top-5 match, residual >= 0.999986 | bit-identical | `[eos]` @61, 70 ms/tok (14.35 tok/s) | 69 ms/tok |
| 0.8B | corr 0.999993 / 0.999992 / 0.999990, argmax + top-5 match, residual >= 0.999991 | bit-identical | `[eos]` @47, 54 ms/tok (18.36 tok/s) | 53 ms/tok |

So native q8 costs about 5 % of wall time on the 9B and roughly nothing on the two small
ones, and it is what the model author shipped rather than a re-quantized copy of it. No
catalogue point was needed: the q8 out projection reduces over the DeltaNet value width
(4096 or 2048), both already validated by the 35B. The 4B is the one size whose hidden width
is not a multiple of the 4 KB activation element, so its glue walks two unequal halves and
its main core is already the largest of the four -- it stays on the re-quantizing fallback
until the FFN tail moves off that core. `.claude/plans/q8m-hw-results.md`.

## q8 weights: run them at q8, or re-quantize them

OFLM's newer converter stores the non-expert projections of the 35B-A3B fine-tunes --
attention q/k/v/o, the linear-attention qkv/z and out projections, the shared expert --
and Qwen3.5's `ssm_out_proj` at **q8** rather than q4_1. There are two ways to serve
that, and the engine has both.

**Run them at q8** (the default where the design can). A container q8 chunk is 8704
bytes and the main cores stream 5120-byte weight elements, so the host splits each chunk
into two 16-row half-tiles that each fill one element -- a byte permutation, no
arithmetic -- and `designs/gemv_q4/gemv_q8.h` consumes them with the same 64-row band and
the same y element the q4 GEMV uses. The pool holds the author's own values; nothing
downstream of the band changes. A q8 projection streams twice the bytes, so a Qwen3.6-MoE
linear layer's per-token weight traffic rises about 60 % (the routed experts, which dominate
it, stay q4_1) -- but measured on seven 35B containers that costs only about **9 % of decode
time** (155-170 ms/token against 140-162 re-quantized), so the projections are not what
decode waits on.

**Re-quantize them** (the fallback, and what every role the designs cannot stream at q8
still gets -- the routed experts and the MoE's shared expert). The packer dequantizes
each q8 chunk and writes the optimal q4_1 of it into the pool. A q8 chunk and a q4_1
chunk hold the same 32 x 256 tile, so the ordinary `std_perm` reads either and the plan,
the manifest and the kernels never learn there was a second format. This is not free:
on the 9B's first layer the q4_1 reading of `ssm_out_proj` tracks the q8 one at
correlation 0.9968 over random inputs, and across a 4-layer slice the logits correlate
0.999682 with the same argmax and the same top 5.

Which projections take which path is not a switch someone sets: `recipes/load.py` reads
the container's safetensors header and writes a per-role weight format into the
`ModelSpec`, so a model asks for what it actually holds. That map is part of the spec
hash, which means **a q8 variant of a shape is a different kernel set** (its instruction
streams bake in the doubled pool offsets and fill sizes) -- its build directories carry
the map's short hash, and a model with no q8 role keeps the directory, the hash and the
manifest it always had. `OPEN_KERNELS_FORCE_Q4_1=1` on the export, with
`make_decode.py --requant` on the reference, puts a whole model back on the fallback,
which is the A/B. `specs/open-engine/spec.md`, OPEN-QUANT-Q8.

**Measured, 2026-09-07** (`.claude/plans/q8-hw-results.md`). Ornith-1.0-35B-A3B at native
q8 -- an 8-layer / 3-token slice against the replica fed the container's own q8 values --
scores logits corr 0.999996 / 0.999998 / 0.999988 with the same argmax and top-5 and every
residual >= 0.999992; the engine reproduces the harness bit for bit (0.000e+00 over 248 320
logits). Six sibling containers, including Atomic-Germ's own `Qwen3.6-35B-A3B-NPU2` mirror,
derive the identical q8 spec and pass on the same kernels. Against the weights the author
shipped, native q8 holds 0.999996 where the re-quantizing fallback holds 0.997631 -- and the
fallback's greedy pick already differs by the second token.

**Qwen3.5 takes this path for every size but the 4B (2026-09-07).** Its containers hold ONE
q8 projection among q4_1 ones, so the main core has to carry both GEMV bodies; folding the
two q4_1 entry points into one -- on a mixed spec only, so no shipped kernel set moves --
made room on the 9B, 2B and 0.8B. The 4B is still over and stays on the fallback. Numbers in
the Qwen3.5 section above; log `.claude/plans/q8m-hw-results.md`.

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

- **Batched prefill -- open on Granite, not yet on the other families.**
  `OFLM_OPEN_GEMM_BLOCK=1` runs T prompt tokens per layer as 5 whole-array GEMM
  dispatches plus T attention dispatches instead of T decode steps (1.95x TTFT
  on a 1005-token prompt). It is read through `utils::getenv_oflm`, so the
  pre-rename `FLM_OPEN_GEMM_BLOCK` still works and prints a one-line notice
  naming the current variable. It needs a kernel set carrying a `gemm_block`
  program, which today is Granite only; every other family still goes through
  the decode step one token at a time. The route writes no M-RoPE position
  records, so a prompt that has had an image stays on the sequential path.
- **Long-context attention cost -- closed on the dense families and Qwen3.5,
  the 35B in progress** (2026-09-08, spec OPEN-ATTN-CONTEXT). The attention
  kernel now batches its softmax exponentials on the vector unit, splits the
  heads over up to six cores and blocks the cached rows per call; a step at
  position 2048 costs about what a step at position 0 does (Qwen3-4B: 5050 ->
  258 ms). A family joins by measurement (`recipes/attnknobs.py`); the 35B's
  kernels at the default knobs are byte-identical to what shipped.
- **Vision -- coded, not yet run end to end.** The vision tower runs on the host
  (`vision/vit.cpp`, checked against transformers with the shipped weights to
  4e-6) and its rows enter the model as embedding vectors at their M-RoPE
  positions (`Core::step_embed`). The app's Qwen3.6 and Qwen3.5 model classes no
  longer need the closed engine for images. `oflm-test --vision` on the open
  engine is the acceptance (OPEN-VISION-EMBED).
- **The weight file** is still OFLM's `.q4nx`. The GGUF path is a separate piece
  of work; this reader is ~150 lines and will go with it. The chunk format is read
  per tensor, so a container mixing q8 and q4_1 -- which is what the 35B fine-tunes
  ship, q8 attention and shared experts over q4_1 routed experts -- loads and packs;
  the q8 projections the kernel set was built for go into the pool at q8, the rest are
  re-quantized to q4_1 (above). **Both weight formats are read:** a container written
  by OFLM 1.0.3+ stores Q4_K super-blocks (4736-byte chunks) where 1.0.2 stored q4_1,
  and those are transcoded into the pool's q4_1 chunk on the way in -- nearly free,
  since Q4_K's scale and min already have the pool's granularity, and no kernel,
  manifest or build key changes (OPEN-QUANT-Q4K). Anything else (the 1280 / 2560-byte
  geometries) is refused, naming the tensor.
- **Memory.** The engine holds 21.6 GB of NPU buffers, and on Windows those
  are managed by the video memory manager and can be evicted under pressure —
  the first server run on a 47 GB box with 0.6 GB free hung a kernel (ERT
  state 8) and the context was dead from then on. Two mitigations are in:
  the container's mapped pages are dropped after packing (they were another
  ~13 GB of working set with no further use), and a failed kernel marks the
  engine so the next request rebuilds it (~90 s) instead of failing forever.
  Budget ~25 GB free for the full model.

## Through the app (2026-09-05)

`oflm.exe` built on this box with the vcpkg route (`src/build-windows-vcpkg.cmd`,
notes in `WinSetup.md`). `oflm serve qwen3.6-moe:35b-a3b` picks the open engine
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
