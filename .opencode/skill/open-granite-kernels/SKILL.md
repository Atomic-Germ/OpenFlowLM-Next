---
name: open-granite-kernels
description: Build, verify and ship the open XDNA2 kernel sets (dx ln lm_head_q4) that run IBM Granite 4.2 3B on the dense recipe. Use when rebuilding those xclbins, adding another Granite size, debugging "no open kernels found" for granite:3b, or when a Granite container's attention_multiplier is refused at load.
---

# Granite 4.2 3B on the dense recipe

Granite is the **fifth family** on `open_kernels/recipes/dense.py`, and the
first at **head_dim 64** — the point every shipped FastFlowLM design refused,
because head_dim is intrinsic to RoPE and cannot be padded. Nothing in the
design changed for it: `ATTN_HD` / `ATTN_NH` are compile-time macros and
`designs/attn/attn.h` already carried HD 64's `kScale = 0.125f`.

Kernels are **built, not checked in** (`.gitignore`: `src/xclbins/BERT-h*/` and
the model's `open_kernels/`). Reference docs: `specs/open-engine/spec.md`
(OPEN-FAMILY-GRANITE), `src/open_qwen36/README.md`, `open_kernels/PROVENANCE.md`.

## The fold — read this before anything else

Granite is Llama plus four scalars: `attention_multiplier` (replacing the
implicit `hd**-0.5`), `embedding_multiplier`, `residual_multiplier`,
`logits_scaling`. `ModelSpec` expresses none of them and `attn.h` hard-codes
`1/sqrt(HD)`.

They do not need expressing, because **q4nx-build folds all four into the
weights** at conversion: `q_proj *= attention_multiplier * sqrt(hd)`,
`o_proj`/`down_proj *= residual_multiplier`, `embed_tokens *=
embedding_multiplier`, `lm_head /= logits_scaling`. For 4.2-3B the only
non-unit factor is `attention_multiplier = 0.015625` at hd 64, so the fold is
`q_proj *= 0.125` and the folded `config.json` reads `attention_multiplier =
0.125 = 64**-0.5` exactly. A power of two, so it is exact in bf16. The
container keeps the originals under `q4nx_folded_multipliers`.

**An unfolded container is refused twice**, by name, and both refusals matter
because an unfolded model does not crash — it returns plausible garbage:

- `recipes/spec.py::_granite_scale_check`, at generation, from HF `config.json`
  and GGUF metadata alike.
- `hf_config_check` carries `attention_multiplier`, so the **engine** refuses at
  load. Swapping a different `model.q4nx` under a built kernel set is otherwise
  silent.

## Build — on Windows, not WSL

This is the opposite of the Qwen3.6 skill's advice, and deliberately so.
`aiecc` emits `insts.elf` through `aiebu-asm`, and `build_design.py` asks for it
unconditionally. **Windows has `aiebu-asm.exe` and `xclbinutil.exe` from
`C:\Xilinx\XRT`**, both on PATH after `iron_env.ps1`. WSL has neither: Ubuntu
24.04's own XRT is `202210.2.13.466` (2022, pre-NPU) and building Xilinx/aiebu
needs Boost + liblzma + liblz4, i.e. root. Nothing in the build path is
OS-specific — an xclbin is a device artifact with no host code in it.

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1          # MUST be dot-sourced
cd <repo>
python open_kernels\export_qwen36_kernels.py --model-dir C:\Users\<you>\.flm\models\Granite-4.2-3B-NPU2
```

Three sets, from build dirs `dense/build_granite_h2560`, `ln/build_2560_1e-05`,
`lm_head_q4/build_100352`, into
`src/xclbins/Granite-4.2-3B-NPU2/open_kernels/{dx,ln,lm_head_q4}` plus
`manifest.json`, `spec.json`, `toolchain.json`.

`OPEN_KERNELS_UNVALIDATED=1` is **no longer needed** — head_dim 64, num_heads 40
and gemv_q4 K 8192 entered `catalogue.py` with the 2026-09-06 run below. It is
needed again only for a *new* unvalidated point.

## The standalone test binaries

Neither ships built. Both need the same three variables:

```powershell
$env:XRT_INCLUDE_DIR = "C:\Xilinx\XRT\include"
$env:XRT_LIB_DIR     = "C:\Xilinx\XRT\lib"
$env:VCVARS64        = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cmd /c open_kernels\harness\build.cmd      # -> open_kernels\harness\out\run_kernel.exe
cmd /c src\open_qwen36\build.cmd           # -> src\open_qwen36\out\open_qwen36_cli.exe
```

`VCVARS64` must be set: the scripts default to VS 2022 **BuildTools**, and this
box has Community **18**.

## Verify — the ladder, in order

The three commands in step 1 are **one chain**, not three choices:
`make_decode.py` writes the fixture and the fp64 reference, `run_kernel.exe`
produces `y_*`, `compare_decode.py` scores one against the other.

```powershell
python open_kernels\model\make_decode.py --model-dir <model dir> --layers 4 --tokens 2 --out open_kernels\model\out_gr
open_kernels\harness\out\run_kernel.exe open_kernels\model\out_gr\run_decode.cfg
python open_kernels\model\compare_decode.py --tokens 2 --out open_kernels\model\out_gr

src\open_qwen36\out\open_qwen36_cli.exe --model <model dir> --kernels src\xclbins\Granite-4.2-3B-NPU2\open_kernels `
    --ids 100264 --max-tokens 3 --layers 4 --dump-logits out_gr\y
python src\open_qwen36\chat.py "Explain what an NPU is in two sentences." --model <model dir> --kernels src\xclbins\Granite-4.2-3B-NPU2\open_kernels
```

**Prompt id 100264 = `<|start_of_role|>`.** *Not* `config.json`'s
`bos_token_id` 100283, which decodes to `</documents>` and disagrees with
`tokenizer_config.json`'s own bos (`<|end_of_text|>`). The container carries
both; only one is how a Granite prompt starts. `make_decode.py`'s per-family
default is 100264 for this reason.

Granite also closes **both** roles with `<|end_of_text|>`, so the turn end and
the eos are the same token — the other three families have a separate pair, and
`chat.py`'s Granite branch says so explicitly.

## Result — 2026-09-06, mlir-aie 1.4.2.dev16+g7e00b57 / Peano 21.0.0.2026080301

Correctness, 4-layer slice: logits corr **0.999998 / 0.999990**, same argmax
(38457) and an **identical top-5** at both positions, residual corr
0.999990–0.999999 in every layer. A coherent two-sentence answer through
`chat.py`, ending on `<|end_of_text|>` at token 52.

**Performance is the open question.** Decode **5.92 tok/s** over 63 tokens, and
the per-step cost grows with position:

| position | `part0` (40 layers) | `lm_head` | per layer |
|---:|---:|---:|---:|
| 18 | 91.9 ms | 4.0 ms | 2300 µs |
| 80 | 235.4 ms | 4.1 ms | 5900 µs |

About **+2.3 ms per position** (~58 µs per layer per position) while `lm_head`
stays flat, so the growth is the KV scan in attention, not the GEMVs.

Against the hand-written four-dispatch Granite kernels in
`vegah/FastFlowLM@feat/kernels` (**1744.7 µs/layer, 13.6 tok/s device time**),
one `dx` dispatch per layer did **not** win, and past roughly position 40 this
path is slower than that branch's CPU host engine (8.7 tok/s) too.

**Do not attribute this to the geometry without evidence.** Granite is the
first family measured at head_dim 64 / 40 heads, *and* the first measured with
a per-position breakdown at all — none of Qwen3-4B, Llama-3.1-8B or Gemma3-4B
has a published number at more than one position. The way to settle it is to
run the same breakdown on one of those (see below); until then the cause is
unknown.

## How to settle the context-growth question

Take one already-validated family, run a single decode step at several
positions, and record `part0`. `open_qwen36_cli --at-position P` seeks the KV
cache to P without decoding P tokens, so a sweep is cheap:

```powershell
foreach ($p in 0, 256, 1024, 2048) {
    src\open_qwen36\out\open_qwen36_cli.exe --model <model dir> --kernels <kernels> `
        --ids <family's first token> --max-tokens 1 --at-position $p
}
```

If `part0` grows the same way for Qwen3-4B or Llama-3.1-8B, the behaviour is
the dense recipe's attention at long context and Granite is merely the first
family anyone measured. If it stays flat there and grows here, it is specific
to 40 heads over 8 kv heads at head_dim 64. Either answer is worth having; the
current state is that nobody has looked.

Neither model is on this machine — `Granite-4.2-3B-NPU2` is the only one in
`~/.flm/models` — so the sweep costs one container download (Qwen3-4B is the
smaller at ~2.5 GB).

## Rules

- Never commit the xclbins; the distributed package ships them pre-built.
- A catalogue point enters `catalogue.py` **after** its hardware run, never
  before. `OPEN_KERNELS_UNVALIDATED=1` is how you get the run, not how you ship.
- Do not assert a validated set's membership in a test. Three tests had to be
  edited when Granite's points landed, all because they spelled the set out —
  which fails on the one event that is never a regression.
- Granite has **no closed engine** to fall back on (`llama_npu` whitelists
  `hidden_size` to {2048, 3072, 4096} and refuses 2560). A missing kernel set is
  a named refusal naming the build command, not a quiet substitution.
