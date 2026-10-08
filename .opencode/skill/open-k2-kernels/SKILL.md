---
name: open-k2-kernels
description: Convert, build, verify and run IFM K2-Horizon (3.7B, 7B) on the open dense recipe. Use when packing a K2 GGUF, rebuilding K2 xclbins, adding another K2 size, or when a K2 model is fluent but "a bit off" (the q/k row order).
---

# K2-Horizon on the dense recipe

K2 is a generic dense family (`model_type: k2_horizon`): GQA 32/8 at head_dim 128,
no q/k norm, gate or bias, split-half RoPE (θ 1e7 under `rope_parameters`), silu FFN,
an untied 250624-row head, and **GroupRMSNorm**: `layernorm_num_groups` contiguous
groups each normalised on its own (3.7B: 2 x 1280, 7B: 4 x 1024). Spec requirement:
OPEN-FAMILY-K2 in `specs/open-engine/spec.md`.

## The two traps

1. **q/k row order.** A `k2-horizon` GGUF already holds HF's split-half q/k rows.
   `q4nx-build -f k2` used to apply Llama's un-interleave anyway (fixed in
   `q4nx/models/k2.py::_gguf_qk_interleaved`). A scrambled pack is still fluent, so
   check a likelihood, not the text: 7B NLL 2.44 correct vs 2.96 scrambled.
   `~/.flm/models/K2-Horizon-3.7B-NPU2-scrambled` is such a pack, kept for comparison.
2. **chat.py picks the wrong template.** K2's tokenizer contains `<|start_header_id|>`,
   which `chat.py` takes as Llama 3. Build ids with transformers'
   `apply_chat_template` on IFM's `chat_template.jinja` and feed `open_qwen36_cli --ids`.
   The template opens a `<ifm|think>` block; the model reasons first.

## Convert (about 3 minutes for the 7B)

```bash
hf download IFM/K2-Horizon-7B-GGUF K2-Horizon-7B-BF16.gguf --local-dir C:/models/k2-7b
# config/tokenizer/template as the skeleton, so -s does not pull 16 GB of safetensors
curl -L -o base/config.json https://huggingface.co/IFM/K2-Horizon-7B/resolve/main/config.json   # + tokenizer*.json, chat_template.jinja
cd utilities/q4nx-build
python convert.py -i C:/models/k2-7b/K2-Horizon-7B-BF16.gguf -o C:/models/k2-7b/K2-Horizon-7B-NPU2 \
    -f k2 -t language -s C:/models/k2-7b/base --source-repo IFM/K2-Horizon-7B-GGUF --quant Q4_1
```

The "rope_freqs.weight absent" warning is expected: K2 has no such tensor. The 7B
container is 7.0 GB.

## Build (Windows, about 20 minutes with -j 4)

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd <repo>\open_kernels
python export_qwen36_kernels.py --model-dir C:\models\k2-7b\K2-Horizon-7B-NPU2 -j 4
```

-> `src/xclbins/K2-Horizon-7B-NPU2/open_kernels/` (`dx`, `ln`, `lm_head_q4`, the
prefill GEMMs, the attention tiers). A new K2 width needs `OPEN_KERNELS_UNVALIDATED=1`
until its `ln (width, groups)` point has passed on hardware.

`src/open_qwen36/build.cmd` stops at `pools_test` (it crashes in
`mixed_container_tests` on main as of f31ecaa, unrelated to K2). To build the CLI
alone, run the `cl` line for `open_qwen36_cli.exe` from that script directly
(vcvars64 from VS 2022 BuildTools).

## Verify

```powershell
# the norm on its own: groups scaled differently, so pooled statistics fail
$env:LN_N="4096"; $env:LN_EPS="1e-06"; $env:LN_GROUPS="4"
python build_design.py designs/ln/ln.py designs/ln/build_4096_g4
cd designs/ln; $env:LN_BUILD="build_4096_g4"; python make_test.py; run_kernel run.cfg; python compare.py
# a 4-layer slice against the fp64 replica (prompt id 0 = <|ifm|begin_of_text|>)
python model/make_decode.py --model-dir C:/models/k2-7b/K2-Horizon-7B-NPU2 --layers 4 --tokens 2 --out model/out_k2
run_kernel model/out_k2/run_decode.cfg; python model/compare_decode.py --tokens 2 --out model/out_k2
# the whole model
open_qwen36_cli --model C:/models/k2-7b/K2-Horizon-7B-NPU2 --kernels src/xclbins/K2-Horizon-7B-NPU2/open_kernels --ids <ids> --max-tokens 120
```

## Result 2026-10-08 (7B, Strix)

`ln` 4096x4 PASS (cos 0.99999999); slice logits corr 0.999999 at both positions,
argmax matching, residual corr >= 0.999999; **105 ms/token (9.56 tok/s)** at
positions 20-83, coherent text.

## K2-Horizon-7B-Uno

The Uno adapter (a diffusion-drafter LoRA over the 7B) is planned in
`specs/open-engine/plans/k2-horizon-7b-uno.md`. CPU reference: `utilities/uno-ref/uno_ref.py`.
