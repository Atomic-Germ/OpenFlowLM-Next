# Open diffusion: FLUX.2 [klein] 4B text-to-image on the NPU

What the open image engine must do, and what has been measured. Directory name gives the
prefix: `OPEN-DIFFUSION`.

The model is `black-forest-labs/FLUX.2-klein-4B` (distilled, 4 steps, no CFG):
- the DiT has 5 double-stream and 20 single-stream blocks (hidden 3072, 24 heads of 128);
- the text encoder is Qwen3-4B layers 1-27 (taps 9/18/27);
- the VAE decoder is AutoencoderKLFlux2.

**The reference is the diffusers pipeline in bf16 on the CPU**
(`utilities/dit-ref/klein_quant_study.py`, 8 fixed prompts, seed 1234 + i). The goal set
by the user is maximum speed without broken images: drift is acceptable, breakage is
not.

The schedule is one list, `open_kernels/klein_pipeline.py`:
- text encoder, conditioning, 4 steps, VAE;
- 1050 dispatches over six kernel sets.

The exporter builds its streams (`open_kernels/export_dit_kernels.py`). Two runners replay
the same list:
- `utilities/dit-chain/generate.py` (pyxrt);
- `src/open_diffusion/` (native), from the bundle `utilities/dit-chain/export_bundle.py`
  writes.

## Requirements

### OPEN-DIFFUSION-NPU-ONLY: no host compute that grows with the data
**Applies to:** `open_kernels/klein_pipeline.py`, `utilities/dit-chain/generate.py`, `src/open_diffusion`
**Verification:** manual

During a generation the host may do only these things:
- tokenize the prompt;
- gather the prompt's 512 embedding rows;
- write the seeded noise;
- patch `te_attn`'s `valid_len`;
- read the RGBA and encode the PNG.

Everything else is an NPU dispatch. The host queues runs within one kernel set and blocks,
without polling, on the last run before switching sets.

**Verification (manual):** run `generate.py` (or the native CLI) on a quiet machine and
compare the process CPU time per image with the NPU wall time. The report prints both
(`host CPU`).

**Measured 2026-09-27** (pyxrt runner): 0.1-0.45 s of host CPU per image, against 6.5-7.4 s
(512²) and 20 s (1024²) of NPU time.

### OPEN-DIFFUSION-QUALITY: not broken, and within the predicted drift
**Applies to:** the whole pipeline
**Verification:** manual

The runner is given the study's fixed noise and prompts (`capture_pipeline_inputs.py`,
`generate.py --study`). Its images must meet all of these:
- they are coherent;
- text prompts render legible text;
- the LPIPS against the bf16 CPU run lands near what the CPU emulation of the NPU
  arithmetic predicts. Materially higher is a bug.

**Verification (manual):**

```
python utilities\dit-chain\generate.py --kernels <set> --size 512 --study C:\dev\ditref-out\goldens_pipe_512 --out <dir>
C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\score_images.py <dir> --test "{:02d}.png" --ref "{:02d}.png" --ref-dir C:\dev\ditref-out\klein_512_s4\bf16
```

Then look at the grid.

**Measured 2026-09-27**, 512², 8 prompts, LPIPS vs bf16 (noise floor: fp32 vs bf16 is
0.013):

| run | LPIPS mean / max | CPU-emulated prediction |
|---|---|---|
| DiT + VAE on the NPU, the bf16 text embeddings | 0.044 / 0.092 | 0.029 / 0.081 (DiT linears + attention only) |
| the same with diffusers' exact modulation vectors injected | 0.047 / 0.180 | |
| **everything on the NPU** | **0.107 / 0.191** | te-npu alone: 0.092 / 0.241 |

- All 16 images are coherent. "OPEN LATE" and "SOUP OF THE DAY: TOMATO" render legibly.
- At 1024² (2 prompts) the images are coherent too: LPIPS 0.33 and 0.15. Prompt 0's sign
  moves within the frame.
- The gap to the DiT prediction is the conditioning GEMMs. The study did not emulate
  them: timestep MLP, modulation and embedders. They share dit_gemm's bf16-accumulator
  arithmetic, which puts the modulation vectors 2-3% from diffusers'.
  - Injecting diffusers' exact vectors takes the 7 non-chaotic prompts from 0.038 to
    0.028.
  - Prompt 0 swings either way under any perturbation.
- The rest of the full pipeline's drift is the text encoder's padding rows. That drift
  is known, accepted, and not NPU-specific (`utilities/dit-chain/README.md`).

### OPEN-DIFFUSION-RESOLUTIONS: the supported sizes, and a named refusal for others
**Applies to:** `open_kernels/klein_pipeline.py`
**Verification:** test
**External tests:** none (`specs/open-diffusion/tests/test_host_setup.py`)

A square size R runs only when all of these hold:
- R is a positive multiple of 16 px;
- (R/16)² image tokens is a multiple of 512 (dit_gemm's M tile).

Any other size is refused with the reason named, never run wrong.

**Acceptance criteria:**
- 512 and 1024 pass `check_resolution`.
- 768 is refused because 2304 image tokens is not a multiple of 512.
- 520 and 0 are refused as not a positive multiple of 16.
- `plan(768)` raises.

### OPEN-DIFFUSION-SCHEDULE: the scheduler matches diffusers
**Applies to:** `open_kernels/klein_pipeline.py`
**Verification:** test

The host computes the flow-match sigmas: the exponential time shift with FLUX.2's
empirical mu, and a terminal 0. They must match diffusers' FlowMatchEulerDiscreteScheduler
bit for bit, because a changed sigma changes every image silently. Euler's dt reaches the
NPU as an fp32 in its parameter run.

**Acceptance criteria:**
- `sigmas(512)` = [1.0, 0.95808536, 0.88398188, 0.71749657, 0.0] (float32, exact).
- `sigmas(1024)` = [1.0, 0.96738404, 0.90814394, 0.76719993, 0.0].
- `dt_params` stores sigma[s+1] - sigma[s] as the fp32 at the parameter run's first
  vector.

### OPEN-DIFFUSION-PERF: what may be called a performance number
**Applies to:** the whole pipeline
**Verification:** manual

A time quoted for this pipeline must meet all of these:
- it is a warm generation, not the first after load;
- the NPU is in turbo mode;
- nothing else is loading the CPU. The CPU shares the package power budget, and a busy
  CPU slows the NPU 1.2-1.6×.

A number measured under load says so.

**Measured 2026-09-27, native engine** (`src/open_diffusion`), turbo, 2 prompts × 3 warm
runs each. **Another ~11-core CPU job (another session's) was running**, so these are
pessimistic:

| | 512² | 1024² |
|---|---:|---:|
| image, warm | 5.2-6.0 s (mean 5.6) | 13.5-15.9 s (mean 15.1) |
| text encoder | 0.65-0.80 s | 0.64-0.75 s |
| conditioning | 0.04-0.06 s | 0.04-0.06 s |
| one denoising step | 1.0-1.25 s | 2.9-3.5 s |
| VAE | 0.46-0.54 s | 1.27-1.46 s |
| first image after load | 5.7-6.3 s | 15.0-15.8 s |
| load (kernels + 7.5 GB weights, cached) | 8-9 s | 6-7 s |
| host CPU per image | 0.03-0.19 s | 0.03-0.11 s |

**Earlier the same day, pyxrt runner** (`generate.py`), with a different 12-core job
running. These are slower: Python dispatch, and more contention. They were pessimistic
too:
- the standalone attention benchmark ran 42.7 ms against its quiet 30.5 ms;
- `sgl_in` ran 68.7 ms against 59.4 ms.

| | 512² | 1024² |
|---|---:|---:|
| image, warm | 6.5-7.4 s | 20.0-20.9 s |
| text encoder | 0.76-0.89 s | 0.86-0.88 s |
| conditioning | 0.06 s | 0.07 s |
| 4 denoising steps | 5.1-5.8 s | 17.2-17.9 s |
| VAE | 0.60-0.67 s | 1.86-2.07 s |
| first image after load | 9.5-11 s | 29 s |
