# K2-Horizon-7B-Uno on the NPU

2026-10-08. Status: **approved, in progress** (no performance gate: build all phases, report
the speed). Defaults taken on the four decisions: near-tie bar, greedy first, one container,
dense-first L-row pass.

## Progress

| Phase | State |
|---|---|
| 1. K2-Horizon-7B AR | **done**: GroupRMSNorm(4), `OPEN-FAMILY-K2`, 9.56 tok/s on the NPU. A converter bug was found and fixed on the way: `-f k2` scrambled the q/k rows. |
| 0a. CPU reference | **done** (`utilities/uno-ref`): 3.94 tokens per cycle at L = 8, 3.45 at L = 4, q4_1 base; a q4_1 LoRA loses nothing |
| 2. L-row pass | **done** (`designs/dxl`, OPEN-DECODE-ROWS): the layer and the head with an NPU argmax are bit-identical to decode; `Core::step_rows`, `--rows-check` PASS; a pass costs 1.17-1.28x one step. |
| 3. LoRA | **done** (OPEN-UNO-LORA): `uno.q4nx` (`q4nx-build --uno-adapter`), the draft stream sharing dxl's context; the draft layer matches fp64 at corr >= 0.999998. |
| 4. Uno decode | **done** (OPEN-UNO-DECODE): `--uno` is identical to plain decode (64 tokens, then 6 prompts x 128); the `oflm` `k2` family uses it for greedy requests. |
| 5. Serving | **done**: minja renders K2's template as transformers does (TOOLS-K2-TEMPLATE, 10/10); reasoning and `<ifm|tool_call>` parsing (TOOLS-K2-REASONING / -CALLS); history fix-ups (TOOLS-K2-HISTORY); `<|ifm|im_end|>` stops generation (OPEN-CONVERT-EOS-GENCONFIG). `oflm serve`: greedy, sampled, streamed, multi-turn, tool call and tool result all correct. |
| Speed (reported) | Uno identical to decode on 6 prompts x 128 tokens, 3.47 tokens a cycle, median 1.55x (range 0.85-1.73x), on a machine at 100% CPU where decode ran ~320 ms/token against 105 clean; `oflm serve` greedy 5.6 tok/s vs sampled 3.8 tok/s on the same machine. A clean absolute Uno number is still missing. |

Findings that changed the design:
- **The L-row pass can be bit-identical to decode, not just within the near-tie bar.** Per
  token, the L-row GEMV does gemv_q4_tile's arithmetic in the same order; the norms, the
  attention (one query row at a time) and silu are dx's own kernels. Measured: 0 of 4096
  values differ on a real layer. So Uno's output equals the NPU's own AR output exactly.
- **L = 4 is the working point.** The GEMV costs ~1.1x at L = 4 and ~1.9x at L = 8: at L = 8
  each chunk takes two tile4 passes on 8 cores. Acceptance beyond three drafts is rare
  (3.45 tokens per cycle at L = 4 against 4.04 at L = 8).
- **The LoRA is stored at q4_1, as base-pool bands**, and runs through the same GEMV:
  - A is 8 extra bands per projection group, one per core (zero-padded to 512 rows).
  - B is one extra k-tile slice per band, zero-padded from 128 to 256 columns.
  - The seed row's mask is a DMA: token 0 of the B slice's x stream is read from a zero row.
  - Overhead: ~12 MB per layer (~10%), on the draft pass only.

## What Uno is

`IFM/K2-Horizon-7B-Uno` is not a model. It is a PEFT LoRA adapter over `IFM/K2-Horizon-7B`:
- 504 fp32 tensors, 1.4 GB, 349M parameters.
- r = 128, lora_alpha = 8192, so the scale is **64** (plain alpha / r, not rsLoRA).
- Targets q/k/v/o/gate/up/down in all 36 layers.

The base weights alone are the normal autoregressive (AR) model. The base plus the LoRA is a
drafter that guesses several tokens at once. IFM's inference code
(`github.com/ifm-ai/uno`, `nano_vllm_uno/engine/two_pass_decoding.py`) runs a **cycle of two
forwards, each L = 8 rows**:

1. **Draft pass.**
   - Input: `[t_n, noise_1 .. noise_{L-1}]`. t_n is the last committed token (the seed). The
     noise tokens are uniform random ids in [1, 250624).
   - Row 0 runs without LoRA. Its logits are exactly the base model's next-token logits; that
     sample is `c`.
   - Rows 1..L-1 run with LoRA and give drafts d_1..d_{L-1} for the positions after c.
   - The KV rows written for the noise positions are dropped.
2. **Verify pass.**
   - Input: `[c, d_1 .. d_{L-1}]`, no LoRA, the normal causal mask.
   - Row j-1 checks d_j: greedy is an argmax match; T > 0 uses standard rejection sampling.
   - The last row gives a bonus token when every draft is accepted.
3. **Commit.** c, the accepted drafts, then either the correction or the bonus token: between 2
   and L+1 tokens per cycle. The KV frontier moves to the new length minus 1.

Facts that shape the port:
- **Lossless.** The output distribution is the base model's under the same sampling settings.
  Under greedy decoding the tokens are identical to plain AR (given identical logits).
- **LoRA is a per-row mask inside each target linear:** `y += 64 · (m ⊙ (x·Aᵀ))·Bᵀ`. The
  embedding, norms and lm_head get no LoRA. The AR path never uses it.
- **Attention is causal in both passes,** not bidirectional, over one shared KV cache.
- **Consequence: L is a runtime knob.** The first k rows of an L = 8 pass are exactly an
  L = k pass.
- **IFM's reported speedup:** an average of 2.71 tokens per forward, about 5.4 tokens per
  cycle. That was measured at T = 1.0 on their benchmarks. Greedy decoding and chat are not
  measured.
- The draft and verify passes are separate; IFM does not fuse them.
- The draft tree is optional, FA3-only and off by default. Out of scope here.
- Nothing closed-source blocks inference.
- The K2 training recipe is unreleased. The noise distribution is taken from IFM's own 8B
  inference defaults (`inference.py:52-55`, `examples/uno_8B/run_eval.sh`).

## What we have and what's missing

**The base model.** `IFM/K2-Horizon-7B` is the same family as `K2-Horizon-3.7B`, which already
runs on the dense recipe (#139: 65 ms/token at ctx 512 on the NPU).

7B shape:
- 36 layers, hidden 4096, 32/8 heads, head_dim 128, FFN 12288.
- Untied head, vocab 250624.
- θ 1e7, eps 1e-6, split-half RoPE. The adapter's q/k rows are already in HF order, so no
  permutation is needed.
- **GroupRMSNorm with 4 groups of 1024.**

The widths are Qwen3-8B's. `gemv_q4` K = 4096 / 12288, attention (128, 32, 8, 128) and
`lm_head_q4` K = 4096 are all in the validated catalogue. **The gaps:**
- **The norm.** `ln.h`, `ln_nr.cc`, `ln_xn.cc`, `ln.py` and `spec.py` accept 1 or 2 groups
  only.
- **No spec requirement.** `tests/test_k2.py` traces to `OPEN-FAMILY-K2`, but that requirement
  is missing from `spec.md`; #139 never added it.

**The Uno path.** None of it exists:
- The dense recipe decodes one token per step. The block route is a 256-token prefill with
  host norms.
- There is no L-row pass, no LoRA, and no on-NPU argmax/top-k. `cli.cpp:64` takes the argmax
  on the host from the full logits.

## Expected speed (reported, not a gate)

Weights read per AR token at q4_1: about 8.0B parameters, about 5.0 GB. Scaling the 3.7B's
measured 65 ms by its weight bytes gives an estimated **110-130 ms/token, about 8 tok/s**
(estimate, not measured).

A cycle costs r_draft(L) + r_verify(L) single-token steps, where r(L) is the cost of an L-row
pass relative to one token. LoRA adds about 7% to the draft pass at q8. At about 5.4 tokens
per cycle (IFM's figure, L = 8):

| r(8) | cycle (steps) | speedup |
|---|---|---|
| 1.5 | 3.07 | ~1.75x |
| 2.0 | 4.07 | ~1.35x |
| 2.7 | 5.47 | ~1.0x |

What we know about r: Bonsai's verify pass costs **1.25x at 2 rows** on this NPU, and its
4-wide GEMV costs 1.36-1.39x (it is core-bound on the per-token epilogue; see
`C:\code\bonsai-specdec\.claude\plans\specker-verify.md`). K2 has no DeltaNet, which was most of
Bonsai's per-row cost. Nobody has measured 8 rows.

**Reported at the end:** AR tok/s, Uno tok/s, and tokens per cycle at L = 2, 4 and 8 on chat,
code and reasoning prompts. The default L is set from those numbers.

## Phases

### Phase 0: CPU oracle (alongside Phase 1)

**0a. CPU oracle,** `utilities/uno-ref/`:
- Python with torch on the CPU, transformers with `trust_remote_code`. The machine has 88 GB;
  bf16 needs about 16 GB.
- Forward hooks apply the per-row LoRA mask, and the oracle runs the two-pass cycle for greedy
  and for T > 0.
- **Outputs:**
  - Tokens per cycle and the accepted-length distribution on chat, code and reasoning prompts.
    Tokens per cycle at L < 8 follows from that distribution, because the attention is causal.
  - The same figures with the base weights round-tripped through q4_1 (what our container
    holds) and the LoRA at bf16, q8 and q4_1.
  - A check that greedy Uno output is identical to greedy AR output.
- This also becomes the oracle for Phases 3 and 4.
- It is offline only, and it must not run while the NPU is being timed.

**0b. NPU ratio** (design input for Phase 2, not a gate): time `designs/gemm_q4` in blocked
mode at M = 1, 2, 4 and 8 on the K2-7B shapes (qkv 6144×4096, o 4096×4096, gate|up
24576×4096, down 4096×12288, the head 250624×4096) against the shipped `gemv_q4`.

### Phase 1: K2-Horizon-7B, AR, on the dense recipe (needed whatever Phase 0 says)

- **Norm kernels:**
  - Generalise `LN_GROUPS` from {1, 2} to any G that divides N with N/G a multiple of 32.
  - Builds at G = 1 and G = 2 must stay byte-identical; compare `insts.bin` against today's sets.
  - Accept 4 in `ln.py`, `ModelSpec.from_dict` and `_k2_hf`.
- **Recipe:**
  - Add the catalogue point `ln (4096, 4)`.
  - Add `recipes/specs/k2-horizon-7b.json` and the name `K2-Horizon-7B-NPU2` in
    `test_spec_model_names.py`.
- **Container:** convert `IFM/K2-Horizon-7B-GGUF` (the BF16 or Q8_0 file) with
  `q4nx-build -f k2` to a q4_1 container. Confirm that `k2.json` needs no change at this size.
- **Hardware:** the OPEN-FAMILY procedure:
  - slice logits against the fp64 replica, every layer;
  - decode through the engine;
  - `chat.py`;
  - `oflm-test --llm`.
- **Ship:**
  - The spec requirement `OPEN-FAMILY-K2`, covering the 3.7B and the 7B.
  - The skill `.opencode/skill/open-k2-kernels/SKILL.md`.
  - `K2-Horizon-7B-NPU2` runnable with `oflm`.
  - Publishing to Hugging Face (`Atomic-Germ/K2-Horizon-7B-OpenNPU2`) only on your go-ahead.

### Phase 2: the L-row pass (OPEN-DECODE-ROWS)

L ≤ 8 positions (chosen at run time) go through every layer:
- One hardware context for the whole pass.
- Causal attention over the cache plus the earlier rows of the block.
- KV is written for all L rows. The caller then commits a prefix; for a dense model, rollback
  only moves the frontier.

The pieces:
- **Projections:** the `gemm_q4` blocked dataflow (M ≤ 24) as a shipped kernel at the K2
  shapes. It reads the same pool chunks as the GEMV, so there is no second copy of the weights.
- **Per-row stages:** GroupRMSNorm, RoPE, L-query attention, SwiGLU and the residual, all on
  the NPU.
- **Head:** the lm_head for L rows in one pass over its weights, with an argmax / top-k (k ≤ 64)
  epilogue. The 8 MB of logits per pass never reach the host.
- **Check:** each row's logits against the 1-row decode step at the same position. The bar is
  decision 1 below.

This is the dense half of the paused 35B plan's verify pass (`OPEN-SPEC-VERIFY`, branch
`feat/spec-decode`), built without the experts that made that one expensive. It also gives every
dense model plain speculative decoding later.

### Phase 3: conditional LoRA on the NPU (OPEN-UNO-LORA)

- **Container:**
  - The adapter's tensors are packed beside the base, with the ×64 folded into B.
  - Format from 0a: q8 is expected; q4_1 if acceptance holds. The drafts' precision changes only
    speed, never the output.
- **The x·Aᵀ product rides inside the base projection's GEMM** as extra weight rows:
  - A_q|A_k|A_v adds 384 rows to qkv.
  - A_g|A_u adds 256 to gate|up.
  - A_o adds 128 to o, and A_d adds 128 to down.
- The row mask zeroes z on the seed row.
- ΔY = z·Bᵀ is a K = 128 GEMM (block-diagonal K = 384 for qkv), added into Y on the NPU.
- **Cost:** about 0.37 GB more weight read per draft pass at q8 (~7%).
- **Check:** draft-row logits against the oracle at the same formats, and tokens per cycle on
  the NPU within noise of the oracle's.

### Phase 4: Uno decode, container, `oflm` (OPEN-UNO-DECODE)

- **Engine:** the cycle exactly as the reference.
  - Greedy first.
  - Then T > 0: accept if u < min(1, p/q); the correction comes from norm(max(p−q, 0)), over
    the NPU's top-k supports.
  - Host work per cycle is choosing among L·k candidates: constant size, with blocking waits.
  - Prefill is the existing block route, unchanged.
- **Converter:** a q4nx-build builder that takes `IFM/K2-Horizon-7B-Uno` plus its base and
  produces `K2-Horizon-7B-Uno-NPU2`. It links to the K2-Horizon-7B kernel set because the spec
  is the same.
- **oflm:**
  - Decode mode `uno` is the default when a container carries the adapter; `--decode-mode ar`
    turns it off.
  - Log tokens per forward and tok/s.
- **Tests:** `oflm-test --llm` on the Uno container. Greedy AR and Uno outputs must be
  identical on its prompts, within the near-tie bar.
- Update the skill.

## Spec impact

New requirements:

| ID | Verification |
|---|---|
| `OPEN-FAMILY-K2` (3.7B + 7B) | test: the derivation, `norm_groups` 4, the catalogue point (extend `test_k2.py`); manual: the hardware procedure |
| `OPEN-DECODE-ROWS` | manual: an L-row pass agrees with L single steps |
| `OPEN-UNO-LORA` | test: the adapter packing in `q4nx-build/tests` (scale folded into B, the A-row concatenation order, q/k rows not permuted); manual: on-NPU numerics |
| `OPEN-UNO-DECODE` | test: the cycle's bookkeeping against a fake core with fixed logits (accept, correction, bonus, frontier), and rejection sampling with a fixed RNG against a reference; manual: lossless on hardware |
| `TOOLS-K2-TEMPLATE` (specs/tool-calling) | manual: `utilities/template-check/check.py` against transformers |
| `TOOLS-K2-HISTORY`, `TOOLS-K2-REASONING`, `TOOLS-K2-CALLS` (specs/tool-calling) | test: `src/test/k2_chat` |

Modified: `OPEN-SPEC-DERIVE` (K2 accepts `layernorm_num_groups: 4`).

## Decisions for you

1. **How strict is "lossless"?** I recommend (b).
   - (a) Bit-identical to the AR decode route. This is Bonsai's bar; it ties each row to the
     1-wide arithmetic and is slower to build.
   - (b) Argmax-identical except on near-ties, with the bound measured. This is the 35B plan's
     bar.
2. **Greedy-only first Uno PR, T > 0 sampling in a follow-up?** I recommend yes.
3. **Packaging.** I recommend (a).
   - (a) One Uno container with the base and the adapter: about 5.4 GB, and `oflm-add` is
     unchanged.
   - (b) An adapter-only container (about 0.4 GB) that resolves an installed
     `K2-Horizon-7B-NPU2`. This needs base resolution in `oflm-add`.
4. **Build the L-row pass dense-first here, then port it to the 35B's verify?** I recommend
   yes; the 35B plan stays paused until then.

## Risks

- **r(8) too high.** Then Uno at L = 8 may be no faster than AR. L is a runtime knob, so the
  default comes from the measurements, and `--decode-mode ar` stays available.
- **Acceptance on a q4_1 base.** The LoRA was trained against bf16 hidden states, and 0a
  measures the drop. If it is large, a q8 attention/FFN container trades AR speed for
  acceptance.
- **Greedy and chat tokens per cycle are unknown.** IFM's 2.71 is T = 1.0 on reasoning and code
  benchmarks.
- **Other sessions share the NPU.** Check `tasklist` for `open_qwen36_cli` / `npu.exe` before
  timing anything.

## Not in this plan

- The draft tree.
- Fusing the next draft into the verify pass. The causal attention makes this possible, but it
  only pays when every draft is accepted; revisit with Phase 4's numbers.
- Other K2 sizes: the 0.9B (YaRN), MoVA, the 375B.
