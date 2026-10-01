# Review: Phase 8, image edits (`edits.md`)

2026-09-30. Reviewed against diffusers 0.40.0 (`C:\dev\ditref-venv`), klein's VAE config in the
HF cache, and the tree at `090f8ad`.

## Summary

The plan adds reference-image edits to the klein engine. It keeps every op on the NPU, adds a
VAE encoder, extends the DiT's joint sequence to [txt | gen | ref], and ships two edit
configurations (512e512 and 1024e1024) with a republished model. It estimates about 3 weeks.

**Verdict:** the kernel design holds up, and its reading of diffusers is accurate. It is not
ready to implement as written. The blockers are at the edges: a CLI flag that already exists,
publish/pull mechanics that don't work the way the plan assumes, untrusted image decoding in
the server, and a `test` requirement with no acceptance criteria.

### What I checked and found correct
- **diffusers mechanics:**
  - encode with argmax, patchify, then BN normalise (K:463-476);
  - preprocessing (K:770-782): the pure centre crop is `Flux2ImageProcessor._resize_and_crop`,
    which only crops;
  - mu counts generated tokens only (K:815-816);
  - references are concatenated last and sliced off (K:845-860);
  - position ids (K:266-366).
- **VAE config:** `batch_norm_eps` 1e-4, `use_quant_conv`, 32 latent channels, 4 down blocks
  (so 3 downsamples), mid attention.
- **Encoder cost:** recomputed layer by layer at 0.558 TMAC for 512², which matches.
- **Speed table:** consistent with the Phase 7 profile (`archive/phase7-speed.md:28-44`).
  - At 1024², attention is ~0.92 s of the 2.63 s step and the GEMMs ~1.58 s.
  - Scaled to L = 8704 that gives ~6.7 s per step, against the plan's ~7 s.
- **Code references:**
  - `dit_ew` uses 21 of `RTP_LEN = 24` slots (`dit_ew.py:75`).
  - `dit_conv` is stride 1 only.
  - `ImageUpload` holds metadata only (`rest_handler.hpp:35-40`).
  - The only stb header in the tree is `stb_image_write.h`.
  - ELFs are keyed by the resolution string (`engine.cpp:333`).
- **Shape constraints at the new lengths:** L = 2560, 5632 and 8704 all satisfy
  `dit_fa`'s `tokens % 512` and `dit_gemm`'s M tile.

---

## Unclear points

1. **"Details in the research notes of 2026-09-30"** gives no path. The only match is
   `.claude/plans/handoff-2026-09-30.md:110`, which just lists the encoder as not done. Link
   the notes or inline the parts the plan depends on.
2. **"(a), built 512 first"** doesn't say whether 512e512 goes through 8.4-8.5 (packaging,
   CLI, server) before 1024e1024 is started, or only through 8.2-8.3.
3. **x_emb on REFLAT "each step".** The result doesn't change from step to step. It still has
   to be rewritten every step, because the blocks overwrite the X buffer's reference rows.
   Say so, so nobody later "optimises" it into a stale read. It costs 1.6 GMAC per step at
   T_ref 4096, which is negligible.
4. **The CPU emulation's scope for the edit gate.** Say which components it emulates (DiT
   linears + attention, as today?) and whether it takes diffusers' reference latents or the
   NPU encoder's. Otherwise EDIT's predicted LPIPS mixes encoder drift with DiT drift.
5. **The CLI line omits the model tag:** `oflm image "<prompt>" --image in.png`. The existing
   grammar is `oflm image <tag> "<prompt>"`.
6. **The first edit after load.** The engine prefetches one size (`engine.cpp:519`). The
   first edit pays kernel creation, as the plan says. Should OPEN-DIFFUSION-PERF state "first
   edit after load" as its own row?

## Production readiness

**Blockers:**

- **B1. `--image` is taken.**
  - It is a serve-only boolean (`vm_args.hpp:96`, "keep the image engine resident") and is
    explicitly refused on every other command (`vm_args.hpp:276`).
  - `oflm image <tag> "<p>" --image in.png` fails either as a non-serve use of `--image` or
    as a bool parse of `in.png`.
  - 8.5's CLI needs another name.
- **B2. Republishing doesn't reach existing installs, and it breaks the open PRs.**
  - `oflm pull` downloads only files that are **missing** (`model_downloader.cpp:421`). A
    hash mismatch is advisory: the file is kept and verification passes
    (`model_downloader.cpp:618`).
    - So "existing installs would be refused until re-pulled" is wrong. A re-pull adds the
      new files and keeps the stale ones with the same names, such as `bundle.json` and
      `schedule_*.json`. The install stays refused until the user deletes the model
      directory.
  - The registry reads its file list and hashes live from HF `tree/main`
    (`src/model_list.json:1342`). Overwriting `main` therefore breaks every build of #136-#141
    (layout refused), including the PACKAGE and DETERMINISM checks already recorded for
    those PRs.
  - Nothing is on `main` yet (#136-#141 are all open), so no released user is affected. The
    other option in decision 6, scoping the layout hash, is moot. The real question is
    revision pinning.
- **B3. The server would decode untrusted bytes with stb_image.**
  - The server accepts bodies up to 256 MB (`server.hpp:163`) and can bind beyond localhost
    (`--host`).
  - stb_image is not hardened against malicious files. It has a long fuzzing and CVE
    history, and it has no default cap on decoded size: a small PNG can declare
    60000×60000.
  - The plan lists "PNG, JPEG, BMP, …", which is more surface than needed.
- **B4. OPEN-DIFFUSION-REFERENCE is `test` with no acceptance criteria**, and its golden is
  undefined.
  - `stb_image_resize2` has no Lanczos filter: box, triangle, cubic B-spline, Catmull-Rom,
    Mitchell, point, or a custom callback. It will not reproduce PIL's LANCZOS.
  - If the golden is PIL's output, the test needs a tolerance.
  - If the golden is the code's own output, it is a snapshot and proves nothing.

**Should fix:**

- **S1. EXIF orientation.**
  - Phone JPEGs carry a rotation tag. stb_image ignores it, and `diffusers.utils.load_image`
    applies it (`exif_transpose`).
  - Without it, edits of phone photos come out sideways, and nothing flags it.
- **S2. The host cost is understated.**
  - "~10-30 ms" holds for a 512² PNG.
  - A 12 MP phone JPEG through stb_image plus a resize is more likely 0.1-0.3 s (estimated,
    not measured).
  - That is still small against 7.5-31 s, but it is host CPU that NPU-ONLY's report counts.
    Measure it.
- **S3. Resident memory across configurations.**
  - The engine keeps each selected configuration's context and activations for its lifetime
    (`engine.cpp:159`, `:270`).
  - A server that has served 512, 1024, 512e512 and 1024e1024 holds all four sets of
    activations (~9 GiB for 1024e1024 alone) plus 7.5 GB of weights, pinned, beside any
    resident chat model (`--image 1`).
  - "87.6 GB, so memory isn't the limit" is argued per configuration. State the resident
    total and the NPU-visible budget, or evict the least recently used configuration.
- **S4. The base branch.**
  - `feat/edits` sits on #141, which the Phase 7 notes call parked, five open PRs deep.
  - The edit numerics inherit #141's `FA_TAU = 32`.
  - Decide whether #141 is un-parked or `feat/edits` rebases onto #140 before 3 weeks of
    work land on a parked base.

## Architecture review

**Sound:**
- the encoder as one more schedule beside `vae_decoder.py`, reusing `dit_conv`, `vae_ew` and
  the rank-128 attention;
- references appended last, so Euler and `proj_out` stay untouched;
- one ELF per configuration, in the same hardware-context model as Phase 6;
- no masks and no multiple references in this phase.

**Weak spots:**

- **A1. Decision 2's rationale is incomplete.**
  - `dit_fa` already has a **valid key length** RTP (`dit_fa.py:38-43`, used for the text
    encoder's padding).
  - Tail-padding the reference to a 512 multiple and masking the padded keys would let one
    DiT configuration take any reference aspect with T_ref ≤ 4096. diffusers caps the area at
    1024², so its references never exceed 4096 tokens.
  - The real per-shape costs are:
    - the encoder: GroupNorm statistics over the whole tensor rule out running it on a
      padded canvas;
    - `qk`'s reference `grid_w`.
  - So square-crop is still the right call for Phase 8, for a reason other than the one
    stated. The growth path (aspect buckets = encoder streams per bucket plus `valid_len`
    on the DiT) should be written down so the API doesn't foreclose it.
- **A2. The product impact of square-crop is understated.**
  - "Matters only for non-square or small inputs" covers most real inputs: phone photos are
    4:3 or 3:4, and screenshots 16:9.
  - Every such edit loses its sides and comes out square.
  - Small inputs are upscaled to R, where diffusers never upscales, so a 300² input edited
    at 1024 is a 3.4× upscale that the model will faithfully reproduce as blur.
  - Both deserve a printed notice (CLI) and a spec line. diffusers' input refusals (a side
    under 64 px, aspect over 8:1) aren't mentioned; with centre-crop, an 8:1 panorama keeps
    1/8 of its width.
- **A3. The default size.**
  - With `--size` defaulting to 1024, the default edit takes ~31 s and upscales small
    references.
  - diffusers' default is that the output follows the reference.
- **A4. Spike 2 (the `qk` RoPE branch) is overstated as a risk.**
  - A reference row is an image row with axis 0 rotated by t = 10. That is `FINE[10]` in
    the existing table: `FINE` covers p < 64, and all four axes share θ and dim
    (`ew.cc:163-178`).
  - In same-size configurations the reference `grid_w` equals the generated one, and
    `n_gen = grid_w²`, so possibly no new RTP at all. `RTP_LEN` is a design constant anyway.
  - The addition is a few instructions inside an already-`noinline` function.
- **A5. A missing spike: the long-L streams.**
  - Nothing in 8.1 checks that the existing kernels run at L = 8704:
    - `dit_fa` walks 136 key chunks per pass, against 72 validated. Its header records shim-BD
      wraps and hangs tied to stream shape (`dit_fa.py:19-22`).
    - The lazy rescale's τ = 32 safety argument is validated only to 4608.
  - This is cheap to check and could sink 1024e1024 after the encoder is already built.
- **A6. The silent-failure guard is too weak.**
  - A DiT that ignores the reference (wrong t offset, reference rows mis-strided or masked)
    still produces coherent images, and "coherent" passes.
  - Only the LPIPS threshold catches it, and that threshold isn't set.
- **A7. The golden distribution is narrow.** Using the model's own 8 study images as the only
  references tests the encoder on synthetic, in-distribution pixels. Natural photos (texture,
  noise, faces, JPEG artefacts) are the actual input.

## Alternative approaches

1. **Stride-2 without a new op (for spike 1).**
   - Run the residual `add` that feeds each downsample as 4 phase dispatches. Each one:
     - reads one (row, pixel) parity with pitch ×2 and `px_stride` 2C;
     - writes `px_stride` 4C at channel offset (2p + q)·C.
   - Every stride is uniform per phase, so ordinary `vae_ew` views may express it.
   - That `add` carries no `stats`, like the decoder's adds before its upsample convs
     (`vae_ew.py:5-6`), so splitting it costs no GroupNorm bookkeeping.
   - Cost: 9 extra dispatches per image.
   - Try this before a `vae_ew` `s2d` op or the 3-day `dit_conv` memtile change.
2. **Aspect buckets via `valid_len`** (A1), as a post-Phase-8 path rather than per-shape DiT
   configurations.
3. **A vertical slice.** Take 512e512 through encoder → DiT → CLI → server → determinism,
   then add 1024e1024 as mostly exporter work. The fast configuration exercises every
   integration seam first, and a late surprise at 1024 doesn't block shipping 512.
4. **Pin the model revision instead of overwriting `main`** (B2). The downloader already
   honours a `resolve/<rev>` base URL (`model_downloader.cpp:423`).

---

## Recommendations

1. **Keep `--image <file>` for the reference and rename serve's flag** (owner's decision,
   2026-09-30).
   - `--image` matches the `image` field of `/v1/images/edits`.
   - Serve's flag was added in #137 (`7f67e86`), which is unmerged, so nothing released
     changes.
   - Proposed name: `--imagegen 1`, beside `--imagemodel`. It keeps the `--asr 1` / `--embed
     1` pattern. Avoid `--images`: one letter from `--image`, it turns a typo into the other
     flag.
   - The new `--image <file>` is refused by every command except `image`, like `-o`,
     `--size` and `--seed`. Fix the plan's example to include the tag.
   - Places to update:
     - `vm_args.hpp:57`, `:96`, `:100`;
     - the comment at `program_args.hpp:54`;
     - the messages at `rest_handler.cpp:431` and `:434`;
     - SERVER-IMAGES-RESIDENCY (`specs/server-api/spec.md:388-416`);
     - `docs/docs/instructions/server/openapi.md:427`, `:430`;
     - `.opencode/skill/open-diffusion/SKILL.md:136`;
     - the comment at `test_images_api.py:275`.
     Archived plans stay as written.
   (B1)
2. **Publish the edit model to a new HF revision or branch,** and point `url` / `file_url` in
   `model_list.json` at it, so builds of #136-#141 keep pulling the layout they can run.
   Rewrite decision 6 around this: the hash-scoping option is moot, since nothing is on
   `main`. (B2)
3. **Fix the upgrade path explicitly.** Either:
   - make `oflm pull` replace files whose hash differs when the model's layout is refused
     (the DiT's `weights.bin` is unchanged, so an upgrade is ~70 MB plus small files); or
   - document "remove, then pull" as the upgrade step.
   Delete "refused until re-pulled". (B2)
4. **Harden reference decoding.**
   - `STBI_ONLY_PNG` + `STBI_ONLY_JPEG`.
   - `stbi_info` first; refuse above a pixel cap (e.g. 64 MP) and below 64 px a side.
   - A per-part byte cap for edits well under the 256 MB body limit.
   - WebP and anything else → a 400 naming the format as not implemented.
   (B3)
5. **Write OPEN-DIFFUSION-REFERENCE's acceptance criteria now.** Proposal:
   - A 512² PNG at R = 512 passes through byte-identical.
   - A 4000×3000 JPEG gives the centred 3000² crop box, resized to R.
   - It lands within ±N levels per channel (max) of PIL's `ImageOps.fit` / LANCZOS golden on
     the same crop. Pick a Lanczos-3 custom filter in stb_image_resize2 to get close.
   - RGBA drops alpha (as PIL's `convert("RGB")` does); grayscale and palette images become
     RGB.
   - EXIF orientation 6 rotates.
   - Too-small, too-large and undecodable inputs are refused, naming the reason.
   (B4, S1)
6. **Apply EXIF orientation** for JPEG (tag 0x0112, a ~40-line parse), and cover it in #5. (S1)
7. **Feed the quality gates diffusers' preprocessed pixels.**
   - 8.0 saves the post-crop RGB8 reference.
   - `generate.py --study` and the chain tests take it directly, so the EDIT and ENCODER
     gates measure NPU drift and not resize drift.
   - REFERENCE's tolerance covers the resize separately.
8. **Add spike 8.1.5, the long-L streams:**
   - build and run the standalone `dit_fa` test at L = 2560 and 8704 (24 heads), with
     accuracy against `fa_emul.py`;
   - `dit_gemm` `sgl_in` / `sgl_out` at M = 8704;
   - `dit_ew` views at T = 8704.
   Do it before 8.2. (A5)
9. **Downgrade spike 2** to a verification task. Implement the reference branch as image rope
   + `rope_axis(0, FINE[10])`, and derive `n_gen` / the reference `grid_w` from the existing
   `grid_w` for same-size configurations. (A4)
10. **Try the 4-phase residual-add route first in spike 1**, before a new `s2d` op. (Alt 1)
11. **Add an ablation criterion to OPEN-DIFFUSION-EDIT:** LPIPS(NPU edit, diffusers edit)
    must be well below LPIPS(NPU edit with a different reference, or with no reference, same
    seed, diffusers edit). This catches a DiT that ignores the reference. (A6)
12. **Add 3-4 CC0 natural photos to the edit study:** one 4:3 JPEG with an EXIF rotation, one
    portrait, one with text. The ENCODER gate runs on them too. (A7)
13. **Rewrite decision 2's rationale.**
    - Name `dit_fa`'s `valid_len` and the encoder as the real per-shape cost.
    - Record aspect buckets as the growth path.
    - Put the square-output consequence and the upscaling of small inputs in the spec. The
      CLI prints "reference centre-cropped W×H → S², resized to R". (A1, A2)
14. **Make the output size follow the reference by default:** the largest configuration ≤ the
    reference's short side, minimum 512, with `--size` overriding. The server's `size: auto`
    (OpenAI's edit default) maps the same way. (A3)
15. **Close the spec-impact gaps:**
    - acceptance criteria for the RESOLUTIONS change (which `(R, R_ref)` pairs pass, and the
      named refusal for others);
    - SERVER-IMAGES-EDITS's criterion at `specs/server-api/spec.md:368` flips from 501 to a
      400 for two images, so state the new criteria;
    - OPEN-DIFFUSION-STEPS: either edits accept `steps` (the reference-row x_emb arguments
      have stride 0 across steps, which the engine's "steps lie on one line" check must
      accept) or edits refuse `steps ≠ 4`, named.
16. **Bound the NPU-ONLY amendment:** "once per request, before the first NPU dispatch, under
    the pixel cap". Measure a 12 MP JPEG's decode + resize and put it in the host-CPU figure.
    (S2)
17. **State the resident-memory total** for an engine holding all four configurations, and
    either confirm the budget or evict the least recently used configuration's activations.
    (S3)
18. **Settle the base branch:** un-park #141 or rebase `feat/edits` onto #140. (S4)
19. **Build as a vertical slice:** 512e512 end to end through 8.5 first, then 1024e1024, with
    one publish at the end. (Alt 3)
20. **Link the 2026-09-30 research notes,** or inline what the plan depends on. (Unclear 1)

---

## Your Decisions

Review the recommendations above and mark your decisions below:

### Accepted Recommendations
- [x] #1 (modified): Keep `--image <file>` for edits; rename serve's `--image 1` (proposed `--imagegen 1`) in #137
- [ ] #2: Publish to a pinned HF revision, not over `main`; rewrite decision 6
- [ ] #3: Fix the upgrade path (pull replaces hash-mismatched files, or document remove + pull)
- [ ] #4: Harden decoding (PNG/JPEG only, pixel cap, byte cap, named 400 for other formats)
- [ ] #5: Acceptance criteria for OPEN-DIFFUSION-REFERENCE (tolerance vs PIL, alpha, EXIF, refusals)
- [ ] #6: Apply EXIF orientation
- [ ] #7: Gates consume diffusers' preprocessed RGB8 (separate resize drift from NPU drift)
- [ ] #8: New spike: dit_fa / dit_gemm / dit_ew at L = 8704 and 2560
- [ ] #9: Downgrade the qk RoPE spike; use FINE[10] and derive n_gen / ref grid_w
- [ ] #10: Spike 1 tries the 4-phase residual-add route before an s2d op
- [ ] #11: Reference-ablation criterion in OPEN-DIFFUSION-EDIT
- [ ] #12: Add CC0 natural photos to the edit study
- [ ] #13: Rewrite decision 2's rationale (valid_len, encoder); document square output + upscale
- [ ] #14: Output size follows the reference by default; server `size: auto` likewise
- [ ] #15: Spec gaps: RESOLUTIONS criteria, SERVER-IMAGES-EDITS criteria, STEPS for edits
- [ ] #16: Bound the NPU-ONLY amendment; measure the 12 MP JPEG host cost
- [ ] #17: State resident memory across all configurations; LRU eviction if needed
- [ ] #18: Settle the base branch (#141 un-parked or rebase onto #140)
- [ ] #19: Vertical slice: 512e512 end to end first, one publish
- [ ] #20: Link the research notes

### Rejected Recommendations
- [ ] #X: [Brief description] - Reason: [your reason]

### Custom Instructions
[Add any additional changes or instructions not covered by the recommendations above]

### Notes
[Any other thoughts or considerations]
