---
name: q4nx-streaming-pack
description: Convert GGUF models larger than system RAM with q4nx-build's --stream / StreamingSafetensorsWriter path, and prune very large MoE models with --prune-moe-ffn / --prune-experts. Use when a pack OOMs mid-conversion, when peak RSS exceeds RAM on an 80GB+ GGUF, when a MoE model must be narrowed to fit, or when the catalogue refuses a pruned expert width ("UNVALIDATED point allowed by OPEN_KERNELS_UNVALIDATED").
---

# Streaming and pruned packs (q4nx-build)

The default pack path accumulates every converted tensor in `conv.q4nx_tensors`
and hands the dict to `safetensors.torch.save_file` at the end, so peak RSS is
roughly the size of the whole packed model. That is fine for a 27 GB GGUF and
impossible for an 80 GB+ one. `--stream` replaces the sink: each tensor is
written to disk the moment it is produced, and its source pages are advised
away immediately after.

Reachable two ways (`convert.py` is a thin wrapper over `q4nx.cli:main`, so the
installed `q4nx-build` console script takes the same flags):

```
cd utilities/q4nx-build
python convert.py -i <big.gguf> -o <out-dir> -s <hf-repo> -t language --stream
```

`--stream` is **not required** for a big model: `cli.py` enables it
automatically when the source is >= 80% of `MemAvailable`. The flag forces it on
for a smaller source on a busy machine. It logs which path it took:

```
[INFO] Streaming export enabled (source 214.3 GB, available RAM 256.0 GB)
```

## How the writer works, and its one hard limit

`q4nx/streaming_exporter.py` emits a valid safetensors file incrementally.
safetensors wants `{u64 length, JSON header, raw bytes}` up front, but the JSON
needs every tensor's offset, which is only known after the last tensor is
written. So the writer reserves a fixed header region up front and patches the
real JSON into it on `close()`. The padding is JSON whitespace, so any
conforming reader accepts it, and `__metadata__` is omitted to match
`save_file(..., metadata=None)`.

`StreamingPack` is the dict-shaped sink: `pack[name] = tensor` becomes
`add_tensor` plus a key-count bookkeeping entry whose value is the name itself.
`len()`, `in` and iteration keep working, which is what the rest of the CLI
depends on (`_layer_count` and the speculative report iterate keys only).

**The limit: `header_reserved` defaults to 4 MiB and nothing raises it.** The
construction site is `models/qwen35moe.py`, which passes no override, and there
is no CLI flag. A pack with enough tensors to overflow 4 MiB of JSON hard-fails
at `close()` with "re-run with a larger reserved header" -- and there is no way
to actually do that re-run. A header entry costs ~80-150 bytes, so 4 MiB is on
the order of 30-50k tensors -- noise at any realistic model size, but it is a
cliff, not a slope. If a future
model trips it, the fix is to plumb a `--stream-header-bytes` flag through
`cli.py` into the `StreamingSafetensorsWriter(...)` call.

## `drop_pages` is advisory, not a guarantee

`q4nx/memfree.py` calls `madvise(MADV_DONTNEED)` on each source tensor's mapped
bytes after it converts, the same eviction Guanaco's herdcache does. It is safe
because the conversion is a one-pass read-mostly scan: once a tensor is
converted its source bytes are never touched again. It is still only advice --
the kernel may decline, and it returns 0 when it does or when libc is
unavailable. Peak RSS therefore tracks the *page cache*, not a hard bound, so
on a machine where the source is genuinely larger than RAM, budget for the
source staying in cache and being reclaimed lazily.

## MoE pruning for very large models

Two imatrix-driven flags, both on the MoE converter only:

- **`--prune-experts K`** — per-layer K most-dispatched experts, ranked by the
  imatrix's calibration `counts` (the signal Guanaco's HerdCache prior pins).
  Gathers the expert axis of `ffn_{gate,up,down}_exps` and the router input
  axis. `config.json num_experts` is narrowed to match.
- **`--prune-moe-ffn K`** — per-layer top-K neurons of the dense-FFN axis by
  imatrix activation energy on `ffn_down_exps` (in_sum2 summed over the expert
  axis), gathered consistently across every expert of the layer. `config.json`
  `moe_intermediate_size` / `intermediate_size` are updated so `--build-spec`
  derives a compatible spec.

Both need an imatrix (`--imatrix`, or the sidecar, or `OFLM_IMATRIX`), which is
recorded in the pack because a pruned container is not reproducible without it.
`--prune-ffn` refuses on a MoE arch with a pointer to these two; conversely both
refuse on the HF-safetensors path, which has no imatrix to rank with.

**The imatrix must describe the same expert set as the GGUF.** `_resolve_moe_prune`
checks the imatrix's expert count against the GGUF tensor and refuses otherwise.
Without that check, a "reaped" 288-expert GGUF paired with a 512-entry imatrix
applies indices into the wrong axis entirely -- a real mismatch the bundled
imatrix has on `qwen3.8-flash-next-reap-288`. If you see that error, regenerate
the imatrix; do not work around it.

**`--prune-moe-ffn K` snaps K to a *packable* width.** The qwen36moe recipe
requires the expert width to be a multiple of 128 whose 128-row stripe count
divides the 8 columns, i.e. K in {128, 256, 512, 1024}. An off-list K would
produce a container the recipe's own checks later refuse, so it is snapped to
the nearest valid width with a message; Ctrl-C at that point to keep the exact
(unbuildable) width.

**Packable is not the same as validated.** The snap rule above is the recipe's
structural stripe law. On top of it, the kernel *catalogue*
(`catalogue.py`'s `Template("moe", ...)`, currently `ff: values(512)`) records
which widths have actually been built and fixture-tested. So 128/256/1024 are
packable but off-catalogue, and 512 is the one validated point today. Two
separate messages report these, do not conflate them:

- the snap `[INFO]` fires for structural un-packability (fixed before the pack),
- `_catalogue_prune_check`'s `[INFO]` fires for a packable-but-unvalidated width.

**The shared expert is pruned too, identically.** `_prune_shexp` narrows
`ffn_{gate,up,down}_shexp` to the per-layer `moe_ffn` index set. This is not
optional polish: the shared expert rides one call-site path in the kernels, so a
container that narrows only the routed experts is structurally refused at load.
`config.json shared_expert_intermediate_size` is narrowed to match.

## The catalogue wall is deliberate, and the message says which flag

`--prune-moe-ffn` ends with `_catalogue_prune_check`, an **info-only** probe: if
the pruned width is outside the validated kernel catalogue it names the set and
surfaces it *before* a 20 GB build instead of after.

`open_kernels/export_qwen36_kernels.py` refuses an off-catalogue spec with an
`OpRangeError` and an explicit deliberate opt-in. The refined message
distinguishes the two cases, and the distinction is real and verifiable:

```
OPEN_KERNELS_UNVALIDATED=1 python open_kernels/export_qwen36_kernels.py --model-dir <dir>
```

- flag **not** set -> the point is off-grid but bypassable; the flag is the way
  through, at your own risk of a build that fails or produces an untested kernel.
- flag **already** set and still refused -> the rule is a recipe *structural*
  law or a missing mapping, which the flag does not soften. Do not keep
  re-running with it; read the rule in the message.

The flag is honoured in exactly one place, `catalogue.py`'s `Param._refuse`.
Structural rules deliberately bypass it by raising `OpRangeError` directly --
see `qwen36moe.py`'s `per_band` ("the refusal is here rather than in a catalogue
entry so OPEN_KERNELS_UNVALIDATED cannot soften it"). Its own docstring names
the case that proved it: **K = 2944**, the width GPT-OSS's container ships, is 23
band chunks, an odd count the rs=2 band law cannot consume, so it is refused
identically with the flag set or unset. Verified:

```
flag=None: per_band(2944) refused -> a 2944-wide band is 23 chunks, an odd count...
flag='1':  per_band(2944) refused -> a 2944-wide band is 23 chunks, an odd count...
```

## Carry-through, and the 35 GB PLE table

Unmapped tensors no longer warn-and-drop. `_carry_through` puts 2D weights that
quantize cleanly through the normal Q8NX pack and travels everything else as
bf16 under its GGUF name, so a speculative pack loses no bytes. Two consequences
worth knowing:

- `per_layer_token_embd.weight` is explicitly **skipped** and reported: it is an
  n-gram-indexed `[160, ~320M]` hash store, 35 GB as Q5_0, neither dequantizable
  nor a GEMM weight.
- The speculative report at the end of the pack now drops config entries that a
  converter fallback actually supplied (tied `lm_head`, synthesized vision
  weights) from the "missing" list, so it only names tensors the runtime will
  really lack.

## Verify before shipping a container

Do not trust the conversion log alone -- a converter bug passes the slice, the
engine's bit-identity and a prompt compare, and the model still answers in
fragments. Check the container against HF first
(`open_kernels/model/container_vs_hf.py`); see `open-qwen38-27b-kernels` for
that workflow and the value-head scramble it catches.

Then, for a pruned MoE pack specifically: confirm `num_experts`,
`moe_intermediate_size` **and** `shared_expert_intermediate_size` in the output
`config.json` are all consistent with the packed tensors, and that the expert
width you ended at is one the kernel set was built for.

## Rules

- `--stream` must keep `model.q4nx` a file any conforming safetensors reader
  accepts. When touching the writer, diff its output against
  `safetensors.torch.save_file` for the same dict (bf16, f64, i32 and a
  zero-length tensor all matter) and confirm the sizes differ only by the
  reserved header.
- The docstring in `streaming_exporter.py` says `torch.save_file`; the engine
  actually calls `safetensors.torch.save_file` (`model_converter.py`). The
  reference is wrong, not the code -- do not "fix" the writer to match it.
- `ironvenv` carries a system torch override (`2.14.1+rocm7.2`) that
  `transformers`/`accelerate` must not perturb. It is not in
  `ironvenv-requirements.txt`, which pins the IRON/Peano toolchain only. After
  installing anything into that venv, check `python -c "import torch;
  print(torch.__version__)"` still reports the ROCm build.
