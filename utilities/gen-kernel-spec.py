#!/usr/bin/env python3
"""Generate a candidate kernel spec for a HuggingFace model.

Usage:
    python utilities/gen-kernel-spec.py --repo Qwen/Qwen3.5-4B --out open_kernels/recipes/specs/qwen35-4b.json

Fetches config.json (and README frontmatter for base_model), derives dims and
family via the same heuristics as oflm-add/q4nx-build, and emits a spec JSON
that still needs manual routing/heuristic review before building.

A repo that ships only GGUF weights has no config.json: the GGUF header names
its own geometry, so `--repo` falls back to it (the same file `oflm add`
installs). `--gguf <file>` pins a local file explicitly.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS_DIR = ROOT / "open_kernels" / "recipes" / "specs"

# Reuse oflm-add alias tables if available.
sys.path.insert(0, str(ROOT / "utilities" / "oflm-add"))
try:
    from oflm_add import CONFIG_MODEL_FAMILIES, FAMILY_ALIASES, kernel_set_name  # type: ignore
except Exception:
    CONFIG_MODEL_FAMILIES = {}
    FAMILY_ALIASES = []

    def kernel_set_name(name: str) -> str:
        """Fallback when oflm-add is not beside us: strip the upload suffix."""
        n = name or ""
        while True:
            stripped = re.sub(r"[-_.](gguf|q4nx|i1)$", "", n, flags=re.IGNORECASE)
            if stripped == n:
                break
            n = stripped
        return (n if n.lower().endswith("-npu2") else f"{n}-NPU2") if n else ""

# The real spec machinery: `ModelSpec.from_hf_config` knows the family builders
# (layer types off full_attn_idxs / layer_types, LFM2's FF adjustment, llama3's
# rope law), and `kernel_hash()` is the key the engine matches installed kernel
# sets by -- so BOTH the existing-spec match and the generated candidate must
# go through it. The hand-rolled family+hidden match linked Llama-3.2-1B to
# minicpm5-2b (same family+hidden, HEAD DIM 64 vs 128 -- different kernels).
sys.path.insert(0, str(ROOT / "open_kernels"))
from recipes.spec import ModelSpec, SpecError, GGUF_FAMILIES  # noqa: E402
from recipes.model_patches import apply_to_dict  # noqa: E402


def _repo_cache(repo: str) -> str:
    """A per-repo download directory.

    hf_hub_download(local_dir=...) writes <local_dir>/<filename>, so ONE shared
    directory means every repo's config.json / tokenizer.json lands on the same
    path and the second repo reads the first one's file: the probe for
    Llama-3.1-70B got Gemma 3 27B's config ("config.json lacks 'vocab_size'"),
    and a spec could be generated from another model's geometry without a word.
    """
    return f"/tmp/hf_cfg/{repo.replace('/', '__')}"


def _hf_token():
    """Forward HF credentials to huggingface_hub (gated bases: meta-llama/*).

    huggingface_hub reads HF_TOKEN from the environment itself, but passing it
    explicitly makes the contract visible and survives a library that stops
    doing so. Never sent to ModelScope.
    """
    return (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None)


def fetch_config(repo: str) -> dict:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id=repo, filename="config.json",
                        local_dir=_repo_cache(repo), token=_hf_token())
    return json.loads(Path(p).read_text())


def fetch_real_vocab(repo: str) -> int | None:
    """The tokenizer's id count for this repo, or None.

    `real_vocab` is the head's masking bound: the kernel is built for `vocab`
    rows (a tile multiple) and the engine masks the logits past the tokenizer's
    real count. Without it a spec says real == vocab, which leaves the padding
    rows in the argmax -- a garbage token can win. The hand-written specs carry
    it (qwen3-4b/8b 151669, qwen35 248070); a GENERATED spec used to lose it,
    so every size added through this script shipped unmasked logits.
    """
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(repo_id=repo, filename="tokenizer.json",
                            local_dir=_repo_cache(repo), token=_hf_token())
    except Exception as exc:                      # noqa: BLE001 - a gated repo may hide it
        print(f"warn: could not fetch tokenizer.json ({exc}); real_vocab = vocab")
        return None
    sys.path.insert(0, str(ROOT / "open_kernels"))
    from recipes.load import tokenizer_vocab
    return tokenizer_vocab(Path(p))


def text_config(cfg: dict) -> dict:
    """The language model's config for a MULTIMODAL checkpoint.

    Gemma 3 / Medgemma / Translategemma configs wrap the text model
    (`text_config`), and the density/vocab/rope knobs live there -- the outer
    dict has the vision tower's. Reading the outer one gives "config.json lacks
    'vocab_size'", which is how a 27B Gemma looked like a broken config.
    """
    for key in ("text_config", "llm_config", "language_config"):
        inner = cfg.get(key)
        if isinstance(inner, dict) and inner.get("hidden_size"):
            merged = dict(inner)
            merged.setdefault("model_type", cfg.get("model_type"))
            return merged
    return cfg


def spec_config(repo: str) -> tuple[dict, int | None]:
    """(the config the recipes want, the tokenizer's real id count).

    Three repairs, all of them things a real repository does: the text model is
    nested inside a multimodal config (`text_config`), Gemma 3's text config
    omits `vocab_size` (the tokenizer defines it), and `real_vocab` -- the
    logits mask -- comes from tokenizer.json, not from the config at all.
    """
    cfg = text_config(fetch_config(repo))
    real_vocab = fetch_real_vocab(repo)
    if not cfg.get("vocab_size") and real_vocab:
        cfg["vocab_size"] = real_vocab
    return cfg, real_vocab


# Some model types are registry families whose kernels are a DIFFERENT recipe's
# (Nanbeige's GGUF/Config says `nanbeige`, and the shipped Nanbeige4.1 spec is a
# llama3). Keep that mapping here, next to the spec derivation, rather than
# adding a recipe alias for a name that is not an architecture.
RECIPE_ALIASES = {"nanbeige": ("llama", "llama3"), "nanbeige4": ("llama", "llama3")}


def recipe_model_type(mt: str) -> str:
    """The model_type the family's ModelSpec builder reads, not the card's."""
    return RECIPE_ALIASES.get(mt.strip().lower(), (mt,))[0]


def family_from_config(cfg: dict) -> str | None:
    mt = cfg.get("model_type")
    if isinstance(mt, str):
        fam = CONFIG_MODEL_FAMILIES.get(mt.strip().lower())
        if fam:
            low = fam.strip().lower()
            if low in RECIPE_ALIASES:
                return RECIPE_ALIASES[low][1]
            return fam
    for arch in (cfg.get("architectures") or []):
        low = arch.lower()
        for prefix, fam in FAMILY_ALIASES:
            if prefix.lower() in low:
                return fam
    return None


def repo_gguf_url(repo: str) -> str | None:
    """A resolve URL for the repo's GGUF, when it ships no config.json.

    A GGUF-only upload is the common shape of a quantized repository, and it
    names its own geometry -- there is nothing in config.json to read. Pick the
    file `oflm add` would install (the same preference order), and fall back to
    any single-file GGUF when every quant is one the kernels do not stream: for
    a SPEC, only the header matters, not whether the quant is servable.
    """
    from oflm_add import choose_gguf_file, hf_file_tree, root_file_names
    try:
        tree = hf_file_tree(repo)
    except Exception as exc:                      # noqa: BLE001 - gated / offline
        print(f"warn: could not list {repo}: {exc}", file=sys.stderr)
        return None
    names = root_file_names(tree)
    chosen, _ = choose_gguf_file(names)
    if not chosen:
        single = [n for n in names if n.endswith(".gguf")
                  and not re.search(r"-\d+-of-\d+\.gguf$", n)]
        chosen = single[0] if single else None
    if not chosen:
        return None
    return f"https://huggingface.co/{repo}/resolve/main/{chosen}"


def spec_config_via_base_chain(repo: str):
    """(base, cfg, real_vocab) from the repo's README base_model chain.

    A quantized repo holds only the GGUF: no config.json, no tokenizer. Its model
    card names the checkpoint it quantizes, and that checkpoint's repo does carry
    both. `oflm add` already climbs that chain (finetune -> base -> base, closest
    first, cycles/depth bounded); do the same here before giving up on config.json
    and deriving from the GGUF header -- the GGUF names the geometry, but an HF
    config is what the family builders (LFM2's FF/rope law, layer types, ...) read.
    """
    try:
        from oflm_add import readme_base_chain
        bases = readme_base_chain(repo)
    except Exception as exc:                      # noqa: BLE001 - offline / no README
        print(f"warn: could not read base_model chain for {repo}: {exc}", file=sys.stderr)
        return None, None, None
    for base in bases:
        try:
            cfg = text_config(fetch_config(base))
        except Exception:
            continue
        real_vocab = fetch_real_vocab(base)
        if not cfg.get("vocab_size") and real_vocab:
            cfg["vocab_size"] = real_vocab
        print(f"config.json not in {repo}; using base model {base}'s config.json")
        return base, cfg, real_vocab
    return None, None, None


def base_model_from_readme(repo: str) -> str | None:
    try:
        from huggingface_hub import hf_hub_download
        # Per-repo cache: one shared /tmp/hf_cfg made every repo's README land on
        # the same path, so a second repo could read the first one's base_model.
        p = hf_hub_download(repo_id=repo, filename="README.md",
                            local_dir=_repo_cache(repo), token=_hf_token())
        text = Path(p).read_text(errors="ignore")
    except Exception:
        return None
    if not text.startswith("---"):
        return None
    end = text.find("---", 3)
    if end == -1:
        return None
    fm = text[3:end]
    lines = fm.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("base_model:"):
            rest = lines[i].split(":", 1)[1].strip()
            if rest and not rest.startswith("[") and not rest.startswith("-"):
                val = rest.strip().strip('"').strip("'")
                if val and val.lower() not in ("null", "none", ""):
                    return val
            # list form
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("-"):
                val = lines[j].strip().lstrip("-").strip().strip('"').strip("'")
                if val and val.lower() not in ("null", "none", ""):
                    return val
                j += 1
        i += 1
    return None


def generic_model_name(repo: str, gguf_source: str = "") -> str:
    """The export/spec label for a repo, from the BASE model it quantizes.

    A GGUF upload's name is an author's slug -- `Llama-3.2-1B-Instruct-GGUF`,
    `Qwen3-8B-GGUF` -- carrying quant/upload suffixes that are not properties of
    the model. The kernels, and `oflm add`'s geometry match, are named after the
    generic base (`Llama-3.2-1B-Instruct`, `Qwen3-8B`). Climb the repo's README
    ``base_model:`` chain first (finetune -> base -> base), then fall back to the
    repo slug with those suffixes stripped by ``kernel_set_name``.

    The returned value is the EXPORT name (with ``-NPU2``). The spec filename
    stem is ``spec_stem()`` below: a second, lower-cased spelling of the same
    model for the repository's on-disk convention.
    """
    base = ""
    if repo:
        base = base_model_from_readme(repo)
        if not base:
            try:
                from oflm_add import readme_base_chain
                chain = readme_base_chain(repo) or []
                if chain:
                    base = chain[-1]
            except Exception:
                pass
    if not base and gguf_source:
        # A local GGUF names itself in general.name; use it before any slug.
        try:
            from oflm_add import read_gguf_inventory
            inv = read_gguf_inventory(gguf_source, limit_mb=96)
            if inv and inv[0].get("general.name"):
                base = str(inv[0]["general.name"]).strip()
        except Exception:
            pass
    if base and "/" in base:
        base = base.split("/")[-1]          # meta-llama/Llama-3.2-1B-Instruct
    fallback = (repo.split("/")[-1] if repo else
                Path(gguf_source).stem if gguf_source else "model")
    return kernel_set_name(base or fallback)


def spec_stem(spec: dict) -> str:
    """The spec file stem for a derived spec: its export name, lower-cased.

    Kept as a separate function so the filename and ``extra.model`` can never
    drift apart (the #54 regression: a 12B spec shipped as "Gemma3-4B-NPU2").
    The export name is the generic base model, so the stem is too:
    ``Llama-3.2-1B-Instruct-NPU2`` -> ``llama-3.2-1b-instruct-npu2``.
    """
    name = (spec.get("extra") or {}).get("model") or spec.get("family") or "model"
    return str(name).lower()


def existing_gguf_spec(spec: dict) -> str | None:
    """An existing spec whose KERNELS cover this GGUF-derived geometry.

    The repo's slug is irrelevant here: a quant of a model whose geometry is
    already shipped must LINK to that spec, not grow a second repo-named one.
    ``kernel_hash`` is the same key the engine uses to match installed sets, so
    a false link (same hidden, different head dim -- Llama-3.2-1B vs minicpm5-2b)
    is refused by the hash, not by a name heuristic.
    """
    try:
        cand = ModelSpec.from_dict(spec)
    except Exception:
        return None
    for sp in sorted(SPECS_DIR.glob("*.json")):
        try:
            d = json.loads(sp.read_text(encoding="utf-8"))
            if d.get("family") != cand.family:
                continue
            if ModelSpec.from_dict(d).kernel_hash() == cand.kernel_hash():
                return sp.stem
        except Exception:
            continue
    return None


def existing_spec_for_family_size(family: str, cfg: dict) -> str | None:
    """The spec whose KERNELS cover this config, by the same key the engine
    matches installed sets with (ModelSpec.kernel_hash: family, hidden, FF,
    heads, head_dim/rotary, rope law, eps, vocab, quant -- NOT the depth).

    A false link would ship a spec whose kernels the model cannot run (the
    Llama-3.2-1B case again: same family+hidden as minicpm5-2b, head_dim 64
    vs 128), so the rule is kernel-hash equality or nothing.
    """
    try:
        cand = ModelSpec.from_hf_config(cfg)
    except SpecError as e:
        print(f"(no spec match possible: {e})")
        return None
    for sp in SPECS_DIR.glob("*.json"):
        try:
            d = json.loads(sp.read_text())
        except Exception:
            continue
        if d.get("family") != cand.family:
            continue
        try:
            existing = ModelSpec.from_dict(d)
        except Exception:
            continue
        if existing.kernel_hash() == cand.kernel_hash():
            return sp.name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="")
    ap.add_argument("--gguf", default="",
                    help="derive the spec from a GGUF file's own metadata instead of the "
                         "repo's config.json (a gated or absent HF config: the llama3 8B "
                         "repos are gated, and a GGUF always names its geometry)")
    ap.add_argument("--model-name", default="",
                    help="extra.model for a --gguf spec (default: from --spec-name/--repo)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--refresh", action="store_true",
                    help="regenerate an existing spec in place (keeps extra.model) instead "
                         "of writing a link file when another spec covers the geometry")
    args = ap.parse_args()

    cfg = real_vocab = None
    base_from_chain = None
    gguf_url = None
    gguf_spec = None
    gguf_error = None

    def write_gguf_spec(spec, source="the GGUF metadata"):
        # The geometry comes from the GGUF; the NAME comes from the base model
        # the repo quantizes, never from the repo's upload slug. A second repo
        # of the same base therefore produces the same spec, and the geometry
        # hash below links it instead of duplicating it.
        #
        # Patch FIRST: the base-model patch supplies rope_scaling (baked into a
        # Llama-3 GGUF's rope_freqs.weight, absent from its header) and the
        # real tokenizer vocab. This changes kernel_hash -- which is the whole
        # point: a scaled Llama-3.2 GGUF must hash-equal the shipped spec so
        # it LINKS, rather than getting a second unbuildable spec.
        patch_key_name = generic_model_name(args.repo, args.gguf)
        before = dict(spec)
        apply_to_dict(spec, patch_key_name)
        if spec != before:
            print(f"applied base-model patch for {patch_key_name}")
        existing = existing_gguf_spec(spec)
        if existing and not args.refresh:
            print(f"Existing spec covers this geometry: {existing} "
                  f"({source}) -- linking instead of generating.")
            out = Path(args.out) if args.out else SPECS_DIR / f"{spec_stem(spec)}.json"
            if out.is_file():
                print(f"{out} already exists -- keeping it")
                return 0
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"link_to": existing,
                                       "base_model": patch_key_name}, indent=2))
            print(f"Wrote link file to {out}")
            return 0
        model_name = args.model_name or patch_key_name
        spec["extra"] = {**(spec.get("extra") or {}), "model": model_name}
        out = Path(args.out) if args.out else SPECS_DIR / f"{spec_stem(spec)}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        print(f"Wrote candidate spec to {out} (from {source})")
        return 0

    def gguf_spec_is_complete(spec: dict) -> bool:
        """Whether the GGUF header (plus its base-model patch) carries the fields
        this family's kernels need.

        A GGUF names dimensions and vocab, but NOT every config field the
        kernels are built from -- Llama-3.2's rope_scaling (factor 32, the
        low/high frequency ramp) is in config.json, not the header. The base-
        model patch supplies such fields by generic base name; only when the
        patch has no value do we fall through to the base config. The GGUF is
        still the authority for size and architecture; this is only about the
        fields it cannot carry.
        """
        patch = None
        try:
            from recipes.model_patches import base_patch
            patch = base_patch(generic_model_name(args.repo, args.gguf))
        except Exception:
            pass
        missing = [f for f in ("layer_types",) if f not in spec]
        # rope_scaling is present-but-None when the GGUF header has no scaling
        # metadata; Llama-3.2's kernels are built WITH the factor-32 ramp. A
        # patch entry with that field fills it, so the spec is complete.
        if spec.get("rope_scaling") is None and not (patch and patch.get("rope_scaling") is not None):
            missing.append("rope_scaling")
        if missing:
            print(f"GGUF header lacks {', '.join(missing)} and the base-model patch has no "
                  "value; using the base model's config for those fields "
                  "(the GGUF remains authoritative for size/architecture)")
            return False
        return True

    if args.gguf:
        # A GGUF names its own geometry: architecture, dimensions, vocab, rope. A gated
        # base repo (meta-llama/*) leaves oflm add with a stub config and leaves this
        # script with nothing to read, and the file is the authority anyway.
        sys.path.insert(0, str(ROOT / "utilities" / "oflm-add"))
        from oflm_add import read_gguf_inventory
        inv = read_gguf_inventory(args.gguf, limit_mb=96)
        if not inv:
            print(f"ERROR: could not read {args.gguf}", file=sys.stderr)
            return 1
        try:
            spec = ModelSpec.from_gguf_metadata(inv[0]).to_dict()
        except SpecError as exc:
            print(f"ERROR: the GGUF header cannot derive a spec ({exc})", file=sys.stderr)
            return 1
        return write_gguf_spec(spec)

    if args.repo:
        # THE FILE FIRST, the same rule `oflm add` uses. A quantized repo's config
        # is a hint for what the GGUF does not carry, never the source of geometry:
        # a base config can be a stub (Llama3-2B-Base claims hidden 4096 / 6 layers),
        # or describe a different family entirely (Nanbeige's card says nanbeige,
        # the GGUF says llama). Try the header first; use config only when the
        # header has no recipe or lacks a key the family builder needs.
        url = repo_gguf_url(args.repo)
        if url:
            sys.path.insert(0, str(ROOT / "utilities" / "oflm-add"))
            from oflm_add import read_gguf_inventory
            try:
                inv = read_gguf_inventory(url, limit_mb=96)
                if inv:
                    gguf_spec = ModelSpec.from_gguf_metadata(inv[0]).to_dict()
            except SpecError as exc:
                gguf_error = exc
        if gguf_spec is not None and gguf_spec_is_complete(gguf_spec):
            return write_gguf_spec(gguf_spec)
        try:
            cfg, real_vocab = spec_config(args.repo)
        except Exception as exc:                  # noqa: BLE001 - 404 on config.json
            base, base_cfg, base_rv = spec_config_via_base_chain(args.repo)
            if base_cfg is not None:
                base_from_chain, cfg, real_vocab = base, base_cfg, base_rv
                if gguf_error is not None:
                    print(f"GGUF header lacks a needed key ({gguf_error}); "
                          f"using base model {base}'s config.json")
            else:
                detail = (str(gguf_error) if gguf_error is not None
                          else "no config.json at the repo root")
                print(f"ERROR: cannot derive a spec for {args.repo}: {detail}; "
                      "no base config and no usable GGUF header",
                      file=sys.stderr)
                return 1
    if cfg is None:
        cfg, real_vocab = spec_config(args.repo)
    family = family_from_config(cfg)
    if not family:
        print(f"ERROR: could not derive family from config for {args.repo}", file=sys.stderr)
        sys.exit(1)

    bm = base_from_chain or base_model_from_readme(args.repo)
    if bm:
        print(f"base_model detected: {bm}")
        if base_from_chain is None:
            # The config came from this repo; the base is a family hint only.
            # When the chain already supplied cfg (and its real_vocab), do not
            # refetch/overwrite it with a raw config that lost the vocab repair.
            try:
                base_cfg = text_config(fetch_config(bm))
                family = family_from_config(base_cfg) or family
                cfg = base_cfg
            except Exception as e:
                print(f"warn: could not fetch base config: {e}")

    out_path = Path(args.out) if args.out else SPECS_DIR / f"{family}-candidate.json"
    kept_model = ""
    if args.refresh and out_path.is_file():
        # Refresh an EXISTING spec in place (a tokenizer or config changed): keep the
        # export name it already has, and skip the link shortcut -- the caller asked
        # for this spec, not for what covers its geometry.
        try:
            kept_model = json.loads(out_path.read_text()).get("extra", {}).get("model", "")
        except ValueError:
            kept_model = ""
        existing = None
    else:
        existing = existing_spec_for_family_size(family, cfg)
    if existing:
        print(f"Existing spec covers this size: {existing} — linking instead of generating.")
        out = Path(args.out) if args.out else Path(f"/tmp/{family}-link.json")
        if out.is_file():
            # never clobber a real spec with a link pointer
            print(f"{out} already exists — keeping it")
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"link_to": existing, "base_model": bm}, indent=2))
        print(f"Wrote link file to {out}")
        return

    # A real ModelSpec from the family builder when the config's model_type has
    # one (layer types off full_attn_idxs / layer_types, the family's FF and
    # rope law); the template derivation stays for unknown model types.
    cand = None
    builder_cfg = dict(cfg)
    if isinstance(builder_cfg.get("model_type"), str):
        builder_cfg["model_type"] = recipe_model_type(builder_cfg["model_type"])
    try:
        cand = ModelSpec.from_hf_config(builder_cfg, real_vocab=real_vocab)
    except SpecError as e:
        print(f"(the family builder refuses this config: {e}; using the derivation)")

    # Use an existing spec of the same family as a template for schema/routing.
    template = None
    for sp in SPECS_DIR.glob("*.json"):
        try:
            d = json.loads(sp.read_text())
        except Exception:
            continue
        if d.get("family") == family:
            template = d
            break
    if template:
        spec = json.loads(json.dumps(template))
    else:
        spec = {"family": family}
    spec["family"] = family
    spec["hidden"] = cfg.get("hidden_size", spec.get("hidden"))
    spec["num_layers"] = cfg.get("num_hidden_layers", spec.get("num_layers"))
    # The derivation for an unknown model_type starts from a template of the
    # recipe family (or a bare dict). A ModelSpec always needs per-layer types,
    # so fill them in -- otherwise load_spec dies with a KeyError on a spec this
    # script itself wrote.
    if not spec.get("layer_types"):
        spec["layer_types"] = ["dense"] * int(spec["num_layers"])
    spec["intermediate"] = cfg.get("intermediate_size", spec.get("intermediate"))
    spec["num_heads"] = cfg.get("num_attention_heads", spec.get("num_heads"))
    spec["num_kv_heads"] = cfg.get("num_key_value_heads", spec.get("num_kv_heads"))
    spec["vocab"] = cfg.get("vocab_size", spec.get("vocab"))
    # the tokenizer's id count, which the head masks past (see fetch_real_vocab)
    spec["real_vocab"] = real_vocab or spec.get("real_vocab") or spec["vocab"]
    rope_theta = cfg.get("rope_theta", spec.get("rope_theta"))
    spec["rope_theta"] = float(rope_theta) if rope_theta is not None else rope_theta
    head_dim = cfg.get("head_dim") or cfg.get("hidden_size", 0) // max(cfg.get("num_attention_heads", 1), 1)
    spec["head_dim"] = head_dim or spec.get("head_dim")
    spec["rotary_dim"] = head_dim or spec.get("rotary_dim")
    model_name = args.model_name or generic_model_name(args.repo, args.gguf)
    spec["extra"] = {"model_type": cfg.get("model_type"), "source": "hf_config", "model": model_name}
    if cand is not None:
        built = cand.to_dict()
        # keep what the FAMILY BUILDER recorded in extra (Granite's pre-fold
        # attention multiplier, Phi's raw rope_scaling, ...) -- overwriting it dropped
        # the one fact the pack plan needs to fold a raw GGUF's weights
        extra = dict(built.get("extra") or {})
        extra.update({"model_type": cfg.get("model_type"), "source": "hf_config",
                      "model": model_name})
        built["extra"] = extra
        spec = built
    # Apply the base-model patch last, so both the family-builder result and
    # the template derivation get the rope/real_vocab the source cannot carry.
    apply_to_dict(spec, generic_model_name(args.repo, args.gguf))
    out = out_path
    if kept_model and spec.get("extra"):
        spec["extra"]["model"] = kept_model
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2))
    print(f"Wrote candidate spec to {out}")


if __name__ == "__main__":
    main()
