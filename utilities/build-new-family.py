#!/usr/bin/env python3
"""Onboard a HuggingFace/ModelScope repo to the open-kernel pipeline.

    oflm dev build-new-family <user>/<repo>
    python utilities/build-new-family.py --repo Qwen/Qwen3-0.6B --dry-run

This is the ORCHESTRATION layer, not a second kernel pipeline. Geometry is still
decided by the weights (GGUF header first, then the base model's config.json),
and `build-new-spec.py` still owns SPEC -> BUILD -> GATE -> CATALOGUE -> REBUILD
-> REGISTER. What this command adds is the missing front door: it derives the
geometry, decides whether the repo is an already-served size, a buildable size
of a known family, or a genuinely new architecture, runs the scriptable stages,
and prints the rest as an evidence checklist instead of claiming success.

Stages:

  1. SPEC        GGUF-first derivation for the repo (or a local --gguf file).
                 The base model from the repo's README is recorded as the
                 tensors/config authority when one is needed; no q4nx builder
                 is involved (q4nx is deprecated).
  2. CLASSIFY    served (a spec/kernel set already covers this geometry),
                 would_build (known family, new size), or would_extend (an
                 architecture no recipe owns).
  3. UNKNOWN ARCH  read the model's own GGUF header/tensor table, pick the
                 nearest existing recipe by geometry, write a candidate spec
                 against it from the standard GGUF keys, and go straight into
                 the normal BUILD/INSTALL path. RESEARCH.md records what the
                 nearest recipe does NOT explain so the numeric gates can be
                 trusted -- no reservation, no wall, no exit 2.
  4. BUILD       for would_build: promote the derived spec and call
                 build-new-spec.py, then locate the exported kernel set.
  5. INSTALL     for known families: `oflm add <repo>` (or the local GGUF),
                 which links the kernel set and writes settings.json.
  6. VALIDATE    layer dumps / decode gates / test suites are run only when the
                 caller supplies the CLI and reference model; otherwise the
                 exact commands are printed as required next steps.

A successful build is not a support claim: the numeric gates and the production
bar in AGENTS.md (context, profiling, features, bench) still apply. If the
architecture's arithmetic genuinely differs from the nearest recipe, the port
in .agents/skills/open-family-addition/SKILL.md is the remaining work.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "open_kernels" / "recipes" / "specs"
BUILD_SPEC = ROOT / "utilities" / "build-new-spec.py"
GEN_SPEC = ROOT / "utilities" / "gen-kernel-spec.py"
EXPORT = ROOT / "utilities" / "export-kernels.py"
PROBE = ROOT / "utilities" / "probe-family-sizes.py"
CMP_LAYERS = ROOT / "utilities" / "quant-compare" / "cmp_layers.py"
DECODE_GATES = ROOT / "utilities" / "decode-gates" / "decode_gates.py"
REF_DUMP = ROOT / "utilities" / "ref-dump" / "dump_ref"
REF_BUILD = ROOT / "utilities" / "ref-dump" / "build.sh"
RUNTIME_FAMILIES = ROOT / "src" / "include" / "AutoModel" / "model_families.hpp"


class StageError(RuntimeError):
    """A stage failed with a user-facing reason."""


def run(cmd: list[Any], *, env: dict[str, str] | None = None,
        cwd: Path | None = None, check: bool = True, capture: bool = False,
        quiet: bool = False) -> tuple[int, str]:
    cmd = [str(c) for c in cmd]
    if not quiet:
        print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, env=env or dict(os.environ), cwd=str(cwd or ROOT),
                       text=True, stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.PIPE if capture else None)
    out = ""
    if capture:
        out = (p.stdout or "")
        if p.stderr:
            out += ("\n" if out else "") + p.stderr
    if check and p.returncode != 0:
        raise StageError(f"FAILED: {' '.join(cmd)}")
    return p.returncode, out


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise StageError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_oflm_add():
    path = ROOT / "utilities" / "oflm-add"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return importlib.import_module("oflm_add")


def load_model_spec():
    path = ROOT / "open_kernels"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from recipes.spec import ModelSpec, SpecError  # noqa: PLC0415
    return ModelSpec, SpecError


def prepare_environment(dry_run: bool) -> tuple[str, dict[str, str]]:
    """The Python and PATH the size pipeline expects.

    Dry-run deliberately avoids ensure_venv(): deriving a spec needs only the
    parser dependencies and must not create a toolchain in the tree.
    """
    env = dict(os.environ)
    py = sys.executable
    if dry_run:
        return py, env
    mod = load_module("_export_kernels_for_new_family", EXPORT)
    py = str(mod.ensure_venv())
    parts: list[str] = []
    aie_bin = mod.llvm_aie_bin()
    if aie_bin:
        parts.append(str(aie_bin))
    parts.append(str(mod.XRT_BIN))
    parts.append(env.get("PATH", ""))
    env["PATH"] = os.pathsep.join(parts)
    lib = Path(mod.XRT_ROOT) / "lib"
    if lib.is_dir():
        env["LD_LIBRARY_PATH"] = os.pathsep.join([str(lib), env.get("LD_LIBRARY_PATH", "")])
    return py, env


def slugify(value: str) -> str:
    # Dots are meaningful in size names (qwen3-0.6b, gemma-3-1b); keep them.
    return re.sub(r"[^A-Za-z0-9.]+", "-", value).strip("-").lower() or "model"


def display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def split_repo(repo: str, add) -> tuple[str, str]:
    """`repo:QUANT` -> (repo, quant). A URL's port is not a quant."""
    if not repo:
        return "", ""
    return add.split_quant_request(repo)


def base_models_for(repo: str, add) -> list[str]:
    """The README base_model chain: the tensors/config authority when needed."""
    if not repo:
        return []
    kind, bare = add.split_remote_repo(repo)
    if not bare:
        return []
    try:
        return list(add.readme_base_chain(bare, modelscope=(kind == "modelscope")))
    except Exception:  # noqa: BLE001 - a private or offline repo has no README
        return []


def gguf_url_for(repo: str, add, quant: str = "") -> str | None:
    """The GGUF `oflm add` would choose, including quantized subdirectories.

    Unsloth puts dynamic quants in `UD-Q4_K_XL/...`; a root-only scan misses
    them, which is how a split family arrived here without a geometry source.
    A `repo:QUANT` request picks that quant instead of the default preference.
    """
    kind, bare = add.split_remote_repo(repo)
    if kind != "huggingface" or not bare:
        return None
    try:
        tree = add.hf_file_tree(bare)
    except Exception:  # noqa: BLE001 - gated/offline repos are allowed
        return None
    names = add.gguf_file_names(tree)
    chosen, _ = add.choose_gguf_file(names, want=quant or None)
    if not chosen:
        shard = getattr(add, "gguf_shard_info", lambda _n: None)
        model_names = getattr(add, "model_gguf_names", lambda ns: ns)(names)
        singles = [n for n in model_names if not shard(n)]
        chosen = singles[0] if singles else None
    if not chosen:
        return None
    return f"https://huggingface.co/{bare}/resolve/main/{chosen}"


@dataclass
class GGUFFacts:
    """What the model's own weight file says, when no recipe can derive a spec."""
    source: str = ""
    arch: str = ""
    name: str = ""
    size_label: str = ""
    family: str = ""
    model_types: list[str] = field(default_factory=list)
    name_prefixes: list[str] = field(default_factory=list)
    kv: dict[str, Any] = field(default_factory=dict)
    tensors: list[Any] = field(default_factory=list)
    base_model: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def trimmed_kv(self) -> dict[str, Any]:
        """The metadata without the tokenizer's 200k-element arrays."""
        out: dict[str, Any] = {}
        for key, value in self.kv.items():
            if key in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges",
                       "tokenizer.ggml.token_type"):
                out[key] = f"<{len(value)} entries>" if isinstance(value, (list, tuple)) else "<array>"
            elif key == "tokenizer.chat_template":
                out[key] = "<chat template>"
            elif isinstance(value, (list, tuple)) and len(value) > 64:
                out[key] = f"<{len(value)} entries: {list(value[:8])}...>"
            else:
                out[key] = value
        return out


@dataclass
class Derived:
    status: str = "error"           # ok | link | error
    path: Path | None = None
    data: dict[str, Any] = field(default_factory=dict)
    family: str = ""
    arch: str = ""
    spec_hash: str = ""
    kernel_hash: str = ""
    matched_spec: str = ""
    output: str = ""


def _hash_of(data: dict[str, Any], attr: str) -> str:
    ModelSpec, _SpecError = load_model_spec()
    try:
        return getattr(ModelSpec.from_dict(data), attr)()
    except Exception:  # noqa: BLE001 - a malformed candidate is reported below
        return ""


def _matching_spec(data: dict[str, Any], family: str) -> str:
    """The checked-in spec whose KERNELS cover this geometry, if any."""
    khash = _hash_of(data, "kernel_hash")
    if not khash or not family:
        return ""
    for path in sorted(SPECS.glob("*.json")):
        try:
            other = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if other.get("family") != family:
            continue
        try:
            if _hash_of(other, "kernel_hash") == khash:
                return path.stem
        except Exception:  # noqa: BLE001 - one bad spec must not hide the rest
            continue
    return ""


def gguf_facts(repo: str, args: argparse.Namespace, add) -> GGUFFacts | None:
    """The model's own GGUF header and tensor table.

    This is the geometry authority when no recipe can name the architecture.
    `gguf_url_for` already skips projectors and draft heads, so the file read
    here is the model's weights, not `dspark-*`'s speculative head.
    """
    source = str(Path(args.gguf).resolve()) if args.gguf else (gguf_url_for(repo, add, "") or "")
    if not source:
        return None
    try:
        inv = add.read_gguf_inventory(source, limit_mb=96)
    except Exception:  # noqa: BLE001 - offline / gated / unreadable
        return None
    if not inv:
        return None
    kv, tensors = inv
    arch = str(kv.get("general.architecture") or "")
    name = str(kv.get("general.name") or kv.get("general.basename") or "")
    facts = GGUFFacts(source=source, arch=arch, name=name,
                      size_label=str(kv.get("general.size_label") or ""),
                      kv=dict(kv), tensors=list(tensors))

    # The HF config supplies the model_type keys a GGUF does not carry. It is a
    # naming hint only: the header remains the geometry authority.
    try:
        gen = load_module("_gen_kernel_spec_facts", GEN_SPEC)
        base, text_cfg, _real = gen.spec_config_via_base_chain(repo) if repo else (None, None, None)
        if base:
            facts.base_model = base
            raw = gen.fetch_config(base)
            facts.config = dict(text_cfg or {})
            for mt in (raw.get("model_type"),
                       (raw.get("text_config") or {}).get("model_type"),
                       (raw.get("llm_config") or {}).get("model_type"),
                       (text_cfg or {}).get("model_type")):
                if isinstance(mt, str) and mt not in facts.model_types:
                    facts.model_types.append(mt)
    except Exception:  # noqa: BLE001 - no base chain / offline
        pass
    if not facts.model_types:
        facts.model_types = [arch, f"{arch}_text"] if arch else []

    for candidate in (name, str(kv.get("general.basename") or "")):
        if not candidate:
            continue
        slug = slugify(candidate)
        if not slug:
            continue
        parts = slug.split("-")
        for prefix in (slug, "-".join(parts[:2]) if len(parts) > 2 else ""):
            if prefix and prefix not in facts.name_prefixes:
                facts.name_prefixes.append(prefix)
    facts.family = (args.family or arch).strip()
    return facts


def candidate_spec(facts: GGUFFacts, family: str = "") -> dict[str, Any]:
    """A ModelSpec candidate from the GGUF's standard keys, under `family`.

    `family` is the nearest recipe chosen by `nearest_recipe()`, not a new
    family name: the candidate is built so `build-new-spec` can extend that
    recipe to this size instead of stopping at a wall.
    """
    ModelSpec, _SpecError = load_model_spec()
    try:
        return ModelSpec.from_gguf_candidate(facts.kv, family or facts.family,
                                             facts.tensors).to_dict()
    except Exception as exc:  # noqa: BLE001 - the research brief still has the facts
        print(f"warn: could not build a candidate spec: {exc}", file=sys.stderr)
        return {}


# Geometry fields a recipe must match for a candidate to be an easy extension.
# Ordered by how strongly each one determines a kernel layout: a wrong attention
# geometry or expert count changes the kernels, a wrong vocab only the lm_head.
_RECIPE_MATCH_FIELDS = (
    ("head_dim", 3), ("num_heads", 3), ("num_kv_heads", 3),
    ("rotary_dim", 2), ("qk_norm", 2), ("attn_gate", 2),
    ("hidden", 2), ("intermediate", 1), ("sliding_window", 1),
    ("conv_kernel", 1), ("lin_key_heads", 1), ("lin_value_heads", 1),
)


def _structural_score(s, data) -> int:
    """Whether the family's LAYER SHAPE matches, apart from exact numbers.

    A model with experts belongs with the MoE recipe even when its exact expert
    count is new; a gated-attention model belongs with a family that has a gate;
    a hybrid with convs belongs with the conv family. These are worth more than
    an exact head_dim match because they decide which kernels exist at all.
    """
    score = 0
    moe = int(data.get("num_experts") or 0) > 0
    if moe == (int(getattr(s, "num_experts", 0) or 0) > 0):
        score += 30          # experts vs. dense is a different kernel set entirely
    gate = bool(data.get("attn_gate"))
    if gate == bool(getattr(s, "attn_gate", False)):
        score += 15          # a gated attention needs a gate stream in the design
    conv = int(data.get("conv_kernel") or 0) > 0
    if conv == (int(getattr(s, "conv_kernel", 0) or 0) > 0):
        score += 10
    swa = int(data.get("sliding_window") or 0) > 0
    if swa == (int(getattr(s, "sliding_window", 0) or 0) > 0):
        score += 3
    return score


def nearest_recipe(data: dict[str, Any], facts: GGUFFacts) -> tuple[str, list[str]]:
    """The existing recipe whose geometry is closest to this candidate.

    Returns (family, reasons). A score is a weighted count of matching spec
    fields; ties go to the family that owns the most checked-in specs at a
    similar size. If nothing scores, `dense` is the fallback: it is the broadest
    recipe and the one most spec extensions land in.

    This never claims the recipes are equivalent -- `RESEARCH.md` lists the
    fields that DID NOT match so the operator knows what the extension must
    change. It picks a STARTING POINT for the build, not a correctness verdict.
    """
    sys.path.insert(0, str(ROOT / "open_kernels"))
    try:
        from recipes import families as F
        from recipes.load import load_spec
    except Exception as exc:  # noqa: BLE001
        return "dense", [f"could not load the recipe registry: {exc}"]
    candidates: list[tuple[int, str]] = []
    for family in F.FAMILIES:
        files = sorted((ROOT / "open_kernels" / "recipes" / "specs").glob("*.json"))
        own = []
        for path in files:
            try:
                s = load_spec(path)
            except Exception:  # noqa: BLE001 - one bad spec must not hide the rest
                continue
            if s.family == family:
                own.append(s)
        if not own:
            continue
        def score_spec(s) -> int:
            total = _structural_score(s, data)
            for field, weight in _RECIPE_MATCH_FIELDS:
                if data.get(field) is not None and getattr(s, field, None) == data[field]:
                    total += weight
            return total
        # The best spec in a family is the evidence; a family with twenty specs
        # should not win just for having twenty.
        candidates.append((max(score_spec(s) for s in own), family))
    if not candidates:
        return "dense", ["no checked-in spec could be scored; using the broadest recipe"]
    candidates.sort(key=lambda x: -x[0])
    top = candidates[0][0]
    best = [f for score, f in candidates if score == top]
    # On a tie, prefer the family whose LAYER SHAPE is closest: that is which
    # kernels exist, not which headline geometry number happens to match.
    family = max(best, key=lambda f: _shape_bonus(f, data))
    reasons = [f"{f}: {s} geometry/shape points match"
               for s, f in candidates if s > 0][:5]
    if not reasons:
        reasons = [f"no geometry field matched any recipe; using {family} as the starting point"]
    return family, reasons


def _shape_bonus(family: str, data: dict[str, Any]) -> int:
    """Tie-break: prefer the family whose recipe has the same layer kinds."""
    bonus = 0
    if int(data.get("num_experts") or 0) > 0 and family in ("qwen36moe", "qwen35moe"):
        bonus += 3
    if data.get("attn_gate") and family in ("qwen35", "qwen35moe", "lfm2", "lfm3"):
        bonus += 2
    if int(data.get("conv_kernel") or 0) > 0 and family in ("lfm2", "lfm2moe", "lfm3"):
        bonus += 2
    return bonus


def tensor_suffixes(facts: GGUFFacts) -> list[str]:
    """Tensor-name suffixes outside every role map, i.e. what the port must explain."""
    sys.path.insert(0, str(ROOT / "open_kernels"))
    known: set[str] = {"token_embd.weight", "output_norm.weight", "output.weight"}
    try:
        from recipes import spec as spec_mod  # noqa: PLC0415
        for roles in spec_mod.ROLE_TENSORS.values():
            known.update(roles)
        known.update(spec_mod._GGUF_ROLE)       # GGUF names for the same roles
        block = spec_mod._LAYER_PREFIX          # model.layers.N.
    except Exception:  # noqa: BLE001
        block = re.compile(r"^model\.layers?\.\d+\.")
    out: set[str] = set()
    for name, _ty, _dims in facts.tensors:
        leaf = block.sub("", str(name))
        leaf = re.sub(r"^blk\.\d+\.", "", leaf)
        if not leaf.endswith((".weight", ".bias")):
            continue
        if leaf in known:
            continue
        out.add(leaf)
    return sorted(out)


_SIGNAL_TOKENS = ("activation", "softcap", "sliding_window", "loop.", "sink",
                  "conv", "rel_bias", "relative", "res_query", "clip", "rope.",
                  "expert", "shared", "bias", "situ", "gate_rank", "varying")


def architecture_signals(facts: GGUFFacts) -> list[str]:
    """Metadata keys that name arithmetic no standard ModelSpec field covers."""
    out: list[str] = []
    prefix = facts.arch + "." if facts.arch else ""
    for key, value in sorted(facts.kv.items()):
        if not (key.startswith(prefix) or key.startswith("general.")):
            continue
        # Match the key WITHOUT the architecture name: `inkling2_loop.` contains
        # "loop" itself, which would match every key of that model.
        low = (key[len(prefix):] if prefix and key.startswith(prefix) else key).lower()
        if not any(tok in low for tok in _SIGNAL_TOKENS):
            continue
        if isinstance(value, (list, tuple)) and len(value) > 16:
            value = f"<{len(value)} entries: {list(value[:6])}...>"
        out.append(f"- `{key}` = `{value}`")
    return out


def write_research(work: Path, facts: GGUFFacts, derived: Derived,
                   base: str, install_target: str, recipe: str = "",
                   recipe_reasons: list[str] | None = None,
                   built: bool = False) -> Path:
    """The build report: what recipe was extended, what matched, what to verify."""
    work.mkdir(parents=True, exist_ok=True)
    meta = facts.trimmed_kv()
    (work / "gguf-meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True),
                                         encoding="utf-8")
    with (work / "tensors.tsv").open("w", encoding="utf-8") as fh:
        fh.write("name\tggml_type\tshape\n")
        for tensor_name, ty, dims in facts.tensors:
            fh.write(f"{tensor_name}\t{ty}\t{'x'.join(str(d) for d in dims)}\n")
    first_block = [t for t in facts.tensors if str(t[0]).startswith(("blk.0.", "model.layers.0."))]
    globals_ = [t for t in facts.tensors if not re.search(r"(blk|layers?)\.\d+\.", str(t[0]))]
    unknown = tensor_suffixes(facts)

    refs: list[str] = []
    ref_repo = base or (install_target.split(":")[0] if install_target else "")
    if ref_repo:
        try:
            add = load_oflm_add()
            tree = add.hf_file_tree(ref_repo)
            names = [e.get("path") if isinstance(e, dict) else str(e) for e in tree]
            wanted = [n for n in names if n and (
                n.endswith("config.json") or n.endswith("README.md")
                or re.search(r"(modeling|configuration|modular|tokenization)_.*\.py$", n))]
            refs = [f"https://huggingface.co/{ref_repo}/blob/main/{n}" for n in wanted[:40]]
        except Exception:  # noqa: BLE001 - offline / gated
            pass
    if facts.arch:
        refs.append(f"https://github.com/ggml-org/llama.cpp/search?q={facts.arch}&type=code")

    lines = [
        f"# Build report: {facts.arch or facts.family}",
        "",
        f"- repo: `{install_target or facts.source}`",
        f"- base / tensor authority: `{base or '(none)'}`",
        f"- weight file: `{facts.source}`",
        f"- `general.architecture`: `{facts.arch}`",
        f"- `general.name`: `{facts.name or '(none)'}`",
        f"- `general.size_label`: `{facts.size_label or '(none)'}`",
        f"- tensors: {len(facts.tensors)}; candidate spec: `{derived.path or '(not written)'}`",
        f"- nearest recipe: `{recipe or '(none)'}`",
        f"- built: **{'yes' if built else 'no'}**",
        "",
        "## Nearest recipe",
        "",
        "The architecture has no recipe of its own, so `build-new-family` extended",
        f"the closest existing one, `{recipe or '(unknown)'}`, with a candidate spec",
        "built from the GGUF's standard keys. It did not stop at a wall: the",
        "exact command to rebuild is printed in the plan and in `PLAN.md`.",
        "",
    ]
    lines += [f"- {r}" for r in (recipe_reasons or [])] or ["- (no geometry reasons recorded)"]
    lines += [
        "",
        "The sections below record what the nearest recipe does NOT explain, so",
        "you can verify the build against the real arithmetic instead of trusting",
        "a successful export. `open-family-addition` is the reference if the",
        "differences turn out to need a real port.",
        "",
        "## What the file says",
        "",
        "```json",
        json.dumps(meta, indent=2, sort_keys=True),
        "```",
        "",
        "## Arithmetic signals",
        "",
        "These keys name behavior no standard `ModelSpec` field covers; the port",
        "has to decide where each belongs (a new spec field, a catalogue/attention",
        "switch, or a design-local constant):",
        "",
    ]
    lines += architecture_signals(facts) or ["- (none: the header is all standard keys)"]
    lines += [
        "",
        "## Tensors the role maps do not explain",
        "",
        "These are the names the port has to account for (a standard attention/FFN",
        "role is not among them):",
        "",
    ]
    lines += [f"- `{leaf}`" for leaf in unknown] or ["- (none: every tensor matches a known role)"]
    lines += [
        "",
        "### Layer 0",
        "",
        "| tensor | type | shape |",
        "|---|---|---|",
    ]
    lines += [f"| `{n}` | {ty} | {'x'.join(str(d) for d in dims)} |" for n, ty, dims in first_block]
    lines += ["", "### Global tensors", "", "| tensor | type | shape |", "|---|---|---|"]
    lines += [f"| `{n}` | {ty} | {'x'.join(str(d) for d in dims)} |" for n, ty, dims in globals_[:80]]
    lines += [
        "",
        "## Verify the extension",
        "",
        "- Run the numeric gate: `cmp_layers.py` against a llama.cpp reference dump.",
        "- Check the arithmetic signals above against the recipe's design; a missing",
        "  stage (loop passes, relative bias, convs, sinks) is where a build can",
        "  succeed yet compute the wrong thing.",
        "- If the layers need different geometries, follow Gemma 4's per-layer",
        "  sub-spec pattern in `open-family-addition`.",
        "",
        "## References",
        "",
    ]
    lines += [f"- {url}" for url in refs] or ["- (no repository file list available)"]
    lines += [
        "",
        "## Re-run / verify",
        "",
        "```sh",
        f"oflm dev build-new-family {install_target} --cli <cli> --ref-model <gguf> --install-dir <dir>",
        "```",
        "",
        "With the reference arguments the layer dumps and decode gates run here.",
        "A successful build is not a support claim: the production bar in",
        "`AGENTS.md` still applies.",
        "",
    ]
    path = work / "RESEARCH.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def derive_spec(args: argparse.Namespace, work: Path, py: str,
                env: dict[str, str], repo: str, add) -> Derived:
    candidate = work / "candidate-spec.json"
    cmd = [py, GEN_SPEC]
    if args.gguf:
        cmd += ["--gguf", str(Path(args.gguf).resolve())]
    else:
        cmd += ["--repo", repo]
    if args.model_name:
        cmd += ["--model-name", args.model_name]
    cmd += ["--out", str(candidate)]
    rc, out = run(cmd, env=env, check=False, capture=True)
    print(out.rstrip(), flush=True)
    d = Derived(output=out)
    if rc != 0 or not candidate.is_file():
        m = re.search(r"general\.architecture '([^']+)'", out) or \
            re.search(r"model_type '([^']+)'", out)
        d.arch = m.group(1) if m else ""
        return d
    try:
        d.data = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        d.output += f"\n{exc}"
        return d
    d.status = "link" if "link_to" in d.data else "ok"
    d.path = candidate
    if d.status == "link":
        # gen-kernel-spec writes a tiny {"link_to": ...} instead of a full spec
        # when an existing spec already covers this kernel geometry. Read the
        # target so the report can still name the family and hashes it matched.
        link = str(d.data.get("link_to") or "")
        target = Path(link) if link.endswith(".json") else SPECS / f"{link}.json"
        if not target.is_absolute() and not target.is_file():
            target = SPECS / target
        if target.is_file():
            try:
                linked = json.loads(target.read_text(encoding="utf-8"))
                linked["link_to"] = link
                d.data = linked
            except (OSError, ValueError):
                pass
    d.family = str(d.data.get("family") or "")
    extra = d.data.get("extra") or {}
    d.arch = str(extra.get("model_type") or extra.get("general.architecture") or "")
    d.spec_hash = _hash_of(d.data, "spec_hash")
    d.kernel_hash = _hash_of(d.data, "kernel_hash")
    if d.status == "ok":
        d.matched_spec = _matching_spec(d.data, d.family)
    return d


def recipe_facts(family: str) -> tuple[bool, str]:
    if not family:
        return False, "no family was derived"
    sys.path.insert(0, str(ROOT / "open_kernels"))
    try:
        from recipes import families  # noqa: PLC0415
        families.family_module(family)
        return True, ""
    except Exception as exc:  # noqa: BLE001 - report the recipe's own refusal
        return False, str(exc)


def runtime_family_known(family: str) -> bool:
    if not family or not RUNTIME_FAMILIES.is_file():
        return False
    text = RUNTIME_FAMILIES.read_text(encoding="utf-8", errors="ignore")
    return bool(re.search(rf"[\"']{re.escape(family)}[\"']", text))


def installed_gguf_name(output: str) -> str:
    """The GGUF `oflm add --dry-run` actually chose, from its `weights` line.

    `choose_gguf_file` in this process uses a different preference order than
    the installer (which falls through quant types the NPU packers reject), so
    re-picking here can point the study at a file that cannot run. The probe's
    own decision is the authority.
    """
    m = re.search(r"^weights\s*:\s*(\S+\.gguf)\s*->", output, re.M)
    return m.group(1).strip() if m else ""


def classify_probe(output: str, rc: int) -> tuple[str, dict[str, Any]]:
    def xclbin_served(text: str) -> tuple[str, dict[str, Any]] | None:
        m = re.search(r"^xclbin source\s*:\s*(.+)$", text, re.M)
        if not m:
            return None
        value = m.group(1).strip()
        # A configured .q4nx/.npue base model is not an open-kernel set.
        if not value or value.lower().startswith("(") or value.lower().endswith(".q4nx"):
            return None
        return "served", {"kernels": value}

    try:
        mod = load_module("_probe_family_sizes_for_new_family", PROBE)
        status, info = mod.classify(output, rc)
        # The installer's chosen weight file is the geometry evidence; the
        # probe's own classifier does not carry it, so attach it here.
        info = dict(info or {})
        chosen = installed_gguf_name(output)
        if chosen:
            info["gguf"] = chosen
        # probe-family-sizes is GGUF-oriented: a safetensors repo reports its
        # serving set on `xclbin source`, with no `gguf kernels` line to parse.
        if status == "unknown":
            return xclbin_served(output) or (status, info)
        return status, info
    except Exception:  # noqa: BLE001 - the small local fallback keeps this usable
        low = output.lower()
        info: dict[str, Any] = {}
        m = re.search(r"gguf kernels\s*:\s*(.+)", output)
        if m:
            info["kernels"] = m.group(1).strip()
        chosen = installed_gguf_name(output)
        if chosen:
            info["gguf"] = chosen
        if "no open kernel set" in low or "no recipe" in low:
            return "unreadable", info
        if "gguf kernels" in low and "(would build" in low:
            return "would_build", info
        if "gguf kernels" in low and "(none" in low:
            return "no_set", info
        served = xclbin_served(output)
        if served:
            return served
        return ("served" if "gguf kernels" in low else "error"), info


def probe_install(args: argparse.Namespace, install_target: str) -> dict[str, Any]:
    oflm = os.environ.get("OFLM_EXECUTABLE") or shutil.which("oflm")
    if not oflm or not install_target:
        return {"status": "skipped", "detail": "`oflm` not found on PATH; run "
                "`oflm add <repo> --dry-run` manually", "output": ""}
    cmd = [oflm, "add", install_target, "--dry-run"]
    if args.family:
        cmd += ["--family", args.family]
    rc, out = run(cmd, check=False, capture=True)
    status, info = classify_probe(out, rc)
    detail = ""
    for line in out.splitlines():
        if any(k in line for k in ("gguf kernels", "family '", "xclbin source",
                                   "no open kernel", "would build")):
            detail = line.strip()
    print(out.rstrip(), flush=True)
    info["alternatives"] = installable_quants(out)
    info["weight_errors"] = weight_errors(out)
    return {"status": status, "detail": detail, "info": info, "output": out, "rc": rc}


def quant_token(name: str) -> str:
    """The quant request token for a GGUF path, as `repo:QUANT` would name it."""
    path = Path(str(name))
    # Unsloth dynamic quants live in a directory named for the quant.
    for part in path.parts[:-1]:
        if re.fullmatch(r"UD-Q\d.*|Q\d.*|IQ\d.*", part, re.I):
            return part.upper()
    for tok in path.stem.split("-"):
        if re.fullmatch(r"(UD-)?(Q|IQ|TQ|PTQ|PQ|MXFP)\d.*|BF16|F16|F32", tok, re.I):
            return tok.upper()
    return ""


def installable_quants(output: str) -> list[str]:
    """Alternative quant tokens the dry run did NOT refuse, for a retry.

    `oflm add --dry-run` prints a `skipped: <path> (<reason>)` line for every
    file it rejected. A quant whose every file appears only in a refused line
    is unusable; any other named quant is a candidate for `repo:QUANT`.
    """
    refused_quants: set[str] = set()
    seen_quants: set[str] = set()
    for line in output.splitlines():
        m = re.search(r"skipped\s*:\s*(\S+?\.gguf)\s*\((.+)\)", line)
        if not m:
            continue
        name, why = m.group(1), m.group(2)
        q = quant_token(name)
        if not q:
            continue
        seen_quants.add(q)
        # A shard "installed beside part 1" is not itself a quant refusal;
        # treat the quant as refused only for a reason that names the file.
        if ("part " in why and "installed beside part 1" in why):
            continue
        refused_quants.add(q)
    return sorted(seen_quants - refused_quants)


def weight_errors(output: str) -> list[str]:
    """One line per reason the dry run refused a weight file, for the report."""
    errors = []
    for line in output.splitlines():
        if "refused (" in line or "not supported by the open kernels" in line:
            errors.append(line.strip())
    return errors[:12]


def install_with_fallback(args: argparse.Namespace, oflm: str, install_target: str,
                          probe: dict[str, Any], env: dict[str, str]) -> int:
    """Run `oflm add`; on failure, try each alternative quant the probe named.

    A first failure is not the end: the dry run already told us which files are
    installable, so retrying them is scriptable. Returns 0 on any success,
    otherwise a clear report and 1 -- never a bare `FAILED` line.
    """
    def invoke(target: str) -> int:
        cmd = [oflm, "add", target]
        if args.family:
            cmd += ["--family", args.family]
        if args.force:
            cmd += ["--force"]
        if args.verbose:
            cmd += ["--verbose"]
        print("-- install:", " ".join(cmd), flush=True)
        return run(cmd, env=env, check=False)

    rc = invoke(install_target)
    if rc == 0:
        return 0
    # Try the other supported quant tokens the probe listed in its
    # `skipped` lines, as `repo:QUANT` requests.
    bare = install_target.split(":")[0]
    tried = set()
    for quant in (probe.get("info") or {}).get("alternatives", []):
        if quant in tried:
            continue
        tried.add(quant)
        rc = invoke(f"{bare}:{quant}")
        if rc == 0:
            print(f"-- installed with quant {quant}", flush=True)
            return 0
    print(f"\n-- install: no compatible GGUF in {install_target}. The probe refused every "
          f"candidate (see its `skipped` lines above); this model cannot be installed "
          f"until it is converted to a supported quant. Geometry matched, but the "
          f"weight file is not loadable.", file=sys.stderr)
    return 1


def spec_name_for(family: str, repo: str, data: dict[str, Any]) -> str:
    """A stable spec stem: family + size, without duplicating the family name."""
    model = str((data.get("extra") or {}).get("model") or "")
    raw = model or repo.rstrip("/").split("/")[-1] or "model"
    slug = slugify(raw)
    slug = re.sub(r"-(gguf|q4nx|i1|npu2)$", "", slug)
    slug = re.sub(r"-(gguf|q4nx|i1)$", "", slug)
    fam = slugify(family.replace(".", ""))
    # "Qwen3.8-27B" slugifies to qwen3-8-27b while the family is qwen38; compare
    # compact forms so a punctuation variant still counts as "family + size".
    compact = re.sub(r"[^a-z0-9]", "", slug)
    if fam and (compact.startswith(fam) or fam.startswith(compact)):
        return slug
    return f"{fam}-{slug}" if fam else slug


def next_commands(args: argparse.Namespace, status: str, name: str,
                  install_target: str, bare_repo: str, *, family: str = "",
                  research: str = "", plan_dir: str = "") -> list[str]:
    cmds: list[str] = []
    if status == "weights_blocked":
        cmds.append("# The kernels fit this geometry, but no GGUF in the repo is loadable "
                    "by the installer. Nothing to build or install -- convert the weights "
                    "to a supported quant first.")
        cmds.append(f"# Probe detail: {install_target}")
        return cmds
    if status == "would_extend":
        cmds.append(f"# Unknown architecture: extended the nearest recipe '{family or name}'.")
        if research:
            cmds.append(f"# Build report (nearest recipe, unmatched tensors, references): {research}")
        cmds.append("# The build ran against the nearest recipe; verify the numeric gates "
                    "below before treating it as supported.")
        cmds.append("# If the arithmetic genuinely differs, follow .agents/skills/open-family-addition/SKILL.md.")
    if status in ("would_build", "would_extend"):
        cmds.append(f"oflm dev build-new-spec --spec {name}")
    if install_target:
        cmds.append(f"oflm add {install_target}")
    cmds += [
        "# Numeric acceptance (llama.cpp reference vs NPU layer dumps):",
        "cmake -S src/open_qwen36 -B build-qwen36 && cmake --build build-qwen36 --target open_qwen36_cli",
        "utilities/ref-dump/dump_ref --model <model.gguf> --ids 1,2,3 --out /tmp/ref/y",
        "OFLM_OPEN_LAYER_NAN=1 OFLM_OPEN_LAYER_DUMP=/tmp/npu build-qwen36/open_qwen36_cli "
        "--model <install> --kernels <install>/open_kernels --ids 1,2,3 --prefill-logits",
        "python utilities/quant-compare/cmp_layers.py /tmp/ref/y /tmp/npu",
        "# Then the production bar: full context, profiling, speculative decoding, oflm-test, bench.",
    ]
    return cmds


def build_plan(args: argparse.Namespace, repo: str, quant: str, bases: list[str],
               derived: Derived, probe: dict[str, Any], add,
               facts: GGUFFacts | None = None, research: str = "",
               recipe: str = "", recipe_reasons: list[str] | None = None,
               extending: bool = False) -> dict[str, Any]:
    recipe_ok, recipe_reason = recipe_facts(derived.family)
    probe_bad = probe.get("status") in ("error", "unreadable", "not_found", "no_set")
    if (derived.status == "link" or derived.matched_spec) and not probe_bad:
        status = "served"
        reason = (f"the base already has kernels: {derived.data.get('link_to')}"
                  if derived.status == "link"
                  else f"checked-in spec {derived.matched_spec} covers this kernel geometry")
    elif (derived.status == "link" or derived.matched_spec) and probe_bad:
        # A geometry hash match says the kernels FIT, not that this weight file
        # can be loaded. The installer's own dry run refused every candidate;
        # reporting "served" here would assert an install that cannot succeed.
        status = "weights_blocked"
        errors = (probe.get("info") or {}).get("weight_errors") or []
        # Keep the reason short: the full probe log is in PLAN.md and the
        # installer's own `skipped` lines were printed above.
        short = []
        for e in errors:
            m = re.search(r"refused \(([^;]+)", e) or re.search(r": (quant [^;]+)", e)
            short.append(m.group(1).strip() if m else e[:80])
        reason = (f"geometry matches {derived.matched_spec or derived.family}, but no "
                  f"installable GGUF"
                  + (f": {'; '.join(dict.fromkeys(short))[:300]}" if short else ""))
    elif extending:
        # The architecture has no recipe of its own; the family here is the
        # nearest existing recipe we're extending, whether or not --family was
        # used to pick it.
        status = "would_extend"
        arch = derived.arch or (facts.arch if facts else "")
        why = "; ".join((recipe_reasons or [])[:2])
        named = "explicitly named" if args.family else "nearest"
        reason = (f"architecture {arch or '(unknown)'} has no recipe of its own; "
                  f"extending the {named} recipe '{recipe or derived.family}'"
                  + (f" ({why})" if why else ""))
    elif derived.status == "error" or not derived.family:
        # The caller builds a candidate from the GGUF under a nearest recipe before
        # this point; if it could not, say so plainly rather than call it a build.
        status = "would_extend"
        arch = derived.arch or (facts.arch if facts else "")
        reason = (f"architecture {arch or '(unknown)'} could not be derived; "
                  f"a GGUF header is needed to choose a recipe")
    elif probe["status"] in ("served",):
        status = "served"
        reason = f"oflm add --dry-run reports a serving set: {probe.get('detail', '')}"
    else:
        status = "would_build"
        reason = (f"known family '{derived.family}', no checked-in spec covers this geometry "
                  f"({probe.get('status', 'unknown')}: {probe.get('detail', '')})")

    name = args.spec_name or spec_name_for(derived.family or derived.arch,
                                          repo or "model", derived.data)
    install_target = args.gguf if (args.gguf and not repo) else (repo or args.gguf or "")
    tensor_repo = args.tensor_repo or (bases[0] if bases else "")
    if facts and not tensor_repo:
        tensor_repo = facts.base_model
    gguf_url = gguf_url_for(repo, add, quant) if repo and not args.gguf else ""
    # The installer's own choice wins: it falls through quant types the NPU
    # packers reject, which our local `choose_gguf_file` call does not model.
    probed = (probe.get("info") or {}).get("gguf") or ""
    if probed and repo:
        bare = add.split_remote_repo(repo)[1] or repo
        gguf_url = f"https://huggingface.co/{bare}/resolve/main/{probed}"
    plan: dict[str, Any] = {
        "repo": repo,
        "repo_argument": install_target,
        "quant": quant,
        "local_gguf": str(Path(args.gguf).resolve()) if args.gguf else "",
        "gguf_url": gguf_url,
        "base_models": bases,
        "tensor_repo": tensor_repo,
        "tensor_repo_reason": ("--tensor-repo override" if args.tensor_repo else
                               "README base_model (tensors/config authority)" if bases else
                               "the GGUF is the geometry authority; no tensors repo needed"
                               if args.gguf else "no base_model chain found"),
        "status": status,
        "reason": reason,
        "spec_name": name,
        "derived": {
            "status": derived.status,
            "family": derived.family,
            "architecture": derived.arch,
            "spec_hash": derived.spec_hash,
            "kernel_hash": derived.kernel_hash,
            "matched_spec": derived.matched_spec,
            "candidate": str(derived.path) if derived.path else "",
        },
        "recipe": {"supported": recipe_ok, "reason": recipe_reason,
                   "nearest": recipe, "reasons": recipe_reasons or []},
        "research": research,
        "runtime_family_known": runtime_family_known(derived.family),
        "probe": {k: v for k, v in probe.items() if k != "output"},
        "next_commands": next_commands(args, status, name, install_target, repo,
                                       family=derived.family, research=research),
    }
    return plan


def print_report(plan: dict[str, Any]) -> None:
    print("\n== build-new-family ==")
    print(f"  repo:        {plan['repo_argument'] or '(none)'}")
    if plan["quant"]:
        print(f"  quant:       {plan['quant']}")
    print(f"  base model:  {plan['tensor_repo'] or '(not needed / not found)'} "
          f"({plan['tensor_repo_reason']})")
    if plan["gguf_url"]:
        print(f"  gguf:        {plan['gguf_url']}")
    d = plan["derived"]
    print(f"  family:      {d['family'] or plan['derived']['architecture'] or '(unknown)'}")
    print(f"  arch:        {d['architecture'] or '(unknown)'}")
    if d["spec_hash"]:
        print(f"  spec hash:   {d['spec_hash']}")
    if d["matched_spec"]:
        print(f"  matched:     {d['matched_spec']}")
    print(f"  probe:       {plan['probe'].get('status', 'skipped')} "
          f"{plan['probe'].get('detail', '')}")
    print(f"  status:      {plan['status']}")
    print(f"  reason:      {plan['reason']}")
    print("\nnext steps:")
    for cmd in plan["next_commands"]:
        print(f"  {cmd}")


def persist_plan(plan: dict[str, Any], args: argparse.Namespace, work: Path,
                 derived: Derived) -> None:
    """Write the authoring record under the work dir (a temp dir in --dry-run)."""
    work.mkdir(parents=True, exist_ok=True)
    plan_path = work / "build-new-family.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    candidate_out = work / "candidate-spec.json"
    if (derived.path and derived.path.is_file()
            and derived.path.resolve() != candidate_out.resolve()):
        shutil.copyfile(derived.path, candidate_out)
    if derived.output.strip():
        (work / "derivation.log").write_text(derived.output, encoding="utf-8")
    if plan["probe"].get("output"):
        (work / "oflm-add-dry-run.log").write_text(plan["probe"]["output"], encoding="utf-8")
    lines = ["# Onboarding plan", "", f"- repo: `{plan['repo_argument']}`",
             f"- status: **{plan['status']}**", f"- reason: {plan['reason']}",
             f"- base/tensor repo: `{plan['tensor_repo'] or 'none'}`",
             f"- nearest recipe: `{plan['recipe'].get('nearest') or 'n/a'}`", ""]
    if plan.get("research"):
        lines += [f"Architecture study: `{plan['research']}`", ""]
    lines += ["## Commands", "", "```sh"]
    lines += plan["next_commands"]
    lines += ["```", ""]
    (work / "PLAN.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nplan: {plan_path}")
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(f"json: {out}")


def promote_spec(derived: Derived, name: str, force: bool) -> Path:
    if not derived.path or not derived.path.is_file():
        raise StageError("no derived spec to promote")
    dest = SPECS / f"{name}.json"
    incoming = json.loads(derived.path.read_text(encoding="utf-8"))
    if dest.is_file():
        try:
            existing = json.loads(dest.read_text(encoding="utf-8"))
        except ValueError:
            existing = None
        if existing == incoming:
            return dest
        if not force:
            raise StageError(f"{display(dest)} already exists with different "
                             "geometry; pass --force to replace it")
    shutil.copyfile(derived.path, dest)
    return dest


def find_exported_set(add, derived: Derived) -> tuple[str, str, list[str]]:
    ModelSpec, _ = load_model_spec()
    try:
        spec = ModelSpec.from_dict(derived.data)
    except Exception:  # noqa: BLE001 - the build already reported its own failure
        return "", "", []
    kdir, name, notes = add.scan_kernel_sets(spec, [add.tree_xclbins()], need_gguf=True)
    return (str(kdir) if kdir else ""), (name or ""), notes


def ensure_ref_dump() -> Path | None:
    if REF_DUMP.is_file():
        return REF_DUMP
    if not REF_BUILD.is_file():
        return None
    run(["bash", REF_BUILD], cwd=REF_BUILD.parent, check=False, capture=True)
    return REF_DUMP if REF_DUMP.is_file() else None


def validate(args: argparse.Namespace, env: dict[str, str], derived: Derived,
             set_name: str) -> None:
    if args.ref_model and args.cli and args.install_dir:
        dump = ensure_ref_dump()
        if not dump:
            print("validation: dump_ref is unavailable; install libllama or run "
                  f"{REF_BUILD}", file=sys.stderr)
            return
        out = Path(args.work or (ROOT / "build" / "new-family")) / "numeric"
        out.mkdir(parents=True, exist_ok=True)
        ids = args.ids or "1,2,3"
        run([dump, "--model", args.ref_model, "--ids", ids, "--out", out / "ref"], env=env)
        cli = Path(args.cli)
        e = dict(env, OFLM_OPEN_LAYER_NAN="1", OFLM_OPEN_LAYER_DUMP=str(out / "npu"))
        install = Path(args.install_dir)
        run([cli, "--model", install, "--kernels", install / "open_kernels",
             "--ids", ids, "--prefill-logits"], env=e)
        run([sys.executable, CMP_LAYERS, out / "ref", out / "npu"], env=env)
        if args.base_set and set_name:
            for mode in ("logits", "corr", "greedy"):
                run([sys.executable, DECODE_GATES, mode, "--cli", cli,
                     "--model", install, "--sets", args.base_set, set_name], env=env)
    else:
        print("\nvalidation: supply --cli, --ref-model and --install-dir to run the "
              "layer/decode gates here; the commands above are the required evidence.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", nargs="?", default="", help="the HF/ModelScope repo (user/repo), "
                    "optionally user/repo:QUANT; a URL or local dir is accepted too")
    ap.add_argument("--repo", dest="repo_opt", default="", help="same as the positional repo")
    ap.add_argument("--gguf", default="", help="derive from a local GGUF (gated/private repos)")
    ap.add_argument("--family", default="", help="the recipe to build against. For an unknown "
                    "architecture this overrides the nearest recipe chosen by geometry; also "
                    "the conservative oflm add hint")
    ap.add_argument("--tensor-repo", default="", help="tensors/config repo override "
                    "(default: the repo's README base_model chain)")
    ap.add_argument("--spec-name", default="", help="the spec stem (default: family + size)")
    ap.add_argument("--model-name", default="", help="extra.model for the derived spec")
    ap.add_argument("--dry-run", action="store_true", help="derive and classify without "
                    "touching the repo; print the exact commands")
    ap.add_argument("--skip-build", action="store_true", help="do not run build-new-spec")
    ap.add_argument("--skip-install", action="store_true", help="do not run oflm add")
    ap.add_argument("--skip-validate", action="store_true", help="do not run numeric gates")
    ap.add_argument("--force", action="store_true", help="replace a different spec of the same name")
    ap.add_argument("--verbose", action="store_true", help="pass --verbose to oflm add")
    ap.add_argument("--work", default="", help="artifact directory (default: build/new-family/<repo>)")
    ap.add_argument("--json", default="", help="also write the plan as JSON to this path")
    ap.add_argument("--cli", default="", help="open_qwen36_cli for numeric validation")
    ap.add_argument("--ref-model", default="", help="llama.cpp-readable GGUF for the reference dump")
    ap.add_argument("--install-dir", default="", help="installed model dir for numeric validation")
    ap.add_argument("--ids", default="1,2,3", help="token ids for numeric validation")
    ap.add_argument("--base-set", default="", help="baseline kernel set for decode-gates")
    args = ap.parse_args()

    repo = args.repo_opt or args.repo
    if not repo and not args.gguf:
        ap.error("provide a repo (user/repo) or --gguf <file>")
    add = load_oflm_add()
    repo, quant = split_repo(repo, add)
    bases = base_models_for(repo, add)
    install_target = args.gguf if (args.gguf and not repo) else (args.repo_opt or args.repo or args.gguf)

    slug = slugify(args.repo or args.gguf or "model")[:64]
    if args.dry_run:
        tmp = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="oflm-new-family-"))
        tmp.mkdir(parents=True, exist_ok=True)
    else:
        tmp = Path(args.work) if args.work else ROOT / "build" / "new-family" / slug
        tmp.mkdir(parents=True, exist_ok=True)
    py, env = prepare_environment(args.dry_run)

    print("-- spec: geometry comes from the weights (GGUF first)", flush=True)
    derived = derive_spec(args, tmp, py, env, repo, add)

    facts: GGUFFacts | None = None
    research = ""
    recipe_name = ""
    recipe_reasons: list[str] = []
    recipe_ok, _recipe_reason = recipe_facts(derived.family)
    needs_family = (not recipe_ok) or (derived.status == "error" and not derived.family)
    if needs_family:
        # No recipe owns this architecture. Read the model's own weights, pick
        # the nearest existing recipe, build a candidate spec against it and go
        # straight into the normal BUILD/INSTALL path. No reservation, no wall.
        print("-- family: no recipe owns this architecture; reading the weight file's facts",
              flush=True)
        facts = gguf_facts(repo, args, add)
        if facts is None:
            print("   no GGUF header was readable; the derivation above is all the evidence there is",
                  file=sys.stderr)
        elif not facts.family:
            print("   the weight file did not name an architecture; pass --family to choose a recipe",
                  file=sys.stderr)
        else:
            if not derived.arch:
                derived.arch = facts.arch
            # Build the geometry first (under a placeholder family), then score
            # the recipes against it. The failed derive_spec left no data.
            data = candidate_spec(facts, args.family or facts.family or "dense")
            recipe_name, recipe_reasons = (
                (args.family, [f"--family {args.family} chosen explicitly"])
                if args.family else nearest_recipe(data or {}, facts))
            if data:
                data["family"] = recipe_name
                data.setdefault("extra", {})["nearest_recipe"] = recipe_name
                data.setdefault("extra", {})["model"] = (
                    args.model_name or (repo.rstrip("/").split("/")[-1] + "-NPU2"))
                p = tmp / "candidate-spec.json"
                p.write_text(json.dumps(data, indent=2), encoding="utf-8")
                derived = Derived(status="ok", path=p, data=data,
                                  family=recipe_name, arch=facts.arch,
                                  output=derived.output)
                print(f"   nearest recipe {recipe_name!r}: "
                      f"{'; '.join(recipe_reasons[:2]) or 'no geometry reason'}", flush=True)
                print(f"   candidate spec {display(p)} (from the GGUF's standard keys)",
                      flush=True)
            base = facts.base_model or (bases[0] if bases else "")
            research = str(write_research(tmp, facts, derived, base, install_target,
                                          recipe=recipe_name,
                                          recipe_reasons=recipe_reasons,
                                          built=not args.dry_run))
            print(f"   build report {display(Path(research))}", flush=True)
        probe = {"status": "skipped",
                 "detail": "not run: the architecture has no recipe; extending the nearest one"}
    else:
        print("-- probe: what oflm add would do", flush=True)
        probe = probe_install(args, install_target)

    plan = build_plan(args, repo, quant, bases, derived, probe, add,
                      facts=facts, research=research, recipe=recipe_name,
                      recipe_reasons=recipe_reasons, extending=bool(facts))
    print_report(plan)
    persist_plan(plan, args, tmp, derived)

    if args.dry_run:
        print("\n-- dry-run: no spec, kernel set or settings were changed")
        return 0
    if plan["status"] == "would_extend":
        print("\nunknown architecture: building against the nearest recipe. Verify the "
              "numeric gates before treating this as supported; if the arithmetic "
              "genuinely differs, follow .agents/skills/open-family-addition/SKILL.md",
              file=sys.stderr)
    set_name = ""
    if plan["status"] == "served":
        print("\nalready served: no new kernels to build; install with oflm add to link them.")
    else:
        if args.skip_build:
            print("\n-- skip-build: the spec was not promoted")
        else:
            name = plan["spec_name"]
            print(f"-- build: promote {name} and run build-new-spec.py", flush=True)
            promoted = promote_spec(derived, name, args.force)
            print(f"   spec {display(promoted)}", flush=True)
            rc, _ = run([py, BUILD_SPEC, "--spec", name], env=env, check=False)
            if rc != 0:
                # The recipe cannot build this shape: build-new-spec parked the
                # spec in recipes/candidates/ with the reason. That is evidence,
                # not a failed contribution -- report it and stop before an
                # install that would have no kernels to link.
                parked = ROOT / "open_kernels" / "recipes" / "candidates" / f"{name}.json"
                print(f"\n-- build wall: the recipe cannot build this shape. "
                      f"The spec was parked at {display(parked)}; "
                      f"see .agents/skills/open-family-addition/SKILL.md to extend the recipe.",
                      file=sys.stderr)
                return 0
            kdir, set_name, notes = find_exported_set(add, derived)
            if kdir:
                print(f"   kernel set: {set_name} ({kdir})", flush=True)
            for note in notes:
                print(f"   note: {note}", file=sys.stderr)

    if plan["status"] == "weights_blocked":
        print("\nno installable weights: the geometry matches, but every GGUF in the repo "
              "was refused by the installer (see its `skipped` lines above). No build, "
              "install or numeric gate can proceed until the weights are converted.",
              file=sys.stderr)
        return 1
    if not args.skip_install and install_target:
        oflm = os.environ.get("OFLM_EXECUTABLE") or shutil.which("oflm") or "oflm"
        rc = install_with_fallback(args, oflm, install_target, probe, env)
        if rc != 0:
            return rc
    if not args.skip_validate:
        validate(args, env, derived, set_name)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StageError as exc:
        print(f"build-new-family: {exc}", file=sys.stderr)
        raise SystemExit(1)
