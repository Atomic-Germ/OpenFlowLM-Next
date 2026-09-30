"""oflm_add - install a pre-converted OFLM (Q4NX) model and register it with OpenFlowLM.

Installable as ``oflm-add`` (e.g. ``uv tool install oflm-add``) or runnable as a
script (``python oflm_add/__init__.py`` or the legacy ``oflm-add.py`` shim).

Python-3 stdlib only (no pip packages). Works with any repo that
already contains the runtime-ready files (config.json, model.q4nx,
tokenizer.json, tokenizer_config.json, optionally chat_template.jinja):

    python3 oflm-add.py Atomic-Germ/Qwen3.5-9B-Claude-4.8-Opus-NPU2

Repo can be a Hugging Face repo id, a ModelScope repo id (--modelscope), a full
Hugging Face URL, a ModelScope URL (www.modelscope.ai/.cn -- implies ModelScope
without the flag), or a local directory holding the model files. The tag is
derived from the repo name (e.g. Qwen3.5-9B-Claude-4.8-Opus-NPU2 ->
qwen3.5-claude:9b); override with --tag. Defaults for the registry entry
(family, engine, size, context length) are copied from the matching official
OpenFlowLM entry.

Open kernels are handled separately. They belong to a model's *spec* (its
shape plus the per-role weight format), not to an official model name, so any
installed set whose manifest.json carries the same spec_hash as this model
drives it. oflm-add derives the spec from the installed config.json (+
tokenizer.json and the model.q4nx header) via the open_kernels/recipes
checkout, finds the matching set, and links it at <model dir>/open_kernels --
the second place open_qwen36::Engine::find_kernels looks.

The script never rewrites the system model list or the system xclbins; it
writes a user-level registry at ~/.config/oflm/model_list.json and adds a single
symlink into ~/.config/oflm/xclbins/ for the new model directory. Custom OFLM
models never ship xclbins (they are closed source), so the kernel symlink is
always taken from the matching official model, keyed by family (engine) and
size -- e.g. Darwin-36B-Opus-NPU2 -> Qwen3.6-35B-A3B-NPU2. OpenFlowLM discovers
both automatically -- it reads the user registry in preference to the shipped
one and resolves each model's kernels under whichever xclbin root carries that
model's directory -- so no environment variables are required.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path

REQUIRED_FILES = ["config.json", "model.q4nx", "tokenizer.json", "tokenizer_config.json"]
OPTIONAL_FILES = ["chat_template.jinja", "vision_weight.q4nx", "audio_weight.q4nx"]
ALL_FILES = REQUIRED_FILES + OPTIONAL_FILES

SYSTEM_LIST_CANDIDATES = [
    "/opt/openflowlm/share/oflm/model_list.json",
    "/usr/share/oflm/model_list.json",
    "/usr/local/share/oflm/model_list.json",
]

SYSTEM_XCLBIN_PREFIXES = [
    Path("/opt/openflowlm/share/oflm"),
    Path("/usr/share/oflm"),
    Path("/usr/local/share/oflm"),
]

# Dir-name prefix -> runtime details.family, used only when no official entry
# can be matched by name. The official model_list.json is the primary source.
FAMILY_ALIASES = [
    ("qwen3.5-omni", "qwen3.5-omni"),
    ("qwen3.6", "qwen3.6-moe"),
    ("qwen3.5-moe", "qwen3.6-moe"),
    ("qwen3.8", "qwen3.5"),      # Qwen3.8-Distilled-*: the Qwen3.5 engine, not Qwen3
    ("qwen3.5", "qwen3.5"),
    ("qwen3", "qwen3"),
    ("qwen2.5vl", "qwen2.5vl"),
    ("qwen2.5", "qwen2"),
    ("qwen2vl", "qwen2vl"),
    ("qwen2", "qwen2"),
    ("gemma4", "gemma4e"),
    ("gemma-4", "gemma4e"),
    ("gemma3", "gemma3"),
    ("llama3", "llama3"),
    ("llama", "llama3"),
    # Granite is its own family now (the dense recipe, head_dim 64 at hidden
    # 2560). It aliased onto llama3 because nothing served it; leaving that
    # would route a Granite directory to the Llama 3 AutoModel, whose sampler
    # defaults and chat-template probe are the wrong ones -- and the closed
    # llama_npu refuses hidden_size 2560 outright.
    ("granite", "granite"),
    ("crow", "qwen3.5"),
    ("huihui", "qwen3.5"),
    ("qwythos", "qwen3.5"),
    ("qwopus", "qwen3.5"),
    ("darwin", "qwen3.6-moe"),
    ("deepseek-r1-0528", "deepseek-r1-0528"),
    ("deepseek-r1", "deepseek-r1"),
    ("deepseek", "deepseek-r1"),
    ("nanbeige4", "nanbeige"),
    ("nanbeige", "nanbeige"),
    ("gpt-oss", "gpt-oss"),
    ("lfm2.5", "lfm2.5-tk"),
    ("lfm2", "lfm2"),
    ("phi4", "phi4"),
    ("whisper-v3", "whisper-v3"),
    ("whisper", "whisper-v3"),
    ("embed-gemma", "embed-gemma"),
]


def log(msg):
    print(msg, file=sys.stderr)


def err(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _engine_dirs():
    """Directories holding an installed engine. The release installs as `flm`;
    `oflm` is a checkout build, so it goes first where both are on PATH, and an
    explicit OFLM_EXECUTABLE goes before either."""
    out = []
    for exe in (os.environ.get("OFLM_EXECUTABLE"), shutil.which("oflm"), shutil.which("flm")):
        if exe:
            d = Path(exe).parent
            if d not in out:
                out.append(d)
    return out


def find_system_model_list(explicit=None):
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise SystemExit(f"--system-list {p} is not a file")
        return p
    candidates = []
    for d in _engine_dirs():
        candidates.append(d / "model_list.json")
        candidates.append((d / ".." / "share" / "oflm" / "model_list.json").resolve())
    candidates += [Path(p) for p in SYSTEM_LIST_CANDIDATES]
    for c in candidates:
        if c.is_file():
            return c
    tried = "\n  ".join(str(c) for c in candidates) or "(nowhere - no engine on PATH)"
    raise SystemExit(
        "Could not locate the system model_list.json. Tried:\n  " + tried +
        "\nPass --system-list with the path to it."
    )


def find_system_xclbin_root():
    """Directory whose <root>/xclbins/ holds the per-model kernel folders."""
    candidates = []
    for d in _engine_dirs():
        candidates.append(d)
        candidates.append((d / ".." / "share" / "oflm").resolve())
    candidates += SYSTEM_XCLBIN_PREFIXES
    for c in candidates:
        if (c / "xclbins").is_dir():
            return c / "xclbins"
    return None


def user_xclbin_dir(arg):
    """Resolve the user-level xclbins directory (where symlinks are added)."""
    if arg:
        base = Path(arg)
    else:
        env = os.environ.get("OFLM_XCLBIN_PATH")
        base = Path(env) if env else Path.home() / ".config" / "oflm"
    return base if base.name == "xclbins" else base / "xclbins"


def user_registry_path(arg):
    if arg:
        return Path(arg)
    env = os.environ.get("OFLM_CONFIG_PATH")
    if env:
        return Path(env)
    return Path.home() / ".config" / "oflm" / "model_list.json"


def models_root_dir(arg):
    if arg:
        return Path(arg)
    env = os.environ.get("OFLM_MODEL_PATH")
    if env:
        return Path(env) / "models"
    return Path.home() / ".config" / "oflm" / "models"


# ---------------------------------------------------------------- tag derivation

def _strip_npu2(name):
    return re.sub(r"-NPU2$", "", name, flags=re.IGNORECASE)


def _extract_size(bare):
    # Trailing size groups like "-A3B" (Qwen3.6-35B-A3B) end in a letter that
    # the digit group must not swallow (previously "-A3B" -> "35b-a3").
    m = re.search(r"(\d+(?:\.\d+)?[Bb](?:-[A-Za-z]+\d+(?:\.\d+)?[A-Za-z]*)*)", bare)
    if not m:
        return None, bare
    size = m.group(1).lower()
    rest = (bare[: m.start()] + " " + bare[m.end():]).strip()
    return size, rest


def derive_tag(dir_name, explicit=None):
    if explicit:
        return explicit
    size, rest = _extract_size(_strip_npu2(dir_name))
    if not size:
        raise SystemExit(
            f"Could not derive a size from '{dir_name}' (no 'NNb' marker). "
            "Pass --tag name:size."
        )
    tokens = [t for t in re.split(r"[-_ ]+", rest) if t]
    if not tokens:
        raise SystemExit("Could not derive a tag from the repo name. Pass --tag name:size.")
    family = tokens[0].lower()
    variant = None
    for t in tokens[1:]:
        if re.fullmatch(r"\d+(\.\d+)?[MmKk]?", t):
            continue
        variant = t.lower()
        break
    return f"{family}-{variant}:{size}" if variant else f"{family}:{size}"


def match_official_entry(system_registry, dir_name):
    """Official entry whose directory name shares the longest token prefix."""
    best = None
    dir_tokens = re.split(r"[-_ ]+", dir_name)
    for bucket, sizes in system_registry.get("models", {}).items():
        for size, info in sizes.items():
            name = info.get("name")
            if not name:
                continue
            common = 0
            for x, y in zip(re.split(r"[-_ ]+", name), dir_tokens):
                if x.lower() != y.lower():
                    break
                common += 1
            if common >= 2 and (best is None or common > best[0]):
                best = (common, bucket, size, info)
    return best


def _official_entries(system_registry, family):
    return [
        (bucket, sz, info)
        for bucket, sizes in system_registry.get("models", {}).items()
        for sz, info in sizes.items()
        if (info.get("details") or {}).get("family") == family
    ]


def match_official_by_family_size(system_registry, family, size):
    """Official entry matching details.family and registry size (bytes).

    Used for repos that share an engine with an official model but not a
    name prefix (e.g. Huihui-Qwythos-9B-... -> qwen3.5 + 9B -> Qwen3.5-9B-NPU2).
    """
    if not family or not size:
        return None
    for bucket, sz, info in _official_entries(system_registry, family):
        if info.get("size") == size:
            return (0, bucket, sz, info)
    return None


def resolve_official(system_registry, dir_name, family, size):
    """Pick the official model that supplies the xclbins for this install.

    Custom OFLM models never ship xclbins (closed source), so the kernels must
    be linked from the matching official model, keyed by family (engine) and
    size. Returns (official_4tuple, note) where note explains any size
    mismatch, or (None, None) when no official model matches.
    """
    official = match_official_entry(system_registry, dir_name)
    if official:
        return official, None
    official = match_official_by_family_size(system_registry, family, size)
    if official:
        return official, None
    entries = _official_entries(system_registry, family)
    if len(entries) == 1:
        bucket, sz, info = entries[0]
        note = None
        if size:
            official_size = info.get("size", 0)
            if official_size and official_size != size:
                note = f"tag size {size/1e9:g}B differs from official {official_size/1e9:g}B"
        return (0, bucket, sz, info), note
    if entries and size:
        best = min(entries, key=lambda e: abs(e[2].get("size", 0) - size))
        bucket, sz, info = best
        return (0, bucket, sz, info), (
            f"no exact size match for {size/1e9:g}B; using {info.get('size', 0)/1e9:g}B kernels"
        )
    return None, None


# A tag that names the family outright, longest first so a specific tag is
# preferred over a general one. These are the tags Atomic-Germ's own conversions
# carry, and they are what let a finetune resolve: a directory named
# `Ornith-1.5-9B-NPU2` says nothing about its architecture, but the repo's README
# frontmatter is tagged `qwen3.5`, and every Qwen3.5-9B finetune wants the same
# kernel set.
_FRONT_FAMILY_TAGS = [
    ("qwen3.6-moe", "qwen3.6-moe"), ("qwen3.6", "qwen3.6-moe"),
    ("qwen3_5_moe", "qwen3.6-moe"), ("qwen3.5-moe", "qwen3.6-moe"),
    ("qwen35moe", "qwen3.6-moe"),
    ("qwen3.8", "qwen3.5"),          # the Qwen3.5 engine, per FAMILY_ALIASES
    ("qwen3.5", "qwen3.5"), ("qwen3_5", "qwen3.5"), ("qwen35", "qwen3.5"),
    ("qwen3.5-omni", "qwen3.5-omni"),
    ("qwen3vl", "qwen3vl"), ("qwen3_vl", "qwen3vl"),
    ("qwen3", "qwen3"), ("qwen2.5vl", "qwen2.5vl"), ("qwen2.5", "qwen2"),
    ("qwen2vl", "qwen2vl"), ("qwen2", "qwen2"),
    ("gemma4e", "gemma4e"), ("gemma4", "gemma4"), ("gemma-4", "gemma4"),
    ("gemma3", "gemma3"), ("gemma-3", "gemma3"),
    ("gpt-oss", "gpt-oss"), ("gpt_oss", "gpt-oss"),
    ("granite", "granite"), ("llama3", "llama3"), ("llama-3", "llama3"),
    ("llama", "llama3"), ("lfm2", "lfm2"), ("lfm2.5", "lfm2"),
    ("hunyuan", "hunyuan"), ("phi4", "phi4"), ("phi-4", "phi4"), ("phi3", "phi3"),
    ("nanbeige", "nanbeige"), ("crow", "qwen3.5"),
]

# Tags that mark a repo as one OFLM can serve at all. Absent on a repo that is
# not an NPU conversion, which is worth saying rather than failing later.
_FRONT_IS_OF = ("npu2", "q4nx", "fastflowlm", "flm", "fastflow")

# The quant format a conversion wrote, when the tag says so. A q4nx-build config
# remains the authority; this only CROSS-CHECKS it, and a disagreement is
# reported rather than silently preferring one.
_FRONT_QUANT_TAGS = {"mxfp4": "mxfp4", "q8": "q8", "q8_0": "q8",
                     "q4_1": "q4_1", "q4_0": "q4_1", "q4_k": "q4_k"}


def _read_frontmatter(repo, timeout=20.0):
    """The YAML frontmatter of a HuggingFace repo's README, as a dict.

    This is the most overlooked metadata a conversion carries. `config.json`
    gives the geometry, but it does not say "this is a Q4NX container for an
    AMD NPU" or which family a finetune belongs to -- and the finetune's NAME is
    arbitrary (`Ornith-1.5-9B-NPU2`), so the tags are the only thing that ties
    it to a kernel set. A one-line failure here is not fatal: family resolution
    falls through to the name, exactly as before.
    """
    for url in (f"https://huggingface.co/{repo}/resolve/main/README.md",
                f"https://huggingface.co/{repo}/raw/main/README.md"):
        try:
            req = urllib.request.Request(url, headers=_hf_headers())
            with urllib.request.urlopen(req, timeout=timeout) as r:
                text = r.read(200_000).decode("utf-8", "replace")
        except Exception:
            continue
        if not text.startswith("---"):
            return {}
        end = text.find("\n---", 3)
        if end < 0:
            return {}
        block = text[3:end]
        out, key = {}, None
        for line in block.splitlines():
            if not line.strip():
                continue
            if line[0] not in " \t-":                 # a new key
                if ":" not in line:
                    continue
                key, _, val = line.partition(":")
                key, val = key.strip(), val.strip()
                out[key] = val if val and val not in ("|", ">") else []
            elif key is not None and line.lstrip().startswith("- "):
                item = line.lstrip()[2:].strip().strip("'\"")
                if isinstance(out.get(key), list):
                    out[key].append(item)
        return out
    return {}


# `config.json`'s `model_type` -> the family string, the same mapping the recipes
# use (recipes/spec.py HF_FAMILIES / _FAMILY_OF). Duplicated rather than imported
# because oflm-add has to work with no source checkout: the recipes are only
# present if this is the repo or an install that ships them, and the family has
# to be knowable before anything else is set up.
_FAMILY_OF_MODEL_TYPE = {
    "qwen3_5_moe": "qwen3.6-moe", "qwen3_5_moe_text": "qwen3.6-moe",
    "qwen3_next": "qwen3.6-moe", "qwen3_5": "qwen3.5", "qwen3_5_text": "qwen3.5",
    "qwen3": "qwen3", "qwen3_vl": "qwen3vl", "qwen3_vl_text": "qwen3vl",
    "qwen2": "qwen2", "qwen2_5_vl": "qwen2.5vl", "qwen2_5_vl_text": "qwen2.5vl",
    # HF publishes Llama 3 as `llama`, not `llama3` -- read off the repos this
    # tree ships, and the reason a Llama finetune that was tagged correctly
    # still resolved to nothing when the tags were the only source.
    "llama": "llama3", "llama2": "llama3",
    "gemma3": "gemma3", "gemma3_text": "gemma3", "gemma3_text_only": "gemma3",
    "gemma4_text": "gemma4", "gemma4": "gemma4",
    "hunyuan_v1_dense": "hunyuan", "granite": "granite", "phi3": "phi3",
    "phi4": "phi4", "lfm2": "lfm2", "gpt_oss": "gpt-oss",
}


def _tags_of(front):
    """The tag list of a README's frontmatter, lowercased."""
    tags = front.get("tags")
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    return [str(t).lower() for t in tags] if isinstance(tags, list) else []


def declared_kernels(front):
    """The `oflm-kernels` value a repo publishes, or None.

    A conversion can declare the exact set it was built and tested against:
    `oflm-kernels: sha256:602fa183...` in the README frontmatter. That is the
    strongest possible statement -- "this container, with this layout, wants
    exactly these kernels" -- and it makes linking a lookup rather than a
    derivation, which matters for a model whose geometry derives fine but whose
    weights were laid out differently.

    OPTIONAL. Most repos in the wild will not carry it, and must not be required
    to: the tags and `config.json` are enough for the overwhelming majority, and
    the point of this is to be better on the models we publish, not to make an
    undeclared tag a hard error for someone else's.
    """
    for tag in _tags_of(front):
        if tag.startswith("oflm-kernels:"):
            v = tag.split(":", 1)[1].strip()
            return v or None
    v = front.get("oflm-kernels")
    return str(v).strip() if v else None


def declared_family(front):
    """The `oflm-family` value a repo publishes, or None.

    The same idea as `oflm-kernels`, for the family: an explicit
    `oflm-family: qwen3.5` outranks anything inferred from a tag or a name,
    because it was written by whoever made the conversion and they know. It is
    honoured for models we did not create too -- a finetune author can add the
    one tag and have `oflm add` do the right thing without a release from us.
    """
    for tag in _tags_of(front):
        if tag.startswith("oflm-family:"):
            v = tag.split(":", 1)[1].strip()
            if v:
                return v
    v = front.get("oflm-family")
    return str(v).strip() if v else None


def issue_search_url(model_name):
    """A GitHub issue search pre-filled with the model name.

    The old answer to "no kernels for this model" was a sentence telling the
    user to run a Python script from a source checkout several directories
    deep, which is a thing they cannot do from an installed package and do not
    want to anyway. A link that opens an issue already naming their model is
    the request we actually want, and it costs them one click.
    """
    from urllib.parse import quote
    return ("https://github.com/Atomic-Germ/OpenFlowLM-Next/issues"
            f"?q=is%3Aissue+is%3Aopen+no+open+kernels+{quote(model_name)}")


def frontmatter_family(front, model_type=None):
    """A best-effort family from a README's tags, or None.

    NOT the authority. `config.json`'s `model_type` is -- the ModelSpec is derived
    from it, and it is always present on a model OFLM can serve. These tags only
    name which q4nx-build config to read, so the cost of being wrong is a
    fallback, not a wrong kernel set: `model_spec_hash` re-derives from
    config.json and falls back to the container header if this is off.

    So this is deliberately forgiving, in the direction of ANSWERING:

      * a finetune carries its whole ancestry in its tags -- `Ornith-1.5-9B` is
        tagged `qwen`, `qwen3` AND `qwen3.5`, where the last is the narrowest
        and is what it actually IS. Ancestry is not a conflict, so the most
        specific tag wins rather than the most common one;
      * a tag naming a variant (`qwen3.5-omni`, `qwen3vl`, `gemma4e`) beats the
        line it belongs to (`qwen3.5`, `qwen3`, `gemma4`), because that is the
        narrower statement about which engine;
      * `model_type`, when the caller has it, settles everything. A tag that
        disagrees with the config is a stale tag, and the config is right.

    Only returns None when there is genuinely nothing to go on, which leaves the
    caller on the name-based path it always had.
    """
    tags = front.get("tags")
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    tags = [str(t).lower() for t in tags] if isinstance(tags, list) else []

    # The config speaks first, when the caller has already read it.
    if model_type:
        want = _FAMILY_OF_MODEL_TYPE.get(str(model_type).lower())
        if want:
            return want

    best: tuple[int, str] | None = None
    for tag in tags:
        for needle, family in _FRONT_FAMILY_TAGS:
            if not (tag == needle or tag.startswith(needle + "-") or tag.startswith(needle + ".")):
                continue
            # A variant tag (a separator inside the needle) is the narrower claim
            # and outranks the bare line; otherwise the longer name is narrower.
            rank = (1 if any(c in needle for c in ".-") else 0, len(needle))
            if best is None or rank > best[0]:
                best = (rank, family)
    return best[1] if best else None


def derive_family(system_registry, dir_name, explicit=None, base_entry=None, front=None):
    if explicit:
        return explicit
    if base_entry:
        fam = base_entry.get("details", {}).get("family")
        if fam:
            return fam
    # The frontmatter before the name: a finetune's directory name is arbitrary,
    # and the tags on its repo are what say which engine it needs. `Ornith-1.5-9B`
    # has no prefix FAMILY_ALIASES recognises; its repo is tagged `qwen3.5`.
    if front:
        fam = frontmatter_family(front)
        if fam:
            return fam
    lower = dir_name.lower()
    for prefix, family in FAMILY_ALIASES:
        if lower.startswith(prefix.lower()):
            return family
    raise SystemExit(
        f"Could not determine details.family for '{dir_name}'. "
        "Pass --family (e.g. qwen3.5, qwen3.6-moe, nanbeige, llama3, ...)."
    )


# ------------------------------------------------------------------- asset fetch

def _hf_headers():
    headers = {"User-Agent": "oflm-add/1.0"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _ms_headers():
    # Never forward Hugging Face credentials to ModelScope hosts.
    return {"User-Agent": "oflm-add/1.0"}


def _http_get_json(url, headers=None):
    req = urllib.request.Request(url, headers=headers if headers is not None else _hf_headers())
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


MODELSCOPE_HOSTS = ("modelscope.ai", "modelscope.cn", "modelscope.com")
# Repos live on either hub (the international .ai site and the original .cn
# site are separate registries), so query both before giving up.
MS_DOMAINS = ["modelscope.ai", "modelscope.cn"]


def is_modelscope_host(host):
    host = host.lower()
    return any(host == h or host.endswith("." + h) for h in MODELSCOPE_HOSTS)


URL_PATH_CUTS = ("resolve", "blob", "tree", "commit", "files", "discuss")


def split_remote_repo(raw):
    """Classify an http(s) model URL: returns (host_kind, "Org/Name").

    host_kind is 'modelscope' or 'huggingface'; a ModelScope URL therefore
    implies ModelScope without --modelscope. Bare Org/Name arguments are not
    URLs and must be classified by the caller (default: Hugging Face).
    """
    if not raw.startswith(("https://", "http://")):
        return "huggingface", raw
    host, _, path = raw.split("://", 1)[1].partition("/")
    segs = [s for s in path.split("/") if s]
    for cut in URL_PATH_CUTS:
        if cut in segs:
            segs = segs[: segs.index(cut)]
    if segs and segs[0] == "models":
        segs = segs[1:]
    kind = "modelscope" if is_modelscope_host(host) else "huggingface"
    return kind, "/".join(segs[:2])


def hf_file_tree(repo_id):
    return _http_get_json(f"https://huggingface.co/api/models/{repo_id}/tree/main?recursive=true")


def ms_file_tree(repo_id):
    """Root-level file listing from ModelScope.

    Returns (domain, {name: meta}) where meta carries Size/Sha256 for download
    verification. Tries each known hub domain; raises SystemExit when the repo
    is found on none of them.
    """
    errors = []
    for domain in MS_DOMAINS:
        url = (
            f"https://{domain}/api/v1/models/{repo_id}"
            "/repo/files?Revision=master&Recursive=false"
        )
        try:
            tree = _http_get_json(url, headers=_ms_headers())
        except Exception as e:
            errors.append(f"{domain}: {e}")
            continue
        files = ((tree.get("Data") or {}).get("Files")) or []
        if tree.get("Code") == 200 and files:
            return domain, {f["Path"]: f for f in files if f.get("Path")}
        errors.append(f"{domain}: {tree.get('Message') or 'not found'}")
    raise SystemExit(
        f"ModelScope repo not found ({repo_id}). Tried: " + "; ".join(errors)
    )


def hf_cache_snapshot(repo_id):
    roots = []
    for env in ("HF_HUB_CACHE", "HF_HOME"):
        if os.environ.get(env):
            p = Path(os.environ[env])
            roots.append(p if p.name == "hub" else p / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    repo_dir_name = "models--" + repo_id.replace("/", "--")
    for root in roots:
        snapshots = root / repo_dir_name / "snapshots"
        if not snapshots.is_dir():
            continue
        for ref in (root / repo_dir_name / "refs").glob("*"):
            try:
                rev = ref.read_text().strip()
            except Exception:
                continue
            d = snapshots / rev
            if d.is_dir():
                return d
        first = next((d for d in snapshots.iterdir() if d.is_dir()), None)
        if first:
            return first
    return None


def ms_cache_snapshot(repo_id):
    """Local ModelScope SDK cache dir holding this repo's files (or None).

    The SDK stores plain files directly under <cache>/models/<org>/<name>
    (newer releases) or <cache>/<org>/<name> (legacy), so unlike the HF flow
    there are no snapshot/blob indirections to resolve.
    """
    org, _, name = repo_id.partition("/")
    roots = []
    env = os.environ.get("MODELSCOPE_CACHE")
    if env:
        roots.append(Path(env))
    roots.append(Path.home() / ".cache" / "modelscope")
    candidates = []
    for root in roots:
        base = root if root.name == "modelscope" else root
        candidates += [base / "models", base]
    for base in candidates:
        d = base / org / name
        if (d / "config.json").is_file():
            return d
    return None


def download_file(url, dest, expected_size=None, expected_sha=None, verify=True, quiet=False,
                  headers=None):
    req = urllib.request.Request(url, headers=headers if headers is not None else _hf_headers())
    tmp = str(dest) + ".part"
    written = 0
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        length = int(resp.headers.get("Content-Length") or 0)
        total = expected_size or length or 0
        last_pct = -1
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            written += len(chunk)
            if total and not quiet:
                pct = int(written * 100 / total)
                if pct != last_pct and pct % 5 == 0:
                    log(f"    {pct:3d}% ({written/1e9:.2f} GB / {total/1e9:.2f} GB)")
                    last_pct = pct
    if expected_size and written != expected_size:
        os.unlink(tmp)
        raise SystemExit(f"Size mismatch for {dest.name}: got {written}, expected {expected_size}")
    if expected_sha and verify:
        h = hashlib.sha256()
        with open(tmp, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        if h.hexdigest() != expected_sha:
            os.unlink(tmp)
            raise SystemExit(f"sha256 mismatch for {dest.name}")
    os.replace(tmp, dest)


def fetch_assets(repo_id, target, modelscope=False, verify=True, force=False, quiet=False):
    """Populate target/ with the model files; returns the list of files present."""
    obtained = []
    target.mkdir(parents=True, exist_ok=True)
    if modelscope:
        domain, entries = ms_file_tree(repo_id)
        for fname in ALL_FILES:
            if fname not in entries:
                continue
            dest = target / fname
            if dest.is_file() and not force:
                obtained.append(fname)
                continue
            meta = entries[fname]
            expected_size = meta.get("Size") or None
            expected_sha = (meta.get("Sha256") or "").lower() or None
            if not quiet:
                gb = f" ({expected_size / 1e9:.2f} GB)" if expected_size else ""
                log(f"Downloading {fname}{gb} from ModelScope ({domain})...")
            download_file(
                f"https://{domain}/models/{repo_id}/resolve/master/{fname}",
                dest,
                expected_size=expected_size,
                expected_sha=expected_sha,
                verify=verify,
                quiet=quiet,
                headers=_ms_headers(),
            )
            obtained.append(fname)
        return obtained

    # Hugging Face: local cache first, then the tree API.
    entries = {}
    for e in hf_file_tree(repo_id):
        p = e.get("path")
        if p and "/" not in p:
            entries[p] = e
    for fname in ALL_FILES:
        if fname not in entries:
            continue
        dest = target / fname
        if dest.is_file() and not force:
            obtained.append(fname)
            continue
        lfs = entries[fname].get("lfs") or {}
        expected_sha = lfs.get("oid")
        expected_size = lfs.get("size") or entries[fname].get("size")
        log(f"Downloading {fname} ({expected_size/1e9:.2f} GB)...")
        download_file(
            f"https://huggingface.co/{repo_id}/resolve/main/{fname}",
            dest,
            expected_size=expected_size,
            expected_sha=expected_sha,
            verify=verify,
            quiet=quiet,
        )
        obtained.append(fname)
    return obtained


def copy_from_dir(src_dir, target, force=False):
    obtained = []
    for fname in ALL_FILES:
        src = src_dir / fname
        if src.is_file():
            dest = target / fname
            if dest.is_file() and not force:
                obtained.append(fname)
                continue
            shutil.copy2(src, dest)
            obtained.append(fname)
    return obtained


# ---------------------------------------------------------------- registry

def size_from_tag(tag):
    """Registry 'size' (bytes) from the tag size marker, e.g. '3b' -> 3000000000,
    '9b-claude-4.8' -> 9000000000, '0.8b' -> 800000000."""
    m = re.match(r".*:(\d+(?:\.\d+)?)b\b", tag, flags=re.IGNORECASE)
    if not m:
        return None
    return int(float(m.group(1)) * 1_000_000_000)


def estimate_size(config_path):
    try:
        cfg = load_json(config_path)
    except Exception:
        return None
    hidden = cfg.get("hidden_size")
    layers = cfg.get("num_hidden_layers")
    if not hidden or not layers:
        return None
    intermediate = cfg.get("intermediate_size")
    per_layer = 12 * hidden * hidden
    if intermediate:
        per_layer += 3 * hidden * intermediate
    total = per_layer * layers + 2 * hidden * (cfg.get("vocab_size") or hidden)
    return max(int(round(total / 1e9 * 2) / 2 * 1e9), 1_000_000_000)


def build_entry(base_entry, dir_name, files, size):
    entry = dict(base_entry) if base_entry else {}
    entry["name"] = dir_name
    entry["files"] = list(files)
    entry["url"] = ""
    entry["file_url"] = ""
    entry["ms_url"] = ""
    entry.setdefault("max_prefill_len", 4096)
    entry.setdefault("default_context_length", 8192)
    entry.setdefault("oflm_min_version", "0.9.45")
    entry.setdefault("details", {}).setdefault("format", "NPU2")
    if size:
        entry["size"] = size
    entry["vlm"] = any(f.startswith("vision") for f in files)
    return entry


def register(user_list_path, tag, entry, system_registry):
    if user_list_path.is_file():
        registry = load_json(user_list_path)
    else:
        registry = json.loads(json.dumps(system_registry))
    registry.setdefault("model_path", "models")
    model_type, size = tag.split(":", 1)
    registry.setdefault("models", {}).setdefault(model_type, {})[size] = entry
    save_json(user_list_path, registry)


# ------------------------------------------------------------------- xclbins

def link_xclbins(system_root, user_root, dir_name, source_name, force=False, quiet=False):
    if not source_name:
        if not quiet:
            log("[WARN] No xclbin source; skipping symlink. Pass --xclbin-from NAME to link an official model's kernels.")
        return
    src = system_root / source_name
    if not src.is_dir():
        if not quiet:
            log(f"[WARN] Official model has no xclbins directory: {source_name}")
        return
    user_root.mkdir(parents=True, exist_ok=True)
    link = user_root / dir_name
    target = str(src)
    if link.exists() or link.is_symlink():
        # resolve() covers a junction too, which does not answer to readlink
        if link.exists() and link.resolve() == src.resolve():
            if not quiet:
                log(f"[INFO] xclbins link already in place: {link}")
            return
        if link.is_symlink():
            link.unlink()
        elif force:
            shutil.rmtree(link)
        else:
            raise SystemExit(
                f"{link} already exists and is not a symlink. Remove it or pass --force."
            )
    if not _make_dir_link(link, src):
        raise SystemExit(
            f"Could not link {link} -> {target}. Windows grants the symlink privilege to "
            f"admins and developer mode only, and the junction fallback failed too."
        )
    if not quiet:
        log(f"[INFO] Linked xclbins: {link} -> {target}")


# -------------------------------------------------------------- open kernels
#
# Closed kernels are per official model; open kernel sets are per ModelSpec --
# the shape plus the per-role weight format. Two shape-identical models (a
# fine-tune, a distill) share one set. The engine finds a set at
# OFLM_OPEN_KERNELS_DIR, then <model dir>/open_kernels, then
# <xclbins root>/<model name>/open_kernels (open_qwen36::Engine::find_kernels).
# We link into the model directory: it is the one root that does not depend on
# how OFLM_XCLBIN_PATH happens to be set.

def open_kernels_checkout():
    """An `open_kernels/` directory holding recipes/spec.py, or None.

    Looked for at $OPEN_KERNELS_DIR, then next to this checkout (oflm-add lives
    in <repo>/utilities/oflm-add), then under the working directory, then -- for
    an INSTALLED oflm, which is the case that matters to a user -- beside the
    recipes in the install tree. The package ships `recipes/` (it has to: the
    ModelSpec derivation is what links the kernels), so a user who installed the
    RPM finds one at <prefix>/share/oflm/open_kernels with no source checkout and
    no environment variable set. That is what makes `oflm add` link a kernel set
    on a machine that has only the package.
    """
    candidates = []
    env = os.environ.get("OPEN_KERNELS_DIR")
    if env:
        candidates.append(Path(env))
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "open_kernels")
    candidates.append(Path.cwd() / "open_kernels")
    # Installed layout: <prefix>/share/oflm/utilities/oflm-add/oflm_add/ here, so
    # <prefix>/share/oflm is two levels up from the package's grandparent.
    here = Path(__file__).resolve()
    share = here.parents[2]                      # <prefix>/share/oflm
    if share.name:
        candidates.append(share / "open_kernels")
    for c in candidates:
        if (c / "recipes" / "spec.py").is_file():
            return c
    return None


def model_spec_hash(model_dir, cat_family=None, size=None):
    """(spec_hash, note) for an installed model directory; (None, why) on failure.

    The quant map -- the part of the hash that says which projections are q8 --
    comes from the q4nx-build config for this family and size, which is the same
    authority the kernel build reads, so the two hashes agree by construction.
    Reading it out of the container instead needed a table of chunk widths
    (`CHUNK_FORMAT`) that only listed the models someone had happened to try, so
    a model at a new width was refused with a message about quant chunk bytes --
    Gemma3-1B among them. The container is no longer consulted when the family
    and size are known; it remains the fallback for a model the catalogue does
    not name, where refusing beats guessing.
    """
    root = open_kernels_checkout()
    if root is None:
        return None, "no open_kernels/recipes found (set OPEN_KERNELS_DIR)"
    added = str(root)
    inserted = added not in sys.path
    if inserted:
        sys.path.insert(0, added)
    try:
        from recipes.load import spec_from_hf_config, spec_from_model_dir, tokenizer_vocab
        from recipes.spec import SpecError
        md = Path(model_dir)
        # Preferred: the q4nx-build config, so this hash equals the kernel build's
        # by construction rather than by coincidence. Needs only config.json and
        # the tokenizer -- both of which an installed model has.
        if cat_family:
            try:
                cfg = json.loads((md / "config.json").read_text(encoding="utf-8"))
                spec = spec_from_hf_config(cfg, tokenizer_vocab(md / "tokenizer.json"),
                                           cat_family, size)
                return spec.spec_hash(), f"spec from {root} + q4nx config {cat_family}/{size}"
            except SpecError:
                pass            # no config for this family: fall through to the container
        # Fallback: a model the catalogue does not name, where the container header
        # is the only statement of its format. It can refuse, and refusing is right
        # -- a guessed format is a wrong hash, which links mismatched kernels.
        spec = spec_from_model_dir(md)
        return spec.spec_hash(), f"spec from {root} (container header)"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    finally:
        if inserted:
            try:
                sys.path.remove(added)
            except ValueError:
                pass


def kernel_set_spec_hash(kernel_dir):
    """The manifest's spec_hash for a kernel set directory, or None."""
    try:
        return load_json(Path(kernel_dir) / "manifest.json").get("spec_hash")
    except Exception:
        return None


def find_open_kernels(spec_hash, roots, dir_name):
    """The installed open kernel set whose manifest matches `spec_hash`.

    Roots are xclbins directories (<root>/<model>/open_kernels). A set sitting
    under this model's own name wins; otherwise the first match in root order.
    Returns (Path, source model name) or (None, None).
    """
    if not spec_hash:
        return None, None
    matches = []
    for root in roots:
        if not root or not Path(root).is_dir():
            continue
        for model in sorted(Path(root).iterdir()):
            k = model / "open_kernels"
            if not (k / "manifest.json").is_file():
                continue
            if kernel_set_spec_hash(k) == spec_hash:
                matches.append((k, model.name))
    for k, name in matches:
        if name == dir_name:
            return k, name
    return matches[0] if matches else (None, None)


def _make_dir_link(link, target):
    """Symlink `link` -> `target`; a junction where symlinks need a privilege
    Windows only grants to admins / developer mode. True on success."""
    try:
        os.symlink(str(target), str(link), target_is_directory=True)
        return True
    except (OSError, NotImplementedError, AttributeError):
        pass
    try:
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
        return True
    except Exception:
        return False


def link_open_kernels(model_dir, kernel_dir, force=False, quiet=False):
    """Put the chosen kernel set where find_kernels looks for this model.

    True when <model dir>/open_kernels resolves to `kernel_dir`.
    """
    kernel_dir = Path(kernel_dir).resolve()
    link = Path(model_dir) / "open_kernels"
    if link.exists() and link.resolve() == kernel_dir:
        if not quiet:
            log(f"[INFO] open kernels already in place: {link}")
        return True
    if link.is_symlink() or link.exists():
        if not (force or link.is_symlink()):
            raise SystemExit(
                f"{link} already exists and is not a link. Remove it or pass --force."
            )
        try:
            if link.is_symlink() or link.is_file():
                link.unlink()
            else:
                shutil.rmtree(link)
        except OSError as e:
            log(f"[WARN] could not replace {link}: {e}")
            return False
    if _make_dir_link(link, kernel_dir):
        if not quiet:
            log(f"[INFO] Linked open kernels: {link} -> {kernel_dir}")
        return True
    log(f"[WARN] Could not link {link} -> {kernel_dir} (a Windows symlink needs "
        "developer mode or admin). Run oflm with:")
    log(f'           OFLM_OPEN_KERNELS_DIR="{kernel_dir}"')
    return False


def setup_open_kernels(model_dir, dir_name, roots, override=None, force=False, quiet=False,
                       cat_family=None, size=None, declared_hash=None):
    """Find and link the open kernel set for this model; say which and why."""
    if override:
        kernel_dir = Path(override)
        if not (kernel_dir / "manifest.json").is_file():
            raise SystemExit(f"--open-kernels {kernel_dir} has no manifest.json")
        log(f"[INFO] open kernels: {kernel_dir} (--open-kernels)")
        return link_open_kernels(model_dir, kernel_dir, force=force, quiet=quiet)

    # A repo that published `oflm-kernels` states the set it was built against,
    # so that is what to look for -- a declared hash beats a derived one, and a
    # mismatch between them is worth saying out loud, because it means the
    # container is not the one the kernel set was made for.
    spec_hash, note = model_spec_hash(model_dir, cat_family, size)
    if declared_hash and spec_hash and declared_hash != spec_hash:
        log(f"[INFO] This repo declares oflm-kernels {declared_hash[:19]} but its "
            f"config.json derives {spec_hash[:19]}. The two disagree, so the "
            f"container is not the one that set was built for; using the derived "
            f"one, and it may not match. This is worth an issue: "
            f"{issue_search_url(dir_name)}")
    elif declared_hash and not spec_hash:
        spec_hash, note = declared_hash, "declared by the repo's oflm-kernels tag"
    if not spec_hash:
        if not quiet:
            log(f"[INFO] No open-kernel spec for this model ({note}); closed kernels only.")
        return False
    kernel_dir, source = find_open_kernels(spec_hash, roots, dir_name)
    if kernel_dir:
        log(f"[INFO] open kernels from '{source}': its manifest spec_hash matches "
            f"this model's ({spec_hash[:19]})")
        return link_open_kernels(model_dir, kernel_dir, force=force, quiet=quiet)
    # No set is installed for this model. Say so in one line and offer the one
    # thing the user can actually do -- open an issue naming their model. The
    # previous message told them to run a Python script from a source checkout
    # several directories deep, which they cannot do from an installed package
    # and would not want to: kernels are built by the distribution, not by the
    # person installing it.
    log(f"[INFO] No installed open kernel set matches this model "
        f"({spec_hash[:19]}); it will run on the closed kernels, which are shipped. "
        f"To get open kernels for it, open an issue -- this link has the model "
        f"named already:\n           {issue_search_url(dir_name)}")
    return False


# ---------------------------------------------------------------------- main


# --------------------------------------------------------------- vision.json
#
# Some containers ship a vision tower and none of the numbers needed to read it.
# Qwen3-VL-4B-Instruct-NPU2 is the one that found this: no `vision_config`, no
# `image_token_id`, no `mrope_section`. Most of the tower's geometry falls out of
# the weight file's own tensor shapes, but two numbers never do -- the attention
# head count (qkv is [3 * hidden, hidden] at any split) and which blocks the
# deepstack mergers hang off -- and neither does the decoder's M-RoPE section.
#
# They cannot be added to config.json. `pull` compares every registry-listed file
# against a REMOTE manifest's byte count and treats a difference as a truncated
# download, so an edited config.json is silently replaced -- on this model that is
# a 4 GB re-pull. vision.json is not in that list, so it survives.
#
# The durable fix is `q4nx-build` writing the keys into the container it converts,
# which is Atomic-Germ's call; see .claude/plans/draft-issue-71-comment.md.

VISION_DEFAULTS = {
    # keyed by the container's own directory name
    "Qwen3-VL-4B-Instruct-NPU2": {
        "vision_heads": 16,
        "vision_deepstack_indexes": [5, 11, 17],
        "rope_scaling": {"rope_type": "default",
                         "mrope_section": [24, 20, 20],
                         "mrope_interleaved": True},
    },
}


def write_vision_sidecar(target, dir_name, quiet=False):
    """Write vision.json when the container declares a tower but omits how to read it.

    Never overwrites one that is already there, and never writes for a container
    whose config.json carries a vision_config -- those describe themselves.
    """
    cfg_path = Path(target) / "config.json"
    if not cfg_path.is_file():
        return
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return
    if not isinstance(cfg, dict) or "vision_model_weight" not in cfg:
        return                                   # not a VLM container
    if cfg.get("vision_config"):
        return                                   # it describes itself
    side = Path(target) / "vision.json"
    if side.exists():
        if not quiet:
            log(f"[INFO] vision.json already present: {side}")
        return
    known = VISION_DEFAULTS.get(dir_name)
    if known is None:
        if not quiet:
            log(f"[WARN] {dir_name} ships a vision tower but no vision_config, and there is no "
                f"known entry for it. Images will be refused until {side} names "
                f"vision_heads and vision_deepstack_indexes (see the model's published config).")
        return
    body = dict(known)
    body["_note"] = (f"What {dir_name} omits. config.json cannot hold this: pull re-fetches any "
                     f"registry-listed file whose size changes. Written by oflm-add.")
    side.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    if not quiet:
        log(f"[INFO] Wrote {side} ({dir_name} carries no vision_config)")


def main():
    ap = argparse.ArgumentParser(
        description="Install a pre-converted OFLM (Q4NX) model and register it with OpenFlowLM.",
    )
    ap.add_argument("repo", help="Hugging Face repo id (Org/Name), ModelScope id (with --modelscope), URL, or local directory")
    ap.add_argument("--tag", help="Registry tag (default: derived from the repo name, e.g. qwen3.5-claude:9b)")
    ap.add_argument("--family", help="details.family for engine dispatch (default: from matching official entry)")
    ap.add_argument("--config", help="model_list.json to update (default: $OFLM_CONFIG_PATH or ~/.config/oflm/model_list.json)")
    ap.add_argument("--models-root", help="models directory (default: $OFLM_MODEL_PATH or ~/.config/oflm/models)")
    ap.add_argument("--xclbin-dir", help="user xclbins directory (default: ~/.config/oflm/xclbins)")
    ap.add_argument("--xclbin-from", help="official model directory name to link xclbins from (default: best match, e.g. Qwen3.6-35B-A3B-NPU2)")
    ap.add_argument("--system-list", help="official model_list.json used for defaults (default: auto-detect)")
    ap.add_argument("--modelscope", action="store_true", help="Treat REPO as a ModelScope repo id (implied by www.modelscope.ai/.cn URLs)")
    ap.add_argument("--open-kernels", help="open kernel set for this model (a directory holding manifest.json); default: the installed set whose manifest spec_hash matches")
    ap.add_argument("--no-xclbin", action="store_true", help="Do not create the xclbins symlink (nor the open-kernels link, unless --open-kernels is given)")
    ap.add_argument("--no-verify", action="store_true", help="Skip sha256 verification of downloads")
    ap.add_argument("--force", action="store_true", help="Overwrite existing model files/links")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan and exit")
    ap.add_argument("--quiet", action="store_true", help="Less output")
    args = ap.parse_args()

    # Hugging Face is the default hub; --modelscope opts in explicitly and a
    # modelscope.ai/.cn URL implies it on its own.
    modelscope = args.modelscope
    repo_arg = args.repo
    local_dir = Path(repo_arg) if os.path.isdir(repo_arg) else None
    if local_dir:
        repo, dir_name = repo_arg, local_dir.name
    else:
        host_kind, repo = split_remote_repo(repo_arg)
        if host_kind == "modelscope":
            modelscope = True
        dir_name = repo.split("/")[-1]
    if not dir_name:
        raise SystemExit("Could not determine a model directory name from the repo.")

    system_list = find_system_model_list(args.system_list)
    system_registry = load_json(system_list)
    user_list = user_registry_path(args.config)
    models_root = models_root_dir(args.models_root)
    target = models_root / dir_name

    tag = derive_tag(dir_name, args.tag)
    bucket, size_token = tag.split(":", 1)
    official = match_official_entry(system_registry, dir_name)
    base_entry = official[3] if official else None
    # The README's tags, when this is a remote repo. Read BEFORE the name, because
    # a finetune's directory name carries no family: `Ornith-1.5-9B-NPU2` is a
    # Qwen3.5 model, and only its frontmatter says so. Not fatal if it fails --
    # family resolution falls back to the name, as it always did.
    front = _read_frontmatter(repo) if repo and not args.no_xclbin else {}
    # `oflm-family` is a declaration by whoever made the conversion, so it
    # outranks --family inference and the registry alike. It is optional: a repo
    # without it resolves from the tags, the registry, or the name, exactly as
    # before, which is what keeps a model we never saw working on a best effort.
    _declared = declared_family(front)
    if _declared and not args.family and not (base_entry or {}).get("details", {}).get("family"):
        if not args.quiet:
            log(f"[INFO] family {_declared} (declared by the repo's oflm-family tag)")
        family = _declared
    else:
        if not args.quiet and front:
            _f = frontmatter_family(front)
            if _f and not args.family and not (base_entry or {}).get("details", {}).get("family"):
                log(f"[INFO] family {_f} from the repo's README tags (name does not say)")
        family = derive_family(system_registry, dir_name, args.family, base_entry, front)
    # A repo that publishes the exact set it was tested against, so linking is a
    # lookup rather than a derivation. Also optional.
    declared_kernels_hash = declared_kernels(front)
    size_value = (base_entry or {}).get("size") or size_from_tag(tag)
    official, official_note = resolve_official(system_registry, dir_name, family, size_value)
    base_entry = official[3] if official else None
    src_tag = f"{official[1]}:{official[2]}" if official else None
    xclbin_source = args.xclbin_from or (base_entry or {}).get("name")
    if not args.dry_run:
        if official:
            note = f" ({official_note})" if official_note else ""
            log(f"[INFO] xclbins from official {src_tag}{note}")
        else:
            log("[WARN] No official model matched; no xclbins link. Pass --xclbin-from NAME (or --no-xclbin).")

    if args.dry_run:
        print(f"repo directory : {dir_name}")
        print(f"tag            : {tag}")
        print(f"details.family : {family}")
        print(f"official match : {src_tag or '(none)'}")
        print(f"xclbin source  : {xclbin_source or '(none)'}")
        probe = args.open_kernels or (local_dir if local_dir else None)
        if args.open_kernels:
            print(f"open kernels   : {args.open_kernels} (--open-kernels)")
        elif local_dir:
            sh, note = model_spec_hash(local_dir, family, size_token)
            roots = [user_xclbin_dir(args.xclbin_dir), find_system_xclbin_root()]
            found, _ = find_open_kernels(sh, roots, dir_name) if sh else (None, None)
            print(f"spec hash      : {sh or '(' + note + ')'}")
            print(f"open kernels   : {found or '(none installed)'}")
        print(f"models dir     : {target}")
        print(f"registry       : {user_list}")
        return

    # --- acquire model files ---
    if local_dir:
        if not args.quiet:
            log(f"[INFO] Using local model directory: {local_dir}")
        target.mkdir(parents=True, exist_ok=True)
        files = copy_from_dir(local_dir, target, force=args.force)
    else:
        snapshot = ms_cache_snapshot(repo) if modelscope else hf_cache_snapshot(repo)
        if snapshot:
            if not args.quiet:
                log(f"[INFO] Found local {'ModelScope' if modelscope else 'HF'} cache: {snapshot}")
            target.mkdir(parents=True, exist_ok=True)
            files = copy_from_dir(snapshot, target, force=args.force)
        else:
            if not args.quiet:
                log(f"[INFO] Downloading model files from {'ModelScope' if args.modelscope else 'Hugging Face'}: {repo}")
            target.mkdir(parents=True, exist_ok=True)
            files = fetch_assets(repo, target, args.modelscope, verify=not args.no_verify, force=args.force, quiet=args.quiet)

    missing = [f for f in REQUIRED_FILES if not (target / f).is_file()]
    if missing:
        raise SystemExit(f"Model is missing required files: {missing}")

    if not size_value:
        size_value = estimate_size(target / "config.json")
    entry = build_entry(base_entry, dir_name, files, size_value)
    entry.setdefault("details", {})["family"] = family

    register(user_list, tag, entry, system_registry)
    log(f"[INFO] Registered tag '{tag}' in {user_list}")

    system_root = find_system_xclbin_root()
    if not args.no_xclbin:
        if system_root is None:
            log("[WARN] Could not locate system xclbins; skipped symlink.")
        else:
            link_xclbins(
                system_root,
                user_xclbin_dir(args.xclbin_dir),
                dir_name,
                xclbin_source,
                force=args.force,
                quiet=args.quiet,
            )

    # Open kernels: keyed by this model's spec, not by an official model name.
    if args.open_kernels or not args.no_xclbin:
        setup_open_kernels(
            target,
            dir_name,
            [user_xclbin_dir(args.xclbin_dir), system_root],
            override=args.open_kernels,
            force=args.force,
            quiet=args.quiet,
            # The quant map is read from the q4nx config for THIS family and size,
            # which is what makes this hash equal the kernel build's. `family` is
            # already resolved above (registry entry, --family, or a name prefix);
            # `size` is the 'NNb' marker, and a finetune whose name carries one
            # gets the same kernel set as any other model of that family and size
            # -- which is the point: nothing here is keyed on an official model name.
            cat_family=family,
            size=size_token,
            declared_hash=declared_kernels_hash,
        )

    write_vision_sidecar(target, dir_name, quiet=args.quiet)

    print()
    print(f"Done: {dir_name} installed to {target}")
    print(f"Run:  oflm run {tag}   (or: oflm serve {tag})")
    print()
    print("OpenFlowLM discovers this model's registry entry and kernels")
    print("automatically; no OFLM_CONFIG_PATH/OFLM_XCLBIN_PATH exports needed.")
