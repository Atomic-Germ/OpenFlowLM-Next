"""flm_add - install a pre-converted FLM (Q4NX) model and register it with FastFlowLM.

Installable as ``flm-add`` (e.g. ``uv tool install flm-add``) or runnable as a
script (``python flm_add/__init__.py`` or the legacy ``flm-add.py`` shim).

Python-3 stdlib only (no pip packages). Works with any repo that
already contains the runtime-ready files (config.json, model.q4nx,
tokenizer.json, tokenizer_config.json, optionally chat_template.jinja):

    python3 flm-add.py Atomic-Germ/Qwen3.5-9B-Claude-4.8-Opus-NPU2

Repo can be a Hugging Face repo id, a ModelScope repo id (--modelscope), a full
Hugging Face URL, a ModelScope URL (www.modelscope.ai/.cn -- implies ModelScope
without the flag), or a local directory holding the model files. The tag is
derived from the repo name (e.g. Qwen3.5-9B-Claude-4.8-Opus-NPU2 ->
qwen3.5-claude:9b); override with --tag. Defaults for the registry entry
(family, engine, size, context length) are copied from the matching official
FastFlowLM entry.

Open kernels are handled separately. They belong to a model's *spec* (its
shape plus the per-role weight format), not to an official model name, so any
installed set whose manifest.json carries the same spec_hash as this model
drives it. flm-add derives the spec from the installed config.json (+
tokenizer.json and the model.q4nx header) via the open_kernels/recipes
checkout, finds the matching set, and links it at <model dir>/open_kernels --
the second place open_qwen36::Engine::find_kernels looks.

The script never rewrites the system model list or the system xclbins; it
writes a user-level registry at ~/.config/flm/model_list.json and adds a single
symlink into ~/.config/flm/xclbins/ for the new model directory. Custom FLM
models never ship xclbins (they are closed source), so the kernel symlink is
always taken from the matching official model, keyed by family (engine) and
size -- e.g. Darwin-36B-Opus-NPU2 -> Qwen3.6-35B-A3B-NPU2. The only thing
you need in your shell rc afterwards is:

    FLM_CONFIG_PATH="$HOME/.config/flm/model_list.json" FLM_XCLBIN_PATH="$HOME/.config/flm"
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

# The GGUF-direct path: a repo that carries llama.cpp-style quantized weights
# (e.g. mradermacher/*-i1-GGUF, Atomic-Germ/*-GGUF) can install one compatible
# file as model.gguf beside the tokenizer files -- no q4nx conversion needed.
# The open kernel sets ship an f32-scale build for this (see
# open_kernels/gguf_pool.py); incompatible quants are refused here, pointing
# at q4nx-build.
GGUF_FILES = ["config.json", "model.gguf", "tokenizer.json", "tokenizer_config.json"]
GGUF_OPTIONAL = ["chat_template.jinja"]
# preference order: 4.5-bit fits the NPU pools best, Q8_0 is the fallback
GGUF_QUANT_PREFERENCE = ["Q4_1", "Q4_0", "Q8_0"]
# Families whose engine loads model.gguf (open_qwen36's dense recipe; the GGUF
# manifest section + f32-scale kernel sets only ship for these). qwen3.5
# (fused attn_qkv/MTP) and qwen3.6-moe are NOT there yet -- a GGUF for those
# installs as plain safetensors instead.
GGUF_CAPABLE_FAMILIES = {"qwen3", "llama3", "gemma3", "granite", "hunyuan"}
GGUF_QUANT_RE = re.compile(r"(?:^|[.\-_ ])(I?Q[0-9](?:_[0-9A-Za-z]+)?)")

def gguf_quant_of(name):
    """The quant specifier in a llama.cpp-style weight file name, e.g.
    'Peach-2.0-9B-8k-Roleplay.i1-Q4_1.gguf' -> 'Q4_1' (None if unmarked)."""
    m = GGUF_QUANT_RE.search(name)
    return m.group(1) if m else None

def choose_gguf_file(names):
    """Pick the best compatible *.gguf from a repo's root file names.
    Returns (chosen, refused) -- refused maps every other gguf to why."""
    refused = {}
    cands = []
    for n in names:
        if not n.endswith(".gguf"):
            continue
        if re.search(r"-\d+-of-\d+\.gguf$", n):
            refused[n] = "multi-part GGUF (not supported; a single-file export needed)"
            continue
        q = gguf_quant_of(n)
        if q not in GGUF_QUANT_PREFERENCE:
            refused[n] = f"quant {q or 'unmarked'} not supported by the open kernels (have "                          f"{'/'.join(GGUF_QUANT_PREFERENCE)}); convert with q4nx-build"
            continue
        cands.append((GGUF_QUANT_PREFERENCE.index(q), n))
    cands.sort()
    if not cands:
        return None, refused
    return cands[0][1], refused

SYSTEM_LIST_CANDIDATES = [
    "/opt/fastflowlm/share/flm/model_list.json",
    "/usr/share/flm/model_list.json",
    "/usr/local/share/flm/model_list.json",
]

SYSTEM_XCLBIN_PREFIXES = [
    Path("/opt/fastflowlm/share/flm"),
    Path("/usr/share/flm"),
    Path("/usr/local/share/flm"),
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


def find_system_model_list():
    exe = shutil.which("flm")
    candidates = []
    if exe:
        candidates.append(Path(exe).parent / "model_list.json")
    candidates += [Path(p) for p in SYSTEM_LIST_CANDIDATES]
    for c in candidates:
        if c.is_file():
            return c
    raise SystemExit(
        "Could not locate the system model_list.json (looked next to `flm` and in "
        "/opt,/usr,/usr/local share/flm). Pass --system-list."
    )


def find_system_xclbin_root():
    """Directory whose <root>/xclbins/ holds the per-model kernel folders."""
    exe = shutil.which("flm")
    candidates = []
    if exe:
        candidates.append(Path(exe).parent)
        candidates.append((Path(exe).parent / ".." / "share" / "flm").resolve())
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
        env = os.environ.get("FLM_XCLBIN_PATH")
        base = Path(env) if env else Path.home() / ".config" / "flm"
    return base if base.name == "xclbins" else base / "xclbins"


def user_registry_path(arg):
    if arg:
        return Path(arg)
    env = os.environ.get("FLM_CONFIG_PATH")
    if env:
        return Path(env)
    return Path.home() / ".config" / "flm" / "model_list.json"


def models_root_dir(arg):
    if arg:
        return Path(arg)
    env = os.environ.get("FLM_MODEL_PATH")
    if env:
        return Path(env) / "models"
    return Path.home() / ".config" / "flm" / "models"


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

    Custom FLM models never ship xclbins (closed source), so the kernels must
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


def derive_family(system_registry, dir_name, explicit=None, base_entry=None):
    if explicit:
        return explicit
    if base_entry:
        fam = base_entry.get("details", {}).get("family")
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
    headers = {"User-Agent": "flm-add/1.0"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _ms_headers():
    # Never forward Hugging Face credentials to ModelScope hosts.
    return {"User-Agent": "flm-add/1.0"}


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


def normalize_tokenizer_config(target, repo, bases, modelscope=False):
    """FLM reads eos_token_id (array) / bos_token_id from tokenizer_config.json;
    stock HF repos keep them in generation_config.json -- merge them over."""
    tc = target / "tokenizer_config.json"
    if not tc.is_file():
        return
    cfg = json.loads(tc.read_text(encoding="utf-8"))
    need_eos = "eos_token_id" not in cfg or not isinstance(cfg.get("eos_token_id"), list)
    need_bos = cfg.get("bos_token") is not None and "bos_token_id" not in cfg
    if not (need_eos or need_bos):
        return
    g = None
    gc = target / "generation_config.json"
    for cand in [gc, fetch_from_repo(repo, "generation_config.json", gc, modelscope=modelscope)] + \
                [fetch_from_repo(b, "generation_config.json", gc, modelscope=modelscope) for b in bases]:
        if cand and Path(cand).is_file():
            try:
                g = json.loads(Path(cand).read_text(encoding="utf-8"))
            except Exception:
                g = None
            if g and (need_eos and isinstance(g.get("eos_token_id"), list) or
                      need_bos and isinstance(g.get("bos_token_id"), int)):
                break
    if not g:
        log("[WARN] no generation_config.json; the model may fail at load if the engine "
            "needs eos_token_id in tokenizer_config.json")
        return
    changed = False
    if need_eos and isinstance(g.get("eos_token_id"), list):
        cfg["eos_token_id"] = g["eos_token_id"]
        changed = True
    if need_bos and isinstance(g.get("bos_token_id"), int):
        cfg["bos_token_id"] = g["bos_token_id"]
        changed = True
    if changed:
        tc.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        log("[INFO] merged eos/bos ids from generation_config.json into tokenizer_config.json")


def readme_base_model(repo_id, modelscope=False):
    """The base model id(s) from the repo's README.md YAML frontmatter.

    GGUF-quant repos (mradermacher etc.) carry no tokenizer/config files, but
    their model card names the original model, whose repo has everything. The
    field is `base_model:` either inline (`[[Org/Name]]` wikilink form or
    plain) or a `- Org/Name` list (merge lineages; the first entry is the
    closest to the quantized checkpoint).
    """
    url = (f"https://www.modelscope.ai/models/{repo_id}/resolve/master/README.md" if modelscope
           else f"https://huggingface.co/{repo_id}/raw/main/README.md")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=_hf_headers()), timeout=60) as r:
            text = r.read().decode("utf-8", errors="replace")
    except Exception:
        return []
    lines = text.splitlines()
    # the frontmatter: the first "---" ... "---" block at the top
    if not lines or lines[0].strip() != "---":
        return []
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        return []
    bases = []
    in_base = False
    for line in lines[1:end]:
        s = line.strip()
        if s.startswith("base_model:"):
            in_base = True
            v = s[len("base_model:"):].strip()
            if v:
                bases.append(v)
        elif in_base and s.startswith("- "):
            bases.append(s[2:].strip())
        elif in_base and s and not s.startswith("-"):
            in_base = False
    out = []
    for b in bases:
        b = re.sub(r"^\[\[|\]\]$", "", b).strip("\"' ")
        # "org/name (size)" style suffixes occasionally appear
        b = re.sub(r"\s*\(.*$", "", b).strip()
        if b and "/" in b and b not in out:
            out.append(b)
    return out


def fetch_from_repo(repo_id, fname, dest, modelscope=False, force=False):
    """One auxiliary file from another repo (local HF cache first, then remote).

    Returns the local path, or None if the file is not available."""
    if dest.is_file() and not force:
        return dest
    snapshot = ms_cache_snapshot(repo_id) if modelscope else hf_cache_snapshot(repo_id)
    src = (snapshot / fname) if snapshot else None
    if src and src.is_file():
        shutil.copyfile(src, dest)
        return dest
    if modelscope:
        tree = ms_file_tree(repo_id)[1]
        if fname not in tree:
            return None
        meta = tree[fname]
        url = f"https://www.modelscope.ai/models/{repo_id}/resolve/master/{fname}"
        headers, sha, size = _ms_headers(), (meta.get("Sha256") or "").lower() or None, meta.get("Size") or None
    else:
        entries = {e.get("path"): e for e in hf_file_tree(repo_id)}
        if fname not in entries:
            return None
        lfs = entries[fname].get("lfs") or {}
        url = f"https://huggingface.co/{repo_id}/resolve/main/{fname}"
        headers, sha, size = None, lfs.get("oid"), lfs.get("size") or entries[fname].get("size")
    dest.parent.mkdir(parents=True, exist_ok=True)
    download_file(url, dest, expected_size=size, expected_sha=sha, verify=True)
    return dest


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


def fetch_assets(repo_id, target, modelscope=False, verify=True, force=False, quiet=False, gguf_name=None):
    """Populate target/ with the model files; returns the list of files present.

    With gguf_name set (the GGUF-direct path) the wanted set is the tokenizer
    files plus that one GGUF, stored canonically as model.gguf."""
    obtained = []
    target.mkdir(parents=True, exist_ok=True)
    if gguf_name:
        want = GGUF_FILES + GGUF_OPTIONAL
    else:
        want = ALL_FILES
    if modelscope:
        domain, entries = ms_file_tree(repo_id)
        for fname in want:
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
            remote = gguf_name if fname == "model.gguf" else fname
            meta = entries.get(remote) or {}
            expected_size = meta.get("Size") or None
            expected_sha = (meta.get("Sha256") or "").lower() or None
            if not quiet:
                gb = f" ({expected_size / 1e9:.2f} GB)" if expected_size else ""
                label = remote if remote != fname else fname
                log(f"Downloading {label}{gb} from ModelScope ({domain})...")
            download_file(
                f"https://{domain}/models/{repo_id}/resolve/master/{remote}",
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
    for fname in want:
        remote = gguf_name if fname == "model.gguf" else fname
        if remote not in entries:
            continue
        dest = target / fname
        if dest.is_file() and not force:
            obtained.append(fname)
            continue
        lfs = entries[remote].get("lfs") or {}
        expected_sha = lfs.get("oid")
        expected_size = lfs.get("size") or entries[remote].get("size")
        label = remote if remote != fname else fname
        log(f"Downloading {label} ({expected_size/1e9:.2f} GB)...")
        download_file(
            f"https://huggingface.co/{repo_id}/resolve/main/{remote}",
            dest,
            expected_size=expected_size,
            expected_sha=expected_sha,
            verify=verify,
            quiet=quiet,
        )
        obtained.append(fname)
    return obtained


def copy_from_dir(src_dir, target, force=False, gguf_name=None):
    obtained = []
    want = GGUF_FILES + GGUF_OPTIONAL if gguf_name else ALL_FILES
    for fname in want:
        src = src_dir / (gguf_name if fname == "model.gguf" else fname)
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
    entry.setdefault("flm_min_version", "0.9.45")
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
    if link.is_symlink():
        if os.readlink(link) == target:
            if not quiet:
                log(f"[INFO] xclbins link already in place: {link}")
            return
        link.unlink()
    elif link.exists():
        if force:
            shutil.rmtree(link)
        else:
            raise SystemExit(
                f"{link} already exists and is not a symlink. Remove it or pass --force."
            )
    os.symlink(target, link)
    if not quiet:
        log(f"[INFO] Linked xclbins: {link} -> {target}")


# -------------------------------------------------------------- open kernels
#
# Closed kernels are per official model; open kernel sets are per ModelSpec --
# the shape plus the per-role weight format. Two shape-identical models (a
# fine-tune, a distill) share one set. The engine finds a set at
# FLM_OPEN_KERNELS_DIR, then <model dir>/open_kernels, then
# <xclbins root>/<model name>/open_kernels (open_qwen36::Engine::find_kernels).
# We link into the model directory: it is the one root that does not depend on
# how FLM_XCLBIN_PATH happens to be set.

def open_kernels_checkout():
    """An `open_kernels/` directory holding recipes/spec.py, or None.

    Looked for at $OPEN_KERNELS_DIR, then next to this checkout (flm-add lives
    in <repo>/utilities/flm-add), then under the working directory.
    """
    candidates = []
    env = os.environ.get("OPEN_KERNELS_DIR")
    if env:
        candidates.append(Path(env))
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "open_kernels")
    candidates.append(Path.cwd() / "open_kernels")
    for c in candidates:
        if (c / "recipes" / "spec.py").is_file():
            return c
    return None


def model_spec_hash(model_dir):
    """(spec_hash, note) for an installed model directory; (None, why) on failure.

    Derived the way the recipes do it -- recipes.load.spec_from_model_dir reads
    config.json, the tokenizer's real vocab, and the per-role weight format off
    the model.q4nx safetensors header (no weight byte is read).
    """
    root = open_kernels_checkout()
    if root is None:
        return None, "no open_kernels/recipes checkout found (set OPEN_KERNELS_DIR)"
    added = str(root)
    inserted = added not in sys.path
    if inserted:
        sys.path.insert(0, added)
    try:
        from recipes.load import spec_from_model_dir
        spec = spec_from_model_dir(Path(model_dir))
        return spec.spec_hash(), f"spec from {root}"
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
        "developer mode or admin). Run flm with:")
    log(f'           FLM_OPEN_KERNELS_DIR="{kernel_dir}"')
    return False


def setup_open_kernels(model_dir, dir_name, roots, override=None, force=False, quiet=False):
    """Find and link the open kernel set for this model; say which and why."""
    if override:
        kernel_dir = Path(override)
        if not (kernel_dir / "manifest.json").is_file():
            raise SystemExit(f"--open-kernels {kernel_dir} has no manifest.json")
        log(f"[INFO] open kernels: {kernel_dir} (--open-kernels)")
        return link_open_kernels(model_dir, kernel_dir, force=force, quiet=quiet)

    spec_hash, note = model_spec_hash(model_dir)
    if not spec_hash:
        if not quiet:
            log(f"[INFO] No open-kernel spec for this model ({note}); closed kernels only.")
        return False
    kernel_dir, source = find_open_kernels(spec_hash, roots, dir_name)
    if kernel_dir:
        log(f"[INFO] open kernels from '{source}': its manifest spec_hash matches "
            f"this model's ({spec_hash[:19]})")
        return link_open_kernels(model_dir, kernel_dir, force=force, quiet=quiet)
    checkout = open_kernels_checkout()
    script = (checkout / "export_qwen36_kernels.py") if checkout else Path("open_kernels/export_qwen36_kernels.py")
    log(f"[INFO] No installed open kernel set has spec_hash {spec_hash[:19]}; "
        "the closed kernels stay in charge. Build one with:")
    log(f'           python "{script}" --model-dir "{model_dir}"')
    return False


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="Install a pre-converted FLM (Q4NX) model and register it with FastFlowLM.",
    )
    ap.add_argument("repo", help="Hugging Face repo id (Org/Name), ModelScope id (with --modelscope), URL, or local directory")
    ap.add_argument("--tag", help="Registry tag (default: derived from the repo name, e.g. qwen3.5-claude:9b)")
    ap.add_argument("--family", help="details.family for engine dispatch (default: from matching official entry)")
    ap.add_argument("--config", help="model_list.json to update (default: $FLM_CONFIG_PATH or ~/.config/flm/model_list.json)")
    ap.add_argument("--models-root", help="models directory (default: $FLM_MODEL_PATH or ~/.config/flm/models)")
    ap.add_argument("--xclbin-dir", help="user xclbins directory (default: ~/.config/flm/xclbins)")
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

    # ---- scan the repo for a compatible GGUF (the GGUF-direct path): a repo
    # that carries llama.cpp-style weights installs one of them as model.gguf;
    # repos that ship model.q4nx take the NPU2 path even if a gguf also sits
    # there (the converted container is the curated one).
    gguf_names = []
    if local_dir:
        gguf_names = [e.name for e in local_dir.iterdir() if e.is_file() and e.suffix == ".gguf"]
    else:
        try:
            tree = ms_file_tree(repo)[1] if (args.modelscope or split_remote_repo(repo_arg)[0] == "modelscope") else hf_file_tree(repo)
            gguf_names = [e.get("path") for e in tree if e.get("path") and "/" not in e["path"] and e["path"].endswith(".gguf")]
        except Exception as ex:
            log(f"[WARN] Could not list the repo tree ({ex}); assuming the NPU2 path.")
    gguf_name, gguf_refused = choose_gguf_file(gguf_names)
    have_q4nx = (local_dir / "model.q4nx").is_file() if local_dir else False
    gguf_mode = gguf_name is not None and not have_q4nx

    system_list = find_system_model_list()
    system_registry = load_json(system_list)
    user_list = user_registry_path(args.config)
    models_root = models_root_dir(args.models_root)
    target = models_root / dir_name

    tag = derive_tag(dir_name, args.tag)
    bucket, size_token = tag.split(":", 1)
    official = match_official_entry(system_registry, dir_name)
    base_entry = official[3] if official else None
    family = derive_family(system_registry, dir_name, args.family, base_entry)
    size_value = (base_entry or {}).get("size") or size_from_tag(tag)
    official, official_note = resolve_official(system_registry, dir_name, family, size_value)
    base_entry = official[3] if official else None
    src_tag = f"{official[1]}:{official[2]}" if official else None
    xclbin_source = args.xclbin_from or (base_entry or {}).get("name")

    # ---- the GGUF-direct gate: the engine must read model.gguf AND the linked
    # xclbins must actually carry the GGUF-direct kernel builds (the nested
    # "gguf" manifest section, produced by the *_f32 sets in utilities/
    # build-all.sh). Otherwise fall back to the normal safetensors install.
    gguf_skip = None
    if gguf_mode:
        if family not in GGUF_CAPABLE_FAMILIES:
            gguf_skip = (f"family '{family}' does not load GGUF yet (GGUF-direct covers "
                         + ", ".join(sorted(GGUF_CAPABLE_FAMILIES))
                         + "); installing the safetensors weights instead")
        else:
            xk = None
            sys_xcl = find_system_xclbin_root()
            if xclbin_source and sys_xcl:
                mj = sys_xcl / xclbin_source / "open_kernels" / "manifest.json"
                if mj.is_file():
                    try:
                        xk = json.loads(mj.read_text(encoding="utf-8"))
                    except Exception:
                        xk = None
            if xk is None or "gguf" not in xk:
                gguf_skip = (f"the kernels this install links ({xclbin_source or 'no official match'}) "
                             "have no GGUF-direct build (missing open_kernels gguf manifest); "
                             "rebuild them with utilities/build-all.sh -- "
                             "installing the safetensors weights instead")
        if gguf_skip:
            gguf_refused[gguf_name] = gguf_skip
            log(f"[INFO] skipping {gguf_name}: {gguf_skip}")
            gguf_name, gguf_mode = None, False

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
        if gguf_mode:
            print(f"weights        : {gguf_name} -> model.gguf (GGUF-direct, f32-scale pools)")
            for n, why in sorted(gguf_refused.items()):
                print(f"  skipped      : {n} ({why})")
        elif gguf_refused:
            print(f"weights        : safetensors (GGUF present but not usable: "
                  + "; ".join(sorted(set(gguf_refused.values()))) + ")")
        probe = args.open_kernels or (local_dir if local_dir else None)
        if args.open_kernels:
            print(f"open kernels   : {args.open_kernels} (--open-kernels)")
        elif local_dir:
            sh, note = model_spec_hash(local_dir)
            roots = [user_xclbin_dir(args.xclbin_dir), find_system_xclbin_root()]
            found, _ = find_open_kernels(sh, roots, dir_name) if sh else (None, None)
            print(f"spec hash      : {sh or '(' + note + ')'}")
            print(f"open kernels   : {found or '(none installed)'}")
        print(f"models dir     : {target}")
        print(f"registry       : {user_list}")
        return

    # --- acquire model files ---
    if gguf_mode and not args.quiet:
        log(f"[INFO] GGUF-direct install: {gguf_name} -> model.gguf")
        for n, why in sorted(gguf_refused.items()):
            log(f"[INFO]   skipping {n}: {why}")
    if local_dir:
        if not args.quiet:
            log(f"[INFO] Using local model directory: {local_dir}")
        target.mkdir(parents=True, exist_ok=True)
        files = copy_from_dir(local_dir, target, force=args.force, gguf_name=gguf_name if gguf_mode else None)
    else:
        snapshot = ms_cache_snapshot(repo) if modelscope else hf_cache_snapshot(repo)
        if snapshot:
            if not args.quiet:
                log(f"[INFO] Found local {'ModelScope' if modelscope else 'HF'} cache: {snapshot}")
            target.mkdir(parents=True, exist_ok=True)
            files = copy_from_dir(snapshot, target, force=args.force, gguf_name=gguf_name if gguf_mode else None)
        else:
            if not args.quiet:
                log(f"[INFO] Downloading model files from {'ModelScope' if args.modelscope else 'Hugging Face'}: {repo}")
            target.mkdir(parents=True, exist_ok=True)
            files = fetch_assets(repo, target, args.modelscope, verify=not args.no_verify, force=args.force,
                                 quiet=args.quiet, gguf_name=gguf_name if gguf_mode else None)

    required = ["model.gguf", "tokenizer.json", "tokenizer_config.json"] if gguf_mode else REQUIRED_FILES
    missing = [f for f in required if not (target / f).is_file()]
    if gguf_mode:
        # GGUF-quant repos (mradermacher etc.) often ship only the GGUF: take the
        # tokenizer/config files from the original model the README names.
        # config.json too: the curated config beats the GGUF-derived one (and
        # Granite's folded multipliers exist only there).
        aux_missing = [f for f in missing if f != "model.gguf"]
        if not (target / "config.json").is_file():
            aux_missing.append("config.json")
        if aux_missing:
            bases = readme_base_model(repo, modelscope)
            if bases:
                log(f"[INFO] base model(s) from README.md: {', '.join(bases)}")
                for f in aux_missing:
                    got = None
                    for b in bases:
                        try:
                            got = fetch_from_repo(b, f, target / f, modelscope=modelscope)
                        except Exception as ex:
                            log(f"[WARN] fetching {f} from {b} failed: {ex}")
                            got = None
                        if got:
                            log(f"[INFO]   {f} <- {b}")
                            break
                missing = [f for f in required if not (target / f).is_file()]
    normalize_tokenizer_config(target, repo, readme_base_model(repo, modelscope), modelscope)
    if missing:
        if gguf_mode and missing == ["config.json"]:
            missing = []
        if missing and not gguf_mode and gguf_refused and "model.safetensors" in missing:
            raise SystemExit(
                f"Model is missing required files: {missing}\n"
                "The repo carries only GGUF weights and they cannot be used directly: "
                + "; ".join(sorted(set(gguf_refused.values())))
                + "\nConvert one to model.q4nx with q4nx-build, or pick a GGUF-capable family (--family).")
        if missing:
            raise SystemExit(f"Model is missing required files: {missing}")
    if gguf_mode and not (target / "config.json").is_file():
        log("[WARN] No config.json in the repo; the engine will derive the config from the GGUF "
            "metadata (llama-family shapes only -- Granite etc. need a config.json).")

    if not size_value:
        size_value = estimate_size(target / "config.json")
    entry = build_entry(base_entry, dir_name, files, size_value)
    entry.setdefault("details", {})["family"] = family
    if gguf_mode:
        entry["details"]["weights"] = "gguf"      # the engine loads model.gguf (f32-scale pools)
        entry.setdefault("flm_min_version", "0.9.45")

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
        )

    print()
    print(f"Done: {dir_name} installed to {target}")
    print(f"Run:  flm run {tag}   (or: flm serve {tag})")
    print()
    print("Make sure your shell has these exports (add to ~/.bashrc):")
    print('    export FLM_CONFIG_PATH="$HOME/.config/flm/model_list.json"')
    print('    export FLM_XCLBIN_PATH="$HOME/.config/flm"')

