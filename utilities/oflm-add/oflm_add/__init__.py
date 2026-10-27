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
import copy
import hashlib
import dataclasses
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import time
import urllib.error
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
# Quant families the open kernels ingest (pools.cpp std_perm_gguf): exact
# pools for Q4_1/Q4_0/Q8_0, host requant for Q4_K/Q6_K (validated end to
# end for Q6_K on gemma3-4b). Order is install preference: exact first,
# then K-quants, Q8_0 last (~2x the bytes of the 4-bit files for little
# quality gain on the NPU path).
GGUF_QUANT_PREFERENCE = ["Q4_1", "Q4_0", "Q4_K", "Q5_K", "Q6_K", "Q8_0", "IQ4"]
# Whether a GGUF can be used is NOT a property of the family NAME: it is a
# property of the kernel set the install links. The recipe emits the manifest's
# f32-scale twin ("gguf" section) and the export builds its kernel set; the
# engine swaps that section in when the model ships a model.gguf. So the gate
# is the manifest itself (manifest_supports_gguf), read from the xclbins the
# install links -- one check, no list to keep in step with the recipes, and no
# name namespaces to confuse: the registry's family ("qwen3.5") and the
# recipe's ("qwen35") are different strings for the same model, and a list
# written in one of them silently refuses the other.
# Embedding GGUF (gemma-embedding) loads via the existing bf16 matmul kernels,
# not the f32-scale GGUF-direct pools; the engine reads model.gguf directly and
# dequantizes, so no open_kernels "gguf" manifest section is required.
EMBEDDING_FAMILIES = {"embed-gemma"}
# NPU-embedding models (the NpuEmbeddings engine): a HuggingFace safetensors
# checkpoint plus a pre-tiled `.npue` container packed on first load. They are
# NOT GGUF/q4nx and do not use the chat-model xclbin linking -- their kernels
# live under <xclbin_root>/xclbins/<npue_design_family>/. The authoritative file
# list and npue_design_family come from the official model_list.json entry
# (matched by name); the engine routes by the exact registered tag.
NPU_EMBED_FAMILIES = {"bge", "nomic", "minilm", "gte"}
NPU_EMBED_FILES = [
    "config.json", "model.safetensors", "tokenizer.json",
    "tokenizer_config.json", "vocab.txt", "1_Pooling/config.json",
]
# The quant must run to the end of the stem: 'Q4_0_4_4' is a different
# tensor layout (ik_llama matmul shuffles), not Q4_0, and matching only a
# prefix once installed exactly that as Q4_0 pools (silent garbage).
GGUF_QUANT_RE = re.compile(r"[.\-_ ](I?Q[0-9](?:_[A-Za-z0-9]+)*)\.gguf$", re.IGNORECASE)

def gguf_quant_suffix(name):
    """The raw trailing quant-ish token of a GGUF file name, e.g.
    'Model-Q4_K_M.gguf' -> 'Q4_K_M', 'Model-Q4_0_4_4.gguf' -> 'Q4_0_4_4'
    (None when the stem ends in nothing quant-like)."""
    m = GGUF_QUANT_RE.search(name)
    return m.group(1) if m else None

def gguf_quant_family(suffix):
    """A raw suffix folded to its tensor family, or None when unknown.
    K-subtype letters stay with their family ('Q4_K_M' -> 'Q4_K': the tensors
    inside are plain Q4_K), but anything with a digit run past the family
    ('Q4_0_4_4', 'Q4_K_L2'?) is a different layout, not a variant."""
    if not suffix:
        return None
    q = suffix.upper()
    for fam in GGUF_QUANT_PREFERENCE:
        if q == fam:
            return fam
        if q.startswith(fam + "_") and re.fullmatch(r"[A-Z]+", q[len(fam) + 1:]):
            return fam
    return None

def gguf_quant_of(name):
    """The quant family in a llama.cpp-style weight file name, e.g.
    'Peach-2.0-9B-8k-Roleplay.i1-Q4_1.gguf' -> 'Q4_1' (None if unmarked or
    unknown -- 'Q4_0_4_4' is NOT 'Q4_0')."""
    return gguf_quant_family(gguf_quant_suffix(name))

def split_quant_request(repo_arg: str):
    """`repo:QUANT` -- name the quant to take from a GGUF repo.

    "mradermacher/Qwen3.5-9B-i1-GGUF:IQ4_NL" asks for that file's quant; the
    suffix must LOOK like a quant (letters, digits, underscores) or the argument
    is a repo id or a URL as it stands -- a URL's port would otherwise be read
    as a quant request.
    """
    if ":" not in repo_arg:
        return repo_arg, None
    head, _, tail = repo_arg.rpartition(":")
    if not head or not re.fullmatch(r"[A-Za-z0-9_]+", tail):
        return repo_arg, None
    return head, tail.upper()


def choose_gguf_file(names, want=None):
    """Pick the best compatible *.gguf from a repo's root file names.
    Returns (chosen, refused) -- refused maps every other gguf to why.

    `want` names the quant the user asked for ("IQ4_NL"): the files whose quant
    SUFFIX matches it exactly win, and the ones whose FAMILY matches are the
    fallback. Without it the preference order decides (Q4_1 first -- it is the
    one the pool takes byte-exactly).
    """
    refused = {}
    cands = []
    for n in names:
        if not n.endswith(".gguf"):
            continue
        if re.search(r"-\d+-of-\d+\.gguf$", n):
            refused[n] = "multi-part GGUF (not supported; a single-file export needed)"
            continue
        suffix = gguf_quant_suffix(n)
        q = gguf_quant_family(suffix)
        if q not in GGUF_QUANT_PREFERENCE:
            refused[n] = f"quant {suffix or 'unmarked'} not supported by the open kernels (have "                          f"{'/'.join(GGUF_QUANT_PREFERENCE)}); convert with q4nx-build"
            continue
        cands.append((GGUF_QUANT_PREFERENCE.index(q), q, suffix, n))
    cands.sort()
    if not cands:
        return None, refused
    if want:
        exact = [c for c in cands if c[2].upper() == want]
        if exact:
            return exact[0][3], refused
        fam = gguf_quant_family(want)
        same = [c for c in cands if c[1] == fam]
        if not same:
            have = ", ".join(sorted({c[2] for c in cands})) or "none"
            for c in cands:
                refused[c[3]] = f"the repo has no {want} GGUF (it has: {have})"
            return None, refused
        return same[0][3], refused
    return cands[0][3], refused


def root_file_names(tree):
    """Root file names from an HF tree list or ModelScope's name-keyed dict."""
    names = tree.keys() if isinstance(tree, dict) else (e.get("path") for e in tree)
    return [n for n in names if n and "/" not in n]


def repo_has_q4nx(tree, cache=None):
    return "model.q4nx" in root_file_names(tree) or bool(
        cache and (cache / "model.q4nx").is_file()
    )


def manifest_supports_gguf(path):
    try:
        return "gguf" in json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False


def tree_xclbins():
    """The repo's own kernel sets (src/xclbins), when oflm-add runs from a
    checkout: a freshly exported set lives there, and it may be newer than the
    installed one (a rebuild adds the f32-scale twins the GGUF path needs)."""
    root = open_kernels_checkout()
    if root is None:
        return None
    repo = Path(root).parent
    for d in (repo / "src" / "xclbins", repo / "xclbins"):
        if d.is_dir():
            return d
    return None


def gguf_kernel_set(name, roots):
    """The kernel set for an official model that carries the GGUF-direct build.

    The roots are searched in order (the user's, the system's, the tree's), and
    the first set whose manifest has the "gguf" section wins. The system's set
    may predate the f32-scale twins, so this is how a fresh set is found rather
    than refusing the GGUF.
    """
    if not name:
        # No official match and no geometry match: the caller wants the "no
        # GGUF-direct set" refusal, not a TypeError from Path / None.
        return None
    for root in roots:
        if not root:
            continue
        mj = Path(root) / name / "open_kernels" / "manifest.json"
        if manifest_supports_gguf(mj):
            return mj.parent
    return None

def _require_same_geometry(kdir, spec, respec, model_name):
    """Refuse a GGUF install whose kernel set is a DIFFERENT geometry.

    The set is found by directory name (the official entry's, or the model's),
    and a size the registry does not carry used to fall back to the nearest
    official size -- so a Qwen3-14B GGUF resolved to Qwen3-8B's kernels and the
    install "succeeded". The kernels decide the arithmetic, so a set whose
    manifest `spec` has a different kernel_hash cannot run this model.
    """
    ModelSpec = _model_spec_class()
    spec = spec or respec
    if spec is None or ModelSpec is None:
        return
    mj = Path(kdir) / "manifest.json"
    try:
        d = json.loads(mj.read_text(encoding="utf-8"))
        laid = d.get("spec")
        if not laid or ModelSpec.from_dict(laid).kernel_hash() == spec.kernel_hash():
            return
    except (OSError, ValueError):
        return
    raise SystemExit(
        f"{model_name}: the GGUF-direct kernel set at {kdir} is a DIFFERENT geometry "
        f"({Path(kdir).parent.name}), so it cannot run this model.\n"
        f"Build this size's kernels first:\n"
        f"  python utilities/build-new-spec.py --repo <this model's base repo>")


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
    # Stock quant repos keep the upstream org/model prefix the fine-tune
    # slugs drop: bartowski/Meta-Llama-3.1-8B-Instruct-GGUF,
    # google/gemma-3-4b-it-GGUF, microsoft/Phi-4-mini-instruct-GGUF.
    ("meta-llama", "llama3"),
    ("gemma-3", "gemma3"),
    ("phi-4", "phi4"),
    # Granite is its own family now (the dense recipe, head_dim 64 at hidden
    # 2560). It aliased onto llama3 because nothing served it; leaving that
    # would route a Granite directory to the Llama 3 AutoModel, whose sampler
    # defaults and chat-template probe are the wrong ones -- and the closed
    # llama_npu refuses hidden_size 2560 outright.
    ("granite", "granite"),
    ("hunyuan", "hunyuan"),
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
    ("embedding-gemma", "embed-gemma"),
    ("embeddinggemma", "embed-gemma"),
    # NPU-embedding (NpuEmbeddings) families: HF safetensors checkpoints, no GGUF.
    ("bge", "bge"),
    ("nomic", "nomic"),
    ("all-minilm", "minilm"),
    ("minilm", "minilm"),
    ("gte", "gte"),
]

# GGUF general.architecture -> runtime details.family, consulted when the repo
# name matches no FAMILY_ALIASES prefix. The name alias above covers the known
# embedding-gemma hosts, but any other repo serving a gemma-embedding GGUF
# (arbitrary org/name) should still resolve without --family. Only arches
# whose shapes an engine here serves are mapped; anything else (moe variants
# with different geometry, one-off merges) must say --family explicitly rather
# than link the wrong kernels.
GGUF_ARCH_FAMILIES = {
    "gemma-embedding": "embed-gemma",
    "qwen3": "qwen3",
    "qwen2": "qwen2",
    "llama": "llama3",
    "gemma3": "gemma3",
    "phi4": "phi4",
    "granite": "granite",
    "hunyuan": "hunyuan",
}

# HuggingFace config.json model_type -> runtime details.family, same role as
# the GGUF map for repos whose names carry no family marker (chaotic finetune
# slugs). Conservative for the same reason: an unmapped model_type refuses
# with a --family hint instead of guessing.
CONFIG_MODEL_FAMILIES = {
    "qwen3": "qwen3",
    "qwen2": "qwen2",
    "llama": "llama3",
    "gemma3": "gemma3",
    "phi4": "phi4",
    "granite": "granite",
    "hunyuan": "hunyuan",
    "gpt_oss": "gpt-oss",
}


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


def save_json_atomic(path: Path, data: dict) -> None:
    """Save JSON atomically (temp file + rename)."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except:
        try:
            os.unlink(tmp)
        except:
            pass
        raise


try:
    import fcntl as _fcntl  # POSIX record locks (fcntl/F_SETLK)
except ImportError:  # Windows
    _fcntl = None
    import msvcrt as _msvcrt


class _settings_lock:
    """File lock on <settings.json>.lock, compatible with the C++ SettingsLock.

    The C++ side takes a fcntl record lock (LockFileEx on Windows) on the
    same "<path>.lock" file and holds it for its whole load-modify-save
    cycle. Registration must hold the same lock across read-modify-replace,
    or a concurrent `config set` (C++) and `oflm add` (Python) can lose each
    other's writes: atomic rename only prevents partial JSON, not lost
    updates. Waits up to 2 seconds, like the C++ side; past the deadline it
    proceeds unlocked rather than failing the install.
    """

    def __init__(self, path):
        self.lock_path = Path(str(path) + ".lock")
        self.fd = None
        self.handle = None

    def __enter__(self):
        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return self
        try:
            if _fcntl is not None:
                self.fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o644)
                deadline = time.monotonic() + 2.0
                while True:
                    try:
                        _fcntl.lockf(self.fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
                        return self
                    except OSError:
                        if time.monotonic() >= deadline:
                            os.close(self.fd)
                            self.fd = None
                            return self  # proceed unlocked, like C++ locked()==false
                        time.sleep(0.1)
            else:
                self.handle = open(str(self.lock_path), "a+b")
                deadline = time.monotonic() + 2.0
                while True:
                    try:
                        self.handle.seek(0)
                        _msvcrt.locking(self.handle.fileno(), _msvcrt.LK_NBLCK, 1)
                        return self
                    except OSError:
                        if time.monotonic() >= deadline:
                            self.handle.close()
                            self.handle = None
                            return self
                        time.sleep(0.1)
        except OSError:
            # Locking is best-effort: an unmanageable lock file must not
            # block an install.
            self.fd = None
            self.handle = None
        return self

    def __exit__(self, *exc):
        try:
            if self.fd is not None:
                _fcntl.lockf(self.fd, _fcntl.LOCK_UN)
                os.close(self.fd)
            elif self.handle is not None:
                self.handle.seek(0)
                _msvcrt.locking(self.handle.fileno(), _msvcrt.LK_UNLCK, 1)
                self.handle.close()
        except OSError:
            pass
        self.fd = None
        self.handle = None


def _engine_dirs():
    """Directories holding an installed engine. An explicit OFLM_EXECUTABLE
    goes before `oflm` on PATH, so a checkout build stays ahead of a release
    install when both are present."""
    out = []
    for exe in (os.environ.get("OFLM_EXECUTABLE"), shutil.which("oflm")):
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
    # The registry oflm itself reads comes first: when it is pointed at
    # another file, that file is the one the tool has to use as well --
    # otherwise a dev tree registers models against a different registry
    # than the one it reads.
    env_registry = os.environ.get("OFLM_CONFIG_PATH")
    if env_registry:
        candidates.append(Path(env_registry))
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
    # The xclbin root oflm itself searches comes first: when it is pointed
    # somewhere else, the kernels are there, and a dev tree has its own.
    env_xcl = os.environ.get("OFLM_XCLBIN_PATH")
    if env_xcl:
        candidates.append(Path(env_xcl))
    for d in _engine_dirs():
        candidates.append(d)
        candidates.append((d / ".." / "share" / "oflm").resolve())
    candidates += SYSTEM_XCLBIN_PREFIXES
    for c in candidates:
        if (c / "xclbins").is_dir():
            return c / "xclbins"
    return None


def user_xclbin_dir(arg):
    """Resolve the user-level xclbins directory (where symlinks are added).

    Intentionally ignores OFLM_XCLBIN_PATH: that variable is a runtime search
    path for the engine (and may point at a read-only Nix store path), not a
    writable destination for oflm-add.
    """
    base = Path(arg) if arg else Path.home() / ".config" / "oflm"
    return base if base.name == "xclbins" else base / "xclbins"


def user_settings_path(cli_config: str | None = None) -> Path:
    """Path to the user's settings.json (overlay file, never overwritten by releases).

    It is the file oflm itself reads: the settings of a model store live in
    the store's own directory, NEXT to the models/ directory -- not inside
    it. oflm-add used to write them inside, where oflm never looked, and a
    model registered that way stayed invisible to `oflm list`.
    """
    if cli_config:
        return Path(cli_config)
    env_path = os.environ.get("OFLM_SETTINGS_PATH")
    if env_path:
        return Path(env_path)
    # Next to the models directory: the directory that holds it.
    models_root = models_root_dir(None)
    return models_root.parent / "settings.json"


# Backward compat alias
def user_registry_path(cli_config: str | None = None) -> Path:
    """Deprecated alias for user_settings_path()."""
    return user_settings_path(cli_config)


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


def derive_tag(dir_name, explicit=None, size_guess=None):
    if explicit:
        return explicit
    size, rest = _extract_size(_strip_npu2(dir_name))
    if not size:
        # Chaotic repo slugs ("My-Finetune-Q4_K_M") carry no size marker; a
        # byte-derived guess keeps the install working (kernel matching keys
        # off the tag size). The bucket stays repo-derived, so two customs of
        # the same size never collide.
        if not size_guess:
            raise SystemExit(
                f"Could not derive a size from '{dir_name}' (no 'NNb' marker). "
                "Pass --tag name:size."
            )
        size, rest = size_guess, _strip_npu2(dir_name)
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


def _strip_org(name):
    """Registry names carry a leading "Org-" (e.g. "BAAI-bge-base-en-v1.5"); HF
    repo slugs drop it ("bge-base-en-v1.5"). Return the part after it."""
    return name.split("-", 1)[1] if "-" in name else name


def match_npue_official(system_registry, family, dir_name):
    """Best official NPU-embedding entry for `family` whose name best matches
    `dir_name`. Registry names sometimes carry a leading "Org-" prefix the HF
    repo slug omits (e.g. "BAAI-bge-base-en-v1.5" vs "bge-base-en-v1.5"), and
    sometimes do not ("nomic-embed-text-v1.5"). Score BOTH the org-stripped name
    and the full name against the dir tokens and keep the best, requiring a >=2
    token common prefix (tolerant of the prefix, unlike match_official_entry).
    Returns the 4-tuple (common, bucket, size, info) or None."""
    best = None
    dir_tokens = re.split(r"[-_ ]+", dir_name)
    for bucket, sz, info in _official_entries(system_registry, family):
        name = info.get("name", "")
        if not name:
            continue
        candidates = [re.split(r"[-_ ]+", name)]
        stripped = _strip_org(name)
        if stripped != name:
            candidates.append(re.split(r"[-_ ]+", stripped))
        common = 0
        for toks in candidates:
            c = 0
            for x, y in zip(toks, dir_tokens):
                if x.lower() != y.lower():
                    break
                c += 1
            common = max(common, c)
        if common >= 2 and (best is None or common > best[0]):
            best = (common, bucket, sz, info)
    return best


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


def derive_family(system_registry, dir_name, explicit=None, base_entry=None,
                  content_families=()):
    """details.family for this install.

    The explicit --family wins; then the official entry's own family; then
    what the model IS (the content routes: the kernels it needs, the base
    model it was built on); and last the name, which is the weakest signal --
    a fine-tune's name usually says nothing about its family ("Ornith-1.5-9B"
    is a Qwen3.5-9B).
    """
    if explicit:
        return explicit
    if base_entry:
        fam = base_entry.get("details", {}).get("family")
        if fam:
            return fam
    for fam in content_families:
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


def family_of_official(system_registry, name):
    """The registry's details.family for an official model's name, or None."""
    if not name:
        return None
    for sizes in system_registry.get("models", {}).values():
        for info in sizes.values():
            if info.get("name") == name:
                return (info.get("details") or {}).get("family")
    return None


def _model_key(name):
    """The comparable name of a model: the registry's container name and the
    HF repo slug name the same model ("Qwen3.5-9B-NPU2" / "Qwen3.5-9B")."""
    n = (name or "").split("/")[-1].lower()
    for suf in ("-opennpu2", "-npu2"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return n


def family_from_base_chain(system_registry, bases):
    """The family of the official model a fine-tune is built on.

    A fine-tune's README names its base model, and the official registry says
    that base's family. The fine-tune needs the base's kernels (the same
    shape, the same engine), so this is where its family comes from when its
    own name says nothing. Returns (family, official name).
    """
    for b in bases or ():
        key = _model_key(b)
        for sizes in system_registry.get("models", {}).values():
            for info in sizes.values():
                name = info.get("name") or ""
                if _model_key(name) != key:
                    continue
                fam = (info.get("details") or {}).get("family")
                if fam:
                    return fam, name
    return None, None


def _model_spec_class():
    """The recipes' ModelSpec, or None when there is no open_kernels checkout."""
    root = open_kernels_checkout()
    if root is None:
        return None
    added = str(root)
    inserted = added not in sys.path
    if inserted:
        sys.path.insert(0, added)
    try:
        from recipes.spec import ModelSpec
        return ModelSpec
    except Exception:
        return None
    finally:
        if inserted:
            try:
                sys.path.remove(added)
            except ValueError:
                pass


def family_from_kernels(system_registry, spec, roots):
    """The family of the installed kernel set whose GEOMETRY this model needs.

    The kernels decide the engine, and so the family: a fine-tune of a
    supported model runs on the base model's kernels -- the depth is the
    manifest's business, not the kernels' -- and the installed set's directory
    is named after the official model, which the registry ties to a family.
    The match is the kernel geometry (the recipes' kernel_hash), so it is the
    kernels that decide, not the depth or the tokenizer's id count.
    Returns (family, set name).
    """
    ModelSpec = _model_spec_class()
    if spec is None or ModelSpec is None:
        return None, None
    want = spec.kernel_hash()
    for root in roots:
        if not root:
            continue
        for m in sorted(Path(root).glob("*/open_kernels/manifest.json")):
            try:
                d = json.loads(m.read_text(encoding="utf-8"))
                s = d.get("spec")
                if not s or ModelSpec.from_dict(s).kernel_hash() != want:
                    continue
            except Exception:
                continue
            name = m.parents[1].name
            fam = family_of_official(system_registry, name)
            if fam:
                return fam, name
            # A set the registry does not list (our own conversion of a model we
            # do not ship as a container) still knows what it is: the manifest's
            # spec carries the family the kernels were built for.
            laid = s.get("family") if isinstance(s, dict) else None
            if laid:
                return laid, name
    return None, None


def spec_from_config_dict(cfg):
    """The ModelSpec a config.json describes, or None (no recipes, or a shape
    they do not know)."""
    ModelSpec = _model_spec_class()
    if ModelSpec is None or not isinstance(cfg, dict):
        return None
    try:
        return ModelSpec.from_hf_config(cfg)
    except Exception:
        return None


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


def kernel_manifest_dir(name, roots):
    """The kernel set directory called `name` in any root, or None."""
    if not name:
        return None
    for root in roots:
        if not root:
            continue
        d = Path(root) / name
        if (d / "open_kernels" / "manifest.json").is_file():
            return d
    return None


def official_repo_id(entry, registry=None):
    """The Hugging Face repo id of an official entry ("Atomic-Germ/Nanbeige4.1-3B-NPU2").

    The registry writes its own repositories' urls with `{hf_owner}`; the engine
    fills it from the list's "hf_owner" (or OFLM_HF_OWNER). oflm add has the same
    registry in hand, so the same substitution names the repo to climb to.
    """
    if not entry:
        return None
    url = entry.get("url") or ""
    owner = os.environ.get("OFLM_HF_OWNER") or (registry or {}).get("hf_owner", "")
    url = url.replace("{hf_owner}", owner or "")
    if "huggingface.co/" in url:
        return url.split("huggingface.co/", 1)[1].strip("/")
    return entry.get("name") or None


def fill_chat_template(target, repo, bases, modelscope=False):
    """Merge a chat template in from the model's chain when the installed
    tokenizer_config.json has none.

    A quant repo's tokenizer_config.json is often minimal (Nanbeige's has no
    chat_template; its sibling Nanbeige4.2-3B-Base does), and `oflm serve` refuses
    a model with no template -- after the weights have loaded. The aux fetch stops
    at the FIRST base that has the file, which is not necessarily the one with the
    template, so look further only for this one field.
    """
    tc = Path(target) / "tokenizer_config.json"
    if not tc.is_file() or (Path(target) / "chat_template.jinja").is_file():
        return None
    try:
        cfg = json.loads(tc.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if cfg.get("chat_template"):
        return None
    tmp = Path(target) / ".tokenizer_config.from_base.json"
    try:
        for src in [repo] + list(bases):
            if Path(src).is_dir():
                continue
            tmp.unlink(missing_ok=True)     # fetch_from_repo skips an existing destination
            try:
                got = fetch_from_repo(src, "tokenizer_config.json", tmp, modelscope=modelscope,
                                      force=True)
            except Exception:
                continue
            if not got or not tmp.is_file():
                continue
            try:
                d = json.loads(tmp.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            t = d.get("chat_template")
            if t:
                cfg["chat_template"] = t
                tc.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
                log(f"[INFO] chat_template from {src} (the installed tokenizer_config had none)")
                return src
        log("[WARN] no chat_template in this model's chain; `oflm serve` will refuse it "
            "(add one to tokenizer_config.json or a chat_template.jinja)")
        return None
    finally:
        tmp.unlink(missing_ok=True)


def reconcile_config_json(target, kernel_dir):
    """Make config.json carry the keys the KERNEL SET validates against.

    A GGUF repo's config.json -- or the finetune base's, which is where add
    takes it from -- is often not the full HF config: empero-ai/Qwen3.8-4B
    ships one without `layer_types` and without `full_attention_interval`, so
    the model INSTALLS (the geometry matched by kernel_hash) and then the
    engine refuses it at load, after the multi-GB download:

        open_qwen36: <model>: config.json lacks 'layer_types'

    The kernel set's manifest already holds the exact values the engine will
    compare against (`hf_config_check`, plus `hf_config_defaults` for a key
    whose absence is legal). A key the config OMITS and the set REQUIRES is
    filled from there: the install already matched this model's geometry, so
    the set is the authority on what it can serve -- and `layer_types` is
    exactly the key a finetune's config drops, because it is redundant for a
    loader that infers the pattern from `full_attention_interval`.

    A key the config CONTRADICTS is never overwritten: that is a different
    model, and the engine must still refuse it. Those are reported instead, at
    install time rather than at the first `oflm run`.

    Returns (filled, conflicting) key lists.
    """
    cfg_path = Path(target) / "config.json"
    mj = Path(kernel_dir) / "open_kernels" / "manifest.json" if kernel_dir else None
    if not cfg_path.is_file() or mj is None or not mj.is_file():
        return [], []
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        man = json.loads(mj.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], []
    check = man.get("hf_config_check") or {}
    defaults = man.get("hf_config_defaults") or {}
    if not isinstance(cfg, dict):
        return [], []

    # 1. A MULTIMODAL config wraps the text tower: empero-ai/Qwen3.8-4B (and
    # Qwen/Qwen3.5-4B itself) keep hidden_size, layer_types,
    # full_attention_interval ... under `text_config`, and the engine merges the
    # GGUF's FLAT metadata over the TOP level -- which supplies hidden_size but
    # has no layer_types to supply ("config.json lacks 'layer_types'" at load,
    # after a 2.8 GB install). Lift the tower's keys up where the top level has
    # none; the wrapper is left intact, so a vision engine still finds it.
    flattened = []
    for wrapper in ("text_config", "llm_config", "language_config"):
        inner = cfg.get(wrapper)
        if not isinstance(inner, dict):
            continue
        for k, v in inner.items():
            if k not in cfg:
                cfg[k] = v
                flattened.append(k)
        break

    def expected(key):
        want = check[key]
        return want[0] if isinstance(want, list) else want

    filled, conflicting = [], []
    for key, want in check.items():
        if key == "layer_types":
            got = cfg.get("layer_types")
            if got is None and cfg.get("full_attention_interval") and cfg.get("num_hidden_layers"):
                iv = int(cfg["full_attention_interval"])
                n = int(cfg["num_hidden_layers"])
                if iv > 0:
                    got = ["full_attention" if (l + 1) % iv == 0 else "linear_attention"
                           for l in range(n)]
            if got is None:
                cfg[key] = list(want)
                filled.append(key)
            elif list(got) != list(want):
                conflicting.append(key)
            continue
        if key in cfg:
            got = cfg[key]
            ok = got in want if isinstance(want, list) else got == want
            if not ok:
                conflicting.append(key)
            continue
        if key in defaults:                # absence is legal (the manifest names a default)
            continue
        cfg[key] = expected(key)
        filled.append(key)

    if filled or flattened:
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    if flattened:
        log(f"[INFO] config.json: flattened {len(flattened)} key(s) from text_config "
            f"(the engine validates at the top level)")
    if filled:
        log(f"[INFO] config.json: filled {', '.join(filled)} from the kernel set "
            f"(the model's config omits it; the engine validates against the set)")
    if conflicting:
        log(f"[WARN] config.json disagrees with the kernel set on {', '.join(conflicting)}; "
            f"the engine will refuse this model at load. The set was built for another "
            f"configuration -- check the base model and the quant repo.")
    return filled, conflicting


def token_id_of(target, token):
    """The id of a special token by its STRING, from the installed tokenizer.json.

    A tokenizer_config can name `bos_token` / `eos_token` and never give the id
    (Hy-MT2: `<|startoftext|>` with no bos_token_id), and the engine needs the
    integer. tokenizer.json has both: the added_tokens list and the base vocab.
    """
    if not token or not isinstance(token, str):
        return None
    tj = Path(target) / "tokenizer.json"
    try:
        t = json.loads(tj.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for a in t.get("added_tokens", []) or []:
        if a.get("content") == token and isinstance(a.get("id"), int):
            return a["id"]
    v = (t.get("model") or {}).get("vocab")
    if isinstance(v, dict) and isinstance(v.get(token), int):
        return v[token]
    return None


def normalize_tokenizer_config(target, repo, bases, modelscope=False):
    """FLM reads eos_token_id (array) / bos_token_id from tokenizer_config.json;
    stock HF repos keep them in generation_config.json -- merge them over."""
    tc = target / "tokenizer_config.json"
    if not tc.is_file():
        return
    cfg = json.loads(tc.read_text(encoding="utf-8"))
    # A token NAMED but not numbered: resolve the id from tokenizer.json BEFORE any
    # early return, so a repo that has the tokenizer but no generation_config still
    # gets its ids (Hy-MT2: `<|startoftext|>` with no bos_token_id).
    _ids_fixed = False
    for _tok_key, _id_key in (("bos_token", "bos_token_id"), ("eos_token", "eos_token_id")):
        if cfg.get(_id_key) is not None:      # a null id is a MISSING id, not a set one
            continue
        _tok = cfg.get(_tok_key)
        if isinstance(_tok, dict):
            _tok = _tok.get("content")
        _tid = token_id_of(target, _tok) if isinstance(_tok, str) else None
        if _tid is None and (Path(target) / "model.gguf").is_file():
            # The GGUF's own metadata names the ids (Hy-MT2: bos 127958, and the
            # string <|startoftext|> is not in tokenizer.json's vocab to look up).
            inv = read_gguf_inventory(str(Path(target) / "model.gguf"), limit_mb=32)
            _kv = inv[0] if inv else {}
            _gg = _kv.get(f"tokenizer.ggml.{_id_key}")
            if isinstance(_gg, int):
                _tid = _gg
        if _tid is not None:
            cfg[_id_key] = [_tid] if _id_key == "eos_token_id" else _tid
            _ids_fixed = True
    if _ids_fixed:
        # persist BEFORE the need_eos/need_bos early return: a repo whose only gap was
        # the id (Hy-MT2's null bos_token_id) has nothing left to fetch, and returning
        # without writing left the engine to refuse the model.
        tc.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    need_eos = "eos_token_id" not in cfg or not isinstance(cfg.get("eos_token_id"), list)
    need_bos = cfg.get("bos_token") is not None and "bos_token_id" not in cfg
    if not (need_eos or need_bos):
        return
    g = None
    gc = target / "generation_config.json"
    repo_gen = None
    # A local-directory install passes the directory path as `repo`; fetching
    # generation_config.json from it as an HF id is meaningless (and raises),
    # so only attempt it when `repo` is a remote id.
    if not Path(repo).is_dir():
        try:
            repo_gen = fetch_from_repo(repo, "generation_config.json", gc, modelscope=modelscope)
        except Exception:
            repo_gen = None
    # Base fetches here must not fail the install: a gated base (401/403)
    # simply has no generation_config to offer, and the caller already fell
    # back to the GGUF-embedded tokenizer when that happened.
    base_gens = []
    for b in bases:
        try:
            base_gens.append(fetch_from_repo(b, "generation_config.json", gc, modelscope=modelscope))
        except Exception:
            base_gens.append(None)
    for cand in [gc, repo_gen] + base_gens:
        if cand and Path(cand).is_file():
            try:
                g = json.loads(Path(cand).read_text(encoding="utf-8"))
            except Exception:
                g = None
            if g and (need_eos and isinstance(g.get("eos_token_id"), list) or
                      need_bos and isinstance(g.get("bos_token_id"), int)):
                break
    source = "generation_config.json"
    # "found a generation_config.json" is not the same as "found the id": Qwen3-4B-Base
    # ships one WITHOUT eos_token_id, so the loop kept the file, the fallback never ran,
    # and the engine refused the model ("eos_token_id is missing or not an array").
    need_fallback = not g or (need_eos and not isinstance(g.get("eos_token_id"), list))
    if need_fallback:
        g = dict(g or {})
        source = "config.json"
        # No generation_config.json anywhere in the chain (the Qwen3.5 line ships
        # none). The config.json names the same ids: the model's own eos and the
        # text tower's -- the engine reads them as an array, so give it one.
        cfg_path = target / "config.json"
        if cfg_path.is_file():
            try:
                mcfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                mcfg = {}
            ids: list[int] = []
            for c in (mcfg, mcfg.get("text_config") or {}):
                v = c.get("eos_token_id")
                if isinstance(v, int):
                    ids.append(v)
                elif isinstance(v, list):
                    ids += [x for x in v if isinstance(x, int)]
            if ids:
                g["eos_token_id"] = sorted(set(ids))
            bos = mcfg.get("bos_token_id")
            if isinstance(bos, int):
                g["bos_token_id"] = bos
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
        log("[INFO] merged eos/bos ids into tokenizer_config.json (from " + source + ")")


def _parse_base_model_frontmatter(text):
    """base_model id(s) from already-read README.md frontmatter text."""
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


def readme_base_model(repo_id, modelscope=False):
    """The base model id(s) from the repo's README.md YAML frontmatter.

    GGUF-quant repos (mradermacher etc.) carry no tokenizer/config files, but
    their model card names the original model, whose repo has everything. The
    field is `base_model:` either inline (`[[Org/Name]]` wikilink form or
    plain) or a `- Org/Name` list (merge lineages; the first entry is the
    closest to the quantized checkpoint).

    When ``repo_id`` is a local directory holding a README.md (a local-dir
    install of a quantized model), the frontmatter is parsed from disk so the
    same base_model crawl applies without touching the network.
    """
    local = Path(repo_id)
    if local.is_dir() and (local / "README.md").is_file():
        try:
            return _parse_base_model_frontmatter(
                (local / "README.md").read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return []
    url = (f"https://www.modelscope.ai/models/{repo_id}/resolve/master/README.md" if modelscope
           else f"https://huggingface.co/{repo_id}/raw/main/README.md")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=_hf_headers()), timeout=60) as r:
            text = r.read().decode("utf-8", errors="replace")
    except Exception:
        return []
    return _parse_base_model_frontmatter(text)


def readme_base_chain(repo_id, modelscope=False, max_depth=3):
    """All base models reachable by climbing README frontmatter, in order.

    A quant repo names its finetune; the finetune names ITS base; only the
    base carries the tokenizer/config files. Each level is tried in order, so
    the closest repo wins per file. Cycles and repeats are visited once, and
    the climb stops after max_depth levels -- every level is a network fetch.
    """
    ordered, seen, frontier = [], set(), [repo_id]
    for _ in range(max_depth):
        nxt = []
        for repo in frontier:
            if repo in seen:
                continue
            seen.add(repo)
            for base in readme_base_model(repo, modelscope):
                if base not in seen and base not in ordered:
                    ordered.append(base)
                    nxt.append(base)
        frontier = nxt
        if not frontier:
            break
    return ordered


def peek_config_dict(repo_id, modelscope=False, bases=(), local_dirs=()):
    """config.json contents for detection only (family/size guesses), or None.

    Local files first (the install dir, an HF/MS cache snapshot): no network.
    Then the repo itself, then its base-model chain -- stock quant repos carry
    no config, but the checkpoint they quantize does. Nothing here is
    installed; the real fetch happens later and is verified there."""
    for local in local_dirs:
        if not local:
            continue
        cfg = Path(local) / "config.json"
        if cfg.is_file():
            try:
                return load_json(cfg)
            except Exception:
                pass
    try:
        with tempfile.TemporaryDirectory(prefix="oflm-add-peek-") as tmp:
            dest = Path(tmp) / "config.json"
            for repo in [repo_id, *bases]:
                if not repo:
                    continue
                try:
                    got = fetch_from_repo(repo, "config.json", dest,
                                          modelscope=modelscope, force=True)
                except Exception:
                    got = None
                if got:
                    try:
                        return load_json(got)
                    except Exception:
                        return None
    except Exception:
        pass
    return None


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


def fetch_assets(repo_id, target, modelscope=False, verify=True, force=False, quiet=False, gguf_name=None, want=None):
    """Populate target/ with the model files; returns the list of files present.

    With gguf_name set (the GGUF-direct path) the wanted set is the tokenizer
    files plus that one GGUF, stored canonically as model.gguf. Otherwise the
    default is ALL_FILES; an explicit `want` list overrides it (used by the
    NPU-embedding path, which fetches a safetensors checkpoint, not model.q4nx)."""
    obtained = []
    target.mkdir(parents=True, exist_ok=True)
    if gguf_name:
        want_list = GGUF_FILES + GGUF_OPTIONAL
    else:
        want_list = want if want is not None else ALL_FILES
    if modelscope:
        domain, entries = ms_file_tree(repo_id)
        for fname in want_list:
            remote = gguf_name if fname == "model.gguf" else fname
            if remote not in entries:
                continue
            dest = target / fname
            if dest.is_file() and not force:
                obtained.append(fname)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            meta = entries[remote]
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

    # Hugging Face: local cache first, then the tree API. Keep nested paths too
    # (e.g. 1_Pooling/config.json for NPU-embedding models).
    entries = {}
    for e in hf_file_tree(repo_id):
        p = e.get("path")
        if p:
            entries[p] = e
    for fname in want_list:
        remote = gguf_name if fname == "model.gguf" else fname
        if remote not in entries:
            continue
        dest = target / fname
        if dest.is_file() and not force:
            obtained.append(fname)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
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


def copy_from_dir(src_dir, target, force=False, gguf_name=None, want=None):
    obtained = []
    want_list = (GGUF_FILES + GGUF_OPTIONAL) if gguf_name else (want if want is not None else ALL_FILES)
    for fname in want_list:
        src = src_dir / (gguf_name if fname == "model.gguf" else fname)
        if src.is_file():
            dest = target / fname
            if dest.is_file() and not force:
                obtained.append(fname)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_file() or dest.is_symlink():
                dest.unlink()
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
    return estimate_size_dict(cfg)


def estimate_size_dict(cfg):
    """Registry 'size' (bytes) from an already-parsed config.json dict."""
    if not isinstance(cfg, dict):
        return None
    try:
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
    except Exception:
        return None


def size_token_from_bytes(num_bytes):
    """A tag size marker ('4.4b', '0.3b') from a byte count, or None."""
    if not num_bytes:
        return None
    return f"{round(num_bytes / 1e9, 1):g}b"


# Bytes per parameter for the GGUF quants the open kernels ingest, used to
# guess a registry size for repos whose names carry no size marker. Only an
# install-time tag/kernel-matching hint (logged as a guess); the real weights
# are verified by hash, never by this number.
GGUF_BYTES_PER_PARAM = {
    "Q4_0": 0.5625,
    "Q4_1": 0.625,
    "Q8_0": 1.0625,
    "Q4_K": 0.5625,
    "Q6_K": 0.8125,
}


def gguf_size_token(gguf_bytes, quant):
    """A tag size marker from a GGUF's byte size and quant family, or None."""
    bpp = GGUF_BYTES_PER_PARAM.get(quant)
    if not gguf_bytes or not bpp:
        return None
    return size_token_from_bytes(gguf_bytes / bpp)


def gguf_byte_size(repo, modelscope, local_dir, cache_dir, gguf_name, tree):
    """On-disk or listed byte size of the chosen GGUF, or None.

    Local files (install dir, HF/MS cache snapshot) stat first: no network.
    Otherwise the size comes from the repo tree listing already fetched for
    the GGUF scan."""
    for d in (local_dir, cache_dir):
        if d is None:
            continue
        cand = Path(d) / gguf_name
        try:
            if cand.is_file():
                return cand.stat().st_size
        except OSError:
            pass
    try:
        if modelscope and isinstance(tree, dict):
            return (tree.get(gguf_name) or {}).get("Size") or None
        if isinstance(tree, list):
            for entry in tree:
                if entry.get("path") == gguf_name:
                    return (entry.get("lfs") or {}).get("size") or entry.get("size")
    except Exception:
        pass
    return None


def link_npue_design(system_prefixes, user_root, design_family, force=False, quiet=False):
    """Symlink the NPU-embedding design set (compiled kernels, keyed by GEMM
    geometry) into the user's xclbins so the NpuEmbeddings engine can find it at
    <xclbin_prefix>/xclbins/<npue_design_family>/gemm_rtp/design.json.

    The design sets ship under one of the system xclbin roots (e.g.
    /usr/local/share/oflm/xclbins); this mirrors the family-kernel symlink the
    chat models get, so an `oflm-add` install needs no manual step to run on NPU."""
    src = None
    for p in system_prefixes:
        cand = Path(p) / "xclbins" / design_family
        if (cand / "gemm_rtp" / "design.json").is_file():
            src = cand
            break
    if src is None:
        if not quiet:
            log(f"[WARN] No npue design set for '{design_family}' under any system "
                 "xclbin root; the engine can pack the container but will not run on "
                 "NPU without it. Build it with npu_offload/gemm_rtp/build.ps1.")
        return
    _link_one(user_root, design_family, str(src), force, quiet)


def build_entry(base_entry, dir_name, files, size):
    # Deep copy: the new entry shares no nested objects with the registry it
    # was seeded from, so stamping details below cannot corrupt the curated
    # entry it was derived from (a shallow dict() copy once leaked
    # details.format/family back into the overlay's official entry).
    entry = copy.deepcopy(base_entry) if base_entry else {}
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


def register(
    entry: dict,
    settings_path: Path,
    system_registry: dict,
    force: bool = False,
    name_override: str | None = None,
    tag: str | None = None,
) -> str:
    """Register a model in the user's settings.json (not model_list.json).

    Returns the tag the model was registered under.
    Raises SystemExit on collision (without force/name_override).

    ``tag`` is the tag derived in main() from the repo name (including
    --tag); it is used as-is. Reconstructing a tag from the entry metadata
    would be wrong: build_entry() carries no ``type`` and its ``size`` is a
    numeric byte count, not the tag suffix -- such an entry would register
    as ``unknown:3000000000``.
    """
    # Hold the same lock the C++ SettingsLock takes, for the whole
    # read-modify-write: two concurrent `oflm add`s (or an add racing a
    # `config set`) would otherwise both read the old document and the
    # second atomic rename would silently drop the first one's changes.
    with _settings_lock(settings_path):
        # Load user settings (create if missing)
        if settings_path.exists():
            with open(settings_path) as f:
                settings = json.load(f)
        else:
            settings = {
                "version": 1,
                "defaults": {},
                "settings": {},
                "models": {},
            }

        # Determine tag: --name wins, then the tag derived in main();
        # the entry-metadata fallback exists for callers without a derived
        # tag (the tests).
        model_type = entry.get("type", "unknown")
        size = entry.get("size", "unknown")
        final_tag = name_override or tag or f"{model_type}:{size}"

        if ":" in final_tag:
            model_type, size = final_tag.split(":", 1)

        # Collision check: shipped registry -- on the FINAL destination tag,
        # and unconditionally. Checking only when --name was absent let
        # `--name qwen3.8:27b` write a user entry that shadows the shipped
        # one through the merged overlay. Exception: the NPU-embedding flow
        # deliberately re-installs the official design under its shipped tag
        # (the engine routes by that exact tag); a same-design re-install
        # is not a takeover of the name.
        shipped_entry = (system_registry.get("models", {})
                         .get(model_type, {}).get(size))
        if shipped_entry is not None:
            same_npue_design = (
                entry.get("npue_design_family")
                and shipped_entry.get("npue_design_family")
                and entry["npue_design_family"] == shipped_entry["npue_design_family"]
            )
            if not same_npue_design:
                raise SystemExit(
                    f"Model {final_tag} already exists in the shipped registry. "
                    f"Use --name to install under a different name."
                )

        # Collision check: user settings
        if (model_type in settings.get("models", {}) and
            size in settings["models"][model_type]):
            if not force:
                raise SystemExit(
                    f"Model {final_tag} is already installed. Use --force to overwrite."
                )

        # Write complete entry to settings.models
        settings.setdefault("models", {}).setdefault(model_type, {})[size] = entry

        # Ensure parent directory exists
        settings_path.parent.mkdir(parents=True, exist_ok=True)

        # Save atomically
        save_json_atomic(settings_path, settings)

    return final_tag


# ------------------------------------------------------------------- xclbins

def _link_one(user_root, name, target, force, quiet):
    """Create symlink user_root/name -> target, reusing it when already correct.

    The junction fallback covers Windows, where directory symlinks need a
    privilege only admin / developer mode grants; _make_dir_link tries it.
    """
    link = user_root / name
    if link.exists() or link.is_symlink():
        # resolve() covers a junction too, which does not answer to readlink
        if link.resolve() == Path(target).resolve():
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
    if not _make_dir_link(link, target):
        raise SystemExit(
            f"Could not link {link} -> {target}. Windows grants the symlink privilege to "
            f"admins and developer mode only, and the junction fallback failed too."
        )
    if not quiet:
        log(f"[INFO] Linked xclbins: {link} -> {target}")


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
    target = str(src)
    # Link under the installed model directory name (used by per-model engine
    # lookups that resolve kernels by the model's directory).
    _link_one(user_root, dir_name, target, force, quiet)
    # Also link under the kernel family/source name. open_embedding's NPU matmul
    # probe looks the matmul kernels up by the family directory (e.g.
    # Embedding-Gemma-300M-OpenNPU2), not by the model directory name, so the
    # family-named link is what makes the NPU path discoverable.
    if source_name != dir_name:
        _link_one(user_root, source_name, target, force, quiet)


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
    in <repo>/utilities/oflm-add), then under the working directory.
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

    The spec is derived the way the recipes do it: from the model's own
    weights where it has them (a GGUF's own metadata) and from config.json
    plus the tokenizer's real vocab otherwise (the model.q4nx safetensors
    header gives the per-role weight format; no weight byte is read).
    """
    spec = model_spec(model_dir)
    if spec is None:
        return None, ("no spec for this model (no open_kernels/recipes checkout, "
                      "or a shape they do not know)")
    return spec.spec_hash(), "spec from the model's own metadata"


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


def _link_target(link, target):
    """What the link should point at: a relative path when the two live in
    the same tree, the absolute path otherwise.

    A link from a model directory to a kernel set is inside one tree (a dev
    checkout, or an install), and a relative one survives the tree being
    moved or copied -- an absolute one does not, and a link that points at
    the machine the tree came from is a link to nowhere. When the two are
    in different trees (a user directory and an install prefix), the
    absolute path is the honest one.
    """
    target = Path(target).resolve()
    try:
        common = Path(os.path.commonpath([Path(link).parent.resolve(), target]))
    except ValueError:          # different drives on Windows
        return target
    if str(common) == common.anchor:  # the root: nothing in common but "/"
        return target
    try:
        return Path(os.path.relpath(target, Path(link).parent.resolve()))
    except ValueError:
        return target


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
    target = _link_target(link, kernel_dir)
    if _make_dir_link(link, target):
        if not quiet:
            log(f"[INFO] Linked open kernels: {link} -> {target}")
        return True
    log(f"[WARN] Could not link {link} -> {target} (a Windows symlink needs "
        "developer mode or admin). Run oflm with:")
    log(f'           OFLM_OPEN_KERNELS_DIR="{kernel_dir}"')
    return False


def model_spec(model_dir):
    """The ModelSpec for an installed model directory, or None (no recipes
    checkout, or a shape they do not know).

    A GGUF-direct model is described by ITS OWN weights. The repo's
    config.json is a claim about the model the repo was built from, and for a
    fine-tune that is often the BASE: empero-ai/Qwen3.8-9B-Distill-GGUF
    ships no config.json at all, and the one its README names (the base's)
    says 32 layers where the file has 33. The geometry that decides which
    kernels serve the model is the file's; the config.json is the fallback for
    a model that has no GGUF.
    """
    root = open_kernels_checkout()
    if root is None:
        return None
    added = str(root)
    inserted = added not in sys.path
    if inserted:
        sys.path.insert(0, added)
    try:
        from recipes.load import spec_from_model_dir
        from recipes.spec import ModelSpec
        gguf = Path(model_dir) / "model.gguf"
        if gguf.is_file():
            try:
                inv = read_gguf_inventory(str(gguf))
                if inv:
                    names = [x[0] if isinstance(x, (list, tuple)) else x for x in inv[1]]
                    spec = ModelSpec.from_gguf_metadata(inv[0], names)
                    # the tokenizer's id count (the logits above it are undefined)
                    try:
                        spec = dataclasses.replace(spec, real_vocab=spec_from_model_dir(Path(model_dir)).real_vocab)
                    except Exception:
                        pass
                    return spec
            except Exception:
                pass
        return spec_from_model_dir(Path(model_dir))
    except Exception:
        return None
    finally:
        if inserted:
            try:
                sys.path.remove(added)
            except ValueError:
                pass


def sibling_open_kernels(model_dir, roots, quiet=False, prefer=None):
    """A kernel set for this model built from the kernels of an installed set
    that serves the same GEOMETRY, with this model's own manifest.

    The kernels are width-keyed: the depth is the manifest's business (the
    layer loop it drives), so a fine-tune of a different depth runs on the base
    model's kernels -- only the manifest differs. A model whose shape no
    installed set covers needs a build; a model whose DEPTH differs does not,
    and asking for a rebuild here would put a kernel build between the user and
    a model that runs today.

    The sibling set is a directory of the base's kernel files (hard-linked
    where the filesystem allows it) plus this model's manifest, and it is what
    the model links to. Returns the directory, or None.
    """
    spec = model_spec(model_dir)
    if spec is None:
        return None
    want = spec.kernel_hash()
    # The preferred base first: the set the GGUF gate picked (the freshest one
    # with the f32-scale twins), so the sibling carries them too.
    manifests = []
    if prefer:
        manifests.append(Path(prefer) / "manifest.json")
    for root in roots:
        if not root:
            continue
        manifests += sorted(Path(root).glob("*/open_kernels/manifest.json"))
    for m in manifests:
        try:
            d = json.loads(m.read_text(encoding="utf-8"))
            s = d.get("spec")
            if not s:
                continue
            ModelSpec = _model_spec_class()
            if ModelSpec is None or ModelSpec.from_dict(s).kernel_hash() != want:
                continue
        except Exception:
            continue
        base = m.parent
        out = Path(model_dir) / "open_kernels"
        try:
            if out.exists() and not out.is_symlink():
                shutil.rmtree(out)
            out.mkdir(parents=True, exist_ok=True)
            for f in base.rglob("*"):
                if not f.is_file():
                    continue
                rel = f.relative_to(base)
                dst = out / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    dst.unlink()
                try:
                    os.link(f, dst)                    # the same bytes, no copy
                except OSError:
                    shutil.copy2(f, dst)
            from recipes.manifest import manifest
            (out / "manifest.json").write_text(json.dumps(manifest(spec), indent=1) + "\n",
                                               encoding="utf-8")
        except Exception as exc:
            if not quiet:
                log(f"[WARN] could not build the sibling kernel set: {exc}")
            return None
        if not quiet:
            log(f"[INFO] open kernels: {out} (the kernels of {base.parent.name}, "
                f"with this model's own manifest -- same geometry, "
                f"{spec.num_layers} layers)")
        return out
    return None


def setup_open_kernels(model_dir, dir_name, roots, override=None, force=False, quiet=False, prefer=None):
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
    # No set was built for this shape exactly. The kernels are width-keyed, so
    # a set that serves the same geometry does: only the manifest (the layer
    # loop) differs, and it is written here -- no build, no kernel set to
    # export, nothing for the user to run.
    sibling = sibling_open_kernels(model_dir, roots, quiet=quiet, prefer=prefer)
    if sibling:
        return True
    checkout = open_kernels_checkout()
    script = (checkout / "export_qwen36_kernels.py") if checkout else Path("open_kernels/export_qwen36_kernels.py")
    log(f"[INFO] No installed open kernel set has spec_hash {spec_hash[:19]}; "
        "the closed kernels stay in charge. Build one with:")
    log(f'           python "{script}" --model-dir "{model_dir}"')
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
# ------------------------------------------------------- GGUF introspection
#
# Stock quant repos (bartowski/*-GGUF and friends) ship ONLY the GGUFs: no
# config, no tokenizer, and a README whose base_model is often a gated repo
# (meta-llama/* answers 403 anonymously). Two stdlib-only readers make that
# installable anyway:
#
#   1. read_gguf_inventory: header + KV + tensor infos from a byte prefix
#      (Range request for remotes, head read for local files). Verifies the
#      chosen file's tensor types are ones the NPU packers ingest AND match
#      the filename claim -- the Q4_0_4_4 trap (ik_llama matmul shuffles
#      misread as Q4_0) becomes a loud refusal instead of silent garbage.
#   2. extract_tokenizer_from_gguf: tokenizer.json + tokenizer_config.json +
#      chat_template.jinja from the GGUF's embedded tokenizer, the last
#      resort when every base repo is unreachable or gated.
#
# ggml type numbers (llama.cpp ggml.h; cross-checked with
# src/gguf/gguf_file.hpp -- only metadata parsing needs these).
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
    27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 31: "Q4_0_4_4",
    32: "Q4_0_4_8", 33: "Q4_0_8_8",
}
# Tensor types the NPU packers ingest (pools.cpp std_perm_gguf for the
# quants, put/pack_norm for the small float weights).
GGUF_ENGINE_TYPES = {"F32", "F16", "BF16", "Q4_0", "Q4_1", "Q8_0", "Q4_K", "Q5_K", "Q6_K",
                     "IQ4_NL", "IQ4_XS"}


class _GgufTruncated(Exception):
    """The byte prefix ends mid-structure; fetch more or give up."""


class _GgufCursor:
    def __init__(self, buf):
        self.buf = buf
        self.off = 0

    def _take(self, n, what):
        if self.off + n > len(self.buf):
            raise _GgufTruncated(f"ends inside {what}")
        out = self.buf[self.off:self.off + n]
        self.off += n
        return out

    def u8(self):
        return self._take(1, "u8")[0]

    def u16(self):
        return struct.unpack("<H", self._take(2, "u16"))[0]

    def u32(self):
        return struct.unpack("<I", self._take(4, "u32"))[0]

    def u64(self):
        return struct.unpack("<Q", self._take(8, "u64"))[0]

    def i8(self):
        return struct.unpack("<b", self._take(1, "i8"))[0]

    def i16(self):
        return struct.unpack("<h", self._take(2, "i16"))[0]

    def i32(self):
        return struct.unpack("<i", self._take(4, "i32"))[0]

    def i64(self):
        return struct.unpack("<q", self._take(8, "i64"))[0]

    def f32(self):
        return struct.unpack("<f", self._take(4, "f32"))[0]

    def f64(self):
        return struct.unpack("<d", self._take(8, "f64"))[0]

    def bool(self):
        return self.u8() != 0

    def string(self):
        # surrogateescape preserves non-UTF8 token bytes round-trip exactly.
        return self._take(self.u64(), "string").decode("utf-8", "surrogateescape")

    def value(self, typ):
        readers = {0: self.u8, 1: self.i8, 2: self.u16, 3: self.i16,
                   4: self.u32, 5: self.i32, 6: self.f32, 7: self.bool,
                   10: self.u64, 11: self.i64, 12: self.f64}
        if typ == 8:
            return self.string()
        if typ == 9:
            etyp, count = self.u32(), self.u64()
            if etyp == 8:
                return [self.string() for _ in range(count)]
            return [self.value(etyp) for _ in range(count)]
        try:
            return readers[typ]()
        except KeyError:
            raise ValueError(f"unknown GGUF metadata type {typ}")


def parse_gguf_inventory(buf):
    """(kv, [(name, type_number, dims)]) from a GGUF header prefix.
    Raises _GgufTruncated when the prefix ends mid-structure, ValueError on
    a bad magic/version. Tensor DATA is never touched."""
    cur = _GgufCursor(buf)
    if cur._take(4, "magic") != b"GGUF":
        raise ValueError("not a GGUF file (bad magic)")
    version = cur.u32()
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version}")
    n_tensors, n_kv = cur.u64(), cur.u64()
    kv = {}
    for _ in range(n_kv):
        key = cur.string()
        typ = cur.u32()
        kv[key] = cur.value(typ)
    tensors = []
    for _ in range(n_tensors):
        name = cur.string()
        n_dims = cur.u32()
        dims = [cur.u64() for _ in range(n_dims)]
        tensors.append((name, cur.u32(), dims))
        cur.u64()  # data offset; the payload itself is never read
    return kv, tensors


def gguf_prefix_bytes(source, limit):
    """First `limit` bytes of a local path or an https URL (Range).
    A server that ignores Range still yields a usable prefix: the read is
    capped either way. Returns b"" when nothing could be read."""
    if isinstance(source, Path) or (isinstance(source, str) and not source.startswith("http")):
        try:
            with open(source, "rb") as f:
                return f.read(limit)
        except OSError:
            return b""
    try:
        req = urllib.request.Request(source, headers={**_hf_headers(), "Range": f"bytes=0-{limit - 1}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            out = bytearray()
            while len(out) < limit:
                chunk = resp.read(min(1024 * 1024, limit - len(out)))
                if not chunk:
                    break
                out += chunk
            return bytes(out)
    except Exception:
        return b""


def read_gguf_inventory(source, limit_mb=32):
    """(kv, tensors) for a GGUF, or None when the prefix is unreachable or
    too short to cover the metadata. `source` is a local path or URL."""
    buf = gguf_prefix_bytes(source, int(limit_mb * 1024 * 1024))
    if len(buf) < 24:
        return None
    try:
        return parse_gguf_inventory(buf)
    except _GgufTruncated:
        return None
    except ValueError:
        return None


def family_from_gguf_arch(gguf_source):
    """details.family from a GGUF's general.architecture, or None.

    The repo name is the primary signal (FAMILY_ALIASES); this covers repos
    whose names say nothing about the model (an arbitrary host of an
    embedding-gemma quant). Unreachable metadata yields None, not an error."""
    inv = read_gguf_inventory(gguf_source)
    arch = inv[0].get("general.architecture") if inv else None
    if not arch:
        return None
    return GGUF_ARCH_FAMILIES.get(str(arch).strip().lower())


def family_from_config_dict(cfg):
    """details.family from a HuggingFace config.json dict, or None.

    model_type first (exact); then the architectures list, substring-matched
    against FAMILY_ALIASES so Qwen3ForCausalLM-style class names resolve.
    Neither fires for unknown shapes -- the caller must ask for --family."""
    if not isinstance(cfg, dict):
        return None
    model_type = cfg.get("model_type")
    if isinstance(model_type, str):
        lowered_type = model_type.strip().lower()
        # Vision-language checkpoints need their vision weights and their own
        # kernels; never fold one silently onto a text family. Unknown VL
        # shapes refuse here so the caller passes --family explicitly.
        if "vl" in lowered_type and lowered_type not in CONFIG_MODEL_FAMILIES:
            return None
        family = CONFIG_MODEL_FAMILIES.get(lowered_type)
        if family:
            return family
    archs = cfg.get("architectures") or []
    if isinstance(archs, str):
        archs = [archs]
    for arch in archs:
        if not isinstance(arch, str):
            continue
        lowered = arch.lower()
        for prefix, family in FAMILY_ALIASES:
            if prefix.lower() in lowered:
                return family
    return None


def verify_gguf_choice(source, gguf_name, claimed_family):
    """The chosen file really holds what its name claims, in types the
    engine packs. Returns (ok, note): unreachable metadata warns and keeps
    the filename claim; a type mismatch refuses loudly."""
    inv = read_gguf_inventory(source)
    if inv is None:
        return True, "metadata unreachable; installing on the filename claim"
    kv, tensors = inv
    unknown = sorted({t for _, t, _ in tensors if t not in GGML_TYPE_NAMES})
    if unknown:
        return False, f"unknown ggml tensor types {unknown}; refusing (see q4nx-build)"
    bad = sorted({GGML_TYPE_NAMES[t] for _, t, _ in tensors
                   if GGML_TYPE_NAMES[t] not in GGUF_ENGINE_TYPES})
    if bad:
        return False, (f"tensors use {'/'.join(bad)}, which the NPU packers do not ingest "
                       "(have F32/F16/BF16/Q4_0/Q4_1/Q8_0/Q4_K/Q6_K); convert with q4nx-build")
    # llama.cpp's quantizers MIX types by importance: the body at the named quant,
    # the output projection and the attention V at a finer one (a Q4_1 file holds
    # Q6_K beside its Q4_1; an IQ4_XS file holds Q5_K). So the name is a hint, not
    # a contract: what must hold is that SOME tensor is of the claimed family --
    # otherwise the name is a lie and the file is not what was asked for.
    quants = {GGML_TYPE_NAMES[t] for _, t, _ in tensors
               if GGML_TYPE_NAMES[t] not in ("F32", "F16", "BF16")}
    fams = {gguf_quant_family(q) for q in quants}
    if claimed_family and claimed_family not in fams:
        return False, (f"the file's tensors are {'/'.join(sorted(quants))} -- none of them is the "
                       f"{claimed_family} its name claims; refusing rather than packing the wrong layout")
    arch = kv.get("general.architecture", "?")
    return True, f"{len(tensors)} tensors ({arch}), quant {sorted(quants)}"


def _bytes_to_unicode():
    """The GPT-2 byte mapping HF tokenizer.json files store BPE vocabs in.
    Printable bytes map to themselves; whitespace/controls map to U+0100 and
    up. Applied uniformly it is bijective over byte strings, so encode/decode
    round-trips exactly and the emitted file is self-consistent."""
    keep = set(range(0x21, 0x7F)) | set(range(0xA1, 0xAD)) | set(range(0xAE, 0x100))
    table, extra = {}, 0x100
    for b in range(256):
        if b in keep:
            table[b] = chr(b)
        else:
            table[b] = chr(extra)
            extra += 1
    return table


def _map_token(raw, table):
    return "".join(table[b] for b in raw)


def extract_tokenizer_from_gguf(gguf_path, target, force=False):
    """tokenizer.json + tokenizer_config.json + chat_template.jinja from the
    GGUF's embedded tokenizer. BPE (merges present) only: a mergeless
    sentencepiece vocab cannot become a correct BPE file, and that raises.
    Returns the written file names."""
    # The vocab KV dominates the prefix; grow until the tensor infos parse.
    buf, inv = b"", None
    with open(gguf_path, "rb") as f:
        for _ in range(4):  # 8/16/32/64 MB ceiling
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            buf += chunk
            try:
                inv = parse_gguf_inventory(buf)
                break
            except _GgufTruncated:
                continue
    if inv is None:
        raise RuntimeError("could not parse the GGUF metadata (truncated header?)")
    kv, _ = inv

    model = kv.get("tokenizer.ggml.model", "llama")
    tokens = kv.get("tokenizer.ggml.tokens")
    merges = kv.get("tokenizer.ggml.merges") or []
    if not tokens:
        raise RuntimeError("the GGUF carries no tokenizer.ggml.tokens; fetch the tokenizer from the base repo")
    if not merges:
        raise RuntimeError(
            f"the embedded tokenizer ({model}) has no merges (sentencepiece-style); "
            "a BPE tokenizer.json cannot be derived from it -- fetch the tokenizer "
            "from the base repo named in the GGUF repo's README")
    types = kv.get("tokenizer.ggml.token_type") or [1] * len(tokens)
    table = _bytes_to_unicode()

    def raw(i):
        return tokens[i].encode("utf-8", "surrogateescape")

    vocab, added, unk_content = {}, [], None
    for i, tok in enumerate(tokens):
        typ = types[i] if i < len(types) else 1
        if typ == 5:  # unused slot
            continue
        content = _map_token(raw(i), table)
        if typ in (2, 3, 4):  # unknown / control / user-defined specials
            added.append({"id": i, "content": content, "single_word": False,
                          "lstrip": False, "rstrip": False, "normalized": False,
                          "special": True})
            if typ == 2:
                unk_content = content
            vocab[content] = i
        else:  # normal (1) and byte (6) tokens live in the BPE vocab
            vocab[content] = i
    hf_merges = []
    for m in merges:
        raw_m = m.encode("utf-8", "surrogateescape")
        # Pairs join on ONE space; a side may itself hold 0x20, so split right.
        left, _, right = raw_m.decode("utf-8", "surrogateescape").rpartition(" ")
        if not _:
            continue
        a = _map_token(left.encode("utf-8", "surrogateescape"), table)
        b = _map_token(right.encode("utf-8", "surrogateescape"), table)
        hf_merges.append(f"{a} {b}")

    def tok_content(tid, default=None):
        if tid is None or not (0 <= tid < len(tokens)):
            return default
        return _map_token(raw(tid), table)

    bos_id = kv.get("tokenizer.ggml.bos_token_id")
    eos_id = kv.get("tokenizer.ggml.eos_token_id")
    unk_id = kv.get("tokenizer.ggml.unknown_token_id")
    if eos_id is None:
        raise RuntimeError("the GGUF names no eos token; fetch the tokenizer from the base repo")
    eos_content = tok_content(eos_id)
    bos_content = tok_content(bos_id)
    if unk_content is None:
        unk_content = tok_content(unk_id, eos_content)
    template = kv.get("tokenizer.chat_template")

    byte_level = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True}
    byte_level_no_trim = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False}
    tokenizer_json = {
        "version": "1.0",
        "truncation": None, "padding": None,
        "added_tokens": sorted(added, key=lambda t: t["id"]),
        "normalizer": byte_level,
        "pre_tokenizer": byte_level,
        "post_processor": byte_level_no_trim,
        "decoder": byte_level_no_trim,
        "model": {"type": "BPE", "dropout": None, "unk_token": unk_content,
                  "continuing_subword_prefix": "", "end_of_word_suffix": "",
                  "fuse_unk": True, "byte_fallback": False,
                  "vocab": vocab, "merges": hf_merges},
    }
    config_json = {
        "bos_token": bos_content,
        "eos_token": eos_content,
        "unk_token": unk_content,
        "bos_token_id": bos_id,
        "eos_token_id": [eos_id],
        "add_bos_token": bool(kv.get("tokenizer.ggml.add_bos_token", True)),
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    if template:
        config_json["chat_template"] = template

    written = []
    for name, doc in (("tokenizer.json", tokenizer_json), ("tokenizer_config.json", config_json)):
        dest = target / name
        if dest.is_file() and not force:
            continue
        dest.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(name)
    if template:
        dest = target / "chat_template.jinja"
        if not dest.is_file() or force:
            dest.write_text(template if template.endswith("\n") else template + "\n", encoding="utf-8")
            written.append("chat_template.jinja")
    return written


def main():
    ap = argparse.ArgumentParser(
        description="Install a pre-converted OFLM (Q4NX) model and register it with OpenFlowLM.",
    )
    ap.add_argument("repo", help="Hugging Face repo id (Org/Name), ModelScope id (with --modelscope), URL, or local directory")
    ap.add_argument("--tag", help="Registry tag (default: derived from the repo name, e.g. qwen3.5-claude:9b)")
    ap.add_argument("--name", help="Install under a custom tag (e.g. my-finetune:3b). Overrides --tag.")
    ap.add_argument("--family", help="details.family for engine dispatch (default: from matching official entry)")
    ap.add_argument("--config", help="your settings.json to update (default: $OFLM_SETTINGS_PATH, or next to the models directory)")
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
    # `repo:QUANT` names the quant to take from a GGUF repo (mradermacher's
    # catalogues carry one file per quant). Without it the preference order
    # decides -- Q4_1 first, the one the pool takes byte-exactly.
    repo_arg, gguf_want = split_quant_request(repo_arg)
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
    tree = {}
    gguf_source = None
    cache = None
    if local_dir:
        gguf_names = [e.name for e in local_dir.iterdir() if e.is_file() and e.suffix == ".gguf"]
    else:
        try:
            tree = ms_file_tree(repo)[1] if modelscope else hf_file_tree(repo)
            gguf_names = [n for n in root_file_names(tree) if n.endswith(".gguf")]
        except Exception as ex:
            log(f"[WARN] Could not list the repo tree ({ex}); assuming the NPU2 path.")
    gguf_name, gguf_refused = choose_gguf_file(gguf_names, gguf_want)
    if gguf_want and gguf_name is None:
        if not gguf_names:
            all_files = sorted(e.name for e in local_dir.iterdir() if e.is_file()) if local_dir else sorted(root_file_names(tree))
            raise SystemExit(
                f"{repo_arg}: ':QUANT' is for GGUF repositories, and this repository has no .gguf files "
                f"(files present: {', '.join(all_files) or 'none'})")
        compat = sorted({gguf_quant_suffix(n).upper() for n in gguf_names
                         if gguf_quant_of(n) in GGUF_QUANT_PREFERENCE and not re.search(r"-\d+-of-\d+\.gguf$", n)})
        files = ", ".join(sorted(gguf_names))
        raise SystemExit(
            f"{repo_arg}: no compatible '{gguf_want}' GGUF in that repository.\n"
            f"  supported quants present : {', '.join(compat) or 'none'} "
            f"(open kernels accept {'/'.join(GGUF_QUANT_PREFERENCE)})\n"
            f"  .gguf files in repo      : {files}")
    cache = None
    if local_dir:
        have_q4nx = (local_dir / "model.q4nx").is_file()
    else:
        cache = ms_cache_snapshot(repo) if modelscope else hf_cache_snapshot(repo)
        have_q4nx = repo_has_q4nx(tree, cache)
    gguf_only = bool(gguf_names) and not have_q4nx
    if gguf_only and gguf_name is None:
        files = ", ".join(sorted(gguf_names))
        raise SystemExit(
            f"{repo_arg}: repository carries only .gguf files, and none uses a quant supported "
            f"by the open kernels (accept {'/'.join(GGUF_QUANT_PREFERENCE)}).\n"
            f"  .gguf files in repo: {files}")
    gguf_mode = gguf_name is not None and not have_q4nx
    if gguf_mode:
        # The tensors must be what the name claims, in types the NPU packers
        # ingest. Remote metadata is best-effort (Range); a local file is on
        # disk, so a failed parse there refuses the file outright.
        if local_dir:
            gguf_source = str(local_dir / gguf_name)
        elif cache and (cache / gguf_name).is_file():
            # The user already has this GGUF on disk in their HF cache: verify
            # the local file (no network round trip) and let the install copy
            # it from there below.
            gguf_source = str(cache / gguf_name)
        elif modelscope:
            domain = ms_file_tree(repo)[0]
            gguf_source = f"https://{domain}/models/{repo}/resolve/master/{gguf_name}"
        else:
            gguf_source = f"https://huggingface.co/{repo}/resolve/main/{gguf_name}"
        ok, note = verify_gguf_choice(gguf_source, gguf_name, gguf_quant_of(gguf_name))
        if ok:
            log(f"[INFO] {gguf_name}: {note}")
        else:
            raise SystemExit(f"{repo_arg}: refusing {gguf_name}: {note}")

    system_list = Path(args.system_list) if args.system_list else find_system_model_list()
    system_registry = load_json(system_list)
    user_list = user_registry_path(args.config)
    models_root = models_root_dir(args.models_root)
    target = models_root / dir_name

    # Detection caches: the failure paths below may each need the base-model
    # chain and a peeked config.json; fetch each at most once per install.
    base_chain = None
    peeked = None
    peeked_done = False

    def get_base_chain():
        nonlocal base_chain
        if base_chain is None:
            base_chain = readme_base_chain(repo if not local_dir else repo_arg,
                                            modelscope)
        return base_chain

    def get_peeked_config():
        nonlocal peeked, peeked_done
        if not peeked_done:
            peeked_done = True
            peeked = peek_config_dict(None if local_dir else repo, modelscope,
                                      bases=get_base_chain(),
                                      local_dirs=[d for d in (local_dir, cache) if d])
        return peeked

    # ---- the family: what the model IS, before what it is CALLED. A fine-tune's
    # name says nothing ("Ornith-1.5-9B"), but the kernels it needs say who it
    # is: the installed set whose geometry matches is the base model's, and the
    # registry ties that base to a family. Then the base chain (the README's
    # base model), then the content (the GGUF header's architecture, the
    # config's model_type), and last the name.
    content_families = []
    # The roots the open kernel sets live in: the user's, the system's, and the
    # tree's own (src/xclbins -- a freshly exported set, which may be newer
    # than the installed copy).
    kernel_roots = [user_xclbin_dir(args.xclbin_dir), find_system_xclbin_root(), tree_xclbins()]
    spec = None
    geom_set = None                     # the set whose GEOMETRY this model needs
    if args.family is None:
        spec = spec_from_config_dict(get_peeked_config())
        fam, set_name = family_from_kernels(system_registry, spec, kernel_roots)
        if fam:
            log(f"[INFO] family '{fam}' from the kernel set '{set_name}' (the same geometry)")
            content_families.append(fam)
            geom_set = set_name
        fam, base_name = family_from_base_chain(system_registry, get_base_chain())
        if fam and fam not in content_families:
            log(f"[INFO] family '{fam}' from the base model '{base_name}'")
            content_families.append(fam)
    try:
        family = derive_family(system_registry, dir_name, args.family,
                               content_families=content_families)
    except SystemExit:
        # No name marker, and no official entry can supply the family yet
        # (that matching happens below). Climb the content instead, cheapest
        # first: the GGUF header names its architecture; otherwise a
        # config.json -- the repo's own, else its base-model chain's -- names
        # its model_type. llama.cpp resolves the same way, from the file, not
        # the repo slug.
        family = None
        if args.family is None:
            if gguf_source is not None:
                family = family_from_gguf_arch(gguf_source)
                if family:
                    log(f"[INFO] family '{family}' from the GGUF architecture")
            if family is None:
                family = family_from_config_dict(get_peeked_config())
                if family:
                    log(f"[INFO] family '{family}' from config.json model_type")
        if family is None:
            raise
    npue_embed = family in NPU_EMBED_FAMILIES
    if npue_embed:
        # NPU-embedding (NpuEmbeddings) model: the authoritative registry entry
        # carries `npue_design_family` + the safetensors file list, and the engine
        # routes by the EXACT registered tag -- so default the tag to that entry
        # rather than deriving a numeric size these repos (bge-base:en-v1.5) lack.
        official = match_npue_official(system_registry, family, dir_name)
        if not official:
            raise SystemExit(
                f"No official entry found for npue-embedding family '{family}' "
                f"(repo '{dir_name}'). The npue_design_family metadata is required; "
                "pass --tag if the model is a known variant.")
        base_entry = official[3]
        official_note = None
        src_tag = f"{official[1]}:{official[2]}"
        xclbin_source = base_entry.get("name")
        tag = f"{official[1]}:{official[2]}" if not args.tag else args.tag
        bucket, size_token = tag.split(":", 1)
        size_value = base_entry.get("size") or size_from_tag(tag)
        gguf_mode = False
        gguf_name = None
    else:
        official = match_official_entry(system_registry, dir_name)
        base_entry = official[3] if official else None
        try:
            tag = derive_tag(dir_name, args.tag)
        except SystemExit:
            # Chaotic slugs carry no size marker: guess from the GGUF's bytes
            # and quant, else from the peeked config's geometry. Kernel
            # matching keys off the tag size, so a logged guess beats a
            # refusal; --tag still overrides.
            if args.tag is not None:
                raise
            size_guess = None
            if gguf_mode:
                size_guess = gguf_size_token(
                    gguf_byte_size(repo, modelscope, local_dir, cache,
                                   gguf_name, tree),
                    gguf_quant_of(gguf_name))
            if size_guess is None:
                size_guess = size_token_from_bytes(
                    estimate_size_dict(get_peeked_config()))
            if size_guess is None:
                raise
            log(f"[INFO] tag size '{size_guess}' guessed for '{dir_name}'; "
                "pass --tag to override")
            tag = derive_tag(dir_name, None, size_guess)
        bucket, size_token = tag.split(":", 1)
        size_value = (base_entry or {}).get("size") or size_from_tag(tag)
        official, official_note = resolve_official(system_registry, dir_name, family, size_value)
        base_entry = official[3] if official else None
        src_tag = f"{official[1]}:{official[2]}" if official else None
        # A set whose KERNELS match this model's geometry is the right answer, and
        # resolve_official's last resort -- the nearest official size -- is not: a
        # Qwen3-14B model was being handed Qwen3-8B's kernels because the registry
        # has no 14B entry, which is a wrong link, not a small one.
        xclbin_source = args.xclbin_from or geom_set or (base_entry or {}).get("name")

    # ---- the GGUF-direct gate: the engine must read model.gguf AND the linked
    # xclbins must actually carry the GGUF-direct kernel builds (the nested
    # "gguf" manifest section, produced by the *_f32 sets in utilities/
    # build-all.sh). A GGUF repository never carries model.q4nx, so a missing
    # kernel set stops here with a direct error rather than falling back to the
    # q4nx file list.
    gguf_kernel_dir = None
    if gguf_mode:
        if family in EMBEDDING_FAMILIES:
            # Embedding GGUF uses the existing bf16 matmul kernels (the engine
            # reads model.gguf and dequantizes); it needs no f32-scale GGUF-direct
            # open_kernels manifest. Just confirm the family is recognized.
            xclbin_source = "Embedding-Gemma-300M-OpenNPU2"
        else:
            # The set that carries the section: the one the install links, or a
            # fresher one with the same name (a rebuild adds the f32-scale
            # twins; the installed copy may predate it).
            if args.open_kernels:
                kdir = Path(args.open_kernels) if manifest_supports_gguf(
                    Path(args.open_kernels) / "manifest.json") else None
            else:
                kdir = gguf_kernel_set(xclbin_source, kernel_roots)
            if kdir is None:
                bases = get_base_chain()
                base_hint = f" --repo {bases[0]}" if bases else ""
                raise SystemExit(
                    f"{repo_arg}: no GGUF-direct open kernel set found for "
                    f"{xclbin_source or dir_name} (family '{family}').\n"
                    f"Build the kernel set for this model size with:\n"
                    f"  python utilities/build-new-spec.py{base_hint}")
            if not args.open_kernels:
                _require_same_geometry(kdir, spec, spec_from_config_dict(get_peeked_config()),
                                       dir_name)
            gguf_kernel_dir = kdir
            if str(kdir) != str(Path(args.open_kernels) if args.open_kernels else ""):
                log(f"[INFO] GGUF-direct kernels: {kdir}")

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
            print(f"quant family   : {gguf_quant_of(gguf_name)} (tensor types verified against the GGUF header)")
            for n, why in sorted(gguf_refused.items()):
                print(f"  skipped      : {n} ({why})")
            bases = get_base_chain()
            print(f"base model(s)  : {', '.join(bases) or '(none in README)'}")
            print("tokenizer      : base repo files, else the GGUF's embedded tokenizer")
        elif npue_embed:
            print(f"weights        : safetensors (NPU-embedding: {base_entry.get('npue_design_family')})")
        if args.open_kernels:
            print(f"open kernels   : {args.open_kernels} (--open-kernels)")
        elif local_dir:
            sh, note = model_spec_hash(local_dir)
            found, _ = find_open_kernels(sh, kernel_roots, dir_name) if sh else (None, None)
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
        files = copy_from_dir(local_dir, target, force=args.force,
                              gguf_name=gguf_name if gguf_mode else None,
                              want=NPU_EMBED_FILES if npue_embed else None)
    else:
        snapshot = ms_cache_snapshot(repo) if modelscope else hf_cache_snapshot(repo)
        # A cache that holds the repo but not the file we want is no use: the
        # GGUF-direct path asks for ONE file by name, and a snapshot with only
        # the config (or with a different quant) is not it. Fall through to the
        # download then, rather than copying a cache that cannot serve.
        if snapshot and gguf_mode and not (Path(snapshot) / gguf_name).is_file():
            if not args.quiet:
                log(f"[INFO] The local cache has no {gguf_name}; downloading it.")
            snapshot = None
        if snapshot:
            if not args.quiet:
                log(f"[INFO] Found local {'ModelScope' if modelscope else 'HF'} cache: {snapshot}")
            target.mkdir(parents=True, exist_ok=True)
            files = copy_from_dir(snapshot, target, force=args.force,
                                  gguf_name=gguf_name if gguf_mode else None,
                                  want=NPU_EMBED_FILES if npue_embed else None)
        else:
            if not args.quiet:
                log(f"[INFO] Downloading model files from {'ModelScope' if modelscope else 'Hugging Face'}: {repo}")
            target.mkdir(parents=True, exist_ok=True)
            files = fetch_assets(repo, target, modelscope, verify=not args.no_verify, force=args.force,
                                 quiet=args.quiet, gguf_name=gguf_name if gguf_mode else None,
                                 want=NPU_EMBED_FILES if npue_embed else None)

    if npue_embed:
        # Core safetensors checkpoint + tokenizer; vocab.txt / 1_Pooling are absent
        # for some families (e.g. gte has no vocab.txt) and are tolerated.
        required = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
    else:
        required = ["model.gguf", "tokenizer.json", "tokenizer_config.json"] if gguf_mode else REQUIRED_FILES
    missing = [f for f in required if not (target / f).is_file()]
    if gguf_mode:
        # GGUF-quant repos (mradermacher etc.) often ship only the GGUF: take the
        # tokenizer/config files from the original model the README names.
        # config.json too: the curated config beats the GGUF-derived one (and
        # Granite's folded multipliers exist only there). The chain climbs past
        # the quant repo: a quant of a finetune of a base tries each level in
        # order until every file resolves.
        aux_missing = [f for f in missing if f != "model.gguf"]
        if not (target / "config.json").is_file():
            aux_missing.append("config.json")
        if not (target / "chat_template.jinja").is_file():
            aux_missing.append("chat_template.jinja")
        if aux_missing:
            bases = get_base_chain()
            gated_warned = set()
            if bases:
                log(f"[INFO] base model(s) from README.md: {', '.join(bases)}")
                for f in aux_missing:
                    got = None
                    for b in bases:
                        try:
                            got = fetch_from_repo(b, f, target / f, modelscope=modelscope)
                        except urllib.error.HTTPError as ex:
                            if ex.code in (401, 403) and b not in gated_warned:
                                gated_warned.add(b)
                                log(f"[WARN] {b} is gated (HTTP {ex.code}); set HF_TOKEN "
                                    "(huggingface-cli login) to fetch from it, or rely on "
                                    "the GGUF-embedded tokenizer below")
                            elif b not in gated_warned:
                                log(f"[WARN] fetching {f} from {b} failed: {ex}")
                            got = None
                        except Exception as ex:
                            log(f"[WARN] fetching {f} from {b} failed: {ex}")
                            got = None
                        if got:
                            log(f"[INFO]   {f} <- {b}")
                            break
            else:
                log("[WARN] no base_model in the GGUF repo README; tokenizer/config must "
                    "come from the GGUF itself")
            # Last resort: the GGUF embeds its own tokenizer (vocab, merges,
            # specials, chat template). Curated base-repo files win when they
            # exist; this covers gated or vanished bases with zero auth.
            if gguf_mode and (target / "tokenizer.json").is_file():
                # tokenizer.json already exists (from base repo); skip GGUF extraction
                log("[INFO] tokenizer.json already present from base repo; skipping GGUF extraction")
            elif (target / "model.gguf").is_file():
                try:
                    wrote = extract_tokenizer_from_gguf(target / "model.gguf", target, force=args.force)
                except Exception as ex:
                    log(f"[WARN] GGUF-embedded tokenizer unusable: {ex}")
                else:
                    log(f"[INFO] tokenizer files from the GGUF itself: {', '.join(wrote)}")
            missing = [f for f in required if not (target / f).is_file()]
    normalize_tokenizer_config(target, repo, get_base_chain(), modelscope)
    # The chain ends at the OFFICIAL converted repo: a quant of a base whose own
    # tokenizer_config has no chat_template (Nanbeige 4.3-Base) still finds one in
    # the family's converted container (Nanbeige4.1-3B-NPU2), which is the repo
    # the kernels came from.
    _chain = list(get_base_chain())
    _off_repo = official_repo_id(base_entry, system_registry)
    if _off_repo and _off_repo not in _chain:
        _chain.append(_off_repo)
    fill_chat_template(target, repo, _chain, modelscope)
    _kset = gguf_kernel_dir or kernel_manifest_dir(xclbin_source, kernel_roots)
    if _kset is not None and (Path(target) / "config.json").is_file():
        reconcile_config_json(target, _kset)
    if missing:
        if gguf_mode and missing == ["config.json"]:
            missing = []
        if missing:
            present = sorted(e.name for e in target.iterdir() if e.is_file())
            raise SystemExit(
                f"Model is missing required files: {missing} (files present: {', '.join(present) or 'none'})")
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
    if npue_embed:
        # build_entry copies npue_design_family / details / label from the official
        # entry, but blanks `url`; restore it so the .npue packer attributes the
        # weights to their source repository (a licensing statement, not optional).
        entry["url"] = base_entry.get("url") or f"https://huggingface.co/{repo}"

    settings_path = user_settings_path(args.config)
    # Pass the tag derived in main() (repo name + --tag); register() must not
    # reconstruct it from the entry, whose `size` is a byte count.
    tag = register(
        entry,
        settings_path,
        system_registry,
        force=args.force,
        name_override=args.name,
        tag=tag,
    )
    log(f"[INFO] Registered tag '{tag}' in {settings_path}")

    system_root = find_system_xclbin_root()
    if not args.no_xclbin:
        if npue_embed:
            design = base_entry.get("npue_design_family")
            if design:
                link_npue_design(SYSTEM_XCLBIN_PREFIXES, user_xclbin_dir(args.xclbin_dir),
                                 design, force=args.force, quiet=args.quiet)
        elif system_root is None:
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
    # NPU-embedding uses the npue_design_family design set (already installed),
    # not the chat-model open-kernel link, so it is skipped here.
    if (args.open_kernels or gguf_kernel_dir or not args.no_xclbin) and not npue_embed:
        setup_open_kernels(
            target,
            dir_name,
            kernel_roots,
            override=args.open_kernels,
            prefer=str(gguf_kernel_dir) if gguf_kernel_dir else None,
            force=args.force,
            quiet=args.quiet,
        )

    write_vision_sidecar(target, dir_name, quiet=args.quiet)

    print()
    print(f"Done: {dir_name} installed to {target}")
    print(f"Run:  oflm run {tag}   (or: oflm serve {tag})")
    print()
    print("OpenFlowLM discovers this model's registry entry and kernels")
    print("automatically; no OFLM_CONFIG_PATH/OFLM_XCLBIN_PATH exports needed.")
