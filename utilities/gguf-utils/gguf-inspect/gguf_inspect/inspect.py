"""Metadata inspection for GGUF model sets.

Produces a structured report of everything an installer must know before
deciding how to load or pack a model: architecture, trunk/MTP structure,
projector components, tensor-carried RoPE, tokenizer/template, a candidate
install tag, and the exact unsupported requirements that block execution.

Nothing here decodes tensor payloads and the NPU is never opened.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from .reader import GgufFile, open_gguf

_MTP_NAMES = ("mtp", "nextn", "block_32", "pred")


@dataclass
class RopeFactors:
    tensor_names: list = field(default_factory=list)
    carried: bool = False


@dataclass
class Projector:
    architecture: str = ""
    base_model: str = ""
    is_clip: bool = False


@dataclass
class Tokenizer:
    model: str = ""
    pre: str = ""
    bos: Optional[int] = None
    eos: Optional[int] = None
    pad: Optional[int] = None
    has_chat_template: bool = False
    template_preview: str = ""


@dataclass
class BlockInfo:
    architecture: str = ""
    block_count: int = 0
    nextn_predict_layers: int = 0
    trunk_layers: int = 0
    has_mtp: bool = False
    mtp_note: str = ""


@dataclass
class ExecutionOptions:
    file_type: Optional[int] = None
    quantization_version: Optional[int] = None
    dtype_counts: dict = field(default_factory=dict)
    total_tensor_bytes: int = 0
    unknown_dtypes: list = field(default_factory=list)


@dataclass
class Unsupported:
    requirements: list = field(default_factory=list)
    notes: list = field(default_factory=list)


@dataclass
class GGUFReport:
    path: str = ""
    version: int = 0
    endian: str = ""
    source: str = ""
    snapshot: str = ""
    architecture: str = ""
    general: dict = field(default_factory=dict)
    blocks: BlockInfo = field(default_factory=BlockInfo)
    projector: Projector = field(default_factory=Projector)
    rope: RopeFactors = field(default_factory=RopeFactors)
    tokenizer: Tokenizer = field(default_factory=Tokenizer)
    candidate_tag: str = ""
    tag_basis: str = ""
    execution: ExecutionOptions = field(default_factory=ExecutionOptions)
    unsupported: Unsupported = field(default_factory=Unsupported)

    def to_dict(self) -> dict:
        return asdict(self)


def _kv(data: dict, prefix: str) -> dict:
    out: dict = {}
    for k, v in data.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


def inspect(path: str, source: str = "", snapshot: str = "") -> GGUFReport:
    gf: GgufFile = open_gguf(path)
    rep = GGUFReport(path=gf.path, version=gf.version or 0, endian=gf.endian)
    rep.source = source or _hf_snapshot(path)
    rep.snapshot = snapshot or _hf_snapshot(path)

    arch = gf.get_field("general.architecture")
    rep.architecture = str(arch) if arch else ""
    for key in ("name", "basename", "size_label", "file_type", "finetune",
                "quantized_by", "license", "repo_url", "source_url"):
        val = gf.get_field("general." + key)
        if val is not None:
            rep.general[key] = val
    rep.general["file_type_raw"] = gf.get_field("general.file_type")

    _inspect_blocks(gf, rep)
    _inspect_projector(gf, rep)
    _inspect_rope(gf, rep)
    _inspect_tokenizer(gf, rep)
    _inspect_execution(gf, rep)
    rep.candidate_tag, rep.tag_basis = _derive_candidate_tag(rep)
    _inspect_unsupported(rep)
    return rep


def _inspect_blocks(gf: GgufFile, rep: GGUFReport) -> None:
    arch = rep.architecture
    for key in ("block_count", "nextn_predict_layers", "context_length",
                "embedding_length", "feed_forward_length"):
        val = gf.get_field(arch + "." + key) if arch else gf.get_field(key)
        if val is not None and val != "":
            try:
                rep.general[arch + "." + key] = int(val)
            except (TypeError, ValueError):
                pass
    bc = rep.general.get(arch + ".block_count")
    info = BlockInfo(architecture=arch)
    if bc is not None:
        info.block_count = bc
        info.nextn_predict_layers = rep.general.get(arch + ".nextn_predict_layers", 0)
        info.trunk_layers = info.block_count - info.nextn_predict_layers
        info.has_mtp = info.nextn_predict_layers > 0
        if info.has_mtp:
            info.mtp_note = (
                f"block_count {info.block_count} includes {info.nextn_predict_layers} "
                f"NextN layer(s); trunk is {info.trunk_layers} layer(s)"
            )
    rep.blocks = info


def _inspect_projector(gf: GgufFile, rep: GGUFReport) -> None:
    proj = Projector()
    has_clip = any(k.startswith("clip.") for k in gf.field_names) or \
        rep.architecture == "clip"
    proj.is_clip = has_clip
    if has_clip:
        proj.architecture = str(gf.get_field("general.architecture") or "")
        bm = gf.get_field("general.base_model.0.name")
        proj.base_model = str(bm) if bm else ""
    rep.projector = proj


def _inspect_rope(gf: GgufFile, rep: GGUFReport) -> None:
    names: list = []
    for key in gf.field_names:
        low = key.lower()
        if low.startswith("rope") and low.endswith(".weight"):
            names.append(key)
    for t in gf.tensors:
        low = t.name.lower()
        if "rope" in low and low.endswith(".weight"):
            names.append(t.name)
    uniq = sorted(set(names))
    rep.rope = RopeFactors(tensor_names=uniq, carried=bool(uniq))


def _inspect_tokenizer(gf: GgufFile, rep: GGUFReport) -> None:
    tok = Tokenizer()
    for key in gf.field_names:
        if not key.startswith("tokenizer."):
            continue
        val = gf.get_field(key)
        if val is None or val == "":
            continue
        suffix = key[len("tokenizer."):].lower()
        if suffix == "ggml.model":
            tok.model = str(val)
        elif suffix == "ggml.pre":
            tok.pre = str(val)
        elif suffix == "ggml.bos_token_id":
            tok.bos = _to_int(val)
        elif suffix == "ggml.eos_token_id":
            tok.eos = _to_int(val)
        elif suffix == "ggml.padding_token_id":
            tok.pad = _to_int(val)
        elif suffix == "chat_template":
            tok.has_chat_template = True
            tok.template_preview = str(val)[:160]
    rep.tokenizer = tok


def _inspect_execution(gf: GgufFile, rep: GGUFReport) -> None:
    counts: Counter = Counter(t.dtype for t in gf.tensors)
    total = sum(t.n_bytes for t in gf.tensors)
    ft = gf.get_field("general.file_type")
    qv = gf.get_field("general.quantization_version")
    unknown = []
    for dtype, _ in counts.items():
        if dtype not in ("Q4_1", "Q8_0", "Q6_K", "Q4_K", "Q5_K", "Q3_K",
                         "Q2_K", "F16", "F32", "BF16", "I8"):
            unknown.append(dtype)
    rep.execution = ExecutionOptions(
        file_type=_to_int(ft),
        quantization_version=_to_int(qv),
        dtype_counts=dict(counts),
        total_tensor_bytes=total,
        unknown_dtypes=sorted(set(unknown)),
    )


def _derive_candidate_tag(rep: GGUFReport) -> tuple[str, str]:
    arch = rep.architecture
    size_label = rep.general.get("size_label") or rep.general.get(arch + ".size_label", "")
    base = (rep.general.get("basename") or arch or "model")
    size = ""
    if size_label:
        size = re.sub(r"[^0-9.]", "", str(size_label))
    if not size:
        return "", "no size_label"
    name = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
    tag = f"{name}:{size}B"
    return tag, f"general.basename={base}, size_label={size_label}"


def _inspect_unsupported(rep: GGUFReport) -> None:
    uns = Unsupported()
    if rep.blocks.has_mtp:
        uns.requirements.append(
            "speculative/MTP execution for the NextN layer(s); "
            "must be supported before the model runs"
        )
    if rep.projector.is_clip:
        uns.requirements.append(
            "vision encoder + projector execution for the clip component"
        )
    if not rep.rope.carried and rep.blocks.architecture in ("llama", "phi3"):
        uns.notes.append(
            "RoPE factors may be derived rather than carried; confirm with the exporter"
        )
    if rep.execution.unknown_dtypes:
        uns.requirements.append(
            f"decode/support for tensor type(s) not in the base set: "
            f"{', '.join(rep.execution.unknown_dtypes)}"
        )
    if not rep.tokenizer.has_chat_template:
        uns.notes.append("no tokenizer.chat_template present; a template is required at serve")
    rep.unsupported = uns


def _to_int(val: Any) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _hf_snapshot(path: str) -> str:
    """Return the HF snapshot hash when `path` is inside a HF cache, else ''.

    HF caches layout: .../models--<repo>__/snapshots/<hash>/<file>. Only the
    snapshot hash is derivable from the path; the repo id must be passed via
    --source.
    """
    parts = Path(path).parts
    if "snapshots" in parts:
        i = parts.index("snapshots")
        if i + 1 < len(parts):
            cand = parts[i + 1]
            if len(cand) == 40 and all(c in "0123456789abcdef" for c in cand):
                return cand
    return ""

