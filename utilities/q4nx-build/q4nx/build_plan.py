"""Derive build parameters (-s skeleton, -o name, -t weights type) from HF metadata.

Most converted builds need a "skeleton" source: an existing FLM NPU2 port of the
true base model that ships exactly the config/tokenizer/vision assets a build
requires (e.g. Atomic-Germ/Qwen3.5-4B-NPU2). Instead of asking the user for
-s/-o/-t, we follow the model card's ``base_model`` frontmatter chain upward and
stop at the first ancestor whose ``{org}/{basename}-NPU2`` mirror exists under a
preferred org. The chain also yields the display name and parameter-size token
for the output folder, and the pipeline tag tells us whether the finetune is a
VLM (convert language + vision weights).

Every network touch is optional: offline or private-repo failures degrade to a
``[WARN]`` and the caller keeps whatever the user passed explicitly.
"""
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .model_assets import _repo_id_from_url

# Orgs searched (in order) for the {basename}-NPU2 skeleton repo.
SKELETON_ORGS = ("Atomic-Germ", "FastFlowLM")
SKELETON_SUFFIX = "-NPU2"
MAX_CHAIN_DEPTH = 8

# Pipeline tags that imply vision-capable weights.
VLM_PIPELINE_TAGS = frozenset({
    "image-text-to-text",
    "image-to-text",
    "video-text-to-text",
    "any-to-any",
    "document-question-answering",
})
# Card tags with the same meaning (some cards skip the pipeline tag).
VLM_TAG_KEYWORDS = ("vision-language", "vlm", "image-text-to-text")

# Size token inside a skeleton name: 4B / 0.8B / 35B-A3B / E4B. Searched as a
# dash-delimited unit so "Claude" never matches, and kept in original case so
# lowercase sources stay lowercase (medgemma-1.5-4b-it -> 4b).
_SIZE_TOKEN_RE = re.compile(
    r"(?:^|-)(\d+(?:\.\d+)?[bB](?:-[Aa]\d+[bB])?|[Ee]\d+[bB])(?:$|-)"
)


@dataclass
class BuildPlan:
    """Everything derivable from an input repo's card metadata."""

    repo_id: str
    chain: List[str] = field(default_factory=list)
    skeleton: Optional[str] = None
    display_name: Optional[str] = None
    size_token: Optional[str] = None
    output_name: Optional[str] = None
    weights_type: str = "language"
    weights_reason: str = ""
    pipeline_tag: Optional[str] = None


def _model_info(repo_id: str):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[WARN] huggingface_hub not installed; cannot read HF metadata")
        return None
    try:
        return HfApi().model_info(repo_id)
    except Exception:
        return None


def _card_value(card, key):
    """Read a card field across huggingface_hub versions (dict-like or attrs)."""
    if card is None:
        return None
    try:
        return card[key]
    except Exception:
        return getattr(card, key, None)


def fetch_card(repo_id: str) -> dict:
    """The subset of a repo's card metadata used for derivation; {} if unavailable."""
    info = _model_info(repo_id)
    if info is None:
        return {}
    card = getattr(info, "cardData", None) or getattr(info, "card_data", None)
    return {
        "base_model": _card_value(card, "base_model"),
        "model_name": _card_value(card, "model_name"),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "tags": list(getattr(info, "tags", None) or []),
    }


def normalize_base_models(value) -> List[str]:
    """base_model frontmatter to repo ids: string or list form, URLs flattened."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            continue
        item = item.strip()
        item = _repo_id_from_url(item) or item
        if item and item not in result:
            result.append(item)
    return result


def walk_base_chain(
    repo_id: str,
    max_depth: int = MAX_CHAIN_DEPTH,
    cache: Optional[Dict[str, dict]] = None,
    fetch: Callable[[str], dict] = fetch_card,
) -> List[str]:
    """Ancestor chain [repo, parent, grandparent, ...] from base_model cards.

    Stops at depth, a cycle, or a repo without base_model. Note this does NOT
    stop at the "true" base: e.g. Qwen/Qwen3.5-4B itself declares
    Qwen/Qwen3.5-4B-Base. Callers pick the interesting node (find_skeleton).
    """
    if cache is None:
        cache = {}
    chain = [repo_id]
    seen = {repo_id}
    current = repo_id
    for _ in range(max_depth - 1):
        card = cache.get(current)
        if card is None:
            card = fetch(current)
            cache[current] = card
        parents = normalize_base_models(card.get("base_model"))
        if not parents:
            break
        nxt = parents[0]
        if nxt in seen:
            break
        chain.append(nxt)
        seen.add(nxt)
        current = nxt
    return chain


def _skeleton_exists(candidate: str) -> bool:
    return _model_info(candidate) is not None


def find_skeleton(
    chain: List[str],
    orgs: List[str] = SKELETON_ORGS,
    probe: Callable[[str], bool] = _skeleton_exists,
) -> Optional[str]:
    """First ancestor with an {org}/{basename}-NPU2 mirror under a preferred org.

    Nearest ancestor wins, so a finetune of Qwen/Qwen3.5-4B resolves to
    Atomic-Germ/Qwen3.5-4B-NPU2 rather than walking all the way to -Base.
    """
    for node in chain:
        basename = os.path.basename(node.rstrip("/"))
        if basename.endswith(SKELETON_SUFFIX):
            continue
        for org in orgs:
            candidate = f"{org}/{basename}{SKELETON_SUFFIX}"
            if candidate == node:
                continue
            if probe(candidate):
                return candidate
    return None


def parse_size_token(name: str) -> Optional[str]:
    """Parameter-size token in a model/skeleton name (4B, 0.8B, 35B-A3B, E4B)."""
    stem = re.sub(rf"{SKELETON_SUFFIX}$", "", name, flags=re.IGNORECASE)
    match = _SIZE_TOKEN_RE.search(stem)
    return match.group(1) if match else None


def derive_display_name(card: dict, repo_id: str) -> str:
    """model_name frontmatter field, else the repo basename minus -GGUF."""
    name = str(card.get("model_name") or "").strip()
    if not name:
        name = os.path.basename(repo_id.rstrip("/"))
        name = re.sub(r"(?:-[Gg][Gg][Uu][Ff])+$", "", name)
    name = re.sub(r"\s+", "-", name).strip("-")
    return name or repo_id


def infer_weights_type(card: dict) -> str:
    """'vision' when the card marks the model VLM, else 'language'."""
    pipeline = str(card.get("pipeline_tag") or "").lower()
    if pipeline in VLM_PIPELINE_TAGS:
        return "vision"
    tags = {str(t).lower() for t in card.get("tags") or []}
    if tags.intersection(VLM_TAG_KEYWORDS):
        return "vision"
    return "language"


def weights_type_reason(card: dict) -> str:
    """Human-readable signal behind infer_weights_type ('' when default)."""
    pipeline = str(card.get("pipeline_tag") or "").lower()
    if pipeline in VLM_PIPELINE_TAGS:
        return f"pipeline_tag {pipeline}"
    tags = {str(t).lower() for t in card.get("tags") or []}
    if tags.intersection(VLM_TAG_KEYWORDS):
        return "card tags"
    return ""


def format_chain(chain: List[str]) -> str:
    return " -> ".join(chain)


def derive_build_plan(
    repo_id: str,
    orgs: List[str] = SKELETON_ORGS,
    fetch: Callable[[str], dict] = fetch_card,
    probe: Callable[[str], bool] = _skeleton_exists,
) -> BuildPlan:
    """Resolve everything about a build that the input repo's metadata implies."""
    plan = BuildPlan(repo_id=repo_id)
    cache: Dict[str, dict] = {}
    plan.chain = walk_base_chain(repo_id, cache=cache, fetch=fetch)
    card = cache.get(repo_id) or {}
    plan.pipeline_tag = card.get("pipeline_tag")
    plan.weights_type = infer_weights_type(card)
    plan.weights_reason = weights_type_reason(card)
    plan.display_name = derive_display_name(card, repo_id)
    plan.skeleton = find_skeleton(plan.chain, orgs, probe)
    if plan.skeleton:
        plan.size_token = parse_size_token(os.path.basename(plan.skeleton))
    if plan.display_name:
        parts = [plan.display_name]
        # Don't duplicate a size the name already carries
        # (Qwen3.5-9B-Uncensored... already has 9B; medgemma-1.5-4b-it has 4b).
        segments = {seg.lower() for seg in plan.display_name.split("-")}
        if plan.size_token and plan.size_token.lower() not in segments:
            parts.append(plan.size_token)
        plan.output_name = "-".join(parts) + SKELETON_SUFFIX
    return plan
