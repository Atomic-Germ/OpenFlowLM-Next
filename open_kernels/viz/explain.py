"""The explainers (src/viz/explain/*.md) as the page reads them, and what a viz.json needs of them (VIZ-EXPLAIN-COVERAGE)."""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

EXPLAIN = Path(__file__).resolve().parents[2] / "src" / "viz" / "explain"
# The concepts the page links to whatever the model: tiles, memory, the host and GPU chips, the timing badge.
PAGE_CONCEPTS = ("concept:shim", "concept:mem", "concept:core", "concept:ddr", "concept:ctx", "concept:gpu",
                 "concept:host", "concept:modelled")
HEAD = re.compile(r"^(Title|Summary|Spec|Source):\s*(.*)$")


def parse(text: str) -> dict[str, dict]:
    """`## kind:key` sections -> {key: {title, summary, spec, source, body}}; keys may hold * globs."""
    out: dict[str, dict] = {}
    for part in ("\n" + text).split("\n## ")[1:]:
        lines = part.split("\n")
        key = lines.pop(0).strip()
        e = {"title": "", "summary": "", "spec": [], "source": "", "body": ""}
        body, head = [], True
        for ln in lines:
            m = HEAD.match(ln) if head else None
            if m:
                if m[1] == "Spec":
                    e["spec"] = [s for s in re.split(r"[,\s]+", m[2]) if s]
                else:
                    e[m[1].lower()] = m[2].strip()
                continue
            if head and not ln.strip():
                continue
            head = False
            body.append(ln)
        e["body"] = "\n".join(body).strip()
        out[key] = e
    return out


def load(root: Path = EXPLAIN) -> dict[str, dict]:
    return parse("\n\n".join(p.read_text(encoding="utf-8") for p in sorted(Path(root).glob("*.md"))))


def find(entries: dict, key: str):
    if key in entries:
        return entries[key]
    for k, e in entries.items():
        if "*" in k and fnmatch.fnmatchcase(key, k):
            return e
    return None


def required(viz: dict) -> list[str]:
    """Every key the page can ask for while showing this viz.json."""
    d = viz["decode"]
    keys: list[str] = []
    for e in d["events"]:
        if e["kind"] == "dispatch":
            keys.append(f"dispatch:{e['name']}")
        elif e["kind"] == "host":
            keys.append(f"host:{e['name']}")
    for img in d["images"].values():
        for c in img["cores"]:
            keys += [f"core:{fn}" for fn in c["funcs"]]
        keys += [f"fifo:{f['name']}" for f in img["fifos"]]
    for tp in d["templates"].values():
        keys += [f"buffer:{a}" for a in (viz["kernels"][tp["kernel"]].get("args") or [])]
    keys += [f"layer:{lt}" for lt in viz["layers"]]
    keys += list(PAGE_CONCEPTS)
    return list(dict.fromkeys(keys))


def missing(viz: dict, entries: dict) -> list[str]:
    return [k for k in required(viz) if find(entries, k) is None]


def unknown_specs(entries: dict, spec_ids: set[str]) -> list[tuple[str, str]]:
    return [(k, s) for k, e in entries.items() for s in e["spec"] if s not in spec_ids]


def spec_ids(specs_root: Path) -> set[str]:
    ids: set[str] = set()
    for p in Path(specs_root).glob("*/spec.md"):
        ids |= set(re.findall(r"^###\s+([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)\b", p.read_text(encoding="utf-8"), re.M))
    return ids
