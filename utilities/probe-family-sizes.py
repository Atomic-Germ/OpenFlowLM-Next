#!/usr/bin/env python3
"""Probe candidate GGUF repos with `oflm add --dry-run` to find size/geometry gaps.

Classifies each candidate as:
  - served: a matching open-kernel set already serves this geometry
  - would_build: family/recipe recognized, different geometry (buildable candidate)
  - no_set: no set and cannot build as stated (no checkout/build path reported)
  - unreadable: geometry could not be read / no recipe for architecture
  - not_found: repo or model not found
  - error: probe failed

Prefers mradermacher *-i1-GGUF repos (often carry Q4_1) and can fall back to
unsloth. Outputs a matrix and optionally JSON for reuse.

Usage examples:
  python3 utilities/probe-family-sizes.py --builtin
  python3 utilities/probe-family-sizes.py --builtin --json out.json
  python3 utilities/probe-family-sizes.py --file candidates.txt
  python3 utilities/probe-family-sizes.py --repo mradermacher/Qwen2.5-7B-Instruct-i1-GGUF

candidates.txt format: one repo id per line, or `label<TAB>repo1,repo2,...`
(blank lines and # comments ignored).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Built-in candidates focused on families with existing recipes/designs where
# filling geometries is the leverage point. Prefer mradermacher -i1-GGUF, then
# unsloth as fallback. This is a starter set, not exhaustive.
BUILTIN: list[tuple[str, list[str]]] = [
    ("qwen2/2.5-0.5b", ["mradermacher/Qwen2.5-0.5B-Instruct-i1-GGUF", "unsloth/Qwen2.5-0.5B-Instruct-GGUF"]),
    ("qwen2/2.5-1.5b", ["mradermacher/Qwen2.5-1.5B-Instruct-i1-GGUF", "unsloth/Qwen2.5-1.5B-Instruct-GGUF"]),
    ("qwen2/2.5-7b",   ["mradermacher/Qwen2.5-7B-Instruct-i1-GGUF", "unsloth/Qwen2.5-7B-Instruct-GGUF"]),
    ("qwen2/2.5-14b",  ["mradermacher/Qwen2.5-14B-Instruct-i1-GGUF", "unsloth/Qwen2.5-14B-Instruct-GGUF"]),
    ("qwen2/2.5-32b",  ["mradermacher/Qwen2.5-32B-Instruct-i1-GGUF", "unsloth/Qwen2.5-32B-Instruct-GGUF"]),
    ("qwen2vl/2.5-3b", ["mradermacher/Qwen2.5-VL-3B-Instruct-i1-GGUF", "unsloth/Qwen2.5-VL-3B-Instruct-GGUF"]),
    ("qwen2vl/2.5-7b", ["mradermacher/Qwen2.5-VL-7B-Instruct-i1-GGUF", "unsloth/Qwen2.5-VL-7B-Instruct-GGUF"]),
    ("qwen3/32b",      ["mradermacher/Qwen3-32B-i1-GGUF", "unsloth/Qwen3-32B-GGUF"]),
    ("qwen3/30b-a3b",  ["mradermacher/Qwen3-30B-A3B-i1-GGUF", "unsloth/Qwen3-30B-A3B-GGUF"]),
    ("qwen3vl/2b",     ["mradermacher/Qwen3-VL-2B-Instruct-i1-GGUF", "unsloth/Qwen3-VL-2B-Instruct-GGUF"]),
    ("qwen3vl/8b",     ["mradermacher/Qwen3-VL-8B-Instruct-i1-GGUF", "unsloth/Qwen3-VL-8B-Instruct-GGUF"]),
    ("qwen3vl/32b",    ["mradermacher/Qwen3-VL-32B-Instruct-i1-GGUF", "unsloth/Qwen3-VL-32B-Instruct-GGUF"]),
    ("qwen3.5/35b-a3b",["mradermacher/Qwen3.5-35B-A3B-i1-GGUF", "unsloth/Qwen3.5-35B-A3B-GGUF"]),
    ("llama3.2/1b",    ["mradermacher/Llama-3.2-1B-Instruct-i1-GGUF", "unsloth/Llama-3.2-1B-Instruct-GGUF"]),
    ("llama3.2/3b",    ["mradermacher/Llama-3.2-3B-Instruct-i1-GGUF", "unsloth/Llama-3.2-3B-Instruct-GGUF"]),
    ("llama3.1/8b",    ["mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF", "unsloth/Meta-Llama-3.1-8B-Instruct-GGUF"]),
    ("llama3.1/70b",   ["mradermacher/Meta-Llama-3.1-70B-Instruct-i1-GGUF", "unsloth/Meta-Llama-3.1-70B-Instruct-GGUF"]),
    ("gemma3/1b",      ["mradermacher/gemma-3-1b-it-i1-GGUF", "unsloth/gemma-3-1b-it-GGUF"]),
    ("gemma3/12b",     ["mradermacher/gemma-3-12b-it-i1-GGUF", "unsloth/gemma-3-12b-it-GGUF"]),
    ("gemma3/27b",     ["mradermacher/gemma-3-27b-it-i1-GGUF", "unsloth/gemma-3-27b-it-GGUF"]),
    ("gemma4/12b",     ["mradermacher/gemma-4-12B-it-i1-GGUF", "unsloth/gemma-4-12B-it-GGUF"]),
    ("gemma4/26b-a4b", ["mradermacher/gemma-4-26B-A4B-it-i1-GGUF", "unsloth/gemma-4-26B-A4B-it-GGUF"]),
    ("gemma4/31b",     ["mradermacher/gemma-4-31B-it-i1-GGUF", "unsloth/gemma-4-31B-it-GGUF"]),
    ("phi3.5/mini",    ["mradermacher/Phi-3.5-mini-instruct-i1-GGUF", "unsloth/Phi-3.5-mini-instruct-GGUF"]),
    ("phi4/mini",      ["mradermacher/Phi-4-mini-instruct-i1-GGUF", "unsloth/Phi-4-mini-instruct-GGUF"]),
    ("phi4/reasoning", ["mradermacher/Phi-4-mini-reasoning-i1-GGUF", "unsloth/Phi-4-mini-reasoning-GGUF"]),
    ("granite4.2/3b",  ["mradermacher/granite-4.0-micro-i1-GGUF", "mradermacher/granite-4.0-h-tiny-i1-GGUF", "unsloth/granite-4.0-micro-GGUF"]),
    ("hunyuan/1.8b",   ["mradermacher/HY-MT1.5-1.8B-i1-GGUF", "unsloth/HY-MT1.5-1.8B-GGUF"]),
    ("hunyuan/7b",     ["mradermacher/HY-MT1.5-7B-i1-GGUF", "unsloth/HY-MT1.5-7B-GGUF"]),
    ("lfm2/1.2b",      ["mradermacher/LFM2-1.2B-i1-GGUF", "unsloth/LFM2-1.2B-GGUF"]),
    ("lfm2/2.6b",      ["mradermacher/LFM2-2.6B-i1-GGUF", "unsloth/LFM2-2.6B-GGUF"]),
    ("lfm2.5/1.2b",    ["mradermacher/LFM2.5-1.2B-Instruct-i1-GGUF", "unsloth/LFM2.5-1.2B-Instruct-GGUF"]),
    ("k2/3.7b",        ["mradermacher/K2-Horizon-3.7B-i1-GGUF", "unsloth/K2-Horizon-3.7B-GGUF"]),
    ("k2/7b",          ["mradermacher/K2-Horizon-7B-i1-GGUF", "unsloth/K2-Horizon-7B-GGUF"]),
]


@dataclass
class Result:
    label: str
    repo: str
    status: str
    family: str = ""
    gguf_arch: str = ""
    kernels: str = ""
    detail: str = ""
    rc: int = 0


def classify(output: str, rc: int) -> tuple[str, dict]:
    info: dict[str, str] = {}
    m = re.search(r"^.*\(([^,]+)\), quant", output, re.M)
    if m:
        info["gguf_arch"] = m.group(1).strip()
    m = re.search(r"family '([^']+)' from", output)
    if m:
        info["family"] = m.group(1)
    m = re.search(r"gguf kernels\s*:\s*(.+)", output)
    if m:
        info["kernels"] = m.group(1).strip()

    low = output.lower()
    if "could not list the repo tree" in low and "404" in low and "no recipe" not in low and "gguf kernels" not in low and "family '" not in low:
        return "not_found", info
    if "404" in low and "assuming the npu2 path" in low and "family '" not in low and "gguf kernels" not in low:
        # weak signal; could still be unknown. fall through to other checks
        pass
    if "could not derive a size" in low and "family '" not in low:
        return "not_found", info
    if "no open kernel set can run this model" in low or "no recipe for its architecture" in low or "this build has no recipe" in low:
        return "unreadable", info
    if "geometry could not be read" in low:
        return "unreadable", info
    k = info.get("kernels", "")
    if k.startswith("(would build"):
        return "would_build", info
    if k.startswith("(none"):
        return "no_set", info
    if k and not k.startswith("("):
        return "served", info
    if "not found" in low or "404" in low:
        return "not_found", info
    if rc != 0:
        return "error", info
    return "unknown", info


def probe(label: str, repo: str, timeout: int = 180) -> Result:
    env = dict(os.environ)
    cmd = ["oflm", "add", repo, "--dry-run"]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout, env=env, cwd=str(ROOT))
        out = p.stdout or ""
        status, info = classify(out, p.returncode)
        # keep a compact detail line
        detail = ""
        for line in out.splitlines():
            if any(s in line for s in ["gguf kernels", "would build", "no open kernel", "geometry could not", "family '", "xclbin source"]):
                detail = line.strip()
        return Result(label=label, repo=repo, status=status,
                      family=info.get("family", ""), gguf_arch=info.get("gguf_arch", ""),
                      kernels=info.get("kernels", ""), detail=detail, rc=p.returncode)
    except subprocess.TimeoutExpired:
        return Result(label=label, repo=repo, status="error", detail="timeout")
    except Exception as e:
        return Result(label=label, repo=repo, status="error", detail=str(e))


def load_candidates(path: Path) -> list[tuple[str, list[str]]]:
    items: list[tuple[str, list[str]]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            label, repos = line.split("\t", 1)
            items.append((label.strip(), [r.strip() for r in repos.split(",") if r.strip()]))
        elif "," in line:
            parts = [r.strip() for r in line.split(",") if r.strip()]
            items.append((parts[0], parts))
        else:
            items.append((line, [line]))
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--builtin", action="store_true", help="use built-in candidate set")
    ap.add_argument("--file", help="candidate file (label<TAB>repo1,repo2 or one repo per line)")
    ap.add_argument("--repo", action="append", default=[], help="probe a specific repo (repeatable)")
    ap.add_argument("--jobs", type=int, default=4, help="parallel probes (default 4)")
    ap.add_argument("--timeout", type=int, default=180, help="per-probe timeout seconds")
    ap.add_argument("--json", help="write full results as JSON")
    ap.add_argument("--first-success", action="store_true", default=True,
                    help="stop trying fallbacks for a label once one resolves (default)")
    ap.add_argument("--all-fallbacks", action="store_false", dest="first_success",
                    help="try all fallback repos per label")
    args = ap.parse_args()

    groups: list[tuple[str, list[str]]] = []
    if args.builtin:
        groups.extend(BUILTIN)
    if args.file:
        groups.extend(load_candidates(Path(args.file)))
    for r in args.repo:
        groups.append((r, [r]))
    if not groups:
        ap.error("provide --builtin, --file, or --repo")

    if not shutil.which("oflm"):
        print("error: `oflm` not found in PATH", file=sys.stderr)
        return 2

    # Flatten to probe tasks but preserve label grouping for fallback logic.
    results: list[Result] = []
    # Simple strategy: probe first candidate per label in parallel waves; if not_found/error try fallback.
    def probe_label(label: str, repos: list[str]) -> Result:
        last: Result | None = None
        for repo in repos:
            r = probe(label, repo, timeout=args.timeout)
            last = r
            if r.status != "not_found":
                return r
            # if not_found, try next fallback
        return last  # type: ignore

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(probe_label, label, repos): (label, repos) for label, repos in groups}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(f"[{r.status:11s}] {r.label:18s} {r.repo}  {r.detail}", flush=True)

    # summary matrix
    order = {"served": 0, "would_build": 1, "no_set": 2, "unreadable": 3, "not_found": 4, "error": 5, "unknown": 6}
    results.sort(key=lambda r: (order.get(r.status, 9), r.label))
    print("\n== summary ==")
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    for k in sorted(counts, key=lambda x: order.get(x, 9)):
        print(f"  {k:11s}: {counts[k]}")
    print("\n== would_build (fill candidates) ==")
    for r in results:
        if r.status == "would_build":
            print(f"  - {r.label:18s} {r.repo} :: {r.detail}")

    if args.json:
        Path(args.json).write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
