"""Generate recipes/specs/<slug>.json from the matrix, and report what will not build.

    python3 recipes/gen_specs.py            # refresh configs + write specs + print the report
    python3 recipes/gen_specs.py --check    # verify the tree matches the matrix (CI)

Why this exists
---------------
The specs used to be fifteen hand-written JSON files, which made open support a
function of who last needed a size: qwen3 had 4B and 8B but not 0.6B or 1.7B,
qwen3.5 had 9B and 27B but not 0.8B/2B/4B. Nothing noticed the gaps, because a
missing spec looks exactly like a model nobody asked for.

A spec carries no authored information. Every field in it comes out of the
upstream `config.json`, which `ModelSpec.from_hf_config` already reads -- the
checked-in JSON is a serialization of a derivation that is already automatic.
So specs.json is now GENERATED, from `recipes/specs/matrix.json`, and the
catalogue's validated-combination tables decide what comes out.

    a row in the matrix          -> a spec, if the catalogue accepts the geometry
    a row the catalogue refuses  -> a line in the report, naming the table that
                                   has to learn the combination first

That inverts the failure. Adding a size becomes a catalogue-validation
exercise, and the remaining gaps are printed instead of being invisible.

Configs are cached under `specs/_configs/<slug>.json` and checked in, so a
run -- and CI -- needs no network and no Hugging Face credentials. Gated repos
(Gemma, Llama) therefore work once a developer has fetched them once with
`HF_TOKEN` set. `gen_specs.py --refresh` re-fetches.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from recipes import manifest as _manifest                      # noqa: E402
from recipes.catalogue import OpRangeError                      # noqa: E402
from recipes.naming import expected_model                       # noqa: E402
from recipes.spec import ModelSpec, SpecError                   # noqa: E402

SPECS = HERE / "specs"
CONFIGS = SPECS / "_configs"
MATRIX = HERE / "matrix.json"
DEFAULT_MAX_CTX = 32768
_UA = {"User-Agent": "openflowlm-gen-specs/1.0"}


def load_matrix() -> list[dict]:
    data = json.loads(MATRIX.read_text(encoding="utf-8"))
    return data["entries"]


def fetch_config(repo: str, refresh: bool) -> dict | None:
    """The upstream config.json, cached on disk so runs and CI need no network."""
    dest = CONFIGS / (repo.split("/")[-1] + ".json")
    if dest.is_file() and not refresh:
        return json.loads(dest.read_text(encoding="utf-8"))
    url = f"https://huggingface.co/{repo}/resolve/main/config.json"
    headers = dict(_UA)
    if token := os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
            cfg = json.loads(r.read())
    except Exception as exc:                                     # noqa: BLE001
        print(f"  cannot fetch {repo}: {exc}", file=sys.stderr)
        return None
    CONFIGS.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    return cfg


def derive(entry: dict, cfg: dict) -> ModelSpec:
    """The ModelSpec for a row, from its upstream config plus the row's overrides.

    `real_vocab` and `extra.model` are the two fields a config.json cannot
    supply. The real vocabulary is max(token id)+1 from the CONTAINER's
    tokenizer.json, which is lower than the config's nominal `vocab_size` (a
    padded-to-256 head): qwen3-8b is 151669 against the config's 151936, and
    lfm2-1.2b 64400 against 65536. They change the lm_head row count, so they
    are declared in the matrix rather than re-derived. Without this a
    regeneration silently rewrote four verified specs to the nominal number.
    """
    spec = ModelSpec.from_hf_config(cfg, real_vocab=entry.get("real_vocab"))
    extra = dict(spec.extra, model=expected_model(entry["slug"]))
    return dataclasses.replace(spec, extra=extra)


def spec_differs(dest: Path, spec: ModelSpec) -> bool:
    """Text compare with the noise stripped, so key order is not a diff.

    A spec is data; `{"a":1,"b":2}` and `{"b":2,"a":1}` are the same spec. Only
    real field changes should show up as drift.
    """
    if not dest.is_file():
        return True
    try:
        return json.loads(dest.read_text(encoding="utf-8")) != spec.to_dict()
    except ValueError:
        return True


def spec_json(spec: ModelSpec) -> str:
    return json.dumps(spec.to_dict(), indent=2) + "\n"


HASHES = HERE / "spec_hashes.json"


def write_hash_ledger() -> None:
    """`spec_hashes.json` (beside the matrix, NOT in specs/): every spec -> its hash.

    Generated, so a new size does not have to be hand-registered to keep a
    regression guard green. The guard's real value is the FROZEN hashes for the
    specs that predate this (test_lfm2.py holds those); this ledger only answers
    "is every spec on disk accounted for", which is the part that had become a
    chore and the part that made a missing spec invisible.
    """
    ledger = {}
    for dest in sorted(SPECS.glob("*.json")):
        try:
            spec = ModelSpec.from_json(dest.read_text(encoding="utf-8"))
        except Exception as exc:                                # noqa: BLE001
            print(f"  {dest.name}: not a spec ({exc})", file=sys.stderr)
            continue
        ledger[dest.name] = spec.spec_hash()
    HASHES.write_text(json.dumps(ledger, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def check_registry_coverage(entries: list[dict]) -> list[str]:
    """Registry tags with no matrix row.

    Not fatal -- a tag may ship closed-kernel-only on purpose. But it is printed
    every run, so adding a model to model_list.json without deciding its spec
    status is visible rather than silent.
    """
    reg = HERE.parents[1] / "src" / "model_list.json"
    if not reg.is_file():
        return []
    models = json.loads(reg.read_text(encoding="utf-8"))["models"]
    tags = {f"{fam}:{sz}" for fam, sizes in models.items()
            for sz in (sizes if isinstance(sizes, dict) else {})}
    covered = {e.get("tag") for e in entries if e.get("tag")}
    return sorted(tags - covered)


def run(refresh: bool, check: bool, write: bool) -> int:
    entries = load_matrix()
    built, refused, drifted = [], [], []

    for entry in entries:
        slug, repo = entry["slug"], entry["repo"]
        cfg = fetch_config(repo, refresh)
        if cfg is None:
            refused.append((slug, f"no config for {repo} (gated? set HF_TOKEN and --refresh)"))
            continue
        try:
            spec = derive(entry, cfg)
        except (SpecError, KeyError) as exc:
            refused.append((slug, f"derive: {exc}"))
            continue
        if spec.family != entry["family"]:
            refused.append((slug, f"derived {spec.family}, matrix says {entry['family']}"))
            continue
        # The catalogue's refusal is the answer to "does this build".
        try:
            _manifest.manifest(spec, DEFAULT_MAX_CTX, slug)
        except (OpRangeError, SpecError) as exc:
            refused.append((slug, str(exc).replace("\n", " ")))
            continue
        except Exception as exc:                                # noqa: BLE001
            refused.append((slug, f"{type(exc).__name__}: {exc}"))
            continue

        dest = SPECS / f"{slug}.json"
        if spec_differs(dest, spec):
            if check:
                drifted.append(slug)
            elif write:
                dest.write_text(spec_json(spec), encoding="utf-8")
        built.append(slug)

    w = max((len(e["slug"]) for e in entries), default=8)
    print(f"{'slug':{w}}  {'family':11} status")
    print("-" * (w + 40))
    for slug in built:
        fam = next(e["family"] for e in entries if e["slug"] == slug)
        print(f"{slug:{w}}  {fam:11} spec {'stale' if slug in drifted else 'ok'}")
    for slug, why in refused:
        print(f"{slug:{w}}  {'-':11} REFUSED: {why}")

    print()
    print(f"buildable: {len(built)}/{len(entries)}    refused: {len(refused)}")
    write_hash_ledger()
    if gaps := check_registry_coverage(entries):
        print(f"\nshipped registry tags with no matrix row ({len(gaps)}):")
        for t in gaps:
            print(f"  {t}")

    if check and drifted:
        print(f"\nFAIL: {len(drifted)} spec(s) differ from the matrix: {', '.join(drifted)}")
        print("      run: python3 recipes/gen_specs.py")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch every upstream config (needs network; HF_TOKEN for gated repos)")
    ap.add_argument("--check", action="store_true",
                    help="verify the checked-in specs match the matrix; do not write")
    args = ap.parse_args()
    return run(args.refresh, args.check, write=not args.check)


if __name__ == "__main__":
    sys.exit(main())