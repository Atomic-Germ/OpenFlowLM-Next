#!/usr/bin/env python3
"""Derive one kernel spec per (family, size) in the model catalogue.

    python open_kernels/gen_catalogue_specs.py [--out DIR] [--check] [--jobs N]

WHY THIS EXISTS. `recipes/specs/*.json` was a hand-maintained list of twelve
files, each naming the model it was for in `extra.model`, and the build exported
exactly those. That is how a model with a perfectly good recipe went unserved:
Qwen3-0.6B derives a spec today and had no file, so `oflm add` computed a hash
that matched nothing and the model fell through to the closed engine.

It is also how two specs ended up pointing at the SAME output directory --
`gemma3-12b.json` and `gemma3-4b.json` both said `extra.model =
"Gemma3-4B-NPU2"` and differ only in `num_layers` (48 vs 34), so whichever ran
last won and the other model's set never existed. A hand-kept list is a
correctness hazard, not just an inconvenience.

So the catalogue is the input. Every model in `model_list.json` gets its
`config.json` fetched, a spec derived with `spec_from_hf_config` (which reads
the weight format from the q4nx-build config -- the same authority `oflm add`
reads, so the two hashes agree), and the result deduplicated by `spec_hash`.
Many catalogue entries are one geometry under several names, so N models
collapse to far fewer sets -- and each set is named for the FAMILY AND SIZE, so
it serves every model of that geometry rather than the one whose name happened
to be written down first.

THAT NAMING IS THE POINT. `Qwen3.5-9B` and `Qwen3.8-Distilled-9B` are the same
architecture at the same size, and a finetune called `Qwen3.5-9B-Claude-4.8`
is the same again: all three want one kernel set, and `oflm add` finds it by
`spec_hash` without caring what the model is called. The set is named for the
geometry so that the directory name cannot become the thing that decides.

A family with no recipe is reported as NOT IMPLEMENTED and skipped. That is the
honest outcome, not a failure: `gptoss` is in `recipes.families.NOT_IMPLEMENTED`
with the reason, and inventing kernels for it would be worse than saying so.

Requires network access (it fetches each model's `config.json` and
`tokenizer.json` from the catalogue's own URLs). It reads no weight bytes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from recipes.load import spec_from_hf_config  # noqa: E402
from recipes.spec import SpecError  # noqa: E402

CATALOGUE = REPO / "src" / "model_list.json"
SPECS_DIR = HERE / "recipes" / "specs"

# Fetched with these; they are what the deriver reads and what every catalogue
# entry already lists in its `files`.
_WANT = ("config.json", "tokenizer.json")


def load_catalogue() -> list[dict]:
    """Every model entry in the registry, flattened out of its nesting."""
    doc = json.loads(CATALOGUE.read_text(encoding="utf-8"))
    out: list[dict] = []

    def rec(node):
        if isinstance(node, dict):
            if "name" in node and "url" in node:
                out.append(node)
            for v in node.values():
                rec(v)
        elif isinstance(node, list):
            for v in node:
                rec(v)

    rec(doc)
    return sorted(out, key=lambda m: m["name"])


def _repo_of(entry: dict) -> str:
    """`owner/name` for the entry, from either URL field.

    The catalogue is not consistent about this: some `url` values are the bare
    repo (`.../Qwen3-4B-NPU2`) and some already carry the file path
    (`.../Qwen3.5-4B-NPU2/resolve/main`). Stripping a known host prefix alone
    leaves `/resolve/main` glued onto the repo name and every fetch 404s, which
    reads as "this model has no config" rather than as a bug here -- so the
    revision suffix is removed too.
    """
    for key in ("url", "ms_url"):
        url = entry.get(key) or ""
        for pre in ("https://huggingface.co/", "https://www.modelscope.cn/models/",
                    "https://modelscope.cn/models/"):
            if url.startswith(pre):
                repo = url[len(pre):].rstrip("/")
                for suf in ("/resolve/main", "/resolve/master", "/tree/main"):
                    if repo.endswith(suf):
                        repo = repo[: -len(suf)]
                return repo
    url = (entry.get("url") or "").rstrip("/")
    return url.rsplit("/", 1)[-1]


def fetch(repo: str, filename: str, timeout: float = 30.0) -> bytes | None:
    """One file from the model's repo, or None when it cannot be read.

    Tries the resolve path on HuggingFace, then ModelScope -- the same two
    sources `oflm add` pulls from, so a model the user can install is a model
    this can derive.

    An `HF_TOKEN` is forwarded when set, for the same reason `oflm add` forwards
    it: a gated repo answers 401 to an anonymous request and 200 to an
    authenticated one, so without the token a private model reports as
    unreadable and is skipped as if it did not exist. A gated model is a
    supported model that this could not see, and the report has to say which.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    hf_headers = {"User-Agent": "oflm-kernel-build/1.0"}
    if token:
        hf_headers["Authorization"] = f"Bearer {token}"
    # ModelScope never gets the HuggingFace credential.
    ms_headers = {"User-Agent": "oflm-kernel-build/1.0"}

    urls = [f"https://huggingface.co/{repo}/resolve/main/{filename}"]
    if not repo.startswith("amd/"):
        urls.append(f"https://modelscope.cn/models/{repo}/resolve/master/{filename}")
    for i, url in enumerate(urls):
        try:
            req = urllib.request.Request(url, headers=ms_headers if i else hf_headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError):
            continue
    return None


def unreadable_reason(repo: str, filename: str) -> str:
    """Why a fetch came back empty -- 401 and 404 are different problems.

    "no config.json" says the model is not there; a gated repo is there and this
    could not see it, which is fixed by HF_TOKEN and not by writing a recipe.
    Conflating them would put a supported model in the NOT IMPLEMENTED list.
    """
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    req = urllib.request.Request(url, headers={"User-Agent": "oflm-kernel-build/1.0"})
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        req.add_header("Authorization", "Bearer " + (
            os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")))
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return f"{filename} unreadable (HTTP {r.status})"
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return (f"{filename} is gated/private (HTTP {e.code}) -- set HF_TOKEN; "
                    f"this is not a missing recipe")
        if e.code == 404:
            return f"no {filename} in {repo} (HTTP 404)"
        return f"{filename} unreadable (HTTP {e.code})"
    except Exception as e:  # noqa: BLE001 - the message is the report
        return f"{filename} unreadable ({type(e).__name__}: {e})"


def vocab_of(raw: bytes | None) -> int | None:
    """The tokenizer's id count, the same measure `tokenizer_vocab` takes.

    Only the max id is needed, and a full tokenizer.json is a few MB per model,
    so the ids are pulled out of the parsed object rather than scanned as text.
    """
    if not raw:
        return None
    try:
        t = json.loads(raw)
    except ValueError:
        return None
    ids = [a["id"] for a in t.get("added_tokens", []) if isinstance(a, dict) and "id" in a]
    vocab = t.get("model", {}).get("vocab", {})
    if isinstance(vocab, dict):
        ids += [v for v in vocab.values() if isinstance(v, int)]
    return max(ids) + 1 if ids else None


def derive(entry: dict) -> dict:
    """One catalogue entry -> a spec, or a named reason there is none."""
    name = entry["name"]
    details = entry.get("details", {}) or {}
    cat_family = details.get("family")
    size = details.get("parameter_size")
    repo = _repo_of(entry)

    cfg_raw = fetch(repo, "config.json")
    if cfg_raw is None:
        return {"name": name, "skip": unreadable_reason(repo, "config.json")}
    try:
        cfg = json.loads(cfg_raw)
    except ValueError as e:
        return {"name": name, "skip": f"config.json does not parse: {e}"}

    rv = vocab_of(fetch(repo, "tokenizer.json"))
    try:
        spec = spec_from_hf_config(cfg, rv, cat_family, size)
    except SpecError as e:
        return {"name": name, "skip": str(e), "family": cat_family, "size": size}
    except Exception as e:  # noqa: BLE001 - the message is the report
        return {"name": name, "skip": f"{type(e).__name__}: {e}"}
    # The DERIVER is not the recipe. A family can have a working spec builder and
    # no kernels to build from it -- `gptoss` is exactly that, and it is in
    # families.NOT_IMPLEMENTED with the reason. Writing a spec for one would have
    # the build discover it forty minutes in, at the first design compile, instead
    # of here. So ask the recipe now, and report what it says.
    try:
        from recipes.families import for_spec
        for_spec(spec)
    except NotImplementedError as e:
        return {"name": name, "skip": str(e).splitlines()[0], "family": cat_family, "size": size}
    except Exception as e:  # noqa: BLE001 - the message is the report
        return {"name": name, "skip": f"recipe {family_of(spec)}: {type(e).__name__}: {e}",
                "family": cat_family, "size": size}
    return {"name": name, "spec": spec, "family": cat_family, "size": size}


def family_of(spec) -> str:
    return getattr(spec, "family", "?")


def set_name(spec) -> str:
    """The directory name for a set: family, size, and the geometry that differs.

    Family and hidden size, because that is what a person recognises. Then the
    layer count, because two members of one family at one nominal size can still
    be different depths, and a name that collides silently overwrites one set
    with the other -- the exact failure `gemma3-12b.json` had when two specs both
    said `Gemma3-4B-NPU2`.

    The tokenizer's id count is deliberately NOT in the name. It is not a kernel
    input (the head is sized from the padded vocab) and it is not in spec_hash
    either, so putting it here would split one set per finetune: LFM2-1.2B and
    LFM2.5-1.2B differ by two ids and want the same kernels. Where it did matter
    to tell sets apart it is now carried by a suffix derived from the hash, which
    is the only thing that can separate two genuinely different specs.
    """
    return f"{spec.family}-h{spec.hidden}-L{spec.num_layers}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=SPECS_DIR,
                    help="where the generated spec files go (default: recipes/specs)")
    ap.add_argument("--check", action="store_true",
                    help="report coverage and exit non-zero if anything is unbuildable")
    ap.add_argument("--jobs", type=int, default=8, help="parallel fetches")
    ap.add_argument("--list", action="store_true", help="print the plan, write nothing")
    a = ap.parse_args()

    models = load_catalogue()
    print(f"-- catalogue: {len(models)} models")

    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        results = list(ex.map(derive, models))

    by_hash: dict[str, dict] = {}
    unbuildable: list[dict] = []
    for r in results:
        if "spec" not in r:
            unbuildable.append(r)
            continue
        h = r["spec"].spec_hash()
        by_hash.setdefault(h, {"spec": r["spec"], "members": []})["members"].append(r["name"])

    print(f"-- derived: {len(by_hash)} distinct kernel set(s) from "
          f"{sum(len(v['members']) for v in by_hash.values())} model(s)")
    for h, v in sorted(by_hash.items(), key=lambda kv: kv[1]["spec"].family):
        s = v["spec"]
        print(f"   {set_name(s):<28} {s.family:<9} hidden={s.hidden:<5} layers={s.num_layers:<3} "
              f"vocab={s.vocab:<7} {h[:19]}")
        for m in v["members"]:
            print(f"       serves {m}")

    if unbuildable:
        # A gated repo is not a missing recipe, and lumping the two together
        # would put a supported model in the list of families needing kernel work.
        gated = [r for r in unbuildable if "gated/private" in r["skip"]]
        real = [r for r in unbuildable if "gated/private" not in r["skip"]]
        if gated:
            print(f"\n-- GATED ({len(gated)}): readable with HF_TOKEN. Not a recipe gap.")
            for r in gated:
                print(f"   {r['name']:<34} {r['skip'].splitlines()[0][:88]}")
        if real:
            print(f"\n-- NOT IMPLEMENTED ({len(real)}): no recipe, or the metadata does "
                  f"not derive. These fall back to the closed engine.")
            for r in real:
                print(f"   {r['name']:<34} {r['skip'].splitlines()[0][:88]}")

    if a.list:
        return 0

    out: Path = a.out
    out.mkdir(parents=True, exist_ok=True)
    # The specs directory is DERIVED and wholly owned by this script, so it is
    # cleared rather than merged into. A spec left over from an earlier naming
    # scheme otherwise stays a valid-looking file that the exporter globs, and
    # the same kernels get built twice under two names -- which is what happened
    # when `real_vocab` left the name and the two naming schemes briefly coexisted.
    # The bookkeeping stamp is rewritten below, so it goes too.
    for stale in sorted(out.glob("*.json")):
        stale.unlink()
    stamp = out / ".stamp.json"
    if stamp.exists():
        stamp.unlink()
    written = 0
    # Two different specs must never land on one file: the exporter reads
    # extra["model"] to decide its output directory, so a collision is one set
    # silently overwriting another -- the exact failure `gemma3-12b.json` had.
    #
    # A name built from family+size+geometry CANNOT be made unique that way.
    # Qwen3-4B and Qwen3-VL-4B are the same family, hidden, layer count and real
    # vocab and differ only in fields the name does not carry (the VL tower's
    # presence), so the name gets a short hash suffix instead. The suffix is
    # derived from the spec, so the same spec always gets the same name and a
    # rebuild is reproducible -- which is what makes it usable as a directory.
    seen_name: dict[str, str] = {}
    for h, v in by_hash.items():
        s = v["spec"]
        nm = set_name(s)
        if nm in seen_name and seen_name[nm] != h:
            nm = f"{nm}-{h[7:13]}"
        seen_name[nm] = h
        s.extra["model"] = nm
        s.extra["serves"] = v["members"]
        p = out / f"{nm}.json"
        p.write_text(s.to_json(), encoding="utf-8", newline="\n")
        written += 1
    print(f"\n-- wrote {written} spec file(s) to {out}")
    if a.check and unbuildable:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
