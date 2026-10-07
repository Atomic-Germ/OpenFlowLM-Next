"""Check that every model in src/model_list.json is where the list says it is.

    python utilities/check-model-list/check_model_list.py [--list PATH] [--owner ACCOUNT]

For each entry with a Hugging Face `file_url`, asks Hugging Face for the repository's files and
reports a repository that does not answer, and any file the entry's `files` names that is not
there. `--owner` checks our own repositories under another account (what OFLM_HF_OWNER does at
run time), to see whether that account is ready before the list's `hf_owner` is changed to it.
Exits 1 if anything is missing. Standard library only; HF_TOKEN is sent when set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

SLOT = "{hf_owner}"
REPO = Path(__file__).resolve().parents[2]


def resolved(entry: dict, owner: str) -> tuple[str, bool]:
    """The entry's file_url with the account filled in, and whether it is one of ours."""
    url = entry.get("file_url") or ""
    return url.replace(SLOT, owner), SLOT in url


def tree(url: str) -> set[str]:
    req = urllib.request.Request(url + ("&" if "?" in url else "?") + "recursive=true")
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return {f["path"] for f in json.load(r) if f.get("type") == "file"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", type=Path, default=REPO / "src" / "model_list.json")
    ap.add_argument("--owner", help="check our repositories under this account instead of the list's hf_owner")
    a = ap.parse_args()
    data = json.loads(a.list.read_text(encoding="utf-8"))
    owner = a.owner or data.get("hf_owner", "")
    bad = checked = 0
    for family, sizes in data["models"].items():
        for size, entry in sizes.items():
            tag = f"{family}:{size}"
            url, ours = resolved(entry, owner)
            if not url.startswith("https://huggingface.co/"):
                continue                       # a local model, or one hosted elsewhere
            if ours and not owner:
                print(f"FAIL {tag}: uses {SLOT}, but the list has no hf_owner")
                bad += 1
                continue
            if a.owner and not ours:
                continue                       # --owner asks about our repositories only
            checked += 1
            try:
                have = tree(url)
            except urllib.error.HTTPError as e:
                print(f"FAIL {tag}: {url} answers HTTP {e.code}")
                bad += 1
                continue
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"FAIL {tag}: {url}: {e}")
                bad += 1
                continue
            missing = [f for f in entry.get("files", []) if f not in have]
            if missing:
                print(f"FAIL {tag}: {url} lacks {', '.join(missing)}")
                bad += 1
    print(f"{checked} entries checked under '{owner}', {bad} with something missing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
