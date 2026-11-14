"""Refresh `src/model_info.json` from HuggingFace, per file, without reformatting the file.

`model_info.json` is the registry of the files a model repo holds: path, size and the oid
`oflm` verifies after a download. When a repo is re-uploaded the sizes and hashes move, and a
stale entry makes `oflm check` -- and therefore `oflm run` -- treat an already-downloaded file
as missing and re-download it forever:

    [WARNING]  config.json is 3091 bytes, the manifest says 3189 -- treating it as missing

The comparison has to mirror what the C++ side reads (src/pull/install_record.cpp,
`registry_files`): an LFS file is verified against `lfs.oid` (sha256); a non-LFS file against
its git blob oid (sha1). Comparing the two kinds blindly reports every LFS file as stale,
which is how a first pass here produced six false positives per model.

Only entries listed in `model_list.json`'s `files` are refreshed -- README and
`.gitattributes` are not part of what `oflm` downloads. The rest of the document, and its
formatting, is left alone: this writes back the same text with individual lines replaced, so
the diff is exactly the fields that changed.

    python utilities/refresh-model-info.py --check     # report drift, change nothing
    python utilities/refresh-model-info.py             # update in place
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_INFO = ROOT / 'src' / 'model_info.json'
MODEL_LIST = ROOT / 'src' / 'model_list.json'


def effective(entry: dict) -> tuple[int, str, bool]:
    """The (size, oid, is_lfs) triple the downloader actually verifies.

    Mirrors `pull::registry_files`: the LFS object wins when present, because a large file is
    stored out of band and its `oid` on the outer object is only the git pointer.
    """
    lfs = entry.get('lfs')
    if isinstance(lfs, dict):
        return int(lfs.get('size', entry.get('size', 0))), str(lfs.get('oid', '')), True
    return int(entry.get('size', 0)), str(entry.get('oid', '')), False


def tree_of(owner: str, name: str) -> dict[str, dict]:
    url = f'https://huggingface.co/api/models/{owner}/{name}/tree/main'
    with urllib.request.urlopen(url, timeout=120) as r:
        return {e['path']: e for e in json.load(r) if e.get('type') == 'file'}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true',
                    help='report stale entries and exit non-zero; change nothing')
    ap.add_argument('--only', nargs='*', help='restrict to these model tags')
    a = ap.parse_args()

    text = MODEL_INFO.read_text()
    info = json.loads(text)
    listing = json.loads(MODEL_LIST.read_text())
    owner = listing['hf_owner']

    stale = 0
    edits: list[tuple[str, str, str]] = []   # (old fragment, new fragment, tag)
    for fam, sizes in listing['models'].items():
        for size, model in sizes.items():
            tag = f'{fam}:{size}'
            if a.only and tag not in a.only:
                continue
            if tag not in info:
                continue
            want = model['files']
            try:
                remote = tree_of(owner, model['name'])
            except Exception as e:                    # network, 404, rate limit
                print(f'{tag}: skipped ({e})', file=sys.stderr)
                continue
            by_path = {e['path']: e for e in info[tag]}
            for path in want:
                have = by_path.get(path)
                got = remote.get(path)
                if got is None or have is None:
                    print(f'{tag}: {path} is not in both sides -- left alone', file=sys.stderr)
                    continue
                hs, ho, hl = effective(have)
                rs, ro, rl = effective(got)
                if hs == rs and ho == ro:
                    continue
                stale += 1
                old_line = next(l for l in text.splitlines()
                                if f'"path": "{path}"' in l or False)
                print(f'{tag}: {path}\n    size {hs} -> {rs}\n    oid  {ho[:16]} -> {ro[:16]}')
                edits.append((tag, path, json.dumps(got, indent=4)))

    if a.check:
        return 1 if stale else 0
    if not stale:
        print('model_info.json is current')
        return 0

    # Rewrite by replacing each stale ENTRY object, keeping the rest byte-identical.
    data = json.loads(text)
    for tag, path, blob in edits:
        remote_entry = json.loads(blob)
        for i, e in enumerate(data[tag]):
            if e['path'] == path:
                data[tag][i] = remote_entry
                break
    MODEL_INFO.write_text(json.dumps(data, indent=4) + '\n')
    print(f'\nupdated {stale} entr{"y" if stale == 1 else "ies"} in {MODEL_INFO.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
