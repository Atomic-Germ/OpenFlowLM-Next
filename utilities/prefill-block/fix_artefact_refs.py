"""Point the `artefact` fields at the tool that produces them, not at a path under runs/.

`runs/` is a scratch directory (see .gitignore): a test that read it was asserting what
somebody once observed, and those tests are gone. The JSON provenance that remains is the
measured NUMBERS, and each block already carries the tool, the command and the fixture that
produced it. The `artefact` key was the one field still naming a file that no longer exists,
which makes the document look self-referential.

    python fix_artefact_refs.py            # rewrite the two files
    python fix_artefact_refs.py --check    # fail if any artefact path is still a runs/ path
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent

# path in the document -> what replaces it
REPLACEMENTS = {
    'tail-crossover.json': {
        'evidence': ("not committed: runs/ is scratch (.gitignore)."
                     " Reproduce with: python crossover_local.py --block-min 1"),
        'superseded_sweep': ("not committed: runs/ is scratch."
                             " Reproduce with the superseded tail_crossover.py, kept as the"
                             " record of how 128 was produced"),
        'route_matrix_after_the_change': ("not committed: runs/ is scratch."
                                          " Reproduce with: python route_matrix.py --mode block"),
    },
    'qwen38-production-baseline.json': {
        'prefill': ("not committed: runs/ is scratch (.gitignore)."
                    " Reproduce with: python oflm_sanity.py --lengths 256 512 1024"),
        'decode': ("not committed: runs/ is scratch."
                   " Reproduce with: python decode_rate.py --tokens 32 --ctx 4096"),
        'generation_headroom': ("not committed: runs/ is scratch."
                                " Reproduce with: python think_boundary.py --ctx 512"),
    },
}


def walk(node, fn):
    """Apply `fn(dict, key)` at every level, so a nested block is covered too."""
    if isinstance(node, dict):
        for key in list(node):
            fn(node, key)
            walk(node[key], fn)
    elif isinstance(node, list):
        for item in node:
            walk(item, fn)


def bad_paths(doc) -> list[str]:
    found: list[str] = []

    def collect(d, key):
        if key == 'artefact' and isinstance(d.get(key), str) and '/runs/' in d[key]:
            found.append(d[key])

    walk(doc, collect)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true',
                    help='fail if any artefact still names a runs/ path')
    a = ap.parse_args()

    problems = 0
    for name, blocks in REPLACEMENTS.items():
        path = HERE / name
        doc = json.loads(path.read_text())
        if a.check:
            bad = bad_paths(doc)
            for b in bad:
                print(f'{name}: still points at {b}', file=sys.stderr)
            problems += len(bad)
            continue
        for block, text in blocks.items():
            if block not in doc:
                print(f'{name}: no block {block!r}', file=sys.stderr)
                problems += 1
                continue
            doc[block]['artefact'] = text
        path.write_text(json.dumps(doc, indent=1) + '\n')
        print(f'{name}: updated {len(blocks)} artefact field(s)')

    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
