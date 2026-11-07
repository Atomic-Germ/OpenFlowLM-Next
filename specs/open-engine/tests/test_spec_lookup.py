# Traces: OPEN-SPEC-LOOKUP
"""The prompt-lookup drafter: given token ids, the drafts. The engine's C++ copy (src/open_qwen36/spec_lookup.hpp) is
checked against the same cases by src/open_qwen36/spec_lookup_test.cpp (build.cmd runs it)."""
import json
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "utilities" / "spec-lookup"))
from spec_lookup import fast_lookup, lookup_drafts  # noqa: E402

CASES = json.loads((Path(__file__).parent / "fixtures" / "spec_lookup_cases.json").read_text())["cases"]


def by_definition(hist, k, nmax, nmin, extend):
    """The rule restated with slices: the latest start of the longest suffix n-gram that has a successor."""
    out, first = [], 0
    while len(out) < k:
        cur = list(hist) + out
        s, hit = len(cur), None
        for g in range(nmax, max(1, nmin) - 1, -1):
            if s < g + 1:
                continue
            starts = [j for j in range(s - g) if cur[j:j + g] == cur[s - g:]]
            if starts:
                hit = (g, max(starts))
                break
        if not hit:
            break
        g, j = hit
        first = first or g
        out += cur[j + g:s][:k - len(out)]
        if not extend:
            break
    return first, out


@pytest.mark.parametrize("c", CASES, ids=[c["what"] for c in CASES])
def test_cases(c):
    got = lookup_drafts(c["hist"], c["k"], c["nmax"], c["nmin"], c["extend"])
    assert got == (c["n"], c["drafts"])
    assert fast_lookup(c["hist"], len(c["hist"]), c["k"], c["nmax"], c["nmin"], c["extend"]) == got


def test_random_histories_match_the_rule():
    rng = random.Random(7)
    for trial in range(400):
        hist = [rng.randrange(1, 5 if trial % 2 else 12) for _ in range(rng.randrange(1, 60))]
        k, nmax = rng.randrange(1, 4), rng.randrange(1, 5)
        nmin, extend = rng.randrange(1, nmax + 1), bool(rng.randrange(2))
        want = by_definition(hist, k, nmax, nmin, extend)
        assert lookup_drafts(hist, k, nmax, nmin, extend) == want, (hist, k, nmax, nmin, extend)
        assert fast_lookup(hist, len(hist), k, nmax, nmin, extend) == want, (hist, k, nmax, nmin, extend)


def test_drafts_only_ever_come_from_the_history():
    rng = random.Random(11)
    for _ in range(200):
        hist = [rng.randrange(1, 6) for _ in range(rng.randrange(2, 40))]
        n, drafts = lookup_drafts(hist, 3)
        assert len(drafts) <= 3 and (n == 0) == (not drafts)
        assert set(drafts) <= set(hist)
