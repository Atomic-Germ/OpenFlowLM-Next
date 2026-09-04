#!/usr/bin/env python3
"""Test a RUNNING flm embeddings endpoint through the official OpenAI client.

    pip install openai
    python utilities/test_embeddings_endpoint.py --model bge-base:en-v1.5

This is NOT a unit test and it starts nothing. It talks to a server you already
have running, over the wire, through `openai.OpenAI` -- so it exercises the real
client's request shaping and response parsing, not a hand-rolled HTTP call. If
this passes, an application using the OpenAI SDK works against this server.

    flm serve <llm> --embed 1 --embeddingmodel bge-base:en-v1.5

FOUR PROPERTIES, and the interesting thing is that only one of them needs a
threshold at all.

0. THE SERVER HONOURS THE `model` FIELD.
   Checked first because everything else is meaningless without it. This server
   used to echo the requested tag back and otherwise ignore it, so a request
   for any model returned the LOADED one's vectors under the name that had been
   asked for -- a wrong answer that actively asserts it is right.

1. IDENTICAL TEXTS -> IDENTICAL VECTORS, exactly.
   Not "close": bit-equal. The encode is deterministic, so five copies of one
   sentence in one request must come back byte for byte the same. This is the
   cheapest possible check for a whole family of real bugs -- cross-lane
   aliasing (one lane reading another's rows), row/tier misindexing, a
   scratch buffer shared between rows. Upstream has hit two of those, and both
   produced *plausible* vectors that differed only between rows.

2. SIMILAR TEXTS -> SIMILAR VECTORS.
3. UNRELATED TEXTS -> UNRELATED VECTORS.

   These two are ONE test, not two, and it needs no magic number. "Relatively
   similar" and "totally different" are only meaningful against each other, so
   the criterion is SEPARATION:

       min(cosine within the similar group) > max(cosine across the unrelated group)

   A fixed threshold like "> 0.7" would encode this machine's model and this
   month's checkpoint into the test. Separation asks the question that actually
   matters -- can this endpoint tell related text from unrelated text? -- and it
   holds for any model worth serving. The margin is printed so a drift shows up
   as a shrinking number long before it becomes a failure.

Exit code 0 on pass, 1 on fail.
"""
from __future__ import annotations

import argparse
import math
import sys

try:
    from openai import OpenAI
except ImportError:
    sys.exit("this test needs the official client: pip install openai")


# Five ways of saying the same thing. Not paraphrases of one sentence pattern --
# the wording, length and structure all vary, so a model cannot pass by matching
# surface forms.
SIMILAR = [
    "A man is playing a guitar on stage.",
    "Someone plays guitar at a concert.",
    "A guitarist performs in front of an audience.",
    "He strummed his guitar during the live show.",
    "The musician was on stage with his guitar.",
]

# Five sentences with nothing in common with each other OR with the group above:
# different topics, different registers, no shared content words.
DIFFERENT = [
    "The central bank raised interest rates on Tuesday.",
    "Add the flour slowly while whisking the batter.",
    "Antarctic sea ice reached a record minimum this year.",
    "The compiler rejected the template instantiation.",
    "She won the 400 metre hurdles in Oslo.",
]

IDENTICAL_N = 5


def unit_dot(a: list[float], b: list[float]) -> float:
    """Cosine, computed in double and NOT assuming the vectors are normalised.

    The endpoint does return unit vectors, and this checks that separately --
    but a similarity metric that silently depends on an unverified property is
    how you get a test that passes for the wrong reason.
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return dot / (na * nb)


def pairs(vs: list[list[float]]) -> list[tuple[int, int, float]]:
    return [(i, j, unit_dot(vs[i], vs[j]))
            for i in range(len(vs)) for j in range(i + 1, len(vs))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="the embedding model tag the server was started with")
    ap.add_argument("--base-url", default="http://127.0.0.1:52625/v1")
    ap.add_argument("--api-key", default="not-needed",
                    help="the OpenAI client requires one; flm ignores it")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key,
                    timeout=args.timeout)

    def embed(texts: list[str]) -> list[list[float]]:
        r = client.embeddings.create(model=args.model, input=texts)
        # The SDK sorts nothing for you: `index` is the contract that says which
        # vector belongs to which input, and a server that gets it wrong would
        # otherwise pass every similarity check below while returning shuffled
        # answers.
        if [d.index for d in r.data] != list(range(len(texts))):
            raise AssertionError(
                f"data[].index is {[d.index for d in r.data]}, expected "
                f"0..{len(texts) - 1} -- the response does not say which vector "
                f"belongs to which input")
        return [d.embedding for d in r.data]

    print(f"endpoint : {args.base_url}")
    print(f"model    : {args.model}")

    fails: list[str] = []

    # ---------------------------------------------------------------- shape
    try:
        probe = client.embeddings.create(model=args.model, input=["probe"])
    except Exception as e:                                  # noqa: BLE001
        print(f"\nFAIL -- the endpoint did not answer: {e}")
        return 1
    dims = len(probe.data[0].embedding)
    print(f"dims     : {dims}")
    if probe.object != "list":
        fails.append(f"response.object is {probe.object!r}, expected 'list'")
    if probe.data[0].object != "embedding":
        fails.append(f"data[0].object is {probe.data[0].object!r}")

    # ------------------------------- 0. does the server honour `model` at all?
    #
    # First in the run, because everything below is meaningless without it. The
    # tag used to be echoed back and otherwise unused, so a request for ANY
    # model returned the loaded one's vectors LABELLED WITH THE TAG THAT WAS
    # ASKED FOR -- a client comparing response.model to its request saw
    # agreement. Measured on a server holding bge-base: asking for
    # gte-multilingual came back as gte-multilingual, byte-identical to
    # bge-base.
    bogus = "definitely-not-a-model:v0"
    try:
        r0 = client.embeddings.create(model=bogus, input=["probe"])
    except Exception:                                       # noqa: BLE001
        print(f"\n[0] a model this server does not serve -> refused, correctly")
    else:
        if r0.model == bogus:
            print(f"\n[0] a model this server does not serve -> ANSWERED, and "
                  f"labelled {r0.model!r}")
            fails.append(
                f"the server answered a request for {bogus!r} with vectors "
                f"labelled {bogus!r}. It serves whatever model IS loaded under "
                f"the name that was asked for, so a client cannot tell which "
                f"model produced its embeddings")
        else:
            print(f"\n[0] a model this server does not serve -> answered, but "
                  f"labelled {r0.model!r} (what is loaded), which is at least "
                  f"honest")

    # ---------------------------------------- 1. identical -> identical
    ident = embed([SIMILAR[0]] * IDENTICAL_N)
    same = all(ident[k] == ident[0] for k in range(1, IDENTICAL_N))
    if same:
        print(f"\n[1] {IDENTICAL_N} identical texts  -> all {IDENTICAL_N} "
              f"vectors EXACTLY equal")
    else:
        worst = max(
            max(abs(a - b) for a, b in zip(ident[k], ident[0]))
            for k in range(1, IDENTICAL_N))
        print(f"[1] {IDENTICAL_N} identical texts  -> NOT equal, "
              f"max component difference {worst:.3e}")
        fails.append(
            f"identical inputs produced different vectors (max diff {worst:.3e}). "
            f"The encode is deterministic, so this is a per-row bug -- lane "
            f"aliasing, tier misindexing, or shared scratch between rows -- not "
            f"a precision issue")

    # -------------------------------------------- norms, before using cosines
    bad_norms = [(i, sum(x * x for x in v))
                 for i, v in enumerate(ident) if abs(sum(x * x for x in v) - 1.0) > 1e-2]
    if bad_norms:
        fails.append(f"vectors are not unit length: {bad_norms[:3]}")

    # --------------------------------- 2 and 3. similar vs unrelated
    sim = embed(SIMILAR)
    dif = embed(DIFFERENT)

    sp = pairs(sim)
    dp = pairs(dif)
    sim_min = min(c for _, _, c in sp)
    sim_max = max(c for _, _, c in sp)
    dif_min = min(c for _, _, c in dp)
    dif_max = max(c for _, _, c in dp)
    # The two groups are unrelated to each other as well, so these pairs belong
    # with the "different" side. Including them makes the test harder, and it is
    # the case a real retrieval system actually runs into.
    cross = [unit_dot(a, b) for a in sim for b in dif]
    cross_max = max(cross)

    print(f"\n[2] 5 similar texts     -> cosine {sim_min:.4f} .. {sim_max:.4f}"
          f"   ({len(sp)} pairs)")
    print(f"[3] 5 unrelated texts   -> cosine {dif_min:.4f} .. {dif_max:.4f}"
          f"   ({len(dp)} pairs)")
    print(f"    similar vs unrelated-> cosine up to {cross_max:.4f}"
          f"   ({len(cross)} pairs)")

    ceiling = max(dif_max, cross_max)
    margin = sim_min - ceiling
    print(f"\n    separation: worst similar pair {sim_min:.4f} vs best "
          f"unrelated pair {ceiling:.4f}  ->  margin {margin:+.4f}")
    if margin > 0:
        print("    every similar pair scores above every unrelated pair")
    else:
        fails.append(
            f"the groups do not separate: the worst similar pair ({sim_min:.4f}) "
            f"scores no higher than the best unrelated one ({ceiling:.4f}). The "
            f"endpoint returns vectors, but they do not carry meaning")

    # ---------------------------------------------------------------- verdict
    print()
    if fails:
        for f in fails:
            print(f"FAIL: {f}")
        return 1
    print("PASS -- identical inputs give identical vectors, and similar text "
          "separates from unrelated text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
