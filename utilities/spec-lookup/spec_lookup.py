r"""Prompt-lookup drafting for Granite's L-row verify pass (OPEN-SPEC-LOOKUP): the drafter, the prompts, and the
offline simulation that replays greedy transcripts through each cycle policy.

    python spec_lookup.py prompts --model <dir> --out <dir>             # <name>.ids for every prompt set
    python spec_lookup.py probe --cli <exe> --model <dir> --kernels <set> --prompts <dir> --out <dir> [--only a,b]
    python spec_lookup.py sim|ngrams --probes <dir> --prompts <dir> [--costs costs.json]   # replayed offline
    python spec_lookup.py bench --cli <exe> --model <dir> --kernels <set> --prompts <dir> --out <jsonl> --modes ...
    python spec_lookup.py matrix ... --units "what:tokens:modes;..." --gap 75; table --dir <matrix out>
    python spec_lookup.py serve-launch ... [--env OFLM_SPECULATIVE=1] --out a.json; serve-compare a.json b.json

`probe` runs `open_qwen36_cli --spec-probe`: greedy decode by draft passes, so one sweep gives the greedy transcript and,
at every position, the drafts a Uno cycle seeded there would verify. Under greedy decoding a draft never changes the
output, so replaying the transcript is exact for lookup and for Uno up to the draft rows' noise ids. Wrap `probe` in
`dibs run npu` and `bench` in `dibs run --measure`.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
L = 4


def lookup_drafts(hist: list[int], k: int, nmax: int = 4, nmin: int = 1, extend: bool = True) -> tuple[int, list[int]]:
    """The engine's drafter (src/open_qwen36/spec_lookup.hpp), line for line: (n of the first match or 0, drafts)."""
    out: list[int] = []
    n = len(hist)
    nmin = max(1, nmin)

    def at(i: int) -> int:
        return hist[i] if i < n else out[i - n]

    first = 0
    while len(out) < k:
        s = n + len(out)
        got, frm = 0, 0
        for g in range(nmax, nmin - 1, -1):
            if s < g + 1:
                continue
            for j in range(s - g - 1, -1, -1):
                i = 0
                while i < g and at(j + i) == at(s - g + i):
                    i += 1
                if i == g:
                    got, frm = g, j + g
                    break
            if got:
                break
        if not got:
            break
        first = first or got
        c = frm
        while c < s and len(out) < k:
            out.append(at(c))
            c += 1
        if not extend:
            break
    return first, out


def fast_lookup(seq, n: int, k: int, nmax: int = 4, nmin: int = 1, extend: bool = True) -> tuple[int, list[int]]:
    """lookup_drafts over seq[:n] with numpy (seq: an int array holding at least the history); same result."""
    import numpy as np
    hist = np.asarray(seq[:n], dtype=np.int64)
    out: list[int] = []
    first = 0
    nmin = max(1, nmin)
    while len(out) < k:
        cur = np.concatenate([hist, np.asarray(out, dtype=np.int64)]) if out else hist
        s = len(cur)
        got, frm = 0, 0
        for g in range(nmax, nmin - 1, -1):
            if s < g + 1:
                continue
            m = cur[: s - g] == cur[s - g]
            for i in range(1, g):
                m &= cur[i: s - g + i] == cur[s - g + i]
            idx = np.flatnonzero(m)
            if idx.size:
                got, frm = g, int(idx[-1]) + g
                break
        if not got:
            break
        first = first or got
        out.extend(int(x) for x in cur[frm: min(s, frm + k - len(out))])
        if not extend:
            break
    return first, out


# ---------------------------------------------------------------- prompts

LOOKUP_ASK = "Reply with the complete updated file in a single code block."


def _bug_file() -> str:
    src = (REPO / "utilities/cmp_greedy.py").read_text(encoding="utf-8")
    bad = src.replace("if ar != an and first_div is None:", "if ar == an and first_div is None:")
    assert bad != src, "the injected bug no longer applies to utilities/cmp_greedy.py"
    return bad


def _section(path: str, start: str, stop: str) -> str:
    text = (REPO / path).read_text(encoding="utf-8")
    i = text.index(start)
    return text[i: text.index(stop, i)].strip()


def lookup_prompts() -> dict[str, str]:
    """Where lookup should shine: the answer repeats the prompt (thinking off, as a coding assistant runs)."""
    code = lambda rel, body, ask: f"Here is `{rel}`:\n\n```python\n{body}\n```\n\n{ask}"
    doc = lambda body, ask: f"Here is a document.\n\n<document>\n{body}\n</document>\n\n{ask}"
    return {
        "edit_py": code("utilities/uno-bench/uno_bench.py", (REPO / "utilities/uno-bench/uno_bench.py").read_text(encoding="utf-8"),
                        "Add a `--reps N` option (default 1) that runs each prompt N times and reports the median Uno and "
                        "decode tok/s for each prompt. " + LOOKUP_ASK),
        "fix_py": code("utilities/cmp_greedy.py", _bug_file(),
                       "When I compare two identical runs, this script reports a divergence at the very first position. "
                       "Find the bug and fix it. " + LOOKUP_ASK),
        "quote_doc": doc(_section("specs/open-engine/spec.md", "### OPEN-UNO-DECODE", "**Result 2026-10-09 (K2"),
                         "Quote both **Procedure (manual)** paragraphs word for word, then explain in plain words what "
                         "each one checks."),
        "qa_doc": doc(_section(".opencode/skill/open-dxl-rows/SKILL.md", "## Traps", "## Build"),
                      "Using only this document, answer: 1) Why must every main-core buffer be a whole number of 64 B? "
                      "2) What goes wrong if the z wait comes after the tile's own drain? 3) How many south ports are "
                      "there in all, and why did Granite run out of them? 4) What compiles in seconds where constant "
                      "skip loops hung llc? Quote the document's own words in each answer."),
    }


def build_prompts(model: Path, out: Path) -> dict[str, dict]:
    from transformers import AutoTokenizer
    sys.path.insert(0, str(REPO / "utilities" / "granite-combined-bench"))
    from bench import chat_prompt, tokenizer  # noqa: E402
    hf = AutoTokenizer.from_pretrained(str(model))
    out.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict] = {}

    def save(name: str, ids: list[int], group: str, tokens: int) -> None:
        (out / f"{name}.ids").write_text(",".join(map(str, ids)))
        index[name] = {"group": group, "prompt_tokens": len(ids), "tokens": tokens}

    for line in (REPO / "utilities/uno-ref/prompts.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)
            text = hf.apply_chat_template(p["messages"], add_generation_prompt=True, tokenize=False)
            save(p["name"], hf.encode(text, add_special_tokens=False), "short", 256)
    tok = tokenizer(model)
    for n in (1024, 2048):
        for kind in ("chat", "code"):
            save(f"{kind}_{n}", chat_prompt(tok, model, kind, n), f"ctx{n}", 256)
    for name, msg in lookup_prompts().items():
        text = hf.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=False,
                                      enable_thinking=False)
        save(name, hf.encode(text, add_special_tokens=False), "lookup", 512)
    (out / "index.json").write_text(json.dumps(index, indent=1))
    return index


# ---------------------------------------------------------------- simulation

EOS = (100257,)     # Granite's <|im_end|>: a transcript is replayed to its reply's end, never past it


def load_probe(path: Path) -> dict:
    d = json.loads(path.read_text())
    d["name"] = path.stem
    P = d["prompt_len"]
    cut = next((i for i, t in enumerate(d["ids"][P:]) if t in EOS), None)
    if cut is not None:
        d["ids"] = d["ids"][:P + cut + 1]
    return d


def simulate(d: dict, policy: str, costs: dict, ntok: int, nmax: int = 4, nmin: int = 1, extend: bool = True,
             pol_args: dict | None = None, rows: int = L) -> dict:
    """Replay one probe transcript through a cycle policy. Costs: ms of a plain step, a verify pass and a draft pass.
    `rows` other than the kernel set's L = 4 is a what-if for lookup alone (Uno's probe drafts are L - 1 = 3)."""
    import numpy as np
    G = np.asarray(d["ids"], dtype=np.int64)
    P = d["prompt_len"]
    U = (d.get("uno") or []) if d.get("draft", True) else []
    end = min(len(G), P + ntok)
    ts, tv, td = costs["step"], costs["verify"], costs["draft"]
    pol = Policy(**(pol_args or {})) if policy.startswith("adaptive") else None
    p = P - 1                        # the pending token's position
    ms = 0.0
    cyc = {"plain": [0, 0], "lookup": [0, 0], "uno": [0, 0]}
    by_n: dict[int, list[int]] = {}

    def acc(drafts: list[int], at: int) -> int:     # drafts for positions at, at+1, ...
        k = 0
        while k < len(drafts) and at + k < end and drafts[k] == G[at + k]:
            k += 1
        return k

    base, _, thr = policy.partition(":")
    thr = int(thr or 1)
    while p + 1 < end:
        matched, drafts = fast_lookup(G, p + 1, rows - 1, nmax, nmin, extend)
        if drafts:
            drafts = drafts + [drafts[-1]] * (rows - 1 - len(drafts))
        has_uno = bool(U) and p - (P - 1) < len(U)
        if base == "plain":
            m = "plain"
        elif base == "uno":
            m = "uno" if has_uno else "plain"
        elif base == "lookup":
            m = "lookup" if matched >= thr else "plain"
        elif base == "lookup_uno":
            m = "lookup" if matched >= thr else ("uno" if has_uno else "plain")
        elif base == "adaptive":
            m = pol.choose(matched, has_uno, costs)
        elif base == "adaptive2":    # a what-if: Uno only where the lookup has no draft
            m = pol.choose(matched, has_uno and not matched, costs)
        elif base == "unofill":      # every cycle Uno, its verify rows from a lookup after c when that matches >= thr
            m = "uno" if has_uno else "plain"
        elif base == "oracle":       # the best of the three for this cycle, known in hindsight: a bound, not a policy
            opts = [("plain", 1 / ts)]
            if matched:
                opts.append(("lookup", (acc(drafts, p + 1) + 1) / tv))
            if has_uno:
                opts.append(("uno", (acc(list(U[p - (P - 1)]), p + 2) + 2) / (td + tv)))
            m = max(opts, key=lambda o: o[1])[0]
        else:
            raise ValueError(policy)
        if m == "plain":
            got, k, cost = 1, 0, ts
        elif m == "lookup":
            k = acc(drafts, p + 1)
            got, cost = k + 1, tv
            by_n.setdefault(matched, [0, 0])
            by_n[matched][0] += 1
            by_n[matched][1] += k
        else:
            ud = list(U[p - (P - 1)])
            if base == "unofill":
                mc, lc = fast_lookup(G, p + 2, rows - 1, nmax, nmin, extend)
                if mc >= thr:
                    ud = lc + [lc[-1]] * (rows - 1 - len(lc))
            k = acc(ud, p + 2)
            got, cost = k + 2, td + tv
        got = min(got, end - (p + 1))
        cyc[m][0] += 1
        cyc[m][1] += got
        ms += cost
        if pol:
            pol.record(m, matched, k)
        p += got
    tokens = sum(c[1] for c in cyc.values())
    cycles = sum(c[0] for c in cyc.values())
    return {"tokens": tokens, "cycles": cycles, "ms": ms, "tok_s": 1000 * tokens / ms, "cyc": cyc, "by_n": by_n,
            "plain_tok_s": 1000 / ts}


class Policy:
    """SpecPolicy (spec_lookup.hpp) with costs given rather than timed."""

    def __init__(self, alpha: float = 0.3, revert: float = 0.1, prior_lookup=(0, 0.3, 0.8, 1.0, 2.0), prior_uno=1.0):
        self.alpha, self.revert = alpha, revert
        self.prior_lookup, self.prior_uno = list(prior_lookup), prior_uno
        self.acc_lookup, self.acc_uno = list(prior_lookup), prior_uno

    def gain(self, m: str, matched: int, c: dict) -> float:
        if m == "plain":
            return 1 / c["step"]
        if m == "lookup":
            return (1 + self.acc_lookup[min(max(matched, 1), 4)]) / c["verify"]
        return (2 + self.acc_uno) / (c["draft"] + c["verify"])

    def choose(self, matched: int, has_uno: bool, c: dict) -> str:
        best, g = "plain", self.gain("plain", 0, c)
        if has_uno and self.gain("uno", 0, c) > g:
            best, g = "uno", self.gain("uno", 0, c)
        if matched and self.gain("lookup", matched, c) > g:
            best = "lookup"
        return best

    def record(self, m: str, matched: int, k: int) -> None:
        for g in range(1, 5):
            if not (m == "lookup" and g == matched):
                self.acc_lookup[g] += self.revert * (self.prior_lookup[g] - self.acc_lookup[g])
        if m != "uno":
            self.acc_uno += self.revert * (self.prior_uno - self.acc_uno)
        if m == "lookup":
            g = min(max(matched, 1), 4)
            self.acc_lookup[g] += self.alpha * (k - self.acc_lookup[g])
        if m == "uno":
            self.acc_uno += self.alpha * (k - self.acc_uno)


DEFAULT_COSTS = {   # ms; replaced by --costs from measured runs
    "short": {"step": 48.0, "verify": 64.2, "draft": 67.0},
    "ctx1024": {"step": 64.5, "verify": 80.0, "draft": 84.0},
    "ctx2048": {"step": 81.0, "verify": 96.0, "draft": 100.0},
    "lookup": {"step": 60.0, "verify": 76.0, "draft": 80.0},
}


def run_sim(a) -> int:
    probes = sorted(Path(a.probes).glob("*.json"))
    index = json.loads((Path(a.prompts) / "index.json").read_text()) if a.prompts else {}
    costs = json.loads(Path(a.costs).read_text()) if a.costs else DEFAULT_COSTS
    pols = a.policies.split(",")
    rows = []
    print(f"{'prompt':<10} {'group':<8} {'plain':>6}  " + "  ".join(f"{p:>17}" for p in pols))
    for f in probes:
        d = load_probe(f)
        grp = index.get(d["name"], {}).get("group", "short")
        c = costs[grp]
        ntok = a.tokens or index.get(d["name"], {}).get("tokens", 256)
        r = {"name": d["name"], "group": grp, "plain": 1000 / c["step"]}
        if a.rows != L:     # a what-if build: the verify pass priced at verify(L) scaled by the rows' slope
            c = {**c, "verify": c["step"] + (c["verify"] - c["step"]) * (a.rows - 1) / (L - 1)}
        for pol in pols:
            r[pol] = simulate(d, pol, c, ntok, nmax=a.nmax, nmin=a.nmin, rows=a.rows)
        rows.append(r)
        print(f"{d['name']:<10} {grp:<8} {r['plain']:6.2f}  " + "  ".join(
            f"{r[p]['tok_s']:6.2f} {r[p]['tok_s'] / r['plain']:4.2f}x {r[p]['tokens'] / r[p]['cycles']:4.2f}" for p in pols),
            flush=True)
    for grp in dict.fromkeys(r["group"] for r in rows):
        rs = [r for r in rows if r["group"] == grp]
        print(f"{'geomean':<10} {grp:<8} {'':>6}  " + "  ".join(
            f"{'':>6} {math.exp(sum(math.log(r[p]['tok_s'] / r['plain']) for r in rs) / len(rs)):4.2f}x {'':>4}" for p in pols))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    return 0


def run_ngram_table(a) -> int:
    """Tokens per lookup verify pass and share of cycles drafted, n-gram sizes x drafts k (L = k + 1 rows)."""
    probes = sorted(Path(a.probes).glob("*.json"))
    index = json.loads((Path(a.prompts) / "index.json").read_text()) if a.prompts else {}
    groups: dict[str, list[dict]] = {}
    for f in probes:
        d = load_probe(f)
        groups.setdefault(index.get(d["name"], {}).get("group", "short"), []).append(d)
    for grp, ds in groups.items():
        print(f"\n{grp}: " + ", ".join(d["name"] for d in ds))
        for extend in (True, False):
            for nmin in (1, 2, 3):
                for nmax in (1, 2, 3, 4):
                    if nmax < nmin:
                        continue
                    cells = []
                    for k in range(1, a.kmax + 1):
                        T = P_ = D = 0
                        for d in ds:
                            ntok = a.tokens or index.get(d["name"], {}).get("tokens", 256)
                            t, pa, dr = lookup_stats(d, ntok, k, nmax, nmin, extend)
                            T, P_, D = T + t, P_ + pa, D + dr
                        cells.append(f"k{k} {T / max(1, P_):.2f} ({100 * D / max(1, P_):.0f}%)")
                    print(f"  n {nmin}..{nmax} {'ext' if extend else '   '}  " + "  ".join(cells))
    return 0


def lookup_stats(d: dict, ntok: int, k: int, nmax: int, nmin: int, extend: bool) -> tuple[int, int, int]:
    """Lookup-only replay: tokens, passes (verify or plain), passes that had a draft."""
    import numpy as np
    G = np.asarray(d["ids"], dtype=np.int64)
    P = d["prompt_len"]
    end = min(len(G), P + ntok)
    p, T, passes, drafted = P - 1, 0, 0, 0
    while p + 1 < end:
        matched, dr = fast_lookup(G, p + 1, k, nmax, nmin, extend)
        a = 0
        if dr:
            dr = dr + [dr[-1]] * (k - len(dr))
            while a < k and p + 1 + a < end and dr[a] == G[p + 1 + a]:
                a += 1
            drafted += 1
        got = min(a + 1, end - (p + 1))
        T += got
        passes += 1
        p += got
    return T, passes, drafted


# ---------------------------------------------------------------- NPU runs

def run_probe(a) -> int:
    index = json.loads((Path(a.prompts) / "index.json").read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    for name, info in index.items():
        if only and name not in only and info["group"] not in only:
            continue
        cmd = [a.cli, "--model", a.model, "--kernels", a.kernels, "--ids-file", str(Path(a.prompts) / f"{name}.ids"),
               "--spec-probe", str(a.tokens or info["tokens"]), "--spec-dump", str(out / f"{name}.json"), "--quiet"]
        if a.seed:
            cmd.append("--spec-probe-seed")
        if info["prompt_tokens"] >= 64:
            cmd.append("--gemm-block")
        r = subprocess.run(cmd, capture_output=True, text=True)
        tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
        print(f"{name}: rc {r.returncode} {tail[0]}", flush=True)
    return 0


SPEC = re.compile(r"spec: (\d+) tokens in (\d+) cycles \(([\d.]+) a cycle\), mode (\S+): ([\d.]+) ms/token \(([\d.]+) tok/s\); "
                  r"lookup ([\d.]+) ms a cycle")
METHOD = re.compile(r"spec (plain|lookup|uno)\s*: (\d+) cycles, (\d+) tokens \(([\d.]+) a cycle\), ([\d.]+) ms a cycle")
PASSMS = re.compile(r"spec pass ms: step ([\d.]+) \((\d+)\), verify ([\d.]+) \((\d+)\), draft ([\d.]+) \((\d+)\)")
DECODE = re.compile(r"decode: (\d+) tokens, ([\d.]+) ms/token \(([\d.]+) tok/s\); spec is ([\d.]+)x")
VERDICT = re.compile(r"SPEC (IDENTICAL to decode|DIFFERS from decode at token (\d+))")


def parse_spec(text: str) -> dict:
    s, d, v, pm = SPEC.search(text), DECODE.search(text), VERDICT.search(text), PASSMS.search(text)
    if not (s and d and v and pm):
        raise ValueError("spec: the CLI's --spec report did not parse:\n" + text[-2000:])
    meth = {m[1]: {"cycles": int(m[2]), "tokens": int(m[3]), "ms_cycle": float(m[5])} for m in METHOD.finditer(text)}
    return {"tokens": int(s[1]), "cycles": int(s[2]), "per_cycle": float(s[3]), "mode": s[4], "spec_tok_s": float(s[6]),
            "lookup_ms": float(s[7]), "methods": meth, "step_ms": float(pm[1]), "verify_ms": float(pm[3]),
            "draft_ms": float(pm[5]), "decode_tok_s": float(d[3]), "speedup": float(d[4]), "identical": v[2] is None,
            "first_diff": None if v[2] is None else int(v[2])}


def run_bench(a) -> int:
    if (Path(a.out).parent / "STOP").exists():     # a matrix is being paused between units
        print("STOP file present: not running", flush=True)
        return 3
    index = json.loads((Path(a.prompts) / "index.json").read_text())
    only = set(a.only.split(",")) if a.only else None
    rows = []
    with open(a.out, "a", encoding="utf-8") as out:
        for rep in range(a.reps):
            for name, info in index.items():
                if only and name not in only and info["group"] not in only:
                    continue
                for mode in a.modes.split(","):
                    cmd = [a.cli, "--model", a.model, "--kernels", a.kernels, "--ids-file",
                           str(Path(a.prompts) / f"{name}.ids"), "--spec", str(a.tokens or info["tokens"]), "--spec-mode",
                           mode, "--quiet", *a.extra]
                    if info["prompt_tokens"] >= 64:
                        cmd.append("--gemm-block")
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    try:
                        row = {"name": name, "group": info["group"], "rep": rep, **parse_spec(r.stdout + r.stderr)}
                    except ValueError as e:
                        row = {"name": name, "group": info["group"], "rep": rep, "mode": mode, "error": str(e)[-1500:]}
                    rows.append(row)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    if "error" in row:
                        print(f"{name:<10} {mode:<10} ERROR", flush=True)
                    else:
                        print(f"{name:<10} {mode:<10} {row['per_cycle']:5.2f}/c spec {row['spec_tok_s']:6.2f} decode "
                              f"{row['decode_tok_s']:6.2f} {row['speedup']:5.2f}x {'identical' if row['identical'] else 'DIFFERS'}",
                              flush=True)
    ok = all(r.get("identical") for r in rows)
    print("ALL SPEC IDENTICAL" if ok else "NOT ALL IDENTICAL", flush=True)
    return 0 if ok else 1


def run_matrix(a) -> int:
    """Each unit `what:tokens:modes` (what: prompt names or groups, '+'-joined; modes '+'-joined, run back to back in one
    process) per rep as one `dibs run --measure`, rerun until dibs says CLEAN; NOISY logs kept aside. `--gap` seconds
    after every unit leave the machine to the other sessions."""
    import shutil
    import time
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    xrt = Path(r"C:\Windows\System32\AMD\xrt-smi.exe")
    if xrt.exists():
        r = subprocess.run([str(xrt), "examine", "-r", "aie-partitions"], capture_output=True, text=True)
        (out / f"aie-partitions-{int(time.time())}.txt").write_text(r.stdout + r.stderr)
    dibs = shutil.which("dibs.cmd") or shutil.which("dibs") or "dibs"
    units = [u.split(":") for u in a.units.split(";")]
    expect = json.loads(Path(a.expect).read_text()) if a.expect else {}
    for rep in range(a.rep_from, a.reps):
        for what, tokens, modes in units:
            name = f"{what.replace('+', '-')}_{modes.replace('+', '-')}_r{rep}"
            if (out / f"{name}.jsonl").exists():
                continue
            for k in range(a.tries):
                if (out / "STOP").exists():
                    print(f"{name}: paused by {out / 'STOP'}", flush=True)
                    return 0
                log = out / f"{name}.log"
                cmd = [dibs, "run", "--measure", "--tag", a.tag, "--label", f"spec {name}", "--eta", a.eta, "--",
                       sys.executable, str(Path(__file__).resolve()), "bench", "--cli", a.cli, "--model", a.model,
                       "--kernels", a.kernels, "--prompts", a.prompts, "--out", str(out / f"{name}.part"),
                       "--only", what.replace("+", ","), "--modes", modes.replace("+", ","), "--tokens", tokens]
                (out / f"{name}.part").unlink(missing_ok=True)
                with open(log, "w", encoding="utf-8") as f:
                    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
                text = log.read_text(encoding="utf-8", errors="replace")
                clean = "dibs: CLEAN after" in text and (out / f"{name}.part").exists()
                if clean and expect:     # dibs lets a sibling subagent's NPU run into a CLEAN unit: the plain decode tells
                    got = [json.loads(l) for l in (out / f"{name}.part").read_text().splitlines() if l.strip()]
                    hit = [r["name"] for r in got if r.get("decode_tok_s", 0) < 0.95 * expect.get(r.get("name"), 0)]
                    if hit:
                        clean = False
                        text += f"\nslow plain decode on {hit}: shared NPU"
                if clean:
                    (out / f"{name}.part").replace(out / f"{name}.jsonl")
                    print(f"{name}: CLEAN (try {k + 1})", flush=True)
                else:
                    log.replace(out / f"{name}.noisy{k + 1}.log")
                    print(f"{name}: NOISY (try {k + 1}) {text.strip().splitlines()[-1][:160] if text.strip() else ''}",
                          flush=True)
                time.sleep(a.gap)
                if clean:
                    break
    return 0


def run_table(a) -> int:
    """The CLEAN units of a matrix: per prompt, plain decode and each mode's median tok/s, tokens a cycle, speedup."""
    import statistics as st
    rows = [json.loads(l) for f in sorted(Path(a.dir).glob("*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "error" not in r]
    # a CLEAN unit can still share the NPU with a sibling's work dibs let in: the run's own pass times say so
    def med_of(n, key):
        v = [r[key] for r in rows if r["name"] == n and r.get(key, 0) > 0]
        return st.median(v) if v else 0.0
    names_all = {r["name"] for r in rows}
    ref = {n: {k: med_of(n, k) for k in ("decode_tok_s", "verify_ms", "step_ms", "draft_ms")} for n in names_all}
    def slow_run(r):
        m = ref[r["name"]]
        if r["decode_tok_s"] < 0.95 * m["decode_tok_s"]:
            return f"plain decode {r['decode_tok_s']:.2f} against {m['decode_tok_s']:.2f}"
        for k in ("verify_ms", "step_ms", "draft_ms"):
            if r.get(k, 0) > 1.05 * m[k] > 0:
                return f"{k} {r[k]:.1f} against {m[k]:.1f}"
        return ""
    slow = [(r, slow_run(r)) for r in rows]
    for r, why in slow:
        if why:
            print(f"set aside: {r['name']} {r['mode']} rep {r['rep']}: {why}")
    bad = [r for r, _ in slow if not r["identical"]]
    total = len(slow)
    rows = [r for r, why in slow if not why]
    order = ["plain", "uno", "lookup", "lookup-uno", "adaptive"]
    modes = sorted(dict.fromkeys(r["mode"] for r in rows), key=lambda m: order.index(m) if m in order else 99)
    groups = ["short", "ctx1024", "ctx2048", "lookup"]
    names = sorted(dict.fromkeys(r["name"] for r in rows),
                   key=lambda n: next((groups.index(r["group"]) for r in rows if r["name"] == n and r["group"] in groups), 99))
    med = lambda v: st.median(v) if v else float("nan")
    print(f"| prompt | plain tok/s | " + " | ".join(f"{m} tok/s (tokens a cycle) | {m} x" for m in modes) + " |")
    print("|---|--:|" + "--:|--:|" * len(modes))
    per_grp: dict[str, dict[str, list[float]]] = {}
    for n in names:
        rs = [r for r in rows if r["name"] == n]
        plain = med([r["decode_tok_s"] for r in rs])
        cells = []
        for m in modes:
            v = [r for r in rs if r["mode"] == m]
            if not v:
                cells += ["", ""]
                continue
            t, c = med([r["spec_tok_s"] for r in v]), med([r["per_cycle"] for r in v])
            sp = med([r["spec_tok_s"] / r["decode_tok_s"] for r in v])
            per_grp.setdefault(rs[0]["group"], {}).setdefault(m, []).append(sp)
            cells += [f"{t:.2f} ({c:.2f}), n={len(v)}", f"{sp:.2f}x"]
        print(f"| {n} | {plain:.2f} | " + " | ".join(cells) + " |")
    for g, d in per_grp.items():
        print(f"geomean {g}: " + ", ".join(f"{m} {math.exp(sum(map(math.log, v)) / len(v)):.2f}x" for m, v in d.items()))
    print(f"{total} runs ({len(rows)} kept), {len(bad)} not identical"
          + "".join(f"\n  DIFFERS: {r['name']} {r['mode']} rep {r['rep']}" for r in bad))
    return 1 if bad else 0


def _chat(url: str, body: dict) -> tuple[str, float, dict]:
    """One /v1/chat/completions request: the reply (reasoning + content), its wall seconds and the server's usage."""
    import time
    import urllib.request
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        if not body.get("stream"):
            j = json.loads(r.read())
            m = j["choices"][0]["message"]
            return (m.get("reasoning_content") or "") + "\x00" + (m.get("content") or ""), time.time() - t0, j.get("usage") or {}
        reasoning, content, usage = [], [], {}
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            j = json.loads(line[5:])
            usage = j.get("usage") or usage
            ch = j.get("choices") or [{}]
            d = ch[0].get("delta") or {}
            reasoning.append(d.get("reasoning_content") or "")
            content.append(d.get("content") or "")
        return "".join(reasoning) + "\x00" + "".join(content), time.time() - t0, usage


def run_serve(a) -> int:
    """The served checks against one running `oflm serve`: greedy, cut, streamed, and a sampled request."""
    url = f"http://127.0.0.1:{a.port}/v1/chat/completions"
    res = {}
    for name, msgs, think in (("chat", [{"role": "user", "content": "Explain what an NPU is in two sentences."}], True),
                              ("edit", [{"role": "user", "content": lookup_prompts()["fix_py"]}], False)):
        greedy = {"model": a.tag, "messages": msgs, "max_tokens": a.tokens, "top_k": 1, "repetition_penalty": 1.0,
                  "think": think}
        for kind, body in (("greedy", greedy), ("cut", {**greedy, "max_tokens": a.cut}),
                           ("streamed", {**greedy, "stream": True}), ("sampled", {"model": a.tag, "messages": msgs,
                                                                                   "max_tokens": 48})):
            text, sec, usage = _chat(url, body)
            res[f"{name}_{kind}"] = {"text": text, "seconds": sec, "usage": usage}
            print(f"{name}_{kind}: {sec:.1f} s, {len(text)} chars, {usage.get('completion_tokens')} tokens, decode "
                  f"{usage.get('decoding_speed_tps', float('nan')):.2f} tok/s", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


def run_serve_launch(a) -> int:
    """Start `oflm serve` with the given env, run `serve` against it, stop it: one call per server configuration."""
    import os
    import time
    import urllib.request
    env = dict(os.environ)
    env.update({"OFLM_CONFIG_PATH": a.registry, "OFLM_MODEL_PATH": a.models_root, "OMP_WAIT_POLICY": "PASSIVE",
                "OFLM_OPEN_KERNELS_DIR": a.kernels})
    env.pop("OFLM_SPECULATIVE", None)
    env.pop("OFLM_SPEC_MODE", None)
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    log = open(a.log, "w", encoding="utf-8")
    srv = subprocess.Popen([a.oflm, "serve", a.tag, "-p", str(a.port)], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while True:
            if srv.poll() is not None:
                raise RuntimeError(f"oflm serve exited with {srv.returncode}; see {a.log}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=5):
                    break
            except Exception:
                if time.time() - t0 > 300:
                    raise RuntimeError("oflm serve did not come up in 300 s")
                time.sleep(2)
        return run_serve(a)
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill()
        log.close()


def run_serve_compare(a) -> int:
    x, y = (json.loads(Path(p).read_text()) for p in (a.a, a.b))
    bad = 0
    for k in x:
        if k.endswith("_sampled"):
            ok = bool(x[k]["text"].strip("\x00")) and bool(y[k]["text"].strip("\x00"))
            print(f"{k}: {'both replied' if ok else 'EMPTY'}")
        else:
            ok = x[k]["text"] == y[k]["text"]
            print(f"{k}: {'SAME' if ok else 'DIFFERENT'} ({x[k]['seconds']:.1f} s against {y[k]['seconds']:.1f} s)")
        bad += not ok
    for name in ("chat", "edit"):
        s = x.get(f"{name}_streamed", {}).get("text")
        if s is not None:
            same = s == x[f"{name}_greedy"]["text"]
            print(f"{name}: streamed {'equals' if same else 'DIFFERS from'} non-streamed in {a.a}")
            bad += not same
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    for name in ("probe", "bench"):
        q = sub.add_parser(name)
        q.add_argument("--cli", required=True)
        q.add_argument("--model", required=True)
        q.add_argument("--kernels", required=True)
        q.add_argument("--prompts", required=True)
        q.add_argument("--out", required=True)
        q.add_argument("--only", default="", help="prompt names or groups (short, ctx1024, ctx2048, lookup)")
        if name == "probe":
            q.add_argument("--tokens", type=int, default=0, help="0: the prompt set's own count")
            q.add_argument("--seed", action="store_true", help="--spec-probe-seed: draft rows start from the lookup's guess")
        if name == "bench":
            q.add_argument("--modes", default="adaptive")
            q.add_argument("--reps", type=int, default=1)
            q.add_argument("--tokens", type=int, default=0, help="0: the prompt set's own count")
            q.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    s = sub.add_parser("sim")
    s.add_argument("--probes", required=True)
    s.add_argument("--prompts", default="")
    s.add_argument("--costs", default="")
    s.add_argument("--tokens", type=int, default=0)
    s.add_argument("--json", default="")
    s.add_argument("--policies", default="uno,lookup,lookup_uno,lookup_uno:2,lookup_uno:3,adaptive,oracle")
    s.add_argument("--nmax", type=int, default=4)
    s.add_argument("--nmin", type=int, default=1)
    s.add_argument("--rows", type=int, default=L, help="what-if verify rows (lookup policies only; the set has 4)")
    x = sub.add_parser("matrix")
    for f in ("--cli", "--model", "--kernels", "--prompts", "--out"):
        x.add_argument(f, required=True)
    x.add_argument("--units", default="short:128:lookup+adaptive;ctx1024:128:lookup+adaptive;ctx2048:128:lookup+adaptive;"
                                      "edit_py+fix_py:256:lookup+adaptive;quote_doc+qa_doc:256:lookup+adaptive",
                   help="what:tokens:modes;... (what and modes '+'-joined)")
    x.add_argument("--reps", type=int, default=3)
    x.add_argument("--rep-from", type=int, default=0)
    x.add_argument("--tries", type=int, default=6)
    x.add_argument("--gap", type=float, default=75, help="seconds to leave the machine after every unit")
    x.add_argument("--expect", default="", help="json {prompt: plain tok/s}: a unit 5%% slower is rerun like a NOISY one")
    x.add_argument("--tag", default="d3")
    x.add_argument("--eta", default="5m")
    sv = sub.add_parser("serve")
    sv.add_argument("--port", type=int, default=52625)
    sv.add_argument("--tag", default="granite:4.2-3b")
    sv.add_argument("--tokens", type=int, default=200)
    sv.add_argument("--cut", type=int, default=37)
    sv.add_argument("--out", required=True)
    sl = sub.add_parser("serve-launch", help="start oflm serve with --env KEY=VALUE ..., run `serve`, stop it")
    sl.add_argument("--oflm", required=True)
    sl.add_argument("--registry", required=True, help="a model_list.json naming the model (OFLM_CONFIG_PATH)")
    sl.add_argument("--models-root", required=True, help="OFLM_MODEL_PATH: <root>/<model_path>/<name>")
    sl.add_argument("--kernels", required=True)
    sl.add_argument("--log", required=True)
    sl.add_argument("--env", nargs="*", default=[])
    sl.add_argument("--port", type=int, default=52625)
    sl.add_argument("--tag", default="granite:4.2-3b")
    sl.add_argument("--tokens", type=int, default=200)
    sl.add_argument("--cut", type=int, default=37)
    sl.add_argument("--out", required=True)
    sc = sub.add_parser("serve-compare")
    sc.add_argument("a")
    sc.add_argument("b")
    tb = sub.add_parser("table")
    tb.add_argument("--dir", required=True, help="a matrix --out directory")
    t = sub.add_parser("ngrams")
    t.add_argument("--probes", required=True)
    t.add_argument("--prompts", default="")
    t.add_argument("--tokens", type=int, default=0)
    t.add_argument("--kmax", type=int, default=3, help="drafts a pass; L = k + 1 rows (the kernel set has L = 4)")
    a = ap.parse_args(argv)
    if a.cmd == "prompts":
        idx = build_prompts(a.model, a.out)
        for k, v in idx.items():
            print(f"{k:<10} {v['group']:<8} {v['prompt_tokens']:>5} prompt tokens, {v['tokens']} to generate")
        return 0
    if a.cmd == "probe":
        return run_probe(a)
    if a.cmd == "bench":
        return run_bench(a)
    if a.cmd == "ngrams":
        return run_ngram_table(a)
    if a.cmd == "matrix":
        return run_matrix(a)
    if a.cmd == "table":
        return run_table(a)
    if a.cmd == "serve":
        return run_serve(a)
    if a.cmd == "serve-launch":
        return run_serve_launch(a)
    if a.cmd == "serve-compare":
        return run_serve_compare(a)
    return run_sim(a)


if __name__ == "__main__":
    sys.exit(main())
