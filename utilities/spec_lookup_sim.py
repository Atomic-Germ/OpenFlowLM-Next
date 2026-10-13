"""Prompt-lookup speculative decoding: tokens per verify pass, replayed exactly from greedy transcripts.

  PRISM_LLAMA_SERVER=<llama-server> BONSAI_GGUF=<PQ2_0 gguf> python utilities/spec_lookup_sim.py capture all   (then: sim)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".claude" / "plans" / "spec_lookup_data"
LLAMA = Path(os.environ.get("PRISM_LLAMA_SERVER", "llama-server"))   # PrismML's fork: stock llama.cpp cannot read PQ2_0
GGUF = Path(os.environ.get("BONSAI_GGUF", "Ternary-Bonsai-2-27B-PQ2_0.gguf"))
PORT = 8097
N_PREDICT = 1024
CTX = 12288
# already their server's first request before capture moved to one server per prompt
FRESH_BEFORE_FIX = {"chat_1", "doc_1"}


def _read(rel: str, start: int | None = None, end: int | None = None) -> str:
    lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[(start - 1 if start else 0):end])


def _code(rel: str, ask: str) -> str:
    """File first, request last: request-first made the agentic fine-tune answer with a hallucinated tool call."""
    lang = "cpp" if rel.endswith((".cpp", ".hpp", ".h")) else "python"
    return f"Here is `{rel}`:\n\n```{lang}\n{_read(rel)}\n```\n\n{ask}"


def _doc(text: str, ask: str) -> str:
    return f"Here is a document.\n\n<document>\n{text}\n</document>\n\n{ask}"


def prompts() -> dict[str, tuple[str, bool]]:
    """id -> (user message, thinking on). Classes are the id prefixes."""
    full = "Reply with the complete updated file in a single code block."
    return {
        "chat_1": ("What's the difference between TCP and UDP, and when would I choose each?", False),
        "chat_2": ("How do I start a sourdough starter from scratch? Give me a day-by-day plan for the first week.", False),
        "chat_3": ("Explain how a heat pump keeps a house warm when it's below freezing outside.", False),
        "chat_4": ("I have five days in Kyoto and Osaka in April. Suggest an itinerary with a mix of temples, food "
                   "and one day trip.", False),
        "chat_5": ("What is a Python decorator? Show a few practical examples.", False),
        "code_1": (_code("open_kernels/designs/gemv_t2/probe.py",
                         "Add a `--csv PATH` option to this script. When given, append one row per build "
                         "(tag, min ms, median ms, check result) to that CSV file, writing a header if the file "
                         "is new. " + full), False),
        "code_2": (_code("open_kernels/t2_pack.py",
                         "Make two changes. `pack_t2_pool` should raise a ValueError with a clear message when k "
                         "is not a multiple of 256 or n is not a multiple of rs. `_selftest` should also check, "
                         "for three random shapes, that `t2_from_q4_1_pool` of the q4_1 pool equals "
                         "`pack_t2_pool` of the same weights. " + full), False),
        "code_3": (_code("open_kernels/recipes/manifest.py",
                         "Add an `--out PATH` argument to `main` that writes the manifest to a file instead of "
                         "stdout, and make `manifest` reject a `max_ctx` that is not a positive multiple of 256. "
                         "Show only the functions you changed."), False),
        "code_4": (_code("src/open_qwen36/q4nx_file.cpp",
                         "Every error thrown in this file should name the container path and, where there is "
                         "one, the tensor. Add a small helper that builds those messages and use it everywhere "
                         "this file throws. " + full), False),
        "code_5": (_code("open_kernels/recipes/cache.py",
                         "Rename `build_key` to `kernel_build_key`, update every use in this file, and give "
                         "each function a one-line docstring if it lacks one. " + full), False),
        "doc_1": (_doc(_read("docs/BUILD.md"),
                       "Summarize this document as a checklist for building on Windows, in at most 15 steps."), False),
        "doc_2": (_doc(_read("REPLACEMENT_PLAN.md"),
                       "Summarize this plan in five bullet points. Then list every open risk or unresolved "
                       "question the document mentions."), False),
        "doc_3": (_doc(_read("specs/server-api/spec.md"),
                       "List every requirement ID in this spec, each with a one-sentence summary of what it "
                       "requires and how it is verified."), False),
        "doc_4": (_doc(_read("docs/linux-getting-started.md"),
                       "Using only this guide, answer: 1) Which distributions are supported? 2) What driver and "
                       "firmware does the NPU need? 3) What should I do if `ulimit -l` is not unlimited? "
                       "4) How do I install on Arch Linux? 5) How do I check the NPU works after install?"), False),
        "doc_5": (_doc(_read("specs/open-engine/spec.md", 654, 828),
                       "Based on this spec, explain in plain words how ternary weights are stored and applied "
                       "on the NPU, for decode and for prefill. Then answer: why are the scales bf16 rather than "
                       "fp16, and how is each requirement verified?"), False),
        "reason_1": ("Find all pairs of positive integers (x, y) with x^2 - y^2 = 2025.", True),
        "reason_2": ("On an island, knights always tell the truth and knaves always lie. A says: \"B is a knave.\" "
                     "B says: \"A and C are the same type.\" C says: \"I am a knight.\" What can be determined "
                     "about A, B and C?", True),
        "reason_3": ("In how many ways can a 2 x 10 rectangle be tiled with 1 x 2 dominoes?", True),
        "reason_4": ("Pencils cost 3 for $1 and pens cost $2 each. Maya bought 25 items and spent exactly $15. "
                     "How many of each did she buy? Are there other solutions?", True),
        "reason_5": ("Write a Python function that returns the length of the longest substring without repeating "
                     "characters, and prove its time complexity.", True),
    }


# Without a system message the agentic fine-tune answered whole-file edits with a hallucinated Bash tool call.
CODE_SYSTEM = ("You are a coding assistant in a chat window. No tools are available: never emit a tool "
               "call. You cannot run commands or see any files except what the user pastes. Answer directly "
               "with code.")
CODE_PROMPT_VERSION = 2   # file first + request last + this system message


def messages(pid: str, user: str) -> list[dict]:
    m = [{"role": "user", "content": user}]
    return ([{"role": "system", "content": CODE_SYSTEM}] + m) if pid.startswith("code_") else m


def render(msgs: list[dict], thinking: bool) -> str:
    """The GGUF's own chat template, rendered the way HF/llama.cpp do (trim/lstrip blocks)."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg):
        raise RuntimeError(msg)

    tmpl = (DATA / "chat_template.jinja").read_text(encoding="utf-8")
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False, **kw)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(tmpl).render(messages=msgs,
                                        add_generation_prompt=True, enable_thinking=thinking)


def save_template() -> None:
    import gguf
    from gguf import gguf_reader
    gguf_reader.GGUFReader._build_tensors = lambda self, offs, fields: setattr(self, "tensors", [])  # gguf cannot parse tensor type 142; only metadata is needed
    r = gguf.GGUFReader(str(GGUF))
    (DATA / "chat_template.jinja").write_text(r.fields["tokenizer.chat_template"].contents(), encoding="utf-8")


def post(path: str, body: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def capture(ids: list[str], force: bool) -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    if not (DATA / "chat_template.jinja").exists():
        save_template()
    ps = prompts()
    todo = [i for i in ids if force or not (DATA / f"{i}.json").exists()]
    if not todo:
        print("nothing to do")
        return 0
    for pid in todo:
        # One server per prompt: even with cache_prompt off, the DeltaNet state leaked into the next request.
        log = open(DATA / f"server_{pid}_{int(time.time())}.log", "w", encoding="utf-8")
        srv = subprocess.Popen([str(LLAMA), "-m", str(GGUF), "-ngl", "99", "-c", str(CTX), "-np", "1",
                                "--host", "127.0.0.1", "--port", str(PORT), "--no-webui"],
                               stdout=log, stderr=subprocess.STDOUT)
        try:
            t0 = time.time()
            while True:
                if srv.poll() is not None:
                    raise RuntimeError(f"llama-server exited with {srv.returncode}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                        if r.status == 200:
                            break
                except Exception:
                    pass
                if time.time() - t0 > 600:
                    raise RuntimeError("llama-server did not come up")
                time.sleep(2)
            user, thinking = ps[pid]
            msgs = messages(pid, user)
            text = render(msgs, thinking)
            try:      # cross-check the local render against the server's own template engine
                srv_text = post("/apply-template", {"messages": msgs,
                                                    "chat_template_kwargs": {"enable_thinking": thinking}})["prompt"]
                same = srv_text == text
            except Exception as e:
                same = f"apply-template failed: {e}"
            prompt_ids = post("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]
            t1 = time.time()
            r = post("/completion", {"prompt": prompt_ids, "n_predict": N_PREDICT, "temperature": 0, "top_k": 1,
                                     "return_tokens": True, "cache_prompt": False, "stream": False})
            gen = r["tokens"]
            rec = {"id": pid, "class": pid.split("_")[0], "thinking": thinking, "template_matches_server": same,
                   "system": msgs[0]["content"] if msgs[0]["role"] == "system" else None,
                   "code_prompt_version": CODE_PROMPT_VERSION if pid.startswith("code_") else None,
                   "prompt_ids": prompt_ids, "gen_ids": gen, "gen_text": r["content"],
                   "stop_type": r.get("stop_type"), "timings": r.get("timings"),
                   "seconds": round(time.time() - t1, 1), "fresh_server": True}
            (DATA / f"{pid}.json").write_text(json.dumps(rec), encoding="utf-8")
            print(f"{pid}: server up in {t1 - t0:.0f} s, prompt {len(prompt_ids)} tok, generated {len(gen)} "
                  f"({r.get('stop_type')}), {rec['seconds']} s, template match {same}", flush=True)
        finally:
            srv.terminate()
            try:
                srv.wait(timeout=30)
            except subprocess.TimeoutExpired:
                srv.kill()
                srv.wait(timeout=30)
            log.close()
    return 0


def simulate(prompt: list[int], gen: list[int], k: int, nmin: int = 1, nmax: int = 4) -> dict:
    """Replay one greedy transcript. gen[0] comes from prefill; every later token from a verify pass."""
    ctx = list(prompt) + [gen[0]]
    idx = [dict() for _ in range(nmax + 1)]   # idx[n][ngram] = most recent start j with a token after it

    def add_upto(L_old: int, L_new: int) -> None:
        # positions p in [L_old, L_new) became the last token; the n-gram ending at p-1 now has a successor
        for p in range(max(L_old, 1), L_new):
            for n in range(1, nmax + 1):
                j = p - n
                if j >= 0:
                    idx[n][tuple(ctx[j:p])] = j

    add_upto(0, len(ctx))
    i, passes, drafted, positions, accepted_total = 1, 0, 0, 0, 0
    while i < len(gen):
        L = len(ctx)
        draft: list[int] = []
        for n in range(nmax, nmin - 1, -1):
            if L < n:
                continue
            j = idx[n].get(tuple(ctx[L - n:]))
            if j is not None:
                draft = ctx[j + n: j + n + k]
                break
        a = 0
        while a < len(draft) and i + a < len(gen) and draft[a] == gen[i + a]:
            a += 1
        step = min(a + 1, len(gen) - i)
        passes += 1
        drafted += bool(draft)
        positions += len(draft) + 1
        accepted_total += a
        ctx.extend(gen[i:i + step])
        add_upto(L, len(ctx))
        i += step
    return {"tokens": len(gen) - 1, "passes": passes, "drafted": drafted, "positions": positions}


def load() -> list[dict]:
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(DATA.glob("*.json"))
            if p.stem.split("_")[0] in ("chat", "code", "doc", "reason")]
    stale = [r["id"] for r in recs if (not r.get("fresh_server") and r["id"] not in FRESH_BEFORE_FIX)
             or (r["class"] == "code" and r.get("code_prompt_version") != CODE_PROMPT_VERSION)]
    if stale:
        print(f"ignoring stale transcripts (a server's second request, or code without the system "
              f"message): {stale}")
    return [r for r in recs if r["id"] not in stale]


def sim(kmax: int, as_json: Path | None) -> int:
    recs = load()
    if not recs:
        print(f"no transcripts in {DATA}")
        return 1
    classes = ["chat", "code", "doc", "reason"]
    out = {"per_prompt": {}, "per_class": {}}
    print(f"{len(recs)} transcripts in {DATA}\n")
    print("per prompt: prompt tokens, generated tokens, stop, tokens/pass at k=1..%d (n>=1)" % kmax)
    for r in recs:
        row = [simulate(r["prompt_ids"], r["gen_ids"], k) for k in range(1, kmax + 1)]
        out["per_prompt"][r["id"]] = row
        tpp = " ".join(f"{x['tokens'] / max(1, x['passes']):.2f}" for x in row)
        print(f"  {r['id']:9s} {len(r['prompt_ids']):5d} {len(r['gen_ids']):5d} {str(r.get('stop_type')):6s} {tpp}")
    for nmin in (1, 2):
        print(f"\ndrafting on a suffix match of length >= {nmin} (n = 4..{nmin}); pooled over each class's prompts")
        print(f"  class   prompts  drafted (k=3)  tokens/pass k=1..{kmax}   |  positions/pass k=1..{kmax}")
        for c in classes:
            rs = [r for r in recs if r["class"] == c]
            if not rs:
                continue
            tpp, ppp, dr = [], [], None
            for k in range(1, kmax + 1):
                s = [simulate(r["prompt_ids"], r["gen_ids"], k, nmin=nmin) for r in rs]
                T, P, D, Q = (sum(x[f] for x in s) for f in ("tokens", "passes", "drafted", "positions"))
                tpp.append(T / P)
                ppp.append(Q / P)
                if k == 3:
                    dr = D / P
                out["per_class"].setdefault(f"n>={nmin}", {}).setdefault(c, []).append(
                    {"k": k, "tokens": T, "passes": P, "drafted": D, "positions": Q})
            print(f"  {c:7s} {len(rs):5d}   {100 * dr:8.0f}%      " + " ".join(f"{v:.2f}" for v in tpp)
                  + "   |  " + " ".join(f"{v:.2f}" for v in ppp))
    if as_json:
        as_json.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("ids", nargs="+", help="prompt ids, or 'all'")
    c.add_argument("--force", action="store_true", help="re-run prompts that already have a transcript")
    s = sub.add_parser("sim")
    s.add_argument("--kmax", type=int, default=7)
    s.add_argument("--json", type=Path)
    p = sub.add_parser("show", help="print a rendered prompt")
    p.add_argument("id")
    a = ap.parse_args(argv)
    if a.cmd == "capture":
        return capture(list(prompts()) if a.ids == ["all"] else a.ids, a.force)
    if a.cmd == "show":
        DATA.mkdir(parents=True, exist_ok=True)
        if not (DATA / "chat_template.jinja").exists():
            save_template()
        user, thinking = prompts()[a.id]
        sys.stdout.buffer.write(render(messages(a.id, user), thinking).encode("utf-8"))
        return 0
    return sim(a.kmax, a.json)


if __name__ == "__main__":
    sys.exit(main())
