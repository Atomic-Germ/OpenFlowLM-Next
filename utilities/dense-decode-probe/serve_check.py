r"""`oflm serve` on Granite through the app, on a chosen prefill route: chat, long prompt, multi-turn, streamed (OPEN-PREFILL-GLUE-PREP).

    python serve_check.py --oflm <oflm.exe> --registry <model_list.json> --models-root <root> --kernels <set> \
        --out <json> [--prefill-mode bfp16] [--env OFLM_OPEN_PREFILL_PREP=0] [--long-tokens 1500]

Greedy (top_k 1) requests, so two runs of different routes can be diffed with `--compare a.json b.json`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("spec_lookup", REPO / "utilities" / "spec-lookup" / "spec_lookup.py")
sl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sl)


def scenarios(a, url):
    def ask(msgs, **kw):
        body = {"model": a.tag, "messages": msgs, "max_tokens": a.tokens, "top_k": 1, "repetition_penalty": 1.0, "think": False, **kw}
        return sl._chat(url, body)

    res = {}
    t, s, u = ask([{"role": "user", "content": "Explain what an NPU is in two sentences."}])
    res["chat"] = {"text": t, "seconds": s, "usage": u}
    t2, s2, u2 = ask([{"role": "user", "content": "Explain what an NPU is in two sentences."}], stream=True)
    res["chat_streamed"] = {"text": t2, "seconds": s2, "usage": u2}
    doc = (REPO / "specs" / "open-engine" / "spec.md").read_text(encoding="utf-8", errors="ignore")
    cut = doc[: a.long_tokens * 4]
    t, s, u = ask([{"role": "user", "content": cut + "\n\nIn one sentence, what is the text above about?"}])
    res["long"] = {"text": t, "seconds": s, "usage": u}
    msgs = []
    for q in ("My name is Ada and I like chess. Remember that.", "What is my name?", "What game do I like?"):
        msgs.append({"role": "user", "content": q})
        t, s, u = ask(msgs, max_tokens=60)
        reply = t.split("\x00")[-1]
        msgs.append({"role": "assistant", "content": reply})
        res[f"turn{len(msgs) // 2}"] = {"text": t, "seconds": s, "usage": u}
    return res


def launch(a) -> int:
    env = dict(os.environ)
    env.update({"OFLM_CONFIG_PATH": a.registry, "OFLM_MODEL_PATH": a.models_root, "OMP_WAIT_POLICY": "PASSIVE",
                "OFLM_OPEN_KERNELS_DIR": a.kernels})
    for k in ("OFLM_SPECULATIVE", "OFLM_SPEC_MODE", "OFLM_OPEN_PREFILL_MODE", "OFLM_OPEN_PREFILL_PREP"):
        env.pop(k, None)
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    cmd = [a.oflm, "serve", a.tag, "-p", str(a.port)] + (["--prefill-mode", a.prefill_mode] if a.prefill_mode else [])
    log = open(a.log, "w", encoding="utf-8")
    srv = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
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
        res = scenarios(a, f"http://127.0.0.1:{a.port}/v1/chat/completions")
        if a.oflm_test:
            import tempfile
            e = dict(env, PYTHONPATH=str(REPO / "utilities" / "oflm-test"))
            with tempfile.TemporaryDirectory() as td:
                r = subprocess.run([sys.executable, "-c", "import sys, oflm_test; sys.argv[0] = 'oflm-test'; oflm_test.main()", "--llm", "--port", str(a.port), "--model", a.tag,
                                    "--gen-lim", "64"], cwd=td, env=e, capture_output=True, text=True)
            print(r.stdout[-2500:], r.stderr[-800:])
            print("oflm-test --llm exit code", r.returncode)
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill()
        log.close()
    for k, v in res.items():
        u = v["usage"]
        print(f"{k}: {v['seconds']:.1f} s, prompt {u.get('prompt_tokens')} tok, {u.get('completion_tokens')} out, usage {json.dumps(u)}")
        print("   ", v["text"].replace("\x00", " | ")[:200].replace("\n", " "))
    Path(a.out).write_text(json.dumps(res, indent=1))
    ok = all(v["text"].strip("\x00").strip() for v in res.values())
    print("streamed equals non-streamed:", res["chat"]["text"] == res["chat_streamed"]["text"])
    print("route line:", [l.strip() for l in Path(a.log).read_text(errors="ignore").splitlines() if "block prefill route" in l or "gemm-block prep" in l][:2])
    return 0 if ok else 1


def compare(a, b) -> int:
    x, y = (json.loads(Path(p).read_text()) for p in (a, b))
    for k in x:
        print(f"{k}: {'SAME' if x[k]['text'] == y[k]['text'] else 'DIFFERENT'}")
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--oflm")
    ap.add_argument("--registry")
    ap.add_argument("--models-root")
    ap.add_argument("--kernels")
    ap.add_argument("--out")
    ap.add_argument("--log", default="serve.log")
    ap.add_argument("--prefill-mode", default="")
    ap.add_argument("--env", nargs="*", default=[])
    ap.add_argument("--port", type=int, default=52631)
    ap.add_argument("--tag", default="granite:4.2-3b")
    ap.add_argument("--oflm-test", action="store_true", help="run utilities/oflm-test --llm against the server too")
    ap.add_argument("--tokens", type=int, default=80)
    ap.add_argument("--long-tokens", type=int, default=1500)
    a = ap.parse_args()
    return compare(*a.compare) if a.compare else launch(a)


if __name__ == "__main__":
    sys.exit(main())
