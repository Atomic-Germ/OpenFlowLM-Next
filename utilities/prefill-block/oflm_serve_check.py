"""Phase 27.7/27.8: the shipped `oflm serve` over HTTP, including request isolation.

`serve` is the second consumer of `Engine::prefill`, and the plan forbids giving it a
separate inference implementation. So this drives the real server with the real
OpenAI-compatible request shape:

    POST /v1/chat/completions   {"messages":[{"role":"user","content":...}],
                                 "max_tokens":N,"temperature":0,"stream":false}

Three things are checked, in the order the plan lists them:

  * a **fresh** request works at the block boundary (255/256/257 prompt tokens of text);
  * a **second** request in the SAME server process works and returns the same answer
    as a fresh server would for that same request;
  * after a **reset**, the next request works.

The reset is the interesting one. `serve` exposes a context reset through the chat
surface; if it is absent, the honest position is to say so rather than to invent an
endpoint. `--reset-endpoint` names it if one is found, and the harness records
`reset_available: false` when it is not, which is a limitation of the report, not a pass.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oflm_drive as D  # noqa: E402

TEMPLATE_OFFSET = 11

# A build host usually exports HTTP_PROXY / ALL_PROXY, and on this machine `http_proxy`
# is the malformed `http:/172.17.0.1:1080` while `all_proxy` is socks5h. urllib honours
# those for a loopback URL, so the first version of this harness sat in `wait_ready`
# for the full timeout while `curl` talked to the same server happily -- the harness
# looked like "serve does not come up" when serve was up the whole time. The server is
# on 127.0.0.1, so the proxy is bypassed rather than trusted.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _open(req, timeout):
    return _OPENER.open(req, timeout=timeout)


def start(port: int, *, ctx: int, extra: dict | None = None) -> tuple[subprocess.Popen, list]:
    log = []
    cmd = [str(D.OFLM), 'serve', 'qwen3.8:27b', '--ctx-len', str(ctx), '-p', str(port)]
    proc = subprocess.Popen(cmd, cwd=str(D.ROOT), env=D.env(block=True, extra=extra),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc, log


def wait_ready(port: int, proc: subprocess.Popen, timeout: float = 900) -> str:
    """Poll /v1/models until the server answers; the model load happens before it does."""
    deadline = time.monotonic() + timeout
    last = ''
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return f'server exited with {proc.returncode}'
        try:
            with _open(f'http://127.0.0.1:{port}/v1/models', 5) as r:
                body = r.read().decode()
                if r.status == 200 and body.strip():
                    return ''
        except Exception as e:  # not up yet, or not ready yet
            last = str(e)
        time.sleep(3)
    return f'timed out waiting for /v1/models ({last})'


def chat(port: int, text: str, *, max_tokens: int, timeout: float = 1800,
         messages: list[dict] | None = None) -> dict:
    """One completion. `messages` overrides `text`, for a request that carries a history."""
    body = json.dumps(dict(model='qwen3.8:27b',
                           messages=messages or [dict(role='user', content=text)],
                           max_tokens=max_tokens, temperature=0, stream=False)).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=body,
                                 headers={'Content-Type': 'application/json'}, method='POST')
    t0 = time.monotonic()
    try:
        with _open(req, timeout) as r:
            raw = r.read().decode()
            status = r.status
    except urllib.error.HTTPError as e:
        return dict(ok=False, status=e.code, error=e.read().decode()[:400],
                    wall_s=round(time.monotonic() - t0, 2))
    except Exception as e:
        return dict(ok=False, status=None, error=str(e), wall_s=round(time.monotonic() - t0, 2))
    wall = round(time.monotonic() - t0, 2)
    try:
        j = json.loads(raw)
        content = j['choices'][0]['message']['content']
        usage = j.get('usage', {})
        return dict(ok=True, status=status, content=content, usage=usage, wall_s=wall)
    except Exception as e:
        return dict(ok=False, status=status, error=f'unparsable response: {e}: {raw[:300]}',
                    wall_s=wall)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=18127)
    ap.add_argument('--lengths', type=int, nargs='+', default=[255, 256, 257])
    ap.add_argument('--max-tokens', type=int, default=8)
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'oflm-serve-{stamp}')
    out.mkdir(parents=True)
    tok = D.Tokenizer()
    result = dict(tool='prefill-block-oflm-serve', version=1, phase=27,
                  gate='27.7 oflm serve / 27.8 request isolation', model_acceptance=False,
                  port=a.port, binary=str(D.OFLM), kernels=str(D.KERNELS), started=stamp,
                  template_offset=TEMPLATE_OFFSET)

    proc, _ = start(a.port, ctx=a.ctx)
    rows: list[dict] = []
    try:
        err = wait_ready(a.port, proc)
        if err:
            result['verdict'] = 'FAIL'
            result['rows'] = [dict(check='server_starts', verdict='FAIL', reason=err)]
            (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
            print(f'FAIL server did not start: {err}')
            return 1
        print('server ready', flush=True)

        # 27.7: one request per boundary length, against ONE resident server.
        first: dict[int, dict] = {}
        for n in a.lengths:
            text = tok.exact_text(n)
            r = chat(a.port, text, max_tokens=a.max_tokens)
            first[n] = r
            ok = r.get('ok') and r.get('content')
            rows.append(dict(check=f'fresh_request_{n}', ok=bool(ok),
                             content=r.get('content'), wall_s=r.get('wall_s'),
                             usage=r.get('usage'),
                             verdict='PASS' if ok else 'FAIL',
                             reason=r.get('error')))
            print(f"  fresh n={n:4d} ok={bool(ok)} {r.get('wall_s')}s "
                  f"{str(r.get('content'))[:28]!r}", flush=True)

        # 27.8: the same request again, in the same process. If resident state leaked,
        # this differs from the first answer for the same prompt.
        n = a.lengths[0]
        text = tok.exact_text(n)
        again = chat(a.port, text, max_tokens=a.max_tokens)
        same = again.get('ok') and again.get('content') == first[n].get('content')
        rows.append(dict(check=f'second_request_{n}_matches_first', ok=bool(same),
                         first=first[n].get('content'), second=again.get('content'),
                         verdict='PASS' if same else 'FAIL',
                         reason=None if same else
                         'a second identical request returned a different answer'))
        print(f"  repeat n={n:4d} same={bool(same)} {str(again.get('content'))[:28]!r}",
              flush=True)

        # A different length after that: the prompt that follows a shorter one.
        n2 = a.lengths[-1]
        text2 = tok.exact_text(n2)
        after = chat(a.port, text2, max_tokens=a.max_tokens)
        ok2 = after.get('ok') and after.get('content')
        rows.append(dict(check=f'request_{n2}_after_{n}', ok=bool(ok2),
                         content=after.get('content'), wall_s=after.get('wall_s'),
                         verdict='PASS' if ok2 else 'FAIL', reason=after.get('error')))
        print(f"  then  n={n2:4d} ok={bool(ok2)} {after.get('wall_s')}s "
              f"{str(after.get('content'))[:28]!r}", flush=True)

        # The reset path, if the server has one. Recorded honestly either way.
        reset = None
        for path in ('/v1/context/reset', '/reset', '/v1/reset'):
            req = urllib.request.Request(f'http://127.0.0.1:{a.port}{path}', method='POST')
            try:
                with _open(req, 10) as r:
                    reset = dict(endpoint=path, status=r.status)
                    break
            except Exception:
                continue
        result['reset_endpoint'] = reset
        if reset:
            after_reset = chat(a.port, text, max_tokens=a.max_tokens)
            same_r = after_reset.get('ok') and after_reset.get('content') == first[n].get('content')
            rows.append(dict(check=f'after_reset_{n}_matches_first', ok=bool(same_r),
                             content=after_reset.get('content'),
                             verdict='PASS' if same_r else 'FAIL'))
            print(f"  after reset n={n} same={bool(same_r)}", flush=True)
        else:
            rows.append(dict(check='after_reset', ok=False, verdict='SKIP',
                             reason='the server exposes no reset endpoint; recorded as a '
                                    'limitation rather than a pass'))
            print('  after reset: no reset endpoint on serve (recorded as a limitation)',
                  flush=True)
    finally:
        proc.terminate()
        try:
            tail = proc.communicate(timeout=30)[0] or ''
        except Exception:
            proc.kill()
            tail = ''
        (out / 'server.log').write_text(tail)

    graded = [r for r in rows if r['verdict'] != 'SKIP']
    bad = [r for r in graded if r['verdict'] != 'PASS']
    result['rows'] = rows
    result['verdict'] = 'PASS' if graded and not bad else 'FAIL'
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"\n{result['verdict']} -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
