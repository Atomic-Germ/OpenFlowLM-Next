"""Matched OpenMP startup-policy validation, using unchanged model/kernel artifacts.

Benchmark: one warm-up pair, then at least five measured pairs in alternating
order. Each process runs the full model; engine prefill/decode timings exclude
model loading. /usr/bin/time records peak RSS. No profiler is enabled.
Use --check production for a multi-turn shipped CLI comparison.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_profile as P
import oflm_drive as D
import oracle as O


def clean_env():
    env = clean_env_from(os.environ)
    env.update(OFLM_OPEN_PREFILL_PRECISION='paired', OFLM_OPEN_LAYER_MAJOR='0')
    return env


def summary(rows):
    measured = [r for r in rows if not r['warmup']]
    by = {label: sorted((r for r in measured if r['arm'] == label), key=lambda r: r['rep'])
          for label in ('before', 'after')}
    if len(by['before']) < 5 or len(by['before']) != len(by['after']):
        raise ValueError('five complete measured pairs required')
    expected = list(range(len(by['before'])))
    if any([r['rep'] for r in arm] != expected for arm in by.values()):
        raise ValueError('duplicate or missing paired repetition')
    if any(a['ids'] != b['ids'] for a, b in zip(by['before'], by['after'])):
        raise ValueError('generated trajectories differ')
    result = {}
    for metric in ('prefill_ms', 'decode_ms', 'rss_kib'):
        a, b = ([r[metric] for r in by[label]] for label in ('before', 'after'))
        if any(not isinstance(v, (int, float)) or not 0 < v < float('inf') for v in a + b):
            raise ValueError('positive finite measurements required')
        ma, mb = statistics.median(a), statistics.median(b)
        result[metric] = dict(before=a, after=b, before_median=ma, after_median=mb,
                              change_pct=100 * (mb / ma - 1),
                              before_range=[min(a), max(a)], after_range=[min(b), max(b)])
    # With five wins, a one-sided paired sign test has p=1/32 under no improvement.
    wins = sum(b['prefill_ms'] < a['prefill_ms'] for a, b in zip(by['before'], by['after']))
    result['prefill_wins'] = wins
    result['all_pairs_faster'] = wins == len(by['before'])
    result['no_decode_regression'] = result['decode_ms']['change_pct'] <= 3
    result['no_peak_memory_regression'] = result['rss_kib']['change_pct'] <= 3
    result['pass'] = all(result[k] for k in ('all_pairs_faster', 'no_decode_regression', 'no_peak_memory_regression'))
    return result


def platform():
    p = subprocess.run(['xrt-smi', 'examine', '--report', 'platform'], capture_output=True, text=True, check=True)
    m = re.search(r'Power Mode\s*:\s*(\S+)', p.stdout)
    if not m:
        raise ValueError('NPU power mode unavailable')
    return dict(mode=m[1], output=p.stdout)


def benchmark(a, out):
    ids = O.fixture(a.tokens, seed=26026)
    ids_file = out / 'ids.txt'
    ids_file.write_text(','.join(map(str, ids)))
    result = dict(check='omp-policy-benchmark', tokens=a.tokens, generated=a.generate,
                  prefill_definition='standalone engine prompt ingestion, excludes load',
                  power=platform(), rows=[], summaries=None,
                  provenance=P.provenance([a.before, a.after, a.model / 'model.q4nx',
                                          a.kernels / 'manifest.json', *a.kernels.glob('*/final.xclbin'),
                                          *a.kernels.glob('*/insts.bin')]))
    # One lock covers the complete paired campaign, including conditions sampling.
    import fcntl
    with O.LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for rep in range(-1, a.reps):
            order = ('before', 'after') if rep % 2 == 0 else ('after', 'before')
            for arm in order:
                tag = f'{rep}-{arm}'
                cmd = [str(getattr(a, arm)), '--model', str(a.model), '--kernels', str(a.kernels),
                       '--ids-file', str(ids_file), '--max-ctx', '4096', '--layers', '64',
                       '--max-tokens', str(a.generate), '--gemm-block', '--block-major', '--pmode', 'none', '--quiet']
                before = P.conditions()
                r = subprocess.run(['/usr/bin/time', '-f', '%M', '-o', str(out / f'{tag}.rss'), *cmd],
                                   env=clean_env(), capture_output=True, text=True, timeout=1800)
                log = r.stdout + '\n' + r.stderr
                (out / f'{tag}.log').write_text(log)
                if r.returncode:
                    raise RuntimeError(f'{tag}: exit {r.returncode}; see retained log')
                prefill = re.search(r'^prefill \d+ tokens: ([\d.]+) ms', log, re.M)
                decode = re.search(r'^decode \d+ tokens: ([\d.]+) ms/token', log, re.M)
                tokens = [int(x) for x in re.findall(r'^token (\d+)$', log, re.M)]
                if not prefill or not decode or len(tokens) != a.generate:
                    raise ValueError(f'{tag}: incomplete timing or tokens')
                power = platform()
                if power['mode'] != result['power']['mode']:
                    raise ValueError('NPU power mode changed during comparison')
                row = dict(rep=rep, warmup=rep < 0, arm=arm, command=cmd,
                           prefill_ms=float(prefill[1]), decode_ms=float(decode[1]), ids=tokens,
                           rss_kib=int((out / f'{tag}.rss').read_text()),
                           conditions_before=before, conditions_after=P.conditions(), power_after=power)
                result['rows'].append(row)
                (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
                print(f'{tag}: prefill={row["prefill_ms"]} ms decode={row["decode_ms"]} ms/token RSS={row["rss_kib"]} KiB', flush=True)
    result['summaries'] = summary(result['rows'])
    result['verdict'] = 'PASS' if result['summaries']['pass'] else 'FAIL'
    return result


def production_outputs(stdout):
    """Read raw answers up to the next CLI prompt, excluding /status output.

    These controlled prompts do not ask the model to print CLI delimiters.
    Comparing the whole remainder would compare changing timing statistics and
    the next turn's streamed answer as if they were part of this response.
    """
    answers = []
    for part in stdout.split('[OFLM]  Model RAW Output:')[1:]:
        end = re.search(r'\n>>> ', part)
        if not end:
            raise ValueError('raw model output has no following CLI prompt')
        answers.append(part[:end.start()].strip())
    return answers


def production(a, out):
    # The old and new policy can be compared with the same production binary:
    # opt-out leaves libgomp's launch-time default; no stale constructor override.
    tok = D.Tokenizer()
    rows = []
    D.OFLM, D.MODEL, D.KERNELS = a.after, a.model, a.kernels
    for arm in ('before', 'after'):
        extra = {'OFLM_OPEN_OMP_PASSIVE': '0' if arm == 'before' else '1',
                 'OFLM_OPEN_LAYER_MAJOR': '0', 'OFLM_OPEN_PREFILL_ROUTE': '1'}
        lines = ['/clear', tok.exact_text(64), '/clear']
        for n in a.lengths:
            lines += [tok.exact_text(n), '/status', '/clear']
        saved = dict(os.environ)
        try:
            os.environ.clear(); os.environ.update(clean_env_from(saved))
            rec = D.run(lines, block=True, gen_lim=a.generate, ctx=4096, timeout=3600, tag=arm, extra=extra)
        finally:
            os.environ.clear(); os.environ.update(saved)
        (out / f'{arm}.log').write_text(rec['text'])
        rec['turns'] = production_outputs(rec['stdout'])
        if rec['rc'] or len(rec['turns']) != len(a.lengths) + 1:
            raise ValueError(f'{arm}: incomplete production session')
        rows.append(rec | dict(arm=arm, conditions_after=P.conditions()))
        (out / 'result.json').write_text(json.dumps(dict(rows=rows), indent=2) + '\n')
    if rows[0]['turns'] != rows[1]['turns']:
        raise ValueError('production outputs differ')
    return dict(check='omp-policy-production', lengths=a.lengths, generated=a.generate,
                rows=rows, verdict='PASS', performance_acceptance=False,
                note='single session per arm; timing cross-check only, not a repeated benchmark')


def clean_env_from(env):
    return {k: v for k, v in env.items() if k not in
            ('OMP_WAIT_POLICY', 'OFLM_OPEN_OMP_REEXEC', 'OFLM_OPEN_OMP_PASSIVE',
             'OFLM_OPEN_PROFILE', 'OFLM_OPEN_PROFILE_STEPS',
             'OFLM_OPEN_PREFILL_ROLES', 'OFLM_OPEN_PREFILL_ROLE_MODE')}



def boundaries(a, out):
    import decode_transition as T
    rows = []
    O.MODEL, O.KERNELS = a.model, a.kernels
    for n in a.lengths:
        ids = O.fixture(n, seed=26026)
        slices = T.state_slices(a.layers, n + a.generate)
        records = {}
        for arm in ('before', 'after'):
            O.CLI = getattr(a, arm)
            wd = out / f'{n}-{arm}'; wd.mkdir()
            flags = [f'{layer}:{offset}:{size}:{T.state_path(wd, layer, offset, size)}'
                     for layer, spans in slices.items() for _, offset, size in spans]
            saved = dict(os.environ)
            try:
                os.environ.clear(); os.environ.update(clean_env_from(saved))
                records[arm] = T.run(ids, wd, arm, chunk=256, layers=a.layers, decode=a.generate,
                                    position=0, dump_state=flags, outdir=str(out))
            finally:
                os.environ.clear(); os.environ.update(saved)
        checks, _ = T.compare_logits(records['before'], records['after'], n, a.generate, strict=True)
        checks += T.compare_state(Path(records['before']['workdir']), Path(records['after']['workdir']), slices)
        # Scheduling only: require exact bytes as well as the unchanged numerical gates.
        left, right = (Path(records[k]['workdir']) for k in ('before', 'after'))
        states_equal = all((left / f.name).read_bytes() == f.read_bytes() for f in right.glob('state_*.bin'))
        logits_equal = all(Path(str(records['before']['prefix']) + suffix).read_bytes() ==
                           Path(str(records['after']['prefix']) + suffix).read_bytes()
                           for suffix in (f'_p{n-1}.bin', *[f'_t{i}.bin' for i in range(a.generate)]))
        row = dict(tokens=n, records=records, checks=checks, state_exact=states_equal,
                   logits_exact=logits_equal, verdict=O.verdict(checks))
        rows.append(row)
        (out / 'result.json').write_text(json.dumps(dict(rows=rows), indent=2) + '\n')
        print(f'boundary {n}: {row["verdict"]} exact={states_equal and logits_equal}', flush=True)
    return dict(check='omp-policy-boundaries', layers=a.layers, rows=rows,
                verdict='PASS' if all(r['verdict'] == 'PASS' and r['state_exact'] and r['logits_exact'] for r in rows) else 'FAIL')


def server(a, out):
    import fcntl
    import oflm_serve_check as S
    import urllib.request
    D.OFLM, D.MODEL, D.KERNELS = a.after, a.model, a.kernels
    tok = D.Tokenizer()
    prompts = {'A': tok.exact_text(256) + '\nReply with only the word RED.',
               'B': tok.exact_text(512) + '\nReply with only the word BLUE.'}
    rows = []
    with O.LOCK.open('a') as lock, (out / 'server.log').open('w') as log:
        fcntl.flock(lock, fcntl.LOCK_EX)
        env = clean_env_from(D.env(block=True, extra={'OFLM_OPEN_LAYER_MAJOR': '0'}))
        proc = subprocess.Popen([str(a.after), 'serve', 'qwen3.8:27b', '--ctx-len', '4096', '-p', str(a.port)],
                                env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            err = S.wait_ready(a.port, proc)
            if err: raise RuntimeError(err)
            for label in ('A', 'B', 'A', 'B', 'A', 'B', 'A'):
                payload = dict(model='qwen3.8:27b', messages=[dict(role='user', content=prompts[label])],
                               temperature=0, max_tokens=a.generate, stream=False, think=False)
                req = urllib.request.Request(f'http://127.0.0.1:{a.port}/v1/chat/completions',
                                             data=json.dumps(payload).encode(),
                                             headers={'Content-Type': 'application/json'})
                with S._open(req, 1800) as response: data = json.loads(response.read())
                status = Path(f'/proc/{proc.pid}/status').read_text()
                rss = int(re.search(r'VmRSS:\s+(\d+)', status)[1])
                rows.append(dict(prompt=label, response=data, rss_kib=rss))
                (out / 'result.json').write_text(json.dumps(dict(rows=rows), indent=2) + '\n')
                print(f'server {label}: RSS={rss} KiB', flush=True)
        finally:
            proc.terminate()
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
    stable = all(r['response']['choices'][0]['message'] == next(x for x in rows if x['prompt'] == r['prompt'])['response']['choices'][0]['message'] for r in rows)
    distinct = rows[0]['response']['choices'][0]['message'] != rows[1]['response']['choices'][0]['message']
    complete = all(0 < r['response'].get('usage', {}).get('completion_tokens', 0) <= a.generate for r in rows)
    rss = [r['rss_kib'] for r in rows[2:]]  # both prompt sizes have allocated once
    bounded = max(rss) <= min(rss) * 1.03
    return dict(check='omp-policy-server', rows=rows, repeated_outputs_identical=stable,
                memory_bounded=bounded, distinct_outputs=distinct,
                verdict='PASS' if stable and complete and bounded and distinct else 'FAIL')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', choices=['benchmark', 'production', 'boundaries', 'server'], default='benchmark')
    ap.add_argument('--before', type=Path)
    ap.add_argument('--after', type=Path, required=True)
    ap.add_argument('--model', type=Path, default=D.MODEL)
    ap.add_argument('--kernels', type=Path, default=D.KERNELS)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--port', type=int, default=18762)
    ap.add_argument('--tokens', type=int, default=256)
    ap.add_argument('--lengths', type=int, nargs='+', default=[256, 512, 1024])
    ap.add_argument('--generate', type=int, default=16)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.reps < 5 or a.generate < 2 or (a.check in ('benchmark', 'boundaries') and not a.before):
        ap.error('benchmark requires --before, at least five pairs and two generated tokens')
    for name in ('before', 'after', 'model', 'kernels'):
        if getattr(a, name): setattr(a, name, getattr(a, name).resolve())
    if a.check in ('production', 'server') and a.model != D.MODEL.resolve():
        ap.error('production checks use the registered qwen3.8:27b model at ' + str(D.MODEL))
    if a.tokens < 1 or a.layers < 1 or any(n < 1 for n in a.lengths):
        ap.error('positive token and layer counts required')
    a.out = a.out.resolve(); a.out.mkdir(parents=True, exist_ok=False)
    result = {'benchmark': benchmark, 'production': production, 'boundaries': boundaries, 'server': server}[a.check](a, a.out)
    (a.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result['verdict'], a.out, flush=True)
    return result['verdict'] != 'PASS'


if __name__ == '__main__':
    sys.exit(main())
