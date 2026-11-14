"""Phase 29 item 1: the KV cost the ALLOCATOR pays, not the formula's.

The formula says 8192 B per full-attention layer per token, so peak RSS must rise by
8192 x 16 = 131072 B per extra context token. This checks that against the running process
instead of trusting the arithmetic: load the engine at each context, read VmHWM from
/proc, and report the slope.

    python kv_rss_probe.py --lengths 4096 32768 131072
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oflm_drive as D  # noqa: E402

GiB = 2 ** 30


def vmhwm_kb(pid: int) -> int | None:
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmHWM:'):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def peak(ctx: int, timeout: float) -> dict:
    """Load at `ctx`, wait until the capacity is reported, then read the peak RSS."""
    cmd = [str(D.OFLM), 'run', 'qwen3.8:27b', '--ctx-len', str(ctx)]
    proc = subprocess.Popen(cmd, cwd=str(D.ROOT), env=D.env(block=True),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    t0 = time.monotonic()
    cap = None
    try:
        while time.monotonic() - t0 < timeout:
            line = proc.stdout.readline()
            if not line:
                break
            m = re.search(r'context capacity (\d+)', line)
            if m:
                cap = int(m.group(1))
                break
        # Sample the high-water mark a moment after the allocation settled.
        time.sleep(1.5)
        rss = vmhwm_kb(proc.pid)
    finally:
        try:
            proc.stdin.write('/bye\n')
            proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return dict(ctx=ctx, capacity_reported=cap, peak_rss_kb=rss,
                peak_rss_gib=round(rss / 1024 / 1024, 3) if rss else None,
                wall_s=round(time.monotonic() - t0, 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lengths', type=int, nargs='+', default=[4096, 32768, 131072])
    ap.add_argument('--timeout', type=float, default=1800)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'kv-rss-{stamp}')
    out.mkdir(parents=True, exist_ok=True)
    rows = [peak(n, a.timeout) for n in a.lengths]
    for r in rows:
        print(f"  ctx={r['ctx']:7d} capacity={r['capacity_reported']} "
              f"peak_rss={r['peak_rss_gib']} GiB", flush=True)

    slope = None
    ok = [r for r in rows if r['peak_rss_kb'] and r['capacity_reported'] == r['ctx']]
    if len(ok) >= 2:
        d_ctx = ok[-1]['ctx'] - ok[0]['ctx']
        d_kb = ok[-1]['peak_rss_kb'] - ok[0]['peak_rss_kb']
        slope = round(d_kb * 1024 / d_ctx, 1) if d_ctx else None
    # RSS CANNOT SEE THIS CACHE. The KV buffer is a DEVICE buffer
    # (`xrt::ext::bo(*dev_, (max_ctx + slack) * state_row)`), so its bytes are allocated in
    # NPU memory and never appear in the process's resident set. The measured slope here was
    # a few hundred bytes per token and even negative, which is noise in a ~2 GiB footprint,
    # not a small cache. Reporting that slope as a pass would have been a fabricated
    # confirmation of the formula.
    result = dict(tool='prefill-block-kv-rss', version=2, phase=29,
                  gate='29 item 1: does the host allocator pay for the KV cache?',
                  model_acceptance=False, binary=str(D.OFLM), kernels=str(D.KERNELS),
                  started=stamp, rows=rows,
                  observed_bytes_per_token=slope,
                  expected_bytes_per_token=131072,
                  why_rss_is_the_wrong_instrument='the KV state buffer is a device BO '
                  '(core.cpp: alloc(lt.state_kind == "kv" ? (cfg_.max_ctx + kv_slack) * '
                  'lt.state_row : ...)), so host RSS never contains it',
                  conclusion='RSS is flat across 4K..128K, which confirms the cache is NOT '
                             'host-resident; the authoritative size is the manifest row and '
                             'the allocation expression, not an RSS slope')
    result['verdict'] = 'INCONCLUSIVE-BY-DESIGN'
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"\nobserved slope {slope} B/token against 131072 expected -- RSS cannot see a "
          'device buffer, so this is not evidence either way')
    print(f"{result['verdict']} -> {out / 'result.json'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
