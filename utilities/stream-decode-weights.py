#!/usr/bin/env python3
"""Reuse one layer-weight BO in a decode harness cfg, retaining every state/capture.

This validates the same sequential kernels with streamed weights; it is not a
resident-runtime or performance test. The original fixture is never rewritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def transform(source):
    lines = source.splitlines()
    pools = {}
    for line in lines:
        words = line.split()
        if 'streamed_pool' in words:
            raise ValueError('streamed_pool already exists')
        if len(words) >= 2 and words[0] == 'buf' and re.fullmatch(r'pool\d+', words[1]):
            if len(words) != 4 or words[1] in pools or int(words[2]) <= 0:
                raise ValueError('pool must have a unique name, positive size and input file')
            pools[words[1]] = (int(words[2]), words[3])
    if not pools:
        raise ValueError('no layer pools')
    out = []
    declared = False
    for line in lines:
        words = line.split()
        used = set(words) & pools.keys()
        if not used:
            out.append(line)
        elif words[0] == 'buf' and words[1] in pools:
            if not declared:
                out.append(f'buf streamed_pool {max(size for size, _ in pools.values())}')
                declared = True
        elif (len(words) >= 3 and words[0] == 'run' and words[1] in ('ax', 'lx')
              and words[2] in pools and used == {words[2]} and declared):
            out.append(f'load streamed_pool {pools[words[2]][1]}')
            words[2] = 'streamed_pool'
            out.append(' '.join(words))
        else:
            raise ValueError(f'unsupported pool use: {line}')
    # A typo in a run must not silently refer to an undeclared pool.
    if any(re.search(r'\bpool\d+\b', line) for line in out if not line.startswith('load ')):
        raise ValueError('undeclared layer pool')
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cfg', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.cfg.resolve() == a.out.resolve():
        raise ValueError('streamed configuration must be a separate file')
    source = a.cfg.read_text()
    result = transform(source)
    a.out.write_text(result)
    a.out.with_suffix('.json').write_text(json.dumps(dict(
        mode='streamed layer weights; all layer state remains resident',
        source=str(a.cfg.resolve()), source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        streamed_sha256=hashlib.sha256(result.encode()).hexdigest()), indent=2)+'\n')
    size = sum(int(s.split()[2]) for s in result.splitlines() if s.startswith('buf '))
    print(f'wrote {a.out}: {size / 2**30:.3f} GiB logical buffers')


if __name__ == '__main__':
    main()
