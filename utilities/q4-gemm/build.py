"""Build a private Q4 GEMM DMA variant, recording compiler inputs and core budgets."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / 'open_kernels/designs/gemm_q4_prefill'
sys.path.insert(0, str(ROOT / 'utilities'))
from core_sizes import sections


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def resources(out: Path) -> dict:
    project = out / 'final.prj'
    elfs = sorted(project.glob('elfs_main_core_*/*.elf'))
    text = {p.stem.removeprefix('elfs_main_core_'): sections(p.read_bytes())['.text'] for p in elfs}
    mlir = project / 'input_with_addresses.mlir'
    ends = {}
    for line in mlir.read_text().splitlines():
        if not re.search(r'aie.buffer\(%tile_\d+_[2-5]\)', line):
            continue
        match = re.search(r'aie.buffer\(%tile_(\d+_[2-5])\).*address = (\d+).*memref<([\dx]+)x(bf16|f32|i32|ui8)>', line)
        if match is None:
            raise ValueError('unrecognized compute-buffer layout')
        tile, address, dims, dtype = match.groups()
        end = int(address) + math.prod(map(int, dims.split('x'))) * {'bf16': 2, 'f32': 4, 'i32': 4, 'ui8': 1}[dtype]
        ends[tile] = max(ends.get(tile, 0), end)
    if len(text) != 32 or text.keys() != ends.keys():
        raise ValueError('incomplete 32-core resource coverage')
    if max(text.values()) >= 16384 or max(ends.values()) > 65536:
        raise ValueError('core memory budget exceeded')
    return dict(text=text, l1_buffer_end=ends, stack_bytes=4096,
                evidence={str(p): digest(p) for p in [mlir, *elfs]})


def build(a: argparse.Namespace) -> None:
    if a.n <= 0 or a.n % 256 or a.k <= 0 or a.k % 256 or a.tokens <= 0 or a.tokens % 256:
        raise ValueError('positive 256-aligned N, K and tokens required')
    if not a.out.is_relative_to(ROOT / 'Models'):
        raise ValueError('private build output must be under Models/')
    a.out.mkdir(parents=True, exist_ok=False)
    record = dict(complete=False, n=a.n, k=a.k, tokens=a.tokens,
                  dma_batch=a.batch, weight_low=a.low)
    try:
        generated = a.out / 'generated'
        generated.mkdir()
        sources = sorted(DESIGN.glob('*.py')) + sorted(DESIGN.glob('*.h'))
        sources += [Path(__file__).resolve(), ROOT / 'open_kernels/build_design.py', ROOT / 'utilities/core_sizes.py']
        record['sources'] = {str(p): digest(p) for p in sources}
        for path in sources:
            if path.parent == DESIGN:
                shutil.copy2(path, generated / path.name)
        env = {k: v for k, v in os.environ.items() if not k.startswith('GQP_')}
        env.update(GQP_N=str(a.n), GQP_K=str(a.k), GQP_T=str(a.tokens), GQP_FMT='q4_1',
                   GQP_BREUSE='0', GQP_WEIGHT_LOW='1' if a.low else '0', GQP_DMA_BATCH=str(a.batch),
                   OFLM_KEEP_FAILED='1', OPENBLAS_NUM_THREADS='1')
        record['environment'] = {k: v for k, v in env.items() if k.startswith('GQP_') or
                                 k in ('PEANO_INSTALL_DIR', 'XILINX_XRT', 'PATH')}
        command = [sys.executable, str(ROOT / 'open_kernels/build_design.py'),
                   str(generated / 'gemm_q4_prefill.py'), str(a.out)]
        record['command'] = command
        with (a.out / 'build.log').open('w') as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        deps = set()
        depfiles = sorted((a.out / 'final.prj').glob('*.o.d'))
        if len(depfiles) < 4:
            raise ValueError('missing compiler dependency evidence')
        for path in depfiles:
            names = shlex.split(path.read_text().replace('\\\n', ' ').split(':', 1)[1])
            deps.update(Path(name).resolve() for name in names)
        if generated / 'gemm_q4_dequant.h' not in deps:
            raise ValueError('private dequantization header was not compiled')
        record['compiler_dependencies'] = {str(p): digest(p) for p in sorted(deps)}
        record['resources'] = resources(a.out)
        record['binaries'] = {str(a.out / name): digest(a.out / name)
                              for name in ('final.xclbin', 'insts.bin', 'insts.elf')}
        record['generated'] = {str(p): digest(p) for p in generated.iterdir() if p.is_file()}
        for path, identity in record['sources'].items():
            if digest(Path(path)) != identity:
                raise ValueError(f'source changed during build: {path}')
        record['complete'] = True
    except Exception as exc:
        record['error'] = str(exc)
        raise
    finally:
        (a.out / 'build.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--tokens', type=int, default=256)
    parser.add_argument('--batch', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--low', action='store_true')
    parser.add_argument('--out', type=lambda v: Path(v).resolve(), required=True)
    build(parser.parse_args())
