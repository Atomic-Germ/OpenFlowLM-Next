"""Stage a private kernel-only ablation without regenerating the base manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stage(base: Path, variants: Path, out: Path) -> None:
    if not out.is_relative_to(ROOT / 'Models'):
        raise ValueError('private output must be under Models/')
    names = {p.name for p in base.glob('gemm_n*') if p.is_dir()}
    if not names or names != {p.name for p in variants.iterdir() if p.is_dir()}:
        raise ValueError('base and variant GEMM sets differ')
    records = {}
    for name in sorted(names):
        record = json.loads((variants / name / 'build.json').read_text())
        expected = f'gemm_n{record["n"]}_k{record["k"]}' + ('_low' if record['weight_low'] else '')
        if not record['complete'] or name != expected or record['tokens'] != 256:
            raise ValueError('incomplete build or incompatible projection shape')
        for file in ('final.xclbin', 'insts.bin', 'insts.elf'):
            path = variants / name / file
            if record['binaries'].get(str(path)) != digest(path):
                raise ValueError('variant binary changed after build')
        records[name] = record
    if len({r['dma_batch'] for r in records.values()}) != 1:
        raise ValueError('mixed DMA variants in one arm')
    out.mkdir(parents=True, exist_ok=False)
    for path in base.iterdir():
        if path.name == 'open_kernels':
            continue
        if path.name == 'manifest.json':
            shutil.copy2(path, out / path.name)
        else:
            target = variants / path.name if path.name in names else path
            (out / path.name).symlink_to(target, target_is_directory=target.is_dir())
    (out / 'stage.json').write_text(json.dumps(dict(
        base=str(base), variants=str(variants), manifest_sha256=digest(out / 'manifest.json'),
        builds=records), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'variants', 'out'):
        parser.add_argument('--' + name, type=lambda v: Path(v).resolve(), required=True)
    args = parser.parse_args()
    stage(args.base, args.variants, args.out)
