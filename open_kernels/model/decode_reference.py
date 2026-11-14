"""Expected decode coverage and immutable reference-file digests.

Written only after make_decode finishes its CPU reference. This records fixture
provenance, not authentication or a hash of every source weight/kernel binary.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FILENAME = 'decode_reference.json'


def sfx(t: int) -> str:
    return '' if t == 0 else f'_t{t}'


def expected_references(meta: dict) -> dict[str, int]:
    files = {}
    for t in range(meta['tokens']):
        suffix = sfx(t)
        files[f'ref_logits{suffix}.bin'] = meta['vocab']
        for layer in range(meta['layers']):
            files[f'ref_res{layer}{suffix}.bin'] = meta['hidden']
    return files


def read_capture(path: Path, count: int, digest: str | None = None) -> np.ndarray:
    data = path.read_bytes()
    if len(data) != count * 4:
        raise ValueError(f'{path.name}: expected {count * 4} bytes ({count} float32), got {len(data)}')
    if digest is not None and hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'{path.name}: reference hash mismatch; do not modify the pinned reference')
    values = np.frombuffer(data, dtype=np.float32).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f'{path.name}: non-finite values')
    return values


def load_reference(out: Path) -> dict:
    path = out / FILENAME
    if not path.is_file():
        raise ValueError(f'{FILENAME} missing; regenerate the reference with make_decode.py (without --cfg-only)')
    meta = json.loads(path.read_text())
    if not isinstance(meta, dict):
        raise ValueError(f'{FILENAME}: expected an object')
    if type(meta.get('version')) is not int or meta['version'] != 1:
        raise ValueError(f'{FILENAME}: unsupported version')
    for field in ('tokens', 'layers', 'hidden', 'vocab', 'max_ctx'):
        if type(meta.get(field)) is not int or meta[field] < (2 if field == 'vocab' else 1):
            raise ValueError(f'{FILENAME}: invalid {field}')
    if meta['tokens'] > meta['max_ctx']:
        raise ValueError(f'{FILENAME}: tokens exceed max_ctx')
    for field in ('spec_hash', 'build_key'):
        if not isinstance(meta.get(field), str) or not meta[field]:
            raise ValueError(f'{FILENAME}: missing {field}')
    if type(meta.get('seed_token')) is not int or meta['seed_token'] < 0:
        raise ValueError(f'{FILENAME}: invalid seed_token')
    if meta.get('routing') not in ('independent', 'device-assisted'):
        raise ValueError(f'{FILENAME}: invalid routing provenance')
    refs = meta.get('references')
    if not isinstance(refs, dict) or set(refs) != set(expected_references(meta)):
        raise ValueError(f'{FILENAME}: references do not cover exactly the declared tokens/layers')
    if any(not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
           for v in refs.values()):
        raise ValueError(f'{FILENAME}: invalid reference hash')
    return meta


def _metadata(manifest: dict, layers: int, tokens: int, seed: int, routing: str) -> dict:
    if not 0 < layers <= len(manifest['layers']) or not 0 < tokens <= manifest['max_ctx_default']:
        raise ValueError('layers and tokens must describe a nonempty run within the manifest')
    logits_bytes = manifest['globals']['logits']
    if logits_bytes % 4:
        raise ValueError('manifest logits size is not float32-aligned')
    return dict(version=1, layers=layers, tokens=tokens, hidden=manifest['layout']['hidden'],
                vocab=logits_bytes // 4, max_ctx=manifest['max_ctx_default'], spec_hash=manifest['spec_hash'],
                build_key=manifest['build_key'], seed_token=seed, routing=routing)


def save_reference(out: Path, manifest: dict, layers: int, tokens: int, seed: int, routing: str) -> None:
    meta = _metadata(manifest, layers, tokens, seed, routing)
    refs = {}
    for name, count in expected_references(meta).items():
        path = out / name
        read_capture(path, count)
        refs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta['references'] = refs
    temporary = out / (FILENAME + '.tmp')
    temporary.write_text(json.dumps(meta, indent=2) + '\n')
    temporary.replace(out / FILENAME)


def check_reference(out: Path, manifest: dict, layers: int, tokens: int, seed: int, routing: str) -> None:
    """A cfg-only rewrite cannot silently relabel old reference files."""
    actual = load_reference(out)
    for field, value in _metadata(manifest, layers, tokens, seed, routing).items():
        if actual[field] != value:
            raise ValueError(f'{FILENAME}: {field} differs; regenerate with make_decode.py without --cfg-only')
    for name, count in expected_references(actual).items():
        read_capture(out / name, count, actual['references'][name])
