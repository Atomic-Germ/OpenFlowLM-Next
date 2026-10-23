"""Select a compatible exported kernel set without changing its build identity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def bundle_path(root: Path, name: str) -> Path:
    root = root.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'kernel export path escapes its directory: {name}')
    return path


def load_bundle(root: Path, expected: dict) -> dict:
    """Require identical runtime/packing semantics; permit a different source build key.

    The selected manifest retains the export's key, and every binary must match
    its toolchain record. This verifies an existing export, not a fresh build
    from the current source tree. No manifest or binary is rewritten.
    """
    selected = json.loads((root / 'manifest.json').read_text())
    record = json.loads((root / 'toolchain.json').read_text())
    if not isinstance(selected, dict) or not isinstance(record, dict):
        raise ValueError('kernel export manifest/toolchain must be objects')
    comparable = lambda m: {k: v for k, v in m.items() if k != 'build_key'}
    if comparable(selected) != comparable(expected):
        raise ValueError('kernel export manifest differs from the current model/recipe')
    for key in ('build_key', 'spec_hash'):
        if not isinstance(selected.get(key), str) or not selected[key] or record.get(key) != selected[key]:
            raise ValueError(f'kernel export toolchain {key} disagrees with manifest')
    hashes = record.get('sha256')
    if not isinstance(hashes, dict):
        raise ValueError('kernel export lacks binary hashes')
    required = {f'{build}/{name}' for build in selected['builds'] for name in ('final.xclbin', 'insts.bin')}
    required.update(selected['contexts'].values())
    required.update(k['insts'] for k in selected['kernels'].values())
    for name in sorted(required):
        if name not in hashes:
            raise ValueError(f'kernel export lacks hash for {name}')
        actual = hashlib.sha256(bundle_path(root, name).read_bytes()).hexdigest()
        if actual != hashes[name]:
            raise ValueError(f'kernel export hash mismatch: {name}')
    return selected
