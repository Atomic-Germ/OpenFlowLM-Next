"""A decode fixture must target an explicit, intact kernel export."""
import copy
import hashlib
import json

import pytest

from make_decode import build_cfg


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / 'export'
    (root / 'dx').mkdir(parents=True)
    m = dict(build_key='old-build', spec_hash='same-spec',
             layout=dict(kv_row=4, ptab_row=4, hidden=4, pool_bytes=4),
             layers=['dense'], globals={'logits': 20}, tail=[],
             contexts={'dx': 'dx/final.xclbin'},
             kernels={'dx': dict(context='dx', insts='dx/insts.bin', build='dx')},
             builds={'dx': dict(build_dir='unrelated-working-build')},
             layer_types={'dense': dict(buffers=dict(consts=4, act=4, state=dict(kind='kv', row=4)),
                                        program=[dict(op='run', kernel='dx', args=['xres'])])})
    hashes = {}
    for name in ('final.xclbin', 'insts.bin'):
        data = name.encode()
        (root / 'dx' / name).write_bytes(data)
        hashes[f'dx/{name}'] = hashlib.sha256(data).hexdigest()
    (root / 'manifest.json').write_text(json.dumps(m))
    (root / 'toolchain.json').write_text(json.dumps(dict(build_key='old-build', spec_hash='same-spec', sha256=hashes)))
    current = copy.deepcopy(m)
    current['build_key'] = 'current-recipe-build'
    return root, current, m


def load(root, current):
    from decode_bundle import load_bundle
    return load_bundle(root, current)


def test_compatible_export_keeps_its_original_build_identity(bundle):
    root, current, exported = bundle
    assert load(root, current) == exported
    assert current['build_key'] == 'current-recipe-build'


@pytest.mark.parametrize('field', ['spec_hash', 'layout', 'layers', 'kernels', 'builds', 'tail'])
def test_recipe_changes_are_not_hidden_by_allowing_an_old_build_key(bundle, field):
    root, current, _ = bundle
    current[field] = 'different'
    with pytest.raises(ValueError, match='manifest'):
        load(root, current)


@pytest.mark.parametrize('damage', ['missing', 'changed', 'unhashed', 'build_key', 'spec_hash'])
def test_incomplete_or_modified_export_fails(bundle, damage):
    root, current, _ = bundle
    path = root / 'toolchain.json'
    record = json.loads(path.read_text())
    if damage == 'missing':
        (root / 'dx/insts.bin').unlink()
    elif damage == 'changed':
        (root / 'dx/final.xclbin').write_bytes(b'other kernel')
    elif damage == 'unhashed':
        del record['sha256']['dx/insts.bin']
    else:
        record[damage] = 'unrelated'
    path.write_text(json.dumps(record))
    with pytest.raises((ValueError, OSError)):
        load(root, current)


def test_export_cfg_reads_export_paths_not_working_builds(bundle, tmp_path):
    root, current, _ = bundle
    m = load(root, current)
    cfg = build_cfg(m, 1, 1, tmp_path, tmp_path, 8, kernel_dir=root)
    assert f'xclbin dx {root}/dx/final.xclbin' in cfg
    assert f'kernelx dx dx {root}/dx/insts.bin' in cfg
    assert 'unrelated-working-build' not in cfg


def test_cfg_only_export_selection_preserves_reference_build_key(bundle, tmp_path):
    from decode_reference import check_reference, save_reference
    import numpy as np
    root, current, _ = bundle
    current['max_ctx_default'] = 8
    exported = copy.deepcopy(current)
    exported['build_key'] = 'old-build'
    (root / 'manifest.json').write_text(json.dumps(exported))
    out = tmp_path / 'refs'
    out.mkdir()
    np.arange(5, dtype=np.float32).tofile(out / 'ref_logits.bin')
    np.arange(4, dtype=np.float32).tofile(out / 'ref_res0.bin')
    m = load(root, current)
    save_reference(out, m, 1, 1, 1, 'independent')
    check_reference(out, load(root, current), 1, 1, 1, 'independent')
    with pytest.raises(ValueError, match='build_key'):
        check_reference(out, current, 1, 1, 1, 'independent')


def test_export_symlink_cannot_read_outside_bundle(bundle, tmp_path):
    root, current, _ = bundle
    binary = root / 'dx/insts.bin'
    outside = tmp_path / 'external.bin'
    binary.rename(outside)
    binary.symlink_to(outside)
    with pytest.raises(ValueError, match='escapes'):
        load(root, current)
