"""Build/staging/reference helpers are testable without a compiler or NPU."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name):
    spec = importlib.util.spec_from_file_location('q4_gemm_' + name, ROOT / f'utilities/q4-gemm/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_high_and_low_references_use_fp64_products():
    bf16 = pytest.importorskip('ml_dtypes').bfloat16
    tool = load('fixtures')
    weights = np.array([[1.001, -2.003], [0.00201, 8.014]], np.float32)
    activations = np.array([[1, 2], [-3, 4]], np.float32).astype(bf16)
    high, low = tool.references(weights, activations)
    rounded = weights.astype(bf16).astype(np.float32)
    residual = (weights - rounded).astype(bf16).astype(np.float64)
    np.testing.assert_array_equal(high, (weights.astype(np.float64) @ activations.astype(np.float64).T).astype('<f4'))
    np.testing.assert_array_equal(low, (residual @ activations.astype(np.float64).T).astype('<f4'))
    assert np.any(low != 0)


def stage_fixture(tmp_path, monkeypatch):
    tool = load('stage')
    monkeypatch.setattr(tool, 'ROOT', tmp_path)
    base, variants, out = tmp_path/'base', tmp_path/'variants', tmp_path/'Models/output'
    base.mkdir(); variants.mkdir()
    (base/'manifest.json').write_bytes(b'{"mapping":"retained"}\n')
    (base/'lx').mkdir()
    name = 'gemm_n256_k256'
    (base/name).mkdir()
    target = variants/name
    target.mkdir()
    for file in ('final.xclbin', 'insts.bin', 'insts.elf'):
        (target/file).write_bytes(file.encode())
    record = dict(complete=True, n=256, k=256, tokens=256, weight_low=False, dma_batch=4,
                  binaries={str(p): tool.digest(p) for p in target.iterdir()})
    (target/'build.json').write_text(json.dumps(record))
    return tool, base, variants, out, target


def test_staging_preserves_manifest_and_unrelated_kernels(tmp_path, monkeypatch):
    tool, base, variants, out, target = stage_fixture(tmp_path, monkeypatch)
    probe = tmp_path/'link-probe'
    try:
        probe.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip('directory symlink capability required for private staging')
    tool.stage(base, variants, out)
    assert (out/'manifest.json').read_bytes() == (base/'manifest.json').read_bytes()
    assert (out/'lx').resolve() == base/'lx'
    assert (out/target.name).resolve() == target
    assert json.loads((out/'stage.json').read_text())['manifest_sha256'] == tool.digest(base/'manifest.json')


def test_staging_refuses_changed_binaries_and_missing_shapes(tmp_path, monkeypatch):
    tool, base, variants, out, target = stage_fixture(tmp_path, monkeypatch)
    (target/'insts.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='binary changed'):
        tool.stage(base, variants, out)
    assert not out.exists()
    (base/'gemm_n512_k256').mkdir()
    with pytest.raises(ValueError, match='sets differ'):
        tool.stage(base, variants, out)


@pytest.mark.parametrize('text_size,buffer_size,passes', [(5000, 61448, True),
                                                        (16384, 61448, False),
                                                        (5000, 65540, False)])
def test_resource_gate_covers_all_cores_and_enforces_budgets(tmp_path, monkeypatch,
                                                          text_size, buffer_size, passes):
    tool = load('build')
    project = tmp_path/'final.prj'
    project.mkdir()
    lines = []
    for col in range(8):
        for row in range(2, 6):
            tile = f'{col}_{row}'
            folder = project/f'elfs_main_core_{tile}'
            folder.mkdir()
            (folder/f'elfs_main_core_{tile}.elf').write_bytes(b'fixture ELF')
            lines.append(f'%b = aie.buffer(%tile_{tile}) {{address = 0 : i32}} : memref<{buffer_size}xui8>')
    (project/'input_with_addresses.mlir').write_text('\n'.join(lines))
    monkeypatch.setattr(tool, 'sections', lambda data: {'.text': text_size})
    if passes:
        result = tool.resources(tmp_path)
        assert len(result['text']) == len(result['l1_buffer_end']) == 32
        assert result['stack_bytes'] == 4096
    else:
        with pytest.raises(ValueError, match='memory budget'):
            tool.resources(tmp_path)
