"""Private prefill trace: causal coverage, control identity and per-row gates."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / 'utilities/prefill-residual-trace'


def load(name):
    s = importlib.util.spec_from_file_location('residual_trace_' + name, TOOL / (name + '.py'))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_writer_causal_chunks_and_padding(tmp_path):
    cpp = tmp_path / 'writer.cpp'
    cpp.write_text(r'''
#include "trace.hpp"
#include <limits>
using namespace open_qwen36::residual_trace;
int main(int argc, char** argv) {
    Writer w(argv[1]);
    const float x[] = {1,2,3,4,5,6,99,99};
    w.append(0,0,3,2,x); // discard the padded fourth row
    w.append(1,0,1,2,x);
    w.append(1,1,2,2,x+2);
    w.append(0,3,1,2,x); // independent continuation
    w.append(1,3,1,2,x);
    w.finish(2,4,2);
}
''')
    exe = tmp_path / 'writer'
    subprocess.run(['c++', '-std=c++17', '-O2', '-I', str(TOOL), str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe), str(tmp_path / 'run')], check=True)
    expected = np.array([1,2,3,4,5,6,1,2], dtype='<f4').tobytes()
    assert (tmp_path / 'run_res0.bin').read_bytes() == expected
    assert (tmp_path / 'run_res1.bin').read_bytes() == expected


@pytest.mark.parametrize('bad', ['gap','duplicate','width','nan','incomplete','existing','closed'])
def test_writer_rejects_invalid_capture(tmp_path, bad):
    actions = {
        'gap': 'w.append(0,2,1,2,x);',
        'duplicate': 'w.append(0,0,1,2,x);',
        'width': 'w.append(0,1,1,1,x);',
        'nan': 'w.append(0,1,1,2,bad);',
        'incomplete': 'w.finish(2,1,2);',
        'existing': 'Writer again(argv[1]); again.append(0,0,1,2,x);',
        'closed': 'w.finish(1,1,2); w.append(0,1,1,2,x);',
    }
    cpp = tmp_path / 'invalid.cpp'
    cpp.write_text('''#include "trace.hpp"
#include <limits>
using namespace open_qwen36::residual_trace;
int main(int argc, char** argv) {
 Writer w(argv[1]); const float x[]={1,2}, bad[]={1,std::numeric_limits<float>::quiet_NaN()};
 w.append(0,0,1,2,x);
 try { ''' + actions[bad] + ''' } catch(const std::runtime_error&) { return 0; }
 return 1;
}''')
    exe = tmp_path / 'invalid'
    subprocess.run(['c++', '-std=c++17', '-I', str(TOOL), str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe), str(tmp_path / 'run')], check=True)


def test_per_position_gate_finds_small_row_hidden_by_global_scale():
    m = load('audit')
    ref = np.array([[1000.,1000.],[1.,1.],[0.,0.]])
    got = ref.copy(); got[1,0] += .01; got[2,0] = 1e-32
    rows = m.score_rows(got, ref)
    assert [r['passed'] for r in rows] == [True,False,False]
    assert rows[1]['maxrel'] == pytest.approx(.01)
    assert rows[2]['maxrel'] is None  # zero reference is exact, no epsilon floor


@pytest.mark.parametrize('damage', ['truncated','nan','extra'])
def test_trace_reader_rejects_invalid_inventory(tmp_path, damage):
    m = load('audit')
    values = np.ones(6, dtype='<f4')
    if damage == 'truncated': values = values[:-1]
    if damage == 'extra': values = np.append(values, np.float32(1))
    if damage == 'nan': values[0] = np.nan
    path = tmp_path / 'trace.bin'; values.tofile(path)
    with pytest.raises(ValueError): m.read_trace(path,3,2)


def test_identity_is_required_before_interpreting_residuals(tmp_path):
    m = load('audit')
    before, after = tmp_path/'before', tmp_path/'after'
    before.mkdir(); after.mkdir()
    for route in m.ROUTES:
        for name in m.paired.base.expected_dumps(route,1):
            (before/name).write_bytes(b'original'); (after/name).write_bytes(b'original')
    (after/'sequential_p0.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='logit identity'):
        m.require_identity(before,after,1)


def test_audit_error_invalidates_previous_success(tmp_path):
    m = load('audit')
    out = tmp_path/'trace-audit.json'; out.write_text('{"capture_valid":true}')
    with pytest.raises(FileNotFoundError):
        m.audit(SimpleNamespace(out=out, build=tmp_path/'missing', baseline=tmp_path/'base', run=tmp_path/'run'))
    result = json.loads(out.read_text())
    assert result['capture_valid'] is False and result['numerical_passed'] is False


def test_patch_rejects_source_drift_and_preserves_paired_body():
    m = load('build')
    cpp = (ROOT/'src/open_qwen36/core.cpp').read_text()
    hpp = (ROOT/'src/open_qwen36/core.hpp').read_text()
    cli = (ROOT/'src/open_qwen36/cli.cpp').read_text()
    paired, _ = m.paired.patch(cpp,hpp)
    traced, traced_cli = m.patch(paired,cli)
    start = paired.index('const float* Core::gemm_run_once(')
    end = paired.index('\nvoid Core::gemm(',start)
    assert paired[start:end] in traced
    assert 'residual_trace::writer.finish' in traced_cli
    with pytest.raises(ValueError): m.patch(traced,traced_cli)
    with pytest.raises(ValueError): m.patch(paired.replace('const bool pipe =', 'const bool renamed ='),cli)


@pytest.mark.parametrize('damage', ['none','numeric','missing','extra','identity'])
def test_complete_audit_distinguishes_invalid_capture_from_numeric_failure(tmp_path, monkeypatch, damage):
    m = load('audit')
    before, after = tmp_path/'before', tmp_path/'after'
    before.mkdir(); after.mkdir()
    meta = dict(ids=[10,20], hidden=2, layers=4)
    for directory in (before,after):
        (directory/'reference.json').write_text(json.dumps(meta))
        (directory/'report.json').write_text('{}')
    for name in ('build','baseline_build'):
        (tmp_path/name).mkdir(); (tmp_path/name/'build-evidence.json').write_text('{}')
    monkeypatch.setattr(m,'verify_build',lambda *args: dict(cli_sha256='cli'))
    monkeypatch.setattr(m.paired,'validate_run',lambda *args: (meta,dict(cli_sha256='cli',passed=True)))
    for route in m.ROUTES:
        for name in m.base.expected_dumps(route,2):
            (before/name).write_bytes(b'logits'); (after/name).write_bytes(b'logits')
    for layer in range(4):
        np.ones((2,2), dtype='<f4').tofile(after/f'ref_res{layer}.bin')
        np.ones(2, dtype='<f4').tofile(after/f'ref_cont_res{layer}.bin')
        for route in m.ROUTES:
            x = np.ones((3,2), dtype='<f4')
            if damage=='numeric' and layer==2: x[1,0] += .01
            x.tofile(after/f'{route}_res{layer}.bin')
    if damage=='missing': (after/'hybrid_res2.bin').unlink()
    if damage=='extra': (after/'hybrid_res4.bin').write_bytes(b'extra')
    if damage=='identity': (after/'hybrid_t1.bin').write_bytes(b'changed')
    out = tmp_path/'audit.json'
    args = SimpleNamespace(build=tmp_path/'build', baseline_build=tmp_path/'baseline_build',
                           baseline=before, run=after, out=out)
    if damage in ('missing','extra','identity'):
        with pytest.raises(ValueError): m.audit(args)
        assert not json.loads(out.read_text())['capture_valid']
    else:
        assert m.audit(args) == (1 if damage=='numeric' else 0)
        result = json.loads(out.read_text())
        assert result['capture_valid']
        assert len(result['rows'])==36 and len(result['trace_hashes'])==12
        first = result['first_failing_position_layer']['sequential']
        if damage=='numeric': assert (first['position'],first['layer'])==(1,2)
        else: assert first is None


def test_runner_memory_refusal_does_not_create_run_directory(tmp_path, monkeypatch):
    m = load('run')
    baseline = tmp_path/'baseline'; baseline.mkdir()
    (baseline/'reference.json').write_text('{"layers":64}')
    def refused(*args): raise ValueError('not enough memory')
    monkeypatch.setattr(m.audit.base, 'memory_preflight', refused)
    out = tmp_path/'run'
    with pytest.raises(ValueError, match='memory'):
        m.run(SimpleNamespace(baseline=baseline, out=out))
    assert not out.exists()


@pytest.mark.parametrize('n', [255,256,257])
def test_writer_matches_all_three_schedules_at_block_boundary(tmp_path, n):
    cpp=tmp_path/'boundary.cpp'
    cpp.write_text(r'''
#include "trace.hpp"
#include <algorithm>
#include <limits>
#include <vector>
using namespace open_qwen36::residual_trace;
int main(int argc,char** argv) {
 const size_t n=std::stoul(argv[2]), hidden=3, padded=((n+255)/256)*256;
 std::vector<float> x(padded*hidden, std::numeric_limits<float>::quiet_NaN());
 for(size_t i=0;i<n*hidden;++i) x[i]=static_cast<float>(i)-1.5f;
 Writer sequential(std::string(argv[1])+"/sequential");
 Writer layer_major(std::string(argv[1])+"/hybrid");
 Writer block_major(std::string(argv[1])+"/block_major");
 for(int l=0;l<4;++l) {
  for(size_t p=0;p<n;++p) sequential.append(l,p,1,hidden,x.data()+p*hidden);
  layer_major.append(l,0,n,hidden,x.data());
 }
 for(size_t p=0;p<n;p+=256)
  for(int l=0;l<4;++l) block_major.append(l,p,std::min(size_t(256),n-p),hidden,x.data()+p*hidden);
 for(auto* w:{&sequential,&layer_major,&block_major}) {
  for(int l=0;l<4;++l) w->append(l,n,1,hidden,x.data());
  w->finish(4,n+1,hidden);
 }
}
''')
    exe=tmp_path/'boundary'
    subprocess.run(['c++','-std=c++17','-I',str(TOOL),str(cpp),'-o',str(exe)],check=True)
    subprocess.run([str(exe),str(tmp_path),str(n)],check=True)
    expected=np.concatenate((np.arange(n*3,dtype='<f4')-1.5,np.arange(3,dtype='<f4')-1.5)).tobytes()
    for route in ('sequential','hybrid','block_major'):
        for layer in range(4): assert (tmp_path/f'{route}_res{layer}.bin').read_bytes()==expected


def test_localization_distinguishes_first_position_from_shallowest_layer():
    m=load('audit')
    rows=[dict(route=route,layer=l,position=p,maxrel=0.001,maxabs=0.01,passed=True)
          for route in m.ROUTES for l in range(4) for p in range(3)]
    for row in rows:
        if row['route']=='sequential' and (row['layer'],row['position']) in ((3,0),(1,2)):
            row.update(passed=False,maxrel=0.02 if row['layer']==3 else 0.03)
    summary=m.localize(rows,4,3)
    seq=summary['sequential']
    assert (seq['first_by_position']['position'],seq['first_by_position']['layer'])==(0,3)
    assert (seq['first_by_layer']['position'],seq['first_by_layer']['layer'])==(2,1)
    assert seq['worst']['maxrel']==0.03
    assert seq['passed']==10 and seq['total']==12
    assert seq['positions'][1]['first_failure'] is None
    assert seq['positions'][2]['first_failure']['layer']==1
    assert seq['layers'][3]['first_failure']['position']==0
    assert summary['hybrid']['first_by_position'] is None


@pytest.mark.parametrize('damage',['missing','duplicate','unknown_route','out_of_range'])
def test_localization_requires_exact_route_layer_position_coverage(damage):
    m=load('audit')
    rows=[dict(route=r,layer=l,position=p,maxrel=0.,maxabs=0.,passed=True)
          for r in m.ROUTES for l in range(2) for p in range(2)]
    if damage=='missing': rows.pop()
    if damage=='duplicate': rows[-1]=rows[0].copy()
    if damage=='unknown_route': rows[-1]['route']='other'
    if damage=='out_of_range': rows[-1]['position']=2
    with pytest.raises(ValueError,match='coverage'): m.localize(rows,2,2)
