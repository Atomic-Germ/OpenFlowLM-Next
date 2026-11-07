"""Private weighted CLI must preserve control arithmetic and output ownership."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import subprocess
import pytest
ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT/'utilities/prefill-weighted-cli'

def load():
    s=importlib.util.spec_from_file_location('weighted_builder',TOOL/'build.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_patch_preserves_single_dispatch_and_scopes_weight_correction():
    m=load();cpp=(ROOT/'src/open_qwen36/core.cpp').read_text();hpp=(ROOT/'src/open_qwen36/core.hpp').read_text()
    c,h=m.patch(cpp,hpp,Path('/tmp/low'))
    start=cpp.index('const float* Core::gemm_run(');end=cpp.index('\nvoid Core::gemm(',start)
    assert cpp[start:end].replace('Core::gemm_run(', 'Core::gemm_run_once(',1) in c
    assert 'T != 256 || K != 5120 || N != 34816 || s.split' in c
    assert 'gb.ffn_program[0].args != s.args' in c
    assert 'stage-major is not supported' in c
    assert 'weighted_gemm::Workspace' in h
    assert '/tmp/low/final.xclbin' in c
    with pytest.raises(ValueError):m.patch(c,h,Path('/tmp/low'))


def test_existing_build_is_never_overwritten(tmp_path):
    p=tmp_path/'build-evidence.json';p.write_text('keep')
    with pytest.raises(FileExistsError):load().build(SimpleNamespace(out=tmp_path))
    assert p.read_text()=='keep'


def test_pair_weight_sum_order_and_device_ownership(tmp_path):
    cpp=tmp_path/'test.cpp';exe=tmp_path/'test'
    cpp.write_text(r'''
#include "weighted_gemm.hpp"
#include <array>
#include <stdexcept>
using open_qwen36::weighted_gemm::Workspace;
void require(bool x) { if (!x) throw std::runtime_error("failed"); }
int main() {
 Workspace w; std::array<float,2> device{}; int calls=0;
 auto high=[&]() { ++calls; device={1e8f,3.f}; return device.data(); };
 auto low=[&]() { ++calls; require(device[0]==1e8f); device={-1e8f,4.f}; return device.data(); };
 auto* y=w.run(2,high,low);
 require(calls==2 && y!=device.data() && y[0]==0 && y[1]==7 && device[1]==4);
 y=w.run(2,high,low);require(calls==4 && y[1]==7);
}
''')
    subprocess.run(['c++','-std=c++17','-O2','-I',str(TOOL),str(cpp),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)


def auditor():
    s=importlib.util.spec_from_file_location('weighted_auditor',TOOL/'audit.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_trace_inventory_rejects_missing_and_extra(tmp_path):
    m=auditor()
    for route in m.trace.ROUTES:
        for layer in range(2):(tmp_path/f'{route}_res{layer}.bin').touch()
    m.inventory(tmp_path,2)
    (tmp_path/'ref_res0.bin').touch();m.inventory(tmp_path,2)
    (tmp_path/'hybrid_res1.bin').unlink()
    with pytest.raises(ValueError):m.inventory(tmp_path,2)
    (tmp_path/'hybrid_res1.bin').touch();(tmp_path/'hybrid_res2.bin').touch()
    with pytest.raises(ValueError):m.inventory(tmp_path,2)


def test_audit_failure_clears_stale_success(tmp_path):
    import json
    out=tmp_path/'comparison.json';out.write_text('{"passed":true}')
    with pytest.raises(FileNotFoundError):auditor().audit(SimpleNamespace(out=tmp_path,baseline=tmp_path/'missing'))
    r=json.loads(out.read_text());assert not r['passed'] and not r['capture_valid'] and not r['model_acceptance']
