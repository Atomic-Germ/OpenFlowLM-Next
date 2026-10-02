"""The activation precision probe is confined to the full FFN validation path."""
from pathlib import Path
import subprocess
import sys


def test_activation_carry_rejects_non_ffn_before_build(tmp_path):
    root=Path(__file__).resolve().parents[3]
    p=subprocess.run([sys.executable,str(root/'utilities/probe-qwen35-wide.py'),
        '--scope','projection','--activation-carry','--out',str(tmp_path/'absent')],capture_output=True,text=True)
    assert p.returncode==2 and '--activation-carry requires --scope ffn' in p.stderr
    assert not (tmp_path/'absent').exists()
