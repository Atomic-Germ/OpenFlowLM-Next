"""A binary comparison cannot borrow an unrelated or modified historical audit."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def module():
    spec = importlib.util.spec_from_file_location("post_identity", ROOT / "utilities/deltanet-post-fp32/identity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("fault", [None, "binary", "capture", "report", "unbound", "missing", "failed"])
def test_archive_binds_audited_build_and_all_capture_bytes(tmp_path, fault):
    m = module()
    reference, run = tmp_path / "build", tmp_path / "run"
    reference.mkdir(); run.mkdir()
    artifacts = {}
    for name in ("final.xclbin", "insts.bin"):
        (reference / name).write_bytes(b"binary")
        artifacts[name] = m.digest(reference / name)
    record = dict(complete=True, mode="fp32", artifacts=artifacts)
    (reference / "build.json").write_text(json.dumps(record))
    captures = {}
    for p in range(243):
        for region in ("act", "state", "xres"):
            name = f"{region}_{p}.bin"
            (run / name).write_bytes(b"capture")
            captures[name] = m.digest(run / name)
    result = dict(complete=True, passed=True, mode="fp32", build=str(reference), artifacts=captures,
                  inputs={str(reference / "build.json"): m.digest(reference / "build.json")})
    if fault == "unbound":
        result["inputs"] = {}
    if fault == "missing":
        result["artifacts"].pop("act_242.bin")
    if fault == "failed":
        result["passed"] = False
    (run / "report.json").write_text(json.dumps(result))
    (run / "audit.json").write_text(json.dumps(dict(complete=True, passed=True,
        report_sha256=m.digest(run / "report.json"))))
    if fault in ("binary", "capture", "report"):
        target = {"binary": reference / "insts.bin", "capture": run / "state_242.bin",
                  "report": run / "report.json"}[fault]
        with target.open("ab") as f:
            f.write(b" ")
    if fault:
        with pytest.raises(ValueError):
            m.archived_build(reference, run)
    else:
        assert m.archived_build(reference, run)[0] == record
