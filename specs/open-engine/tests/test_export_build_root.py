"""Private kernel builds must retain locking and cache invalidation after merge."""
import io

import export_qwen36_kernels as exporter


def test_private_build_root_cache_and_source_invalidation(tmp_path, monkeypatch):
    designs = tmp_path / "open_kernels" / "designs"
    design = designs / "probe"
    design.mkdir(parents=True)
    source = design / "probe.py"
    source.write_text("# original source\n")
    out = tmp_path / "private" / "probe"
    monkeypatch.setattr(exporter, "DESIGNS", designs)
    monkeypatch.setattr(exporter, "HERE", designs.parent)
    monkeypatch.setattr(exporter, "LOCKS", designs / ".locks")
    monkeypatch.setattr(exporter, "fullest_core", lambda _: 0)
    calls = []

    class Compiler:
        returncode = 0

        def __init__(self, argv, **kwargs):
            calls.append(argv)
            out.mkdir(parents=True, exist_ok=True)
            for name in exporter.FILES:
                (out / name).write_bytes(b"fixture")
            self.stdout = io.BytesIO()

        def wait(self):
            return self.returncode

    monkeypatch.setattr(exporter.subprocess, "Popen", Compiler)
    sets = {"probe": {"design": "probe/probe.py", "build_dir": str(out), "env": {"N": "128"}}}
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    assert exporter.build("probe", sets, spec) == out
    assert exporter.build("probe", sets, spec) == out
    assert len(calls) == 1
    assert len(list(exporter.LOCKS.glob("*.lock"))) == 1
    source.write_text("# changed source\n")
    exporter.build("probe", sets, spec)
    assert len(calls) == 2
    sets["probe"]["env"]["N"] = "256"
    exporter.build("probe", sets, spec)
    assert len(calls) == 3
    (out / "insts.bin").unlink()
    exporter.build("probe", sets, spec)
    assert len(calls) == 4
