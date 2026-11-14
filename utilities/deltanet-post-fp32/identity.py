"""Compare shared post builds with immutable, previously audited causal binaries."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load("_shared_post_builder", HERE / "build.py")
exporter = load("_shared_post_exporter", builder.ROOT / "open_kernels/export_qwen36_kernels.py")
digest = builder.digest


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError("changed archived artifact: " + str(path))


def archived_build(reference, run):
    """Historical source paths may change; accepted binary/capture bytes may not."""
    record = json.loads((reference / "build.json").read_text())
    result = json.loads((run / "report.json").read_text())
    proof = json.loads((run / "audit.json").read_text())
    if not (record["complete"] and result["complete"] and result["passed"]
            and proof["complete"] and proof["passed"]
            and proof["report_sha256"] == digest(run / "report.json")
            and Path(result["build"]).resolve() == reference.resolve()
            and result["mode"] == record["mode"]
            and result["inputs"].get(str(reference / "build.json")) == digest(reference / "build.json")):
        raise ValueError("reference build lacks a bound passing causal audit")
    names = {f"{region}_{p}.bin" for p in range(243) for region in ("act", "state", "xres")}
    if set(result["artifacts"]) != names:
        raise ValueError("incomplete historical capture inventory")
    if set(record["artifacts"]) != {"final.xclbin", "insts.bin"}:
        raise ValueError("incomplete historical binary inventory")
    hashes = {str(reference / n): h for n, h in record["artifacts"].items()}
    hashes.update({str(run / n): h for n, h in result["artifacts"].items()})
    hashes.update({str(p): digest(p) for p in
                   (reference / "build.json", run / "report.json", run / "audit.json")})
    verify_hashes(hashes)
    return record, hashes


def compare(build, reference, run):
    current = builder.verify(build)
    old, hashes = archived_build(reference, run)
    if current["mode"] != old["mode"] or current["geometry"] != old["geometry"]:
        raise ValueError("different post mode or scratch/state geometry")
    resources = json.loads((build / "resources.json").read_text())
    if not resources["passed"] or resources["build"] != current:
        raise ValueError("missing current resource audit")
    verify_hashes(resources["evidence_sha256"])
    exact = (build / "insts.bin").read_bytes() == (reference / "insts.bin").read_bytes()
    equivalent, detail = exporter.xclbin_equivalent(
        (build / "final.xclbin").read_bytes(), (reference / "final.xclbin").read_bytes())
    if not exact or not equivalent:
        raise ValueError(f"binary identity failed: instructions={exact}; {detail}")
    paths = [Path(__file__), Path(builder.__file__), Path(exporter.__file__),
             build / "build.json", build / "resources.json", build / "insts.bin", build / "final.xclbin"]
    hashes.update({str(p): digest(p) for p in paths})
    return dict(passed=True, model_acceptance=False, fresh_numerical_run=False,
                build=str(build), reference=str(reference), reference_run=str(run),
                mode=current["mode"], instructions_exact=exact, xclbin_equivalent=equivalent,
                xclbin_detail=detail, inputs=hashes,
                max_text=max(resources["core_text"].values()),
                max_l1_buffer_end=max(resources["l1_buffer_end"].values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("build", "reference", "reference-run", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    result = dict(passed=False, model_acceptance=False, fresh_numerical_run=False)
    # Refuse to overwrite previous evidence, including a failed attempt.
    with args.out.open("x") as output:
        try:
            result = compare(args.build.resolve(), args.reference.resolve(), args.reference_run.resolve())
        except Exception as error:
            result["error"] = str(error)
            raise
        finally:
            output.write(json.dumps(result, indent=2) + "\n")
