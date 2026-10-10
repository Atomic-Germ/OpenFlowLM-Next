"""Regenerate fixtures/ from a build tree: python specs/viz/tests/make_fixtures.py DESIGNS_DIR TWO_CONTEXT_SET ONE_CONTEXT_SET"""
import gzip
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "open_kernels"))
from viz.topology import extract_prj  # noqa: E402

# Placed and unplaced logical tiles, unplaced shims only, mem tiles with links: every form the parser maps.
MLIR = {"lx0": "layer_x/build_lx0_q663fca7b", "lx1": "layer_x/build_lx1_q663fca7b", "ln": "ln/build",
        "lm_head_q8": "lm_head_q8/build_full", "gemm_n2048_k512": "gemm_q4_prefill/build_n2048_k512_t256"}


def gz(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f, open(dst, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as g:
        shutil.copyfileobj(f, g)


def gz_json(obj, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as g:
        g.write(json.dumps(obj, separators=(",", ":")).encode("utf-8"))


def main(designs: Path, two: Path, one: Path) -> None:
    out = HERE / "fixtures"
    for name, b in MLIR.items():
        for f in ("aie.mlir", "input_with_addresses.mlir"):
            gz(designs / b / "final.prj" / f, out / "mlir" / name / (f + ".gz"))
    shutil.copyfile(designs / MLIR["lx0"] / "insts.bin", out / "mlir" / "lx0" / "insts.bin")
    for tag, s in (("two", two), ("one", one)):
        m = json.loads((s / "manifest.json").read_text(encoding="utf-8"))
        runs = {st["kernel"] for lt in m["layer_types"].values() for st in lt["program"] if st["op"] == "run"}
        runs |= {st["kernel"] for st in m["tail"] if st["op"] == "run"}
        keep = {k: m["kernels"][k] for k in runs}
        m["kernels"] = keep
        m["builds"] = {d: m["builds"][d] for d in {Path(k["insts"]).parent.as_posix() for k in keep.values()}}
        m["contexts"] = {c: m["contexts"][c] for c in {k["context"] for k in keep.values()}}
        gz_json(m, out / tag / "manifest.json.gz")
        for k, kd in keep.items():
            gz_json(extract_prj(designs / m["builds"][Path(kd["insts"]).parent.as_posix()]["build_dir"] / "final.prj"),
                    out / tag / "topo" / f"{k}.json.gz")


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:4]))
