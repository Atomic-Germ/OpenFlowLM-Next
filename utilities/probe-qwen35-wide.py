#!/usr/bin/env python3
"""Compile-only resource probe of wide Qwen3.5; never exports a model manifest.

Run with ironvenv/bin/python. The FFN is deliberately synthetic until segmented
FFN exists. Logs/spec/toolchain metadata remain in the selected build directory.
"""
import argparse
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "open_kernels"))
from recipes.spec import ModelSpec


def isolate_glue(source):
    """Compile the actual lx glue worker/schedule without unrelated norm/GEMV workers.

    This is diagnostic AST extraction, not another production kernel family.
    Keep the same five BO types, glue placement, functions, FIFOs and buffers.
    """
    tree = ast.parse(source)
    design = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "lx")
    body = []
    for node in design.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("main_body", "post_body", "dense_sequence", "sequence"):
            continue
        if isinstance(node, ast.For):  # main-core Worker construction
            continue
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "workers" for t in node.targets):
            body.extend(ast.parse("workers = []").body)
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "workers.append":
            if ast.unparse(node.value.args[0].args[0]) != "glue_body":
                continue
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "rt" for t in node.targets):
            body.extend(ast.parse('''
def sequence(a_pool, c_xres, a_consts, a_state, a_act, side_p, gact_p, gout_c, xn_p):
    ps = Pipeline(3)
    wide_side_sequence(ps, side_p, xn_p, a_consts, a_act)
    pipe = Pipeline(3)
    for tt in range(NT):
        pipe.drain(gout_c, a_state, rows3(tt))
        if tt >= KEY_TILES:
            pipe.drain(gout_c, a_act, bt(A_BYTES, A_VEC + (tt - KEY_TILES) * D.HEADS_PER_TILE * D.RECORD_BYTES,
                                       D.HEADS_PER_TILE * D.RECORD_BYTES))
        pipe.fill(gact_p, a_act, bt(A_BYTES, A_QKV + tt * TILE * 4, TILE * 4))
        pipe.fill(gact_p, a_state, rows3(tt))
    pipe.finish()
    ps.finish()
rt = Runtime(sequence, [pool_ty, xres_ty, consts_ty, state_ty, act_ty,
                       of_side.prod(tile=Tile(2, 0)), of_gact.prod(tile=Tile(3, 0)),
                       of_gout.cons(tile=Tile(2, 0)), of_xn_side.prod(tile=Tile(5, 0))])
''').body)
            continue
        body.append(node)
    design.body = body
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ffn", type=int, default=8192, help="synthetic FFN width (default 8192)")
    p.add_argument("--scope", choices=("layer", "glue", "projection"), default="layer")
    p.add_argument("--projection-k", type=int, default=5120, help="isolated Q4 projection width")
    p.add_argument("--out", type=Path, default=ROOT / "open_kernels/designs/layer_x/build_wide_probe")
    args = p.parse_args()
    out = args.out.resolve()
    if args.scope == "glue":
        out = out / "glue"
    elif args.scope == "projection":
        out = out / f"projection_k{args.projection_k}"
    out.mkdir(parents=True, exist_ok=True)
    fixture = ROOT / "specs/open-engine/tests/fixtures/config_qwen38_27b.json"
    spec = ModelSpec.from_hf_config(json.loads(fixture.read_text())).to_dict()
    spec.update(intermediate=args.ffn, quant="q4_1")
    spec["extra"] = {"probe": "synthetic FFN; NOT a validated 27B kernel"}
    spec_path = out / "probe-spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    env = os.environ.copy()
    env.update(OPEN_KERNELS_SPEC=str(spec_path), OPEN_KERNELS_UNVALIDATED="1",
               OPEN_KERNELS_WIDE_GLUE_PROBE="1")
    env["PATH"] = "/opt/xilinx/xrt/bin:" + env["PATH"]
    env["XILINX_XRT"] = "/opt/xilinx/xrt"
    env["LD_LIBRARY_PATH"] = "/opt/xilinx/xrt/lib:" + env.get("LD_LIBRARY_PATH", "")
    layer_dir = ROOT / "open_kernels/designs/layer_x"
    env["PYTHONPATH"] = str(layer_dir) + os.pathsep + env.get("PYTHONPATH", "")
    design_path = layer_dir / "lx.py"
    if args.scope == "projection":
        design_path = layer_dir / "projection_probe.py"
        env["PROBE_K"] = str(args.projection_k)
    if args.scope == "glue":
        source = design_path.read_text()
        # HERE must continue to name the production source directory, not the
        # output directory containing this generated diagnostic module.
        source = source.replace("HERE = Path(__file__).parent", f"HERE = Path({str(layer_dir)!r})")
        design_path = out / "glue_probe.py"
        design_path.write_text(isolate_glue(source))
    metadata = {"python": sys.version, "ffn": args.ffn, "scope": args.scope, "hardware_validated": False,
                "packages": {n: importlib.metadata.version(n) for n in ("mlir-aie", "llvm-aie", "numpy")}}
    if args.scope == "projection":
        metadata.update(projection_k=args.projection_k, projection_n=1024, weight_format="q4_1")
    (out / "probe-toolchain.json").write_text(json.dumps(metadata, indent=2) + "\n")
    commands = [
        [sys.executable, str(ROOT / "open_kernels/designs/layer_x/gen_kernels.py")],
        [sys.executable, str(ROOT / "open_kernels/build_design.py"),
         str(design_path), str(out)],
    ]
    for i, command in enumerate(commands):
        log = out / f"{i}-build.log"
        print(f"Running {command}; log: {log}", flush=True)
        with log.open("w") as f:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text()[-12000:])
            return result.returncode
    print("Compile-only probe succeeded; no hardware validation or model export.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
