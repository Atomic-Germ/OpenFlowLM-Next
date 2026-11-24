#!/usr/bin/env python3
"""Wrap the NPU-free model/family checks and report the production-bar status.

    python utilities/verify-family.py --gguf model.gguf
    python utilities/verify-family.py --model-dir ~/.config/oflm/models/MyModel-NPU2
    python utilities/verify-family.py --model-dir DIR --hf-shard model-00001-of-00003.safetensors \
        --mmproj mmproj.gguf --safetensors org/Model
    oflm dev verify-family --gguf model.gguf --json

Adding a family (open-family-addition) ends with a set of checks chosen by hand:
`container_vs_hf.py` before any chat test, `check-mmproj.py` for a VL tower,
`gguf-embed-check.py` for an embedding GGUF, and the AGENTS.md production bar.
This script runs the ones that need no NPU, marks each PASS / SKIP / FAIL, and
prints the production-bar items as `needs-measurement` unless an evidence file
says otherwise. It never *claims* a measurement it did not take: the hardware
numerics (`ref-dump` + `cmp_layers.py`) stay an explicit `--npu` opt-in, and the
product bar items stay red until a dev supplies evidence.

An evidence file is a JSON object mapping bar item ids (see BAR_ITEMS) to
`{"status": "pass", "note": "...", "measured": "..."}`. It is a claim by the
dev, not a verification -- the point is to make the claim explicit and visible
in the PR, not to hide it in prose.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The AGENTS.md production bar, as machine-readable ids. The text is kept
# verbatim so the report can be pasted into a PR.
BAR_ITEMS = [
    ("prefill_block_route", "Prefill uses the block route (state any exception)."),
    ("near_optimal", "Decode and prefill are near optimal; profile and name what is left."),
    ("context_length", "No unusual context-length limits; test up to the real context."),
    ("all_on_npu", "Everything runs on the NPU (usually); name host stages and cost."),
    ("speculative_decoding", "Speculative decoding explored: ship one or reject with measured reason."),
    ("add_and_serve", "`oflm add` and `oflm serve` run the model, not only a CLI harness."),
    ("quantized", "Ships quantized (usually q4); prefer Unsloth GGUFs, else state why."),
    ("benchmarks", "Benchmarks in docs/docs/benchmarks/ and a README 'Recommended models' row."),
    ("features", "Vision/audio/MTP/reasoning/tools exposed; name each omission and why."),
    ("registry_entry", "src/model_list.json entry exists with real quant, family and context length."),
]


class Result:
    def __init__(self, name, status, detail=""):
        self.name, self.status, self.detail = name, status, detail

    def as_dict(self):
        return {"name": self.name, "status": self.status, "detail": self.detail}


def run_helper(script, args, timeout=600):
    """Run a utility script; return (ok, combined output)."""
    path = ROOT / script
    if not path.is_file():
        return False, f"helper script not found: {script}"
    try:
        p = subprocess.run([sys.executable, str(path), *args], cwd=ROOT,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    out = (p.stdout + p.stderr).strip()
    return p.returncode == 0, out


def check_install(model_dir):
    """The NPU-free install invariants: files, kernel link, spec, patch."""
    problems = []
    for f in ("config.json", "tokenizer.json", "model.gguf"):
        if not (model_dir / f).is_file():
            problems.append(f"missing {f}")
    link = model_dir / "open_kernels"
    if not (link.exists() or link.is_symlink()):
        problems.append("no open_kernels link")
    # The spec the engine sees must derive and must not be a stub.
    sys.path.insert(0, str(ROOT / "utilities" / "oflm-add"))
    try:
        import oflm_add
        spec = oflm_add.model_spec(str(model_dir))
        if spec is None:
            problems.append("no ModelSpec can be derived (no recipes or unknown shape)")
        else:
            if spec.rope_scaling is None and getattr(spec, "family", "") == "llama3":
                problems.append("llama3 spec has no rope_scaling (base-model patch not applied?)")
            if spec.real_vocab == spec.vocab and spec.vocab % 128 == 0:
                problems.append("real_vocab == padded vocab (logits mask missing?)")
    except Exception as exc:                          # noqa: BLE001
        problems.append(f"spec derivation failed: {exc}")
    return problems


def check_container(model_dir, hf_shard):
    ok, out = run_helper("open_kernels/model/container_vs_hf.py",
                         ["--model-dir", str(model_dir), "--hf-shard", str(hf_shard)])
    return ok, out


def check_mmproj(mmproj, safetensors):
    return run_helper("utilities/check-mmproj.py",
                      ["--mmproj", str(mmproj), "--safetensors", str(safetensors)])


def check_embed(gguf, vocab):
    args = [str(gguf)]
    if vocab:
        args += ["--vocab", str(vocab)]
    return run_helper("utilities/gguf-embed-check.py", args)


def check_npu(model_dir, evidence):
    """The hardware numerics path: ref-dump + cmp_layers, opt-in and best-effort."""
    if not evidence or "logits_ref" not in evidence or "ids" not in evidence:
        return False, "--npu needs an evidence file with 'logits_ref' and 'ids'"
    ref = Path(evidence["logits_ref"])
    ids = evidence["ids"]
    dump = ROOT / "utilities" / "ref-dump" / "dump_ref"
    if not shutil.which("dump_ref") and not dump.is_file():
        return False, "ref-dump not built (see utilities/ref-dump/build.sh)"
    # Without a prebuilt engine CLI and kernel link there is nothing to compare.
    # Kept as a SKIP (never a PASS): a measurement the script did not take is
    # not a measurement. Run cmp_layers.py directly and record the result in
    # the evidence file instead.
    return False, "--npu numerics require a built open_qwen36_cli; run cmp_layers.py directly and record the result in the evidence file"


def bar_status(evidence):
    ev = evidence or {}
    out = []
    for key, text in BAR_ITEMS:
        item = ev.get(key)
        if isinstance(item, dict) and str(item.get("status", "")).lower() in ("pass", "ok", "yes"):
            out.append({"id": key, "item": text, "status": "pass",
                        "note": str(item.get("note", ""))})
        else:
            out.append({"id": key, "item": text, "status": "needs-measurement",
                        "note": str((item or {}).get("note", ""))})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", help="an installed model directory to validate")
    ap.add_argument("--hf-shard", help="HF safetensors shard for container_vs_hf.py")
    ap.add_argument("--mmproj", help="mmproj GGUF for check-mmproj.py")
    ap.add_argument("--safetensors", help="HF repo the mmproj was converted from")
    ap.add_argument("--gguf", help="an embedding GGUF for gguf-embed-check.py")
    ap.add_argument("--vocab", help="authoritative vocab.txt for the embedding check")
    ap.add_argument("--evidence", help="JSON file of production-bar claims (see module docstring)")
    ap.add_argument("--npu", action="store_true", help="attempt the hardware numerics path")
    ap.add_argument("--json", action="store_true", help="emit a machine-readable report")
    args = ap.parse_args()

    evidence = None
    if args.evidence:
        try:
            evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"verify-family: cannot read evidence file: {exc}", file=sys.stderr)
            return 2

    checks = []
    if args.model_dir:
        problems = check_install(Path(args.model_dir))
        checks.append(Result("install", "FAIL" if problems else "PASS",
                             "; ".join(problems)))
        if args.hf_shard:
            ok, out = check_container(Path(args.model_dir), args.hf_shard)
            checks.append(Result("container_vs_hf", "PASS" if ok else "FAIL", out))
    if args.mmproj and args.safetensors:
        ok, out = check_mmproj(args.mmproj, args.safetensors)
        checks.append(Result("mmproj", "PASS" if ok else "FAIL", out))
    if args.gguf:
        ok, out = check_embed(args.gguf, args.vocab)
        checks.append(Result("gguf_embed", "PASS" if ok else "FAIL", out))
    if not checks:
        ap.error("nothing to check: pass --model-dir, --gguf, or --mmproj/--safetensors")
    if args.npu:
        ok, out = check_npu(args.model_dir, evidence)
        checks.append(Result("npu_numerics", "PASS" if ok else "SKIP", out))

    bar = bar_status(evidence)
    failures = [c for c in checks if c.status == "FAIL"]
    if args.json:
        print(json.dumps({"checks": [c.as_dict() for c in checks], "production_bar": bar,
                          "ok": not failures}, indent=2))
    else:
        for c in checks:
            print(f"[{c.status:4}] {c.name}" + (f": {c.detail}" if c.detail else ""))
        print("\nProduction bar (AGENTS.md):")
        for b in bar:
            suffix = f" -- {b['note']}" if b["note"] else ""
            print(f"  [{b['status']:17}] {b['id']}: {b['item']}{suffix}")
        if failures:
            print(f"\nverify-family: {len(failures)} check(s) FAILED")
        else:
            print("\nverify-family: all requested checks passed; "
                  "production-bar items still need measurements unless evidenced.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
