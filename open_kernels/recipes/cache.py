"""The build key: what a kernel set was built from. Any change to it rebuilds.

Covers the recipe package's sources, every kernel source the family's designs
include (the family module's KERNEL_SOURCES), the ModelSpec (without its
informational `extra`) and the quant format, plus any PROBE environment
variable the family exposes (`probe_env()`) -- those change the compiled kernel
and nothing else in the key can see them, so without this a probe build and a
real one share a key and the second is skipped. The KV / ptab capacity is NOT in
it: in this tree every position-dependent word of the attention stream is
patched per token, so the capacity is a runtime buffer size, not a kernel
input.

Traces: OPEN-BUILD-CACHE (specs/open-engine/spec.md).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .families import for_spec
from .spec import ModelSpec

ROOT = Path(__file__).resolve().parents[1]      # open_kernels/


def source_files(spec: ModelSpec, root: Path = ROOT) -> list[Path]:
    files = sorted((root / "recipes").glob("*.py"))
    for pat in for_spec(spec).KERNEL_SOURCES:
        files += sorted(root.glob(pat))
    # generated TUs are outputs of gen_kernels.py, not inputs; the generator is already included
    seen, out = set(), []
    for f in files:
        if f.is_file() and f not in seen:
            seen.add(f)
            out.append(f)
    return out


def build_key(spec: ModelSpec, root: Path = ROOT) -> str:
    h = hashlib.sha256()
    for f in source_files(spec, root):
        h.update(f.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    d = spec.to_dict()
    d.pop("extra", None)
    h.update(json.dumps(d, sort_keys=True).encode())
    h.update(b"\0quant=" + spec.quant.encode())
    # Only when something is set, so an ordinary build's key is untouched.
    probes = getattr(for_spec(spec), "probe_env", dict)()
    if probes:
        h.update(b"\0probes=" + json.dumps(probes, sort_keys=True).encode())
    return "sha256:" + h.hexdigest()
