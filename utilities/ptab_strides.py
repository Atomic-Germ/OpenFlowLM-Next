"""For every recipes/specs/*.json manifest: each attnpos kernel's position-table row against the layout's (2026-10-17)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "open_kernels"
sys.path.insert(0, str(ROOT))

from recipes.load import load_spec  # noqa: E402
from recipes.manifest import manifest  # noqa: E402


def strides(m: dict) -> list[tuple[str, int, int]]:
    rows = {k: v["per_row"] for k, v in m.get("globals", {}).items() if isinstance(v, dict) and "per_row" in v}
    out = []
    for lt in m["layer_types"].values():
        for s in lt["program"]:
            if m["kernels"].get(s.get("kernel"), {}).get("patch") != "attnpos":
                continue
            for a in s.get("args", []):
                if a in rows:
                    out.append((s["kernel"], rows[a], m["layout"]["ptab_row"]))
    return out


def main() -> int:
    differ = 0
    for p in sorted((ROOT / "recipes" / "specs").glob("*.json")):
        try:
            m = manifest(load_spec(p))
        except Exception as e:  # noqa: BLE001
            print(f"{p.stem}: no manifest ({type(e).__name__}: {e})")
            continue
        for view, mm in (("top", m), ("gguf", m.get("gguf"))):
            if not mm:
                continue
            for k, row, lay in sorted(set(strides(mm))):
                tag = "same" if row == lay else "DIFFERS"
                differ += row != lay
                print(f"{tag:8} {p.stem} [{view}] {k}: table row {row}, layout ptab_row {lay}")
    print(f"{differ} kernel(s) whose table row differs from the layout's")
    return 0


if __name__ == "__main__":
    sys.exit(main())
