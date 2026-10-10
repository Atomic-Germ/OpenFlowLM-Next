"""python -m viz SET_DIR [--designs DIR]: write a kernel set's topology.json files and viz.json."""
import argparse
import json
import sys
from pathlib import Path

from .export import DESIGNS, write_viz

ap = argparse.ArgumentParser(prog="python -m viz")
ap.add_argument("set_dir", type=Path)
ap.add_argument("--designs", type=Path, default=DESIGNS, help="where the manifest's build_dirs live")
a = ap.parse_args()
try:
    out = write_viz(a.set_dir, a.designs)
except ValueError as e:
    sys.exit(f"[viz] {e}")
v = json.loads(out.read_text(encoding="utf-8"))
d = v["decode"]
print(f"-> {out}  ({out.stat().st_size // 1024} KB: {len(v['kernels'])} kernels, "
      f"{sum(e['kind'] == 'dispatch' for e in d['events'])} decode dispatches, modelled step {d['total_us'] / 1000:.1f} ms)")
