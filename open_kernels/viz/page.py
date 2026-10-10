"""python -m viz.page SET_DIR -o out.html: the page `oflm viz` writes, for working on it without oflm."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[2] / "src" / "viz"
SLOTS = re.compile(r"/\*VIZ_(FONTS|DATA|EXPLAIN|MODEL)\*/")


def explain_text(root: Path = PAGE) -> str:
    return "\n\n".join(p.read_text(encoding="utf-8") for p in sorted((root / "explain").glob("*.md")))


def render(viz: str, explain: str, model: str = "{}", root: Path = PAGE) -> str:
    """Fill each slot of the template once; `</` is escaped so no payload can close its script tag."""
    fill = {"FONTS": (root / "fonts.css").read_text(encoding="utf-8"),
            "DATA": viz.replace("</", "<\\/"), "EXPLAIN": explain.replace("</", "<\\/"),
            "MODEL": model.replace("</", "<\\/")}
    return SLOTS.sub(lambda m: fill[m[1]], (root / "viz.html").read_text(encoding="utf-8"), count=4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m viz.page")
    ap.add_argument("set_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    model = json.dumps({"tag": a.tag} if a.tag else {})
    a.out.write_text(render((a.set_dir / "viz.json").read_text(encoding="utf-8"), explain_text(), model), encoding="utf-8")
    print(f"-> {a.out}")
