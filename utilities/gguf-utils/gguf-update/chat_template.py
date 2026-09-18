#!/usr/bin/env python3
"""Update tokenizer.chat_template in GGUF files in the current directory.

Run from a directory containing GGUF files. Rewrites each file with the new
template via gguf.scripts.gguf_new_metadata (ships with the gguf package).

  python chat_template.py --family qwen3.x            # bundled template
  python chat_template.py --jinja my_template.jinja   # custom template file
  python chat_template.py --family qwen3.x --file a.gguf b.gguf

Template families live in chat_templates/<family>/chat_template.jinja; a family
is considered supported only when its directory exists.
"""
from __future__ import annotations

import difflib
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

# Initial range-request size for remote GGUF headers. Tokenizer vocab can be
# tens of MB; bump if extraction fails on a particular model.
REMOTE_CHUNK_BYTES = 64 * 1024 * 1024


def list_gguf_files() -> list[Path]:
    return sorted(p for p in Path.cwd().glob("*.gguf") if p.is_file())


# ---- GGUF header parser (just enough to find tokenizer.chat_template) -----

# GGUFValueType enum (gguf/constants.py)
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32, _BOOL, _STRING, \
    _ARRAY, _UINT64, _INT64, _FLOAT64 = range(13)

_SCALAR_FMT = {
    _UINT8: ("<B", 1), _INT8: ("<b", 1),
    _UINT16: ("<H", 2), _INT16: ("<h", 2),
    _UINT32: ("<I", 4), _INT32: ("<i", 4),
    _FLOAT32: ("<f", 4), _BOOL: ("<?", 1),
    _UINT64: ("<Q", 8), _INT64: ("<q", 8), _FLOAT64: ("<d", 8),
}


def _read_gguf_string(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8")


def _read_gguf_value(f, vtype: int):
    if vtype == _STRING:
        return _read_gguf_string(f)
    if vtype == _ARRAY:
        (etype,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [_read_gguf_value(f, etype) for _ in range(n)]
    fmt, size = _SCALAR_FMT[vtype]
    return struct.unpack(fmt, f.read(size))[0]


def extract_template_from_gguf(path: Path) -> str:
    """Parse GGUF header from a (possibly truncated) file, return the value of
    tokenizer.chat_template. Works on partial downloads as long as the KV
    section fits inside the truncated portion."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"Not a GGUF file: {path}")
        f.read(4)   # version
        f.read(8)   # tensor count
        (kv_count,) = struct.unpack("<Q", f.read(8))

        for _ in range(kv_count):
            key = _read_gguf_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            value = _read_gguf_value(f, vtype)
            if key == "tokenizer.chat_template":
                if not isinstance(value, str):
                    raise TypeError(
                        f"tokenizer.chat_template is {type(value).__name__}, expected str. "
                        "Multi-template GGUFs are not supported by this script."
                    )
                return value
    raise KeyError(
        "tokenizer.chat_template not found. The KV section may extend past the "
        "downloaded chunk; try increasing REMOTE_CHUNK_BYTES."
    )


# ---- Template sources ------------------------------------------------------

# Bundled family templates live in chat_templates/<family>/chat_template.jinja,
# next to this script. Family support is gated on that directory existing: a
# missing directory means the family is not supported yet.
TEMPLATES_DIR = Path(__file__).parent / "chat_templates"


def available_families() -> list[str]:
    if not TEMPLATES_DIR.is_dir():
        return []
    return sorted(p.name for p in TEMPLATES_DIR.iterdir() if p.is_dir())


def load_family_template(family: str) -> Path:
    """Return the jinja path for a model family, or exit if unsupported.

    Compatibility is tested simply by the presence of chat_templates/<family>/.
    """
    family_dir = TEMPLATES_DIR / family
    if not family_dir.is_dir():
        fams = ", ".join(available_families()) or "none"
        sys.exit(
            f"No template for family '{family}': "
            f"chat_templates/{family}/ does not exist.\n"
            f"Available families: {fams}"
        )
    tmpl = family_dir / "chat_template.jinja"
    if not tmpl.is_file():
        sys.exit(f"Family '{family}' exists but {tmpl.name} is missing.")
    return tmpl


def load_custom_template(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_file():
        sys.exit(f"File not found: {p}")
    return p


# ---- Template validation ---------------------------------------------------

def validate_template(template: str) -> None:
    """Parse the template with Jinja2 and try a dry render. Aborts on failure."""
    try:
        import jinja2
    except ImportError:
        print("WARNING: jinja2 not available, skipping validation.", file=sys.stderr)
        return

    env = jinja2.Environment(
        trim_blocks=True, lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"],
    )
    try:
        compiled = env.from_string(template)
    except jinja2.TemplateSyntaxError as e:
        sys.exit(f"Jinja2 syntax error at line {e.lineno}: {e.message}")

    sample = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        "add_generation_prompt": True,
        "bos_token": "<bos>", "eos_token": "<eos>",
        "tools": None, "tool_choice": None,
    }
    try:
        rendered = compiled.render(**sample)
    except Exception as e:
        print(f"WARNING: template parsed but failed to render with sample messages: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        if input("Continue anyway? [y/N] ").strip().lower() not in ("y", "yes"):
            sys.exit("Aborted.")
        return
    print(f"Template validated (renders to {len(rendered)} chars on sample input).")


# ---- Diff display ----------------------------------------------------------

_USE_COLOUR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
_RESET = "\033[0m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"


def _paint(line: str) -> str:
    if not _USE_COLOUR:
        return line
    if line.startswith("+++") or line.startswith("---"):
        return f"{_BOLD}{line}{_RESET}"
    if line.startswith("@@"):
        return f"{_CYAN}{line}{_RESET}"
    if line.startswith("+"):
        return f"{_GREEN}{line}{_RESET}"
    if line.startswith("-"):
        return f"{_RED}{line}{_RESET}"
    return line


def get_current_template(path: Path) -> str | None:
    try:
        return extract_template_from_gguf(path)
    except KeyError:
        return None


def show_diff(name: str, current: str | None, new: str, max_lines: int = 500,
              suppress: bool = False) -> bool:
    """Print a unified diff. Returns True if there's a change, False if identical."""
    current_str = current if current is not None else ""
    if current_str == new:
        print(f"\n=== {name}: no change ===")
        return False
    if suppress:
        return True

    print(f"\n=== diff for {name} ===")
    if current is None:
        print("(no existing tokenizer.chat_template — adding new)")

    diff = list(difflib.unified_diff(
        current_str.splitlines(),
        new.splitlines(),
        fromfile=f"current: {name}",
        tofile="new",
        lineterm="",
    ))
    shown = diff[:max_lines]
    print("\n".join(_paint(line) for line in shown))
    if len(diff) > max_lines:
        print(f"... ({len(diff) - max_lines} more diff lines truncated) ...")
    return True


# ---- GGUF rewrite ----------------------------------------------------------

def free_bytes(p: Path) -> int:
    return shutil.disk_usage(p.parent).free


def update_gguf(input_path: Path, template_file: Path) -> None:
    out_path = input_path.with_suffix(input_path.suffix + ".new")
    needed = input_path.stat().st_size + 64 * 1024 * 1024
    if free_bytes(input_path) < needed:
        sys.exit(
            f"Not enough free space on {input_path.parent} for {input_path.name} "
            f"(need ~{needed // 1024**3} GB)."
        )
    print(f"\n→ {input_path.name}: writing {out_path.name}")
    cmd = [
        sys.executable, "-m", "gguf.scripts.gguf_new_metadata",
        "--chat-template-file", str(template_file),
        "--force",
        str(input_path), str(out_path),
    ]
    subprocess.run(cmd, check=True)
    print(f"→ {input_path.name}: replacing original")
    out_path.replace(input_path)


def main() -> int:
    import argparse
    import importlib.util

    if importlib.util.find_spec("gguf") is None:
        sys.exit(
            "gguf package not available. Run via './venv/bin/python chat_template.py' "
            "or install gguf into the active environment."
        )

    parser = argparse.ArgumentParser(
        description="Update tokenizer.chat_template in GGUF files in the "
                    "current directory.",
    )
    parser.add_argument(
        "--family", metavar="FAMILY",
        help="Bundled template for a model family, selected from "
             "chat_templates/<FAMILY>/chat_template.jinja. Support is gated on "
             "that directory existing.",
    )
    parser.add_argument(
        "--jinja", metavar="PATH",
        help="Path to a custom jinja template file (alternative to --family).",
    )
    parser.add_argument(
        "--file", metavar="GGUF", action="append", default=[],
        help="Specific .gguf file to update; may be repeated. Defaults to all "
             "*.gguf in the current directory.",
    )
    parser.add_argument(
        "--no-diff", action="store_true",
        help="Show only a summary instead of the full before/after diff.",
    )
    args = parser.parse_args()

    if not args.family and not args.jinja:
        parser.error("one of --family or --jinja is required.")
    if args.family and args.jinja:
        parser.error("--family and --jinja are mutually exclusive.")

    if args.family:
        tmpl_path = load_family_template(args.family)
    else:
        tmpl_path = load_custom_template(args.jinja)
    template = tmpl_path.read_text(encoding="utf-8")

    if not template.strip():
        sys.exit("Empty template, aborting.")

    validate_template(template)

    if args.file:
        selected = [Path(f).expanduser() for f in args.file]
        missing = [orig for orig, p in zip(args.file, selected) if not p.is_file()]
        if missing:
            sys.exit("File not found: " + ", ".join(missing))
    else:
        selected = list_gguf_files()
    if not selected:
        sys.exit("No .gguf files in current directory (use --file to specify).")

    origin = f"--family {args.family}" if args.family else f"--jinja {args.jinja}"
    print(f"\nTemplate: {tmpl_path}")
    print(f"Source:   {origin}")
    print(f"Files:    {len(selected)} selected")

    preview = template[:300].replace("\n", "\\n")
    print(f"\n--- Template preview ({len(template)} chars) ---")
    print(preview + ("..." if len(template) > 300 else ""))
    print("---")

    to_update: list[Path] = []
    for f in selected:
        current = get_current_template(f)
        if show_diff(f.name, current, template, suppress=bool(args.no_diff)):
            to_update.append(f)

    if not to_update:
        print("\nAll selected files already have this template, nothing to do.")
        return 0

    fd, tmpl_name = tempfile.mkstemp(suffix=".jinja", prefix="chat_template_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(template)
    template_file = Path(tmpl_name)

    try:
        for f in to_update:
            update_gguf(f, template_file)
        print("\nDone.")
    finally:
        template_file.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
