#!/usr/bin/env python3
"""Try different runtime families for an already-installed OFLM model.

Useful when a converted model loads under the wrong engine or asserts during
weight loading. The script only touches the user registry and the xclbins
symlink; it never re-downloads weights.

Example:
    python3 utilities/try-model-families.py huihui-qwythos:9b \
        --families qwen3.5,qwen3.5-omni,qwen3.6-moe,qwen3vl,qwen3 \
        --timeout 20
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_FAMILIES = [
    "qwen3.5",
    "qwen3.5-omni",
    "qwen3.6-moe",
    "qwen3vl",
    "qwen3",
    "qwen2vl",
    "qwen2",
    "llama3",
    "gemma4e",
    "gemma4-12b",
    "gemma3",
    "gemma3-text",
    "gpt-oss",
    "nanbeige",
    "phi4",
    "lfm2",
    "lfm2.5-tk",
    "deepseek-r1",
    "deepseek-r1-0528",
]

SYSTEM_LIST_CANDIDATES = [
    "/opt/openflowlm/share/oflm/model_list.json",
    "/usr/share/oflm/model_list.json",
    "/usr/local/share/oflm/model_list.json",
]


def log(msg):
    print(msg, file=sys.stderr)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def find_system_model_list():
    candidates = []
    for exe in (os.environ.get("OFLM_EXECUTABLE"), shutil.which("oflm"), shutil.which("flm")):
        if exe:
            d = Path(exe).parent
            candidates.append(d / "model_list.json")
            candidates.append((d / ".." / "share" / "oflm" / "model_list.json").resolve())
    candidates += [Path(p) for p in SYSTEM_LIST_CANDIDATES]
    for c in candidates:
        if c.is_file():
            return c
    raise SystemExit("Could not locate system model_list.json")


def find_system_xclbin_root():
    for exe in (os.environ.get("OFLM_EXECUTABLE"), shutil.which("oflm"), shutil.which("flm")):
        if exe:
            d = Path(exe).parent
            if (d / "xclbins").is_dir():
                return d / "xclbins"
            r = (d / ".." / "share" / "oflm" / "xclbins").resolve()
            if r.is_dir():
                return r
    for prefix in [Path("/opt/openflowlm/share/oflm"), Path("/usr/share/oflm"), Path("/usr/local/share/oflm")]:
        if (prefix / "xclbins").is_dir():
            return prefix / "xclbins"
    return None


def user_registry_path():
    env = os.environ.get("OFLM_CONFIG_PATH")
    if env:
        return Path(env)
    return Path.home() / ".config" / "oflm" / "model_list.json"


def user_xclbin_dir():
    # oflm-add deliberately ignores OFLM_XCLBIN_PATH for writes.
    return Path.home() / ".config" / "oflm" / "xclbins"


def official_entries_by_family(system_registry, family):
    out = []
    for bucket, sizes in system_registry.get("models", {}).items():
        for sz, info in sizes.items():
            if (info.get("details") or {}).get("family") == family:
                out.append((bucket, sz, info))
    return out


def pick_official(system_registry, family, size_bytes):
    """Best official entry for (family, size)."""
    if not family:
        return None
    entries = official_entries_by_family(system_registry, family)
    if size_bytes:
        for bucket, sz, info in entries:
            if info.get("size") == size_bytes:
                return info
    if len(entries) == 1:
        return entries[0][2]
    if entries and size_bytes:
        best = min(entries, key=lambda e: abs(e[2].get("size", 0) - size_bytes))
        return best[2]
    return entries[0][2] if entries else None


def make_link(link, target):
    """Create a directory symlink, replacing an existing link."""
    link = Path(link)
    target = Path(target).resolve()
    if link.is_symlink() or link.exists():
        if link.is_symlink():
            link.unlink()
        else:
            shutil.rmtree(link)
    os.symlink(str(target), str(link), target_is_directory=True)


def run_load(tag, timeout):
    """Run `oflm run <tag>` with a non-interactive prompt and capture output."""
    try:
        proc = subprocess.run(
            ["oflm", "run", tag],
            input="\n",
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
        status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
        return status, combined
    except subprocess.TimeoutExpired as e:
        combined = (e.stdout or "") + (e.stderr or "")
        return "timeout", combined
    except FileNotFoundError:
        return "no oflm", ""


def first_error_line(output):
    """Extract the most informative error line."""
    lines = output.splitlines()
    for line in reversed(lines):
        if "error" in line.lower() or "assert" in line.lower() or "aborted" in line.lower():
            return line.strip()
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("["):
            return stripped
    return output[:200].strip()


def main():
    ap = argparse.ArgumentParser(
        description="Try different runtime families for an installed OFLM model."
    )
    ap.add_argument("tag", help="Model tag to test, e.g. huihui-qwythos:9b")
    ap.add_argument(
        "--families",
        help="Comma-separated list of families to try (default: built-in candidate list)",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=25,
        help="Seconds to wait for `oflm run <tag>` before killing it (default: 25)",
    )
    ap.add_argument(
        "--keep-best",
        action="store_true",
        help="Leave the registry/symlink set to the family with the cleanest load",
    )
    args = ap.parse_args()

    if ":" not in args.tag:
        raise SystemExit("tag must include a size, e.g. huihui-qwythos:9b")
    bucket, size_token = args.tag.split(":", 1)

    families = [f.strip() for f in args.families.split(",")] if args.families else list(DEFAULT_FAMILIES)

    reg_path = user_registry_path()
    if not reg_path.is_file():
        raise SystemExit(f"User registry not found: {reg_path}")
    original_registry = load_json(reg_path)
    registry = json.loads(json.dumps(original_registry))

    if bucket not in registry.get("models", {}) or size_token not in registry["models"][bucket]:
        raise SystemExit(f"Tag '{args.tag}' is not in user registry {reg_path}")
    entry = registry["models"][bucket][size_token]
    size_bytes = entry.get("size")
    dir_name = entry.get("name") or bucket

    system_list = find_system_model_list()
    system_registry = load_json(system_list)
    system_root = find_system_xclbin_root()
    user_root = user_xclbin_dir()
    link_path = user_root / dir_name

    original_family = entry.get("details", {}).get("family")
    original_link_target = None
    if link_path.is_symlink():
        try:
            original_link_target = link_path.readlink()
        except OSError:
            pass

    results = []
    best = None
    print(f"Testing {args.tag} (current family: {original_family or 'unknown'})")
    print(f"Model size: {size_bytes / 1e9:g}B" if size_bytes else "Model size: unknown")
    print(f"Families: {', '.join(families)}\n")

    for family in families:
        official = pick_official(system_registry, family, size_bytes)
        if official is None:
            results.append((family, "no official entry", "no matching official model in system list"))
            continue

        entry.setdefault("details", {})["family"] = family
        save_json(reg_path, registry)

        if system_root is not None:
            src = system_root / official.get("name", "")
            if src.is_dir():
                make_link(link_path, src)
            else:
                results.append((family, "no xclbin dir", f"official {official.get('name')} has no xclbins"))
                continue

        print(f"  -> trying family={family}, xclbins={official.get('name')}")
        status, output = run_load(args.tag, args.timeout)
        one_liner = first_error_line(output)
        results.append((family, status, one_liner))
        print(f"     {status}: {one_liner[:120]}")

        if status == "ok" and (best is None or best[1] != "ok"):
            best = (family, status, one_liner, official.get("name"))
        elif best is None and status == "timeout":
            # timeout often means it got past load and is waiting for input
            best = (family, status, one_liner, official.get("name"))

    if args.keep_best and best:
        family, status, _note, xclbin_name = best
        entry.setdefault("details", {})["family"] = family
        save_json(reg_path, registry)
        if system_root is not None:
            make_link(link_path, system_root / xclbin_name)
        print(f"\nKept best family: {family} ({status})")
    else:
        # restore original
        save_json(reg_path, original_registry)
        if original_link_target is not None and system_root is not None:
            try:
                make_link(link_path, original_link_target)
            except Exception as e:
                log(f"[WARN] could not restore xclbin link: {e}")
        elif original_link_target is None and link_path.is_symlink():
            link_path.unlink()
        print("\nRestored original family and xclbin link.")

    print("\nResults:")
    for family, status, note in results:
        mark = "✅" if status == "ok" else "⏬" if status == "timeout" else "❌"
        print(f"  {mark} {family:20s} -> {status}: {note[:120]}")


if __name__ == "__main__":
    main()
