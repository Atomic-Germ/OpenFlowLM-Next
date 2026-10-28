#!/usr/bin/env python3
"""register-bundle: put a locally built diffusion bundle on `oflm image`'s tag list.

`oflm add` installs Q4NX containers and GGUF models; a FLUX.2 [klein] bundle is neither
(it is the schedule + weights + VAE the diffusion engine replays), and the shipped one
arrives through `oflm pull` from the model repo. While a bundle is being built or tested
locally, this registers a directory that is already on disk:

    python utilities/dit-chain/register-bundle.py ~/klein-bundle --tag klein-gguf:512
    oflm image klein-gguf:512 "a red fox in snow" --size 512

It links the directory into the models root (no copy) and writes the same registry entry
`oflm add` would, cloned from the official `flux2-klein` entry so the engine, sizes and
family are the shipped ones. The `--tag`'s size names the entry only; the bundle's own
`bundle.json` says which resolutions it carries.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "utilities" / "oflm-add"))
import oflm_add as A  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bundle", help="the directory export_bundle.py wrote (bundle.json inside)")
    ap.add_argument("--tag", required=True, help="family:size to serve it under (klein-gguf:512)")
    ap.add_argument("--system-list", default=None,
                    help="official model_list.json (default: the tree's src/model_list.json)")
    a = ap.parse_args(argv)

    src = Path(a.bundle).expanduser().resolve()
    man = src / "bundle.json"
    if not man.is_file():
        print(f"{src} has no bundle.json; is it an export_bundle.py output?", file=sys.stderr)
        return 1
    bundle = json.loads(man.read_text(encoding="utf-8"))
    family, _, size = a.tag.partition(":")
    if not family or not size:
        print("--tag wants family:size (klein-gguf:512)", file=sys.stderr)
        return 1

    system = Path(a.system_list) if a.system_list else HERE.parents[2] / "src" / "model_list.json"
    reg = json.loads(Path(system).read_text(encoding="utf-8"))
    template = (reg.get("models", {}).get("flux2-klein") or {}).get("4b")
    if template is None:
        print(f"{system} has no flux2-klein:4b entry to clone", file=sys.stderr)
        return 1

    root = A.models_root_dir(None)
    root.mkdir(parents=True, exist_ok=True)
    target = root / src.name
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.exists():
        print(f"{target} exists and is not a link; refusing to touch it", file=sys.stderr)
        return 1
    target.symlink_to(src)

    entry = json.loads(json.dumps(template))
    entry["name"] = src.name
    entry["files"] = sorted(p.name for p in src.iterdir() if p.is_file())
    entry["size"] = sum((src / f).stat().st_size for f in entry["files"])
    entry["url"] = entry["file_url"] = entry["ms_url"] = ""
    entry.setdefault("details", {})["family"] = "flux2-klein"
    # the sizes this bundle actually carries, straight from its own manifest
    resolutions = sorted(int(r) for r in bundle.get("resolutions", {}))
    if resolutions:
        entry["image_sizes"] = resolutions
        entry["image_edit_sizes"] = sorted(int(r) for r in bundle.get("edits", {}))

    tag = A.register(entry, A.user_settings_path(None), reg, force=True, tag=a.tag)
    print(f"registered {tag} -> {target} ({len(entry['files'])} files, "
          f"{entry['size'] / 2**30:.2f} GiB, sizes {resolutions})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
