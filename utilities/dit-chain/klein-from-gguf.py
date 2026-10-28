#!/usr/bin/env python3
"""klein-from-gguf: a FLUX.2 [klein] GGUF -> a bundle `oflm image` can run.

One command for what used to be four:

    python utilities/dit-chain/klein-from-gguf.py --repo unsloth/FLUX.2-klein-4B-GGUF \
        --tag klein-gguf:512 --image "a red fox in fresh snow"

It

  1. takes the DiT GGUF from --gguf, or picks one from --repo and downloads it (Q4_1 first:
     the quant this tree runs everywhere; BF16/F16/Q8_0 also decode);
  2. resolves the checkpoint that supplies the text encoder and the VAE -- the GGUF repo's
     README names it (`base_model`), the same climb the text models use;
  3. builds the bundle (export_bundle.py: the DiT's weights packed from the GGUF, the text
     encoder's from the checkpoint, the schedules, the VAE);
  4. builds the kernel set for the same resolutions if it is not already built, and installs
     it beside the bundle (the layout hash must match, and both sides compute it);
  5. registers the tag so `oflm image <tag> ...` works;
  6. optionally generates an image, which is the only check that says the weights are right.

Steps 3 and 4 are cached: a second run is seconds, a first run is ~25 minutes (the kernel
build dominates). This is the local path until `q4nx-build --open-diffusion` takes a GGUF
repo and `oflm add` knows what a bundle is.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DIT_CACHE = Path.home() / ".cache" / "oflm" / "dit-gguf"

# The quants gguf_weights decodes, best first. Q4_1 is what the rest of the tree runs and
# what these repos publish; BF16/F16 are lossless and much larger.
QUANTS = ["Q4_1", "BF16", "F16", "Q8_0"]


def log(msg):
    print(msg, flush=True)


def pick_gguf(repo: str) -> str:
    """(repo-relative) name of the DiT GGUF to take from a repo, Q4_1 first."""
    from huggingface_hub import list_repo_files

    files = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
    dit = [f for f in files if "mmproj" not in f.lower() and "mtp" not in f.lower()]
    for q in QUANTS:
        cand = sorted(f for f in dit if q.lower() in f.lower())
        if cand:
            return cand[0]
    raise SystemExit(f"{repo}: no DiT GGUF in {QUANTS} (have: {', '.join(sorted(files)) or 'none'})")


def ensure_gguf(repo: str, name: str) -> Path:
    target = DIT_CACHE / f"{repo.replace('/', '--')}--{Path(name).name}"
    if target.is_file():
        log(f"DiT GGUF: {target} (cached)")
        return target
    from huggingface_hub import hf_hub_download

    target.parent.mkdir(parents=True, exist_ok=True)
    log(f"DiT GGUF: {name} <- {repo}")
    got = hf_hub_download(repo_id=repo, filename=name, local_dir=str(target.parent))
    got = Path(got)
    if got != target:
        got.replace(target)
    return target


def run(cmd, what):
    log(f"-- {what}")
    rc = subprocess.run([str(c) for c in cmd]).returncode
    if rc:
        raise SystemExit(f"{what} failed (exit {rc})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=None, help="the GGUF repo (its README names the checkpoint)")
    ap.add_argument("--gguf", default=None, help="a DiT GGUF already on disk (skips the download)")
    ap.add_argument("--resolutions", default="512", help="square sizes the bundle carries (512,1024)")
    ap.add_argument("--out", default=str(Path.home() / "klein-bundle"), help="the bundle directory")
    ap.add_argument("--pack-cache", default=str(Path.home() / "klein-pack"),
                    help="keep the per-weight packed files here (reused next run)")
    ap.add_argument("--tag", default=None, help="register as this tag (klein-gguf:512); skip to not register")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-kernels", action="store_true",
                    help="do not build/install the kernel set (it is already beside the bundle)")
    ap.add_argument("--image", default=None, help="generate this prompt afterwards (the real check)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image-out", default=None, help="default ./klein-<tag>-<seed>.png")
    ap.add_argument("--oflm", default="oflm", help="the oflm binary (default: from PATH)")
    a = ap.parse_args(argv)

    if not a.gguf and not a.repo:
        ap.error("give --gguf <file> or --repo <owner/name>")
    resolutions = [int(r) for r in a.resolutions.split(",")]

    # 1. the DiT
    repo = a.repo
    if a.gguf:
        gguf = Path(a.gguf).expanduser()
        if not gguf.is_file():
            ap.error(f"{gguf} is not a file")
        if not repo:
            # the checkpoint cannot be resolved without a repo; say so before the 9-minute build
            ap.error("--gguf needs --repo too: the text encoder and VAE are in the checkpoint "
                     "its README names (no GGUF of either exists)")
    else:
        name = pick_gguf(repo)
        gguf = ensure_gguf(repo, name)

    # 2+3. the bundle (the checkpoint is resolved inside export_bundle from --gguf-repo)
    out = Path(a.out).expanduser().resolve()
    run([sys.executable, HERE / "export_bundle.py",
         "--gguf-repo", repo, "--transformer-gguf", gguf,
         "--resolutions", ",".join(str(r) for r in resolutions),
         "--out", out, "--pack-cache", Path(a.pack_cache).expanduser(), "--jobs", a.jobs],
        f"bundle -> {out}")

    # 4. the kernels, for the same resolutions (the layout hash ties them to this bundle)
    layout = json.loads((out / "config.json").read_text())["layout"]
    installed = out / "open_kernels" / "diffusion_kernels.json"
    if a.skip_kernels:
        log("-- kernels: skipped (--skip-kernels)")
    elif installed.is_file() and json.loads(installed.read_text()).get("layout") == layout:
        log(f"-- kernels: already beside the bundle (layout {layout})")
    else:
        built = REPO / "src" / "xclbins" / "FLUX.2-klein-4B-NPU2" / "open_kernels"
        run([sys.executable, REPO / "open_kernels" / "export_dit_kernels.py",
             "--resolutions", ",".join(str(r) for r in resolutions), "--out", built], "kernel build")
        run([sys.executable, REPO / "open_kernels" / "export_dit_kernels.py",
             "--resolutions", ",".join(str(r) for r in resolutions),
             "--install", out / "open_kernels", "--out", built], "kernel install")

    # 5. the tag
    if a.tag:
        run([sys.executable, HERE / "register-bundle.py", out, "--tag", a.tag], f"register {a.tag}")

    # 6. the only check that says the weights are right
    if a.image:
        if not a.tag:
            ap.error("--image needs --tag (oflm image takes a tag)")
        png = Path(a.image_out).expanduser() if a.image_out else Path.cwd() / f"klein-{a.tag.replace(':', '-')}-{a.seed}.png"
        run([a.oflm, "image", a.tag, a.image, "--size", a.size, "--seed", a.seed, "--out", png],
            f"image -> {png}")
        log(f"\nopen {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
