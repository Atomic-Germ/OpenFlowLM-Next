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
sys.path.insert(0, str(HERE))
import gguf_source  # noqa: E402

def log(msg):
    print(msg, flush=True)


def _shipped_kernels(layout):
    """A kernel set whose layout matches, under any xclbins root (or beside a model).

    Same rule the engine and `oflm add` use: a bundle names its kernels by layout hash, and
    the set that ships with the app is the normal answer -- building one is the exception.
    """
    import os

    roots = [Path(p) for p in os.environ.get("OFLM_XCLBIN_PATH", "").split(os.pathsep) if p]
    roots += [REPO / "src" / "xclbins", Path.home() / ".config" / "oflm" / "xclbins"]
    for root in roots:
        for m in sorted(Path(root).glob("*/open_kernels/diffusion_kernels.json")):
            try:
                if json.loads(m.read_text()).get("layout") == layout:
                    return m.parent
            except (OSError, ValueError):
                continue
    return None


def run(cmd, what):
    log(f"-- {what}")
    rc = subprocess.run([str(c) for c in cmd]).returncode
    if rc:
        raise SystemExit(f"{what} failed (exit {rc})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=None, help="the GGUF repo (its README names the checkpoint)")
    ap.add_argument("--gguf", default=None, help="a DiT GGUF already on disk (skips the download)")
    ap.add_argument("--resolutions", default="512,1024",
                    help="square sizes the bundle carries (default: the shipped set's 512,1024)")
    ap.add_argument("--edits", default="512,1024",
                    help="resolutions that also get an edit configuration (default: the shipped "
                         "set's 512,1024). The layout hash covers BOTH lists, so the defaults are "
                         "what lets the bundle use the kernel set that ships with the app; name "
                         "others only if you will build kernels for them too")
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
        name = gguf_source.pick(repo)
        gguf = gguf_source.ensure(repo, name)
        if any(q.lower() in name.lower() for q in gguf_source.LOSSY):
            log("[WARN] a lossy DiT quant: klein's 4-step trajectory is sensitive enough that "
                "Q8_0 and Q4_1 both come out grainy (measured against the shipped bf16 bundle). "
                "Use the repo's BF16 or F16 file for the image the model is capable of.")

    # 2+3. the bundle (the checkpoint is resolved inside export_bundle from --gguf-repo)
    out = Path(a.out).expanduser().resolve()
    run([sys.executable, HERE / "export_bundle.py",
         "--gguf-repo", repo, "--transformer-gguf", gguf,
         "--resolutions", ",".join(str(r) for r in resolutions), "--edits", a.edits,
         "--out", out, "--pack-cache", Path(a.pack_cache).expanduser(), "--jobs", a.jobs,
         "--reference-repo", repo],
        f"bundle -> {out}")

    # 4. the kernels, for the same resolutions (the layout hash ties them to this bundle).
    # A set that SHIPS with the app is enough: the engine looks beside the model first and
    # then under the xclbins roots, so building one is only for a bundle whose layout no
    # installed set carries.
    layout = json.loads((out / "config.json").read_text())["layout"]
    installed = out / "open_kernels" / "diffusion_kernels.json"
    if a.skip_kernels:
        log("-- kernels: skipped (--skip-kernels)")
    elif installed.is_file() and json.loads(installed.read_text()).get("layout") == layout:
        log(f"-- kernels: already beside the bundle (layout {layout})")
    elif (shipped := _shipped_kernels(layout)) is not None:
        log(f"-- kernels: {shipped} already matches layout {layout}")
    else:
        built = REPO / "src" / "xclbins" / "FLUX.2-klein-4B-NPU2" / "open_kernels"
        run([sys.executable, REPO / "open_kernels" / "export_dit_kernels.py",
             "--resolutions", ",".join(str(r) for r in resolutions), "--edits", a.edits,
             "--out", built], "kernel build")
        run([sys.executable, REPO / "open_kernels" / "export_dit_kernels.py",
             "--resolutions", ",".join(str(r) for r in resolutions), "--edits", a.edits,
             "--install", out / "open_kernels", "--out", built], "kernel install")

    # 5. the tag
    if a.tag:
        # `oflm add` knows what a bundle is: it copies/links the files, registers the tag
        # with the bundle's own sizes and edits, and links the kernel set whose layout
        # hash matches (open_kernels/export_dit_kernels.py --install writes those).
        run([sys.executable, REPO / "utilities" / "oflm-add" / "oflm-add.py", out,
             "--tag", a.tag, "--force"], f"install {a.tag}")

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
