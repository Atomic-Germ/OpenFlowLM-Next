"""Builder for the open diffusion engine's model repo (FLUX.2 [klein] 4B).

    q4nx-build --open-diffusion -i black-forest-labs/FLUX.2-klein-4B -o FLUX.2-klein-4B-NPU2

The work is utilities/dit-chain/export_bundle.py's (its docstring lists the files): the
packed GEMM weights, the VAE's, the schedules, the embedding table. They are packed for
dit_gemm's tile layout, not left row-major, because packing 7.5 GiB at every load would
take minutes; bundle.json and config.json carry the layout hash that ties them to a
kernel set, and the engine refuses any other. This module checks the checkpoint is the
one the kernels are built for, adds the README, and writes model_info_entry.json.

It imports the packing code from this repo's checkout (open_kernels/, utilities/dit-chain),
so run it from a checkout, not an installed wheel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from .open_causal import resolve_source
from .open_embedding import sha256_file

MODEL_INFO_ARTIFACT = "model_info_entry.json"
REPO_ROOT = Path(__file__).resolve().parents[3]

# The configurations the shipped kernel set is built for (export_dit_kernels.py --resolutions
# 512,1024 --edits 512,1024): the model's schedules and layout hash must name the same ones.
RESOLUTIONS, EDITS = [512, 1024], [512, 1024]

README = """---
license: apache-2.0
base_model: black-forest-labs/FLUX.2-klein-4B
pipeline_tag: text-to-image
tags:
- oflm
- npu
- image-to-image
---

# FLUX.2 [klein] 4B for OpenFlowLM (XDNA2 NPU)

[FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) converted for
OpenFlowLM's open diffusion engine: text encoder, 4 denoising steps and VAE decoder all run
on an AMD XDNA2 NPU -- and for an edit, the VAE encoder too.

    oflm image flux2-klein:4b "a red fox in fresh snow" -o fox.png
    oflm image flux2-klein:4b "make it night" --image photo.jpg -o night.png

Sizes: 512 and 1024 square. An edit's reference is centre-cropped to a square at the
output's size.

The GEMM weights are packed in dit_gemm's bfp16 tile layout for the kernel set OpenFlowLM
ships (layout `{layout}`); they are not usable by other runtimes. Built with
`q4nx-build --open-diffusion` from the checkpoint above. License: Apache 2.0, as the
original (LICENSE.md).
"""


def _git_blob_oid(data: bytes) -> str:
    import hashlib
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def hub_entry(path: Path, rel: str) -> dict:
    """The file as the HF tree API lists it once uploaded (what src/model_info.json holds and
    oflm pull verifies against): *.bin and files of 10 MB or more go to LFS -- `lfs.oid` is
    the SHA-256, `oid` the blob id of the LFS pointer -- and the rest are plain git blobs."""
    size = path.stat().st_size
    if rel.endswith(".bin") or size >= 10 * 1000 * 1000:
        sha = sha256_file(path)
        pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {size}\n"
        return {"type": "file", "oid": _git_blob_oid(pointer.encode()), "size": size,
                "lfs": {"oid": sha, "size": size}, "path": rel}
    return {"type": "file", "oid": _git_blob_oid(path.read_bytes()), "size": size, "path": rel}


def build_open_diffusion_repo(source: str, output_dir: str, npu_assets: Optional[str] = None,
                              pack_cache: Optional[str] = None, jobs: int = 4) -> dict:
    if npu_assets:
        raise ValueError("--npu-assets is not used by --open-diffusion: kernels are built by "
                         "open_kernels/export_dit_kernels.py")
    if not (REPO_ROOT / "utilities" / "dit-chain" / "export_bundle.py").is_file():
        raise RuntimeError(f"--open-diffusion runs from an OpenFlowLM checkout; "
                           f"{REPO_ROOT} is not one")
    sys.path.insert(0, str(REPO_ROOT / "utilities" / "dit-chain"))
    # A GGUF source carries only the DiT: the checkpoint that supplies the text encoder and
    # the VAE is the one the GGUF repo's README names, exactly as a text quant repo names
    # its base for the tokenizer. `q4nx-build --open-diffusion <gguf-repo>` therefore works
    # on a GGUF repo alone.
    transformer_gguf = None
    if str(source).lower().endswith(".gguf") and Path(source).is_file():
        raise ValueError("a local GGUF needs the repo it came from for the checkpoint: pass "
                         "the HF repo id instead, or use utilities/dit-chain/klein-from-gguf.py")
    import gguf_source  # noqa: E402

    if not Path(source).is_dir() and gguf_source.is_gguf_repo(source):
        import klein_checkpoint  # noqa: E402

        transformer_gguf = gguf_source.ensure(source, gguf_source.pick(source))
        src, base = klein_checkpoint.resolve(source)
        print(f"checkpoint: {src} (from {base}, for the text encoder and the VAE)", flush=True)
    else:
        src = resolve_source(source)
    read = lambda p: json.loads((src / p).read_text(encoding="utf-8"))  # noqa: E731
    if read("model_index.json").get("_class_name") != "Flux2KleinPipeline":
        raise ValueError(f"{src} is not a Flux2KleinPipeline checkpoint")
    # The geometry used to be checked here against hand-written dicts (attention_head_dim,
    # num_layers, hidden_size ...) that restated the schedule's constants -- a checkpoint
    # whose config.json agreed but whose tensors did not would pass and pack garbage.
    # export_bundle.build now asks the TENSORS (klein_pipeline.validate_sources), which is
    # also the only check that works when the DiT is a GGUF and there is no config.json.

    import export_bundle  # noqa: E402

    out = Path(output_dir)
    files = export_bundle.build(src, out.resolve(), RESOLUTIONS, jobs,
                                Path(pack_cache) if pack_cache else None, EDITS,
                                transformer_gguf=transformer_gguf)
    layout = json.loads((out / "config.json").read_text(encoding="utf-8"))["layout"]
    (out / "README.md").write_text(README.replace("{layout}", layout), encoding="utf-8")
    files = sorted(files + ["README.md"])

    model_info = [hub_entry(out / rel, rel) for rel in files]
    (out / MODEL_INFO_ARTIFACT).write_text(json.dumps(model_info, indent=1) + "\n", encoding="utf-8")
    return {"output_dir": str(out), "source": str(src), "files": files,
            "layout": layout, "model_info": model_info}
