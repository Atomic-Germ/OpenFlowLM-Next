#!/usr/bin/env python3
"""Console-script entry point for q4nx-build."""
import os
import re
import shutil
import sys
from pathlib import Path

from q4nx import create_converter, create_hf_converter
from q4nx.arch_detect import family_from_text
from q4nx.build_plan import derive_build_plan, format_chain
from q4nx.model_assets import (
    assemble_model_assets,
    assemble_model_assets_hf,
    get_default_oflm_version,
    find_repo_gguf,
    select_repo_gguf,
)


def _is_hf_repo_id(path: str) -> bool:
    """True if path looks like an 'org/name' HF repo id (not a local path, not a .gguf)."""
    if not path or path.endswith(".gguf") or os.path.exists(path):
        return False
    if path.startswith(("http://", "https://", "file:")):
        return False
    parts = path.split("/")
    return len(parts) == 2 and all(parts) and "\\" not in path


def _split_repo_quant(spec: str):
    """`-i org/name:Q8_0` -> (repo_id, "q8_0"); anything else -> (spec, None).

    llama.cpp's syntax, so the coordinate a reader already has off a model page
    works unchanged. The quant is a PIN rather than a preference: it is the only
    thing that keeps a recorded recipe producing the same artifact once the repo
    gains a better-quantized file.

    Only split when what precedes the colon is itself a repo id. A Windows path
    (`D:/models/x.gguf`) contains a colon and must survive intact, and a quant
    token is not a path -- so requiring the `org/name` shape on the left is what
    keeps the two apart.
    """
    if not spec or ":" not in spec or spec.endswith(".gguf"):
        return spec, None
    head, _, tail = spec.rpartition(":")
    if not head or "/" not in head:
        return spec, None
    if not _is_hf_repo_id(head):
        return spec, None
    return head, tail.strip().lower()


def _is_hf_source(path: str) -> bool:
    """True if path is a local HF-safetensors model dir."""
    if os.path.isdir(path):
        return (
            os.path.exists(os.path.join(path, "model.safetensors"))
            or os.path.exists(os.path.join(path, "model.safetensors.index.json"))
        )
    return False


def _report_speculative(model) -> None:
    """End-of-pack roll call for the speculative (best-effort) path.

    _WarnDict already prints each unknown tensor as it appears; this closes
    the loop so the user is not left assembling a mid-log of warnings. If
    anything was passed through unmapped, the container *may* not load in any
    runtime -- say so plainly rather than leaving a plausible-looking
    model.q4nx on disk.
    """
    unknown = getattr(getattr(model, "forward_name_map", None), "unknown", [])
    missing = getattr(model, "_missing_config_entries", [])
    unknown_types = getattr(getattr(model, "tensor_q4nx_type_map", None), "unknown", [])
    if unknown or missing or unknown_types:
        print("\n[WARN] Speculative pack: the converter produced a best-effort container.")
        if unknown:
            print(f"[WARN]   {len(unknown)} GGUF tensor(s) had no config mapping and were "
                  f"carried through under their GGUF names:")
            for n in unknown:
                print(f"           {n}")
        if unknown_types:
            print(f"[WARN]   {len(unknown_types)} tensor(s) fell back to the default type; "
                  f"check the dtype policy matches this model:")
            for n in unknown_types:
                print(f"           {n}")
        if missing:
            print(f"[WARN]   {len(missing)} config tensor(s) were absent from the GGUF:")
            for n in missing:
                print(f"           {n}")
        print("[WARN] If this build is wrong, an OFLM runtime will refuse to load it or "
              "misdecode weights; do not publish it as support for this architecture.")
        # Which supported family would have mapped this model best? Pure
        # coverage scoring over the configs' own templates -- it does not make
        # the model supported, it only tells the user the least-wrong -f.
        # Only meaningful for a GGUF source.
        ranked = []
        if getattr(model, "gguf_reader", None) is not None:
            try:
                from q4nx.model_converter import config_coverage_report
                ranked = config_coverage_report(list(getattr(model, "gguf_tensors", {}).keys()))
            except Exception:
                ranked = []
        if ranked:
            print("[WARN] Nearest config families by tensor coverage (support guessing aid):")
            for fname, covered, total, frac in ranked[:3]:
                print(f"         {fname:<22} {covered}/{total} templates ({frac:.0%})")
            best, bc, bt, bf = ranked[0]
            if bf >= 0.99:
                print(f"[WARN] {best} covers the GGUF's tensor layout fully; retrying with "
                      f"'-f {best.replace('.json', '')}' should produce a proper pack.")


# mradermacher splits one model's files across TWO repos of the same name: the
# bare `-GGUF` carries the K-quants, Q8_0, f16 and the mmproj, while `-i1-GGUF`
# carries the I-quants AND the imatrix they were made with. Neither half packs a
# pruned model on its own -- `--prune-ffn` needs the imatrix, and the imatrix is
# only in the second repo -- so someone naming the first gets a refusal that
# names neither repo. The convention is `Base-GGUF` / `Base-i1-GGUF`, i.e. `-i1`
# spliced in before the suffix.
#
# Scoped to that one publisher deliberately. The name pattern is a quirk of his
# layout, not a Hub convention, and guessing at sibling repos for an arbitrary
# owner means network calls nobody asked for and a real chance of joining two
# unrelated models that happen to share a stem. A wrong imatrix prunes the FFN
# by the wrong neurons, which is silent.
IMATRIX_SIBLING_OWNERS = ("mradermacher/",)
IMATRIX_SIBLING_TAG = "-i1-GGUF"


def _imatrix_sibling_repo(repo_id):
    """The `-i1-GGUF` sibling of a split repo, or None when there is no rule.

    Refuses a repo whose stem already carries an `-i<N>` tag, so that naming the
    I-quant half does not produce `...-i1-i1-GGUF` -- that repo is where the
    imatrix already is, and there is nothing further to look up.
    """
    if not repo_id or not repo_id.endswith("-GGUF"):
        return None
    if not repo_id.startswith(IMATRIX_SIBLING_OWNERS):
        return None
    stem = repo_id[: -len("-GGUF")]
    if re.search(r"-i\d+$", stem):
        return None
    return stem + IMATRIX_SIBLING_TAG


def _resolve_imatrix_hint(args, input_path, source_repo=None):
    """Where the importance matrix lives, for --prune-ffn.

    An explicit --imatrix wins; then OFLM_IMATRIX, so a pack driven through
    `oflm pack` (which shells out to this tool) can name it in the environment
    rather than requiring the flag to be threaded through; then a sidecar next
    to the model file; then, for a split publisher, the sibling repo that
    actually holds it. Returns None when there is nothing, which the converter
    turns into an explanation instead of a silent full-width pack.
    """
    from q4nx.imatrix_prune import imatrix_path, env_imatrix
    explicit = args.imatrix or env_imatrix()
    try:
        p = imatrix_path(explicit, _model_dir(input_path))
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        raise SystemExit(2)
    if explicit and p is None:
        print(f"[ERROR] --imatrix {explicit} is not an importance matrix "
              f"(no *.in_sum2 tensors)")
        raise SystemExit(2)
    if args.prune_ffn and p is None:
        sibling = _imatrix_sibling_repo(source_repo)
        if sibling:
            from q4nx.model_assets import find_repo_imatrix
            found = find_repo_imatrix(sibling)
            if found:
                path, filename = found
                print(f"[INFO] {source_repo} does not publish an imatrix; this "
                      f"model's files are split across two repos. Took "
                      f"{filename} from {sibling} instead.")
                return path
            print(f"[WARN] {sibling} publishes no imatrix either; continuing "
                  f"without one.")
        print(f"[ERROR] --prune-ffn {args.prune_ffn} needs an imatrix and none was "
              f"found. Pass --imatrix PATH or put the imatrix GGUF beside the model.")
        raise SystemExit(2)
    return p


def _layer_count(model):
    """How many layers the converted container holds, from its own tensors.

    The right source of truth for num_hidden_layers: the GGUF's block_count
    counts the MTP block, the source config may or may not have excluded it,
    and only the tensors just written settle it.
    """
    n = 0
    for name in getattr(model, "q4nx_tensors", {}):
        parts = name.split(".")
        if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers" \
                and parts[2].isdigit():
            n = max(n, int(parts[2]) + 1)
    return n or None


IMATRIX_NAME = "imatrix.gguf"
# A pruned container is only reproducible with the imatrix that chose its
# neurons, and --prune-ffn refuses without one. Recording the flag therefore
# records a dependency the reader does not have, which is how "the base was
# pruned with an imatrix" ends up living in one person's head. An imatrix is
# activation SUMS, not weights: 13.6 MB for a 27B, so shipping it costs
# nothing beside a 16 GB container. Past this it is still copied -- dropping it
# silently would reintroduce the gap -- but it is worth saying out loud.
IMATRIX_LARGE_BYTES = 256 * 1024 * 1024


def _stage_imatrix(model, output_folder, prune_meta):
    """Copy the imatrix that chose this container's neurons into the container.

    Returns the path to record in the reproduction command, or None when no
    prune ran (nothing to stage) or the copy could not be made -- in which case
    the command falls back to naming a path the reader must supply.
    """
    if not prune_meta.get("kept"):
        return None
    src = getattr(model, "imatrix_path_hint", None)
    if not src:
        return None
    src = Path(src)
    dst = Path(output_folder) / IMATRIX_NAME
    if not src.is_file():
        print(f"[WARN] {src} is gone; the recorded command will name it as a path "
              f"the reader has to supply rather than shipping it with the container.")
        return None
    try:
        if src.resolve() == dst.resolve():
            return dst                      # already packed in place; nothing to copy
        n = src.stat().st_size
        shutil.copy2(src, dst)
    except OSError as e:
        print(f"[WARN] could not copy the imatrix into the container ({e}); the "
              f"recorded command will name it as a path the reader must supply.")
        return None
    print(f"[INFO] Shipped the imatrix that chose these neurons: {IMATRIX_NAME} "
          f"({n / 1e6:.1f} MB)")
    if n > IMATRIX_LARGE_BYTES:
        print(f"[WARN] that imatrix is {n / 1e6:.1f} MB, which is large next to a "
              f"model card. It was copied anyway: without it the prune cannot be "
              f"reproduced at all.")
    return dst


def _packed_command(args, input_path, output_folder, source_model, prune_meta,
                    imatrix_ref=None, source_repo=None, pin_quant=None):
    """The `oflm pack` line that reproduces this container, for the model card.

Reconstructed from the parsed arguments rather than read from sys.argv,
    because `oflm pack` re-invokes this module through `python -c` and argv
    would only show that wrapper. What determines the artifact is these flags, so
    these flags are what the card records -- a finetune can then be packed the
    same way as the base was, without anyone having to remember.

    `-i` records the repo id when the pack came from one, never the file it
    downloaded to. A resolved HF cache path names a content-addressed snapshot
    on ONE machine: it does not resolve on the reader's, and it is not even the
    same path after a cache eviction. The repo id re-resolves, and the exact
    filename chosen is recorded beside it as the card's "Source GGUF" row, so
    the pair says both which repository and which file in it.
    """
    # The pin goes back into the recorded command: dropping it would leave a
    # recipe that re-runs the preference search and can pick a different file.
    src = source_repo or input_path
    if pin_quant:
        src = f"{src}:{pin_quant.upper()}"
    cmd = ["oflm pack", "-i", str(src), "-o", str(output_folder)]
    if source_model:
        cmd += ["-s", str(source_model)]
    if getattr(args, "force_model_type", ""):
        cmd += ["-f", str(args.force_model_type)]
    if getattr(args, "quant", None):
        cmd += ["--quant", str(args.quant)]
    if getattr(args, "pad_to_fit", False):
        cmd.append("--pad-to-fit")
    if prune_meta.get("kept"):
        cmd += ["--prune-ffn", str(args.prune_ffn)]
        # The staged copy when there is one, so the command reproduces from the
        # container alone. The fallback names a path rather than pretending:
        # an unrunnable recipe is better than one that points at a file the
        # reader has to guess the name of.
        cmd += ["--imatrix", str(imatrix_ref or args.imatrix
                                 or "<path to the imatrix GGUF>")]
    if getattr(args, "deploy_tag", None):
        cmd += ["--deploy", str(args.deploy_tag)]
    return " ".join(cmd)


def _prune_meta(model):
    """What the converter recorded about an imatrix prune, for the card + config.

    `mtp_dropped` and `layers_actual` are always reported: a Qwen3.5 export
    carries one EXTRA transformer block for multi-token prediction, this runtime
    has no speculative decoding, so that block is never converted and the depth
    config.json declares must be what model.q4nx actually holds (Qwen3.8-27B:
    block_count 65, num_hidden_layers 64). The prune-specific keys stay absent
    when no prune ran, so an unpruned pack's FFN declarations are untouched.
    """
    if not getattr(model, "imatrix", None):
        return {
            "mtp_dropped": getattr(model, "mtp_dropped", 0),
            "layers_actual": _layer_count(model),
        }
    return {
        "kept": getattr(model, "prune_ffn_kept", None),
        "frm": getattr(model, "prune_ffn_from", None),
        "retained": getattr(model, "prune_ffn_retained", None),
        "mtp_dropped": getattr(model, "mtp_dropped", 0),
        # measured, not declared: the layer count the manifest will carry
        "layers_actual": _layer_count(model),
    }


def _model_dir(input_path):
    from pathlib import Path
    if _is_hf_repo_id(input_path) or _is_hf_source(input_path):
        return None
    p = Path(input_path)
    return p if p.is_dir() else p.parent


def _parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser(
        prog="q4nx-build",
        description=(
            "Convert GGUF or HF-safetensors model files to Q4NX format (output always named "
            "model.q4nx). -i also accepts an HF repo id: a quantized GGUF is auto-selected in "
            "family-preferred order."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_file", nargs="?", help="Input GGUF file (positional)")
    parser.add_argument(
        "-i", "--input", dest="input_flag", help="Input GGUF file, or an HF repo id"
    )
    parser.add_argument(
        "-o", "--output", dest="output_flag", help="Output folder (optional)"
    )
    parser.add_argument(
        "-t",
        "--type",
        dest="weights_type",
        default=None,
        choices=["language", "vision", "audio"],
        help="Type of weights to convert (default: inferred from the repo card; "
             "VLM pipeline tags convert language + vision, otherwise language)",
    )
    parser.add_argument(
        "-f", "--force", dest="force_model_type", default="", help="Model type override"
    )
    parser.add_argument(
        "-s", "--source-model", dest="source_model", default=None,
        help="Source HF/ModelScope model for tokenizer/config assets (the NPU2 "
             "skeleton). Default: the first ancestor in the repo card's "
             "base_model chain with an {org}/{base}-NPU2 mirror "
             "(orgs: Atomic-Germ, then OpenFlowLM).",
    )
    parser.add_argument(
        "--source-repo", dest="source_repo", default=None, metavar="ORG/NAME",
        help="Declare the HF repo a LOCAL GGUF came from, so the recorded pack "
             "command names a repo instead of a path only this machine has. Use "
             "with -i <file.gguf>; ignored when -i is already a repo id.",
    )
    parser.add_argument(
        "--dry-run", dest="dry_run", action="store_true",
        help="Resolve and print the build plan (GGUF choice, base_model chain, "
             "skeleton source, output name, weights type) without converting.",
    )
    parser.add_argument(
        "--pad-to-fit", dest="pad_to_fit", action="store_true",
        help="When the model's hidden size is smaller than the selected engine "
             "variant's official dim, zero-pad the hidden axis so the weights "
             "fit the compiled variant (padded channels are inert).",
    )
    parser.add_argument(
        "--prune-ffn", dest="prune_ffn", type=int, default=None, metavar="K",
        help="Narrow the dense FFN to K intermediate neurons, chosen PER LAYER by "
             "imatrix importance. For a model whose FFN is too wide for a core's L1 "
             "(Qwen3.8-27B at 17408 will not build; 12288-13312 will). Needs an "
             "imatrix: --imatrix PATH, or a sidecar GGUF beside the model. Refuses "
             "without one rather than chopping the first K, which retains only "
             "K/width of the activation mass by construction.",
    )
    parser.add_argument(
        "--imatrix", dest="imatrix", default=None, metavar="PATH",
        help="Importance-matrix GGUF (*.in_sum2 tensors) used by --prune-ffn. "
             "Defaults to a sidecar next to the model file. OFLM_IMATRIX is "
             "honoured too, so `oflm pack` can name it without re-typing the flag.",
    )
    parser.add_argument(
        "--oflm-version", dest="oflm_version", default=None, help="oflm_version to write into config.json"
    )
    parser.add_argument(
        "--quant", dest="quant", default=None, choices=["Q4_0", "Q4_1", "Q8_0", "Q4_K"],
        help="Override the family config's default weight format. Q4_K is the 4736-byte "
             "super-block layout OFLM 1.0.3+ requires for the 35B MoE projections; the "
             "configs still default to what each family has shipped.",
    )
    parser.add_argument(
        "-d", "--deploy", dest="deploy_tag", default=None, metavar="NAME:SIZE",
        help="Deploy the converted model into oflm's models dir and register it under this tag (e.g. 'qwen3.5-claude:9b')",
    )
    parser.add_argument(
        "--deploy-from", dest="deploy_from", default=None, metavar="SOURCE_TAG",
        help="Official registry entry to copy defaults from (e.g. 'qwen3.5:9b')",
    )
    parser.add_argument(
        "--deploy-name", dest="deploy_name", default=None, metavar="DIR",
        help="Directory name inside oflm's models dir (default: derived from the deploy tag)",
    )
    parser.add_argument(
        "--open-embedding", dest="open_embedding", action="store_true",
        help="Build an open (unquantized) embedding model repo instead of Q4NX. "
             "Packs safetensors as-is, writes a portable weights_manifest.json, "
             "and emits model_info_entry.json for src/model_info.json.",
    )
    parser.add_argument(
        "--npu-assets", dest="npu_assets", default=None, metavar="DIR",
        help="With --open-embedding/--open-causal-lm: directory of "
             "m{M}_{K}x{N}.{xclbin,insts} artifacts to ship under npu_matmul_f32/",
    )
    parser.add_argument(
        "--make-reference", dest="make_reference", default=None, metavar="OUT.json",
        help="Run the independent NumPy reference implementation over a model "
             "directory and write a fixture file for engine validation. With "
             "--open-causal-lm it references the built dir; otherwise it reads "
             "-i directly. NumPy only (no torch/transformers).",
    )
    parser.add_argument(
        "--open-causal-lm", dest="open_causal_lm", action="store_true",
        help="Build an open (unquantized) causal-LM text repo (Phase 0 of the "
             "open Gemma3 text work). Packs bf16 safetensors verbatim, records "
             "per-tensor dtype and tied-embedding mapping in the manifest, and "
             "emits model_info_entry.json for src/model_info.json.",
    )
    parser.add_argument(
        "--open-whisper", dest="open_whisper", action="store_true",
        help="Build the open Whisper engine's model repo (issue #72): encoder GEMM "
             "operands as bf16 W^T with the fused Q|K|V and cross K|V the kernel set "
             "expects, f32 biases/norms, the decoder in bf16, and a tokenizer_config.json "
             "carrying the bos/eos ids the host reads.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    input_path = args.input_flag or args.input_file
    if not input_path:
        sys.exit("Error: Input file is required. Use -i <file> or provide as positional argument.")

    # What was ASKED for, before any resolution. A repo id is about to be
    # replaced by the local file it downloads to -- a path under the HF cache
    # carrying a content-addressed snapshot hash -- and that path is useless to
    # anyone but this machine, so the repo id has to be kept to be recorded.
    requested_input = input_path
    input_path, pin_quant = _split_repo_quant(input_path)
    # --source-repo says "this local GGUF came from that repo". Without it a
    # locally downloaded file has no coordinate at all, and its card cannot be
    # reproduced from by anyone -- including its author, once the file moves.
    source_repo = args.source_repo or (input_path if _is_hf_repo_id(input_path) else None)
    if args.source_repo and _is_hf_repo_id(input_path):
        print("[WARN] --source-repo names the repo the GGUF came from, but -i is "
              "already a repo id. Ignoring it; -i wins.")
        source_repo = input_path

    # Reference oracle: usable either alongside a build (reference the freshly
    # built dir) or standalone against an existing model directory.
    if args.make_reference and not (args.open_embedding or args.open_causal_lm):
        from q4nx.reference import generate_reference

        ref = generate_reference(input_path, args.make_reference)
        print(f"[INFO] Reference fixtures written to {ref['output']}")
        print(f"[INFO] Prompts: {ref['prompt_count']}, tokens: {ref['token_counts']}")
        return 0

    # Open (unquantized) builders: no GGUF, no quantization, no config
    # assembly from a repo card. One shot produces an uploadable HF repo dir.
    if args.open_whisper:
        from q4nx.open_whisper import MODEL_INFO_ARTIFACT, build_open_whisper_repo

        output_folder = os.path.abspath(args.output_flag or ".")
        result = build_open_whisper_repo(input_path, output_folder, npu_assets=args.npu_assets)
        print(f"[INFO] Open Whisper repo built at {result['output_dir']}")
        print(f"[INFO] Source: {result['source']}")
        print(f"[INFO] Tensors: {result['tensor_count']}")
        for name in result["files"]:
            print(f"  - {name}")
        print(f"[INFO] Registry metadata written to "
              f"{os.path.join(result['output_dir'], MODEL_INFO_ARTIFACT)}")
        return 0

    if args.open_embedding or args.open_causal_lm:
        # --make-reference combined with a build references the built dir.
        make_ref = args.make_reference
        if args.open_embedding:
            from q4nx.open_embedding import (
                MODEL_INFO_ARTIFACT,
                build_open_embedding_repo,
            )

            build = build_open_embedding_repo
        else:
            from q4nx.open_causal import MODEL_INFO_ARTIFACT, build_open_causal_repo

            build = build_open_causal_repo

        output_folder = os.path.abspath(args.output_flag or ".")
        result = build(input_path, output_folder, npu_assets=args.npu_assets)
        kind = "Open embedding" if args.open_embedding else "Open causal LM"
        print(f"[INFO] {kind} repo built at {result['output_dir']}")
        print(f"[INFO] Source: {result['source']}")
        print(f"[INFO] Tensors: {result['tensor_count']}")
        for name in result["files"]:
            print(f"  - {name}")
        print(
            f"[INFO] Registry metadata written to "
            f"{os.path.join(result['output_dir'], MODEL_INFO_ARTIFACT)}; "
            "merge it into src/model_info.json under the model tag."
        )

        if make_ref:
            from q4nx.reference import generate_reference

            ref = generate_reference(result["output_dir"], make_ref)
            print(f"[INFO] Reference fixtures written to {ref['output']}")
            print(f"[INFO] Prompts: {ref['prompt_count']}, tokens: {ref['token_counts']}")
        return 0

    # Local paths must exist; HF repo ids are resolved later by the converter.
    if not _is_hf_repo_id(input_path) and not os.path.exists(input_path):
        sys.exit(f"Error: Input file does not exist: {input_path}")

    oflm_version = args.oflm_version or get_default_oflm_version()

    # Fill unspecified -s/-o/-t from the repo card's base_model chain
    # (q4nx.build_plan): walk ancestors until one has an {org}/{base}-NPU2
    # mirror, use it as the skeleton source, and read the output name and VLM
    # detection from the same cards. Explicit flags always win.
    weights_type = args.weights_type
    source_model = args.source_model
    output_folder = args.output_flag
    family_hint = None
    plan = None
    if _is_hf_repo_id(input_path):
        plan = derive_build_plan(input_path)
        print(f"[INFO] Base chain: {format_chain(plan.chain)}")
        if source_model is None:
            if plan.skeleton:
                print(f"[INFO] Skeleton source: {plan.skeleton}")
            else:
                print("[WARN] No {org}/{base}-NPU2 skeleton found on the chain; "
                      "pass -s to name the asset source explicitly.")
            source_model = plan.skeleton
        if weights_type is None:
            weights_type = plan.weights_type
            suffix = f" ({plan.weights_reason})" if plan.weights_reason else ""
            print(f"[INFO] Weights type: {weights_type}{suffix}")
        if output_folder is None and plan.output_name:
            output_folder = plan.output_name
            print(f"[INFO] Output folder: {output_folder} (from card metadata)")
        family_hint = family_from_text(" ".join(plan.chain))

    weights_type = weights_type or "language"
    output_folder = output_folder or os.path.dirname(input_path) or "."
    # A pruned FFN is a different artifact from the one the card names, so the
    # DIRECTORY says so too -- not just the README. Someone with both on disk
    # should be able to tell them apart from `ls`, and a stale name that resolves
    # to the wrong widths later is the failure this prevents. Only applied when
    # the name came from the card (an explicit -o is the caller's to choose).
    if args.prune_ffn and plan is not None and plan.output_name \
            and output_folder == plan.output_name:
        base = Path(plan.output_name)
        tagged = f"{base.stem}-imx{args.prune_ffn}{base.suffix}"
        if tagged != os.path.basename(output_folder):
            output_folder = str(base.parent / tagged) if str(base.parent) != "." else tagged
            print(f"[INFO] Output folder: {output_folder} (FFN pruned to "
                  f"{args.prune_ffn}; tagged so it is not mistaken for the "
                  f"unpruned model)")

    # Resolve the weight source before touching absolute paths: an HF repo id
    # must stay in 'org/name' form or _is_hf_repo_id/create_hf_converter won't
    # recognize it.
    #
    # -i <hf-repo-id> prefers a quantized GGUF shipped in the repo itself,
    # chosen in a family-preferred order (default q4_1, then q4_0, then q8_0).
    # The order is driven by -f when given, then by the chain-derived family
    # hint. The chosen GGUF is downloaded via the HF cache; if the repo has
    # none, we fall back to the HF-safetensors source path below.
    hf_input = None
    source_file = None
    selected_gguf = None
    if _is_hf_repo_id(input_path):
        if args.dry_run:
            selected_gguf = select_repo_gguf(input_path, args.force_model_type, family_hint,
                                             pin_quant)
            if selected_gguf is None:
                hf_input = input_path
        else:
            found = find_repo_gguf(input_path, args.force_model_type,
                                   family_hint=family_hint, pin_quant=pin_quant)
            if found is not None:
                input_path, source_file = found
                source_model = source_model or input_path
            else:
                hf_input = input_path
    elif _is_hf_source(input_path):
        hf_input = input_path

    if args.dry_run:
        requested = args.input_flag or args.input_file
        print("[DRY-RUN] Build plan (nothing was converted or downloaded):")
        print(f"  input:         {requested}")
        if selected_gguf:
            print(f"  source GGUF:   {selected_gguf}")
        elif hf_input:
            print("  source:        HF safetensors (no quantized GGUF in the repo)")
        elif os.path.exists(requested):
            print(f"  source GGUF:   {requested}")
        print(f"  skeleton (-s): {source_model or '(GGUF provenance fallback)'}")
        print(f"  weights (-t):  {weights_type}")
        print(f"  output (-o):   {os.path.abspath(output_folder)}")
        return 0

    output_folder = os.path.abspath(output_folder)
    os.makedirs(os.path.dirname(output_folder) or ".", exist_ok=True)

    print(f"[INFO] Converting {input_path} to {output_folder}...")

    if hf_input is not None:
        model = create_hf_converter(hf_input, args.force_model_type)
        model.pad_to_fit = args.pad_to_fit
        model.prune_ffn = args.prune_ffn
        model.imatrix_path_hint = _resolve_imatrix_hint(args, input_path, source_repo)
        if args.quant:
            model.set_default_tensor_type(args.quant)
        if weights_type == "vision":
            model.convert(q4nx_path=output_folder, weights_type="language")
            model.convert(q4nx_path=output_folder, weights_type="vision")
        else:
            model.convert(q4nx_path=output_folder, weights_type=weights_type)
        # Always empty here: --prune-ffn needs a GGUF source (the safetensors
        # path has no imatrix to rank columns with), so there is nothing to ship.
        prune_meta = _prune_meta(model)
        assemble_model_assets_hf(
            model.hf_source,
            model.q4nx_config,
            output_folder,
            source_model=source_model or hf_input,
            oflm_version=oflm_version,
            source_file=source_file,
            model_arch=model.model_arch,
            prune_meta=prune_meta,
            packed_with=_packed_command(
                args, input_path, output_folder, source_model or hf_input, prune_meta,
                source_repo=source_repo, pin_quant=pin_quant),
        )
    else:
        model = create_converter(input_path, args.force_model_type)
        model.pad_to_fit = args.pad_to_fit
        model.prune_ffn = args.prune_ffn
        model.imatrix_path_hint = _resolve_imatrix_hint(args, input_path, source_repo)
        if args.quant:
            model.set_default_tensor_type(args.quant)
        if weights_type == "vision":
            model.convert(q4nx_path=output_folder, weights_type="language")
            model.convert(q4nx_path=output_folder, weights_type="vision")
        else:
            model.convert(q4nx_path=output_folder, weights_type=weights_type)
        prune_meta = _prune_meta(model)
        _report_speculative(model)
        # Staged AFTER the conversion and BEFORE the assets are written, so the
        # recorded command names a file that is already in the directory it names.
        imatrix_ref = _stage_imatrix(model, output_folder, prune_meta)
        if source_repo and not source_file:
            # A local GGUF has no repo-relative path, so the card could name the
            # repo but not the file in it. These publishers keep GGUFs at the
            # repository ROOT, so the basename IS the repo-relative name and the
            # two rows pair up into something a reader can resolve by hand.
            source_file = os.path.basename(input_path)
        assemble_model_assets(
            model.gguf_reader,
            model.q4nx_config,
            output_folder,
            source_model=source_model,
            oflm_version=oflm_version,
            source_file=source_file,
            model_arch=model.model_arch,
            prune_meta=prune_meta,
            packed_with=_packed_command(
                args, input_path, output_folder, source_model, prune_meta,
                imatrix_ref=imatrix_ref, source_repo=source_repo, pin_quant=pin_quant),
            imatrix_name=imatrix_ref.name if imatrix_ref else None,
            source_repo=source_repo,
        )

    if args.deploy_tag:
        from q4nx.deploy import deploy_model

        deploy_model(
            output_folder,
            args.deploy_tag,
            model.model_arch,
            model_dir_name=args.deploy_name,
            deploy_from=args.deploy_from,
        )

    print(f"[INFO] Conversion complete! Output saved to {output_folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())