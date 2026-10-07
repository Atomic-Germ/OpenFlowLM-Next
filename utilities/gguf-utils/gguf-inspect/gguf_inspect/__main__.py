"""Command line for `gguf-inspect`.

Usage:
    gguf-inspect path/to/file.gguf            # human-readable report
    gguf-inspect path/to/file.gguf --json    # machine-readable report
    gguf-inspect path/to/file.gguf --only tags|blocks|rope|tokenizer|exec  # focused
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap

from . import inspect as _inspect
from . import __version__


def _human(rep) -> str:
    b = rep.blocks
    lines = [
        f"path:            {rep.path}",
        f"gguf version:    {rep.version}",
        f"endian:          {rep.endian}",
        f"architecture:    {rep.architecture}",
    ]
    if rep.snapshot:
        lines.append(f"HF snapshot:     {rep.snapshot}")
    if rep.general:
        lines.append(f"general:         {rep.general}")
    if b.block_count:
        lines.append(
            f"blocks:          {b.block_count} total, {b.trunk_layers} trunk"
            f"{(' + ' + str(b.nextn_predict_layers) + ' NextN (MTP) layers, ' if b.has_mtp else ' ')}"
            + (b.mtp_note + "" if b.has_mtp else "no MTP/NextN layers")
        )
    if rep.projector.is_clip:
        lines.append(
            f"projector:       clip base_model={rep.projector.base_model or 'n/a'}"
        )
    if rep.rope.tensor_names:
        prefix = "carried in tensors: " if rep.rope.carried else ""
        lines.append(f"rope factors:    {prefix}{rep.rope.tensor_names}")
    t = rep.tokenizer
    if t.model or t.pre:
        lines.append(
            f"tokenizer:       model={t.model or 'n/a'} pre={t.pre or 'n/a'} "
            f"BOS={t.bos} EOS={t.eos} PAD={t.pad} chat={t.has_chat_template}"
        )
    if rep.candidate_tag:
        lines.append(f"candidate tag:   {rep.candidate_tag}")
    e = rep.execution
    if e.file_type is not None or e.dtype_counts:
        lines.append(
            f"execution:       file_type={e.file_type} quant_version={e.quantization_version} "
            f"dtype_counts={e.dtype_counts} unknown_dtypes={e.unknown_dtypes}"
        )
    u = rep.unsupported
    if u.requirements:
        lines.append("requires:")
        lines.extend(f"  - {r}" for r in u.requirements)
    if u.notes:
        lines.append("notes:")
        lines.extend(f"  - {n}" for n in u.notes)
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="gguf-inspect",
        description="Inspect GGUF model-set metadata without opening the NPU.",
    )
    parser.add_argument("path", help="GGUF model file")
    parser.add_argument(
        "--source",
        help="source repo id (e.g. Atomic-Germ/Qwen3.8-9B-Distill-GGUF); "
             "the HF snapshot hash is auto-detected from the path",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["tags", "blocks", "rope", "tokenizer", "exec"],
        help="restrict output to these sections",
    )
    parser.add_argument("--version", action="store_true", help="show version and exit")
    args = parser.parse_args(argv)
    if args.version:
        print(f"gguf-inspect {__version__}")
        return 0

    rep = _inspect(args.path, source=args.source)

    if args.json:
        data = rep.to_dict()
        if args.only:
            keys = {"tags": "candidate_tag", "blocks": "blocks", "rope": "rope",
                    "tokenizer": "tokenizer", "exec": "execution"}
            data = {k: data.get(k) for k in args.only if k in data}
        print(json.dumps(data, indent=2, default=str))
        return 0

    if args.only:
        print(json.dumps(rep.to_dict(), indent=2, default=str))
        return 0
    print(_human(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
