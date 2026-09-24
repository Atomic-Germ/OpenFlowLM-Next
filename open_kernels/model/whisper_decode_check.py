r"""Does a given encoder output still produce the right transcript? transformers' decoder
(float64) is run greedily on an encoder output produced elsewhere -- the bf16 replica's, or
later the engine's -- and its token path compared with the golden one.

    python open_kernels/model/whisper_decode_check.py --model-dir DIR --goldens DIR \
        --enc-dir DIR [--proto hf|host]

The decoder here is exact, so any difference is the encoder's. The prefix is the golden's
own forced prefix (SOT, language, transcribe, first timestamp for `hf`), then plain argmax
until EOT: the same procedure whisper_goldens.py used, so identical tokens mean the encoder
output is as good as the fp64 one for this clip. Also reported: at how many decoding steps
the golden token is still the argmax when the decoder is teacher-forced along the golden
path (argmax agreement), which does not stop at the first divergence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EOT = 50257


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--goldens", required=True, type=Path)
    ap.add_argument("--enc-dir", required=True, type=Path, help="<clip>.enc_out.npy files")
    ap.add_argument("--proto", default="hf", choices=("hf", "host", "both"))
    ap.add_argument("--write-baseline", type=Path, default=None,
                    help="write the token path this encoder output produces, per clip and "
                         "protocol, as the bf16 datapath's own reference (see below)")
    args = ap.parse_args()

    # Which clips can actually be checked, decided BEFORE the model is loaded. A missing
    # file used to be skipped silently, so a typo in --enc-dir checked nothing and still
    # exited 0 -- a gate that passes without measuring anything. Deciding it here also
    # means a wrong path costs a second rather than the minute the float64 model takes.
    meta = json.loads((args.goldens / "meta.json").read_text(encoding="utf-8"))
    protos = ("hf", "host") if args.proto == "both" else (args.proto,)
    clips = [n for n in meta["clips"] if (args.enc_dir / f"{n}.enc_out.npy").is_file()]
    missing = [n for n in meta["clips"] if n not in clips]
    for n in missing:
        print(f"MISSING {args.enc_dir / (n + '.enc_out.npy')}", flush=True)
    if not clips:
        print(f"nothing to check: no <clip>.enc_out.npy in {args.enc_dir}", flush=True)
        return 1
    all_ok = not missing

    import torch
    from safetensors.numpy import load_file
    from transformers import WhisperForConditionalGeneration, WhisperTokenizer

    torch.set_grad_enabled(False)
    tok = WhisperTokenizer.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir, torch_dtype=torch.float32)
    model = model.double().eval()
    baseline: dict = {}
    for proto, name in [(p, n) for p in protos for n in clips]:
      prefix_len = 4 if proto == "hf" else 3
      if True:
        f = args.enc_dir / f"{name}.enc_out.npy"
        g = load_file(str(args.goldens / f"{name}.safetensors"))
        gold = [int(t) for t in g[f"{proto}.tokens"]]
        out = torch.from_numpy(np.load(f).astype(np.float64))[None]

        # teacher-forced along the golden path: one pass, argmax at every position
        lg = model(encoder_outputs=(out,), decoder_input_ids=torch.tensor([gold])).logits[0]
        am = lg.argmax(-1).tolist()
        steps = range(prefix_len - 1, len(gold) - 1)
        agree = sum(am[i] == gold[i + 1] for i in steps)

        ids = gold[:prefix_len]
        o = model(encoder_outputs=(out,), decoder_input_ids=torch.tensor([ids]), use_cache=True)
        while len(ids) < 440:
            nxt = int(torch.argmax(o.logits[0, -1]))
            ids.append(nxt)
            if nxt == EOT:
                break
            o = model(encoder_outputs=(out,), decoder_input_ids=torch.tensor([[nxt]]),
                      past_key_values=o.past_key_values, use_cache=True)
        same = ids == gold
        all_ok &= same
        first_diff = next((i for i, (a, b) in enumerate(zip(ids, gold)) if a != b),
                          None if len(ids) == len(gold) else min(len(ids), len(gold)))
        baseline.setdefault(name, {})[proto] = {
            "tokens": ids, "matches_fp64": same, "first_divergence": first_diff,
            "forced_argmax": [agree, len(steps)]}
        print(f"{'SAME' if same else 'DIFF'} {name} [{proto}]: forced argmax {agree}/{len(steps)}, "
              f"free-run {len(ids)} vs golden {len(gold)} tokens"
              + ("" if same else f", first divergence at {first_diff}"), flush=True)
        if not same:
            print("   golden:", tok.decode(gold, skip_special_tokens=True)[:200])
            print("   got   :", tok.decode(ids, skip_special_tokens=True)[:200])
    if args.write_baseline:
        args.write_baseline.write_text(
            json.dumps({"format": "oflm-open-whisper-bf16-token-baseline-v1",
                        "model_sha256": meta.get("model_sha256", ""),
                        "clips": baseline}, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.write_baseline}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
