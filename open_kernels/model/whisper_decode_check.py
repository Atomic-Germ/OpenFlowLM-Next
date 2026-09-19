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
    ap.add_argument("--proto", default="hf", choices=("hf", "host"))
    args = ap.parse_args()

    import torch
    from safetensors.numpy import load_file
    from transformers import WhisperForConditionalGeneration, WhisperTokenizer

    torch.set_grad_enabled(False)
    tok = WhisperTokenizer.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir, torch_dtype=torch.float32)
    model = model.double().eval()
    meta = json.loads((args.goldens / "meta.json").read_text(encoding="utf-8"))
    prefix_len = 4 if args.proto == "hf" else 3
    all_ok = True
    for name in meta["clips"]:
        f = args.enc_dir / f"{name}.enc_out.npy"
        if not f.is_file():
            continue
        g = load_file(str(args.goldens / f"{name}.safetensors"))
        gold = [int(t) for t in g[f"{args.proto}.tokens"]]
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
        print(f"{'SAME' if same else 'DIFF'} {name}: forced argmax {agree}/{len(steps)}, "
              f"free-run {len(ids)} vs golden {len(gold)} tokens", flush=True)
        if not same:
            print("   golden:", tok.decode(gold, skip_special_tokens=True)[:200])
            print("   got   :", tok.decode(ids, skip_special_tokens=True)[:200])
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
