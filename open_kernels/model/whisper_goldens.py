r"""Golden data for the open Whisper engine (issue #72), from transformers in float64.

    python open_kernels/model/whisper_goldens.py --model-dir DIR --out OUT clip.mp3 [...]

The oracle is `WhisperForConditionalGeneration` loaded from the HF checkpoint
(openai/whisper-large-v3-turbo, stored fp16) and promoted to float64, so every figure the
engine is later compared against carries no rounding of its own. It needs torch and
transformers; run it from a reference venv, not from the IRON one. Audio is decoded with
openai/whisper's own ffmpeg command -- 16 kHz mono **s16le**, then /32768 -- because the
output sample format changes the stereo downmix gain (see decode_audio() below), and an
oracle that does not consume the runtime's own audio measures a different clip.

Per clip, the FIRST 30 s window only -- that is the engine's unit (encode_audio takes one
[128][3000] mel):

  mel                  [128, 3000]   WhisperFeatureExtractor output (the engine's input)
  conv1                [3000, 1280]  GELU(conv1(mel)), time-major
  conv2                [1500, 1280]  GELU(conv2(conv1)) + embed_positions (layer 0's input)
  enc.hidden.{i}       [1500, 1280]  input of encoder layer i, i = 0..31, then i = 32 is
                                     the last layer's output BEFORE the final LayerNorm
  enc.out              [1500, 1280]  after the final LayerNorm (what cross-attention reads)
  dec.{l}.xk / .xv     [1500, 1280]  cross-attention K and V of decoder layer l (v has its
                                     bias, k_proj has none)
  {proto}.tokens       [T]   int32   the decoder input sequence, teacher-forced
  {proto}.logits       [T, 51866]    raw logits after each input token (no processors)

Two protocols, because the host never feeds the language token it detects
(modeling_whisper.cpp:148-162):

  hf    [SOT, <lang>, transcribe, <|0.00|>, greedy...]   what transformers does
  host  [SOT, transcribe, <ts>, greedy...]                what oflm's Whisper::generate does
        (its first timestamp is the argmax over timestamp tokens, as _sample_in_time_stamp)

Greedy here is plain argmax until EOT, with no watchdog and no suppression, so the path is
a property of the model alone; the engine is checked by feeding it the same tokens.
The `hf` path's text is the reference transcript for a window. `meta.json` also carries
transformers' own generate() output for the whole clip, with and without timestamps, for
context only: generate() applies suppression and timestamp processors, so its text differs
from plain greedy on 5 of the first 6 clips (a capital, a word, a whole extra sentence).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SR = 16000
N_SAMPLES = 30 * SR
SOT, EOT, TRANSCRIBE, NOTS = 50258, 50257, 50360, 50364
LANG_LO, LANG_HI = 50259, 50358          # inclusive, as _sample_in_language
TS_BEGIN = 50365                          # <|0.00|>


def decode_audio(path: Path) -> np.ndarray:
    """openai/whisper's own `load_audio` command, verbatim, then s16 -> float.

    THE OUTPUT SAMPLE FORMAT CHANGES THE DOWNMIX GAIN. `-ac 1` to **f32le** is
    exactly sqrt(2) louder than `-ac 1` to **s16le** on a correlated stereo
    source: libswresample normalises the stereo->mono matrix to preserve power
    for float output and to preserve amplitude for integer output. Measured on
    a dual-mono clip: rms 0.068495 (f32le) against 0.048433 (s16le), ratio
    1.414214.

    That is not a cosmetic difference for Whisper: the mel is log-scaled and
    then clamped against its own maximum, so a global gain shifts every value
    by log10(2)/4 = 0.0753 and the transcript changes. This oracle used f32le
    at first, which made its goldens disagree with the runtime's own FFmpeg
    path (s16, via libswresample) and made live transcripts look wrong when
    they were not.
    """
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
         "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SR), "-"],
        check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-steps", type=int, default=440)
    ap.add_argument("--no-generate", action="store_true",
                    help="skip transformers' generate() transcripts (slow on long clips)")
    ap.add_argument("clips", nargs="+", type=Path)
    args = ap.parse_args()

    import torch
    from safetensors.numpy import save_file
    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

    torch.set_grad_enabled(False)
    fe = WhisperFeatureExtractor.from_pretrained(args.model_dir)
    tok = WhisperTokenizer.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir, torch_dtype=torch.float32)
    model = model.double().eval()
    enc, dec = model.model.encoder, model.model.decoder
    cfg = model.config
    assert (cfg.d_model, cfg.encoder_layers, cfg.decoder_layers, cfg.num_mel_bins, cfg.vocab_size) \
        == (1280, 32, 4, 128, 51866), "not whisper-large-v3-turbo's geometry"

    args.out.mkdir(parents=True, exist_ok=True)
    index = {"model_dir": str(args.model_dir),
             "model_sha256": sha256(args.model_dir / "model.safetensors"),
             "torch": torch.__version__, "clips": {}}

    for clip in args.clips:
        t0 = time.perf_counter()
        name = clip.stem
        pcm = decode_audio(clip)
        first = pcm[:N_SAMPLES]
        feats = fe(first, sampling_rate=SR, return_tensors="np").input_features[0]  # [128, 3000]
        x = torch.from_numpy(feats).double()[None]

        t: dict[str, np.ndarray] = {"mel": feats.astype(np.float32)}
        h1 = torch.nn.functional.gelu(enc.conv1(x))                      # [1, 1280, 3000]
        h2 = torch.nn.functional.gelu(enc.conv2(h1))                     # [1, 1280, 1500]
        h2 = h2.permute(0, 2, 1) + enc.embed_positions.weight[:1500]
        t["conv1"] = h1[0].T.float().numpy()
        t["conv2"] = h2[0].float().numpy()

        eo = enc(x, output_hidden_states=True)
        hs = eo.hidden_states
        assert len(hs) == 33
        assert torch.allclose(hs[0], h2), "conv stem replica disagrees with the encoder"
        for i in range(32):
            t[f"enc.hidden.{i}"] = hs[i][0].float().numpy()
        # hidden_states[32] is post-final-LayerNorm in transformers; recompute the pre-LN one
        # by running the last layer on hs[31] so the engine can be checked layer by layer.
        last = enc.layers[31](hs[31], attention_mask=None)
        last = last[0] if isinstance(last, tuple) else last
        t["enc.hidden.32"] = last[0].float().numpy()
        out = eo.last_hidden_state
        assert torch.allclose(enc.layer_norm(last), out)
        t["enc.out"] = out[0].float().numpy()
        for l, layer in enumerate(dec.layers):
            t[f"dec.{l}.xk"] = layer.encoder_attn.k_proj(out)[0].float().numpy()
            t[f"dec.{l}.xv"] = layer.encoder_attn.v_proj(out)[0].float().numpy()

        meta: dict = {"clip": clip.name, "clip_sha256": sha256(clip),
                      "seconds": len(pcm) / SR, "window_seconds": len(first) / SR}

        def step(ids: list[int]) -> np.ndarray:
            """Logits after every token of `ids`, in one uncached pass."""
            o = model(encoder_outputs=(out,), decoder_input_ids=torch.tensor([ids]))
            return o.logits[0].float().numpy()          # [len(ids), vocab]

        def greedy(ids: list[int]) -> list[int]:
            """Plain argmax until EOT, with a KV cache (an uncached loop is O(T^2))."""
            o = model(encoder_outputs=(out,), decoder_input_ids=torch.tensor([ids]), use_cache=True)
            while len(ids) < args.max_steps:
                nxt = int(torch.argmax(o.logits[0, -1]))
                ids.append(nxt)
                if nxt == EOT:
                    break
                o = model(encoder_outputs=(out,), decoder_input_ids=torch.tensor([[nxt]]),
                          past_key_values=o.past_key_values, use_cache=True)
            return ids

        lg0 = step([SOT])[-1]
        lang = LANG_LO + int(np.argmax(lg0[LANG_LO:LANG_HI + 1]))
        for proto in ("hf", "host"):
            ids = [SOT, lang, TRANSCRIBE, TS_BEGIN] if proto == "hf" else [SOT, TRANSCRIBE]
            if proto == "host":
                lg = step(ids)[-1]
                ids.append(TS_BEGIN + int(np.argmax(lg[TS_BEGIN:])))
            ids = greedy(ids)
            logits = step(ids)                          # one pass: logits after every input
            first_free = 4 if proto == "hf" else 3
            path = np.argmax(logits[first_free - 1:-1], axis=1)
            assert np.array_equal(path, ids[first_free:]), \
                f"{proto}: the cached greedy path is not the uncached argmax path"
            t[f"{proto}.tokens"] = np.array(ids, dtype=np.int32)
            t[f"{proto}.logits"] = logits
            meta[proto] = {"language": tok.decode([lang]), "n_tokens": len(ids),
                           "ended": ids[-1] == EOT,
                           "text": tok.decode(ids, skip_special_tokens=True),
                           "text_with_ts": tok.decode(ids, decode_with_timestamps=True)}

        if not args.no_generate:
            feats_all = fe(pcm, sampling_rate=SR, return_tensors="pt", truncation=False,
                           padding="longest", return_attention_mask=True)
            xin = feats_all.input_features.double()
            if xin.shape[-1] < 3000:
                xin = torch.nn.functional.pad(xin, (0, 3000 - xin.shape[-1]))
            kw = dict(task="transcribe")
            if xin.shape[-1] > 3000:
                kw["attention_mask"] = feats_all.attention_mask
            for ts in (False, True):
                g = model.generate(xin, return_timestamps=ts or xin.shape[-1] > 3000, **kw)
                seq = g if isinstance(g, torch.Tensor) else g["sequences"]
                meta[f"generate_ts{int(ts)}"] = tok.decode(
                    seq[0].tolist(), skip_special_tokens=not ts, decode_with_timestamps=ts)

        # safetensors.numpy writes a non-contiguous array in MEMORY order, silently: a
        # transposed or permuted tensor (conv1's .T, every hidden state -- transformers
        # keeps the stem's permuted layout through the residual adds) round-trips as a
        # different matrix of the same shape. Found when the replica matched enc.out to
        # 1e-8 and every hidden state at cosine 0.1-0.2.
        t = {k: np.ascontiguousarray(v) for k, v in t.items()}
        save_file(t, str(args.out / f"{name}.safetensors"))
        meta["elapsed_s"] = round(time.perf_counter() - t0, 1)
        index["clips"][name] = meta
        print(f"{name}: {meta['hf']['language']} {meta['hf']['n_tokens']} tok, "
              f"{meta['elapsed_s']} s -- {meta['hf']['text'][:80]!r}", flush=True)

    (args.out / "meta.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
