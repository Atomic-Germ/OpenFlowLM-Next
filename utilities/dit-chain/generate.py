r"""generate: FLUX.2 [klein] 4B text-to-image with every op on the NPU.

    . C:\dev\mlir-aie\iron_env.ps1
    python open_kernels\export_dit_kernels.py --resolutions 512,1024 --out C:\dev\klein-kernels
    python utilities\dit-chain\generate.py --kernels C:\dev\klein-kernels --size 512 --prompt "a red fox in fresh snow" --out C:\dev\gen

Runs open_kernels/klein_pipeline.py's schedule (text encoder, conditioning, 4 denoising
steps, VAE; 1050 dispatches over six kernel sets) through the pyxrt host. Runs on one
hardware context are queued back to back; before the next kernel set the host blocks on
the last one (XRT's wait sleeps). The host does only what does not grow with the data:
tokenize, gather the prompt's 512 embedding rows, write the noise, patch te_attn's
valid_len, and encode the PNG.

    --study <goldens_pipe dir>   utilities/dit-ref/capture_pipeline_inputs.py's prompts and
                                 fixed noise: images comparable with klein_quant_study's bf16
                                 CPU run (score with utilities/dit-ref/score_images.py)
    --ctx-ref                    with --study: the bf16 text encoder's embeddings instead of
                                 the NPU's (isolates the DiT + VAE)
    --profile                    also time every op (a blocking wait after each)
    --edit                       an edit (klein_pipeline.plan(R, edit=True)): with --study, a
                                 capture_edit_goldens.py directory (prompts, prepared
                                 references ref_<i>.npy, noise); with --prompt, --ref <npy|png>
    --ref-tokens                 with --edit --study: diffusers' bf16 reference tokens
                                 (reflat_<i>.npy) instead of the NPU encoder's (isolates the DiT)
    --swap-ref K                 with --edit --study: edit i gets reference i + K (the
                                 reference-ablation of OPEN-DIFFUSION-EDIT)

Weights are packed once into <kernels>\packed\ (dit_gemm's bfp16 layout; ~8 GB).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import inspect
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "open_kernels"))
sys.path.insert(0, str(ROOT / "open_kernels" / "harness"))
sys.path.insert(0, str(HERE))
import klein_pipeline as kp  # noqa: E402
from safetensors_np import SafeTensors  # noqa: E402

SETS = {"gemm": ".", "fa": "fa", "ew": "ew", "conv": "conv", "conv1": "conv1", "vew": "vew"}


def model_dir() -> Path:
    return Path(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--black-forest-labs"
                              "--FLUX.2-klein-4B/snapshots/*"))[0])


def _pack_one(sub: str, name: str, cache: str, ckpt: str) -> tuple[str, float]:
    t0 = time.time()
    specs = kp.dit_weight_specs() if sub == "transformer" else kp.te_weight_specs()
    K, N, build = specs[name]
    st = SafeTensors(Path(ckpt) / sub)
    b = build(st.get)
    assert b.shape == (K, N), (name, b.shape, (K, N))
    out = Path(cache) / f"{name}.bin"
    tmp = out.with_suffix(".tmp")
    kp.pack_b_cols(b).tofile(tmp)
    tmp.replace(out)
    return name, time.time() - t0


def pack_fingerprint(ckpt: Path) -> str:
    """Identifies what a packed file was made from: the checkpoint's weight files (name, size and
    samples of the head, middle and tail, so a retrained same-shape checkpoint differs) and the
    packing code."""
    h = hashlib.sha256(inspect.getsource(kp.pack_b_cols).encode())
    for sub in ("transformer", "text_encoder"):
        for p in sorted((ckpt / sub).glob("*.safetensors")):
            n = p.stat().st_size
            h.update(f"{sub}/{p.name}:{n}".encode())
            with open(p, "rb") as f:
                for at in (0, max(0, n // 2 - (1 << 19)), max(0, n - (1 << 20))):
                    f.seek(at)
                    h.update(f.read(1 << 20))
    return h.hexdigest()


def ensure_packed(cache: Path, jobs: int, ckpt: Path | None = None) -> dict[str, tuple[Path, int]]:
    """{name: (file, bytes)} for every DiT and text-encoder GEMM weight, packing the missing.
    ckpt: the checkpoint directory (default: the HF cache's snapshot)."""
    ckpt = ckpt or model_dir()
    cache.mkdir(parents=True, exist_ok=True)
    stamp, fp = cache / "provenance.txt", pack_fingerprint(ckpt)
    if stamp.exists() and stamp.read_text().strip() != fp:
        print(f"{cache} was packed from another checkpoint or packing format; repacking", flush=True)
        for p in cache.glob("*.bin"):
            p.unlink()
    stamp.write_text(fp + "\n")
    want = {n: ("transformer", K * N * 9 // 8) for n, (K, N, _) in kp.dit_weight_specs().items()}
    want |= {n: ("text_encoder", K * N * 9 // 8) for n, (K, N, _) in kp.te_weight_specs().items()}
    todo = [n for n, (_, nb) in want.items()
            if not (cache / f"{n}.bin").exists() or (cache / f"{n}.bin").stat().st_size != nb]
    if todo:
        print(f"packing {len(todo)} weights -> {cache} ({jobs} processes)", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(jobs) as ex:
            for i, (n, s) in enumerate(ex.map(_pack_one, [want[n][0] for n in todo], todo,
                                              [str(cache)] * len(todo), [str(ckpt)] * len(todo))):
                if i % 20 == 0 or i == len(todo) - 1:
                    print(f"  [{i + 1}/{len(todo)}] {n} {s:.1f} s", flush=True)
        print(f"packed in {time.time() - t0:.0f} s", flush=True)
    return {n: (cache / f"{n}.bin", nb) for n, (_, nb) in want.items()}


def embed_rows(st: SafeTensors, ids: np.ndarray) -> np.ndarray:
    """model.embed_tokens rows for ids as bf16, zero-padded to the 3072 element: XT."""
    f, meta = st._where["model.embed_tokens.weight"]
    mm, base = st._maps[f]
    a, _ = meta["data_offsets"]
    V, D = meta["shape"]
    table = np.ndarray((V, D), dtype=np.uint16, buffer=mm, offset=base + a)
    x = np.zeros((len(ids), kp.TE_PAD), np.uint16)
    x[:, :D] = table[ids]
    return x


class Runner:
    def __init__(self, kdir: Path, R: int, pack_jobs: int, edit: bool = False):
        from npu_host import Npu
        import chain_test_vae as ctv

        t0 = time.time()
        self.R, self.pl = R, kp.plan(R, edit=edit)
        md = model_dir()
        self.dit, self.te = SafeTensors(md / "transformer"), SafeTensors(md / "text_encoder")
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(md / "tokenizer" / "tokenizer.json"))
        packed = ensure_packed(kdir / "packed", pack_jobs)

        self.npu = npu = Npu()
        self.sets = {n: npu.kernel_set(n, kdir / d) for n, d in SETS.items()}
        fa_meta = json.loads((kdir / "fa" / "dit_fa.json").read_text(encoding="utf-8"))
        self.vl_words = fa_meta["patch"]["te_attn"]["valid_len"]

        self.bufs = {n: npu.buf(n, nb) for n, nb in self.pl.buffers.items()}
        for b in self.bufs.values():
            b.zero()
        self.bufs["PARAMS"].write(kp.fill_params(self.dit.get, self.te.get))
        vp = self.pl.vae
        qin = vp.buffers["QIN"]
        q = np.zeros((qin.H * qin.W, qin.C), bfloat16)
        q[:, 512] = 1                                  # the VAE attention GEMM's bias row
        self.bufs["v_QIN"].write(q)
        wbytes, self.vtable, blocks, self.vindex = ctv.packed_weights(kdir / "vae_packed.npz")
        self.vW, self.vS = npu.buf("vae_W", wbytes.size), npu.buf("vae_S", blocks.size * 2)
        self.vW.write(wbytes)
        self.vS.write(blocks)
        self.blk_bytes = ctv.vd.BLOCK * ctv.vd.EL * 2
        if edit:
            import chain_test_vae_enc as cte
            wbytes, self.etable, blocks, self.eindex = cte.packed_weights(kdir / "venc_packed.npz")
            self.eW, self.eS = npu.buf("venc_W", wbytes.size), npu.buf("venc_S", blocks.size * 2)
            self.eW.write(wbytes)
            self.eS.write(blocks)

        t1 = time.time()
        self.w = {}
        for n, (path, nb) in packed.items():
            self.w[n] = npu.buf(f"w:{n}", nb)
            self.w[n].load(path)
        gb = sum(nb for _, nb in packed.values()) / 2 ** 30
        print(f"weights: {gb:.2f} GiB in {time.time() - t1:.1f} s", flush=True)

        self.sig = kp.sigmas(R, self.pl.steps)
        self.bufs["TF"].write(kp.timestep_features(self.sig, self.pl.steps))
        self.bufs["DT"].write(kp.dt_params(self.sig, self.pl.steps))

        self.ops = []
        for o in self.pl.ops:
            st = self.sets[o["set"]].stream(o["stream"])
            self.ops.append((o["set"], st, [self._arg(a) for a in o["args"]], o["phase"], o))
        self.vl_stream = self.sets["fa"].stream("te_attn")
        print(f"loaded {R}x{R}: {len(self.ops)} dispatches in {time.time() - t0:.1f} s", flush=True)

    def _arg(self, ref):
        kind = ref[0]
        if kind == "buf":
            _, n, off, nb = ref
            return self.bufs[n].view(off, nb)
        if kind == "w":
            return self.w[ref[1]].bo
        if kind == "vae_w":
            off, n = self.vtable[ref[1]]
            return self.vW.view(off, n)
        if kind == "vae_gn":
            return self.vS.view(self.vindex[ref[1]] * self.blk_bytes, self.blk_bytes)
        if kind == "venc_w":
            off, n = self.etable[ref[1]]
            return self.eW.view(off, n)
        if kind == "venc_gn":
            return self.eS.view(self.eindex[ref[1]] * self.blk_bytes, self.blk_bytes)
        raise ValueError(ref)

    def set_prompt(self, prompt: str, ids_check: np.ndarray | None = None) -> int:
        ids, n_real = kp.token_ids(self.tok, prompt)
        if ids_check is not None and not np.array_equal(ids, ids_check):
            raise SystemExit(f"tokenization differs from the pipeline's for {prompt!r}")
        self.bufs["XT"].write(embed_rows(self.te, ids))
        self.vl_stream.patch(self.vl_words, n_real)
        return n_real

    def set_ctx(self, ctx_bits: np.ndarray) -> None:
        """CTX from outside (the bf16 text encoder's embeddings [512, 7680] as bf16 bits)."""
        c = np.zeros((kp.L_TXT, kp.CTX_LD), np.uint16)
        c[:, :ctx_bits.shape[1]] = ctx_bits
        self.bufs["CTX"].write(c)

    def set_noise(self, lat_bits: np.ndarray) -> None:
        self.bufs["LAT"].write(np.ascontiguousarray(lat_bits, np.uint16))

    def set_reference(self, rgb: np.ndarray) -> None:
        """An edit's prepared reference (uint8 [R, R, 3]) as the encoder reads it: bf16
        2 (x / 255) - 1 (float32, diffusers' arithmetic) in channels 0-2 of its
        zero-bordered input."""
        b = self.pl.enc.buffers["IN"]
        x = np.zeros((b.H + 2, b.pitch, b.C), bfloat16)
        x[1:b.H + 1, 1:b.W + 1, :3] = (np.float32(2) * (rgb.astype(np.float32) / np.float32(255))
                                       - np.float32(1)).astype(bfloat16)
        self.bufs[self.pl.enc_buffers["IN"]].write(x)

    def set_ref_tokens(self, tok_bits: np.ndarray) -> None:
        """REFLAT from outside (diffusers' packed, normalised reference tokens, bf16 bits);
        run with skip=("encode",)."""
        self.bufs["REFLAT"].write(np.ascontiguousarray(tok_bits, np.uint16))

    def run(self, skip=(), profile=False):
        """Every op in order. Returns (seconds per phase, per-op ms if profile)."""
        from npu_host import COMPLETED

        def wait(h, what):
            st = h.wait()
            if st != COMPLETED:
                raise RuntimeError(f"{what}: {st}")

        phases: dict[str, float] = {}
        per_op = []
        cur_set, cur_phase, last = None, None, None
        t_phase = t0 = time.perf_counter()
        for kset, st, bos, phase, o in self.ops:
            if phase in skip:
                continue
            if kset != cur_set or phase != cur_phase:
                if last is not None:
                    wait(last, f"{cur_set}/{o['what']}")
                    last = None
                if phase != cur_phase:
                    now = time.perf_counter()
                    if cur_phase is not None:
                        phases[cur_phase] = now - t_phase
                    t_phase, cur_phase = now, phase
                cur_set = kset
            ts = time.perf_counter()
            last = st.start(*bos)
            if profile:
                wait(last, f"{kset}/{st.name} ({o['what']})")
                last = None
                per_op.append((time.perf_counter() - ts) * 1e3)
        if last is not None:
            wait(last, "last op")
        now = time.perf_counter()
        phases[cur_phase] = now - t_phase
        phases["total"] = now - t0
        return phases, per_op

    def image(self) -> np.ndarray:
        R = self.R
        return self.bufs["v_RGBA"].read(np.uint8).reshape(-1, 8192)[:, :4096] \
            .reshape(R, R, 4)[..., :3].copy()

    def latents(self) -> np.ndarray:
        T = kp.image_tokens(self.R)
        return self.bufs["LAT"].read(np.uint16, 0, T * kp.LAT_CH).reshape(T, kp.LAT_CH)


def load_ref(path: str, R: int) -> np.ndarray:
    """An edit's reference as uint8 [R, R, 3]: a prepared .npy as is; any image file the way
    capture_edit_goldens.prepare does (EXIF orientation, RGB, centre crop, LANCZOS)."""
    if path.lower().endswith(".npy"):
        return np.load(path)
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = min(img.size)
    left, top = (img.width - s) // 2, (img.height - s) // 2
    img = img.crop((left, top, left + s, top + s))
    if s != R:
        img = img.resize((R, R), Image.Resampling.LANCZOS)
    return np.asarray(img, np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--study", default=None, help="capture_pipeline_inputs.py output dir")
    ap.add_argument("--prompts", type=int, default=8, help="with --study: how many")
    ap.add_argument("--ctx-ref", action="store_true")
    ap.add_argument("--ref-latents", default=None,
                    help="with --study: dir of the bf16 pipeline's final packed latents "
                         "lat_<i>.npy (capture_vae_goldens.py's goldens_vae_<size>)")
    ap.add_argument("--runs", type=int, default=1, help="generations of each prompt (timing)")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--edit", action="store_true")
    ap.add_argument("--ref", action="append", default=[],
                    help="with --edit --prompt: the prepared reference (uint8 [R, R, 3] .npy)")
    ap.add_argument("--ref-tokens", action="store_true")
    ap.add_argument("--swap-ref", type=int, default=0)
    ap.add_argument("--pack-jobs", type=int, default=4)
    ap.add_argument("--pack-only", action="store_true", help="pack the weights and exit")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.pack_only:
        ensure_packed(Path(a.kernels) / "packed", a.pack_jobs)
        return 0
    if not a.out:
        raise SystemExit("--out is required")
    why = kp.check_resolution(a.size)
    if why:
        raise SystemExit(f"unsupported size: {why}")
    import chain_test_vae as ctv

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r = Runner(Path(a.kernels), a.size, a.pack_jobs, edit=a.edit)
    T = kp.image_tokens(a.size)
    jobs = []                       # (tag, prompt, noise bits, ids, ctx bits, reference, tokens)
    if a.study:
        sd = Path(a.study)
        prompts = json.loads((sd / "prompts.json").read_text(encoding="utf-8"))
        n = min(a.prompts, len(prompts))
        for i in range(n):
            ctx = np.load(sd / f"ctx_{i}.npy") if a.ctx_ref else None
            ref = tok = None
            if a.edit:
                k = (i + a.swap_ref) % len(prompts)
                ref = np.load(sd / f"ref_{k}.npy")
                tok = np.load(sd / f"reflat_{k}.npy") if a.ref_tokens else None
            p = prompts[i]["prompt"] if isinstance(prompts[i], dict) else prompts[i]
            jobs.append((f"{i:02d}", p, np.load(sd / f"noise_{i}.npy"),
                         np.load(sd / f"ids_{i}.npy"), ctx, ref, tok))
    for j, p in enumerate(a.prompt):
        rng = np.random.default_rng(a.seed + j)
        noise = rng.standard_normal((T, kp.LAT_CH), np.float32).astype(bfloat16).view(np.uint16)
        ref = load_ref(a.ref[j], a.size) if a.edit else None
        jobs.append((f"p{j:02d}", p, noise, None, None, ref, None))

    report = {"size": a.size, "edit": a.edit, "images": []}
    for tag, prompt, noise, ids, ctx, ref, tok in jobs:
        for run in range(a.runs):
            h0 = time.perf_counter()
            c0 = time.process_time()
            skip = []
            if ctx is not None:
                r.set_ctx(ctx)
                n_real = None
                skip.append("text")
            else:
                n_real = r.set_prompt(prompt, ids)
            r.set_noise(noise)
            if tok is not None:
                r.set_ref_tokens(tok)
                skip.append("encode")
            elif ref is not None:
                r.set_reference(ref)
            host_ms = (time.perf_counter() - h0) * 1e3
            ph, per_op = r.run(skip=tuple(skip), profile=a.profile)
            cpu_s = time.process_time() - c0
            img = r.image()
            ctv.write_png(out / f"{tag}.png", np.ascontiguousarray(img))
            rec = {"tag": tag, "prompt": prompt, "run": run, "tokens": n_real,
                   "host_setup_ms": host_ms, "npu_s": ph["total"], "host_cpu_s": cpu_s,
                   "phases_s": {k: v for k, v in ph.items() if k != "total"}}
            report["images"].append(rec)
            steps = sum(v for k, v in ph.items() if k.startswith("step"))
            enc = f", encode {ph['encode']:.2f}" if "encode" in ph else ""
            print(f"[{tag} run {run}] {ph['total']:.2f} s on the NPU (text {ph.get('text', 0):.2f}, "
                  f"cond {ph.get('cond', 0):.3f}{enc}, steps {steps:.2f}, vae {ph.get('vae', 0):.2f}); "
                  f"host setup {host_ms:.0f} ms, host CPU {cpu_s:.2f} s  -> {out / f'{tag}.png'}",
                  flush=True)
            if a.profile:
                profile(r, per_op, skip=tuple(skip))
        lat = r.latents()
        np.save(out / f"lat_{tag}.npy", lat)
        ref = Path(a.ref_latents or "") / f"lat_{int(tag)}.npy" if a.ref_latents and tag.isdigit() else None
        if ref is not None and ref.exists():
            got = lat.view(bfloat16).astype(np.float64)
            want = np.load(ref).view(bfloat16).astype(np.float64)
            rf = float(np.linalg.norm(got - want) / np.linalg.norm(want))
            print(f"  final latents vs the bf16 pipeline's: rel_fro {rf:.3e}, finite "
                  f"{bool(np.isfinite(got).all())}", flush=True)
            report["images"][-1]["latents_rel_fro"] = rf
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return 0


def profile(r: Runner, per_op: list[float], skip=()) -> None:
    ops = [o for o in r.ops if o[3] not in skip]
    by_set, by_phase_set = defaultdict(float), defaultdict(float)
    after, same = 0.0, 0.0
    for k, ((kset, st, _, phase, o), ms) in enumerate(zip(ops, per_op)):
        by_set[kset] += ms
        by_phase_set[(phase if not phase.startswith("step") else "steps", kset)] += ms
        if k and ops[k - 1][0] != kset:
            after += ms
        else:
            same += ms
    n_sw = sum(1 for x, y in zip(ops, ops[1:]) if x[0] != y[0])
    print(f"  per set (blocking waits, ms): " + ", ".join(f"{s} {v:.0f}" for s, v in by_set.items()))
    print(f"  {n_sw} kernel-set switches; ops right after a switch {after:.0f} ms, the rest {same:.0f} ms")
    for (ph, s), v in sorted(by_phase_set.items()):
        print(f"    {ph:6s} {s:5s} {v:8.1f} ms")
    kinds = defaultdict(lambda: [0, 0.0])
    for (kset, st, _, phase, o), ms in zip(ops, per_op):
        kinds[(kset, st.name)][0] += 1
        kinds[(kset, st.name)][1] += ms
    print("  heaviest streams (count, total ms, mean ms):")
    for (kset, name), (n, v) in sorted(kinds.items(), key=lambda kv: -kv[1][1])[:20]:
        print(f"    {kset:5s} {name:20s} {n:4d} {v:9.1f} {v / n:8.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
