r"""Harness program for a whole-model decode on the merged `lax` kernels (designs/layer_x/lax.py):
every whole layer (attention + router + routed experts retargeted ON-DEVICE + shared expert) is
one run, and `--per` consecutive layers are ONE `xrt::runlist` submit on one hw_context.

    python open_kernels/model/make_decode.py --requant --layers 40 --tokens N --out OUT
    python open_kernels/model/lax_decode_cfg.py --out OUT --lax-l BUILD_L --lax-a BUILD_A --per 40
    open_kernels/harness/build/run_kernel OUT/run_lax_p40.cfg
    python open_kernels/model/compare_decode.py --out OUT --tokens N

make_decode.py writes the per-layer pools / consts / zero state and the fp64 reference; this
file only re-spells its run_decode.cfg for lax. `--per 40` is the whole token in ONE submit;
`--per 8` is five. The layer kinds come from run_decode.cfg (which kernel runs each pool).
Each layer's cfg holds its own pool base (`poolbase`, words 0..1): the emitters form the
routed-expert addresses from it. Words 2..9 are the per-column w-channel MM2S queue registers
(the merged lax design's shim allocation, designs/router/check_ondv_channels.py), which
ondv_ctrl_col reads at run time.

The lax builds must be ONDV builds with the three fixes (MOE_ONDEVICE_ROUTE=1 ONDV_EMIT_SHARED=1
ONDV_PKTDONE_ACQ=1, see 35b-whole-layer-runlist-status.md "2026-09-24 (lax decode)").
`setup()` / `position()` are also the building blocks of model/lax_chat.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GLOBALS = ("xres", "zero", "normw", "xresf", "hn", "logits", "lmpool", "ptab")
# lax's w{c} fifos: MM2S ch1 (0x1D21C) or ch0 (0x1D214) per column
QUEUES = (0x1D21C, 0x1D214, 0x1D21C, 0x1D21C, 0x1D21C, 0x1D21C, 0x1D214, 0x1D214)
POOL, CONSTS, ACT, CFG, STATE, KV = 536870912, 11882496, 190464, 4096, 2342912, 8388608
HEAD = ["run ln xres zero normw xresf hn", "run lm lmpool hn logits"]    # final norm + lm_head


def sfx(t):
    return "" if t == 0 else f"_t{t}"


def setup(out: Path, lax_l: Path, lax_a: Path, per: int) -> tuple[list[str], int, str]:
    """Kernels, buffers and the runlists of a lax decode over make_decode.py's `out`:
    (lines, number of submits per position, the layer kinds as 'l'/'f')."""
    src = (out / "run_decode.cfg").read_text().splitlines()
    kinds = {}                                # layer -> 'l' | 'f', from the kernel that runs its pool
    pool_dir = None
    for line in src:
        p = line.split()
        if p[:1] == ["run"] and len(p) > 2 and p[2].startswith("pool") and p[2][4:].isdigit():
            kinds.setdefault(int(p[2][4:]), "f" if p[1].startswith("ax") else "l")
        if p[:2] == ["buf", "pool0"]:
            pool_dir = Path(p[3]).parent
    nl = len(kinds)
    tail = [line for line in src if line.split()[:1] == ["xclbin"] and line.split()[1] in ("ln", "lm")]
    tail += [line for line in src if line.split()[:1] == ["kernelx"] and line.split()[1] in ("ln", "lm")]

    c = ["device", "attngeom 2048 1024"]
    ng = (nl + per - 1) // per
    for g in range(ng):
        c += [f"xclbin X{g} {lax_l}/final.xclbin", f"kernelx lxf{g} X{g} {lax_l}/insts.bin",
              f"kernelx axf{g} X{g} {lax_a}/insts.bin"]
    c += tail
    c += [line for line in src if line.startswith("buf ") and line.split()[1] in GLOBALS]
    c += [f"buf dkv {KV}", f"buf dstate {STATE}"]            # the unused kv / state argument
    qmap = out / "qmap_lax.bin"
    qmap.write_bytes(b"".join(q.to_bytes(4, "little") for q in QUEUES))
    c.append(f"buf qmap 32 {qmap}")
    for l in range(nl):
        c += [f"buf pool{l} {POOL} {pool_dir}/pool_L{l}.bin", f"buf consts{l} {CONSTS} {out}/consts_{l}.bin",
              f"buf act{l} {ACT}", f"buf cfg{l} {CFG}", f"poolbase cfg{l} 0 pool{l}",
              f"copy cfg{l} 8 qmap 0 32"]
        c.append(f"buf state{l} {STATE} {out}/zstate_linear_attention.bin" if kinds[l] == "l" else f"buf kv{l} {KV}")

    def args(l):
        if kinds[l] == "l":
            return f"pool{l} xres consts{l} dkv act{l} ptab state{l} cfg{l}"
        return f"pool{l} xres consts{l} kv{l} act{l} ptab dstate cfg{l}"

    for g in range(ng):
        c.append(f"runlist r{g}")
        for l in range(g * per, min(g * per + per, nl)):
            c.append(f"runlist_add r{g} {'lxf' if kinds[l] == 'l' else 'axf'}{g} {args(l)}")
    c.append("attngeom 2048 1024 0")
    return c, ng, "".join(kinds[l] for l in range(nl))


def position(pos: int, ng: int, feed: str | None = None) -> list[str]:
    """One position: optionally feed a token's embedding (`feed` = an id or 'last'; needs the
    `embed` buffer), then the attention position and every submit."""
    c = [f"feed xres embed {feed}"] if feed is not None else []
    c += [f"attnpos axf{g} {pos}" for g in range(ng)]
    c += [f"runlist_exec r{g}" for g in range(ng)]
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="make_decode.py's --out")
    ap.add_argument("--lax-l", required=True, help="lax build, LAX_KIND=0 (linear attention)")
    ap.add_argument("--lax-a", required=True, help="lax build, LAX_KIND=1 (full attention)")
    ap.add_argument("--per", type=int, default=40, help="layers per hw_context / runlist submit")
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--dump-res", action="store_true", help="dump xres after every submit")
    ap.add_argument("--prompt-ids", default=None,
                    help="generation mode: comma-separated prompt token ids, fed one position at a time")
    ap.add_argument("--gen", type=int, default=0, help="generation mode: greedy tokens after the prompt")
    ap.add_argument("--embed", default=None, help="generation mode: the bf16 embedding table [vocab, hidden]")
    ap.add_argument("--vocab", type=int, default=248070, help="generation mode: real vocab (argmax range)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    c, ng, kinds = setup(out, Path(a.lax_l).resolve(), Path(a.lax_a).resolve(), a.per)
    if a.prompt_ids:
        # greedy generation: every position is one feed + one submit; the norm + lm_head (and the
        # argmax, fed back as the next input) only where a next token is wanted
        ids = [int(v) for v in a.prompt_ids.split(",")]
        emb = Path(a.embed).resolve()
        c.append(f"buf embed {emb.stat().st_size} {emb}")
        for pos in range(len(ids) + a.gen):
            c += position(pos, ng, str(ids[pos]) if pos < len(ids) else "last")
            if pos >= len(ids) - 1:
                c += HEAD + [f"greedy logits {a.vocab}"]
            c.append("tick")
        path = out / f"run_lax_gen_p{a.per}.cfg"
        path.write_text("\n".join(c) + "\n", newline="\n")
        print(f"wrote {path}: {len(ids)} prompt + {a.gen} generated positions, {ng} submit(s)/position")
        return 0
    for t in range(a.tokens):
        s = sfx(t)
        if t:
            c.append(f"load xres {out}/xres{t}.bin")
        c += [f"attnpos axf{g} {t}" for g in range(ng)]
        for g in range(ng):
            c.append(f"runlist_exec r{g}")
            if a.dump_res:
                c.append(f"dump xres {out}/y_res{min(g * a.per + a.per, len(kinds)) - 1}{s}.bin 8192")
        c += HEAD + [f"dump logits {out}/y_logits{s}.bin 993280"]
    path = out / f"run_lax_p{a.per}.cfg"
    path.write_text("\n".join(c) + "\n", newline="\n")
    print(f"wrote {path}: {len(kinds)} layers ({kinds}), {ng} submit(s)/token, {a.tokens} token(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
