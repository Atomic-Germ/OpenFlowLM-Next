"""Drive the shipped `oflm` non-interactively, and build prompts of an EXACT token count.

Phase 27 asked for `oflm run` and `oflm serve` to be validated at 255/256/257/511/512/513
prompt tokens. Two things made that look impossible, and both were wrong:

  * "run is interactive only" -- [`cli_wide_posix.cpp:224`](../../src/runner/cli_wide_posix.cpp:224)
    maps stdin EOF to `/bye`, so a pipe drives it to completion and to exit.
  * "the token count cannot be pinned" -- `oflm` prints
    `Prefill chunk 1/1 with N tokens`, which is the engine's own count of what it
    prefilled. That number is the ground truth, so the prompt does not have to be
    guessed at: it is measured.

The chat template wraps the user text, so `prompt_tokens = template_offset + tokens(text)`.
One run at a known text length fixes `template_offset`; every other length then follows
from the tokenizer alone, with no further calibration and no model reload. The helper
below still reports the engine's own count for every run, so a template that stopped
being a constant offset would show up as a mismatch instead of being assumed away.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OFLM = ROOT / 'build/src/oflm'
DATA = ROOT / 'oflm-data'
MODEL = DATA / 'models/Qwen3.8-27B-NPU2'
KERNELS = ROOT / 'Models/qwen38-27b/production-v2/kernels-model'
TOKENIZER = MODEL / 'tokenizer.json'

# The same device lock the resident validators take: never overlap another NPU job.
LOCK = ROOT / 'Models/qwen38-27b/.npu.lock'

# Distinct words, used as filler. They were chosen to be unremarkable: BPE merges
# change what an append costs depending on what came before it, so no word costs a
# fixed amount and none of this is assumed -- see `exact_text`, which measures the
# whole string after every append instead of adding up prices.
_POOL = (
    'alpha beta gamma delta epsilon zeta theta iota kappa lambda mu nu xi omicron pi '
    'rho sigma tau upsilon phi chi psi omega red blue green gold iron copper silver '
    'zinc neon argon xenon radon north south east west left right one two three four '
    'five six seven eight nine ten first second third cat dog bird fish tree rock sand '
    'wave cloud rain snow wind fire earth water metal glass paper cloth wood stone brick'
).split()
_BASE = 'Report:'


def env(*, block: bool | None = None, extra: dict | None = None) -> dict:
    """The environment `oflm` needs here: PATH, the NPU libs, the config root, kernels."""
    e = dict(os.environ)
    e['PATH'] = f'{ROOT / "build/src"}:{e.get("PATH", "")}'
    ld = f'{ROOT / "src/lib/xrt"}:{ROOT / "src/lib/hrx"}'
    e['LD_LIBRARY_PATH'] = ld + (':' + e['LD_LIBRARY_PATH'] if e.get('LD_LIBRARY_PATH') else '')
    e['OFLM_MODEL_PATH'] = str(DATA)
    # Without this, `oflm` reports "model_list.json not found" and exits before it ever
    # reaches the model. It names the FILE, not the directory that holds it.
    e['OFLM_CONFIG_PATH'] = str(ROOT / 'src/model_list.json')
    e['OFLM_OPEN_KERNELS_DIR'] = str(KERNELS)
    # The block route is on by default; `block=False` is how the sequential path is
    # reached without rebuilding anything.
    if block is True:
        e.pop('OFLM_OPEN_GEMM_BLOCK', None)
    elif block is False:
        e['OFLM_OPEN_GEMM_BLOCK'] = '0'
    if extra:
        e.update(extra)
    return e


class Tokenizer:
    """Just enough of tokenizer.json to count tokens the way the engine will."""

    def __init__(self, path: Path = TOKENIZER):
        from tokenizers import Tokenizer as HfTokenizer  # noqa: N814
        self._t = HfTokenizer.from_file(str(path))

    def count(self, text: str) -> int:
        return len(self._t.encode(text, add_special_tokens=False).ids)

    def exact_text(self, want: int) -> str:
        """Text whose token count is exactly `want`.

        BPE merges mean an appended word does not cost a fixed number of tokens: the
        price depends on what precedes it, so any additive bookkeeping drifts. So the
        count is never accumulated -- the whole string is re-tokenized after every
        append, and the append is kept only if it raised the count. Because a word
        costs at least one token, every accepted append strictly increases the count
        and the loop is bounded by `want`, not by a guess.
        """
        if want <= 0:
            return ''
        text = _BASE
        if self.count(text) > want:
            raise RuntimeError(f'even the base is longer than {want} tokens')
        guard = 0
        while self.count(text) < want:
            guard += 1
            if guard > 8 * want + 64:
                raise RuntimeError(f'exact_text({want}) made no progress')
            before = self.count(text)
            for w in _POOL:
                cand = text + ' ' + w
                if self.count(cand) > before:
                    text = cand
                    break
            else:
                raise RuntimeError(f'no word raised the count past {before}')
        got = self.count(text)
        if got != want:
            raise RuntimeError(f'exact_text({want}) landed on {got}')
        return text


def run(session: list[str], *, block: bool | None = None, ctx: int = 4096,
        gen_lim: int = 8, timeout: float = 3600, tag: str = '',
        extra: dict | None = None) -> dict:
    """One whole `oflm run` session: the lines go in on stdin, the process exits at EOF."""
    script = ''
    if gen_lim > 0:
        script += f'/set gen-lim {gen_lim}\n'
    # Greedy, so two routes that agree should agree exactly rather than probably.
    script += '/set temp 0\n/set topk 1\n'
    for line in session:
        script += line + '\n'
    script += '/bye\n'
    cmd = [str(OFLM), 'run', 'qwen3.8:27b', '--ctx-len', str(ctx)]
    t0 = time.monotonic()
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a') as handle:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_EX)
        proc = subprocess.run(cmd, input=script, cwd=str(ROOT), env=env(block=block, extra=extra),
                              capture_output=True, text=True, timeout=timeout)
    wall = time.monotonic() - t0
    text = proc.stdout + '\n' + proc.stderr
    return dict(tag=tag, block=block, rc=proc.returncode, wall_s=round(wall, 3),
                stdout=proc.stdout, stderr=proc.stderr, text=text,
                chunks=parse_chunks(text), output=parse_output(proc.stdout),
                command=' '.join(cmd))


def parse_chunks(text: str) -> list[dict]:
    """`Prefill chunk i/n with N tokens` -- the engine's own count of what it prefilled."""
    rows = []
    for m in re.finditer(r'Prefill chunk (\d+)/(\d+) with (\d+) tokens', text):
        rows.append(dict(index=int(m.group(1)), count=int(m.group(2)),
                         tokens=int(m.group(3))))
    return rows


def prefilled_tokens(rec: dict) -> int:
    """How many prompt tokens the engine actually prefilled, across all chunks."""
    return sum(c['tokens'] for c in rec['chunks'])


_NOISE = re.compile(
    r'^(?:\[OFLM\]|open_qwen36:|>>>|\.\.\.|/set|/bye|Error:|WARNING)'
    r'|^Using custom|^Goodbye!$|^\s*$')


def parse_output(stdout: str) -> str:
    """Only the model's own text: no banners, no prompts, no open_qwen36 diagnostics."""
    keep = []
    for line in stdout.splitlines():
        if _NOISE.search(line):
            continue
        keep.append(line)
    return '\n'.join(keep).strip()


def main() -> int:
    """Self-check: the tokenizer builds exact lengths and `oflm` agrees. CPU only."""
    tok = Tokenizer()
    want = [int(x) for x in sys.argv[1:]] or [32, 255, 256, 257]
    for n in want:
        text = tok.exact_text(n)
        got = tok.count(text)
        print(f'{n:5d} -> {got:5d} tokens, {len(text):6d} chars  {"ok" if got == n else "MISMATCH"}')
        if got != n:
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
