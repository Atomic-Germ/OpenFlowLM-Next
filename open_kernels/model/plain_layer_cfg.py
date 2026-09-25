"""Transform the --per 1 --dump-res cfg into one plain `run` per layer on a single context."""
import sys
src = open(sys.argv[1]).read().splitlines()
out = sys.argv[2]
args, lines, seen = {}, [], False
for line in src:
    p = line.split()
    if not p: continue
    if p[0] == 'runlist_add':
        args[p[1][1:]] = (p[2], ' '.join(p[3:])); continue
    if p[0] == 'runlist': continue
    if p[0] == 'runlist_exec':
        kern, a = args[p[1][1:]]
        lines.append(f"run {kern[:3]}0 {a}"); continue
    if p[0] == 'xclbin':
        if p[1] in ('ln', 'lm'): lines.append(line)
        elif not seen: lines.append(f"xclbin X0 {sys.argv[3]}/final.xclbin"); seen = True
        continue
    if p[0] == 'kernelx':
        if p[1] in ('ln', 'lm'): lines.append(line)
        elif p[1].startswith('lxf'): lines.append(f"kernelx lxf0 X0 {sys.argv[3]}/insts.bin")
        elif p[1].startswith('axf'): lines.append(f"kernelx axf0 X0 {sys.argv[4]}/insts.bin")
        continue
    if p[0] == 'attnpos': lines.append(f"attnpos axf0 {p[2]}"); continue
    lines.append(line)
# dedupe kernelx + repeated attnpos
res, seen2, prev = [], set(), None
for l in lines:
    if l.startswith('kernelx'):
        n = l.split()[1]
        if n in seen2: continue
        seen2.add(n)
    if l.startswith('attnpos') and l == prev: continue
    res.append(l); prev = l
open(out, 'w').write('\n'.join(res) + '\n')
print('wrote', out, len(res), 'lines')
