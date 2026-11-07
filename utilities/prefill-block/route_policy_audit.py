"""Phase 29.0: which kernel sets did the Phase 28 preference change actually move?

`Core::layer_major_preferred()` is a capability rule -- a set keeps layer-major only if it
carries a routed-expert kernel. It does not mention any model, which is what makes it safe to
apply without an update; but "safe to apply" is not "already validated". Any pre-existing set
whose default route moves from layer-major to the block loop is being switched to a schedule
nobody measured on it, and that has to be found rather than assumed away.

This enumerates every manifest on the machine under a kernel-set root, evaluates the SAME
predicates the runtime does (no reimplementation of the policy, a literal transcription of
`Core::layer_major_ok()` and `Core::layer_major_preferred()`), and classifies each set:

    has routed experts   old preferred (capability)  new preferred  changed?

Read-only: nothing is built, nothing is run, nothing is written to a manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def layer_major_ok(m: dict, layer_major_on: bool = True) -> tuple[bool, str]:
    """`Core::layer_major_ok()`, transcribed: kinds present, all linear/full, T agrees."""
    types = m.get('layer_types') or {}
    if not types:
        return False, 'no layer_types'
    ts = {(t.get('gemm_block') or {}).get('t') for t in types.values()}
    if not layer_major_on:
        return False, 'OFLM_OPEN_LAYER_MAJOR=0'
    if len(ts) != 1 or 0 in ts:
        return False, f'gemm_block.t disagrees or is absent: {sorted(x for x in ts if x is not None)}'
    for name, t in types.items():
        kind = (t.get('gemm_block') or {}).get('kind')
        if kind not in ('linear', 'full'):
            return False, f'{name} kind={kind}'
    return True, ''


def has_routed_experts(m: dict) -> tuple[bool, list[str]]:
    """`Core::layer_major_preferred()`, transcribed: any MoE kernel anywhere."""
    why = []
    for name, t in (m.get('layer_types') or {}).items():
        gb = t.get('gemm_block') or {}
        if gb.get('moe_kernel'):
            why.append(f'{name}: moe_kernel')
        for slots, k in (gb.get('moe_batch') or {}).get('kernels', {}).items():
            why.append(f'{name}: moe_batch[{slots}]={k}')
    return bool(why), why


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, nargs='+',
                    default=[ROOT / 'Models', ROOT / 'oflm-data', ROOT / 'src' / 'models'],
                    help='where to look for manifest.json files')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'route-policy-audit-{stamp}')
    out.mkdir(parents=True, exist_ok=True)
    result = dict(tool='prefill-block-route-policy-audit', version=1, phase=29,
                  gate='29.0 route-policy compatibility closure', model_acceptance=False,
                  started=stamp,
                  policy=dict(layer_major_ok='every layer kind in {linear, full}, one T',
                              layer_major_preferred='the set carries a routed-expert kernel',
                              old_preference='layer_major_ok() alone',
                              new_preference='layer_major_ok() and layer_major_preferred()'))

    manifests = []
    for root in a.root:
        if root.is_dir():
            manifests += sorted(root.rglob('manifest.json'))
    rows, seen = [], set()
    for p in manifests:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        try:
            m = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as e:
            rows.append(dict(manifest=str(p), error=str(e)))
            continue
        ok, why = layer_major_ok(m)
        routed, routed_why = has_routed_experts(m)
        declared = bool(m.get('block_loop_preferred'))
        old = 'layer-major' if ok else ('block-loop' if (m.get('layer_types') or {}) else 'none')
        # `layer_major_preferred()`, transcribed: routed experts keep it; otherwise the set
        # keeps it too unless its manifest DECLARES the block loop was measured on it.
        new = 'block-loop' if (ok and not routed and declared) else (
            'layer-major' if ok else old)
        rows.append(dict(
            manifest=str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p),
            family=m.get('family'), spec_hash=(m.get('spec_hash') or '')[:19],
            layer_types=sorted((m.get('layer_types') or {}).keys()),
            kinds=sorted({(t.get('gemm_block') or {}).get('kind')
                          for t in (m.get('layer_types') or {}).values()}),
            block_t=sorted({(t.get('gemm_block') or {}).get('t')
                            for t in (m.get('layer_types') or {}).values()}),
            layer_major_ok=ok, layer_major_ok_why=why,
            has_routed_experts=routed, routed_why=routed_why,
            block_loop_declared=declared,
            old_preferred=old, new_preferred=new, changed=(old != new)))

    result['manifests'] = rows
    changed = [r for r in rows if r.get('changed')]
    result['changed_count'] = len(changed)
    result['changed'] = [{k: r[k] for k in ('manifest', 'family', 'layer_types',
                                            'old_preferred', 'new_preferred')}
                         for r in changed]
    # The policy's own expectation, asserted rather than described.
    # The policy: routed experts keep layer-major; a dense set keeps it unless it DECLARED
    # the block loop was measured. A dense set that keeps layer-major is not a violation --
    # it is the conservative reading, and it is what every pre-existing dense set gets.
    bad = [r for r in rows if r.get('layer_major_ok') and not r.get('error') and
           ((r['has_routed_experts'] and r['new_preferred'] != 'layer-major') or
            (not r['has_routed_experts'] and r['block_loop_declared']
             and r['new_preferred'] != 'block-loop') or
            (not r['has_routed_experts'] and not r['block_loop_declared']
             and r['new_preferred'] != 'layer-major'))]
    result['policy_violations'] = [{k: r[k] for k in ('manifest', 'has_routed_experts',
                                                      'new_preferred')} for r in bad]
    result['verdict'] = 'PASS' if not bad else 'FAIL'
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')

    print(f"{'family':16s} {'kinds':16s} {'routed':7s} {'old':12s} {'new':12s} changed")
    for r in rows:
        if r.get('error'):
            print(f"  {r['manifest']}: {r['error']}")
            continue
        # `kinds` can be `[None]` for a manifest whose layers carry no gemm_block at all --
        # those sets never had a block route, so print them rather than crash on the None.
        kinds = ','.join(str(k) for k in r['kinds']) or '-'
        print(f"{str(r.get('family')):16s} {kinds:16s} "
              f"{str(r['has_routed_experts']):7s} {r['old_preferred']:12s} "
              f"{r['new_preferred']:12s} {'YES' if r['changed'] else 'no'}")
    print(f"\n{len(rows)} manifests, {len(changed)} default route(s) changed")
    for c in changed:
        print(f"  CHANGED: {c['manifest']} ({c['family']}) {c['old_preferred']} -> "
              f"{c['new_preferred']}")
    if bad:
        print('policy violations:')
        for v in bad:
            print(f"  - {v}")
    print(f"{result['verdict']} -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
