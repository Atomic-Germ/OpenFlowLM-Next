"""Verify control reproduction, common inputs and primary FP32 entry replay gates."""
import argparse
import json
from pathlib import Path
from entry_common import digest
from entry_check import verify_inputs


def compare_records(control,candidate,left_inputs,right_inputs):
    if control.get('mode')!='bf16' or not control.get('control_identical') or candidate.get('mode')!='fp32':
        raise ValueError('invalid or non-identical control')
    for k in ('layers','tokens','fixture_build_key'):
        if control[k]!=candidate[k]:raise ValueError('experiment identity mismatch')
    expected={(l,t) for l in control['layers'] for t in range(control['tokens'])}
    for r in (control,candidate):
        if not expected or len(r['results'])!=len(expected) or {(x['layer'],x['token']) for x in r['results']}!=expected:
            raise ValueError('incomplete result coverage')
    if not left_inputs or left_inputs!=right_inputs:raise ValueError('experiment input hashes differ')
    return dict(passed=candidate['passed'] and all(r['passed'] for r in candidate['results']),
                control_failed_cases=sum(not r['passed'] for r in control['results']),
                candidate_failed_cases=sum(not r['passed'] for r in candidate['results']),
                cases=len(expected),common_input_hashes=len(left_inputs),control_identical=True)


def load_verified(out):
    score=json.loads((out/'score.json').read_text());audit=json.loads((out/'audit.json').read_text())
    if digest(out/'audit.json')!=score['audit_sha256']:raise ValueError('audit hash mismatch')
    verify_inputs(audit,[Path(p) for p in audit['sha256']])
    for name,h in score['capture_sha256'].items():
        if digest(out/name)!=h:raise ValueError('capture hash mismatch')
    # Input paths outside the variant's output/build are required to agree exactly.
    shared={p:h for p,h in audit['sha256'].items()
            if not Path(p).is_relative_to(out) and not Path(p).is_relative_to(out.parent/(score['mode']+'-build'))}
    return score,shared


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--baseline-score',type=Path,required=True);a=ap.parse_args();root=a.root.resolve()
    out=root/'comparison.json';out.unlink(missing_ok=True)
    control,left=load_verified(root/'bf16');candidate,right=load_verified(root/'fp32')
    baseline=json.loads(a.baseline_score.read_text())
    if control['results']!=baseline['results'] or control['weight_sha256']!=baseline['weight_sha256']:
        raise ValueError('control oracle differs from retained baseline')
    if control['weight_sha256']!=candidate['weight_sha256']:raise ValueError('effective weights differ')
    r=compare_records(control,candidate,left,right)
    r['evidence_sha256']={str(p.resolve()):digest(p) for p in [root/'bf16/score.json',root/'fp32/score.json',a.baseline_score]}
    out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));return 0 if r['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
