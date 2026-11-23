"""Final content/provenance audit; reads archives and preserves every raw capture."""
import argparse
import subprocess
from pathlib import Path

import pipeline_capture as C
import pipeline_evidence as E
import fold_dedup_experiment as X


def audit(root):
    E.require(E.read_json(root/'application-passed.json')['passed'], 'application gate absent')
    E.require(E.read_json(root/'numerical-v1/numerical-passed.json')['passed'], 'numerical gate absent')
    E.require(E.read_json(root/'timing-v1/completed.json')['completed'], 'timing incomplete')
    current = {}
    for arm in ('control', 'candidate'):
        build = X.BASE / arm
        cli, identity_path = build/'build/open_qwen36_cli', build/'identity.json'
        identity = E.build_identity(identity_path, cli)
        snapshot = E.snapshot(E.provenance_inputs(cli, identity_path, identity, C.MODEL, X.KERNELS))
        E.validate_attested_snapshot(snapshot, cli, identity_path, identity)
        current.update(snapshot)
    rows = E.read_json(root/'timing-v1/selection.json')['samples']
    runtime = E.read_json(root/'recovery-cross-audit.json')['runtime']
    for row in rows:
        directory = Path(row['capture'])
        record = E.read_json(directory/'capture.json')
        E.require(E.sha256(directory/'capture.json') == row['capture_sha256'], 'capture metadata changed')
        E.require(record['before'] == record['after'], 'capture inputs changed')
        E.require(record['runtime']['before'] == runtime, 'capture runtime differs')
        E.validate_runtime(record['runtime'])
        for path, value in record['before'].items():
            E.require(current[path] == value, 'current attested input differs: ' + path)
        for name, digest in record['raw_hashes'].items():
            E.require(E.sha256(directory/name) == digest, 'raw changed: ' + str(directory/name))
    previous = []
    for name in ('correctness', 'campaign', 'continuation-v1', 'recovery-20261123-1246'):
        for path in (X.BASE/name).rglob('capture.json'):
            record = E.read_json(path)
            for raw, digest in record.get('raw_hashes', {}).items():
                E.require(E.sha256(path.parent/raw) == digest, 'old raw changed: ' + str(path.parent/raw))
            previous.append(dict(path=str(path), sha256=E.sha256(path),
                                 declared_raw_hashes=len(record.get('raw_hashes', {}))))
    old_reports = {}
    for name in ('qwen38-host-fold-dedup.md', 'qwen38-host-fold-dedup.json',
                 'qwen38-host-fold-final-acceptance-plan.md'):
        path = C.ROOT/'specs/open-engine/reports'/name
        if path.exists():
            committed = subprocess.run(['git', 'show', '037f2fcb:'+str(path.relative_to(C.ROOT))],
                                       check=True, capture_output=True).stdout
            E.require(path.read_bytes() == committed, 'previous committed report changed')
            old_reports[str(path)] = E.sha256(path)
    policy = E.read_json(root/'timing-v1/policy.json')
    for path, value in policy['tool_snapshot'].items():
        E.require(E.fingerprint(path) == value, 'pinned timing tool changed')
    C.write_json(root/'final-audit.json', dict(passed=True, timing_captures=len(rows),
        current_attested_inputs=current, runtime=runtime, previous_captures=previous,
        committed_reports_unchanged=old_reports,
        config_incident=E.read_json(root/'model-identity-incident.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root.absolute())
