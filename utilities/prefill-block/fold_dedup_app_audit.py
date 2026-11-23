"""Audit actual application outputs; missing evidence never passes."""
import argparse
from pathlib import Path
import re

import pipeline_capture as C
import pipeline_evidence as E


def audit(root):
    records = {}
    requests = {}
    traces = {}
    for arm in ('off', 'on'):
        directory = root / f'app-serve-{arm}-local-v4'
        record = E.read_json(directory / 'capture.json')
        E.require(record['before'] == record['after'], 'application inputs changed')
        E.validate_runtime(record['runtime'])
        records[f'serve-{arm}'] = record
        import json
        rows = json.loads((directory / 'requests.json').read_text())
        E.require(len(rows) == 4 and all(r.get('status') == 200 for r in rows),
                  'missing/failed server requests')
        posts = rows[1:]
        E.require(all(r['body']['usage']['completion_tokens'] == 17 for r in posts),
                  'incomplete server continuation')
        outputs = [r['body']['choices'] for r in posts]
        E.require(outputs[0] == outputs[2], 'server A/B/A differs')
        requests[arm] = posts
        text = (directory / 'run.log').read_text()
        E.require(not re.search(r'ERT_CMD_STATE_TIMEOUT|NOAVAIL|request failed:', text),
                  'application hardware/request error')
        segments = re.split(r'\[OFLM\]\s+Start prefill\.\.\.', text)[1:]
        E.require(len(segments) == 3, 'missing request trace boundaries')
        trace = [[int(t) for t in re.findall(r'open_qwen36: embed t=(\d+)', s)] for s in segments]
        E.require(all(len(t) >= 56+16 for t in trace), 'incomplete consumed-token trace')
        E.require(trace[0] == trace[2], 'server consumed-token A/B/A differs')
        traces[arm] = trace
    E.require(traces['off'] == traces['on'], 'server consumed engine IDs differ')
    for i in range(3):
        E.require(requests['off'][i]['request'] == requests['on'][i]['request'], 'API request differs')
        E.require(requests['off'][i]['body']['choices'] == requests['on'][i]['body']['choices'],
                  'server OFF/ON output differs')
    run_traces = {}
    for arm in ('off', 'on'):
        directory = root / f'app-run-{arm}-local-v2'
        record = E.read_json(directory / 'capture.json')
        E.require(record['before'] == record['after'], 'run inputs changed')
        E.validate_runtime(record['runtime'])
        text = (directory / 'run.log').read_text()
        E.require(re.search(r'Model RAW Output:\s*\napple\s*\n', text), 'short application answer absent')
        E.require(not re.search(r'ERT_CMD_STATE_TIMEOUT|NOAVAIL|request failed:', text), 'run error')
        run_traces[arm] = re.findall(r'open_qwen36: embed t=(\d+)', text)
        E.require(len(run_traces[arm]) > 10, 'run trace absent')
        records[f'run-{arm}'] = record
    E.require(run_traces['off'] == run_traces['on'], 'run OFF/ON consumed engine IDs differ')
    runtime = records['serve-off']['runtime']['before']
    E.require(all(r['runtime']['before'] == runtime for r in records.values()), 'application runtimes differ')
    numerical = E.read_json(root/'numerical-v1/isolation/original-fresh-a/capture.json')
    E.require(numerical['runtime']['before'] == runtime, 'numerical/application runtime differs')
    captures = [root/f'app-{key}-local-{"v4" if key.startswith("serve") else "v2"}/capture.json'
                for key in records]
    result = dict(passed=True, captures=E.snapshot(captures),
        application_control='same exact c5d20c85 application OFF; timing baseline remains ad867144',
        server_requests='distinct A/B/A, 17 completion tokens each, temperature zero, both arms',
        consumed_engine_ids=traces, run_consumed_engine_ids=run_traces,
        scope='exact API choices and consumed engine-token traces; no API logits or complete state dumps',
        final_sampled_id='Not independently exposed by API; complete 17-ID greedy trajectory covered by numerical CLI gates',
        scratch_metadata='local file sizes and git/LFS hashes; shipped metadata differs; not an unmodified-install gate')
    C.write_json(root/'application-passed.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root.absolute())
