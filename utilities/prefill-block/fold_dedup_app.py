"""Bounded real candidate application attempts, with live runtime provenance."""
import argparse
import os
from pathlib import Path
import signal
import threading
import time
import urllib.request
import json
import hashlib
import shutil

import pipeline_capture as C
import pipeline_evidence as E
import fold_dedup_experiment as X


def attempt(root, stage, arm, label=''):
    build = root / 'app-buildroot-v1'
    cli = build / 'build/src/oflm'
    out = root / f'app-{stage}-{arm}{label}'
    out.mkdir(exist_ok=False)
    settings = out / 'settings.json'
    settings.write_text('{}\n')
    store = out / 'store/models'
    store.mkdir(parents=True)
    local = store / C.MODEL.name
    local.mkdir()
    for source in C.MODEL.iterdir():
        if source.is_file():
            if source.name == 'model.q4nx':
                (local / source.name).symlink_to(source)
            else:
                shutil.copy2(source, local / source.name)
    metadata = E.read_json(C.ROOT / 'src/model_info.json')
    entries = []
    registry = E.read_json(C.ROOT / 'src/model_list.json')
    for name in registry['models']['qwen3.8']['27b']['files']:
        path = local / name
        entry = dict(type='file', path=name, size=path.stat().st_size)
        if name == 'model.q4nx':
            entry['lfs'] = dict(oid=E.sha256(path), size=path.stat().st_size)
        else:
            raw = path.read_bytes()
            entry['oid'] = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        entries.append(entry)
    metadata['qwen3.8:27b'] = entries
    C.write_json(out / 'model_info.json', metadata)
    env, removed = C.environment(out / 'unused', None, 'prefill', '0', 'block', X.THREADS,
                                 dedup_input_split=arm == 'on')
    for key in ('OFLM_OPEN_PROFILE', 'OFLM_OPEN_PROFILE_STEPS'):
        env.pop(key)
    env.update(OFLM_CONFIG_PATH=str(C.ROOT / 'src/model_list.json'),
               OFLM_MODELINFO_PATH=str(out / 'model_info.json'),
               OFLM_MODEL_PATH=str(out / 'store'), OFLM_SETTINGS_PATH=str(settings),
               OFLM_XCLBIN_PATH=str(C.ROOT / 'src'), OFLM_OPEN_KERNELS_DIR=str(X.KERNELS),
               LD_LIBRARY_PATH='/opt/xilinx/xrt/lib:' + str(build / 'src/lib/xrt'))
    cmd = [str(cli), stage, 'qwen3.8:27b', '--ctx-len', '4096', '--pmode', 'default']
    if stage == 'serve':
        cmd += ['--host', '127.0.0.1', '--port', '52639']
    inputs = [cli, build / 'source.tar', build / 'open-kernels-source.tar',
              build / 'LICENSE_OPEN_RUNTIME.md', build / 'build/CMakeCache.txt',
              C.ROOT / 'src/model_list.json', out / 'model_info.json']
    inputs += list((build / 'src/lib/xrt').glob('*.so*'))
    inputs += E.provenance_inputs(X.BASE / 'candidate/build/open_qwen36_cli',
        X.BASE / 'candidate/identity.json',
        E.build_identity(X.BASE / 'candidate/identity.json', X.BASE / 'candidate/build/open_qwen36_cli'),
        C.MODEL, X.KERNELS)
    before = E.snapshot(inputs)
    invocation = dict(argv=cmd, environment=env, sanitized_inherited_oflm=removed, before=before,
                      source_commit='c5d20c85', stage=stage, arm=arm,
                      power='Linux application default label; no power-policy write; unchanged/unverified')
    C.write_json(out / 'invocation.json', invocation)
    requests = []
    stop = threading.Event()

    def serve_requests():
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 180
            while not stop.is_set():
                try:
                    with opener.open('http://127.0.0.1:52639/v1/models', timeout=2) as r:
                        requests.append(dict(method='GET', path='/v1/models', status=r.status,
                                             body=r.read().decode()))
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise RuntimeError('server readiness deadline')
                    time.sleep(1)
            if stop.is_set():
                return
            for prompt in ('Reply with the word apple.', 'Reply with the word blue.',
                           'Reply with the word apple.'):
                payload = dict(model='qwen3.8:27b', messages=[dict(role='user', content=prompt)],
                               temperature=0, max_tokens=17, stream=False)
                req = urllib.request.Request('http://127.0.0.1:52639/v1/chat/completions',
                    data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
                with opener.open(req, timeout=600) as response:
                    requests.append(dict(method='POST', path='/v1/chat/completions', request=payload,
                        status=response.status, body=json.loads(response.read())))
                C.write_json(out / 'requests.json', requests)
        except Exception as exc:
            requests.append(dict(error=repr(exc)))
        finally:
            C.write_json(out / 'requests.json', requests)
            children = Path(f'/proc/{os.getpid()}/task/{os.getpid()}/children').read_text().split()
            for child in children:
                try:
                    if Path(f'/proc/{child}/exe').resolve() == cli.resolve():
                        os.kill(int(child), signal.SIGINT)
                except ProcessLookupError:
                    pass

    stdin = out / 'stdin.txt'
    stdin.write_text('/set temp 0\n/set gen-lim 17\nReply with the word apple.\n/bye\n')
    with C.lock_exclusive(C.LOCK):
        X.preflight()
        worker = None
        if stage == 'serve':
            worker = threading.Thread(target=serve_requests)
            worker.start()
        try:
            with stdin.open('rb') as input_file:
                result = C.capture_process(cmd, env, out / 'run.log', timeout=2100,
                    executable=before[str(cli)], stdin=input_file,
                    allowed_returncodes=(0, -signal.SIGINT) if stage == 'serve' else (0,))
            C.write_json(out / 'capture.json', dict(**invocation, **result,
                after=E.snapshot(inputs), acceptance='NOT AUTOMATIC: inspect actual inference output'))
        except Exception as exc:
            C.write_json(out / 'failed.json', dict(error=repr(exc), after=E.snapshot(inputs)))
            raise
        finally:
            stop.set()
            if worker:
                worker.join(timeout=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=['run', 'serve'], required=True)
    parser.add_argument('--arm', choices=['off', 'on'], required=True)
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    attempt(args.root.absolute(), args.stage, args.arm, args.label)
