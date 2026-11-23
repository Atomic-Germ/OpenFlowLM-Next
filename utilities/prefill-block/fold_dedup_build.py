"""Build immutable PR4B git archives with before/after native input attestation."""
import argparse
import io
import shutil
import subprocess
import tarfile
from pathlib import Path

import pipeline_capture as C
import pipeline_evidence as E


def build(revision, out):
    out = out.absolute()
    out.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(['git', 'rev-parse', revision], text=True).strip()
    archive = subprocess.check_output(['git', 'archive', revision, 'src/open_qwen36',
                                      'src/include', 'src/gguf', 'open_kernels/harness',
                                      'utilities/prefill-block/build-cli.sh'])
    (out / 'sources.tar').write_bytes(archive)
    source = out / 'source'
    source.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        tf.extractall(source, filter='data')
    sources = E.tree_files(source)
    compiler = Path(shutil.which('g++')).resolve()
    headers = Path('/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/include')
    inputs = [compiler, *E.tree_files(headers)]
    for pattern in ('libxrt*.so*',):
        inputs += [p.resolve() for p in Path('/opt/xilinx/xrt/lib').glob(pattern) if p.is_file()]
    for name in ('cc1plus', 'libstdc++.so', 'libgomp.so', 'libgcc_s.so'):
        p = Path(subprocess.check_output([str(compiler), '-print-file-name=' + name], text=True).strip())
        if p.is_file():
            inputs.append(p.resolve())
    before = E.snapshot(sources + inputs)
    C.write_json(out / 'before.json', before)
    cmd = ['bash', str(source / 'utilities/prefill-block/build-cli.sh'), str(out / 'build')]
    with (out / 'build.log').open('w') as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    after = E.snapshot(sources + inputs)
    C.write_json(out / 'after.json', after)
    E.require(before == after, 'build inputs changed')
    exe = out / 'build/open_qwen36_cli'
    inventory = lambda paths: {str(p): before[str(p)]['sha256'] for p in paths}
    identity = dict(version=1, source_revision=revision, executable_sha256=E.sha256(exe),
                    build_command=cmd, source_files=inventory(sources),
                    build_files=inventory(inputs), build_inputs=inventory(inputs),
                    compiler_version=subprocess.check_output([str(compiler), '--version'], text=True),
                    source_archive_sha256=E.sha256(out / 'sources.tar'))
    C.write_json(out / 'identity.json', identity)
    E.build_identity(out / 'identity.json', exe)
    for p in sources:
        p.chmod(p.stat().st_mode & ~0o222)
    print(exe, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--revision', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    build(a.revision, a.out)
