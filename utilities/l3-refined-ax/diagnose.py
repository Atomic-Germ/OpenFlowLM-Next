"""Separate integrated preparation and FP32-add rounding using retained real inputs."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import numpy as np
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_refined_diagnosis',HERE/'replay.py');r=importlib.util.module_from_spec(s);s.loader.exec_module(r)


def compare(table,high_table,low_table,got,expected):
    if len(table)!=23040 or len(high_table)!=11520 or len(low_table)!=11520 or any(v.shape!=(14336,) or v.dtype!=np.dtype('<f4') or not np.isfinite(v).all() for v in (got,expected)):
        raise ValueError('invalid complete probe captures')
    return dict(high_table_exact=table[:11520]==high_table,low_table_exact=table[11520:]==low_table,
                sum_exact=got.tobytes()==expected.tobytes(),sum_differing_values=int(np.count_nonzero(got.view('<u4')!=expected.view('<u4'))),
                sum_maxabs=float(np.max(abs(got.astype(float)-expected))))


def check(out,prep):
    source=Path(prep['source']);original=Path(prep['original']);rows=[]
    names={f'{k}{p}.bin' for k in ('y','tab') for p in range(240)}
    if {p.name for pattern in ('y*.bin','tab*.bin') for p in out.glob(pattern)}!=names:raise ValueError('incomplete probe inventory')
    for p in range(240):
        high,low=[r.tail.refine.read(source/f'{k}-y{p}.bin',14336) for k in ('hi','lo')]
        rows.append(dict(position=p,**compare((out/f'tab{p}.bin').read_bytes(),(source/f'hi-table{p}.bin').read_bytes(),
              (source/f'lo-table{p}.bin').read_bytes(),r.tail.refine.read(out/f'y{p}.bin',14336),r.tail.refine.combine(high,low))))
    return dict(cases=rows,high_tables=sum(c['high_table_exact'] for c in rows),low_tables=sum(c['low_table_exact'] for c in rows),
                sums=sum(c['sum_exact'] for c in rows),different_sum_values=sum(c['sum_differing_values'] for c in rows),
                artifacts={n:r.digest(out/n) for n in sorted(names)})



def audit(out):
    proof=dict(complete=False,model_acceptance=False)
    try:
        record=json.loads((out/'report.json').read_text())
        if (record.get('complete') is not True or record.get('model_acceptance') is not False or
            record['log_sha256']!=r.digest(out/'native.log') or record['harness_sha256']!=r.digest(record['harness']) or
            record['harness_evidence_sha256']!=r.native.verify_harness(record['harness'])):raise ValueError('unbound probe run')
        r.base.verify(record['inputs']);result=check(out,record)
        if result!=record['result']:raise ValueError('changed diagnostic measurements')
        r.base.verify(record['inputs'])
        proof.update(complete=True,report_sha256=r.digest(out/'report.json'),artifacts=len(result['artifacts']),
                     high_tables=result['high_tables'],low_tables=result['low_tables'],sums=result['sums'])
    except Exception as e:proof['error']=str(e);raise
    finally:r.native.write_report(out/'audit.json',proof)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--build',type=Path);p.add_argument('--source',type=Path);p.add_argument('--harness',type=Path);p.add_argument('--audit',action='store_true')
    a=p.parse_args();out=a.out.resolve()
    if a.audit:return audit(out)
    if any(v is None for v in (a.build,a.source,a.harness)):p.error('build, source and harness required')
    out.mkdir(parents=True,exist_ok=False);record=dict(complete=False,model_acceptance=False)
    try:
        source=a.source.resolve();_,upstream=r.tail.attn.validate_source(source)
        old=r.tail.refine.verified(source);original=Path(old['source']);build=a.build.resolve()
        b=json.loads((build/'build.json').read_text())
        if not b.get('complete'):raise ValueError('incomplete diagnostic build')
        inputs=b['inputs']|b['sha256']|upstream|old['inputs']|old['artifacts']|old['sources']
        for path in (Path(__file__),HERE/'replay.py',build/'build.json'):inputs[str(path.resolve())]=r.digest(path)
        for n in r.tail.refine.names():inputs[str(source/n)]=r.digest(source/n)
        r.base.verify(inputs);h=a.harness.resolve();evidence=r.native.verify_harness(h)
        lines=['device',f'xclbin P {build}/final.xclbin',f'kernelx probe P {build}/insts.bin',
               'buf x 20480','buf pairs 114688','buf y 57344','buf tab 23040']
        for pos in range(240):
            parts=[r.tail.refine.read(source/f'{k}-y{pos}.bin',14336).reshape(224,64) for k in ('hi','lo')]
            path=out/f'pair{pos}.bin';np.stack(parts,axis=1).tofile(path);inputs[str(path)]=r.digest(path)
            lines += [f'load x {original}/x{pos}.bin',f'load pairs {out}/pair{pos}.bin',
                      'run probe x pairs y tab',f'dump y {out}/y{pos}.bin 57344',f'dump tab {out}/tab{pos}.bin 23040']
        (out/'run.cfg').write_text('\n'.join(lines)+'\n');inputs[str(out/'run.cfg')]=r.digest(out/'run.cfg')
        record.update(source=str(source),original=str(original),inputs=inputs,harness=str(h),harness_sha256=r.digest(h),harness_evidence_sha256=evidence)
        with r.native.drift.drift.boundary.runner.device_lock(),(out/'native.log').open('w') as log:
            subprocess.run([str(h),str(out/'run.cfg')],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,HARNESS_TIMEOUT_MS='30000'),check=True,timeout=1800)
        result=check(out,record);r.base.verify(inputs)
        if r.digest(h)!=record['harness_sha256'] or r.native.verify_harness(h)!=evidence:raise ValueError('changed harness')
        record.update(complete=True,result=result,log_sha256=r.digest(out/'native.log'))
    except Exception as e:record['error']=str(e);raise
    finally:r.native.write_report(out/'report.json',record)

if __name__=='__main__':main()
