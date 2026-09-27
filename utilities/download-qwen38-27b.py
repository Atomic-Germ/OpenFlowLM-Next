#!/usr/bin/env python3
"""Download pinned Qwen38 config/tokenizer and ggml-org Q8 GGUF; keep weights local."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

SOURCES = [
    ('Qwen/Qwen3.8-27B','1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
     ['config.json','tokenizer.json','tokenizer_config.json','generation_config.json','chat_template.jinja','LICENSE','README.md','model.safetensors.index.json']),
    ('ggml-org/Qwen3.8-27B-GGUF','71bc7b627595dc8a91039addd9c791ae548d6747',
     ['Qwen3.8-27B-Q8_0.gguf','.src_sha']),
]


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def valid(path,record):
    if not path.exists() or path.stat().st_size!=record['size']: return False
    expected=record.get('lfs',{}).get('sha256')
    if expected: return sha(path)==expected
    digest=hashlib.sha1(f"blob {record['size']}\0".encode(),usedforsecurity=False)
    with path.open('rb') as f:
        while chunk:=f.read(1024*1024): digest.update(chunk)
    return digest.hexdigest()==record['blobId']


def download(repo,revision,record,dest):
    name=record['rfilename'];expected=record.get('lfs',{}).get('sha256');size=record['size']
    if valid(dest,record): return
    partial=dest.with_name(dest.name+'.partial')
    for attempt in range(5):
        offset=partial.stat().st_size if partial.exists() else 0
        if offset==size: break
        if offset>size: raise ValueError(f'oversized partial file: {partial}')
        url=f'https://huggingface.co/{repo}/resolve/{revision}/{name}?download=true&attempt={attempt}'
        request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
        try:
            with urllib.request.urlopen(request,timeout=60) as response:
                resume=response.status==206
                if resume and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                    raise ValueError('incorrect resume range')
                with partial.open('ab' if resume else 'wb') as f:
                    last=f.tell()//(1024**3)
                    while data:=response.read(4*1024**2):
                        f.write(data)
                        if f.tell()//(1024**3)>last:
                            last=f.tell()//(1024**3)
                            print(name,f'{f.tell()/1e9:.2f}/{size/1e9:.2f} GB',flush=True)
            if partial.stat().st_size==size: break
        except (OSError,TimeoutError) as e:
            print('Retry',attempt+1,name,str(e),flush=True)
            time.sleep(2)
    if not partial.exists() or partial.stat().st_size!=size: raise ValueError(f'incomplete download: {name}')
    if not valid(partial,record): raise ValueError(f'content hash mismatch: {name}; remove the partial file before retrying')
    partial.replace(dest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'Models/qwen38-27b/source')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    provenance=[]
    for repo,revision,names in SOURCES:
        with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true',timeout=60) as r:
            info=json.load(r)
        for name in names:
            record=next(v for v in info['siblings'] if v['rfilename']==name)
            dest=args.out/name
            download(repo,revision,record,dest)
            provenance.append(dict(repo=repo,revision=revision,file=name,bytes=dest.stat().st_size,sha256=sha(dest)))
            print('Verified',name,flush=True)
    (args.out/'sources.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__': main()
