"""Write a safetensors/Q4NX container without retaining all tensors in RAM."""
import json
from pathlib import Path
import shutil
import struct
import tempfile

import torch


class TensorWriter:
    TYPES = {torch.bfloat16:'BF16',torch.float32:'F32',torch.int8:'I8',torch.uint8:'U8'}

    def __init__(self,path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.exists(): raise FileExistsError(self.path)
        self.data=tempfile.TemporaryFile(dir=self.path.parent)
        self.header={}

    def __enter__(self): return self

    def add(self,name,parts):
        if name in self.header or name=='__metadata__': raise ValueError(f'duplicate/reserved tensor: {name}')
        start=self.data.tell();shape=None;dtype=None
        for tensor in parts:
            tensor=tensor.detach().cpu().contiguous()
            if tensor.ndim==0 or tensor.dtype not in self.TYPES: raise ValueError('unsupported streamed tensor dtype/rank')
            if shape is None: shape=list(tensor.shape);shape[0]=0;dtype=tensor.dtype
            if list(tensor.shape[1:])!=shape[1:] or tensor.dtype!=dtype: raise ValueError('inconsistent tensor parts')
            shape[0]+=tensor.shape[0]
            self.data.write(memoryview(tensor.view(torch.uint8).numpy()).cast('B'))
        if shape is None: raise ValueError('empty tensor stream')
        self.header[name]=dict(dtype=self.TYPES[dtype],shape=shape,data_offsets=[start,self.data.tell()])

    def __exit__(self,kind,value,traceback):
        try:
            if kind is None:
                header=json.dumps(self.header,separators=(',',':')).encode()
                header+=b' '*((-len(header))%8)
                # Keep the final path absent until the complete container is on disk.
                with tempfile.NamedTemporaryFile(dir=self.path.parent,delete=False) as f:
                    temp=Path(f.name)
                    try:
                        f.write(struct.pack('<Q',len(header)));f.write(header)
                        self.data.seek(0);shutil.copyfileobj(self.data,f,4*1024*1024)
                    except BaseException:
                        temp.unlink(missing_ok=True);raise
                temp.replace(self.path)
        finally: self.data.close()
