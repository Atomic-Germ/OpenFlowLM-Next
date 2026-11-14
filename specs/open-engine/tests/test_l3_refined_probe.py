import importlib.util
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]


def load():
    s=importlib.util.spec_from_file_location('refined_probe_check',ROOT/'utilities/l3-refined-ax/diagnose.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_probe_separates_preparation_and_sum_failures():
    m=load();high=bytes(11520);low=bytes([1])*11520;y=np.ones(14336,dtype='<f4')
    r=m.compare(high+low,high,low,y,y)
    assert r==dict(high_table_exact=True,low_table_exact=True,sum_exact=True,sum_differing_values=0,sum_maxabs=0.)
    tab=bytearray(high+low);tab[11520]=2
    r=m.compare(bytes(tab),high,low,y,y)
    assert r['high_table_exact'] and not r['low_table_exact'] and r['sum_exact']
    changed=y.copy();changed[7]=np.nextafter(changed[7],np.float32(2))
    r=m.compare(high+low,high,low,changed,y)
    assert r['high_table_exact'] and r['low_table_exact'] and not r['sum_exact']
    assert r['sum_differing_values']==1
    with pytest.raises(ValueError):m.compare(high,high,low,y,y)
