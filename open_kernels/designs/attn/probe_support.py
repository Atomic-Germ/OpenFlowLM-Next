"""Diagnostic extraction of the production ax worker; no duplicated attention math."""
import ast
from pathlib import Path

AX = Path(__file__).resolve().parents[1] / 'layer_x/ax.py'


def worker(d, loop):
    tree = ast.parse(AX.read_text())
    ax = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'ax')
    body = next(n for n in ax.body if isinstance(n, ast.FunctionDef) and n.name == '_attn')
    ns = dict(D=d, NHL=d.NHL, KVW=d.KVW, RB=d.RB, ACORES=d.ACORES,
              N_OG=d.NHL // d.HPO, range_=loop)
    exec(compile(ast.Module(body=[body], type_ignores=[]), str(AX), 'exec'), ns)
    return ns['_attn']


def sequence(d, rows, pipe, tap, meta, qg, kvn, cache, new, og, inp, outputs):
    pipe.drain(outputs[0], new, tap(2*d.KVW, 0, 2*d.KVW))
    for c, out in enumerate(outputs):
        pipe.drain(out, og, tap(d.QW, c*d.NHL*d.HD, d.NHL*d.HD))
    pipe.fill(inp, meta, tap(2*d.E_A, 0, 2*d.E_A))
    pipe.fill(inp, qg, tap(2*d.QW, 0, d.QW))
    pipe.fill(inp, kvn, tap(2*d.KVW, 0, d.KVW))
    pipe.fill(inp, kvn, tap(2*d.KVW, d.KVW, d.KVW))
    pipe.fill(inp, cache, tap(rows*2*d.KVW, 0, rows*2*d.KVW))
    pipe.fill(inp, qg, tap(2*d.QW, d.QW, d.QW))
    pipe.finish()
