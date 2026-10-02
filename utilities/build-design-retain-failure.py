#!/usr/bin/env python3
"""Run build_design.py but retain failed IRON artifacts for ELF size diagnosis.

Usage: ironvenv/bin/python utilities/build-design-retain-failure.py DESIGN OUT
Uses the installed mlir-aie cleanup hook only in this diagnostic process.
"""
from pathlib import Path
import runpy
import aie.utils.compile.jit.compilabledesign as compiler

if __name__=='__main__':
    cleanup=compiler._cleanup_failed_compilation
    compiler._cleanup_failed_compilation=lambda path: print(f'Retained failed build: {path}',flush=True)
    try:
        runpy.run_path(str(Path(__file__).resolve().parents[1]/'open_kernels/build_design.py'),run_name='__main__')
    finally:
        compiler._cleanup_failed_compilation=cleanup
