#!/usr/bin/env python3
"""Legacy entry point. All argument parsing and conversion logic lives in
q4nx.cli (the installed `q4nx-build` console script); this wrapper keeps
`python convert.py ...` invocations from the README working.
"""
import sys

from q4nx.cli import main

if __name__ == "__main__":
    sys.exit(main())
