#!/usr/bin/env python3
"""oflm-add - install a pre-converted OFLM (Q4NX) model and register it with OpenFlowLM.

Shim for running the tool from a repo checkout without installing it. The
implementation lives in the installable ``oflm_add`` package (``uv tool install
oflm-add`` / ``uv tool install .``). Copies of the old standalone script that
were packed with model repos remain self-contained and keep working.
"""

from oflm_add import main

if __name__ == "__main__":
    main()
