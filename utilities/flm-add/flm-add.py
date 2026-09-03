#!/usr/bin/env python3
"""flm-add - install a pre-converted FLM (Q4NX) model and register it with FastFlowLM.

Shim for running the tool from a repo checkout without installing it. The
implementation lives in the installable ``flm_add`` package (``uv tool install
flm-add`` / ``uv tool install .``). Copies of the old standalone script that
were packed with model repos remain self-contained and keep working.
"""

from flm_add import main

if __name__ == "__main__":
    main()
