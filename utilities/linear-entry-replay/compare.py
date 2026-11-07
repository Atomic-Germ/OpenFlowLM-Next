"""Compare pinned linear-entry variants using the shared controlled-replay checks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'attention-entry-replay'))
from entry_compare import main

if __name__=='__main__':raise SystemExit(main())
