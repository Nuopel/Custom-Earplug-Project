from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACOUSTMM_SRC = ROOT / "src"
TOOLKITS_ROOT = ROOT.parent

candidate_paths = [ACOUSTMM_SRC]
candidate_paths.extend(sorted(TOOLKITS_ROOT.glob("Toolkitsd_*/src")))

for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)
