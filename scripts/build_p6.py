from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from a_share_p6.build import build_p6


if __name__ == "__main__":
    manifest = build_p6()
    print(
        "[P6] status="
        f"{manifest['status']} "
        f"experiments={manifest['experiment_count']} "
        f"p5_hashes_match={manifest['p5_output_hashes_all_match']}"
    )
