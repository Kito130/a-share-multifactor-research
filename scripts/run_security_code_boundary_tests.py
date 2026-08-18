from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from a_share_p2.boundary import run_boundary_tests


if __name__ == "__main__":
    summary, tests = run_boundary_tests()
    print(
        "P2代码边界闸门："
        f"{int((summary['boundary_status'] == 'PASS').sum())} 主体 PASS；"
        f"{int((tests['status'] == 'PASS').sum())} 项 PASS；"
        f"{int((tests['status'] == 'FAIL').sum())} 项 FAIL"
    )
