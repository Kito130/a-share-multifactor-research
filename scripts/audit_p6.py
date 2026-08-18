from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from a_share_p6.audit import audit_p6


if __name__ == "__main__":
    manifest = audit_p6()
    audit = manifest["audit"]
    print(
        "[P6 AUDIT] "
        f"{manifest['status']} "
        f"PASS={audit['pass_count']} "
        f"WARN={audit['warn_count']} "
        f"FAIL={audit['fail_count']}"
    )
