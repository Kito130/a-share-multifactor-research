from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "data" / "demo_synthetic"


def _run_script(name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / name)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_synthetic_fixture_contract() -> None:
    generated = _run_script("generate_demo_data.py")
    assert "generated 180 synthetic rows" in generated.stdout

    metadata = json.loads((DEMO_DIR / "metadata.json").read_text(encoding="utf-8"))
    panel = pd.read_csv(DEMO_DIR / "monthly_factor_panel.csv")

    assert metadata["classification"] == "SYNTHETIC"
    assert metadata["formal_research_result"] is False
    assert metadata["seed"] == 20260818
    assert metadata["rows"] == len(panel) == 180
    assert panel["signal_date"].nunique() == 6
    assert panel["canonical_ts_code"].nunique() == 30
    assert set(panel["sample_scope"]) == {"fully_synthetic_software_demo"}
    assert panel["canonical_ts_code"].str.fullmatch(r"SYN\d{4}").all()


def test_public_factor_diagnostic_contract() -> None:
    _run_script("generate_demo_data.py")
    completed = _run_script("run_public_demo.py")
    summary = json.loads(completed.stdout)

    assert summary == {
        "classification": "SYNTHETIC_SOFTWARE_DEMO",
        "factor_count": 3,
        "formal_oos_reproduction": False,
        "monthly_ic_rows": 18,
        "panel_rows": 180,
        "signal_months": 6,
        "top_minus_bottom_rows": 18,
    }

