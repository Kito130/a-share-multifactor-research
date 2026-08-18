"""Run a data-safe factor diagnostic over the synthetic public fixture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from a_share_p2.research import _monthly_ic, _quintile_returns  # noqa: E402


def main() -> int:
    panel = pd.read_csv(
        PROJECT_ROOT / "data/demo_synthetic/monthly_factor_panel.csv",
        parse_dates=["signal_date"],
    )
    config = {
        "statistics": {
            "rank_ic_method": "spearman",
            "minimum_cross_section_observations": 5,
            "quintiles": 5,
        }
    }
    monthly_ic = _monthly_ic(panel, config)
    quintiles = _quintile_returns(panel, config)
    top_minus_bottom = quintiles.loc[
        quintiles["quintile"] == "TOP_MINUS_BOTTOM"
    ]
    summary = {
        "classification": "SYNTHETIC_SOFTWARE_DEMO",
        "formal_oos_reproduction": False,
        "factor_count": int(monthly_ic["factor"].nunique()),
        "monthly_ic_rows": int(len(monthly_ic)),
        "panel_rows": int(len(panel)),
        "signal_months": int(panel["signal_date"].nunique()),
        "top_minus_bottom_rows": int(len(top_minus_bottom)),
    }
    expected = {
        "factor_count": 3,
        "monthly_ic_rows": 18,
        "panel_rows": 180,
        "signal_months": 6,
        "top_minus_bottom_rows": 18,
    }
    if any(summary[key] != value for key, value in expected.items()):
        raise RuntimeError(f"synthetic demo contract failed: {summary}")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

