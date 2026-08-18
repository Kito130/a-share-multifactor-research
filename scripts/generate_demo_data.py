"""Generate a deterministic synthetic cross-sectional factor fixture."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data/demo_synthetic"
SEED = 20260818
SIGNAL_DATES = (
    "2020-01-31",
    "2020-02-28",
    "2020-03-31",
    "2020-04-30",
    "2020-05-29",
    "2020-06-30",
)
FIELDS = (
    "signal_date",
    "canonical_ts_code",
    "industry_code",
    "total_mv_cny",
    "universe_eligible",
    "bm_proxy",
    "bm_proxy_winsorized",
    "bm_proxy_z",
    "momentum_12_1",
    "momentum_12_1_winsorized",
    "momentum_12_1_z",
    "lowvol_60",
    "lowvol_60_winsorized",
    "lowvol_60_z",
    "next_month_return",
    "sample_scope",
)


def zscores(values: list[float]) -> list[float]:
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values)
    return [(value - mean) / deviation for value in values]


def build_rows() -> list[dict[str, object]]:
    rng = random.Random(SEED)
    rows: list[dict[str, object]] = []
    for month_index, signal_date in enumerate(SIGNAL_DATES):
        month: list[dict[str, object]] = []
        for stock_index in range(30):
            bm = 0.55 + 0.045 * (stock_index % 10) + rng.uniform(-0.03, 0.03)
            momentum = math.sin((stock_index + month_index * 2) / 4.5)
            momentum += rng.uniform(-0.12, 0.12)
            lowvol = -(0.12 + 0.008 * (stock_index % 7))
            lowvol += rng.uniform(-0.008, 0.008)
            month.append(
                {
                    "signal_date": signal_date,
                    "canonical_ts_code": f"SYN{stock_index + 1:04d}",
                    "industry_code": f"IND{stock_index % 5 + 1}",
                    "total_mv_cny": 2_000_000_000 + stock_index * 125_000_000,
                    "universe_eligible": stock_index % 17 != 0,
                    "bm_proxy": bm,
                    "momentum_12_1": momentum,
                    "lowvol_60": lowvol,
                }
            )
        for factor in ("bm_proxy", "momentum_12_1", "lowvol_60"):
            standardized = zscores([float(row[factor]) for row in month])
            for row, value in zip(month, standardized, strict=True):
                row[f"{factor}_winsorized"] = row[factor]
                row[f"{factor}_z"] = value
        for row in month:
            composite = statistics.fmean(
                float(row[name])
                for name in ("bm_proxy_z", "momentum_12_1_z", "lowvol_60_z")
            )
            row["next_month_return"] = (
                0.004 + 0.006 * composite + rng.uniform(-0.012, 0.012)
            )
            row["sample_scope"] = "fully_synthetic_software_demo"
            rows.append(row)
    return rows


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    with (OUTPUT_DIR / "monthly_factor_panel.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "classification": "SYNTHETIC",
        "formal_research_result": False,
        "generated_by": "scripts/generate_demo_data.py",
        "rows": len(rows),
        "seed": SEED,
        "signal_dates": list(SIGNAL_DATES),
        "stocks_per_cross_section": 30,
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"generated {len(rows)} synthetic rows in {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

