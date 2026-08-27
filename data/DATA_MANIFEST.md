# Public Data Manifest

| Path | Classification | Source | Publication rationale | Limitation |
| --- | --- | --- | --- | --- |
| `data/demo_synthetic/metadata.json` | `SYNTHETIC` | `scripts/generate_demo_data.py` | Records the deterministic fixture seed, dates, and row counts | Not a formal research result |
| `data/demo_synthetic/monthly_factor_panel.csv` | `SYNTHETIC` | `scripts/generate_demo_data.py` | Exercises the public factor-diagnostic contract with invented securities and returns | Cannot reproduce the licensed-data backtest |
| `results/p5_oos/*.csv` | `DERIVED_AGGREGATE` | Frozen private licensed-data run | Publishes bounded aggregate evidence without security-level panels, orders, or holdings | Not live performance, capacity evidence, or redistributable source data |
| `reports/oos_summary.md` and `reports/oos/p5_result_report.md` | `DERIVED_AGGREGATE` | Frozen private licensed-data run | Documents the published OOS metrics and research limitations | Does not make the private inputs reproducible |
| Raw, static, manual, and processed market-data paths | `PRIVATE_OR_LICENSED_EXCLUDED` | Licensed providers and local research inputs | Not included because redistribution rights do not follow from data access | Required for the formal backtest and private-stage tests |

The synthetic files contain no real security identifiers, market observations,
orders, holdings, or performance. The derived OOS files are retained only as
aggregate research evidence. The MIT License applies to original code, not to
excluded market data or third-party materials; see [`../DATA_LICENSE.md`](../DATA_LICENSE.md).
