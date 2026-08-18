# A-Share Multifactor Research

Point-in-time A-share cross-sectional research on whether a simple combination
of value, momentum, and low-volatility signals can produce risk-adjusted excess
return after realistic execution constraints.

This repository publishes the V1 research surface only. It contains the frozen
research protocol, curated aggregate evidence, and a deterministic synthetic
software demo. Licensed market data, raw security-level panels, local
credentials, and the A-share machine-learning/V3 experiments are excluded.

Chinese documentation: [README.zh-CN.md](README.zh-CN.md)

## Research Question

The strategy ranks the eligible A-share universe each month using three
cross-sectional signals:

- `B/M proxy`: `1 / historical PB`, with positive PB required;
- `12-1 momentum`: approximately twelve months of adjusted-price return,
  skipping the most recent month;
- `60-day low volatility`: the negative sample standard deviation of recent
  daily returns.

The signals are winsorized at the 1st and 99th percentiles, standardized within
each month, and combined with equal weights. The portfolio holds the top 100
names with a maximum target weight of 2% per name.

## Study Design

| Stage | Dates | Purpose |
| --- | --- | --- |
| Research | 2016-01-01 to 2019-12-31 | factor diagnostics and portfolio design |
| Validation | 2020-01-01 to 2021-12-31 | one-time validation under frozen rules |
| Final OOS | 2022-01-01 to 2025-12-31 | one authorized final evaluation |

The validation period was not used for parameter retuning. The final OOS used
the frozen configuration and was run once. Later robustness and report work is
classified as post-OOS analysis; it does not create a new untouched OOS.

## Data Flow

```text
licensed point-in-time inputs
        |
        v
P1: normalized daily panel + code-history + execution status
        |
        v
P2: month-end panel -> factor values -> cross-sectional IC/quintiles
        |
        v
P3: equal-weight composite -> next-open orders -> costs/cash/actions
        |
        v
P4: validation checks -> frozen configuration
        |
        v
P5: one-shot final OOS -> aggregate performance and audit manifest
```

Signals are formed at month-end. Orders are executed on the next market
trading day open. The backtest models commissions/slippage, historical stamp
duty, suspensions, open price limits, failed orders, cash, stale valuation, and
the documented 600270.SH to 601598.SH corporate action. This is a research
backtest, not live performance or a capacity study.

## Frozen OOS Result

The baseline cost scenario is 10 bps one-way commission/slippage. The figures
below are derived aggregates from a private licensed-data run and are included
only so the published claim can be checked against machine-readable evidence.

| Metric | Strategy | Reference |
| --- | ---: | --- |
| Total return | 35.85% | 2022-01-04 to 2025-12-31 |
| Annualized return | 8.29% | 969 trading days |
| Annualized return difference | +8.52 pp | versus CSI All Share (`000985.CSI`) |
| Zero-risk-free-rate Sharpe | 0.543 | daily returns, annualized |
| Maximum drawdown | -19.81% | within final OOS |
| Annualized volatility | 17.49% | daily returns, annualized |
| Total trading cost | CNY 4,499,738.44 | baseline scenario |

The benchmark had a -0.85% total return over the same period. The strategy's
2025 return was below the benchmark, so the result is not a claim of stable
outperformance in every calendar year.

Evidence: [OOS summary](reports/oos_summary.md), [frozen result report](reports/oos/p5_result_report.md), and [machine-readable performance](results/p5_oos/oos_performance.csv).

## Public Demo

The public fixture is deterministic and entirely fictitious. It contains six
signal months, 30 synthetic securities per cross-section, and 180 rows. It is a
software contract test, not the source of the formal OOS result.

```powershell
python -m pip install -r requirements.txt
python scripts/generate_demo_data.py
python scripts/run_public_demo.py
python -m pytest -q
```

The expected demo summary includes three factors, 18 monthly IC rows, 180 panel
rows, six signal months, and 18 top-minus-bottom rows. In a clean public clone,
the historical `test_p*.py` modules are skipped because their licensed inputs
and frozen private artifacts are not distributed. See
[data/README.md](data/README.md) for the explicit maintainer opt-in.

## Repository Map

```text
configs/     research rules, cost policy, and frozen protocol references
src/         reusable P1-P6 research and audit implementations
scripts/     phase entry points and the public synthetic demo
tests/       public contract tests plus private-artifact test modules
data/        schemas, manifest, and synthetic fixture only
results/     curated aggregate OOS evidence
reports/     methodology, OOS summary, limitations, and research report
```

## Scope and Limitations

- The `B/M` signal is a `1/PB` proxy. Point-in-time book equity was not
  independently reconstructed and vendor revision policy was not fully verified.
- Three delisting positions lack an auditable terminal event; the original
  result carries forward the last available adjusted close. A post-OOS recovery
  sensitivity is disclosed separately and does not replay the trading path.
- The final OOS covers four calendar years and does not establish capacity,
  market impact, live execution quality, or statistical significance.
- Post-OOS robustness variants are diagnostic only and were not used to select
  or retune the frozen strategy.
- Synthetic data cannot reproduce the formal backtest. Raw and processed market
  data are excluded under [DATA_LICENSE.md](DATA_LICENSE.md).

Detailed methods: [reports/methodology.md](reports/methodology.md). Detailed
limitations: [reports/limitations.md](reports/limitations.md).

## License and Contributions

Original source code is released under [MIT License](LICENSE). Market data,
index data, provider responses, trademarks, and third-party documents are not
licensed by MIT; see [DATA_LICENSE.md](DATA_LICENSE.md).

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md) before
opening an issue or contribution.
