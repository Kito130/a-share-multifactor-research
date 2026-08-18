# Final OOS Summary

This document is the public-facing summary of the frozen P5 final OOS. It is
not a new backtest and it does not change the frozen configuration.

## Scope

- Signal and research rules were frozen after the 2020-2021 validation period.
- Final OOS dates are 2022-01-01 to 2025-12-31; the first executed trading date
  is 2022-01-04.
- The baseline scenario is `BASE_10BPS`, with 10 bps one-way
  commission/slippage and the configured historical stamp duty.
- The benchmark is CSI All Share (`000985.CSI`).

## Result

| Metric | Value |
| --- | ---: |
| Trading days | 969 |
| Strategy total return | 35.8547% |
| Strategy annualized return | 8.2948% |
| Strategy annualized volatility | 17.4883% |
| Zero-risk-free-rate Sharpe | 0.5434 |
| Maximum drawdown | -19.8052% |
| Benchmark total return | -0.8490% |
| Annualized return difference | +8.5163 percentage points |
| Information ratio | 0.5444 |
| Two-way turnover | 18.4729 |
| Total trading cost | CNY 4,499,738.44 |
| Failed buy orders | 24 |
| Failed sell orders | 85 |

The benchmark and strategy comparison is descriptive evidence for this sample,
not a significance test, capacity estimate, or live track record.

## Factor Diagnostics

Final OOS monthly Rank IC was available for 47 months because the final signal
month has no next-month label within the defined sample. The average Rank ICs
were `0.0633` for `bm_proxy`, `0.0838` for `lowvol_60`, and `-0.0232` for
`momentum_12_1`. The negative momentum result is retained as a negative finding;
the project does not claim that all three factors were stable in every period.

## Cost Sensitivity

| Scenario | Annualized return | Maximum drawdown | Total cost |
| --- | ---: | ---: | ---: |
| `STRESS_5BPS` | 8.55% | -19.64% | CNY 2,913,768.73 |
| `BASE_10BPS` | 8.29% | -19.81% | CNY 4,499,738.44 |
| `STRESS_20BPS` | 7.78% | -20.13% | CNY 7,457,146.74 |

## Evidence and Boundaries

The authoritative machine-readable row is in
`results/p5_oos/oos_performance.csv`. The detailed frozen report is
`reports/oos/p5_result_report.md`. The P5 result was produced from excluded
licensed inputs; the public synthetic data is not a reproduction of this row.
Post-OOS robustness and delisting recovery sensitivity are interpretive checks
only and must not be presented as fresh OOS evidence.
