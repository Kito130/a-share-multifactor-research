# Methodology

## Research Question

The study asks whether an equal-weight composite of value, 12-1 momentum, and
60-day low volatility can produce robust cross-sectional excess return after
point-in-time universe rules, next-open execution, trading costs, and common
corporate-action constraints.

## Universe and Inputs

The formal run uses licensed daily market data, security master history,
industry/name history, execution-status fields, a benchmark series, and a
manually reviewed corporate-action table. The benchmark is CSI All Share
(`000985.CSI`). The public repository includes schemas and synthetic fixtures,
not these licensed inputs.

P1 normalizes units, preserves raw and canonical security codes, checks listing
intervals, and creates daily and month-end panels. The stock universe requires:

- a Shenzhen or Shanghai listing with a valid historical listing interval;
- at least 120 trading days since listing;
- positive PB and all three factor values;
- sufficient 20-day liquidity;
- no historical ST flag;
- no invalid code-history or listing-reference record.

## Factor Definitions

For a security `i` at signal month `t`:

```text
B/M proxy(i,t)       = 1 / PB(i,t), when PB(i,t) > 0
12-1 momentum(i,t)    = P_adj(i,t-21) / P_adj(i,t-252) - 1
low volatility(i,t)  = - sample_std(daily_returns over the last 60 observations)
```

The lookback values are point-in-time observations available at the signal
date. Each factor is winsorized cross-sectionally at the 1st and 99th
percentiles and then standardized with the sample standard deviation. The
composite is the equal-weight average of the three z-scores.

## Portfolio and Execution

The portfolio selects the 100 highest composite scores and targets equal
weights, capped at 2% per name. Signals use month-end close. Rebalance orders
are scheduled for the next market trading day open, with sells processed before
buys. The backtest records cash, failed orders, suspensions, open price limits,
commission/slippage, historical stamp duty, stale valuation, and the documented
600270.SH to 601598.SH share-exchange action.

The baseline cost scenario uses 10 bps one-way commission/slippage. Stress cases
use 5 bps and 20 bps. The simulation allows fractional adjusted-price units to
avoid introducing a board-lot assumption that the source data cannot support.

## Time Split and Freeze

```text
2016-2019 research -> 2020-2021 validation -> freeze -> 2022-2025 final OOS
```

Validation was used to assess the pre-specified implementation, not to retune
parameters. The final OOS was authorized and executed once. The frozen
configuration hash is recorded in the shipped protocol and P5 manifest. The
public synthetic fixture never enters this formal result.

## Audit and Reproduction Boundary

P1-P4 tests verify data uniqueness, code-history intervals, factor formulas,
forward-label timing, universe rules, portfolio accounting, execution timing,
cost formulas, corporate actions, and freeze hashes. P5 records input/output
hashes and aggregate results. A public clone can run the synthetic software
contract tests, but cannot regenerate the formal result without the excluded
licensed data.
