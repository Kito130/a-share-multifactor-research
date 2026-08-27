# A-Share Multifactor Research

这是一个带现实交易约束的 A 股横截面因子研究。项目以价值、12-1 动量和低波动率
构建月频组合，严格区分研究期、验证期与一次性最终 OOS，并完整披露成本、负面因子
结果和数据限制。[中文完整说明](README.zh-CN.md)

This repository studies whether a simple cross-sectional combination of value,
12-1 momentum, and low volatility can deliver risk-adjusted return after
point-in-time universe rules and realistic execution constraints. The public
release contains the research method, bounded aggregate OOS evidence, and a
deterministic synthetic software demo. Licensed security-level data, orders,
holdings, local credentials, and later machine-learning experiments are not
published.

## At a Glance

| Item | Design |
| --- | --- |
| Research question | Do three simple cross-sectional signals survive validation, costs, and execution constraints? |
| Portfolio | Monthly top-100 composite, equal weighted, 2% maximum target weight per name |
| Time split | Research 2016-2019; validation 2020-2021; final OOS 2022-2025 |
| Execution | Month-end signal, next-market-day open, sells before buys |
| Public boundary | Synthetic factor panel plus aggregate evidence from a private licensed-data run |

## Signals and Portfolio

For security `i` at signal month `t`:

```text
B/M proxy(i,t)      = 1 / PB(i,t), when PB(i,t) > 0
12-1 momentum(i,t)  = P_adj(i,t-21) / P_adj(i,t-252) - 1
low volatility(i,t) = - sample_std(last 60 daily returns)
```

Each signal is winsorized at the monthly 1st and 99th percentiles and
standardized cross-sectionally with the sample standard deviation. The
composite is the equal-weight average of the three z-scores. Eligible stocks
must satisfy listing-history, seasoning, liquidity, positive-PB, factor
availability, historical ST, and security-code-history checks.

## Research Design

```text
licensed point-in-time inputs
        |
        v
normalized daily and month-end panels
        |
        v
factor diagnostics on the research period
        |
        v
constrained portfolio simulation and independent validation
        |
        v
frozen configuration -> one authorized final OOS evaluation
```

Validation assessed the pre-specified implementation without parameter
retuning. The 2022-2025 final OOS was executed once under the frozen
configuration. Later robustness checks are explicitly post-OOS and cannot be
used to redefine the strategy or create a new untouched test set.

The simulator records commissions/slippage, historical stamp duty,
suspensions, open price limits, failed orders, cash, stale valuations, and the
documented `600270.SH` to `601598.SH` share-exchange event. It does not claim to
model capacity, queue position, or full market impact.

## Frozen OOS Evidence

The baseline applies 10 bps one-way commission/slippage to CNY 100 million of
initial capital. These figures are aggregate evidence from the excluded
licensed-data run, not live performance.

| Metric | Result |
| --- | ---: |
| Total return | 35.8547% |
| Annualized return | 8.2948% |
| Annualized return difference vs. CSI All Share | +8.5163 pp |
| Zero-risk-free-rate Sharpe | 0.5434 |
| Maximum drawdown | -19.8052% |
| Information ratio | 0.5444 |
| Two-way turnover | 18.4729 |
| Total trading cost | CNY 4,499,738.44, or about 4.50% of initial capital |

The benchmark total return was -0.8490% over the same sample, but the strategy
underperformed it in 2025. The momentum signal also failed out of sample:
47-month mean Rank IC was `-0.02325` and annualized ICIR was `-0.5167`. The
factor remains in the frozen result because removing it after observing OOS
would turn the test into model selection.

Evidence: [OOS summary](reports/oos_summary.md),
[detailed frozen output](reports/oos/p5_result_report.md), and
[machine-readable metrics](results/p5_oos/oos_performance.csv).

## Public Reproduction

The public fixture contains 180 fictitious rows: six signal months and 30
synthetic securities per cross-section. It validates factor-diagnostic code but
cannot reproduce the formal backtest.

```powershell
python -m pip install -r requirements.txt
python scripts/generate_demo_data.py
python scripts/run_public_demo.py
python -m pytest -q
```

Expected demo output: three factors, 18 monthly IC rows, 180 panel rows, six
signal months, and 18 top-minus-bottom rows. Tests that require licensed inputs
are skipped in a clean public clone.

## Repository Guide

```text
configs/  research rules, execution costs, and frozen configuration
src/      panel construction, factor research, backtest, and audit code
scripts/  workflow entry points and the public synthetic demo
tests/    public contracts plus private-artifact acceptance tests
data/     public schema, manifest, and synthetic fixture
results/  curated aggregate final-OOS evidence
reports/  methodology, OOS summary, limitations, and research report
```

## Limits

- `1/PB` is a B/M proxy; point-in-time book equity was not independently
  reconstructed, and the provider's historical revision policy remains
  unverified.
- Three delisted positions lack auditable terminal events. The frozen result
  carries the last available adjusted close; recovery scenarios are post-OOS
  sensitivity checks only.
- Four OOS calendar years do not establish statistical significance, capacity,
  or live execution quality.
- Synthetic data do not support claims about real securities or formal OOS
  reproducibility.

Detailed methods: [reports/methodology.md](reports/methodology.md). Detailed
limitations: [reports/limitations.md](reports/limitations.md). Data rights:
[DATA_LICENSE.md](DATA_LICENSE.md).

Original source code is released under the [MIT License](LICENSE). Market data,
provider responses, trademarks, and third-party materials are not licensed by
MIT. See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).
