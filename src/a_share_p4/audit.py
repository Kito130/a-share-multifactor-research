from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml

from .build import PB_DISCLOSURE
from .config import absolute, load_config, sql_path


@dataclass
class AuditResult:
    check_id: str
    category: str
    status: str
    observed: str
    expected: str
    details: str


class AuditCollector:
    def __init__(self) -> None:
        self.results: list[AuditResult] = []

    def add(
        self,
        check_id: str,
        category: str,
        passed: bool,
        observed: Any,
        expected: Any,
        details: str,
        *,
        warning: bool = False,
    ) -> None:
        self.results.append(
            AuditResult(
                check_id=check_id,
                category=category,
                status=(
                    "WARN"
                    if warning
                    else ("PASS" if passed else "FAIL")
                ),
                observed=str(observed),
                expected=str(expected),
                details=details,
            )
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(item) for item in self.results)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv_atomic(frame: pd.DataFrame, relative_path: str) -> None:
    output = absolute(relative_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, output)


def _write_text_atomic(text: str, relative_path: str) -> None:
    output = absolute(relative_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, output)


def _write_json_atomic(
    payload: dict[str, Any], relative_path: str
) -> None:
    output = absolute(relative_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, output)


def _read_yaml(relative_path: str) -> dict[str, Any]:
    with absolute(relative_path).open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle)
    return payload


def _all_paths_relative(config: dict[str, Any]) -> bool:
    values = (
        list(config["inputs"].values())
        + list(config["outputs"].values())
        + list(config["protected_p4_inputs"])
    )
    return all(not Path(str(value)).is_absolute() for value in values)


def _report(
    audit: pd.DataFrame,
    overall: str,
    period_performance: pd.DataFrame,
    ic_summary: pd.DataFrame,
    reproduction: pd.DataFrame,
    freeze_hash: str,
    audited_at: str,
) -> str:
    baseline = period_performance.loc[
        period_performance["is_baseline"].astype(bool)
        & (
            period_performance["period"]
            == "VALIDATION_2020_2021"
        )
    ].iloc[0]
    pass_count = int((audit["status"] == "PASS").sum())
    warn_count = int((audit["status"] == "WARN").sum())
    fail_count = int((audit["status"] == "FAIL").sum())
    lines = [
        "# P4 验证与冻结审计报告",
        "",
        f"- 审计时间（UTC）：{audited_at}",
        f"- 总体状态：**{overall}**",
        (
            f"- 检查结果：{pass_count} PASS / {warn_count} WARN / "
            f"{fail_count} FAIL"
        ),
        "- 研究期：2016-01-01 至 2019-12-31。",
        "- 验证期：2020-01-01 至 2021-12-31。",
        "- 最终 OOS：2022-01-01 至 2025-12-31，未运行、未预览。",
        "",
        "## 验证期基准成本情景",
        "",
        f"- 策略累计收益：{baseline['strategy_total_return']:.2%}",
        f"- 策略年化收益：{baseline['strategy_annualized_return']:.2%}",
        (
            "- 策略期内最大回撤："
            f"{baseline['strategy_max_drawdown_within_period']:.2%}"
        ),
        f"- 中证全指累计收益：{baseline['benchmark_total_return']:.2%}",
        f"- 中证全指年化收益：{baseline['benchmark_annualized_return']:.2%}",
        (
            "- 年化收益差："
            f"{baseline['annualized_return_difference']:.2%}"
        ),
        f"- 总交易成本：{baseline['total_trading_cost']:,.2f} 元",
        f"- 买入失败：{int(baseline['failed_buy_orders'])} 单",
        f"- 卖出失败：{int(baseline['failed_sell_orders'])} 单",
        "",
        "验证期表现只用于冻结前检验，不构成最终 OOS 结果，也不触发调参。",
        "",
        "## 验证期单因子 Rank IC",
        "",
        "| 因子 | 可评价月份 | Rank IC均值 | 年化ICIR |",
        "|---|---:|---:|---:|",
    ]
    for row in ic_summary.itertuples(index=False):
        lines.append(
            f"| {row.factor} | {int(row.months)} | "
            f"{row.mean_rank_ic:.6f} | "
            f"{row.rank_icir_annualized:.6f} |"
        )
    lines.extend(
        [
            "",
            (
                "2021-12 信号的下一月收益落入最终 OOS，故验证期 IC "
                "只评价 23 个月；该标签未读取。"
            ),
            "",
            "## 研究期精确复现",
            "",
            "| 检查 | 旧行数 | 新行数 | 最大数值误差 | 状态 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in reproduction.itertuples(index=False):
        lines.append(
            f"| {row.check} | {int(row.old_rows)} | "
            f"{int(row.new_rows)} | "
            f"{row.maximum_numeric_error:.3e} | {row.status} |"
        )
    lines.extend(
        [
            "",
            "## 冻结与人工闸门",
            "",
            "- 冻结配置：`configs/frozen_config.yaml`。",
            f"- 冻结配置 SHA-256：`{freeze_hash}`。",
            "- 冻结配置记录全部受保护输入的 SHA-256。",
            "- 冻结后禁止根据验证期或最终 OOS 结果调参。",
            "- P5 必须等待用户新的明确授权；本次未生成或运行 P5。",
            "",
            "## 已披露限制",
            "",
            f"- {PB_DISCLOSURE}",
            "",
            "## 检查明细",
            "",
            "| ID | 类别 | 状态 | 观测 | 期望 |",
            "|---|---|---|---|---|",
        ]
    )
    for row in audit.itertuples(index=False):
        lines.append(
            f"| {row.check_id} | {row.category} | {row.status} | "
            f"{row.observed} | {row.expected} |"
        )
    lines.append("")
    return "\n".join(lines)


def audit_p4() -> pd.DataFrame:
    audited_at = datetime.now(UTC).isoformat()
    config = load_config()
    outputs = config["outputs"]
    collector = AuditCollector()
    required_keys = (
        "factor_panel",
        "composite_signals",
        "target_holdings",
        "daily_portfolio",
        "actual_holdings",
        "orders",
        "failed_orders",
        "cash_ledger",
        "corporate_action_events",
        "validation_factor_coverage",
        "validation_monthly_rank_ic",
        "validation_ic_summary",
        "validation_quintile_returns",
        "validation_annual_results",
        "validation_factor_correlations_monthly",
        "validation_factor_correlations_summary",
        "validation_industry_exposure",
        "validation_size_exposure",
        "validation_worst_periods",
        "rebalance_schedule",
        "rebalance_summary",
        "research_reproduction",
        "p4_input_hashes",
        "performance_summary",
        "annual_performance",
        "period_performance",
        "validation_comparison",
        "failure_reason_summary",
        "stale_position_summary",
        "frozen_config",
        "frozen_protocol",
        "config_sha256",
        "p4_run_manifest",
    )
    missing = [
        outputs[key]
        for key in required_keys
        if not absolute(outputs[key]).is_file()
    ]
    collector.add(
        "P4-001",
        "文件",
        not missing,
        f"missing={missing}",
        "missing=[]",
        "P4规定输出必须全部存在。",
    )
    if missing:
        frame = collector.frame()
        _write_csv_atomic(frame, outputs["p4_audit_summary"])
        raise FileNotFoundError(f"缺少P4输出：{missing}")

    collector.add(
        "P4-002",
        "路径",
        _all_paths_relative(config),
        _all_paths_relative(config),
        True,
        "输入、输出和受保护文件均使用项目相对路径。",
    )

    manifest = json.loads(
        absolute(outputs["p4_run_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    p2_manifest = json.loads(
        absolute(config["inputs"]["p2_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    p3_manifest = json.loads(
        absolute(config["inputs"]["p3_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    gates_pass = (
        str(p2_manifest.get("status", "")).startswith("P2_ACCEPTED")
        and str(p3_manifest.get("status", "")).startswith("P3_ACCEPTED")
        and p3_manifest.get("audit", {}).get("fail_count") == 0
        and not p3_manifest.get("scope_guards", {}).get(
            "oos_period_read_or_run", False
        )
    )
    collector.add(
        "P4-003",
        "前置闸门",
        gates_pass,
        (
            f"P2={p2_manifest.get('status')}, "
            f"P3={p3_manifest.get('status')}, "
            f"P3_FAIL={p3_manifest.get('audit', {}).get('fail_count')}"
        ),
        "P2/P3 accepted; P3 FAIL=0; OOS untouched",
        "P4只能从已验收的P2/P3冻结实现进入。",
    )

    p2_config = _read_yaml(config["inputs"]["p2_config"])
    p3_config = _read_yaml(config["inputs"]["p3_config"])
    factor_match = all(
        config[section] == p2_config[section]
        for section in ("factors", "universe", "statistics")
    )
    portfolio_match = all(
        config[section] == p3_config[section]
        for section in (
            "portfolio",
            "composite",
            "cost_scenarios",
            "valuation",
            "corporate_actions",
            "metrics",
        )
    )
    collector.add(
        "P4-004",
        "冻结参数",
        factor_match and portfolio_match,
        f"P2_factor={factor_match}, P3_portfolio={portfolio_match}",
        "both=True",
        "验证期必须原样复用P2/P3参数，不得调参。",
    )

    input_hashes = pd.read_csv(absolute(outputs["p4_input_hashes"]))
    current_mismatches = 0
    for row in input_hashes.itertuples(index=False):
        path = absolute(row.path)
        if (
            not path.is_file()
            or path.stat().st_size != row.size_bytes_after
            or _sha256(path) != row.sha256_after
        ):
            current_mismatches += 1
    collector.add(
        "P4-005",
        "输入完整性",
        bool(input_hashes["match"].all()) and current_mismatches == 0,
        (
            f"recorded={bool(input_hashes['match'].all())}, "
            f"current_mismatches={current_mismatches}, "
            f"files={len(input_hashes)}"
        ),
        "recorded=True, current_mismatches=0",
        "P4运行前后及审计时的全部受保护输入必须一致。",
    )

    reproduction = pd.read_csv(
        absolute(outputs["research_reproduction"])
    )
    reproduction_pass = (
        len(reproduction) == 4
        and bool((reproduction["status"] == "PASS").all())
        and int(reproduction["missing_or_extra_rows"].sum()) == 0
        and int(reproduction["categorical_mismatches"].sum()) == 0
    )
    collector.add(
        "P4-006",
        "研究期复现",
        reproduction_pass,
        (
            f"checks={len(reproduction)}, "
            f"pass={(reproduction['status'] == 'PASS').sum()}, "
            f"max_error={reproduction['maximum_numeric_error'].max():.3e}"
        ),
        "4/4 PASS; no row or categorical mismatch",
        "扩展到验证期后，2016—2019输出必须精确复现P2/P3。",
    )

    view_keys = (
        "factor_panel",
        "composite_signals",
        "target_holdings",
        "daily_portfolio",
        "actual_holdings",
        "orders",
        "failed_orders",
        "corporate_action_events",
    )
    with duckdb.connect() as connection:
        for key in view_keys:
            connection.execute(
                f"""
                CREATE VIEW {key} AS
                SELECT * FROM read_parquet('{sql_path(outputs[key])}')
                """
            )

        factor_stats = connection.execute(
            """
            SELECT
                count(*),
                count(DISTINCT signal_date),
                min(signal_date),
                max(signal_date),
                count(*) FILTER (
                    WHERE signal_date BETWEEN DATE '2020-01-01'
                                          AND DATE '2021-12-31'
                ),
                count(DISTINCT signal_date) FILTER (
                    WHERE signal_date BETWEEN DATE '2020-01-01'
                                          AND DATE '2021-12-31'
                ),
                count(*) FILTER (
                    WHERE signal_date >= DATE '2022-01-01'
                )
            FROM factor_panel
            """
        ).fetchone()
        collector.add(
            "P4-007",
            "因子面板",
            factor_stats[0] > 0
            and factor_stats[1] == 72
            and pd.Timestamp(
                factor_stats[2]
            ).date().isoformat()
            == "2016-01-29"
            and pd.Timestamp(
                factor_stats[3]
            ).date().isoformat()
            == "2021-12-31"
            and factor_stats[4] > 0
            and factor_stats[5] == 24
            and factor_stats[6] == 0,
            (
                f"rows={factor_stats[0]}, months={factor_stats[1]}, "
                f"range={factor_stats[2]}..{factor_stats[3]}, "
                f"validation_rows={factor_stats[4]}, "
                f"validation_months={factor_stats[5]}, "
                f"oos_rows={factor_stats[6]}"
            ),
            "72 months through 2021-12-31; validation=24 months; OOS=0",
            "P4因子面板仅覆盖研究期和验证期。",
        )

        signal_stats = connection.execute(
            """
            SELECT
                count(*),
                count(DISTINCT signal_date),
                min(signal_date),
                max(signal_date),
                count(*) FILTER (
                    WHERE signal_date >= DATE '2022-01-01'
                ),
                max(abs(
                    composite_score
                    - (bm_proxy_z + momentum_12_1_z + lowvol_60_z)
                      / 3.0
                ))
            FROM composite_signals
            """
        ).fetchone()
        collector.add(
            "P4-008",
            "复合因子",
            signal_stats[0] > 0
            and signal_stats[1] == 72
            and pd.Timestamp(
                signal_stats[2]
            ).date().isoformat()
            == "2016-01-29"
            and pd.Timestamp(
                signal_stats[3]
            ).date().isoformat()
            == "2021-12-31"
            and signal_stats[4] == 0
            and signal_stats[5] <= 1e-12,
            (
                f"rows={signal_stats[0]}, months={signal_stats[1]}, "
                f"range={signal_stats[2]}..{signal_stats[3]}, "
                f"oos={signal_stats[4]}, "
                f"formula_error={signal_stats[5]:.3e}"
            ),
            "72 months; OOS=0; formula error<=1e-12",
            "复合分数必须为三个冻结z-score的等权平均。",
        )
        signal_columns = {
            row[0]
            for row in connection.execute(
                "DESCRIBE composite_signals"
            ).fetchall()
        }
        collector.add(
            "P4-009",
            "无前视",
            "next_month_return" not in signal_columns,
            sorted(signal_columns & {"next_month_return"}),
            [],
            "组合信号不得携带下一月收益标签。",
        )

        target_stats = connection.execute(
            """
            WITH monthly AS (
                SELECT
                    signal_date,
                    count(*) AS names,
                    sum(target_weight) AS gross_weight,
                    max(target_weight) AS maximum_weight,
                    min(selection_rank) AS minimum_rank,
                    max(selection_rank) AS maximum_rank
                FROM target_holdings
                GROUP BY signal_date
            )
            SELECT
                count(*),
                min(names), max(names),
                min(gross_weight), max(gross_weight),
                max(maximum_weight),
                min(minimum_rank), max(maximum_rank)
            FROM monthly
            """
        ).fetchone()
        collector.add(
            "P4-010",
            "目标持仓",
            target_stats[0] == 72
            and target_stats[1] == 100
            and target_stats[2] == 100
            and abs(target_stats[3] - 1.0) <= 1e-12
            and abs(target_stats[4] - 1.0) <= 1e-12
            and target_stats[5] <= 0.02
            and target_stats[6] == 1
            and target_stats[7] == 100,
            target_stats,
            "72 months; 100 names; gross=1; max_weight<=0.02; ranks=1..100",
            "验证期继续使用冻结Top-100等权组合。",
        )

        daily_stats = connection.execute(
            """
            SELECT
                count(*),
                count(DISTINCT cost_scenario),
                min(trade_date),
                max(trade_date),
                count(*) FILTER (
                    WHERE trade_date >= DATE '2022-01-01'
                ),
                min(cash_cny),
                min(total_nav_cny),
                max(abs(
                    total_nav_cny - cash_cny - market_value_cny
                )),
                max(abs(
                    cash_weight + market_value_weight - 1.0
                ))
            FROM daily_portfolio
            """
        ).fetchone()
        collector.add(
            "P4-011",
            "样本闸门",
            daily_stats[0] > 0
            and daily_stats[1] == 3
            and pd.Timestamp(
                daily_stats[2]
            ).date().isoformat()
            == "2016-01-29"
            and pd.Timestamp(
                daily_stats[3]
            ).date().isoformat()
            == "2021-12-31"
            and daily_stats[4] == 0,
            (
                f"rows={daily_stats[0]}, scenarios={daily_stats[1]}, "
                f"range={daily_stats[2]}..{daily_stats[3]}, "
                f"oos={daily_stats[4]}"
            ),
            "3 scenarios through 2021-12-31; OOS=0",
            "P4组合只能覆盖研究期和验证期。",
        )
        collector.add(
            "P4-012",
            "会计恒等式",
            daily_stats[5] >= -1e-8
            and daily_stats[6] > 0
            and daily_stats[7] <= 1e-6
            and daily_stats[8] <= 1e-12,
            (
                f"min_cash={daily_stats[5]}, min_nav={daily_stats[6]}, "
                f"nav_error={daily_stats[7]}, "
                f"weight_error={daily_stats[8]}"
            ),
            "cash>=0, nav>0, errors within tolerance",
            "组合不得借现金或使用杠杆，净值等于现金加持仓市值。",
        )

        return_errors = connection.execute(
            """
            WITH lagged AS (
                SELECT
                    *,
                    lag(strategy_nav) OVER (
                        PARTITION BY cost_scenario ORDER BY trade_date
                    ) AS prior_strategy_nav,
                    first_value(benchmark_close) OVER (
                        PARTITION BY cost_scenario ORDER BY trade_date
                    ) AS first_benchmark_close,
                    lag(benchmark_nav) OVER (
                        PARTITION BY cost_scenario ORDER BY trade_date
                    ) AS prior_benchmark_nav
                FROM daily_portfolio
            )
            SELECT
                max(abs(strategy_nav - total_nav_cny / 100000000.0)),
                max(abs(
                    strategy_daily_return
                    - (strategy_nav / prior_strategy_nav - 1.0)
                )) FILTER (WHERE prior_strategy_nav IS NOT NULL),
                max(abs(
                    benchmark_nav
                    - benchmark_close / first_benchmark_close
                )),
                max(abs(
                    benchmark_daily_return
                    - (benchmark_nav / prior_benchmark_nav - 1.0)
                )) FILTER (WHERE prior_benchmark_nav IS NOT NULL),
                count(DISTINCT benchmark_code)
            FROM lagged
            """
        ).fetchone()
        collector.add(
            "P4-013",
            "净值收益",
            all(value <= 1e-12 for value in return_errors[:4])
            and return_errors[4] == 1,
            return_errors,
            "all errors<=1e-12; one benchmark",
            "策略及中证全指净值和日收益必须可全量复算。",
        )

        order_stats = connection.execute(
            """
            SELECT
                count(*),
                count(DISTINCT cost_scenario),
                count(*) FILTER (
                    WHERE trade_date >= DATE '2022-01-01'
                ),
                count(*) FILTER (
                    WHERE trade_date <= signal_date
                ),
                count(*) FILTER (
                    WHERE executed_notional > 0
                      AND (
                        execution_price_source <> 'ADJUSTED_OPEN'
                        OR adjusted_open IS NULL
                        OR adjusted_open <= 0
                      )
                ),
                count(*) FILTER (
                    WHERE order_status = 'FAILED'
                      AND executed_notional <> 0
                )
            FROM orders
            """
        ).fetchone()
        collector.add(
            "P4-014",
            "订单时序",
            order_stats[0] > 0
            and order_stats[1] == 3
            and all(value == 0 for value in order_stats[2:]),
            order_stats,
            "orders>0; scenarios=3; all violations=0",
            "订单必须在信号后、2021年末前，以可得复权开盘价执行。",
        )

        execution_violations = connection.execute(
            f"""
            SELECT count(*)
            FROM orders
            LEFT JOIN read_parquet(
                '{sql_path(config["inputs"]["execution_status"])}'
            ) AS execution
              ON orders.execution_ts_code = execution.ts_code
             AND orders.trade_date = execution.trade_date
             AND execution.security_code_interval_valid
            WHERE orders.executed_notional > 0
              AND (
                execution.ts_code IS NULL
                OR (
                    orders.side = 'BUY'
                    AND execution.cannot_buy_at_open
                )
                OR (
                    orders.side = 'SELL'
                    AND execution.cannot_sell_at_open
                )
              )
            """
        ).fetchone()[0]
        collector.add(
            "P4-015",
            "成交限制",
            execution_violations == 0,
            execution_violations,
            0,
            "任何成交都不得违反P1开盘可买卖状态。",
        )

        cost_errors = connection.execute(
            """
            SELECT
                max(abs(
                    commission_slippage_cost
                    - executed_notional * commission_slippage_rate
                )),
                max(abs(
                    stamp_duty_cost
                    - executed_notional * stamp_duty_rate
                )),
                count(*) FILTER (
                    WHERE side = 'BUY' AND stamp_duty_cost <> 0
                ),
                count(*) FILTER (
                    WHERE side = 'SELL'
                      AND executed_notional > 0
                      AND abs(stamp_duty_rate - 0.001) > 1e-15
                ),
                max(abs(
                    total_trading_cost
                    - commission_slippage_cost - stamp_duty_cost
                ))
            FROM orders
            """
        ).fetchone()
        collector.add(
            "P4-016",
            "交易成本",
            cost_errors[0] <= 1e-12
            and cost_errors[1] <= 1e-12
            and cost_errors[2] == 0
            and cost_errors[3] == 0
            and cost_errors[4] <= 1e-12,
            cost_errors,
            "formula errors<=1e-12; buy tax=0; sell tax=0.001",
            "2016—2021按冻结历史印花税和三档单边成本执行。",
        )

        failed_stats = connection.execute(
            """
            SELECT
                count(*),
                count(*) FILTER (WHERE executed_notional <> 0),
                count(*) FILTER (
                    WHERE side = 'SELL'
                      AND remaining_adjusted_units_after <= 0
                )
            FROM failed_orders
            """
        ).fetchone()
        failed_preservation = connection.execute(
            """
            SELECT count(*)
            FROM failed_orders
            LEFT JOIN actual_holdings
              ON failed_orders.cost_scenario
                    = actual_holdings.cost_scenario
             AND failed_orders.trade_date = actual_holdings.trade_date
             AND failed_orders.canonical_ts_code
                    = actual_holdings.canonical_ts_code
            WHERE failed_orders.side = 'SELL'
              AND (
                actual_holdings.canonical_ts_code IS NULL
                OR abs(
                    actual_holdings.adjusted_units
                    - failed_orders.remaining_adjusted_units_after
                ) > 1e-8
              )
            """
        ).fetchone()[0]
        collector.add(
            "P4-017",
            "失败订单",
            failed_stats[0] > 0
            and failed_stats[1] == 0
            and failed_stats[2] == 0
            and failed_preservation == 0,
            (
                f"failed={failed_stats[0]}, filled={failed_stats[1]}, "
                f"lost_sell_positions={failed_stats[2]}, "
                f"preservation_errors={failed_preservation}"
            ),
            "failed>0; filled=0; preservation errors=0",
            "失败订单不得成交，失败卖出后旧仓必须保留。",
        )

        stale_execution = connection.execute(
            """
            SELECT count(*)
            FROM orders
            WHERE executed_notional > 0
              AND execution_price_source
                    = 'LAST_AVAILABLE_ADJUSTED_CLOSE'
            """
        ).fetchone()[0]
        terminal_stale = connection.execute(
            """
            SELECT
                count(*),
                coalesce(sum(position_market_value), 0.0)
            FROM actual_holdings
            WHERE trade_date = (
                SELECT max(trade_date) FROM actual_holdings
            )
              AND valuation_price_source
                    = 'LAST_AVAILABLE_ADJUSTED_CLOSE'
            """
        ).fetchone()
        collector.add(
            "P4-018",
            "停牌估值",
            stale_execution == 0 and terminal_stale[0] == 0,
            (
                f"stale_executions={stale_execution}, "
                f"terminal_stale_positions={terminal_stale[0]}, "
                f"terminal_value={terminal_stale[1]}"
            ),
            "stale executions=0; terminal stale positions=0",
            "上一可得收盘价只用于估值，且验证期末不得残留陈旧估值。",
        )

        action_stats = connection.execute(
            """
            SELECT
                count(*),
                count(DISTINCT cost_scenario),
                count(*) FILTER (
                    WHERE activity_type <> 'CORPORATE_ACTION'
                       OR old_ts_code <> '600270.SH'
                       OR successor_ts_code <> '601598.SH'
                       OR action_type <> 'STOCK_SWAP_ABSORPTION'
                       OR effective_date <> DATE '2018-12-28'
                       OR abs(exchange_ratio - 3.8225) > 1e-12
                ),
                max(abs(
                    successor_share_quantity_after
                    - old_share_quantity_before * exchange_ratio
                )),
                max(abs(portfolio_value_difference_cny)),
                max(abs(total_action_cost_cny))
            FROM corporate_action_events
            """
        ).fetchone()
        terminal_old = connection.execute(
            """
            SELECT count(*)
            FROM actual_holdings
            WHERE trade_date = (
                SELECT max(trade_date) FROM actual_holdings
            )
              AND (
                    ts_code = '600270.SH'
                 OR canonical_ts_code = '600270.SH'
              )
            """
        ).fetchone()[0]
        collector.add(
            "P4-019",
            "公司行动",
            action_stats[0] == 3
            and action_stats[1] == 3
            and action_stats[2] == 0
            and action_stats[3] <= 1e-8
            and action_stats[4] <= 1e-6
            and action_stats[5] <= 1e-12
            and terminal_old == 0,
            (
                f"events={action_stats[0]}, scenarios={action_stats[1]}, "
                f"metadata_errors={action_stats[2]}, "
                f"quantity_error={action_stats[3]}, "
                f"value_error={action_stats[4]}, "
                f"cost_error={action_stats[5]}, terminal_old={terminal_old}"
            ),
            "3 events; quantity/value continuous; zero cost; terminal old=0",
            "600270换股处理必须在延长样本中保持有效。",
        )

        all_oos_rows = connection.execute(
            """
            SELECT
                (SELECT count(*) FROM factor_panel
                  WHERE signal_date >= DATE '2022-01-01')
              + (SELECT count(*) FROM composite_signals
                  WHERE signal_date >= DATE '2022-01-01')
              + (SELECT count(*) FROM target_holdings
                  WHERE signal_date >= DATE '2022-01-01')
              + (SELECT count(*) FROM daily_portfolio
                  WHERE trade_date >= DATE '2022-01-01')
              + (SELECT count(*) FROM actual_holdings
                  WHERE trade_date >= DATE '2022-01-01')
              + (SELECT count(*) FROM orders
                  WHERE trade_date >= DATE '2022-01-01')
              + (SELECT count(*) FROM corporate_action_events
                  WHERE effective_date >= DATE '2022-01-01')
            """
        ).fetchone()[0]
        collector.add(
            "P4-020",
            "OOS隔离",
            all_oos_rows == 0,
            all_oos_rows,
            0,
            "P4所有因子、信号、组合、订单和公司行动输出不得含OOS行。",
        )

    schedule = pd.read_csv(
        absolute(outputs["rebalance_schedule"]),
        parse_dates=["signal_date", "scheduled_trade_date"],
    )
    benchmark = pd.read_parquet(
        absolute(config["inputs"]["benchmark_daily"])
    )
    benchmark["trade_date"] = pd.to_datetime(benchmark["trade_date"])
    schedule_errors = 0
    for row in schedule.dropna(
        subset=["scheduled_trade_date"]
    ).itertuples(index=False):
        expected = benchmark.loc[
            (benchmark["trade_date"] > row.signal_date)
            & (benchmark["trade_date"] <= pd.Timestamp("2021-12-31")),
            "trade_date",
        ].min()
        if row.scheduled_trade_date != expected:
            schedule_errors += 1
    schedule_pass = (
        len(schedule) == 72
        and schedule["scheduled_trade_date"].notna().sum() == 71
        and schedule_errors == 0
        and schedule.iloc[-1]["schedule_status"]
        == "OUT_OF_SCOPE_NOT_EXECUTED"
    )
    collector.add(
        "P4-021",
        "调仓日历",
        schedule_pass,
        (
            f"signals={len(schedule)}, "
            f"scheduled={schedule['scheduled_trade_date'].notna().sum()}, "
            f"errors={schedule_errors}, "
            f"last={schedule.iloc[-1]['schedule_status']}"
        ),
        "72 signals; 71 next-market-day executions; final out of scope",
        "2021-12月末信号不得跨入2022年执行。",
    )

    ic_summary = pd.read_csv(
        absolute(outputs["validation_ic_summary"])
    )
    monthly_ic = pd.read_csv(
        absolute(outputs["validation_monthly_rank_ic"]),
        parse_dates=["signal_date"],
    )
    expected_factors = {"bm_proxy", "momentum_12_1", "lowvol_60"}
    december_ic = monthly_ic.loc[
        monthly_ic["signal_date"] == pd.Timestamp("2021-12-31")
    ]
    evaluable_ic = monthly_ic.loc[monthly_ic["rank_ic"].notna()]
    ic_pass = (
        set(ic_summary["factor"]) == expected_factors
        and set(ic_summary["months"]) == {23}
        and set(monthly_ic["factor"]) == expected_factors
        and monthly_ic["signal_date"].min() >= pd.Timestamp("2020-01-01")
        and len(december_ic) == 3
        and bool(december_ic["rank_ic"].isna().all())
        and bool((december_ic["observations"] == 0).all())
        and evaluable_ic["signal_date"].max()
        <= pd.Timestamp("2021-11-30")
    )
    collector.add(
        "P4-022",
        "验证因子统计",
        ic_pass,
        (
            f"factors={sorted(ic_summary['factor'])}, "
            f"months={sorted(ic_summary['months'].unique())}, "
            f"evaluable_range="
            f"{evaluable_ic['signal_date'].min().date()}.."
            f"{evaluable_ic['signal_date'].max().date()}, "
            f"december_empty_rows={len(december_ic)}"
        ),
        "3 factors; 23 evaluable months; 3 empty 2021-12 audit rows",
        "不得为计算2021-12信号的下一月收益而读取2022 OOS。",
    )

    period_performance = pd.read_csv(
        absolute(outputs["period_performance"])
    )
    validation_rows = period_performance.loc[
        period_performance["period"] == "VALIDATION_2020_2021"
    ]
    period_pass = (
        len(period_performance) == 6
        and len(validation_rows) == 3
        and set(validation_rows["cost_scenario"])
        == {"STRESS_5BPS", "BASE_10BPS", "STRESS_20BPS"}
        and set(validation_rows["trading_days"]) == {486}
        and set(validation_rows["start_date"]) == {"2020-01-02"}
        and set(validation_rows["end_date"]) == {"2021-12-31"}
    )
    collector.add(
        "P4-023",
        "验证绩效",
        period_pass,
        (
            f"rows={len(period_performance)}, "
            f"validation_rows={len(validation_rows)}, "
            f"trading_days={sorted(validation_rows['trading_days'].unique())}"
        ),
        "6 rows; 3 validation scenarios; 486 trading days",
        "研究期与验证期绩效必须分段留档。",
    )

    cost_order = (
        validation_rows.set_index("cost_scenario")[
            ["strategy_total_return", "total_trading_cost"]
        ]
    )
    monotonic_cost = (
        cost_order.loc["STRESS_5BPS", "strategy_total_return"]
        > cost_order.loc["BASE_10BPS", "strategy_total_return"]
        > cost_order.loc["STRESS_20BPS", "strategy_total_return"]
        and cost_order.loc["STRESS_5BPS", "total_trading_cost"]
        < cost_order.loc["BASE_10BPS", "total_trading_cost"]
        < cost_order.loc["STRESS_20BPS", "total_trading_cost"]
    )
    collector.add(
        "P4-024",
        "成本压力",
        monotonic_cost,
        cost_order.to_dict(orient="index"),
        "return decreases and cost increases with bps",
        "更高成本情景不得机械地产生更高净收益或更低总成本。",
    )

    frozen = _read_yaml(outputs["frozen_config"])
    hash_text = absolute(outputs["config_sha256"]).read_text(
        encoding="utf-8"
    )
    match = re.search(r"sha256=([0-9a-f]{64})", hash_text)
    recorded_hash = match.group(1) if match else ""
    current_freeze_hash = _sha256(absolute(outputs["frozen_config"]))
    manifest_freeze_hash = manifest.get("freeze", {}).get(
        "frozen_config_sha256"
    )
    collector.add(
        "P4-025",
        "冻结哈希",
        bool(recorded_hash)
        and recorded_hash == current_freeze_hash
        and manifest_freeze_hash == current_freeze_hash,
        (
            f"file={current_freeze_hash}, recorded={recorded_hash}, "
            f"manifest={manifest_freeze_hash}"
        ),
        "all three hashes identical",
        "冻结配置必须由独立SHA-256锚定。",
    )

    frozen_sources = frozen.get("source_sha256", {})
    source_hash_pass = (
        set(frozen_sources) == set(config["protected_p4_inputs"])
        and all(
            frozen_sources[path] == _sha256(absolute(path))
            for path in config["protected_p4_inputs"]
        )
    )
    collector.add(
        "P4-026",
        "冻结输入",
        source_hash_pass,
        (
            f"frozen={len(frozen_sources)}, "
            f"protected={len(config['protected_p4_inputs'])}"
        ),
        "all protected input paths and current hashes recorded",
        "冻结配置必须记录全部受保护输入，而非只记录参数文件。",
    )

    rules = frozen.get("post_freeze_rules", {})
    rules_pass = (
        frozen.get("status") == "FROZEN_AFTER_VALIDATION"
        and rules.get("allow_parameter_changes") is False
        and rules.get("allow_validation_retuning") is False
        and rules.get("final_oos_requires_new_explicit_authorization")
        is True
        and rules.get("final_oos_has_been_run") is False
        and rules.get("p5_implementation_generated") is False
    )
    collector.add(
        "P4-027",
        "冻结规则",
        rules_pass,
        rules,
        "no retuning; OOS authorization required; OOS/P5 not run",
        "冻结后的人工闸门必须写入机器可读配置。",
    )

    protocol = absolute(outputs["frozen_protocol"]).read_text(
        encoding="utf-8"
    )
    protocol_pass = all(
        phrase in protocol
        for phrase in (
            "2020-01-01 至 2021-12-31",
            "2022-01-01 至 2025-12-31，尚未运行",
            "不允许根据验证期或最终OOS结果修改参数",
            current_freeze_hash,
            PB_DISCLOSURE,
        )
    )
    collector.add(
        "P4-028",
        "冻结协议",
        protocol_pass,
        f"length={len(protocol)}, hash_present={current_freeze_hash in protocol}",
        "dates, no-retuning rule, hash and PB disclosure present",
        "人类可读冻结协议必须和机器配置一致。",
    )

    scope = manifest.get("scope_guards", {})
    scope_pass = (
        scope.get("validation_period_evaluated") is True
        and scope.get("validation_parameters_retuned") is False
        and scope.get("oos_rows_written") is False
        and scope.get("oos_results_computed") is False
        and scope.get("oos_results_previewed") is False
        and scope.get("p5_code_generated") is False
        and scope.get("p5_run") is False
        and scope.get("p6_run") is False
    )
    collector.add(
        "P4-029",
        "范围声明",
        scope_pass,
        scope,
        "validation only; every OOS/P5/P6 flag false",
        "P4清单必须明确声明未触碰最终OOS。",
    )

    p5_candidates = (
        Path("src/a_share_p5"),
        Path("scripts/build_p5_oos.py"),
        Path("results/p5_oos"),
        Path("reports/oos"),
    )
    existing_p5 = [
        path.as_posix()
        for path in p5_candidates
        if absolute(path.as_posix()).exists()
    ]
    collector.add(
        "P4-030",
        "P5人工闸门",
        not existing_p5,
        existing_p5,
        [],
        "P4不得提前生成P5实现或OOS结果目录。",
    )

    output_mismatches: list[str] = []
    for key, expected_hash in manifest.get("output_sha256", {}).items():
        path = absolute(outputs[key])
        if not path.is_file() or _sha256(path) != expected_hash:
            output_mismatches.append(key)
    collector.add(
        "P4-031",
        "输出完整性",
        not output_mismatches,
        output_mismatches,
        [],
        "构建清单内所有输出哈希必须在审计时保持一致。",
    )

    collector.add(
        "P4-032",
        "PB历史口径",
        True,
        "NEEDS_MANUAL_CONFIRMATION",
        "已披露，不阻碍P4",
        PB_DISCLOSURE,
        warning=True,
    )

    audit = collector.frame()
    fail_count = int((audit["status"] == "FAIL").sum())
    warn_count = int((audit["status"] == "WARN").sum())
    pass_count = int((audit["status"] == "PASS").sum())
    overall = (
        "P4_ACCEPTED_AND_FROZEN_WITH_DISCLOSED_LIMITATION"
        if fail_count == 0
        else "P4_AUDIT_FAILED"
    )
    report = _report(
        audit,
        overall,
        period_performance,
        ic_summary,
        reproduction,
        current_freeze_hash,
        audited_at,
    )
    _write_csv_atomic(audit, outputs["p4_audit_summary"])
    _write_text_atomic(report, outputs["p4_audit_report"])

    manifest["status"] = overall
    manifest["audit"] = {
        "audited_at_utc": audited_at,
        "pass_count": pass_count,
        "warn_count": warn_count,
        "fail_count": fail_count,
        "summary_path": outputs["p4_audit_summary"],
        "report_path": outputs["p4_audit_report"],
    }
    manifest["output_sha256"]["p4_audit_summary"] = _sha256(
        absolute(outputs["p4_audit_summary"])
    )
    manifest["output_sha256"]["p4_audit_report"] = _sha256(
        absolute(outputs["p4_audit_report"])
    )
    _write_json_atomic(manifest, outputs["p4_run_manifest"])
    print(
        f"[P4 AUDIT] {overall}: "
        f"{pass_count} PASS / {warn_count} WARN / {fail_count} FAIL",
        flush=True,
    )
    if fail_count:
        failed = audit.loc[
            audit["status"] == "FAIL",
            ["check_id", "observed", "expected"],
        ]
        raise RuntimeError(
            "P4审计失败："
            f"{failed.to_dict(orient='records')}"
        )
    return audit
