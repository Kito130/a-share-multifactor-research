from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .config import absolute, load_config, sql_path


PB_DISCLOSURE = (
    "使用供应商历史PB构造1/PB代理，未自行重建严格 "
    "point-in-time book equity，供应商历史修订政策未完全核验。"
)
CORPORATE_ACTION_DISCLOSURE = (
    "600270.SH 于2018-12-28按人工公司行动表换股为601598.SH；"
    "原始股数按3.8225倍转换，假定不行使现金选择权，允许非整数股，"
    "不计佣金、滑点或印花税；601598.SH首个可得市场价格出现前"
    "使用价值连续的公司行动承接估值。"
)


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


def _write_json_atomic(payload: dict[str, Any], relative_path: str) -> None:
    output = absolute(relative_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, output)


def _all_paths_relative(config: dict[str, Any]) -> bool:
    values = (
        list(config["inputs"].values())
        + list(config["outputs"].values())
        + list(config["protected_p3_inputs"])
    )
    return all(not Path(str(value)).is_absolute() for value in values)


def _report(
    audit: pd.DataFrame,
    overall: str,
    performance: pd.DataFrame,
    failures: pd.DataFrame,
    stale: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    impact: pd.DataFrame,
    original_hashes: pd.DataFrame,
    audited_at: str,
) -> str:
    baseline = performance.loc[performance["is_baseline"]].iloc[0]
    pass_count = int((audit["status"] == "PASS").sum())
    warn_count = int((audit["status"] == "WARN").sum())
    fail_count = int((audit["status"] == "FAIL").sum())
    lines = [
        "# P3 复合因子与基准回测审计报告",
        "",
        f"- 审计时间（UTC）：{audited_at}",
        f"- 总体状态：**{overall}**",
        (
            f"- 检查结果：{pass_count} PASS / {warn_count} WARN / "
            f"{fail_count} FAIL"
        ),
        "- 样本范围：仅 2016–2019 研究期",
        "- 未运行：2020–2021验证期、2022–2025 OOS、P4冻结协议",
        "",
        "## 冻结实现",
        "",
        "- 复合分数：三个月度 z-score 等权平均。",
        "- 每月选择前100只，目标等权1%，单票上限2%，目标总仓位100%。",
        "- 月末收盘形成信号，下一市场交易日开盘成交。",
        "- 先卖后买；现金不足时买单同比例缩放。",
        "- 买入受阻时资金留在现金；卖出受阻时旧仓继续持有。",
        "- 成交只使用复权开盘价；上一可得复权收盘价只用于估值。",
        "- 持仓为可分割复权单位，不模拟100股整手约束。",
        (
            "- 换股先把复权单位还原为原始股数，再按换股比例转换；"
            "事件记为CORPORATE_ACTION，不记为TRADE。"
        ),
        "",
        "## 基准成本情景",
        "",
        f"- 策略累计收益：{baseline['strategy_total_return']:.2%}",
        f"- 策略年化收益：{baseline['strategy_annualized_return']:.2%}",
        f"- 策略最大回撤：{baseline['strategy_max_drawdown']:.2%}",
        f"- 中证全指累计收益：{baseline['benchmark_total_return']:.2%}",
        f"- 中证全指年化收益：{baseline['benchmark_annualized_return']:.2%}",
        f"- 中证全指最大回撤：{baseline['benchmark_max_drawdown']:.2%}",
        (
            "- 年化收益差："
            f"{baseline['annualized_return_difference']:.2%}"
        ),
        f"- 总交易成本：{baseline['total_trading_cost']:,.2f} 元",
        f"- 买入失败：{int(baseline['failed_buy_orders'])} 单",
        f"- 卖出失败：{int(baseline['failed_sell_orders'])} 单",
        "",
        "以上仅为冻结研究期回测，不是验证期或OOS结果。",
        "",
        "## 成本压力",
        "",
        "| 情景 | 策略累计收益 | 年化收益 | 最大回撤 | 总成本 |",
        "|---|---:|---:|---:|---:|",
    ]
    order = {"STRESS_5BPS": 5, "BASE_10BPS": 10, "STRESS_20BPS": 20}
    for row in performance.sort_values(
        "cost_scenario",
        key=lambda values: values.map(order),
    ).itertuples(index=False):
        lines.append(
            f"| {row.cost_scenario} | {row.strategy_total_return:.2%} | "
            f"{row.strategy_annualized_return:.2%} | "
            f"{row.strategy_max_drawdown:.2%} | "
            f"{row.total_trading_cost:,.2f} |"
        )
    lines.extend(
        [
            "",
            "## 成交失败",
            "",
            "| 情景 | 方向 | 原因 | 数量 |",
            "|---|---|---|---:|",
        ]
    )
    for row in failures.itertuples(index=False):
        lines.append(
            f"| {row.cost_scenario} | {row.side} | "
            f"{row.failure_reason} | {int(row.failed_orders)} |"
        )
    action = corporate_actions.loc[
        corporate_actions["is_baseline"].astype(bool)
    ].iloc[0]
    baseline_impact = impact.loc[
        impact["is_baseline"].astype(bool)
    ].iloc[0]
    lines.extend(
        [
            "",
            "## 600270.SH公司行动修复",
            "",
            f"- {CORPORATE_ACTION_DISCLOSURE}",
            (
                "- 换股前600270原始股数："
                f"{action.old_share_quantity_before:,.9f}"
            ),
            (
                "- 换股后601598原始股数："
                f"{action.successor_share_quantity_after:,.9f}"
            ),
            (
                "- 即时组合价值差："
                f"{action.portfolio_value_difference_cny:,.8f}元"
            ),
            (
                "- 旧版→修复版基准年化收益："
                f"{baseline_impact.annualized_return_before:.6%} → "
                f"{baseline_impact.annualized_return_after:.6%}"
            ),
            (
                "- 旧版→修复版基准累计收益："
                f"{baseline_impact.total_return_before:.6%} → "
                f"{baseline_impact.total_return_after:.6%}"
            ),
            (
                "- 旧版→修复版基准最大回撤："
                f"{baseline_impact.max_drawdown_before:.6%} → "
                f"{baseline_impact.max_drawdown_after:.6%}"
            ),
            (
                "- 原始数据SHA-256："
                f"{len(original_hashes):,}个文件全部未变化。"
            ),
            "",
            "## 已披露限制",
            "",
            f"- {PB_DISCLOSURE}",
        ]
    )
    lines.extend(
        [
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


def audit_p3() -> pd.DataFrame:
    audited_at = datetime.now(UTC).isoformat()
    config = load_config()
    outputs = config["outputs"]
    collector = AuditCollector()
    required_keys = (
        "composite_signals",
        "target_holdings",
        "daily_portfolio",
        "actual_holdings",
        "orders",
        "failed_orders",
        "cash_ledger",
        "corporate_action_events",
        "rebalance_schedule",
        "rebalance_summary",
        "p3_input_hashes",
        "original_data_hash_check",
        "performance_summary",
        "annual_performance",
        "cost_scenario_comparison",
        "corporate_action_impact",
        "failure_reason_summary",
        "stale_position_summary",
        "p3_run_manifest",
    )
    missing = [
        outputs[key]
        for key in required_keys
        if not absolute(outputs[key]).is_file()
    ]
    collector.add(
        "P3-001",
        "文件",
        not missing,
        f"missing={missing}",
        "missing=[]",
        "P3规定输出必须全部存在。",
    )
    if missing:
        frame = collector.frame()
        _write_csv_atomic(frame, outputs["p3_audit_summary"])
        raise FileNotFoundError(f"缺少P3输出：{missing}")
    collector.add(
        "P3-002",
        "路径",
        _all_paths_relative(config),
        _all_paths_relative(config),
        True,
        "全部运行路径来自项目相对配置。",
    )

    p2_manifest = json.loads(
        absolute(config["inputs"]["p2_run_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    collector.add(
        "P3-003",
        "前置闸门",
        str(p2_manifest.get("status", "")).startswith("P2_ACCEPTED"),
        p2_manifest.get("status"),
        "P2_ACCEPTED*",
        "只有P2验收通过后才允许运行P3。",
    )
    hashes = pd.read_csv(absolute(outputs["p3_input_hashes"]))
    current_mismatches = 0
    for row in hashes.itertuples(index=False):
        path = absolute(row.path)
        if (
            not path.is_file()
            or path.stat().st_size != row.size_bytes_after
            or _sha256(path) != row.sha256_after
        ):
            current_mismatches += 1
    collector.add(
        "P3-004",
        "输入完整性",
        bool(hashes["match"].all()) and current_mismatches == 0,
        (
            f"recorded={bool(hashes['match'].all())}, "
            f"current_mismatches={current_mismatches}"
        ),
        "recorded=True, current_mismatches=0",
        "P2/P1面板、成交状态、基准和成本配置不得变化。",
    )

    paths = {key: sql_path(outputs[key]) for key in (
        "composite_signals",
        "target_holdings",
        "daily_portfolio",
        "actual_holdings",
        "orders",
        "failed_orders",
        "corporate_action_events",
    )}
    execution_path = sql_path(config["inputs"]["execution_status"])
    with duckdb.connect() as connection:
        for name, path in paths.items():
            connection.execute(
                f"""
                CREATE VIEW {name} AS
                SELECT * FROM read_parquet('{path}')
                """
            )
        signal_stats = connection.execute(
            """
            SELECT
                count(*),
                count(DISTINCT signal_date),
                min(signal_date),
                max(signal_date),
                count(*) FILTER (
                    WHERE signal_date >= DATE '2020-01-01'
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
            "P3-005",
            "复合因子",
            signal_stats[0] == 107292
            and signal_stats[1] == 48
            and signal_stats[4] == 0
            and signal_stats[5] <= 1e-12,
            (
                f"rows={signal_stats[0]}, months={signal_stats[1]}, "
                f"range={signal_stats[2]}..{signal_stats[3]}, "
                f"validation={signal_stats[4]}, "
                f"formula_error={signal_stats[5]}"
            ),
            "rows=107292, months=48, validation=0, error<=1e-12",
            "复合分数必须是三个P2标准化因子的等权平均。",
        )
        signal_columns = {
            row[0]
            for row in connection.execute(
                "DESCRIBE composite_signals"
            ).fetchall()
        }
        collector.add(
            "P3-006",
            "无前视",
            "next_month_return" not in signal_columns,
            sorted(signal_columns & {"next_month_return"}),
            [],
            "复合信号不得携带或使用下一月收益标签。",
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
            "P3-007",
            "目标持仓",
            target_stats[0] == 48
            and target_stats[1] == 100
            and target_stats[2] == 100
            and abs(target_stats[3] - 1.0) <= 1e-12
            and abs(target_stats[4] - 1.0) <= 1e-12
            and target_stats[5] <= 0.02
            and target_stats[6] == 1
            and target_stats[7] == 100,
            target_stats,
            "48 months; 100 names; gross=1; max_weight<=0.02; ranks=1..100",
            "目标组合必须Top-100、等权且不超过冻结上限。",
        )
        daily_stats = connection.execute(
            """
            SELECT
                count(*),
                count(DISTINCT cost_scenario),
                min(trade_date),
                max(trade_date),
                count(*) FILTER (
                    WHERE trade_date >= DATE '2020-01-01'
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
            "P3-008",
            "样本闸门",
            daily_stats[0] == 2868
            and daily_stats[1] == 3
            and pd.Timestamp(daily_stats[2]).date().isoformat()
            == "2016-01-29"
            and pd.Timestamp(daily_stats[3]).date().isoformat()
            == "2019-12-31"
            and daily_stats[4] == 0,
            (
                f"rows={daily_stats[0]}, scenarios={daily_stats[1]}, "
                f"range={daily_stats[2]}..{daily_stats[3]}, "
                f"validation={daily_stats[4]}"
            ),
            "rows=2868, scenarios=3, research only",
            "三个成本情景各956个市场交易日。",
        )
        collector.add(
            "P3-009",
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
            "组合不得借现金或使用杠杆，净值必须等于现金加持仓市值。",
        )
        initial = connection.execute(
            """
            SELECT
                count(*),
                min(total_nav_cny),
                max(total_nav_cny),
                min(cash_cny),
                max(cash_cny),
                max(holding_count)
            FROM daily_portfolio
            WHERE trade_date = DATE '2016-01-29'
            """
        ).fetchone()
        collector.add(
            "P3-010",
            "初始状态",
            initial[0] == 3
            and initial[1] == 100000000.0
            and initial[2] == 100000000.0
            and initial[3] == 100000000.0
            and initial[4] == 100000000.0
            and initial[5] == 0,
            initial,
            "3 scenarios, cash=nav=100,000,000, holdings=0",
            "首个信号日收盘仍为全现金，下一交易日才成交。",
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
            "P3-011",
            "净值收益",
            all(value <= 1e-12 for value in return_errors[:4])
            and return_errors[4] == 1,
            return_errors,
            "all errors<=1e-12, one benchmark",
            "策略及中证全指净值和日收益全量复算。",
        )
        order_stats = connection.execute(
            """
            SELECT
                count(*),
                count(DISTINCT cost_scenario),
                count(*) FILTER (
                    WHERE trade_date >= DATE '2020-01-01'
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
            "P3-012",
            "订单时序",
            order_stats[0] == 17586
            and order_stats[1] == 3
            and all(value == 0 for value in order_stats[2:]),
            order_stats,
            "orders=17586, scenarios=3, violations=0",
            "所有订单必须在信号后的研究期内交易日执行。",
        )
        execution_violations = connection.execute(
            f"""
            SELECT count(*)
            FROM orders
            LEFT JOIN read_parquet('{execution_path}') AS execution
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
            "P3-013",
            "成交限制",
            execution_violations == 0,
            execution_violations,
            0,
            "任何已成交订单都不得违反P1开盘成交状态。",
        )
        cost_errors = connection.execute(
            """
            SELECT
                max(abs(
                    commission_slippage_cost
                    - executed_notional * commission_slippage_rate
                )),
                count(*) FILTER (
                    WHERE side = 'BUY' AND stamp_duty_cost <> 0
                ),
                max(abs(
                    stamp_duty_cost - executed_notional * 0.001
                )) FILTER (
                    WHERE side = 'SELL' AND executed_notional > 0
                ),
                max(abs(
                    total_trading_cost
                    - commission_slippage_cost - stamp_duty_cost
                ))
            FROM orders
            """
        ).fetchone()
        collector.add(
            "P3-014",
            "交易成本",
            cost_errors[0] <= 1e-12
            and cost_errors[1] == 0
            and cost_errors[2] <= 1e-12
            and cost_errors[3] <= 1e-12,
            cost_errors,
            "errors<=1e-12, buy_stamp_rows=0",
            "5/10/20bps单边成本及研究期卖方0.001印花税全量复算。",
        )
        failed_count = connection.execute(
            """
            SELECT
                (SELECT count(*) FROM failed_orders),
                (SELECT count(*) FROM orders
                 WHERE order_status = 'FAILED'),
                count(*) FILTER (
                    WHERE side = 'BUY'
                ),
                count(*) FILTER (
                    WHERE side = 'SELL'
                ),
                count(*) FILTER (
                    WHERE side = 'SELL'
                      AND remaining_adjusted_units_after <= 0
                )
            FROM failed_orders
            """
        ).fetchone()
        collector.add(
            "P3-015",
            "失败订单",
            failed_count[0] == failed_count[1]
            and failed_count[0] > 0
            and failed_count[2] > 0
            and failed_count[3] > 0
            and failed_count[4] == 0,
            failed_count,
            (
                "failed output equals failed orders; both sides present; "
                "failed sells retain units"
            ),
            "失败买单零成交，失败卖单保留旧仓。",
        )
        preservation_violations = connection.execute(
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
            "P3-016",
            "失败卖出",
            preservation_violations == 0,
            preservation_violations,
            0,
            "失败卖出后的单位必须原样出现在当日实际持仓。",
        )
        holding_errors = connection.execute(
            """
            WITH held AS (
                SELECT
                    cost_scenario,
                    trade_date,
                    count(*) AS names,
                    sum(position_market_value) AS market_value,
                    sum(actual_weight) AS market_weight
                FROM actual_holdings
                GROUP BY cost_scenario, trade_date
            )
            SELECT
                min(actual_holdings.adjusted_units),
                max(abs(
                    daily_portfolio.market_value_cny
                    - coalesce(held.market_value, 0.0)
                )),
                max(abs(
                    daily_portfolio.market_value_weight
                    - coalesce(held.market_weight, 0.0)
                )),
                max(coalesce(held.names, 0)),
                max(actual_holdings.actual_weight)
            FROM daily_portfolio
            LEFT JOIN held USING (cost_scenario, trade_date)
            LEFT JOIN actual_holdings USING (cost_scenario, trade_date)
            """
        ).fetchone()
        collector.add(
            "P3-017",
            "实际持仓",
            holding_errors[0] > 0
            and holding_errors[1] <= 1e-6
            and holding_errors[2] <= 1e-12
            and holding_errors[3] >= 100,
            holding_errors,
            "units>0, value_error<=1e-6, weight_error<=1e-12",
            "实际持仓与每日组合市值一致；失败卖出可使持仓数超过100。",
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
        stale_rows = connection.execute(
            """
            SELECT
                count(*) FILTER (
                    WHERE valuation_price_source
                        <> 'CURRENT_ADJUSTED_CLOSE'
                ),
                count(*) FILTER (
                    WHERE valuation_price_source
                        <> 'CURRENT_ADJUSTED_CLOSE'
                      AND stale_calendar_days <= 0
                )
            FROM actual_holdings
            """
        ).fetchone()
        collector.add(
            "P3-018",
            "估值与成交",
            stale_execution == 0
            and stale_rows[0] > 0
            and stale_rows[1] == 0,
            (
                f"stale_executions={stale_execution}, "
                f"stale_valuations={stale_rows[0]}, "
                f"invalid_stale_days={stale_rows[1]}"
            ),
            "stale_executions=0, stale_valuations>0, invalid=0",
            "上一可得价格只能估值，绝不能虚构成交。",
        )

    schedule = pd.read_csv(
        absolute(outputs["rebalance_schedule"]),
        parse_dates=["signal_date", "scheduled_trade_date"],
    )
    benchmark = pd.read_parquet(
        absolute(config["inputs"]["benchmark_daily"])
    )
    benchmark["trade_date"] = pd.to_datetime(benchmark["trade_date"])
    scheduled = schedule.dropna(subset=["scheduled_trade_date"])
    next_date_errors = 0
    for row in scheduled.itertuples(index=False):
        expected = benchmark.loc[
            (benchmark["trade_date"] > row.signal_date)
            & (benchmark["trade_date"] <= pd.Timestamp("2019-12-31")),
            "trade_date",
        ].min()
        if expected != row.scheduled_trade_date:
            next_date_errors += 1
    collector.add(
        "P3-019",
        "调仓日",
        len(schedule) == 48
        and len(scheduled) == 47
        and next_date_errors == 0
        and schedule.iloc[-1]["schedule_status"]
        == "OUT_OF_SCOPE_NOT_EXECUTED",
        (
            f"signals={len(schedule)}, scheduled={len(scheduled)}, "
            f"next_date_errors={next_date_errors}, "
            f"last={schedule.iloc[-1]['schedule_status']}"
        ),
        "48 signals, 47 scheduled, errors=0, last out of scope",
        "2019-12信号不得越界到2020年成交。",
    )

    rebalances = pd.read_csv(
        absolute(outputs["rebalance_summary"]),
        parse_dates=["signal_date", "trade_date"],
    )
    orders = pd.read_parquet(absolute(outputs["orders"]))
    orders["trade_date"] = pd.to_datetime(orders["trade_date"])
    blocked = (
        orders.loc[
            (orders["side"] == "BUY")
            & (orders["order_status"] == "FAILED")
        ]
        .assign(
            reserved_cash=lambda frame: (
                frame["desired_order_notional"]
                * frame["cash_scaling_factor"]
                * (1.0 + frame["commission_slippage_rate"])
            )
        )
        .groupby(["cost_scenario", "trade_date"], as_index=False)[
            "reserved_cash"
        ]
        .sum()
    )
    reserved_check = rebalances.merge(
        blocked,
        on=["cost_scenario", "trade_date"],
        how="left",
    )
    reserved_check["reserved_cash"] = reserved_check[
        "reserved_cash"
    ].fillna(0.0)
    reserve_violations = int(
        (
            reserved_check["cash_after_open"] + 0.01
            < reserved_check["reserved_cash"]
        ).sum()
    )
    collector.add(
        "P3-020",
        "失败买入",
        reserve_violations == 0,
        reserve_violations,
        0,
        "失败买单对应的同比缩放资金必须留在现金中。",
    )

    performance = pd.read_csv(
        absolute(outputs["performance_summary"])
    )
    annual = pd.read_csv(absolute(outputs["annual_performance"]))
    collector.add(
        "P3-021",
        "绩效输出",
        len(performance) == 3
        and len(annual) == 12
        and set(performance["cost_scenario"])
        == {"STRESS_5BPS", "BASE_10BPS", "STRESS_20BPS"},
        f"performance={len(performance)}, annual={len(annual)}",
        "performance=3, annual=12",
        "三个成本情景和四个研究年份必须完整。",
    )
    returns_by_cost = performance.set_index("cost_scenario")[
        "strategy_total_return"
    ]
    costs_by_cost = performance.set_index("cost_scenario")[
        "total_trading_cost"
    ]
    monotonic = (
        returns_by_cost["STRESS_5BPS"]
        > returns_by_cost["BASE_10BPS"]
        > returns_by_cost["STRESS_20BPS"]
        and costs_by_cost["STRESS_5BPS"]
        < costs_by_cost["BASE_10BPS"]
        < costs_by_cost["STRESS_20BPS"]
    )
    collector.add(
        "P3-022",
        "成本压力",
        monotonic,
        (
            f"returns={returns_by_cost.to_dict()}, "
            f"costs={costs_by_cost.to_dict()}"
        ),
        "return 5bps>10bps>20bps; cost 5bps<10bps<20bps",
        "成本压力方向必须单调合理。",
    )
    daily = pd.read_parquet(absolute(outputs["daily_portfolio"]))
    metric_errors: list[float] = []
    for row in performance.itertuples(index=False):
        group = daily.loc[
            daily["cost_scenario"] == row.cost_scenario
        ].sort_values("trade_date")
        metric_errors.extend(
            [
                abs(
                    row.strategy_total_return
                    - (group.iloc[-1]["strategy_nav"] - 1.0)
                ),
                abs(
                    row.benchmark_total_return
                    - (group.iloc[-1]["benchmark_nav"] - 1.0)
                ),
                abs(
                    row.strategy_max_drawdown
                    - group["strategy_drawdown"].min()
                ),
                abs(
                    row.benchmark_max_drawdown
                    - group["benchmark_drawdown"].min()
                ),
            ]
        )
    collector.add(
        "P3-023",
        "绩效复算",
        max(metric_errors) <= 1e-12,
        max(metric_errors),
        "<=1e-12",
        "累计收益及最大回撤从每日净值独立复算。",
    )

    stale = pd.read_csv(
        absolute(outputs["stale_position_summary"]),
        parse_dates=[
            "first_stale_date",
            "last_stale_date",
            "last_available_price_date",
        ],
    )
    holdings = pd.read_parquet(absolute(outputs["actual_holdings"]))
    holdings["trade_date"] = pd.to_datetime(holdings["trade_date"])
    events = pd.read_parquet(
        absolute(outputs["corporate_action_events"])
    )
    for column in (
        "effective_date",
        "last_trade_date",
        "suspension_start_date",
        "record_date",
        "delist_date",
        "successor_first_price_date",
    ):
        events[column] = pd.to_datetime(events[column])
    manual_actions = pd.read_csv(
        absolute(config["inputs"]["corporate_actions"]),
        dtype={
            "last_trade_date": "string",
            "suspension_start_date": "string",
            "record_date": "string",
            "delist_date": "string",
        },
    )
    baseline_event = events.loc[events["is_baseline"].astype(bool)]
    ratio_error = (
        events["successor_share_quantity_after"]
        - events["old_share_quantity_before"]
        * events["exchange_ratio"]
    ).abs().max()
    value_error = events[
        [
            "position_value_difference_cny",
            "portfolio_value_difference_cny",
        ]
    ].abs().to_numpy().max()
    action_cost = events[
        [
            "commission_cost_cny",
            "stamp_duty_cost_cny",
            "slippage_cost_cny",
            "total_action_cost_cny",
        ]
    ].abs().to_numpy().max()
    old_after_effective = holdings.loc[
        (holdings["canonical_ts_code"] == "600270.SH")
        & (holdings["trade_date"] >= pd.Timestamp("2018-12-28"))
    ]
    terminal_old = holdings.loc[
        (holdings["canonical_ts_code"] == "600270.SH")
        & (holdings["trade_date"] == pd.Timestamp("2019-12-31"))
    ]
    successor_on_effective = holdings.loc[
        (holdings["canonical_ts_code"] == "601598.SH")
        & (holdings["trade_date"] == pd.Timestamp("2018-12-28"))
        & (
            holdings["position_origin_activity_type"]
            == "CORPORATE_ACTION"
        )
    ]
    successor_first_price = holdings.loc[
        (holdings["canonical_ts_code"] == "601598.SH")
        & (holdings["trade_date"] == pd.Timestamp("2019-01-18"))
        & (
            holdings["valuation_price_source"]
            == "CURRENT_ADJUSTED_CLOSE"
        )
    ]
    old_suspension_holdings = holdings.loc[
        (holdings["canonical_ts_code"] == "600270.SH")
        & (
            holdings["trade_date"].between(
                pd.Timestamp("2018-12-13"),
                pd.Timestamp("2018-12-27"),
            )
        )
    ]
    old_trades_after_suspension = orders.loc[
        (orders["canonical_ts_code"] == "600270.SH")
        & (orders["trade_date"] >= pd.Timestamp("2018-12-13"))
        & (orders["executed_notional"] > 0)
    ]
    successor_orders = orders.loc[
        orders["canonical_ts_code"] == "601598.SH"
    ].sort_values(["cost_scenario", "trade_date"])
    filled_successor_sells = successor_orders.loc[
        (successor_orders["side"] == "SELL")
        & (successor_orders["executed_notional"] > 0)
    ]
    manual_valid = (
        len(manual_actions) == 1
        and manual_actions.iloc[0]["old_ts_code"] == "600270.SH"
        and manual_actions.iloc[0]["successor_ts_code"] == "601598.SH"
        and manual_actions.iloc[0]["action_type"]
        == "STOCK_SWAP_ABSORPTION"
        and manual_actions.iloc[0]["last_trade_date"] == "20181212"
        and manual_actions.iloc[0]["suspension_start_date"]
        == "20181213"
        and manual_actions.iloc[0]["record_date"] == "20181227"
        and manual_actions.iloc[0]["delist_date"] == "20181228"
        and abs(
            float(manual_actions.iloc[0]["exchange_ratio"]) - 3.8225
        )
        <= 1e-12
        and float(manual_actions.iloc[0]["cash_component"]) == 0
        and float(manual_actions.iloc[0]["commission_rate"]) == 0
        and float(manual_actions.iloc[0]["stamp_duty_rate"]) == 0
        and manual_actions.iloc[0]["assumption"]
        == "CASH_OPTION_NOT_EXERCISED_AUTOMATIC_SHARE_EXCHANGE"
    )
    action_passed = (
        manual_valid
        and len(events) == 3
        and events["cost_scenario"].nunique() == 3
        and set(events["activity_type"]) == {"CORPORATE_ACTION"}
        and set(events["old_ts_code"]) == {"600270.SH"}
        and set(events["successor_ts_code"]) == {"601598.SH"}
        and set(events["effective_date"])
        == {pd.Timestamp("2018-12-28")}
        and ratio_error <= 1e-8
        and value_error <= 1e-6
        and action_cost <= 1e-12
        and old_after_effective.empty
        and terminal_old.empty
        and len(successor_on_effective) == 3
        and len(successor_first_price) == 3
        and not old_suspension_holdings.empty
        and set(old_suspension_holdings["valuation_price_source"])
        == {"LAST_AVAILABLE_ADJUSTED_CLOSE"}
        and old_trades_after_suspension.empty
        and not successor_orders.empty
        and set(orders["activity_type"]) == {"TRADE"}
        and set(successor_orders["activity_type"]) == {"TRADE"}
        and set(successor_orders.groupby("cost_scenario")[
            "trade_date"
        ].min()) == {pd.Timestamp("2019-01-02")}
        and len(filled_successor_sells) == 3
        and set(filled_successor_sells["trade_date"])
        == {pd.Timestamp("2019-02-01")}
        and len(baseline_event) == 1
    )
    collector.add(
        "P3-024",
        "公司行动",
        action_passed,
        {
            "manual_valid": manual_valid,
            "events": len(events),
            "ratio_error": ratio_error,
            "value_error_cny": value_error,
            "maximum_action_cost_cny": action_cost,
            "old_holdings_after_effective": len(old_after_effective),
            "terminal_old_holdings": len(terminal_old),
            "successor_holdings_on_effective": len(
                successor_on_effective
            ),
            "successor_first_market_price_rows": len(
                successor_first_price
            ),
            "old_trades_after_suspension": len(
                old_trades_after_suspension
            ),
            "first_successor_order_date": (
                successor_orders["trade_date"].min()
                if not successor_orders.empty
                else None
            ),
            "filled_successor_sell_date": (
                filled_successor_sells["trade_date"].min()
                if not filled_successor_sells.empty
                else None
            ),
        },
        (
            "3 CORPORATE_ACTION events; exact 3.8225 ratio; "
            "value difference<=1e-6; zero action costs; old terminal "
            "position=0; successor sold only at normal rebalance"
        ),
        CORPORATE_ACTION_DISCLOSURE,
    )
    collector.add(
        "P3-025",
        "PB风险",
        True,
        "disclosed",
        "disclosed",
        PB_DISCLOSURE,
        warning=True,
    )

    original_hashes = pd.read_csv(
        absolute(outputs["original_data_hash_check"])
    )
    original_current_mismatches = 0
    for row in original_hashes.itertuples(index=False):
        path = absolute(str(row.path))
        if (
            not path.is_file()
            or path.stat().st_size != row.size_bytes_after
            or _sha256(path) != row.sha256_after
        ):
            original_current_mismatches += 1
    original_hash_passed = (
        len(original_hashes) > 17000
        and original_hashes[
            [
                "matches_p1_before",
                "matches_p1_after",
                "unchanged_during_build",
            ]
        ].all().all()
        and (
            original_hashes["sha256_before"]
            == original_hashes["sha256_after"]
        ).all()
        and original_current_mismatches == 0
    )
    collector.add(
        "P3-029",
        "原始数据完整性",
        original_hash_passed,
        {
            "files": len(original_hashes),
            "build_mismatches": int(
                (
                    ~original_hashes[
                        "unchanged_during_build"
                    ].astype(bool)
                ).sum()
            ),
            "current_mismatches": original_current_mismatches,
        },
        "files>17000; build_mismatches=0; current_mismatches=0",
        "P3修复不得改动raw、static、_parts或external原始文件。",
    )

    impact = pd.read_csv(
        absolute(outputs["corporate_action_impact"])
    )
    archive_manifest = pd.read_csv(
        absolute(config["inputs"]["legacy_p3_archive_manifest"])
    )
    archived_performance = absolute(
        config["inputs"]["legacy_p3_performance_summary"]
    )
    archived_hash_row = archive_manifest.loc[
        archive_manifest["source_path"]
        == "results/p3_backtest/performance_summary.csv"
    ]
    impact_passed = (
        len(impact) == 3
        and impact["cost_scenario"].nunique() == 3
        and impact["is_baseline"].astype(bool).sum() == 1
        and len(archived_hash_row) == 1
        and _sha256(archived_performance)
        == archived_hash_row.iloc[0]["sha256"]
        and impact[
            [
                "annualized_return_before",
                "annualized_return_after",
                "annualized_return_difference",
                "total_return_before",
                "total_return_after",
                "total_return_difference",
                "max_drawdown_before",
                "max_drawdown_after",
                "max_drawdown_difference",
            ]
        ].notna().all().all()
    )
    collector.add(
        "P3-030",
        "修复前后对比",
        impact_passed,
        {
            "archive_files": len(archive_manifest),
            "scenarios": len(impact),
            "legacy_performance_sha256": (
                _sha256(archived_performance)
            ),
        },
        "archive hash matches; 3 scenario comparisons; no missing metrics",
        "旧版P3结果必须保留并用SHA-256锁定，修复差异必须可复算。",
    )

    manifest_path = absolute(outputs["p3_run_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scope_violations = [
        key
        for key, value in manifest["scope_guards"].items()
        if bool(value)
    ]
    collector.add(
        "P3-026",
        "范围",
        not scope_violations,
        scope_violations,
        [],
        "P3不得运行验证期、OOS或生成P4冻结协议。",
    )
    hash_mismatches: list[str] = []
    for key, expected_hash in manifest["output_sha256"].items():
        path = absolute(outputs[key])
        if not path.is_file() or _sha256(path) != expected_hash:
            hash_mismatches.append(key)
    collector.add(
        "P3-027",
        "输出完整性",
        not hash_mismatches,
        hash_mismatches,
        [],
        "所有P3构建输出必须与manifest SHA-256一致。",
    )
    collector.add(
        "P3-028",
        "实现假设",
        (
            manifest["frozen_methodology"]["position_units"]
            == "fractional adjusted-price units; no board-lot rounding"
            and manifest["frozen_methodology"][
                "corporate_action_activity"
            ]
            == (
                "CORPORATE_ACTION, never TRADE; zero commission, "
                "slippage and stamp duty"
            )
            and manifest["frozen_methodology"]["terminal_liquidation"]
            is False
        ),
        {
            "position_units": manifest["frozen_methodology"][
                "position_units"
            ],
            "terminal_liquidation": manifest["frozen_methodology"][
                "terminal_liquidation"
            ],
            "corporate_action_activity": manifest[
                "frozen_methodology"
            ]["corporate_action_activity"],
        },
        (
            "fractional adjusted units; corporate action is not trade; "
            "terminal_liquidation=False"
        ),
        "整手取整、把换股伪装成交易或期末强制清仓均不得加入。",
    )

    frame = collector.frame().sort_values(
        "check_id", kind="mergesort"
    ).reset_index(drop=True)
    fail_count = int((frame["status"] == "FAIL").sum())
    warn_count = int((frame["status"] == "WARN").sum())
    overall = (
        "FAIL"
        if fail_count
        else (
            (
                "PASS_WITH_DISCLOSED_LIMITATION"
                if warn_count == 1
                else "PASS_WITH_DISCLOSED_LIMITATIONS"
            )
            if warn_count
            else "PASS"
        )
    )
    _write_csv_atomic(frame, outputs["p3_audit_summary"])
    failures = pd.read_csv(absolute(outputs["failure_reason_summary"]))
    report = _report(
        frame,
        overall,
        performance,
        failures,
        stale,
        events,
        impact,
        original_hashes,
        audited_at,
    )
    _write_text_atomic(report, outputs["p3_audit_report"])

    manifest["status"] = (
        "P3_AUDIT_FAILED"
        if fail_count
        else (
            (
                "P3_ACCEPTED_WITH_DISCLOSED_LIMITATION"
                if warn_count == 1
                else "P3_ACCEPTED_WITH_DISCLOSED_LIMITATIONS"
            )
            if warn_count
            else "P3_ACCEPTED"
        )
    )
    manifest["audit"] = {
        "audited_at_utc": audited_at,
        "overall_status": overall,
        "pass_count": int((frame["status"] == "PASS").sum()),
        "warn_count": warn_count,
        "fail_count": fail_count,
        "audit_summary_sha256": _sha256(
            absolute(outputs["p3_audit_summary"])
        ),
        "audit_report_sha256": _sha256(
            absolute(outputs["p3_audit_report"])
        ),
    }
    _write_json_atomic(manifest, outputs["p3_run_manifest"])
    print(
        f"[P3 AUDIT] {overall}: "
        f"PASS={manifest['audit']['pass_count']}, "
        f"WARN={warn_count}, FAIL={fail_count}",
        flush=True,
    )
    if fail_count:
        failed = frame.loc[frame["status"] == "FAIL"]
        raise RuntimeError(
            "P3审计失败："
            + failed[
                ["check_id", "observed", "expected"]
            ].to_dict(orient="records").__repr__()
        )
    return frame
