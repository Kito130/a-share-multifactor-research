from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml

from a_share_p4.build import PB_DISCLOSURE, _sha256

from .build import validate_p5_preflight
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


def _all_paths_relative(config: dict[str, Any]) -> bool:
    values = (
        list(config["inputs"].values())
        + list(config["outputs"].values())
        + list(config["protected_p5_inputs"])
    )
    return all(not Path(str(value)).is_absolute() for value in values)


def _report(
    audit: pd.DataFrame,
    overall: str,
    oos: pd.DataFrame,
    annual: pd.DataFrame,
    ic: pd.DataFrame,
    reproduction: pd.DataFrame,
    terminal_disclosure: dict[str, Any],
    frozen_hash: str,
    audited_at: str,
) -> str:
    baseline = oos.loc[oos["is_baseline"].astype(bool)].iloc[0]
    yearly = annual.loc[
        annual["is_baseline"].astype(bool)
        & annual["year"].between(2022, 2025)
    ].sort_values("year")
    pass_count = int((audit["status"] == "PASS").sum())
    warn_count = int((audit["status"] == "WARN").sum())
    fail_count = int((audit["status"] == "FAIL").sum())
    lines = [
        "# P5 最终 OOS 审计报告",
        "",
        f"- 审计时间（UTC）：{audited_at}",
        f"- 总体状态：**{overall}**",
        (
            f"- 检查结果：{pass_count} PASS / {warn_count} WARN / "
            f"{fail_count} FAIL"
        ),
        "- 冻结 OOS 研究门禁已在首次运行之前满足。",
        "- 最终 OOS：2022-01-01 至 2025-12-31。",
        f"- 冻结配置 SHA-256：`{frozen_hash}`。",
        "- 冻结配置未修改，验证后未调参，P6未运行。",
        "",
        "## 基准成本情景",
        "",
        f"- 策略累计收益：{baseline['strategy_total_return']:.2%}",
        f"- 策略年化收益：{baseline['strategy_annualized_return']:.2%}",
        (
            "- 策略最大回撤："
            f"{baseline['strategy_max_drawdown_within_period']:.2%}"
        ),
        f"- 策略年化波动率：{baseline['strategy_annualized_volatility']:.2%}",
        f"- 夏普比率：{baseline['strategy_sharpe_zero_rf']:.3f}",
        f"- 中证全指累计收益：{baseline['benchmark_total_return']:.2%}",
        f"- 中证全指年化收益：{baseline['benchmark_annualized_return']:.2%}",
        (
            "- 年化收益差："
            f"{baseline['annualized_return_difference']:.2%}"
        ),
        f"- 信息比率：{baseline['information_ratio']:.3f}",
        f"- 总交易成本：{baseline['total_trading_cost']:,.2f} 元",
        "",
        "## 年度结果",
        "",
        "| 年份 | 策略 | 中证全指 | 差值 |",
        "|---:|---:|---:|---:|",
    ]
    for row in yearly.itertuples(index=False):
        lines.append(
            f"| {row.year} | {row.strategy_return:.2%} | "
            f"{row.benchmark_return:.2%} | "
            f"{row.return_difference:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 单因子OOS诊断",
            "",
            "| 因子 | 可评价月份 | 平均Rank IC | 年化ICIR |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in ic.itertuples(index=False):
        lines.append(
            f"| {row.factor} | {int(row.months)} | "
            f"{row.mean_rank_ic:.6f} | "
            f"{row.rank_icir_annualized:.6f} |"
        )
    lines.extend(
        [
            "",
            "## P4前缀复现",
            "",
            "| 检查 | 旧行数 | 新行数 | 最大误差 | 状态 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in reproduction.itertuples(index=False):
        lines.append(
            f"| {row.check} | {row.old_rows} | {row.new_rows} | "
            f"{row.maximum_numeric_error:.3e} | {row.status} |"
        )
    lines.extend(
        [
            "",
            "## 期末退市估值限制",
            "",
            (
                "- 基准情景期末按最后可得价格估值的退市状态证券："
                f"{terminal_disclosure['baseline_codes']}。"
            ),
            (
                "- 基准情景相关市值："
                f"{terminal_disclosure['baseline_value']:,.2f} 元，"
                f"占净值 {terminal_disclosure['baseline_weight']:.4%}。"
            ),
            (
                "- 静态表将这些证券标为 `list_status=D`，但 processed "
                "数据缺少可用于冻结回测的精确退市日、换股、现金回收或"
                "清算口径。"
            ),
            (
                "- P5 按事先冻结的“上一可得复权收盘价估值、期末不强制"
                "清仓”规则保留结果；没有在看到OOS后补写公司行动、归零"
                "或重跑。该项可能高估或低估最终净值。"
            ),
            "",
            "## 已披露限制",
            "",
            f"- {PB_DISCLOSURE}",
            "- 2025-12信号无项目范围外下一月标签，IC只评价47个月。",
            "- 最终OOS结果被冻结保留，不得用于覆盖配置或重定义本次OOS。",
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


def audit_p5() -> pd.DataFrame:
    audited_at = datetime.now(UTC).isoformat()
    gate = validate_p5_preflight(require_fresh_run=False)
    config = gate["control"]
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
        "oos_factor_panel",
        "oos_composite_signals",
        "oos_target_holdings",
        "oos_daily_portfolio",
        "oos_actual_holdings",
        "oos_orders",
        "oos_failed_orders",
        "oos_cash_ledger",
        "oos_corporate_action_events",
        "oos_factor_coverage",
        "oos_monthly_rank_ic",
        "oos_ic_summary",
        "oos_quintile_returns",
        "oos_annual_results",
        "oos_factor_correlations_monthly",
        "oos_factor_correlations_summary",
        "oos_industry_exposure",
        "oos_size_exposure",
        "oos_worst_periods",
        "full_rebalance_schedule",
        "oos_rebalance_schedule",
        "oos_rebalance_summary",
        "p4_reproduction",
        "p5_input_hashes",
        "run_intent",
        "performance_summary_full",
        "annual_performance",
        "period_performance",
        "oos_performance",
        "oos_cost_scenario_comparison",
        "oos_failure_reason_summary",
        "oos_stale_position_summary",
        "oos_terminal_holdings",
        "p5_result_report",
        "p5_run_manifest",
    )
    missing = [
        outputs[key]
        for key in required_keys
        if not absolute(outputs[key]).is_file()
    ]
    collector.add(
        "P5-001",
        "文件",
        not missing,
        f"missing={missing}",
        "missing=[]",
        "P5规定输出必须全部存在。",
    )
    if missing:
        frame = collector.frame()
        _write_csv_atomic(frame, outputs["p5_audit_summary"])
        raise FileNotFoundError(f"缺少P5输出：{missing}")

    collector.add(
        "P5-002",
        "路径",
        _all_paths_relative(config),
        _all_paths_relative(config),
        True,
        "全部输入、输出和受保护文件均使用项目相对路径。",
    )

    manifest = json.loads(
        absolute(outputs["p5_run_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    intent = json.loads(
        absolute(outputs["run_intent"]).read_text(encoding="utf-8")
    )
    authorization_pass = (
        intent.get("attempt_number") == 1
        and intent.get("status") == "COMPLETED"
        and intent.get("authorization_obtained_before_run") is True
        and intent.get("one_shot_final_oos") is True
        and intent.get("parameters_retuned") is False
        and intent.get("frozen_config_modified") is False
        and manifest.get("authorization", {}).get(
            "obtained_before_run"
        )
        is True
    )
    collector.add(
        "P5-003",
        "OOS 执行许可",
        authorization_pass,
        {
            "attempt": intent.get("attempt_number"),
            "status": intent.get("status"),
            "reference": intent.get("authorization_reference"),
        },
        "attempt=1, completed, authorization before run",
        "最终 OOS 必须在冻结研究闸门通过后一次性运行。",
    )

    frozen_path = absolute(config["inputs"]["frozen_config"])
    current_hash = _sha256(frozen_path)
    recorded = re.search(
        r"sha256=([0-9a-f]{64})",
        absolute(config["inputs"]["frozen_config_sha256"]).read_text(
            encoding="utf-8"
        ),
    )
    hash_pass = (
        recorded is not None
        and current_hash == gate["frozen_hash"]
        and current_hash == recorded.group(1)
        and current_hash
        == manifest.get("frozen_config", {}).get("sha256")
        and manifest.get("frozen_config", {}).get("modified") is False
        and manifest.get("frozen_config", {}).get(
            "parameters_retuned"
        )
        is False
    )
    collector.add(
        "P5-004",
        "冻结配置",
        hash_pass,
        current_hash,
        gate["frozen_hash"],
        "P5前后必须是同一份P4冻结配置。",
    )

    hashes = pd.read_csv(absolute(outputs["p5_input_hashes"]))
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
        "P5-005",
        "输入完整性",
        len(hashes) == len(config["protected_p5_inputs"])
        and bool(hashes["match"].all())
        and current_mismatches == 0,
        (
            f"files={len(hashes)}, "
            f"recorded={bool(hashes['match'].all())}, "
            f"current_mismatches={current_mismatches}"
        ),
        (
            f"files={len(config['protected_p5_inputs'])}, "
            "recorded=True, current_mismatches=0"
        ),
        "冻结配置、P4前缀及底层数据均不得变化。",
    )

    reproduction = pd.read_csv(
        absolute(outputs["p4_reproduction"])
    )
    reproduction_pass = (
        len(reproduction) == 7
        and bool((reproduction["status"] == "PASS").all())
        and int(reproduction["missing_or_extra_rows"].sum()) == 0
        and int(reproduction["categorical_mismatches"].sum()) == 0
    )
    collector.add(
        "P5-006",
        "P4前缀复现",
        reproduction_pass,
        (
            f"checks={len(reproduction)}, "
            f"pass={(reproduction['status'] == 'PASS').sum()}, "
            f"max_error={reproduction['maximum_numeric_error'].max():.3e}"
        ),
        "7/7 PASS; no row or categorical mismatch",
        "进入最终OOS后，截止2021年末的冻结输出必须不变。",
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
        "oos_factor_panel",
        "oos_daily_portfolio",
        "oos_actual_holdings",
        "oos_orders",
    )
    with duckdb.connect() as connection:
        for key in view_keys:
            connection.execute(
                f"""
                CREATE VIEW {key} AS
                SELECT * FROM read_parquet('{sql_path(outputs[key])}')
                """
            )

        factor = connection.execute(
            """
            SELECT
                count(*),
                count(DISTINCT signal_date),
                min(signal_date),
                max(signal_date),
                count(*) FILTER (
                    WHERE signal_date BETWEEN DATE '2022-01-01'
                                          AND DATE '2025-12-31'
                ),
                count(DISTINCT signal_date) FILTER (
                    WHERE signal_date BETWEEN DATE '2022-01-01'
                                          AND DATE '2025-12-31'
                ),
                count(*) FILTER (
                    WHERE signal_date > DATE '2025-12-31'
                )
            FROM factor_panel
            """
        ).fetchone()
        collector.add(
            "P5-007",
            "因子范围",
            factor[0] > 0
            and factor[1] == 120
            and pd.Timestamp(factor[2]).date().isoformat()
            == "2016-01-29"
            and pd.Timestamp(factor[3]).date().isoformat()
            == "2025-12-31"
            and factor[4] > 0
            and factor[5] == 48
            and factor[6] == 0,
            (
                f"rows={factor[0]}, months={factor[1]}, "
                f"range={factor[2]}..{factor[3]}, "
                f"oos_rows={factor[4]}, "
                f"oos_months={factor[5]}, post_end={factor[6]}"
            ),
            "120 full months; 48 OOS months; post-end=0",
            "因子构建只能延长到最终冻结截止日。",
        )

        duplicate_factors = connection.execute(
            """
            SELECT count(*)
            FROM (
                SELECT canonical_ts_code, signal_date
                FROM factor_panel
                GROUP BY canonical_ts_code, signal_date
                HAVING count(*) > 1
            )
            """
        ).fetchone()[0]
        collector.add(
            "P5-008",
            "代码连续性",
            duplicate_factors == 0,
            duplicate_factors,
            0,
            "最终OOS按canonical主体形成的月度面板不得重复。",
        )

        code_boundary = connection.execute(
            """
            SELECT
                max(signal_date) FILTER (
                    WHERE canonical_ts_code = '302132.SZ'
                      AND ts_code = '300114.SZ'
                ),
                min(signal_date) FILTER (
                    WHERE canonical_ts_code = '302132.SZ'
                      AND ts_code = '302132.SZ'
                ),
                count(*) FILTER (
                    WHERE canonical_ts_code = '302132.SZ'
                      AND (
                          (
                              signal_date < DATE '2025-02-17'
                              AND ts_code <> '300114.SZ'
                          )
                          OR (
                              signal_date >= DATE '2025-02-17'
                              AND ts_code <> '302132.SZ'
                          )
                      )
                )
            FROM factor_panel
            """
        ).fetchone()
        collector.add(
            "P5-009",
            "2025换码",
            code_boundary[0] is not None
            and code_boundary[1] is not None
            and pd.Timestamp(code_boundary[0])
            < pd.Timestamp("2025-02-17")
            and pd.Timestamp(code_boundary[1])
            >= pd.Timestamp("2025-02-17")
            and code_boundary[2] == 0,
            code_boundary,
            "old before 2025-02-17; new on/after; violations=0",
            "300114→302132必须使用真实交易代码且主体连续。",
        )

        signals = connection.execute(
            """
            SELECT
                count(*),
                count(DISTINCT signal_date),
                max(abs(
                    composite_score
                    - (bm_proxy_z + momentum_12_1_z + lowvol_60_z)
                      / 3.0
                )),
                count(*) FILTER (
                    WHERE signal_date > DATE '2025-12-31'
                )
            FROM composite_signals
            """
        ).fetchone()
        signal_columns = {
            row[0]
            for row in connection.execute(
                "DESCRIBE composite_signals"
            ).fetchall()
        }
        collector.add(
            "P5-010",
            "复合信号",
            signals[0] > 0
            and signals[1] == 120
            and signals[2] <= 1e-12
            and signals[3] == 0
            and "next_month_return" not in signal_columns,
            (
                f"rows={signals[0]}, months={signals[1]}, "
                f"formula_error={signals[2]}, post_end={signals[3]}, "
                f"label_present={'next_month_return' in signal_columns}"
            ),
            "120 months; formula error<=1e-12; no label; post-end=0",
            "最终OOS信号继续使用冻结等权公式且不携带未来标签。",
        )

        targets = connection.execute(
            """
            WITH monthly AS (
                SELECT
                    signal_date,
                    count(*) AS names,
                    sum(target_weight) AS gross,
                    min(selection_rank) AS min_rank,
                    max(selection_rank) AS max_rank,
                    max(target_weight) AS max_weight
                FROM target_holdings
                GROUP BY signal_date
            )
            SELECT
                count(*), min(names), max(names),
                min(gross), max(gross),
                min(min_rank), max(max_rank), max(max_weight)
            FROM monthly
            """
        ).fetchone()
        collector.add(
            "P5-011",
            "目标持仓",
            targets[0] == 120
            and targets[1:3] == (100, 100)
            and abs(targets[3] - 1.0) <= 1e-12
            and abs(targets[4] - 1.0) <= 1e-12
            and targets[5:7] == (1, 100)
            and targets[7] <= 0.02,
            targets,
            "120 months; 100 names; gross=1; ranks=1..100",
            "最终OOS不得改变Top-100等权规则。",
        )

        daily = connection.execute(
            """
            SELECT
                count(*),
                count(DISTINCT cost_scenario),
                min(trade_date),
                max(trade_date),
                count(*) FILTER (
                    WHERE trade_date BETWEEN DATE '2022-01-01'
                                         AND DATE '2025-12-31'
                ),
                count(*) FILTER (
                    WHERE trade_date > DATE '2025-12-31'
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
            "P5-012",
            "组合范围",
            daily[0] > 0
            and daily[1] == 3
            and pd.Timestamp(daily[2]).date().isoformat()
            == "2016-01-29"
            and pd.Timestamp(daily[3]).date().isoformat()
            == "2025-12-31"
            and daily[4] > 0
            and daily[5] == 0,
            (
                f"rows={daily[0]}, scenarios={daily[1]}, "
                f"range={daily[2]}..{daily[3]}, "
                f"oos_rows={daily[4]}, post_end={daily[5]}"
            ),
            "3 scenarios through 2025-12-31; post-end=0",
            "完整连续模拟不得越过最终OOS截止日。",
        )
        collector.add(
            "P5-013",
            "会计恒等式",
            daily[6] >= -1e-8
            and daily[7] > 0
            and daily[8] <= 1e-6
            and daily[9] <= 1e-12,
            (
                f"min_cash={daily[6]}, min_nav={daily[7]}, "
                f"nav_error={daily[8]}, weight_error={daily[9]}"
            ),
            "cash>=0, nav>0, errors within tolerance",
            "最终OOS不得借现金或使用杠杆。",
        )

        return_errors = connection.execute(
            """
            WITH lagged AS (
                SELECT
                    *,
                    lag(strategy_nav) OVER (
                        PARTITION BY cost_scenario ORDER BY trade_date
                    ) AS prior_strategy_nav,
                    lag(benchmark_nav) OVER (
                        PARTITION BY cost_scenario ORDER BY trade_date
                    ) AS prior_benchmark_nav,
                    first_value(benchmark_close) OVER (
                        PARTITION BY cost_scenario ORDER BY trade_date
                    ) AS first_benchmark_close
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
                )) FILTER (WHERE prior_benchmark_nav IS NOT NULL)
            FROM lagged
            """
        ).fetchone()
        collector.add(
            "P5-014",
            "净值收益",
            all(value <= 1e-12 for value in return_errors),
            return_errors,
            "all errors<=1e-12",
            "策略与中证全指净值和日收益必须可全量复算。",
        )

        oos_daily = connection.execute(
            """
            SELECT
                count(*),
                count(DISTINCT cost_scenario),
                min(trade_date),
                max(trade_date),
                count(*) FILTER (
                    WHERE trade_date < DATE '2022-01-01'
                       OR trade_date > DATE '2025-12-31'
                )
            FROM oos_daily_portfolio
            """
        ).fetchone()
        collector.add(
            "P5-015",
            "OOS切片",
            oos_daily[0] > 0
            and oos_daily[1] == 3
            and pd.Timestamp(oos_daily[2]) >= pd.Timestamp(
                "2022-01-01"
            )
            and pd.Timestamp(oos_daily[3]).date().isoformat()
            == "2025-12-31"
            and oos_daily[4] == 0,
            oos_daily,
            "3 scenarios strictly within 2022–2025",
            "发布的OOS明细不得混入研究期或验证期。",
        )

        order_stats = connection.execute(
            """
            SELECT
                count(*),
                count(DISTINCT cost_scenario),
                count(*) FILTER (
                    WHERE trade_date <= signal_date
                       OR trade_date > DATE '2025-12-31'
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
            "P5-016",
            "订单时序",
            order_stats[0] > 0
            and order_stats[1] == 3
            and all(value == 0 for value in order_stats[2:]),
            order_stats,
            "orders>0; scenarios=3; violations=0",
            "所有成交必须在信号后、截止日前使用可得复权开盘价。",
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
            "P5-017",
            "成交限制",
            execution_violations == 0,
            execution_violations,
            0,
            "最终OOS成交不得违反停牌或开盘涨跌停限制。",
        )

        costs = connection.execute(
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
                max(abs(
                    total_trading_cost
                    - commission_slippage_cost - stamp_duty_cost
                )),
                count(*) FILTER (
                    WHERE side = 'BUY' AND stamp_duty_cost <> 0
                ),
                count(*) FILTER (
                    WHERE side = 'SELL'
                      AND executed_notional > 0
                      AND trade_date < DATE '2023-08-28'
                      AND abs(stamp_duty_rate - 0.001) > 1e-15
                ),
                count(*) FILTER (
                    WHERE side = 'SELL'
                      AND executed_notional > 0
                      AND trade_date >= DATE '2023-08-28'
                      AND abs(stamp_duty_rate - 0.0005) > 1e-15
                )
            FROM orders
            """
        ).fetchone()
        collector.add(
            "P5-018",
            "交易成本",
            costs[0] <= 1e-12
            and costs[1] <= 1e-12
            and costs[2] <= 1e-12
            and costs[3:] == (0, 0, 0),
            costs,
            "formula errors<=1e-12; buy=0; sell policy violations=0",
            "2023-08-28印花税减半必须按冻结生效日进入最终OOS。",
        )

        failed = connection.execute(
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
        preservation = connection.execute(
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
            "P5-019",
            "失败订单",
            failed[0] > 0
            and failed[1:] == (0, 0)
            and preservation == 0,
            f"stats={failed}, preservation_errors={preservation}",
            "failed>0; fills=0; lost positions=0",
            "失败订单不得成交，卖不出的持仓必须继续保留。",
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
        collector.add(
            "P5-020",
            "停牌估值",
            stale_execution == 0,
            stale_execution,
            0,
            "上一可得收盘价只允许估值，不允许成交。",
        )

        actions = connection.execute(
            """
            SELECT
                count(*),
                count(DISTINCT cost_scenario),
                count(*) FILTER (
                    WHERE activity_type <> 'CORPORATE_ACTION'
                       OR old_ts_code <> '600270.SH'
                       OR successor_ts_code <> '601598.SH'
                       OR abs(exchange_ratio - 3.8225) > 1e-12
                ),
                max(abs(
                    successor_share_quantity_after
                    - old_share_quantity_before * exchange_ratio
                )),
                max(abs(portfolio_value_difference_cny)),
                max(abs(total_action_cost_cny)),
                count(*) FILTER (
                    WHERE effective_date >= DATE '2022-01-01'
                )
            FROM corporate_action_events
            """
        ).fetchone()
        collector.add(
            "P5-021",
            "公司行动",
            actions[0] == 3
            and actions[1] == 3
            and actions[2] == 0
            and actions[3] <= 1e-8
            and actions[4] <= 1e-6
            and actions[5] <= 1e-12
            and actions[6] == 0,
            actions,
            "3 frozen prefix events; continuous; zero cost; OOS events=0",
            "已冻结600270换股必须保持不变；人工表没有OOS公司行动。",
        )

        terminal_delisted = connection.execute(
            f"""
            WITH terminal AS (
                SELECT *
                FROM oos_actual_holdings
                WHERE trade_date = (
                    SELECT max(trade_date)
                    FROM oos_actual_holdings
                )
            ),
            listing AS (
                SELECT
                    canonical_ts_code,
                    max(delist_date) AS delist_date
                FROM read_parquet(
                    '{sql_path(config["inputs"]["daily_panel"])}'
                )
                GROUP BY canonical_ts_code
            )
            SELECT
                count(*),
                coalesce(sum(terminal.position_market_value), 0.0),
                coalesce(sum(terminal.actual_weight), 0.0)
            FROM terminal
            LEFT JOIN listing USING (canonical_ts_code)
            WHERE listing.delist_date IS NOT NULL
              AND listing.delist_date <= DATE '2025-12-31'
            """
        ).fetchone()
        collector.add(
            "P5-022",
            "精确退市日持仓",
            terminal_delisted[0] == 0,
            terminal_delisted,
            "count=0, value=0, weight=0",
            (
                "processed delist_date 已明确且不晚于期末的持仓必须为0；"
                "仅有list_status=D但缺少精确日期的记录由P5-023单独披露。"
            ),
        )

        terminal_stale = connection.execute(
            f"""
            WITH terminal AS (
                SELECT *
                FROM oos_actual_holdings
                WHERE trade_date = (
                    SELECT max(trade_date)
                    FROM oos_actual_holdings
                )
                  AND valuation_price_source
                        <> 'CURRENT_ADJUSTED_CLOSE'
            ),
            status AS (
                SELECT
                    ts_code,
                    any_value(list_status) AS list_status
                FROM read_parquet(
                    '{sql_path(config["inputs"]["stock_basic"])}'
                )
                GROUP BY ts_code
            )
            SELECT
                count(*) FILTER (
                    WHERE terminal.actual_weight > 1e-8
                ),
                count(DISTINCT terminal.ts_code) FILTER (
                    WHERE terminal.actual_weight > 1e-8
                ),
                string_agg(
                    DISTINCT terminal.ts_code, ','
                    ORDER BY terminal.ts_code
                ) FILTER (
                    WHERE terminal.is_baseline
                      AND terminal.actual_weight > 1e-8
                ),
                coalesce(sum(terminal.position_market_value) FILTER (
                    WHERE terminal.is_baseline
                      AND terminal.actual_weight > 1e-8
                ), 0.0),
                coalesce(sum(terminal.actual_weight) FILTER (
                    WHERE terminal.is_baseline
                      AND terminal.actual_weight > 1e-8
                ), 0.0),
                count(*) FILTER (
                    WHERE terminal.actual_weight > 1e-8
                      AND status.list_status <> 'D'
                ),
                coalesce(max(terminal.stale_calendar_days), 0)
            FROM terminal
            LEFT JOIN status USING (ts_code)
            """
        ).fetchone()
        terminal_disclosure = {
            "material_rows_all_scenarios": int(terminal_stale[0]),
            "material_codes": int(terminal_stale[1]),
            "baseline_codes": terminal_stale[2] or "",
            "baseline_value": float(terminal_stale[3]),
            "baseline_weight": float(terminal_stale[4]),
            "non_delisted_status_rows": int(terminal_stale[5]),
            "maximum_stale_calendar_days": int(terminal_stale[6]),
        }
        collector.add(
            "P5-023",
            "期末退市估值",
            True,
            terminal_disclosure,
            "冻结结果保留并作重大限制披露",
            (
                "三只实质权重证券在静态表中为list_status=D，但缺少冻结"
                "回测可用的精确退市日和回收口径；P5保留最后价格估值，"
                "不因OOS结果补写规则或重跑。"
            ),
            warning=terminal_stale[0] > 0,
        )

    schedule = pd.read_csv(
        absolute(outputs["full_rebalance_schedule"]),
        parse_dates=["signal_date", "scheduled_trade_date"],
    )
    oos_schedule = pd.read_csv(
        absolute(outputs["oos_rebalance_schedule"]),
        parse_dates=["signal_date", "scheduled_trade_date"],
    )
    schedule_pass = (
        len(schedule) == 120
        and schedule["scheduled_trade_date"].notna().sum() == 119
        and len(oos_schedule) == 49
        and oos_schedule["scheduled_trade_date"].notna().sum() == 48
        and schedule.iloc[-1]["schedule_status"]
        == "OUT_OF_SCOPE_NOT_EXECUTED"
    )
    collector.add(
        "P5-024",
        "调仓日历",
        schedule_pass,
        (
            f"full={len(schedule)}/"
            f"{schedule['scheduled_trade_date'].notna().sum()}, "
            f"oos={len(oos_schedule)}/"
            f"{oos_schedule['scheduled_trade_date'].notna().sum()}, "
            f"last={schedule.iloc[-1]['schedule_status']}"
        ),
        "full=120/119; OOS boundary=49/48; final out of scope",
        "2021-12信号应在OOS首日执行，2025-12信号不得跨到2026。",
    )

    ic = pd.read_csv(absolute(outputs["oos_ic_summary"]))
    monthly_ic = pd.read_csv(
        absolute(outputs["oos_monthly_rank_ic"]),
        parse_dates=["signal_date"],
    )
    december = monthly_ic.loc[
        monthly_ic["signal_date"] == pd.Timestamp("2025-12-31")
    ]
    evaluable = monthly_ic.loc[monthly_ic["rank_ic"].notna()]
    expected_factors = {"bm_proxy", "momentum_12_1", "lowvol_60"}
    ic_pass = (
        set(ic["factor"]) == expected_factors
        and set(ic["months"]) == {47}
        and set(monthly_ic["factor"]) == expected_factors
        and len(december) == 3
        and bool(december["rank_ic"].isna().all())
        and bool((december["observations"] == 0).all())
        and evaluable["signal_date"].max()
        <= pd.Timestamp("2025-11-30")
    )
    collector.add(
        "P5-025",
        "OOS因子统计",
        ic_pass,
        (
            f"factors={sorted(ic['factor'])}, "
            f"months={sorted(ic['months'].unique())}, "
            f"december_empty={len(december)}"
        ),
        "3 factors; 47 evaluable months; 3 empty 2025-12 rows",
        "不得读取项目截止日之后的收益标签。",
    )

    oos = pd.read_csv(absolute(outputs["oos_performance"]))
    annual = pd.read_csv(absolute(outputs["annual_performance"]))
    performance_pass = (
        len(oos) == 3
        and set(oos["cost_scenario"])
        == {"STRESS_5BPS", "BASE_10BPS", "STRESS_20BPS"}
        and set(oos["period"]) == {"FINAL_OOS_2022_2025"}
        and set(oos["start_date"]) == {"2022-01-04"}
        and set(oos["end_date"]) == {"2025-12-31"}
        and set(
            annual.loc[
                annual["year"].between(2022, 2025), "year"
            ]
        )
        == {2022, 2023, 2024, 2025}
    )
    collector.add(
        "P5-026",
        "OOS绩效",
        performance_pass,
        (
            f"rows={len(oos)}, start={sorted(oos['start_date'].unique())}, "
            f"end={sorted(oos['end_date'].unique())}"
        ),
        "3 scenarios; 2022-01-04..2025-12-31; four annual rows",
        "最终OOS绩效必须独立分段并覆盖四个完整市场年份。",
    )

    cost = oos.set_index("cost_scenario")[
        ["strategy_total_return", "total_trading_cost"]
    ]
    monotonic = (
        cost.loc["STRESS_5BPS", "strategy_total_return"]
        > cost.loc["BASE_10BPS", "strategy_total_return"]
        > cost.loc["STRESS_20BPS", "strategy_total_return"]
        and cost.loc["STRESS_5BPS", "total_trading_cost"]
        < cost.loc["BASE_10BPS", "total_trading_cost"]
        < cost.loc["STRESS_20BPS", "total_trading_cost"]
    )
    collector.add(
        "P5-027",
        "成本压力",
        monotonic,
        cost.to_dict(orient="index"),
        "return decreases and cost increases with bps",
        "冻结成本压力方向必须符合机械成本关系。",
    )

    scope = manifest.get("scope_guards", {})
    scope_pass = (
        scope.get("final_oos_authorized_before_run") is True
        and scope.get("final_oos_results_computed") is True
        and scope.get(
            "final_oos_results_reported_after_authorization"
        )
        is True
        and scope.get("oos_previewed_before_authorization") is False
        and scope.get("parameters_retuned_after_validation") is False
        and scope.get("frozen_config_modified") is False
        and scope.get("p6_code_generated") is False
        and scope.get("p6_run") is False
    )
    collector.add(
        "P5-028",
        "范围声明",
        scope_pass,
        scope,
        "authorized OOS only; no retuning; frozen unchanged; P6 false",
        "P5清单必须准确声明授权、冻结和阶段边界。",
    )

    output_mismatches: list[str] = []
    for key, expected_hash in manifest.get("output_sha256", {}).items():
        path = absolute(outputs[key])
        if not path.is_file() or _sha256(path) != expected_hash:
            output_mismatches.append(key)
    collector.add(
        "P5-029",
        "输出完整性",
        not output_mismatches,
        output_mismatches,
        [],
        "构建清单内所有OOS输出哈希必须保持一致。",
    )

    p6_candidates = (
        Path("src/a_share_p6"),
        Path("scripts/build_p6_robustness.py"),
        Path("results/p6_robustness"),
    )
    existing_p6 = [
        path.as_posix()
        for path in p6_candidates
        if absolute(path.as_posix()).exists()
    ]
    collector.add(
        "P5-030",
        "P6闸门",
        not existing_p6,
        existing_p6,
        [],
        "P5不得提前生成稳健性变体或P6实现。",
    )

    collector.add(
        "P5-031",
        "PB历史口径",
        True,
        "NEEDS_MANUAL_CONFIRMATION",
        "已披露，不阻碍冻结OOS",
        PB_DISCLOSURE,
        warning=True,
    )

    audit = collector.frame()
    fail_count = int((audit["status"] == "FAIL").sum())
    warn_count = int((audit["status"] == "WARN").sum())
    pass_count = int((audit["status"] == "PASS").sum())
    overall = (
        "P5_ACCEPTED_FINAL_OOS_WITH_DISCLOSED_LIMITATIONS"
        if fail_count == 0
        else "P5_AUDIT_FAILED"
    )
    report = _report(
        audit,
        overall,
        oos,
        annual,
        ic,
        reproduction,
        terminal_disclosure,
        gate["frozen_hash"],
        audited_at,
    )
    _write_csv_atomic(audit, outputs["p5_audit_summary"])
    _write_text_atomic(report, outputs["p5_audit_report"])
    manifest["status"] = overall
    manifest["audit"] = {
        "audited_at_utc": audited_at,
        "pass_count": pass_count,
        "warn_count": warn_count,
        "fail_count": fail_count,
        "summary_path": outputs["p5_audit_summary"],
        "report_path": outputs["p5_audit_report"],
    }
    manifest["material_oos_limitations"] = {
        "terminal_delisted_status_last_price_valuation": {
            **terminal_disclosure,
            "status": "WARN",
            "treatment": (
                "PRESERVED_FROZEN_LAST_AVAILABLE_ADJUSTED_CLOSE"
            ),
            "oos_rerun_performed": False,
        },
        "historical_pb_revision_policy": {
            "status": "NEEDS_MANUAL_CONFIRMATION",
            "treatment": "DISCLOSED_NO_RETUNING",
        },
    }
    manifest["output_sha256"]["p5_audit_summary"] = _sha256(
        absolute(outputs["p5_audit_summary"])
    )
    manifest["output_sha256"]["p5_audit_report"] = _sha256(
        absolute(outputs["p5_audit_report"])
    )
    _write_json_atomic(manifest, outputs["p5_run_manifest"])
    print(
        f"[P5 AUDIT] {overall}: "
        f"{pass_count} PASS / {warn_count} WARN / "
        f"{fail_count} FAIL",
        flush=True,
    )
    if fail_count:
        failed = audit.loc[
            audit["status"] == "FAIL",
            ["check_id", "observed", "expected"],
        ]
        raise RuntimeError(
            "P5审计失败："
            f"{failed.to_dict(orient='records')}"
        )
    return audit
