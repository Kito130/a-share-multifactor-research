from __future__ import annotations

import json
import math
import os
import platform
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow
import yaml

from a_share_p2.research import _build_statistics, _factor_panel_query
from a_share_p3.build import (
    _annual_performance,
    _build_composite_signals,
    _build_schedule,
    _failure_summary,
    _load_corporate_actions,
    _load_market_inputs,
    _load_stamp_policy,
    _performance_summary,
    _prepare_corporate_action_references,
    _simulate_scenario,
    _stale_position_summary,
)
from a_share_p4.build import (
    PB_DISCLOSURE,
    _copy_query_atomic,
    _input_snapshot,
    _sha256,
    _write_csv_atomic,
    _write_json_atomic,
    _write_parquet_atomic,
    _write_text_atomic,
)

from .config import absolute, load_config


BUILDER_VERSION = "p5.1"


def _log(message: str) -> None:
    print(f"[P5] {message}", flush=True)


def _read_yaml(relative_path: str) -> dict[str, Any]:
    with absolute(relative_path).open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle)
    return payload


def _read_json(relative_path: str) -> dict[str, Any]:
    return json.loads(
        absolute(relative_path).read_text(encoding="utf-8")
    )


def _recorded_freeze_hash(relative_path: str) -> str:
    text = absolute(relative_path).read_text(encoding="utf-8")
    match = re.search(r"sha256=([0-9a-f]{64})", text)
    if match is None:
        raise RuntimeError("冻结配置SHA-256记录格式无效")
    return match.group(1)


def validate_p5_preflight(
    *,
    require_fresh_run: bool,
) -> dict[str, Any]:
    control = load_config()
    inputs = control["inputs"]
    frozen = _read_yaml(inputs["frozen_config"])
    current_hash = _sha256(absolute(inputs["frozen_config"]))
    expected_hash = str(
        control["project"]["expected_frozen_sha256"]
    )
    recorded_hash = _recorded_freeze_hash(
        inputs["frozen_config_sha256"]
    )
    p4_manifest = _read_json(inputs["p4_manifest"])

    hashes = {
        "expected": expected_hash,
        "recorded": recorded_hash,
        "current": current_hash,
        "p4_manifest": p4_manifest.get("freeze", {}).get(
            "frozen_config_sha256"
        ),
    }
    if len(set(hashes.values())) != 1:
        raise RuntimeError(f"P5冻结配置哈希闸门失败：{hashes}")
    if frozen.get("status") != "FROZEN_AFTER_VALIDATION":
        raise RuntimeError("P5只能读取P4正式冻结配置")
    rules = frozen.get("post_freeze_rules", {})
    if (
        rules.get("allow_parameter_changes") is not False
        or rules.get("allow_validation_retuning") is not False
        or rules.get(
            "final_oos_requires_new_explicit_authorization"
        )
        is not True
        or rules.get("final_oos_has_been_run") is not False
        or rules.get("p5_implementation_generated") is not False
    ):
        raise RuntimeError("P4冻结后的P5人工闸门无效")
    if not str(p4_manifest.get("status", "")).startswith(
        "P4_ACCEPTED_AND_FROZEN"
    ):
        raise RuntimeError("P4未正式验收冻结，不能运行P5")
    if p4_manifest.get("audit", {}).get("fail_count") != 0:
        raise RuntimeError("P4仍有审计FAIL，不能运行P5")
    p4_scope = p4_manifest.get("scope_guards", {})
    forbidden_p4_scope = (
        "oos_rows_written",
        "oos_results_computed",
        "oos_results_previewed",
        "p5_code_generated",
        "p5_run",
        "p6_run",
    )
    if any(bool(p4_scope.get(key)) for key in forbidden_p4_scope):
        raise RuntimeError(
            "P4清单显示最终OOS或后续阶段已被触碰"
        )
    if (
        control["project"]["authorization_reference"]
        != "FROZEN_OOS_RELEASE_GATE"
        or control["project"]["one_shot_final_oos"] is not True
    ):
        raise RuntimeError("P5 缺少冻结 OOS 执行许可记录")

    source_hashes = frozen.get("source_sha256", {})
    if not source_hashes:
        raise RuntimeError("冻结配置未记录受保护输入哈希")
    source_mismatches = [
        path
        for path, expected in source_hashes.items()
        if not absolute(path).is_file()
        or _sha256(absolute(path)) != expected
    ]
    if source_mismatches:
        raise RuntimeError(
            f"冻结输入在P5前已变化：{source_mismatches}"
        )

    p2_config = _read_yaml(inputs["p2_config"])
    p3_config = _read_yaml(inputs["p3_config"])
    for section in ("factors", "universe"):
        if frozen[section] != p2_config[section]:
            raise RuntimeError(
                f"冻结配置与受保护P2配置不一致：{section}"
            )
    for section in (
        "portfolio",
        "composite",
        "cost_scenarios",
        "valuation",
        "corporate_actions",
        "metrics",
    ):
        if frozen[section] != p3_config[section]:
            raise RuntimeError(
                f"冻结配置与受保护P3配置不一致：{section}"
            )

    missing = [
        path
        for path in control["protected_p5_inputs"]
        if not absolute(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"P5受保护输入缺失：{missing}")
    if require_fresh_run:
        existing = [
            control["outputs"][key]
            for key in ("run_intent", "p5_run_manifest")
            if absolute(control["outputs"][key]).exists()
        ]
        if existing:
            raise RuntimeError(
                "P5是一次性最终OOS，检测到既有运行记录，拒绝覆盖："
                f"{existing}"
            )
    return {
        "control": control,
        "frozen": frozen,
        "p2_config": p2_config,
        "p3_config": p3_config,
        "p4_manifest": p4_manifest,
        "frozen_hash": current_hash,
    }


def _runtime_config(gate: dict[str, Any]) -> dict[str, Any]:
    control = gate["control"]
    frozen = gate["frozen"]
    sample = frozen["sample"]
    return {
        "project": {
            "warmup_start": sample["warmup"][0],
            "research_start": sample["research"][0],
            "research_end": sample["final_oos"][1],
            "validation_start": sample["validation"][0],
            "validation_end": sample["validation"][1],
            "oos_start": sample["final_oos"][0],
            "oos_end": sample["final_oos"][1],
        },
        "inputs": {
            **control["inputs"],
            "p2_single_factor_panel": control["outputs"][
                "factor_panel"
            ],
        },
        "outputs": control["outputs"],
        "factors": frozen["factors"],
        "universe": frozen["universe"],
        "statistics": gate["p2_config"]["statistics"],
        "portfolio": frozen["portfolio"],
        "composite": frozen["composite"],
        "cost_scenarios": frozen["cost_scenarios"],
        "valuation": frozen["valuation"],
        "corporate_actions": frozen["corporate_actions"],
        "metrics": frozen["metrics"],
    }


def _maximum_error(values: tuple[Any, ...]) -> float:
    return max(float(value or 0.0) for value in values)


def _p4_reproduction(
    control: dict[str, Any],
    frozen: dict[str, Any],
) -> pd.DataFrame:
    paths = {
        key: absolute(value).as_posix().replace("'", "''")
        for key, value in {
            "old_factor": control["inputs"]["p4_factor_panel"],
            "new_factor": control["outputs"]["factor_panel"],
            "old_signal": control["inputs"][
                "p4_composite_signals"
            ],
            "new_signal": control["outputs"]["composite_signals"],
            "old_target": control["inputs"]["p4_target_holdings"],
            "new_target": control["outputs"]["target_holdings"],
            "old_daily": control["inputs"]["p4_daily_portfolio"],
            "new_daily": control["outputs"]["daily_portfolio"],
            "old_holdings": control["inputs"]["p4_actual_holdings"],
            "new_holdings": control["outputs"]["actual_holdings"],
            "old_orders": control["inputs"]["p4_orders"],
            "new_orders": control["outputs"]["orders"],
            "old_actions": control["inputs"][
                "p4_corporate_action_events"
            ],
            "new_actions": control["outputs"][
                "corporate_action_events"
            ],
        }.items()
    }
    end = frozen["sample"]["validation"][1]
    rows: list[dict[str, Any]] = []

    def add(
        check: str,
        result: tuple[Any, ...],
        tolerance: float,
    ) -> None:
        numeric_error = _maximum_error(result[4:])
        passed = (
            result[0] == result[1]
            and result[2] == 0
            and result[3] == 0
            and numeric_error <= tolerance
        )
        rows.append(
            {
                "check": check,
                "old_rows": int(result[0]),
                "new_rows": int(result[1]),
                "missing_or_extra_rows": int(result[2]),
                "categorical_mismatches": int(result[3]),
                "maximum_numeric_error": numeric_error,
                "tolerance": tolerance,
                "status": "PASS" if passed else "FAIL",
            }
        )

    with duckdb.connect() as connection:
        factor = connection.execute(
            f"""
            WITH old AS (
                SELECT * FROM read_parquet('{paths["old_factor"]}')
            ),
            new AS (
                SELECT * FROM read_parquet('{paths["new_factor"]}')
                WHERE signal_date <= DATE '{end}'
            )
            SELECT
                (SELECT count(*) FROM old),
                (SELECT count(*) FROM new),
                count(*) FILTER (
                    WHERE old.ts_code IS NULL OR new.ts_code IS NULL
                ),
                count(*) FILTER (
                    WHERE old.ts_code IS DISTINCT FROM new.ts_code
                       OR old.universe_eligible
                          IS DISTINCT FROM new.universe_eligible
                ),
                max(abs(old.bm_proxy - new.bm_proxy)),
                max(abs(old.momentum_12_1 - new.momentum_12_1)),
                max(abs(old.lowvol_60 - new.lowvol_60)),
                max(abs(old.bm_proxy_z - new.bm_proxy_z)),
                max(abs(
                    old.momentum_12_1_z - new.momentum_12_1_z
                )),
                max(abs(old.lowvol_60_z - new.lowvol_60_z))
            FROM old
            FULL OUTER JOIN new
              USING (canonical_ts_code, signal_date)
            """
        ).fetchone()
        add("P4_FACTOR_PREFIX", factor, 1e-12)

        for label, old_key, new_key in (
            (
                "P4_COMPOSITE_PREFIX",
                "old_signal",
                "new_signal",
            ),
            ("P4_TARGET_PREFIX", "old_target", "new_target"),
        ):
            result = connection.execute(
                f"""
                WITH old AS (
                    SELECT * FROM read_parquet('{paths[old_key]}')
                ),
                new AS (
                    SELECT * FROM read_parquet('{paths[new_key]}')
                    WHERE signal_date <= DATE '{end}'
                )
                SELECT
                    (SELECT count(*) FROM old),
                    (SELECT count(*) FROM new),
                    count(*) FILTER (
                        WHERE old.canonical_ts_code IS NULL
                           OR new.canonical_ts_code IS NULL
                    ),
                    count(*) FILTER (
                        WHERE old.ts_code IS DISTINCT FROM new.ts_code
                           OR old.is_selected
                              IS DISTINCT FROM new.is_selected
                           OR old.selection_rank
                              IS DISTINCT FROM new.selection_rank
                    ),
                    max(abs(
                        old.composite_score - new.composite_score
                    )),
                    max(abs(
                        old.target_weight - new.target_weight
                    ))
                FROM old
                FULL OUTER JOIN new
                  USING (canonical_ts_code, signal_date)
                """
            ).fetchone()
            add(label, result, 1e-12)

        daily = connection.execute(
            f"""
            WITH old AS (
                SELECT * FROM read_parquet('{paths["old_daily"]}')
            ),
            new AS (
                SELECT * FROM read_parquet('{paths["new_daily"]}')
                WHERE trade_date <= DATE '{end}'
            )
            SELECT
                (SELECT count(*) FROM old),
                (SELECT count(*) FROM new),
                count(*) FILTER (
                    WHERE old.cost_scenario IS NULL
                       OR new.cost_scenario IS NULL
                ),
                count(*) FILTER (
                    WHERE old.signal_date_executed
                          IS DISTINCT FROM new.signal_date_executed
                       OR old.holding_count
                          IS DISTINCT FROM new.holding_count
                ),
                max(abs(old.total_nav_cny - new.total_nav_cny)),
                max(abs(old.cash_cny - new.cash_cny)),
                max(abs(
                    old.strategy_daily_return
                    - new.strategy_daily_return
                ))
            FROM old
            FULL OUTER JOIN new
              USING (cost_scenario, trade_date)
            """
        ).fetchone()
        add("P4_DAILY_PREFIX", daily, 1e-6)

        holdings = connection.execute(
            f"""
            WITH old AS (
                SELECT * FROM read_parquet('{paths["old_holdings"]}')
            ),
            new AS (
                SELECT * FROM read_parquet('{paths["new_holdings"]}')
                WHERE trade_date <= DATE '{end}'
            )
            SELECT
                (SELECT count(*) FROM old),
                (SELECT count(*) FROM new),
                count(*) FILTER (
                    WHERE old.cost_scenario IS NULL
                       OR new.cost_scenario IS NULL
                ),
                count(*) FILTER (
                    WHERE old.ts_code IS DISTINCT FROM new.ts_code
                       OR old.valuation_price_source
                          IS DISTINCT FROM new.valuation_price_source
                       OR old.position_origin_activity_type
                          IS DISTINCT FROM
                             new.position_origin_activity_type
                       OR old.is_current_target
                          IS DISTINCT FROM new.is_current_target
                ),
                max(abs(
                    old.adjusted_units - new.adjusted_units
                )),
                max(abs(
                    old.position_market_value
                    - new.position_market_value
                )),
                max(abs(old.actual_weight - new.actual_weight))
            FROM old
            FULL OUTER JOIN new
              USING (
                  cost_scenario, trade_date, canonical_ts_code
              )
            """
        ).fetchone()
        add("P4_HOLDINGS_PREFIX", holdings, 1e-6)

        orders = connection.execute(
            f"""
            WITH old AS (
                SELECT * FROM read_parquet('{paths["old_orders"]}')
            ),
            new AS (
                SELECT * FROM read_parquet('{paths["new_orders"]}')
                WHERE trade_date <= DATE '{end}'
            )
            SELECT
                (SELECT count(*) FROM old),
                (SELECT count(*) FROM new),
                count(*) FILTER (
                    WHERE old.order_id IS NULL OR new.order_id IS NULL
                ),
                count(*) FILTER (
                    WHERE old.order_status
                          IS DISTINCT FROM new.order_status
                       OR old.failure_reason
                          IS DISTINCT FROM new.failure_reason
                       OR old.execution_ts_code
                          IS DISTINCT FROM new.execution_ts_code
                ),
                max(abs(
                    old.executed_notional - new.executed_notional
                )),
                max(abs(
                    old.total_trading_cost
                    - new.total_trading_cost
                )),
                max(abs(
                    old.remaining_adjusted_units_after
                    - new.remaining_adjusted_units_after
                ))
            FROM old
            FULL OUTER JOIN new USING (order_id)
            """
        ).fetchone()
        add("P4_ORDERS_PREFIX", orders, 1e-6)

        actions = connection.execute(
            f"""
            WITH old AS (
                SELECT * FROM read_parquet('{paths["old_actions"]}')
            ),
            new AS (
                SELECT * FROM read_parquet('{paths["new_actions"]}')
                WHERE effective_date <= DATE '{end}'
            )
            SELECT
                (SELECT count(*) FROM old),
                (SELECT count(*) FROM new),
                count(*) FILTER (
                    WHERE old.event_id IS NULL OR new.event_id IS NULL
                ),
                count(*) FILTER (
                    WHERE old.activity_type
                          IS DISTINCT FROM new.activity_type
                       OR old.old_ts_code
                          IS DISTINCT FROM new.old_ts_code
                       OR old.successor_ts_code
                          IS DISTINCT FROM new.successor_ts_code
                ),
                max(abs(
                    old.successor_share_quantity_after
                    - new.successor_share_quantity_after
                )),
                max(abs(
                    old.portfolio_value_difference_cny
                    - new.portfolio_value_difference_cny
                ))
            FROM old
            FULL OUTER JOIN new USING (event_id)
            """
        ).fetchone()
        add("P4_CORPORATE_ACTION_PREFIX", actions, 1e-8)
    return pd.DataFrame(rows)


def _period_performance(
    daily: pd.DataFrame,
    orders: pd.DataFrame,
    runtime: dict[str, Any],
    frozen: dict[str, Any],
) -> pd.DataFrame:
    periods = (
        ("RESEARCH_2016_2019", *frozen["sample"]["research"]),
        ("VALIDATION_2020_2021", *frozen["sample"]["validation"]),
        ("FINAL_OOS_2022_2025", *frozen["sample"]["final_oos"]),
    )
    trading_days = int(runtime["metrics"]["trading_days_per_year"])
    rows: list[dict[str, Any]] = []
    for scenario, scenario_daily in daily.groupby(
        "cost_scenario", sort=True
    ):
        scenario_daily = scenario_daily.sort_values("trade_date")
        for period, start_text, end_text in periods:
            start = pd.Timestamp(start_text)
            end = pd.Timestamp(end_text)
            group = scenario_daily.loc[
                scenario_daily["trade_date"].between(start, end)
            ].copy()
            if group.empty:
                raise RuntimeError(f"P5期间没有组合记录：{period}")
            returns = group["strategy_daily_return"]
            benchmark_returns = group["benchmark_daily_return"]
            excess = returns - benchmark_returns
            strategy_path = (1.0 + returns).cumprod()
            benchmark_path = (1.0 + benchmark_returns).cumprod()
            strategy_total = float(strategy_path.iloc[-1] - 1.0)
            benchmark_total = float(benchmark_path.iloc[-1] - 1.0)
            years = len(group) / trading_days
            period_orders = orders.loc[
                (orders["cost_scenario"] == scenario)
                & orders["trade_date"].between(start, end)
            ]
            strategy_std = float(returns.std(ddof=1))
            excess_std = float(excess.std(ddof=1))
            rows.append(
                {
                    "cost_scenario": scenario,
                    "is_baseline": bool(group.iloc[0]["is_baseline"]),
                    "period": period,
                    "start_date": group["trade_date"].min(),
                    "end_date": group["trade_date"].max(),
                    "trading_days": len(group),
                    "strategy_total_return": strategy_total,
                    "strategy_annualized_return": (
                        (1.0 + strategy_total) ** (1.0 / years)
                        - 1.0
                    ),
                    "strategy_annualized_volatility": (
                        strategy_std * math.sqrt(trading_days)
                    ),
                    "strategy_sharpe_zero_rf": (
                        float(returns.mean())
                        / strategy_std
                        * math.sqrt(trading_days)
                        if strategy_std > 0
                        else math.nan
                    ),
                    "strategy_max_drawdown_within_period": float(
                        (
                            strategy_path / strategy_path.cummax()
                            - 1.0
                        ).min()
                    ),
                    "benchmark_total_return": benchmark_total,
                    "benchmark_annualized_return": (
                        (1.0 + benchmark_total) ** (1.0 / years)
                        - 1.0
                    ),
                    "benchmark_annualized_volatility": float(
                        benchmark_returns.std(ddof=1)
                        * math.sqrt(trading_days)
                    ),
                    "benchmark_max_drawdown_within_period": float(
                        (
                            benchmark_path / benchmark_path.cummax()
                            - 1.0
                        ).min()
                    ),
                    "annualized_return_difference": (
                        (1.0 + strategy_total) ** (1.0 / years)
                        - (1.0 + benchmark_total) ** (1.0 / years)
                    ),
                    "terminal_relative_nav": float(
                        strategy_path.iloc[-1]
                        / benchmark_path.iloc[-1]
                    ),
                    "tracking_error_annualized": float(
                        excess_std * math.sqrt(trading_days)
                    ),
                    "information_ratio": (
                        float(excess.mean())
                        / excess_std
                        * math.sqrt(trading_days)
                        if excess_std > 0
                        else math.nan
                    ),
                    "average_cash_weight": float(
                        group["cash_weight"].mean()
                    ),
                    "maximum_cash_weight": float(
                        group["cash_weight"].max()
                    ),
                    "maximum_stale_price_weight": float(
                        group["stale_price_weight"].max()
                    ),
                    "terminal_stale_price_weight": float(
                        group.iloc[-1]["stale_price_weight"]
                    ),
                    "terminal_stale_price_market_value": float(
                        group.iloc[-1]["stale_price_market_value"]
                    ),
                    "two_way_turnover": float(
                        group["turnover_ratio"].sum()
                    ),
                    "total_commission_slippage_cost": float(
                        period_orders[
                            "commission_slippage_cost"
                        ].sum()
                    ),
                    "total_stamp_duty_cost": float(
                        period_orders["stamp_duty_cost"].sum()
                    ),
                    "total_trading_cost": float(
                        period_orders["total_trading_cost"].sum()
                    ),
                    "failed_buy_orders": int(
                        (
                            (period_orders["side"] == "BUY")
                            & (
                                period_orders["order_status"]
                                == "FAILED"
                            )
                        ).sum()
                    ),
                    "failed_sell_orders": int(
                        (
                            (period_orders["side"] == "SELL")
                            & (
                                period_orders["order_status"]
                                == "FAILED"
                            )
                        ).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _result_report(
    frozen_hash: str,
    oos_performance: pd.DataFrame,
    annual: pd.DataFrame,
    ic_summary: pd.DataFrame,
    reproduction: pd.DataFrame,
) -> str:
    baseline = oos_performance.loc[
        oos_performance["is_baseline"].astype(bool)
    ].iloc[0]
    baseline_annual = annual.loc[
        annual["is_baseline"].astype(bool)
        & annual["year"].between(2022, 2025)
    ].sort_values("year")
    lines = [
        "# P5 最终 OOS 结果",
        "",
        "- 研究闸门：冻结配置后，最终 OOS 仅执行一次。",
        "- 最终 OOS：2022-01-01 至 2025-12-31。",
        "- 参数：只读 P4 冻结配置，未调参。",
        f"- 冻结配置 SHA-256：`{frozen_hash}`。",
        "",
        "## 基准成本情景",
        "",
        f"- 策略累计收益：{baseline['strategy_total_return']:.6%}",
        f"- 策略年化收益：{baseline['strategy_annualized_return']:.6%}",
        (
            "- 策略最大回撤："
            f"{baseline['strategy_max_drawdown_within_period']:.6%}"
        ),
        f"- 策略年化波动率：{baseline['strategy_annualized_volatility']:.6%}",
        f"- 零无风险利率夏普：{baseline['strategy_sharpe_zero_rf']:.6f}",
        f"- 中证全指累计收益：{baseline['benchmark_total_return']:.6%}",
        f"- 中证全指年化收益：{baseline['benchmark_annualized_return']:.6%}",
        (
            "- 年化收益差："
            f"{baseline['annualized_return_difference']:.6%}"
        ),
        f"- 信息比率：{baseline['information_ratio']:.6f}",
        f"- 总交易成本：{baseline['total_trading_cost']:,.2f} 元",
        f"- 买入失败：{int(baseline['failed_buy_orders'])} 单",
        f"- 卖出失败：{int(baseline['failed_sell_orders'])} 单",
        "",
        "## 年度结果",
        "",
        "| 年份 | 策略 | 中证全指 | 差值 |",
        "|---:|---:|---:|---:|",
    ]
    for row in baseline_annual.itertuples(index=False):
        lines.append(
            f"| {row.year} | {row.strategy_return:.2%} | "
            f"{row.benchmark_return:.2%} | "
            f"{row.return_difference:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 成本情景",
            "",
            "| 情景 | 累计收益 | 年化收益 | 最大回撤 | 总成本 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    order = {
        "STRESS_5BPS": 5,
        "BASE_10BPS": 10,
        "STRESS_20BPS": 20,
    }
    for row in oos_performance.sort_values(
        "cost_scenario",
        key=lambda values: values.map(order),
    ).itertuples(index=False):
        lines.append(
            f"| {row.cost_scenario} | "
            f"{row.strategy_total_return:.2%} | "
            f"{row.strategy_annualized_return:.2%} | "
            f"{row.strategy_max_drawdown_within_period:.2%} | "
            f"{row.total_trading_cost:,.2f} |"
        )
    lines.extend(
        [
            "",
            "## 最终 OOS 单因子 Rank IC",
            "",
            "| 因子 | 可评价月份 | 平均Rank IC | 年化ICIR |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in ic_summary.itertuples(index=False):
        lines.append(
            f"| {row.factor} | {int(row.months)} | "
            f"{row.mean_rank_ic:.6f} | "
            f"{row.rank_icir_annualized:.6f} |"
        )
    lines.extend(
        [
            "",
            "2025-12 信号没有项目范围外的下一月标签，故只评价47个月。",
            "",
            "## P4 前缀复现",
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
            "## 限制",
            "",
            f"- {PB_DISCLOSURE}",
            "- 本文件是冻结策略的最终OOS结果，不用于再次调参。",
            "- 稳健性变体、正式研究报告和简历证据属于P6，本阶段未运行。",
            "",
        ]
    )
    return "\n".join(lines)


def _build_p5_after_intent(
    gate: dict[str, Any],
    before: dict[str, dict[str, Any]],
    intent: dict[str, Any],
) -> dict[str, Any]:
    started_at = datetime.fromisoformat(intent["started_at_utc"])
    control = gate["control"]
    frozen = gate["frozen"]
    runtime = _runtime_config(gate)
    outputs = control["outputs"]
    oos_start = pd.Timestamp(frozen["sample"]["final_oos"][0])
    oos_end = pd.Timestamp(frozen["sample"]["final_oos"][1])

    _log("按冻结P2方法构建2016—2025因子面板")
    with duckdb.connect() as connection:
        connection.execute("SET preserve_insertion_order = false")
        connection.execute("SET threads = 4")
        _copy_query_atomic(
            connection,
            _factor_panel_query(runtime),
            outputs["factor_panel"],
        )
    panel = pd.read_parquet(absolute(outputs["factor_panel"]))
    panel["signal_date"] = pd.to_datetime(panel["signal_date"])
    panel["next_signal_date"] = pd.to_datetime(
        panel["next_signal_date"]
    )
    if panel["signal_date"].max() > oos_end:
        raise RuntimeError("P5因子面板越过最终研究截止日")
    oos_panel = panel.loc[
        panel["signal_date"].between(oos_start, oos_end)
    ].copy()
    _write_parquet_atomic(oos_panel, outputs["oos_factor_panel"])

    _log("计算最终OOS单因子统计")
    oos_statistics = _build_statistics(oos_panel, runtime)
    for key, frame in oos_statistics.items():
        _write_csv_atomic(frame, outputs[f"oos_{key}"])

    _log("构建冻结复合信号、目标持仓和调仓日历")
    signals, targets = _build_composite_signals(runtime)
    benchmark = pd.read_parquet(
        absolute(control["inputs"]["benchmark_daily"])
    )
    benchmark["trade_date"] = pd.to_datetime(benchmark["trade_date"])
    benchmark = benchmark.loc[
        benchmark["trade_date"].between(
            signals["signal_date"].min(), oos_end
        )
    ].sort_values("trade_date")
    if set(benchmark["benchmark_code"]) != {"000985.CSI"}:
        raise RuntimeError("P5基准不是中证全指000985.CSI")
    schedule = _build_schedule(signals, benchmark, oos_end)
    targets = targets.merge(
        schedule,
        on="signal_date",
        how="left",
        validate="many_to_one",
    )
    _write_parquet_atomic(signals, outputs["composite_signals"])
    _write_parquet_atomic(targets, outputs["target_holdings"])
    oos_signals = signals.loc[
        signals["signal_date"].between(oos_start, oos_end)
    ].copy()
    oos_targets = targets.loc[
        targets["signal_date"].between(oos_start, oos_end)
    ].copy()
    _write_parquet_atomic(
        oos_signals, outputs["oos_composite_signals"]
    )
    _write_parquet_atomic(
        oos_targets, outputs["oos_target_holdings"]
    )
    _write_csv_atomic(schedule, outputs["full_rebalance_schedule"])
    oos_schedule = schedule.loc[
        schedule["signal_date"].between(oos_start, oos_end)
        | schedule["scheduled_trade_date"].between(
            oos_start, oos_end
        )
    ].copy()
    _write_csv_atomic(
        oos_schedule, outputs["oos_rebalance_schedule"]
    )

    corporate_actions = _load_corporate_actions(runtime)
    stamp_policy = _load_stamp_policy(runtime)
    prices, executions, benchmark = _load_market_inputs(
        runtime,
        targets,
        schedule,
        benchmark,
        corporate_actions,
    )
    action_references = _prepare_corporate_action_references(
        corporate_actions, prices
    )

    daily_frames: list[pd.DataFrame] = []
    holding_frames: list[pd.DataFrame] = []
    order_frames: list[pd.DataFrame] = []
    rebalance_frames: list[pd.DataFrame] = []
    action_frames: list[pd.DataFrame] = []
    for scenario in frozen["cost_scenarios"]:
        _log(f"运行冻结成本情景 {scenario['scenario']}")
        daily, holdings, orders, rebalances, actions = (
            _simulate_scenario(
                runtime,
                scenario,
                targets,
                schedule,
                benchmark,
                prices,
                executions,
                stamp_policy,
                action_references,
            )
        )
        daily_frames.append(daily)
        holding_frames.append(holdings)
        order_frames.append(orders)
        rebalance_frames.append(rebalances)
        action_frames.append(actions)

    daily = pd.concat(daily_frames, ignore_index=True)
    holdings = pd.concat(holding_frames, ignore_index=True)
    orders = pd.concat(order_frames, ignore_index=True)
    rebalances = pd.concat(rebalance_frames, ignore_index=True)
    actions = pd.concat(action_frames, ignore_index=True)
    failed = orders.loc[orders["order_status"] == "FAILED"].copy()
    cash = daily[
        [
            "cost_scenario",
            "is_baseline",
            "trade_date",
            "signal_date_executed",
            "cash_cny",
            "cash_weight",
            "executed_buy_notional",
            "executed_sell_notional",
            "commission_slippage_cost",
            "stamp_duty_cost",
            "total_trading_cost",
            "failed_order_count",
            "partial_order_count",
            "corporate_action_count",
            "corporate_action_value_difference_cny",
        ]
    ].copy()
    for key, frame in (
        ("daily_portfolio", daily),
        ("actual_holdings", holdings),
        ("orders", orders),
        ("failed_orders", failed),
        ("cash_ledger", cash),
        ("corporate_action_events", actions),
    ):
        _write_parquet_atomic(frame, outputs[key])

    oos_frames = {
        "oos_daily_portfolio": daily.loc[
            daily["trade_date"].between(oos_start, oos_end)
        ].copy(),
        "oos_actual_holdings": holdings.loc[
            holdings["trade_date"].between(oos_start, oos_end)
        ].copy(),
        "oos_orders": orders.loc[
            orders["trade_date"].between(oos_start, oos_end)
        ].copy(),
        "oos_failed_orders": failed.loc[
            failed["trade_date"].between(oos_start, oos_end)
        ].copy(),
        "oos_cash_ledger": cash.loc[
            cash["trade_date"].between(oos_start, oos_end)
        ].copy(),
        "oos_corporate_action_events": actions.loc[
            actions["effective_date"].between(oos_start, oos_end)
        ].copy(),
    }
    for key, frame in oos_frames.items():
        _write_parquet_atomic(frame, outputs[key])
    oos_rebalances = rebalances.loc[
        rebalances["trade_date"].between(oos_start, oos_end)
    ].copy()
    _write_csv_atomic(
        oos_rebalances, outputs["oos_rebalance_summary"]
    )

    _log("验证P4前缀精确复现")
    reproduction = _p4_reproduction(control, frozen)
    _write_csv_atomic(reproduction, outputs["p4_reproduction"])
    if not bool((reproduction["status"] == "PASS").all()):
        raise RuntimeError(
            "P5未精确复现P4冻结前缀："
            f"{reproduction.to_dict(orient='records')}"
        )

    performance_full = _performance_summary(
        daily, orders, runtime
    )
    annual = _annual_performance(daily, orders, runtime)
    period = _period_performance(daily, orders, runtime, frozen)
    oos_performance = period.loc[
        period["period"] == "FINAL_OOS_2022_2025"
    ].copy()
    failure_summary = _failure_summary(oos_frames["oos_orders"])
    stale_summary = _stale_position_summary(
        oos_frames["oos_actual_holdings"], oos_end
    )
    terminal_holdings = oos_frames["oos_actual_holdings"].loc[
        oos_frames["oos_actual_holdings"]["trade_date"] == oos_end
    ].copy()
    for key, frame in (
        ("performance_summary_full", performance_full),
        ("annual_performance", annual),
        ("period_performance", period),
        ("oos_performance", oos_performance),
        ("oos_cost_scenario_comparison", oos_performance),
        ("oos_failure_reason_summary", failure_summary),
        ("oos_stale_position_summary", stale_summary),
        ("oos_terminal_holdings", terminal_holdings),
    ):
        _write_csv_atomic(frame, outputs[key])

    _write_text_atomic(
        _result_report(
            gate["frozen_hash"],
            oos_performance,
            annual,
            oos_statistics["ic_summary"],
            reproduction,
        ),
        outputs["p5_result_report"],
    )

    _log("复核冻结配置与全部受保护输入")
    after = _input_snapshot(control["protected_p5_inputs"])
    input_hashes = pd.DataFrame(
        [
            {
                "path": path,
                "size_bytes_before": before[path]["size_bytes"],
                "size_bytes_after": after[path]["size_bytes"],
                "sha256_before": before[path]["sha256"],
                "sha256_after": after[path]["sha256"],
                "match": before[path] == after[path],
            }
            for path in control["protected_p5_inputs"]
        ]
    )
    _write_csv_atomic(input_hashes, outputs["p5_input_hashes"])
    if not bool(input_hashes["match"].all()):
        raise RuntimeError("P5运行期间受保护输入发生变化")
    if _sha256(absolute(control["inputs"]["frozen_config"])) != gate[
        "frozen_hash"
    ]:
        raise RuntimeError("P5运行期间冻结配置发生变化")

    intent["status"] = "COMPLETED"
    intent["completed_at_utc"] = datetime.now(UTC).isoformat()
    intent["results_report"] = outputs["p5_result_report"]
    intent["parameters_retuned"] = False
    intent["frozen_config_modified"] = False
    _write_json_atomic(intent, outputs["run_intent"])

    generated_keys = [
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
        *[f"oos_{key}" for key in oos_statistics],
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
    ]
    output_hashes = {
        key: _sha256(absolute(outputs[key]))
        for key in generated_keys
    }
    baseline = oos_performance.loc[
        oos_performance["is_baseline"].astype(bool)
    ].iloc[0]
    completed_at = datetime.now(UTC)
    manifest: dict[str, Any] = {
        "stage": "P5_FINAL_OOS",
        "builder_version": BUILDER_VERSION,
        "status": "BUILT_PENDING_AUDIT",
        "authorization": {
            "reference": control["project"][
                "authorization_reference"
            ],
            "obtained_before_run": True,
            "one_shot_final_oos": True,
        },
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "elapsed_seconds": round(
            (completed_at - started_at).total_seconds(), 3
        ),
        "runtime": {
            "python": platform.python_version(),
            "duckdb": duckdb.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "numpy": np.__version__,
        },
        "frozen_config": {
            "path": control["inputs"]["frozen_config"],
            "sha256": gate["frozen_hash"],
            "modified": False,
            "parameters_retuned": False,
        },
        "sample": {
            "research": frozen["sample"]["research"],
            "validation": frozen["sample"]["validation"],
            "final_oos": frozen["sample"]["final_oos"],
            "factor_panel_min": str(
                panel["signal_date"].min().date()
            ),
            "factor_panel_max": str(
                panel["signal_date"].max().date()
            ),
            "portfolio_min": str(daily["trade_date"].min().date()),
            "portfolio_max": str(daily["trade_date"].max().date()),
        },
        "counts": {
            "factor_rows": len(panel),
            "oos_factor_rows": len(oos_panel),
            "signal_months_full": int(
                signals["signal_date"].nunique()
            ),
            "signal_months_oos": int(
                oos_signals["signal_date"].nunique()
            ),
            "scheduled_rebalances_full": int(
                schedule["scheduled_trade_date"].notna().sum()
            ),
            "oos_executed_rebalances": int(len(oos_rebalances) / 3),
            "daily_rows_full": len(daily),
            "daily_rows_oos": len(
                oos_frames["oos_daily_portfolio"]
            ),
            "orders_full": len(orders),
            "orders_oos": len(oos_frames["oos_orders"]),
            "failed_orders_oos": len(
                oos_frames["oos_failed_orders"]
            ),
            "corporate_actions_oos": len(
                oos_frames["oos_corporate_action_events"]
            ),
        },
        "p4_reproduction_all_pass": bool(
            (reproduction["status"] == "PASS").all()
        ),
        "oos_baseline_result": {
            key: (
                int(baseline[key])
                if key in {
                    "trading_days",
                    "failed_buy_orders",
                    "failed_sell_orders",
                }
                else float(baseline[key])
            )
            for key in (
                "trading_days",
                "strategy_total_return",
                "strategy_annualized_return",
                "strategy_annualized_volatility",
                "strategy_sharpe_zero_rf",
                "strategy_max_drawdown_within_period",
                "benchmark_total_return",
                "benchmark_annualized_return",
                "benchmark_annualized_volatility",
                "benchmark_max_drawdown_within_period",
                "annualized_return_difference",
                "tracking_error_annualized",
                "information_ratio",
                "average_cash_weight",
                "maximum_stale_price_weight",
                "terminal_stale_price_weight",
                "terminal_stale_price_market_value",
                "two_way_turnover",
                "total_trading_cost",
                "failed_buy_orders",
                "failed_sell_orders",
            )
        },
        "protected_inputs_all_match": bool(
            input_hashes["match"].all()
        ),
        "output_sha256": output_hashes,
        "scope_guards": {
            "final_oos_authorized_before_run": True,
            "final_oos_results_computed": True,
            "final_oos_results_reported_after_authorization": True,
            "oos_previewed_before_authorization": False,
            "parameters_retuned_after_validation": False,
            "frozen_config_modified": False,
            "p6_code_generated": False,
            "p6_run": False,
        },
        "disclosures": [
            PB_DISCLOSURE,
            (
                "最终OOS只运行冻结策略；结果无论好坏均不得用于覆盖"
                "冻结配置或重定义本次OOS。"
            ),
        ],
    }
    _write_json_atomic(manifest, outputs["p5_run_manifest"])
    _log(
        "最终OOS构建完成："
        f"return={baseline['strategy_total_return']:.2%}，"
        f"benchmark={baseline['benchmark_total_return']:.2%}，"
        "等待P5独立审计"
    )
    return manifest


def build_p5() -> dict[str, Any]:
    gate = validate_p5_preflight(require_fresh_run=True)
    control = gate["control"]
    before = _input_snapshot(control["protected_p5_inputs"])
    started_at = datetime.now(UTC)
    intent: dict[str, Any] = {
        "stage": "P5_FINAL_OOS",
        "attempt_number": 1,
        "status": "STARTED",
        "authorization_reference": control["project"][
            "authorization_reference"
        ],
        "authorization_obtained_before_run": True,
        "one_shot_final_oos": True,
        "frozen_config_path": control["inputs"]["frozen_config"],
        "frozen_config_sha256": gate["frozen_hash"],
        "parameters_locked": True,
        "started_at_utc": started_at.isoformat(),
    }
    _write_json_atomic(intent, control["outputs"]["run_intent"])
    try:
        return _build_p5_after_intent(gate, before, intent)
    except Exception as exc:
        intent["status"] = "FAILED"
        intent["failed_at_utc"] = datetime.now(UTC).isoformat()
        intent["error_type"] = type(exc).__name__
        intent["error_message"] = str(exc)
        intent["parameters_retuned"] = False
        intent["frozen_config_modified"] = (
            _sha256(absolute(control["inputs"]["frozen_config"]))
            != gate["frozen_hash"]
        )
        _write_json_atomic(intent, control["outputs"]["run_intent"])
        raise
