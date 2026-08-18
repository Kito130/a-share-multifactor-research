from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd
import pyarrow
import yaml

from .config import CONFIG_PATH, PROJECT_ROOT, absolute, load_config, sql_path


BUILDER_VERSION = "p3.2"

CORPORATE_ACTION_COLUMNS = [
    "old_ts_code",
    "successor_ts_code",
    "action_type",
    "last_trade_date",
    "suspension_start_date",
    "record_date",
    "delist_date",
    "exchange_ratio",
    "cash_component",
    "commission_rate",
    "stamp_duty_rate",
    "assumption",
    "source_note",
]
ORIGINAL_DATA_PREFIXES = (
    "data/_parts/",
    "data/raw/",
    "data/static/",
    "data/external/",
)


def _log(message: str) -> None:
    print(f"[P3] {message}", flush=True)


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


def _write_parquet_atomic(frame: pd.DataFrame, relative_path: str) -> None:
    output = absolute(relative_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    frame.to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
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


def _input_snapshot(
    paths: Iterable[str],
) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for relative_path in paths:
        path = absolute(relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"P3受保护输入不存在：{relative_path}")
        snapshot[relative_path] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return snapshot


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    project = config["project"]
    research_end = pd.Timestamp(project["research_end"])
    validation_start = pd.Timestamp(project["validation_start"])
    oos_start = pd.Timestamp(project["oos_start"])
    if not research_end < validation_start < oos_start:
        raise ValueError("P3研究期、验证期和OOS闸门无效")
    portfolio = config["portfolio"]
    if int(portfolio["maximum_holdings"]) != 100:
        raise ValueError("P3冻结最大持仓数必须为100")
    if float(portfolio["maximum_single_name_target_weight"]) != 0.02:
        raise ValueError("P3冻结单票目标上限必须为2%")
    if not bool(portfolio["allow_fractional_adjusted_units"]):
        raise ValueError("当前P3实现固定使用可分割复权单位")
    if bool(portfolio["board_lot_rounding"]):
        raise ValueError("当前P3实现不做未冻结的整手取整")
    weights = config["composite"]
    weight_sum = sum(float(value) for value in weights.values())
    if abs(weight_sum - 1.0) > 1e-12:
        raise ValueError("复合因子权重之和必须为1")
    scenarios = config["cost_scenarios"]
    if len(scenarios) != 3:
        raise ValueError("P3必须包含5/10/20 bps三个成本情景")
    baseline = [item for item in scenarios if item["is_baseline"]]
    if len(baseline) != 1:
        raise ValueError("P3必须且只能有一个基准成本情景")
    p2_manifest = json.loads(
        absolute(config["inputs"]["p2_run_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    if not str(p2_manifest.get("status", "")).startswith("P2_ACCEPTED"):
        raise RuntimeError("P2尚未通过验收，不能开始P3")
    if p2_manifest.get("panel", {}).get("validation_or_later_rows") != 0:
        raise RuntimeError("P2输入包含验证期或更晚记录")
    return p2_manifest


def _load_stamp_policy(config: dict[str, Any]) -> dict[str, Any]:
    with absolute(config["inputs"]["trading_costs"]).open(
        "r", encoding="utf-8"
    ) as handle:
        policy: dict[str, Any] = yaml.safe_load(handle)
    if policy.get("policy_status") != "CONFIRMED":
        raise RuntimeError("历史印花税配置未确认")
    return policy


def _stamp_rate(trade_date: pd.Timestamp, policy: dict[str, Any]) -> float:
    normalized = pd.Timestamp(trade_date).normalize()
    for period in policy["stamp_duty"]["periods"]:
        start = pd.Timestamp(period["effective_from"])
        end = pd.Timestamp(period["effective_to"])
        if start <= normalized <= end:
            return float(period["sell_rate"])
    raise ValueError(f"交易日缺少印花税配置：{normalized.date()}")


def _load_corporate_actions(config: dict[str, Any]) -> pd.DataFrame:
    path = absolute(config["inputs"]["corporate_actions"])
    actions = pd.read_csv(
        path,
        dtype={
            "old_ts_code": "string",
            "successor_ts_code": "string",
            "action_type": "string",
            "last_trade_date": "string",
            "suspension_start_date": "string",
            "record_date": "string",
            "delist_date": "string",
            "assumption": "string",
            "source_note": "string",
        },
    )
    missing = [
        column
        for column in CORPORATE_ACTION_COLUMNS
        if column not in actions.columns
    ]
    if missing:
        raise ValueError(f"人工公司行动表缺少字段：{missing}")
    actions = actions[CORPORATE_ACTION_COLUMNS].copy()
    if actions.empty:
        raise ValueError("人工公司行动表不得为空")
    for column in (
        "last_trade_date",
        "suspension_start_date",
        "record_date",
        "delist_date",
    ):
        actions[column] = pd.to_datetime(
            actions[column], format="%Y%m%d", errors="raise"
        )
    for column in (
        "exchange_ratio",
        "cash_component",
        "commission_rate",
        "stamp_duty_rate",
    ):
        actions[column] = pd.to_numeric(actions[column], errors="raise")
    effective_field = str(
        config["corporate_actions"]["effective_date_field"]
    )
    if effective_field not in actions.columns:
        raise ValueError(
            f"公司行动生效日字段不存在：{effective_field}"
        )
    actions["effective_date"] = actions[effective_field]
    if actions.duplicated(["old_ts_code", "effective_date"]).any():
        raise ValueError("人工公司行动表存在旧代码/生效日重复")
    if (
        (actions["old_ts_code"] == actions["successor_ts_code"]).any()
        or (actions["action_type"] != "STOCK_SWAP_ABSORPTION").any()
        or (actions["exchange_ratio"] <= 0).any()
    ):
        raise ValueError("公司行动代码、类型或换股比例无效")
    if (
        (actions["cash_component"] != 0).any()
        or (actions["commission_rate"] != 0).any()
        or (actions["stamp_duty_rate"] != 0).any()
    ):
        raise ValueError("本次换股必须为零现金、零佣金、零印花税")
    date_order_valid = (
        (actions["last_trade_date"] < actions["suspension_start_date"])
        & (actions["suspension_start_date"] <= actions["record_date"])
        & (actions["record_date"] < actions["delist_date"])
    )
    if not bool(date_order_valid.all()):
        raise ValueError("公司行动日期顺序无效")
    if actions[
        [
            "old_ts_code",
            "successor_ts_code",
            "action_type",
            "assumption",
            "source_note",
        ]
    ].isna().any().any():
        raise ValueError("公司行动关键文本字段不得为空")
    frozen = config["corporate_actions"]
    if (
        frozen["quantity_basis"] != "raw_share_equivalent"
        or not bool(frozen["fractional_shares_allowed"])
        or bool(frozen["automatic_successor_sale"])
        or frozen["event_activity_type"] != "CORPORATE_ACTION"
    ):
        raise ValueError("公司行动冻结配置与人工确认不一致")
    return actions.sort_values(
        ["effective_date", "old_ts_code"], kind="mergesort"
    ).reset_index(drop=True)


def _prepare_corporate_action_references(
    actions: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for action in actions.to_dict(orient="records"):
        old_code = str(action["old_ts_code"])
        successor_code = str(action["successor_ts_code"])
        last_trade_date = pd.Timestamp(action["last_trade_date"])
        effective_date = pd.Timestamp(action["effective_date"])
        old_reference = prices.loc[
            (prices["canonical_ts_code"] == old_code)
            & (prices["ts_code"] == old_code)
            & (prices["trade_date"] == last_trade_date)
        ]
        if len(old_reference) != 1:
            raise RuntimeError(
                f"公司行动旧代码最后交易日行情不唯一："
                f"{old_code} {last_trade_date.date()} rows={len(old_reference)}"
            )
        unexpected_old_prices = prices.loc[
            (prices["canonical_ts_code"] == old_code)
            & (prices["trade_date"] > last_trade_date)
        ]
        if not unexpected_old_prices.empty:
            raise RuntimeError(
                f"旧代码最后交易日后仍存在行情：{old_code}"
            )
        successor_prices = prices.loc[
            (prices["canonical_ts_code"] == successor_code)
            & (prices["trade_date"] >= effective_date)
            & prices["adjusted_close"].notna()
            & (prices["adjusted_close"] > 0)
        ].sort_values("trade_date")
        if successor_prices.empty:
            raise RuntimeError(
                f"继任证券缺少生效日后的可得行情：{successor_code}"
            )
        old = old_reference.iloc[0]
        successor = successor_prices.iloc[0]
        old_factor = float(old["adj_factor"])
        successor_factor = float(successor["adj_factor"])
        old_raw_close = float(old["raw_close"])
        old_adjusted_close = float(old["adjusted_close"])
        if (
            old_factor <= 0
            or successor_factor <= 0
            or old_raw_close <= 0
            or abs(old_raw_close * old_factor - old_adjusted_close) > 1e-8
        ):
            raise RuntimeError(
                f"公司行动参考价格或复权因子无效：{old_code}"
            )
        rows.append(
            {
                **action,
                "old_last_raw_close": old_raw_close,
                "old_last_adjusted_close": old_adjusted_close,
                "old_adj_factor_at_last_trade": old_factor,
                "successor_first_price_date": pd.Timestamp(
                    successor["trade_date"]
                ),
                "successor_initial_adj_factor": successor_factor,
                "successor_first_raw_close": float(
                    successor["raw_close"]
                ),
                "successor_first_adjusted_close": float(
                    successor["adjusted_close"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _original_data_snapshot(
    config: dict[str, Any],
) -> pd.DataFrame:
    recorded = pd.read_csv(
        absolute(config["inputs"]["p1_protected_input_hashes"])
    )
    recorded = recorded.loc[
        recorded["path"].astype(str).str.startswith(
            ORIGINAL_DATA_PREFIXES
        )
    ].copy()
    if recorded.empty:
        raise RuntimeError("P1原始数据哈希清单为空")
    rows: list[dict[str, Any]] = []
    for row in recorded.itertuples(index=False):
        relative_path = str(row.path)
        path = absolute(relative_path)
        exists = path.is_file()
        current_size = path.stat().st_size if exists else pd.NA
        current_hash = _sha256(path) if exists else ""
        rows.append(
            {
                "path": relative_path,
                "p1_recorded_size_bytes": int(row.size_bytes_after),
                "current_size_bytes": current_size,
                "p1_recorded_sha256": str(row.sha256_after),
                "current_sha256": current_hash,
                "matches_p1_record": (
                    exists
                    and current_size == int(row.size_bytes_after)
                    and current_hash == str(row.sha256_after)
                    and str(row.match).lower() == "true"
                ),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_path_hash(
    frame: pd.DataFrame, hash_column: str
) -> str:
    digest = hashlib.sha256()
    for row in frame.sort_values("path").itertuples(index=False):
        digest.update(str(row.path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(getattr(row, hash_column)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_legacy_performance(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = pd.read_csv(
        absolute(config["inputs"]["legacy_p3_archive_manifest"])
    )
    performance_source = "results/p3_backtest/performance_summary.csv"
    match = manifest.loc[manifest["source_path"] == performance_source]
    if len(match) != 1:
        raise RuntimeError("旧版P3归档清单缺少唯一绩效文件")
    legacy_path = absolute(
        config["inputs"]["legacy_p3_performance_summary"]
    )
    expected_hash = str(match.iloc[0]["sha256"])
    actual_hash = _sha256(legacy_path)
    if actual_hash != expected_hash:
        raise RuntimeError("旧版P3绩效归档SHA-256不匹配")
    performance = pd.read_csv(legacy_path)
    return performance, {
        "archive_manifest_rows": len(manifest),
        "legacy_performance_sha256": actual_hash,
        "legacy_performance_hash_matches_manifest": True,
    }


def _corporate_action_impact(
    legacy: pd.DataFrame,
    current: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "cost_scenario",
        "is_baseline",
        "strategy_annualized_return",
        "strategy_total_return",
        "strategy_max_drawdown",
    ]
    before = legacy[columns].rename(
        columns={
            "strategy_annualized_return": "annualized_return_before",
            "strategy_total_return": "total_return_before",
            "strategy_max_drawdown": "max_drawdown_before",
        }
    )
    after = current[columns].rename(
        columns={
            "strategy_annualized_return": "annualized_return_after",
            "strategy_total_return": "total_return_after",
            "strategy_max_drawdown": "max_drawdown_after",
        }
    )
    impact = before.merge(
        after,
        on=["cost_scenario", "is_baseline"],
        how="inner",
        validate="one_to_one",
    )
    if len(impact) != len(current):
        raise RuntimeError("旧版与修复版P3成本情景无法一一对应")
    for metric in ("annualized_return", "total_return", "max_drawdown"):
        impact[f"{metric}_difference"] = (
            impact[f"{metric}_after"] - impact[f"{metric}_before"]
        )
    return impact


def _build_composite_signals(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "ts_code",
        "canonical_ts_code",
        "signal_date",
        "bm_proxy_z",
        "momentum_12_1_z",
        "lowvol_60_z",
        "universe_eligible",
    ]
    panel = pd.read_parquet(
        absolute(config["inputs"]["p2_single_factor_panel"]),
        columns=columns,
    )
    panel["signal_date"] = pd.to_datetime(panel["signal_date"])
    project = config["project"]
    panel = panel.loc[
        panel["universe_eligible"].fillna(False)
        & (panel["signal_date"] >= pd.Timestamp(project["research_start"]))
        & (panel["signal_date"] <= pd.Timestamp(project["research_end"]))
    ].copy()
    weights = config["composite"]
    panel["composite_score"] = (
        panel["bm_proxy_z"] * float(weights["bm_proxy_z_weight"])
        + panel["momentum_12_1_z"]
        * float(weights["momentum_12_1_z_weight"])
        + panel["lowvol_60_z"]
        * float(weights["lowvol_60_z_weight"])
    )
    panel = panel.sort_values(
        ["signal_date", "composite_score", "canonical_ts_code"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    panel["selection_rank"] = (
        panel.groupby("signal_date", sort=False).cumcount() + 1
    )
    maximum_holdings = int(config["portfolio"]["maximum_holdings"])
    panel["is_selected"] = panel["selection_rank"] <= maximum_holdings
    selected_counts = (
        panel.loc[panel["is_selected"]]
        .groupby("signal_date")["canonical_ts_code"]
        .transform("count")
    )
    panel["target_weight"] = 0.0
    target_gross = float(config["portfolio"]["target_gross_weight"])
    maximum_weight = float(
        config["portfolio"]["maximum_single_name_target_weight"]
    )
    panel.loc[panel["is_selected"], "target_weight"] = np.minimum(
        target_gross / selected_counts.astype(float), maximum_weight
    )
    panel["composite_definition"] = (
        "EQUAL_WEIGHT_BM_MOMENTUM_LOWVOL_Z"
    )
    targets = panel.loc[panel["is_selected"]].copy()
    return panel.reset_index(drop=True), targets.reset_index(drop=True)


def _build_schedule(
    signals: pd.DataFrame,
    benchmark: pd.DataFrame,
    research_end: pd.Timestamp,
) -> pd.DataFrame:
    market_dates = pd.DatetimeIndex(
        benchmark.loc[
            benchmark["trade_date"] <= research_end, "trade_date"
        ].sort_values()
    )
    rows: list[dict[str, Any]] = []
    for signal_date in sorted(signals["signal_date"].unique()):
        signal = pd.Timestamp(signal_date)
        location = market_dates.searchsorted(signal, side="right")
        trade_date = (
            market_dates[location] if location < len(market_dates) else pd.NaT
        )
        rows.append(
            {
                "signal_date": signal,
                "scheduled_trade_date": trade_date,
                "schedule_status": (
                    "SCHEDULED_WITHIN_RESEARCH"
                    if pd.notna(trade_date)
                    else "OUT_OF_SCOPE_NOT_EXECUTED"
                ),
            }
        )
    return pd.DataFrame(rows)


def _load_market_inputs(
    config: dict[str, Any],
    targets: pd.DataFrame,
    schedule: pd.DataFrame,
    benchmark: pd.DataFrame,
    corporate_actions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    action_codes = set(corporate_actions["old_ts_code"].astype(str)) | set(
        corporate_actions["successor_ts_code"].astype(str)
    )
    codes = pd.DataFrame(
        {
            "canonical_ts_code": sorted(
                set(targets["canonical_ts_code"].astype(str))
                | action_codes
            )
        }
    )
    trade_dates = pd.DataFrame(
        {
            "trade_date": sorted(
                schedule["scheduled_trade_date"].dropna().unique()
            )
        }
    )
    first_signal = pd.Timestamp(schedule["signal_date"].min())
    research_end = pd.Timestamp(config["project"]["research_end"])
    daily_path = sql_path(config["inputs"]["daily_panel"])
    execution_path = sql_path(config["inputs"]["execution_status"])
    with duckdb.connect() as connection:
        connection.register("selected_codes", codes)
        connection.register("scheduled_trade_dates", trade_dates)
        prices = connection.execute(
            f"""
            SELECT
                daily.ts_code,
                daily.canonical_ts_code,
                daily.trade_date,
                daily.open AS raw_open,
                daily.close AS raw_close,
                daily.adj_factor,
                daily.open * daily.adj_factor AS adjusted_open,
                daily.adjusted_close
            FROM read_parquet('{daily_path}') AS daily
            INNER JOIN selected_codes USING (canonical_ts_code)
            WHERE daily.trade_date BETWEEN DATE '{first_signal.date()}'
                                       AND DATE '{research_end.date()}'
              AND daily.is_sh_sz
              AND daily.is_within_listing_window
              AND daily.security_code_interval_valid
            ORDER BY daily.trade_date, daily.canonical_ts_code
            """
        ).fetchdf()
        executions = connection.execute(
            f"""
            SELECT
                execution.ts_code,
                execution.canonical_ts_code,
                execution.trade_date,
                execution.open_price,
                execution.has_daily_price,
                execution.has_suspend_record,
                execution.suspended_at_open,
                execution.up_limit,
                execution.down_limit,
                execution.open_at_up_limit,
                execution.open_at_down_limit,
                execution.cannot_buy_at_open,
                execution.cannot_sell_at_open,
                execution.execution_block_reason
            FROM read_parquet('{execution_path}') AS execution
            INNER JOIN selected_codes USING (canonical_ts_code)
            INNER JOIN scheduled_trade_dates USING (trade_date)
            WHERE execution.security_code_interval_valid
            ORDER BY execution.trade_date, execution.canonical_ts_code
            """
        ).fetchdf()
    for frame, date_column in (
        (prices, "trade_date"),
        (executions, "trade_date"),
        (benchmark, "trade_date"),
    ):
        frame[date_column] = pd.to_datetime(frame[date_column])
    duplicate_prices = prices.duplicated(
        ["canonical_ts_code", "trade_date"]
    ).sum()
    duplicate_executions = executions.duplicated(
        ["canonical_ts_code", "trade_date"]
    ).sum()
    if duplicate_prices or duplicate_executions:
        raise RuntimeError(
            "P3市场输入存在canonical重复："
            f"prices={duplicate_prices}, executions={duplicate_executions}"
        )
    return prices, executions, benchmark


def _records_by_date(
    frame: pd.DataFrame, date_column: str
) -> dict[pd.Timestamp, dict[str, dict[str, Any]]]:
    result: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}
    for date_value, group in frame.groupby(date_column, sort=True):
        result[pd.Timestamp(date_value)] = {
            str(row["canonical_ts_code"]): row
            for row in group.to_dict(orient="records")
        }
    return result


def _execution_reason(
    side: str,
    execution: dict[str, Any] | None,
    price: dict[str, Any] | None,
) -> str:
    if execution is None:
        return "NO_EXECUTION_STATUS"
    if (
        not bool(execution.get("has_daily_price", False))
        or price is None
        or pd.isna(price.get("adjusted_open"))
        or float(price.get("adjusted_open", 0.0)) <= 0
    ):
        if bool(execution.get("suspended_at_open", False)):
            return "NO_OPEN_PRICE_AND_SUSPENDED_AT_OPEN"
        return "NO_OPEN_PRICE"
    if bool(execution.get("suspended_at_open", False)):
        return "SUSPENDED_AT_OPEN"
    if side == "BUY" and bool(
        execution.get("open_at_up_limit", False)
    ):
        return "OPEN_AT_UP_LIMIT"
    if side == "SELL" and bool(
        execution.get("open_at_down_limit", False)
    ):
        return "OPEN_AT_DOWN_LIMIT"
    blocked = (
        bool(execution.get("cannot_buy_at_open", False))
        if side == "BUY"
        else bool(execution.get("cannot_sell_at_open", False))
    )
    if blocked:
        return str(
            execution.get("execution_block_reason")
            or "UNCLASSIFIED_EXECUTION_BLOCK"
        )
    return ""


def _simulate_scenario(
    config: dict[str, Any],
    scenario: dict[str, Any],
    targets: pd.DataFrame,
    schedule: pd.DataFrame,
    benchmark: pd.DataFrame,
    prices: pd.DataFrame,
    executions: pd.DataFrame,
    stamp_policy: dict[str, Any],
    corporate_action_references: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    scenario_name = str(scenario["scenario"])
    cost_rate = float(scenario["commission_slippage_rate_one_way"])
    is_baseline = bool(scenario["is_baseline"])
    portfolio_config = config["portfolio"]
    initial_capital = float(portfolio_config["initial_capital_cny"])
    minimum_order = float(
        portfolio_config["minimum_order_notional_cny"]
    )
    unit_epsilon = float(portfolio_config["position_unit_epsilon"])
    research_end = pd.Timestamp(config["project"]["research_end"])
    first_signal = pd.Timestamp(schedule["signal_date"].min())
    market_calendar = benchmark.loc[
        (benchmark["trade_date"] >= first_signal)
        & (benchmark["trade_date"] <= research_end)
    ].sort_values("trade_date")
    price_by_date = _records_by_date(prices, "trade_date")
    execution_by_date = _records_by_date(executions, "trade_date")
    actions_by_date: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    for effective_date, group in corporate_action_references.groupby(
        "effective_date", sort=True
    ):
        actions_by_date[pd.Timestamp(effective_date)] = group.to_dict(
            orient="records"
        )
    target_by_signal: dict[
        pd.Timestamp, dict[str, dict[str, Any]]
    ] = {}
    for signal_date, group in targets.groupby("signal_date", sort=True):
        target_by_signal[pd.Timestamp(signal_date)] = {
            str(row["canonical_ts_code"]): row
            for row in group.to_dict(orient="records")
        }
    trade_to_signal = {
        pd.Timestamp(row.scheduled_trade_date): pd.Timestamp(row.signal_date)
        for row in schedule.itertuples(index=False)
        if pd.notna(row.scheduled_trade_date)
    }

    positions: dict[str, float] = {}
    last_close: dict[str, float] = {}
    last_close_date: dict[str, pd.Timestamp] = {}
    last_price_source: dict[str, str] = {}
    last_ts_code: dict[str, str] = {}
    position_origin: dict[str, str] = {}
    current_targets: dict[str, dict[str, Any]] = {}
    cash = initial_capital
    order_sequence = 0
    order_rows: list[dict[str, Any]] = []
    holding_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    rebalance_rows: list[dict[str, Any]] = []
    corporate_action_rows: list[dict[str, Any]] = []
    corporate_action_sequence = 0

    for calendar_row in market_calendar.itertuples(index=False):
        trade_date = pd.Timestamp(calendar_row.trade_date)
        day_prices = price_by_date.get(trade_date, {})
        day_executions = execution_by_date.get(trade_date, {})
        signal_date = trade_to_signal.get(trade_date)
        day_buy_notional = 0.0
        day_sell_notional = 0.0
        day_commission = 0.0
        day_stamp = 0.0
        day_failed_orders = 0
        day_partial_orders = 0
        day_corporate_actions = 0
        day_corporate_action_value_difference = 0.0
        pretrade_nav_open = math.nan

        for action in actions_by_date.get(trade_date, []):
            old_code = str(action["old_ts_code"])
            successor_code = str(action["successor_ts_code"])
            old_units = positions.get(old_code, 0.0)
            if old_units <= unit_epsilon:
                continue
            existing_successor_units = positions.get(successor_code, 0.0)
            if existing_successor_units > unit_epsilon:
                raise RuntimeError(
                    "公司行动生效前已存在继任证券持仓，"
                    f"当前冻结实现不允许合并两种估值基准：{successor_code}"
                )
            if old_code not in last_close:
                raise RuntimeError(
                    f"公司行动旧持仓缺少最后估值：{old_code}"
                )
            old_factor = float(action["old_adj_factor_at_last_trade"])
            successor_factor = float(
                action["successor_initial_adj_factor"]
            )
            exchange_ratio = float(action["exchange_ratio"])
            old_share_quantity = old_units * old_factor
            successor_share_quantity = (
                old_share_quantity * exchange_ratio
            )
            successor_adjusted_units = (
                successor_share_quantity / successor_factor
            )
            old_position_value = old_units * last_close[old_code]
            expected_old_value = (
                old_share_quantity
                * float(action["old_last_raw_close"])
            )
            if abs(old_position_value - expected_old_value) > 1e-6:
                raise RuntimeError(
                    f"换股前复权单位与原始股数估值不一致：{old_code}"
                )
            successor_carry_price = (
                old_position_value / successor_adjusted_units
            )
            portfolio_value_before = cash + sum(
                units * last_close[code]
                for code, units in positions.items()
            )

            positions.pop(old_code)
            positions[successor_code] = successor_adjusted_units
            last_close.pop(old_code, None)
            last_close_date.pop(old_code, None)
            last_price_source.pop(old_code, None)
            last_ts_code.pop(old_code, None)
            position_origin.pop(old_code, None)
            last_close[successor_code] = successor_carry_price
            last_close_date[successor_code] = pd.Timestamp(
                action["last_trade_date"]
            )
            last_price_source[successor_code] = (
                "CORPORATE_ACTION_CARRY_VALUE"
            )
            last_ts_code[successor_code] = successor_code
            position_origin[successor_code] = "CORPORATE_ACTION"

            successor_position_value = (
                successor_adjusted_units * successor_carry_price
            )
            portfolio_value_after = cash + sum(
                units * last_close[code]
                for code, units in positions.items()
            )
            corporate_action_sequence += 1
            corporate_action_rows.append(
                {
                    "event_id": (
                        f"{scenario_name}-CA-"
                        f"{corporate_action_sequence:04d}"
                    ),
                    "activity_type": "CORPORATE_ACTION",
                    "cost_scenario": scenario_name,
                    "is_baseline": is_baseline,
                    "effective_date": trade_date,
                    "old_ts_code": old_code,
                    "successor_ts_code": successor_code,
                    "action_type": action["action_type"],
                    "last_trade_date": action["last_trade_date"],
                    "suspension_start_date": action[
                        "suspension_start_date"
                    ],
                    "record_date": action["record_date"],
                    "delist_date": action["delist_date"],
                    "old_adjusted_units_before": old_units,
                    "old_adj_factor_at_last_trade": old_factor,
                    "old_share_quantity_before": old_share_quantity,
                    "exchange_ratio": exchange_ratio,
                    "successor_share_quantity_after": (
                        successor_share_quantity
                    ),
                    "successor_initial_adj_factor": successor_factor,
                    "successor_adjusted_units_after": (
                        successor_adjusted_units
                    ),
                    "old_last_raw_close": action[
                        "old_last_raw_close"
                    ],
                    "old_last_adjusted_close": action[
                        "old_last_adjusted_close"
                    ],
                    "successor_carry_adjusted_price": (
                        successor_carry_price
                    ),
                    "successor_first_price_date": action[
                        "successor_first_price_date"
                    ],
                    "successor_first_raw_close": action[
                        "successor_first_raw_close"
                    ],
                    "successor_first_adjusted_close": action[
                        "successor_first_adjusted_close"
                    ],
                    "old_position_value_before_cny": (
                        old_position_value
                    ),
                    "successor_position_value_after_cny": (
                        successor_position_value
                    ),
                    "position_value_difference_cny": (
                        successor_position_value - old_position_value
                    ),
                    "portfolio_value_before_action_cny": (
                        portfolio_value_before
                    ),
                    "portfolio_value_after_action_cny": (
                        portfolio_value_after
                    ),
                    "portfolio_value_difference_cny": (
                        portfolio_value_after - portfolio_value_before
                    ),
                    "cash_component": action["cash_component"],
                    "commission_rate": action["commission_rate"],
                    "commission_cost_cny": 0.0,
                    "stamp_duty_rate": action["stamp_duty_rate"],
                    "stamp_duty_cost_cny": 0.0,
                    "slippage_cost_cny": 0.0,
                    "total_action_cost_cny": 0.0,
                    "fractional_shares_allowed": True,
                    "automatic_successor_sale": False,
                    "assumption": action["assumption"],
                    "source_note": action["source_note"],
                }
            )
            day_corporate_actions += 1
            day_corporate_action_value_difference += (
                portfolio_value_after - portfolio_value_before
            )

        if signal_date is not None:
            target_map = target_by_signal[signal_date]
            open_position_values: dict[str, float] = {}
            for code, units in positions.items():
                price_record = day_prices.get(code)
                open_value_price = (
                    float(price_record["adjusted_open"])
                    if price_record is not None
                    and pd.notna(price_record["adjusted_open"])
                    and float(price_record["adjusted_open"]) > 0
                    else last_close.get(code)
                )
                if open_value_price is None:
                    raise RuntimeError(
                        f"持仓缺少开盘或上一估值价格：{code} {trade_date}"
                    )
                open_position_values[code] = units * open_value_price
            pretrade_nav_open = cash + sum(open_position_values.values())
            if pretrade_nav_open <= 0:
                raise RuntimeError("P3组合净值非正")

            sell_requests: list[dict[str, Any]] = []
            buy_requests: list[dict[str, Any]] = []
            candidate_codes = sorted(set(positions) | set(target_map))
            for code in candidate_codes:
                current_value = open_position_values.get(code, 0.0)
                target_record = target_map.get(code)
                target_weight = (
                    float(target_record["target_weight"])
                    if target_record is not None
                    else 0.0
                )
                target_value = pretrade_nav_open * target_weight
                difference = target_value - current_value
                request = {
                    "canonical_ts_code": code,
                    "target_record": target_record,
                    "current_value": current_value,
                    "target_value": target_value,
                    "target_weight": target_weight,
                    "desired_notional": abs(difference),
                }
                if difference < -minimum_order:
                    sell_requests.append(request)
                elif difference > minimum_order:
                    buy_requests.append(request)

            stamp_rate = _stamp_rate(trade_date, stamp_policy)
            rebalance_order_start = len(order_rows)
            for request in sell_requests:
                code = request["canonical_ts_code"]
                price_record = day_prices.get(code)
                execution_record = day_executions.get(code)
                reason = _execution_reason(
                    "SELL", execution_record, price_record
                )
                adjusted_open = (
                    float(price_record["adjusted_open"])
                    if price_record is not None
                    and pd.notna(price_record["adjusted_open"])
                    else math.nan
                )
                current_units = positions.get(code, 0.0)
                desired_units = (
                    min(
                        current_units,
                        request["desired_notional"] / adjusted_open,
                    )
                    if not math.isnan(adjusted_open)
                    and adjusted_open > 0
                    else math.nan
                )
                executed_units = 0.0
                executed_notional = 0.0
                commission = 0.0
                stamp = 0.0
                status = "FAILED" if reason else "FILLED"
                if not reason:
                    executed_units = desired_units
                    executed_notional = executed_units * adjusted_open
                    commission = executed_notional * cost_rate
                    stamp = executed_notional * stamp_rate
                    cash += executed_notional - commission - stamp
                    remaining = current_units - executed_units
                    if remaining <= unit_epsilon:
                        positions.pop(code, None)
                        position_origin.pop(code, None)
                    else:
                        positions[code] = remaining
                else:
                    day_failed_orders += 1
                order_sequence += 1
                target_record = request["target_record"]
                order_rows.append(
                    {
                        "order_id": (
                            f"{scenario_name}-{order_sequence:06d}"
                        ),
                        "activity_type": "TRADE",
                        "cost_scenario": scenario_name,
                        "is_baseline": is_baseline,
                        "signal_date": signal_date,
                        "trade_date": trade_date,
                        "side": "SELL",
                        "canonical_ts_code": code,
                        "signal_ts_code": (
                            target_record["ts_code"]
                            if target_record is not None
                            else None
                        ),
                        "execution_ts_code": (
                            execution_record.get("ts_code")
                            if execution_record is not None
                            else (
                                price_record.get("ts_code")
                                if price_record is not None
                                else last_ts_code.get(code)
                            )
                        ),
                        "selection_rank": (
                            target_record["selection_rank"]
                            if target_record is not None
                            else pd.NA
                        ),
                        "composite_score": (
                            target_record["composite_score"]
                            if target_record is not None
                            else math.nan
                        ),
                        "target_weight": request["target_weight"],
                        "pretrade_nav_open": pretrade_nav_open,
                        "current_position_value_open": request[
                            "current_value"
                        ],
                        "target_position_value_open": request[
                            "target_value"
                        ],
                        "desired_order_notional": request[
                            "desired_notional"
                        ],
                        "desired_adjusted_units": desired_units,
                        "cash_scaling_factor": 1.0,
                        "order_status": status,
                        "failure_reason": reason,
                        "execution_price_source": (
                            "ADJUSTED_OPEN" if not reason else ""
                        ),
                        "raw_open": (
                            price_record.get("raw_open")
                            if price_record is not None
                            else math.nan
                        ),
                        "adj_factor": (
                            price_record.get("adj_factor")
                            if price_record is not None
                            else math.nan
                        ),
                        "adjusted_open": adjusted_open,
                        "executed_adjusted_units": executed_units,
                        "executed_notional": executed_notional,
                        "commission_slippage_rate": cost_rate,
                        "commission_slippage_cost": commission,
                        "stamp_duty_rate": stamp_rate,
                        "stamp_duty_cost": stamp,
                        "total_trading_cost": commission + stamp,
                        "remaining_adjusted_units_after": positions.get(
                            code, 0.0
                        ),
                    }
                )
                day_sell_notional += executed_notional
                day_commission += commission
                day_stamp += stamp

            total_buy_demand = sum(
                request["desired_notional"] for request in buy_requests
            )
            affordable_scale = (
                min(
                    1.0,
                    max(cash, 0.0)
                    / (total_buy_demand * (1.0 + cost_rate)),
                )
                if total_buy_demand > 0
                else 1.0
            )
            for request in buy_requests:
                code = request["canonical_ts_code"]
                price_record = day_prices.get(code)
                execution_record = day_executions.get(code)
                reason = _execution_reason(
                    "BUY", execution_record, price_record
                )
                adjusted_open = (
                    float(price_record["adjusted_open"])
                    if price_record is not None
                    and pd.notna(price_record["adjusted_open"])
                    else math.nan
                )
                planned_notional = (
                    request["desired_notional"] * affordable_scale
                )
                executed_notional = 0.0
                executed_units = 0.0
                commission = 0.0
                status = "FAILED" if reason else "FILLED"
                if not reason and planned_notional > 0:
                    executed_notional = planned_notional
                    executed_units = executed_notional / adjusted_open
                    commission = executed_notional * cost_rate
                    required_cash = executed_notional + commission
                    if required_cash > cash + 0.01:
                        raise RuntimeError(
                            "买单同比缩放后仍超过可用现金"
                        )
                    cash -= required_cash
                    prior_units = positions.get(code, 0.0)
                    positions[code] = prior_units + executed_units
                    prior_origin = position_origin.get(code)
                    if (
                        prior_units > unit_epsilon
                        and prior_origin
                        and prior_origin != "TRADE"
                    ):
                        position_origin[code] = (
                            "MIXED_TRADE_AND_CORPORATE_ACTION"
                        )
                    else:
                        position_origin[code] = "TRADE"
                    if affordable_scale < 1.0 - 1e-12:
                        status = "PARTIAL_CASH_CONSTRAINT"
                        day_partial_orders += 1
                elif not reason:
                    reason = "INSUFFICIENT_CASH"
                    status = "FAILED"
                if reason:
                    day_failed_orders += 1
                order_sequence += 1
                target_record = request["target_record"]
                order_rows.append(
                    {
                        "order_id": (
                            f"{scenario_name}-{order_sequence:06d}"
                        ),
                        "activity_type": "TRADE",
                        "cost_scenario": scenario_name,
                        "is_baseline": is_baseline,
                        "signal_date": signal_date,
                        "trade_date": trade_date,
                        "side": "BUY",
                        "canonical_ts_code": code,
                        "signal_ts_code": target_record["ts_code"],
                        "execution_ts_code": (
                            execution_record.get("ts_code")
                            if execution_record is not None
                            else (
                                price_record.get("ts_code")
                                if price_record is not None
                                else None
                            )
                        ),
                        "selection_rank": target_record[
                            "selection_rank"
                        ],
                        "composite_score": target_record[
                            "composite_score"
                        ],
                        "target_weight": request["target_weight"],
                        "pretrade_nav_open": pretrade_nav_open,
                        "current_position_value_open": request[
                            "current_value"
                        ],
                        "target_position_value_open": request[
                            "target_value"
                        ],
                        "desired_order_notional": request[
                            "desired_notional"
                        ],
                        "desired_adjusted_units": (
                            request["desired_notional"] / adjusted_open
                            if not math.isnan(adjusted_open)
                            and adjusted_open > 0
                            else math.nan
                        ),
                        "cash_scaling_factor": affordable_scale,
                        "order_status": status,
                        "failure_reason": reason,
                        "execution_price_source": (
                            "ADJUSTED_OPEN"
                            if executed_notional > 0
                            else ""
                        ),
                        "raw_open": (
                            price_record.get("raw_open")
                            if price_record is not None
                            else math.nan
                        ),
                        "adj_factor": (
                            price_record.get("adj_factor")
                            if price_record is not None
                            else math.nan
                        ),
                        "adjusted_open": adjusted_open,
                        "executed_adjusted_units": executed_units,
                        "executed_notional": executed_notional,
                        "commission_slippage_rate": cost_rate,
                        "commission_slippage_cost": commission,
                        "stamp_duty_rate": 0.0,
                        "stamp_duty_cost": 0.0,
                        "total_trading_cost": commission,
                        "remaining_adjusted_units_after": positions.get(
                            code, 0.0
                        ),
                    }
                )
                day_buy_notional += executed_notional
                day_commission += commission

            if -0.01 < cash < 0:
                cash = 0.0
            current_targets = target_map
            after_open_value = 0.0
            for code, units in positions.items():
                price_record = day_prices.get(code)
                valuation_price = (
                    float(price_record["adjusted_open"])
                    if price_record is not None
                    and pd.notna(price_record["adjusted_open"])
                    and float(price_record["adjusted_open"]) > 0
                    else last_close[code]
                )
                after_open_value += units * valuation_price
            rebalance_orders = order_rows[rebalance_order_start:]
            rebalance_rows.append(
                {
                    "cost_scenario": scenario_name,
                    "is_baseline": is_baseline,
                    "signal_date": signal_date,
                    "trade_date": trade_date,
                    "selected_target_count": len(target_map),
                    "pretrade_nav_open": pretrade_nav_open,
                    "target_gross_weight": sum(
                        float(record["target_weight"])
                        for record in target_map.values()
                    ),
                    "sell_order_count": sum(
                        row["side"] == "SELL"
                        for row in rebalance_orders
                    ),
                    "buy_order_count": sum(
                        row["side"] == "BUY"
                        for row in rebalance_orders
                    ),
                    "failed_sell_order_count": sum(
                        row["side"] == "SELL"
                        and row["order_status"] == "FAILED"
                        for row in rebalance_orders
                    ),
                    "failed_buy_order_count": sum(
                        row["side"] == "BUY"
                        and row["order_status"] == "FAILED"
                        for row in rebalance_orders
                    ),
                    "partial_buy_order_count": sum(
                        row["side"] == "BUY"
                        and row["order_status"]
                        == "PARTIAL_CASH_CONSTRAINT"
                        for row in rebalance_orders
                    ),
                    "executed_sell_notional": sum(
                        row["executed_notional"]
                        for row in rebalance_orders
                        if row["side"] == "SELL"
                    ),
                    "executed_buy_notional": sum(
                        row["executed_notional"]
                        for row in rebalance_orders
                        if row["side"] == "BUY"
                    ),
                    "commission_slippage_cost": sum(
                        row["commission_slippage_cost"]
                        for row in rebalance_orders
                    ),
                    "stamp_duty_cost": sum(
                        row["stamp_duty_cost"]
                        for row in rebalance_orders
                    ),
                    "total_trading_cost": sum(
                        row["total_trading_cost"]
                        for row in rebalance_orders
                    ),
                    "cash_scaling_factor": affordable_scale,
                    "cash_after_open": cash,
                    "market_value_after_open": after_open_value,
                    "nav_after_open": cash + after_open_value,
                    "holding_count_after_open": len(positions),
                }
            )

        position_values: list[
            tuple[str, float, float, str, str, pd.Timestamp]
        ] = []
        stale_count = 0
        for code, units in list(positions.items()):
            if units <= unit_epsilon:
                positions.pop(code, None)
                position_origin.pop(code, None)
                continue
            price_record = day_prices.get(code)
            if (
                price_record is not None
                and pd.notna(price_record["adjusted_close"])
                and float(price_record["adjusted_close"]) > 0
            ):
                valuation_price = float(price_record["adjusted_close"])
                price_source = "CURRENT_ADJUSTED_CLOSE"
                last_close[code] = valuation_price
                last_close_date[code] = trade_date
                last_price_source[code] = (
                    "LAST_AVAILABLE_ADJUSTED_CLOSE"
                )
                last_ts_code[code] = str(price_record["ts_code"])
            else:
                valuation_price = last_close.get(code, math.nan)
                price_source = last_price_source.get(
                    code, "LAST_AVAILABLE_ADJUSTED_CLOSE"
                )
                stale_count += 1
            if math.isnan(valuation_price):
                raise RuntimeError(
                    f"持仓缺少收盘估值价格：{code} {trade_date}"
                )
            position_values.append(
                (
                    code,
                    units,
                    units * valuation_price,
                    price_source,
                    last_ts_code.get(code, ""),
                    last_close_date[code],
                )
            )
        market_value = sum(item[2] for item in position_values)
        stale_market_value = sum(
            item[2]
            for item in position_values
            if item[3] != "CURRENT_ADJUSTED_CLOSE"
        )
        total_nav = cash + market_value
        if cash < -0.01 or total_nav <= 0:
            raise RuntimeError(
                f"P3现金或净值约束失败：cash={cash}, nav={total_nav}"
            )
        for (
            code,
            units,
            value,
            price_source,
            actual_ts_code,
            available_price_date,
        ) in position_values:
            target_record = current_targets.get(code)
            holding_rows.append(
                {
                    "cost_scenario": scenario_name,
                    "is_baseline": is_baseline,
                    "trade_date": trade_date,
                    "canonical_ts_code": code,
                    "ts_code": actual_ts_code,
                    "adjusted_units": units,
                    "valuation_price": value / units,
                    "valuation_price_source": price_source,
                    "position_origin_activity_type": (
                        position_origin.get(code, "TRADE")
                    ),
                    "last_available_price_date": available_price_date,
                    "stale_calendar_days": (
                        trade_date - available_price_date
                    ).days,
                    "position_market_value": value,
                    "actual_weight": value / total_nav,
                    "is_current_target": target_record is not None,
                    "current_target_weight": (
                        float(target_record["target_weight"])
                        if target_record is not None
                        else 0.0
                    ),
                    "current_selection_rank": (
                        target_record["selection_rank"]
                        if target_record is not None
                        else pd.NA
                    ),
                    "current_composite_score": (
                        target_record["composite_score"]
                        if target_record is not None
                        else math.nan
                    ),
                }
            )
        daily_rows.append(
            {
                "cost_scenario": scenario_name,
                "is_baseline": is_baseline,
                "trade_date": trade_date,
                "signal_date_executed": signal_date,
                "pretrade_nav_open": pretrade_nav_open,
                "cash_cny": cash,
                "market_value_cny": market_value,
                "total_nav_cny": total_nav,
                "holding_count": len(position_values),
                "stale_price_position_count": stale_count,
                "stale_price_market_value": stale_market_value,
                "stale_price_weight": (
                    stale_market_value / total_nav
                    if total_nav > 0
                    else math.nan
                ),
                "maximum_stale_calendar_days": max(
                    (
                        (trade_date - item[5]).days
                        for item in position_values
                        if item[3] != "CURRENT_ADJUSTED_CLOSE"
                    ),
                    default=0,
                ),
                "executed_buy_notional": day_buy_notional,
                "executed_sell_notional": day_sell_notional,
                "turnover_notional": (
                    day_buy_notional + day_sell_notional
                ),
                "commission_slippage_cost": day_commission,
                "stamp_duty_cost": day_stamp,
                "total_trading_cost": day_commission + day_stamp,
                "failed_order_count": day_failed_orders,
                "partial_order_count": day_partial_orders,
                "corporate_action_count": day_corporate_actions,
                "corporate_action_value_difference_cny": (
                    day_corporate_action_value_difference
                ),
                "benchmark_code": calendar_row.benchmark_code,
                "benchmark_close": calendar_row.close,
            }
        )

    daily = pd.DataFrame(daily_rows).sort_values("trade_date")
    daily["strategy_nav"] = daily["total_nav_cny"] / initial_capital
    daily["strategy_daily_return"] = (
        daily["strategy_nav"].pct_change().fillna(0.0)
    )
    benchmark_base = float(daily.iloc[0]["benchmark_close"])
    daily["benchmark_nav"] = daily["benchmark_close"] / benchmark_base
    daily["benchmark_daily_return"] = (
        daily["benchmark_nav"].pct_change().fillna(0.0)
    )
    daily["excess_daily_return"] = (
        daily["strategy_daily_return"]
        - daily["benchmark_daily_return"]
    )
    daily["strategy_drawdown"] = (
        daily["strategy_nav"]
        / daily["strategy_nav"].cummax()
        - 1.0
    )
    daily["benchmark_drawdown"] = (
        daily["benchmark_nav"]
        / daily["benchmark_nav"].cummax()
        - 1.0
    )
    daily["relative_nav"] = (
        daily["strategy_nav"] / daily["benchmark_nav"]
    )
    daily["cash_weight"] = daily["cash_cny"] / daily["total_nav_cny"]
    daily["market_value_weight"] = (
        daily["market_value_cny"] / daily["total_nav_cny"]
    )
    daily["turnover_ratio"] = np.where(
        daily["pretrade_nav_open"].notna()
        & (daily["pretrade_nav_open"] > 0),
        daily["turnover_notional"] / daily["pretrade_nav_open"],
        0.0,
    )
    return (
        daily,
        pd.DataFrame(holding_rows),
        pd.DataFrame(order_rows),
        pd.DataFrame(rebalance_rows),
        pd.DataFrame(corporate_action_rows),
    )


def _performance_summary(
    daily: pd.DataFrame,
    orders: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    trading_days = int(config["metrics"]["trading_days_per_year"])
    rows: list[dict[str, Any]] = []
    for scenario_name, group in daily.groupby(
        "cost_scenario", sort=False
    ):
        group = group.sort_values("trade_date")
        returns = group["strategy_daily_return"].iloc[1:]
        benchmark_returns = group["benchmark_daily_return"].iloc[1:]
        excess = returns - benchmark_returns
        years = max((len(group) - 1) / trading_days, 1 / trading_days)
        strategy_total = group.iloc[-1]["strategy_nav"] - 1.0
        benchmark_total = group.iloc[-1]["benchmark_nav"] - 1.0
        strategy_annualized = (
            (1.0 + strategy_total) ** (1.0 / years) - 1.0
        )
        benchmark_annualized = (
            (1.0 + benchmark_total) ** (1.0 / years) - 1.0
        )
        strategy_vol = returns.std(ddof=1) * math.sqrt(trading_days)
        benchmark_vol = (
            benchmark_returns.std(ddof=1) * math.sqrt(trading_days)
        )
        tracking_error = excess.std(ddof=1) * math.sqrt(trading_days)
        scenario_orders = orders.loc[
            orders["cost_scenario"] == scenario_name
        ]
        first_trade_date = group.loc[
            group["signal_date_executed"].notna(), "trade_date"
        ].min()
        rows.append(
            {
                "cost_scenario": scenario_name,
                "is_baseline": bool(group.iloc[0]["is_baseline"]),
                "start_date": group.iloc[0]["trade_date"],
                "end_date": group.iloc[-1]["trade_date"],
                "trading_days": len(group),
                "strategy_total_return": strategy_total,
                "strategy_annualized_return": strategy_annualized,
                "strategy_annualized_volatility": strategy_vol,
                "strategy_sharpe_zero_rf": (
                    returns.mean() / returns.std(ddof=1)
                    * math.sqrt(trading_days)
                    if returns.std(ddof=1) > 0
                    else math.nan
                ),
                "strategy_max_drawdown": group[
                    "strategy_drawdown"
                ].min(),
                "benchmark_total_return": benchmark_total,
                "benchmark_annualized_return": benchmark_annualized,
                "benchmark_annualized_volatility": benchmark_vol,
                "benchmark_max_drawdown": group[
                    "benchmark_drawdown"
                ].min(),
                "annualized_return_difference": (
                    strategy_annualized - benchmark_annualized
                ),
                "terminal_relative_nav": group.iloc[-1][
                    "relative_nav"
                ],
                "tracking_error_annualized": tracking_error,
                "information_ratio": (
                    excess.mean() / excess.std(ddof=1)
                    * math.sqrt(trading_days)
                    if excess.std(ddof=1) > 0
                    else math.nan
                ),
                "average_cash_weight": group["cash_weight"].mean(),
                "maximum_cash_weight_after_first_trade": group.loc[
                    group["trade_date"] >= first_trade_date,
                    "cash_weight",
                ].max(),
                "maximum_stale_price_weight": group[
                    "stale_price_weight"
                ].max(),
                "terminal_stale_price_weight": group.iloc[-1][
                    "stale_price_weight"
                ],
                "terminal_stale_price_market_value": group.iloc[-1][
                    "stale_price_market_value"
                ],
                "maximum_stale_calendar_days": group[
                    "maximum_stale_calendar_days"
                ].max(),
                "cumulative_two_way_turnover": group[
                    "turnover_ratio"
                ].sum(),
                "annualized_two_way_turnover": (
                    group["turnover_ratio"].sum() / years
                ),
                "total_commission_slippage_cost": scenario_orders[
                    "commission_slippage_cost"
                ].sum(),
                "total_stamp_duty_cost": scenario_orders[
                    "stamp_duty_cost"
                ].sum(),
                "total_trading_cost": scenario_orders[
                    "total_trading_cost"
                ].sum(),
                "failed_buy_orders": int(
                    (
                        (scenario_orders["side"] == "BUY")
                        & (scenario_orders["order_status"] == "FAILED")
                    ).sum()
                ),
                "failed_sell_orders": int(
                    (
                        (scenario_orders["side"] == "SELL")
                        & (scenario_orders["order_status"] == "FAILED")
                    ).sum()
                ),
                "partial_cash_constrained_buy_orders": int(
                    (
                        scenario_orders["order_status"]
                        == "PARTIAL_CASH_CONSTRAINT"
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _annual_performance(
    daily: pd.DataFrame,
    orders: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    trading_days = int(config["metrics"]["trading_days_per_year"])
    frame = daily.copy()
    frame["year"] = frame["trade_date"].dt.year
    rows: list[dict[str, Any]] = []
    for (scenario_name, year), group in frame.groupby(
        ["cost_scenario", "year"], sort=True
    ):
        group = group.sort_values("trade_date")
        strategy_return = (
            (1.0 + group["strategy_daily_return"]).prod() - 1.0
        )
        benchmark_return = (
            (1.0 + group["benchmark_daily_return"]).prod() - 1.0
        )
        strategy_path = (
            1.0 + group["strategy_daily_return"]
        ).cumprod()
        benchmark_path = (
            1.0 + group["benchmark_daily_return"]
        ).cumprod()
        year_orders = orders.loc[
            (orders["cost_scenario"] == scenario_name)
            & (orders["trade_date"].dt.year == year)
        ]
        rows.append(
            {
                "cost_scenario": scenario_name,
                "is_baseline": bool(group.iloc[0]["is_baseline"]),
                "year": int(year),
                "trading_days": len(group),
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "return_difference": strategy_return - benchmark_return,
                "strategy_annualized_volatility": group[
                    "strategy_daily_return"
                ].std(ddof=1)
                * math.sqrt(trading_days),
                "benchmark_annualized_volatility": group[
                    "benchmark_daily_return"
                ].std(ddof=1)
                * math.sqrt(trading_days),
                "strategy_max_drawdown_within_year": (
                    strategy_path / strategy_path.cummax() - 1.0
                ).min(),
                "benchmark_max_drawdown_within_year": (
                    benchmark_path / benchmark_path.cummax() - 1.0
                ).min(),
                "average_cash_weight": group["cash_weight"].mean(),
                "two_way_turnover": group["turnover_ratio"].sum(),
                "total_trading_cost": year_orders[
                    "total_trading_cost"
                ].sum(),
                "failed_buy_orders": int(
                    (
                        (year_orders["side"] == "BUY")
                        & (year_orders["order_status"] == "FAILED")
                    ).sum()
                ),
                "failed_sell_orders": int(
                    (
                        (year_orders["side"] == "SELL")
                        & (year_orders["order_status"] == "FAILED")
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _failure_summary(orders: pd.DataFrame) -> pd.DataFrame:
    failed = orders.loc[orders["order_status"] == "FAILED"].copy()
    if failed.empty:
        return pd.DataFrame(
            columns=[
                "cost_scenario",
                "side",
                "failure_reason",
                "failed_orders",
                "desired_notional",
                "executed_notional",
            ]
        )
    return (
        failed.groupby(
            ["cost_scenario", "side", "failure_reason"],
            as_index=False,
            dropna=False,
        )
        .agg(
            failed_orders=("order_id", "count"),
            desired_notional=("desired_order_notional", "sum"),
            executed_notional=("executed_notional", "sum"),
        )
        .sort_values(
            ["cost_scenario", "side", "failed_orders"],
            ascending=[True, True, False],
        )
    )


def _stale_position_summary(
    holdings: pd.DataFrame, final_date: pd.Timestamp
) -> pd.DataFrame:
    stale = holdings.loc[
        holdings["valuation_price_source"]
        != "CURRENT_ADJUSTED_CLOSE"
    ].copy()
    if stale.empty:
        return pd.DataFrame(
            columns=[
                "cost_scenario",
                "canonical_ts_code",
                "first_stale_date",
                "last_stale_date",
                "stale_valuation_days",
                "maximum_stale_calendar_days",
                "last_available_price_date",
                "terminal_position_present",
                "terminal_market_value",
                "terminal_weight",
            ]
        )
    rows: list[dict[str, Any]] = []
    for (scenario, code), group in stale.groupby(
        ["cost_scenario", "canonical_ts_code"], sort=True
    ):
        terminal = group.loc[group["trade_date"] == final_date]
        rows.append(
            {
                "cost_scenario": scenario,
                "canonical_ts_code": code,
                "first_stale_date": group["trade_date"].min(),
                "last_stale_date": group["trade_date"].max(),
                "stale_valuation_days": len(group),
                "maximum_stale_calendar_days": group[
                    "stale_calendar_days"
                ].max(),
                "last_available_price_date": group.loc[
                    group["stale_calendar_days"].idxmax(),
                    "last_available_price_date",
                ],
                "terminal_position_present": not terminal.empty,
                "terminal_market_value": (
                    terminal.iloc[-1]["position_market_value"]
                    if not terminal.empty
                    else 0.0
                ),
                "terminal_weight": (
                    terminal.iloc[-1]["actual_weight"]
                    if not terminal.empty
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def build_p3() -> dict[str, Any]:
    started_at = datetime.now(UTC)
    config = load_config()
    p2_manifest = _validate_config(config)
    stamp_policy = _load_stamp_policy(config)
    corporate_actions = _load_corporate_actions(config)
    legacy_performance, legacy_archive = _load_legacy_performance(config)
    protected_paths = list(config["protected_p3_inputs"])
    _log("记录P2/P1输入哈希")
    before = _input_snapshot(protected_paths)
    _log("核对原始数据P1基线SHA-256")
    original_before = _original_data_snapshot(config)
    if not bool(original_before["matches_p1_record"].all()):
        raise RuntimeError("P3修复前原始数据已偏离P1哈希基线")

    _log("构建等权复合因子与Top-100目标持仓")
    signals, targets = _build_composite_signals(config)
    benchmark = pd.read_parquet(
        absolute(config["inputs"]["benchmark_daily"])
    )
    benchmark["trade_date"] = pd.to_datetime(benchmark["trade_date"])
    benchmark = benchmark.loc[
        (benchmark["trade_date"] >= signals["signal_date"].min())
        & (
            benchmark["trade_date"]
            <= pd.Timestamp(config["project"]["research_end"])
        )
    ].sort_values("trade_date")
    if set(benchmark["benchmark_code"]) != {"000985.CSI"}:
        raise RuntimeError("P3宽基不是冻结的中证全指000985.CSI")
    schedule = _build_schedule(
        signals,
        benchmark,
        pd.Timestamp(config["project"]["research_end"]),
    )
    targets = targets.merge(
        schedule,
        on="signal_date",
        how="left",
        validate="many_to_one",
    )
    _write_parquet_atomic(
        signals, config["outputs"]["composite_signals"]
    )
    _write_parquet_atomic(
        targets, config["outputs"]["target_holdings"]
    )
    _write_csv_atomic(schedule, config["outputs"]["rebalance_schedule"])

    _log("加载目标证券日频估值与47个调仓日成交状态")
    prices, executions, benchmark = _load_market_inputs(
        config, targets, schedule, benchmark, corporate_actions
    )
    corporate_action_references = (
        _prepare_corporate_action_references(
            corporate_actions, prices
        )
    )

    daily_frames: list[pd.DataFrame] = []
    holding_frames: list[pd.DataFrame] = []
    order_frames: list[pd.DataFrame] = []
    rebalance_frames: list[pd.DataFrame] = []
    corporate_action_frames: list[pd.DataFrame] = []
    for scenario in config["cost_scenarios"]:
        _log(f"运行成本情景 {scenario['scenario']}")
        daily, holdings, orders, rebalances, action_events = (
            _simulate_scenario(
                config,
                scenario,
                targets,
                schedule,
                benchmark,
                prices,
                executions,
                stamp_policy,
                corporate_action_references,
            )
        )
        daily_frames.append(daily)
        holding_frames.append(holdings)
        order_frames.append(orders)
        rebalance_frames.append(rebalances)
        corporate_action_frames.append(action_events)

    daily_portfolio = pd.concat(daily_frames, ignore_index=True)
    actual_holdings = pd.concat(holding_frames, ignore_index=True)
    orders = pd.concat(order_frames, ignore_index=True)
    rebalances = pd.concat(rebalance_frames, ignore_index=True)
    corporate_action_events = pd.concat(
        corporate_action_frames, ignore_index=True
    )
    failed_orders = orders.loc[
        orders["order_status"] == "FAILED"
    ].copy()
    cash_ledger = daily_portfolio[
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
        ("daily_portfolio", daily_portfolio),
        ("actual_holdings", actual_holdings),
        ("orders", orders),
        ("failed_orders", failed_orders),
        ("cash_ledger", cash_ledger),
        ("corporate_action_events", corporate_action_events),
    ):
        _write_parquet_atomic(frame, config["outputs"][key])
    _write_csv_atomic(rebalances, config["outputs"]["rebalance_summary"])

    performance = _performance_summary(daily_portfolio, orders, config)
    corporate_action_impact = _corporate_action_impact(
        legacy_performance, performance
    )
    annual = _annual_performance(daily_portfolio, orders, config)
    failure_summary = _failure_summary(orders)
    stale_summary = _stale_position_summary(
        actual_holdings,
        pd.Timestamp(config["project"]["research_end"]),
    )
    _write_csv_atomic(
        performance, config["outputs"]["performance_summary"]
    )
    _write_csv_atomic(
        annual, config["outputs"]["annual_performance"]
    )
    _write_csv_atomic(
        performance.sort_values(
            "cost_scenario",
            key=lambda values: values.map(
                {
                    "STRESS_5BPS": 5,
                    "BASE_10BPS": 10,
                    "STRESS_20BPS": 20,
                }
            ),
        ),
        config["outputs"]["cost_scenario_comparison"],
    )
    _write_csv_atomic(
        failure_summary, config["outputs"]["failure_reason_summary"]
    )
    _write_csv_atomic(
        stale_summary, config["outputs"]["stale_position_summary"]
    )
    _write_csv_atomic(
        corporate_action_impact,
        config["outputs"]["corporate_action_impact"],
    )

    _log("复核P2/P1输入哈希")
    after = _input_snapshot(protected_paths)
    comparison = pd.DataFrame(
        [
            {
                "path": relative_path,
                "size_bytes_before": before[relative_path]["size_bytes"],
                "size_bytes_after": after[relative_path]["size_bytes"],
                "sha256_before": before[relative_path]["sha256"],
                "sha256_after": after[relative_path]["sha256"],
                "match": before[relative_path] == after[relative_path],
            }
            for relative_path in protected_paths
        ]
    )
    _write_csv_atomic(comparison, config["outputs"]["p3_input_hashes"])
    if not bool(comparison["match"].all()):
        raise RuntimeError("P3构建期间受保护输入发生变化")
    _log("复核原始数据SHA-256未变化")
    original_after = _original_data_snapshot(config)
    original_check = (
        original_before.rename(
            columns={
                "current_size_bytes": "size_bytes_before",
                "current_sha256": "sha256_before",
                "matches_p1_record": "matches_p1_before",
            }
        )
        .merge(
            original_after.rename(
                columns={
                    "current_size_bytes": "size_bytes_after",
                    "current_sha256": "sha256_after",
                    "matches_p1_record": "matches_p1_after",
                }
            ),
            on=[
                "path",
                "p1_recorded_size_bytes",
                "p1_recorded_sha256",
            ],
            how="outer",
            validate="one_to_one",
        )
    )
    original_check["unchanged_during_build"] = (
        original_check["size_bytes_before"]
        == original_check["size_bytes_after"]
    ) & (
        original_check["sha256_before"]
        == original_check["sha256_after"]
    )
    _write_csv_atomic(
        original_check, config["outputs"]["original_data_hash_check"]
    )
    if not bool(
        original_check[
            [
                "matches_p1_before",
                "matches_p1_after",
                "unchanged_during_build",
            ]
        ].all().all()
    ):
        raise RuntimeError("P3修复期间原始数据SHA-256发生变化")

    generated_keys = [
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
    ]
    output_hashes = {
        key: _sha256(absolute(config["outputs"][key]))
        for key in generated_keys
    }
    completed_at = datetime.now(UTC)
    baseline = performance.loc[performance["is_baseline"]].iloc[0]
    baseline_action = corporate_action_events.loc[
        corporate_action_events["is_baseline"]
    ].iloc[0]
    baseline_impact = corporate_action_impact.loc[
        corporate_action_impact["is_baseline"]
    ].iloc[0]
    manifest: dict[str, Any] = {
        "stage": "P3_COMPOSITE_AND_BENCHMARK_BACKTEST",
        "builder_version": BUILDER_VERSION,
        "status": "BUILT_PENDING_AUDIT",
        "project_root_policy": (
            "runtime_discovery_and_project_relative_config_only"
        ),
        "sample": {
            "research_start": config["project"]["research_start"],
            "research_end": config["project"]["research_end"],
            "validation_start_guard": config["project"][
                "validation_start"
            ],
            "oos_start_guard": config["project"]["oos_start"],
            "first_signal_date": str(signals["signal_date"].min().date()),
            "last_signal_date": str(signals["signal_date"].max().date()),
            "first_portfolio_date": str(
                daily_portfolio["trade_date"].min().date()
            ),
            "last_portfolio_date": str(
                daily_portfolio["trade_date"].max().date()
            ),
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
        "config_sha256": _sha256(PROJECT_ROOT / CONFIG_PATH),
        "p2_manifest_sha256": _sha256(
            absolute(config["inputs"]["p2_run_manifest"])
        ),
        "p2_status_at_start": p2_manifest["status"],
        "frozen_methodology": {
            "composite": (
                "(bm_proxy_z + momentum_12_1_z + lowvol_60_z) / 3"
            ),
            "selection": "top 100, composite descending",
            "target_weight": (
                "equal weight, single-name cap 2%, gross cap 100%"
            ),
            "execution": (
                "next market trading day open; sells before buys"
            ),
            "blocked_buy": "unexecuted target allocation remains cash",
            "blocked_sell": "old adjusted units remain held",
            "valuation": (
                "adjusted close; last available adjusted close only "
                "for valuation, never execution"
            ),
            "position_units": (
                "fractional adjusted-price units; no board-lot rounding"
            ),
            "corporate_action_quantity": (
                "old adjusted units x old adj_factor = old raw-share "
                "quantity; successor raw-share quantity = old raw-share "
                "quantity x exchange_ratio; successor adjusted units = "
                "successor raw-share quantity / first successor adj_factor"
            ),
            "corporate_action_valuation": (
                "value-preserving carry from effective date until the "
                "successor has an available market price"
            ),
            "corporate_action_activity": (
                "CORPORATE_ACTION, never TRADE; zero commission, "
                "slippage and stamp duty"
            ),
            "cash_constraint": (
                "buy demands scaled pro rata including blocked demand"
            ),
            "terminal_liquidation": False,
        },
        "counts": {
            "signal_months": int(signals["signal_date"].nunique()),
            "target_rows": len(targets),
            "scheduled_rebalances": int(
                schedule["scheduled_trade_date"].notna().sum()
            ),
            "out_of_scope_signals": int(
                schedule["scheduled_trade_date"].isna().sum()
            ),
            "cost_scenarios": daily_portfolio[
                "cost_scenario"
            ].nunique(),
            "daily_portfolio_rows": len(daily_portfolio),
            "actual_holding_rows": len(actual_holdings),
            "orders": len(orders),
            "failed_orders": len(failed_orders),
            "corporate_action_events": len(corporate_action_events),
        },
        "baseline_result": {
            "cost_scenario": baseline["cost_scenario"],
            "strategy_total_return": baseline[
                "strategy_total_return"
            ],
            "strategy_annualized_return": baseline[
                "strategy_annualized_return"
            ],
            "strategy_max_drawdown": baseline[
                "strategy_max_drawdown"
            ],
            "benchmark_total_return": baseline[
                "benchmark_total_return"
            ],
            "benchmark_annualized_return": baseline[
                "benchmark_annualized_return"
            ],
            "benchmark_max_drawdown": baseline[
                "benchmark_max_drawdown"
            ],
            "annualized_return_difference": baseline[
                "annualized_return_difference"
            ],
        },
        "corporate_action_result": {
            "old_ts_code": baseline_action["old_ts_code"],
            "successor_ts_code": baseline_action["successor_ts_code"],
            "effective_date": str(
                pd.Timestamp(
                    baseline_action["effective_date"]
                ).date()
            ),
            "old_adjusted_units_before": float(
                baseline_action["old_adjusted_units_before"]
            ),
            "old_share_quantity_before": float(
                baseline_action["old_share_quantity_before"]
            ),
            "exchange_ratio": float(
                baseline_action["exchange_ratio"]
            ),
            "successor_share_quantity_after": float(
                baseline_action["successor_share_quantity_after"]
            ),
            "successor_adjusted_units_after": float(
                baseline_action["successor_adjusted_units_after"]
            ),
            "portfolio_value_before_action_cny": float(
                baseline_action[
                    "portfolio_value_before_action_cny"
                ]
            ),
            "portfolio_value_after_action_cny": float(
                baseline_action[
                    "portfolio_value_after_action_cny"
                ]
            ),
            "portfolio_value_difference_cny": float(
                baseline_action["portfolio_value_difference_cny"]
            ),
            "successor_first_price_date": str(
                pd.Timestamp(
                    baseline_action["successor_first_price_date"]
                ).date()
            ),
            "automatic_successor_sale": False,
        },
        "legacy_comparison": {
            **legacy_archive,
            "baseline_annualized_return_before": float(
                baseline_impact["annualized_return_before"]
            ),
            "baseline_annualized_return_after": float(
                baseline_impact["annualized_return_after"]
            ),
            "baseline_annualized_return_difference": float(
                baseline_impact["annualized_return_difference"]
            ),
            "baseline_total_return_before": float(
                baseline_impact["total_return_before"]
            ),
            "baseline_total_return_after": float(
                baseline_impact["total_return_after"]
            ),
            "baseline_total_return_difference": float(
                baseline_impact["total_return_difference"]
            ),
            "baseline_max_drawdown_before": float(
                baseline_impact["max_drawdown_before"]
            ),
            "baseline_max_drawdown_after": float(
                baseline_impact["max_drawdown_after"]
            ),
            "baseline_max_drawdown_difference": float(
                baseline_impact["max_drawdown_difference"]
            ),
        },
        "original_data_integrity": {
            "file_count": len(original_check),
            "p1_mismatch_count_before": int(
                (~original_check["matches_p1_before"]).sum()
            ),
            "p1_mismatch_count_after": int(
                (~original_check["matches_p1_after"]).sum()
            ),
            "changed_during_build_count": int(
                (~original_check["unchanged_during_build"]).sum()
            ),
            "aggregate_sha256_before": _aggregate_path_hash(
                original_check, "sha256_before"
            ),
            "aggregate_sha256_after": _aggregate_path_hash(
                original_check, "sha256_after"
            ),
        },
        "protected_inputs_all_match": bool(comparison["match"].all()),
        "output_sha256": output_hashes,
        "scope_guards": {
            "validation_period_run": False,
            "oos_period_read_or_run": False,
            "p4_freeze_protocol_generated": False,
            "p5_oos_run": False,
            "parameters_tuned_on_validation_or_oos": False,
        },
        "disclosures": [
            (
                "使用供应商历史PB构造1/PB代理，未自行重建严格 "
                "point-in-time book equity，供应商历史修订政策未完全核验。"
            ),
            (
                "持仓使用可分割复权价格单位，不模拟A股100股整手约束；"
                "上一可得复权收盘价仅用于估值，绝不用于成交。"
            ),
            (
                "600270.SH 于2018-12-28按人工公司行动表换股为"
                "601598.SH；原始股数按3.8225倍转换，假定不行使"
                "现金选择权，允许非整数股，不计佣金、滑点或印花税；"
                "601598.SH首个可得市场价格出现前使用价值连续的"
                "公司行动承接估值。"
            ),
        ],
    }
    _write_json_atomic(manifest, config["outputs"]["p3_run_manifest"])
    _log(
        "构建完成："
        f"signals={manifest['counts']['signal_months']}，"
        f"rebalances={manifest['counts']['scheduled_rebalances']}，"
        f"orders={manifest['counts']['orders']}，"
        f"failed={manifest['counts']['failed_orders']}；等待P3审计"
    )
    return manifest
