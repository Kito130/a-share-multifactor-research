from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import duckdb
import matplotlib
import numpy as np
import pandas as pd
import pyarrow
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from a_share_p2.research import _factor_panel_query
from a_share_p3.build import (
    _annual_performance,
    _build_schedule,
    _failure_summary,
    _load_corporate_actions,
    _load_market_inputs,
    _load_stamp_policy,
    _prepare_corporate_action_references,
    _simulate_scenario,
)
from a_share_p5.build import _period_performance, _runtime_config
from a_share_p5.config import load_config as load_p5_config

from .config import PROJECT_ROOT, absolute, load_config


BUILDER_VERSION = "p6.1"
P6_STATUS = "PROJECT_COMPLETE_WITH_DISCLOSED_LIMITATIONS"
BASELINE_SCENARIO = "BASE_10BPS"
OOS_PERIOD = "FINAL_OOS_2022_2025"
DELIST_CODES = ("000413.SZ", "000666.SZ", "000671.SZ")
FACTOR_COLUMNS = ("bm_proxy_z", "momentum_12_1_z", "lowvol_60_z")


def _log(message: str) -> None:
    print(f"[P6] {message}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(paths: Iterable[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative in paths:
        path = absolute(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Required protected input is missing: {relative}")
        result[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return result


def _write_csv(frame: pd.DataFrame, relative: str) -> None:
    path = absolute(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _write_parquet(frame: pd.DataFrame, relative: str) -> None:
    path = absolute(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _write_text(text: str, relative: str) -> None:
    path = absolute(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_json(payload: dict[str, Any], relative: str) -> None:
    _write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        relative,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _load_yaml(relative: str) -> dict[str, Any]:
    with absolute(relative).open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle)
    return payload


def _load_json(relative: str) -> dict[str, Any]:
    return json.loads(absolute(relative).read_text(encoding="utf-8"))


def _p5_output_hash_check(
    control: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    p5_control = load_p5_config()
    manifest = _load_json(control["inputs"]["p5_manifest"])
    if manifest.get("status") != control["project"]["p5_status_required"]:
        raise RuntimeError(
            "P5 has not been accepted with the required disclosed limitations"
        )
    if manifest.get("audit", {}).get("fail_count") != 0:
        raise RuntimeError("P5 still contains audit failures")
    expected_freeze = control["project"]["expected_frozen_sha256"]
    if (
        manifest.get("frozen_config", {}).get("sha256") != expected_freeze
        and manifest.get("frozen_config", {}).get("frozen_config_sha256")
        != expected_freeze
    ):
        raise RuntimeError("P5 manifest freeze hash does not match P6 protocol")
    rows: list[dict[str, Any]] = []
    for key, expected_hash in manifest["output_sha256"].items():
        if key not in p5_control["outputs"]:
            raise KeyError(f"P5 output mapping missing for manifest key: {key}")
        relative = p5_control["outputs"][key]
        path = absolute(relative)
        actual_hash = _sha256(path) if path.is_file() else None
        rows.append(
            {
                "output_key": key,
                "relative_path": relative,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "matches": actual_hash == expected_hash,
            }
        )
    frame = pd.DataFrame(rows)
    if not bool(frame["matches"].all()):
        bad = frame.loc[~frame["matches"], "relative_path"].tolist()
        raise RuntimeError(f"P5 original outputs changed before P6: {bad}")
    return frame, manifest, p5_control


def _preflight(
    control: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    dict[str, dict[str, Any]],
]:
    if control["project"]["preserve_p5_original_results"] is not True:
        raise RuntimeError("P6 must preserve P5 original results")
    if control["project"]["experiment_classification"] != "POST_OOS_ROBUSTNESS":
        raise RuntimeError("P6 experiments must be classified post-OOS")
    frozen = _load_yaml(control["inputs"]["frozen_config"])
    freeze_hash = _sha256(absolute(control["inputs"]["frozen_config"]))
    if freeze_hash != control["project"]["expected_frozen_sha256"]:
        raise RuntimeError("Frozen configuration SHA-256 changed")
    if frozen.get("status") != "FROZEN_AFTER_VALIDATION":
        raise RuntimeError("P6 requires the P4 frozen configuration")
    hash_check, p5_manifest, _ = _p5_output_hash_check(control)
    intent = _load_json(control["inputs"]["p5_run_intent"])
    if (
        intent.get("attempt_number") != 1
        or intent.get("status") != "COMPLETED"
    ):
        raise RuntimeError("P5 one-shot run intent is not completed attempt 1")
    protected = _snapshot(control["protected_p6_inputs"])
    p5_gate = {
        "control": load_p5_config(),
        "frozen": frozen,
        "p2_config": _load_yaml(control["inputs"]["p2_config"]),
    }
    runtime = _runtime_config(p5_gate)
    runtime["inputs"]["p2_single_factor_panel"] = control["inputs"][
        "p5_factor_panel"
    ]
    return frozen, runtime, p5_manifest, hash_check, protected


def _sample_zscore(values: pd.Series) -> pd.Series:
    standard_deviation = values.std(ddof=1)
    if pd.isna(standard_deviation) or standard_deviation <= 0:
        return pd.Series(np.nan, index=values.index)
    return (values - values.mean()) / standard_deviation


def _industry_neutralize(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    industry = result["industry_code"].fillna("UNKNOWN").astype(str)
    for column in FACTOR_COLUMNS:
        demeaned = result[column] - result.groupby(
            [result["signal_date"], industry], sort=False
        )[column].transform("mean")
        result[column] = demeaned.groupby(
            result["signal_date"], sort=False
        ).transform(_sample_zscore)
    return result


def _size_neutralize(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_log_size"] = np.log(result["total_mv_cny"].where(
        result["total_mv_cny"] > 0
    ))
    for column in FACTOR_COLUMNS:
        residual = pd.Series(np.nan, index=result.index, dtype=float)
        for _, group in result.groupby("signal_date", sort=False):
            valid = group[[column, "_log_size"]].dropna()
            if len(valid) < 3 or valid["_log_size"].std(ddof=1) <= 0:
                continue
            design = np.column_stack(
                [np.ones(len(valid)), valid["_log_size"].to_numpy()]
            )
            coefficients = np.linalg.lstsq(
                design, valid[column].to_numpy(), rcond=None
            )[0]
            residual.loc[valid.index] = (
                valid[column].to_numpy() - design @ coefficients
            )
        result[column] = residual.groupby(
            result["signal_date"], sort=False
        ).transform(_sample_zscore)
    return result.drop(columns="_log_size")


def _build_signals(
    factor_panel: pd.DataFrame,
    frozen: dict[str, Any],
    experiment_id: str,
    experiment: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    panel = factor_panel.copy()
    panel["signal_date"] = pd.to_datetime(panel["signal_date"])
    eligible = panel.loc[panel["universe_eligible"].fillna(False)].copy()
    eligible_before = len(eligible)
    if experiment_id == "MICROCAP_EXCLUDE_BOTTOM20":
        quantile = float(experiment["excluded_market_cap_quantile"])
        cutoffs = eligible.groupby("signal_date")["total_mv_cny"].transform(
            lambda values: values.quantile(quantile)
        )
        eligible = eligible.loc[eligible["total_mv_cny"] >= cutoffs].copy()
    elif experiment_id == "INDUSTRY_NEUTRAL":
        eligible = _industry_neutralize(eligible)
    elif experiment_id == "LOG_SIZE_NEUTRAL":
        eligible = _size_neutralize(eligible)
    eligible = eligible.dropna(subset=list(FACTOR_COLUMNS)).copy()
    weights = frozen["composite"]
    eligible["composite_score"] = (
        eligible["bm_proxy_z"] * float(weights["bm_proxy_z_weight"])
        + eligible["momentum_12_1_z"]
        * float(weights["momentum_12_1_z_weight"])
        + eligible["lowvol_60_z"] * float(weights["lowvol_60_z_weight"])
    )
    eligible = eligible.sort_values(
        ["signal_date", "composite_score", "canonical_ts_code"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    eligible["selection_rank"] = (
        eligible.groupby("signal_date", sort=False).cumcount() + 1
    )
    maximum = int(frozen["portfolio"]["maximum_holdings"])
    eligible["is_selected"] = eligible["selection_rank"] <= maximum
    eligible["target_weight"] = 0.0
    counts = eligible.loc[eligible["is_selected"]].groupby(
        "signal_date"
    )["canonical_ts_code"].transform("count")
    eligible.loc[eligible["is_selected"], "target_weight"] = np.minimum(
        float(frozen["portfolio"]["target_gross_weight"])
        / counts.astype(float),
        float(frozen["portfolio"]["maximum_single_name_target_weight"]),
    )
    eligible["composite_definition"] = (
        "P6_POST_OOS_" + experiment_id
    )
    targets = eligible.loc[eligible["is_selected"]].copy()
    diagnostics = {
        "eligible_rows_before_variant": eligible_before,
        "eligible_rows_after_variant": len(eligible),
        "signal_months": int(eligible["signal_date"].nunique()),
        "minimum_selected": int(
            targets.groupby("signal_date").size().min()
        ),
        "maximum_selected": int(
            targets.groupby("signal_date").size().max()
        ),
        "average_selected": float(
            targets.groupby("signal_date").size().mean()
        ),
    }
    return eligible.reset_index(drop=True), targets.reset_index(drop=True), diagnostics


def _variant_factor_panel(
    runtime: dict[str, Any],
    experiment_id: str,
    experiment: dict[str, Any],
    baseline_panel: pd.DataFrame,
) -> pd.DataFrame:
    if experiment_id not in {"MOMENTUM_9_1", "LOWVOL_120"}:
        return baseline_panel
    variant = copy.deepcopy(runtime)
    for key in (
        "momentum_recent_skip_observations",
        "momentum_lookback_observations",
        "lowvol_window_observations",
    ):
        variant["factors"][key] = int(experiment[key])
    _log(f"computing factor panel for {experiment_id}")
    with duckdb.connect() as connection:
        panel = connection.execute(_factor_panel_query(variant)).fetchdf()
    return panel


def _scenario_for_baseline(frozen: dict[str, Any]) -> dict[str, Any]:
    scenarios = [
        item for item in frozen["cost_scenarios"]
        if item["scenario"] == BASELINE_SCENARIO
    ]
    if len(scenarios) != 1 or not scenarios[0]["is_baseline"]:
        raise RuntimeError("Frozen baseline cost scenario is invalid")
    return copy.deepcopy(scenarios[0])


def _run_variant(
    experiment_id: str,
    experiment: dict[str, Any],
    factor_panel: pd.DataFrame,
    runtime: dict[str, Any],
    frozen: dict[str, Any],
    benchmark: pd.DataFrame,
    actions: pd.DataFrame,
    stamp_policy: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    _log(f"running robustness variant {experiment_id}")
    signals, targets, diagnostics = _build_signals(
        factor_panel, frozen, experiment_id, experiment
    )
    schedule = _build_schedule(
        signals, benchmark, pd.Timestamp(runtime["project"]["research_end"])
    )
    prices, executions, benchmark_input = _load_market_inputs(
        runtime, targets, schedule, benchmark.copy(), actions
    )
    references = _prepare_corporate_action_references(actions, prices)
    scenario = _scenario_for_baseline(frozen)
    daily, holdings, orders, rebalances, corporate_events = _simulate_scenario(
        runtime,
        scenario,
        targets,
        schedule,
        benchmark_input,
        prices,
        executions,
        stamp_policy,
        references,
    )
    for frame, column in (
        (daily, "trade_date"),
        (holdings, "trade_date"),
        (orders, "trade_date"),
        (rebalances, "trade_date"),
        (corporate_events, "effective_date"),
    ):
        if not frame.empty:
            frame[column] = pd.to_datetime(frame[column])
    period = _period_performance(daily, orders, runtime, frozen)
    oos = period.loc[period["period"] == OOS_PERIOD].iloc[0]
    annual = _annual_performance(daily, orders, runtime)
    annual_oos = annual.loc[annual["year"].between(2022, 2025)].copy()
    prefix = f"data/processed/p6_robustness/{experiment_id.lower()}"
    _write_parquet(signals, f"{prefix}_signals.parquet")
    _write_parquet(targets, f"{prefix}_targets.parquet")
    _write_parquet(daily, f"{prefix}_daily.parquet")
    _write_parquet(holdings, f"{prefix}_holdings.parquet")
    _write_parquet(orders, f"{prefix}_orders.parquet")
    _write_parquet(rebalances, f"{prefix}_rebalances.parquet")
    _write_parquet(corporate_events, f"{prefix}_corporate_actions.parquet")
    if experiment_id in {"MOMENTUM_9_1", "LOWVOL_120"}:
        _write_parquet(factor_panel, f"{prefix}_factor_panel.parquet")
    metric = {
        "experiment_id": experiment_id,
        "category": experiment["category"],
        "classification": "POST_OOS_ROBUSTNESS",
        "implementation": experiment["implementation"],
        "status": "PASS",
        "source": "P6_BASE_10BPS_RERUN",
        "cost_scenario": BASELINE_SCENARIO,
        "start_date": str(oos["start_date"])[:10],
        "end_date": str(oos["end_date"])[:10],
        "trading_days": int(oos["trading_days"]),
        "strategy_total_return": float(oos["strategy_total_return"]),
        "strategy_annualized_return": float(oos["strategy_annualized_return"]),
        "strategy_annualized_volatility": float(
            oos["strategy_annualized_volatility"]
        ),
        "strategy_sharpe_zero_rf": float(oos["strategy_sharpe_zero_rf"]),
        "strategy_max_drawdown": float(
            oos["strategy_max_drawdown_within_period"]
        ),
        "benchmark_total_return": float(oos["benchmark_total_return"]),
        "benchmark_annualized_return": float(
            oos["benchmark_annualized_return"]
        ),
        "annualized_return_difference": float(
            oos["annualized_return_difference"]
        ),
        "information_ratio": float(oos["information_ratio"]),
        "two_way_turnover": float(oos["two_way_turnover"]),
        "total_trading_cost": float(oos["total_trading_cost"]),
        "failed_buy_orders": int(oos["failed_buy_orders"]),
        "failed_sell_orders": int(oos["failed_sell_orders"]),
        "terminal_stale_price_weight": float(
            oos["terminal_stale_price_weight"]
        ),
        "p5_original_modified": False,
        **diagnostics,
        "artifact_prefix": prefix,
    }
    return metric, daily, annual_oos


def _reference_metrics(
    control: dict[str, Any],
    frozen_hash: str,
) -> list[dict[str, Any]]:
    performance = pd.read_csv(absolute(control["inputs"]["p5_oos_performance"]))
    mapping = {
        "P5_BASELINE_REFERENCE": BASELINE_SCENARIO,
        "COST_5BPS_REFERENCE": "STRESS_5BPS",
        "COST_20BPS_REFERENCE": "STRESS_20BPS",
    }
    experiments = {
        item["experiment_id"]: item for item in control["experiments"]
    }
    rows: list[dict[str, Any]] = []
    for experiment_id, scenario in mapping.items():
        source = performance.loc[
            performance["cost_scenario"] == scenario
        ].iloc[0]
        rows.append(
            {
                "experiment_id": experiment_id,
                "category": experiments[experiment_id]["category"],
                "classification": "POST_OOS_ROBUSTNESS",
                "implementation": "reuse_p5_no_rerun",
                "status": "PASS",
                "source": "P5_FROZEN_REFERENCE",
                "cost_scenario": scenario,
                "start_date": source["start_date"],
                "end_date": source["end_date"],
                "trading_days": int(source["trading_days"]),
                "strategy_total_return": float(source["strategy_total_return"]),
                "strategy_annualized_return": float(
                    source["strategy_annualized_return"]
                ),
                "strategy_annualized_volatility": float(
                    source["strategy_annualized_volatility"]
                ),
                "strategy_sharpe_zero_rf": float(
                    source["strategy_sharpe_zero_rf"]
                ),
                "strategy_max_drawdown": float(
                    source["strategy_max_drawdown_within_period"]
                ),
                "benchmark_total_return": float(
                    source["benchmark_total_return"]
                ),
                "benchmark_annualized_return": float(
                    source["benchmark_annualized_return"]
                ),
                "annualized_return_difference": float(
                    source["annualized_return_difference"]
                ),
                "information_ratio": float(source["information_ratio"]),
                "two_way_turnover": float(source["two_way_turnover"]),
                "total_trading_cost": float(source["total_trading_cost"]),
                "failed_buy_orders": int(source["failed_buy_orders"]),
                "failed_sell_orders": int(source["failed_sell_orders"]),
                "terminal_stale_price_weight": float(
                    source["terminal_stale_price_weight"]
                ),
                "p5_original_modified": False,
                "eligible_rows_before_variant": None,
                "eligible_rows_after_variant": None,
                "signal_months": None,
                "minimum_selected": None,
                "maximum_selected": None,
                "average_selected": None,
                "artifact_prefix": "results/p5_oos",
                "frozen_config_sha256": frozen_hash,
            }
        )
    return rows


def _delisting_sensitivity(
    control: dict[str, Any],
) -> pd.DataFrame:
    performance = pd.read_csv(absolute(control["inputs"]["p5_oos_performance"]))
    holdings = pd.read_parquet(
        absolute(control["inputs"]["p5_oos_actual_holdings"])
    )
    holdings["trade_date"] = pd.to_datetime(holdings["trade_date"])
    last_date = holdings["trade_date"].max()
    terminal = holdings.loc[
        (holdings["trade_date"] == last_date)
        & holdings["canonical_ts_code"].isin(DELIST_CODES)
        & (holdings["adjusted_units"].abs() > 1e-12)
    ].copy()
    rows: list[dict[str, Any]] = []
    for performance_row in performance.itertuples(index=False):
        scenario_terminal = terminal.loc[
            terminal["cost_scenario"] == performance_row.cost_scenario
        ]
        carrying_value = float(
            scenario_terminal["position_market_value"].sum()
        )
        terminal_nav = carrying_value / float(
            performance_row.terminal_stale_price_weight
        )
        starting_nav = terminal_nav / (
            1.0 + float(performance_row.strategy_total_return)
        )
        years = float(performance_row.trading_days) / 252.0
        for rate in next(
            item["recovery_rates"]
            for item in control["experiments"]
            if item["experiment_id"] == "DELIST_TERMINAL_RECOVERY"
        ):
            adjusted_terminal = terminal_nav - carrying_value * (1.0 - rate)
            total_return = adjusted_terminal / starting_nav - 1.0
            annualized = (1.0 + total_return) ** (1.0 / years) - 1.0
            rows.append(
                {
                    "experiment_id": "DELIST_TERMINAL_RECOVERY",
                    "classification": "POST_OOS_ROBUSTNESS",
                    "method": "TERMINAL_VALUATION_SENSITIVITY_NO_PATH_RERUN",
                    "cost_scenario": performance_row.cost_scenario,
                    "is_baseline": bool(performance_row.is_baseline),
                    "recovery_rate": float(rate),
                    "affected_codes": "|".join(
                        sorted(scenario_terminal["canonical_ts_code"].unique())
                    ),
                    "affected_position_count": len(scenario_terminal),
                    "terminal_carrying_value_cny": carrying_value,
                    "original_terminal_nav_cny": terminal_nav,
                    "adjusted_terminal_nav_cny": adjusted_terminal,
                    "adjusted_strategy_total_return": total_return,
                    "adjusted_strategy_annualized_return": annualized,
                    "total_return_change_vs_p5": (
                        total_return
                        - float(performance_row.strategy_total_return)
                    ),
                    "annualized_return_change_vs_p5": (
                        annualized
                        - float(performance_row.strategy_annualized_return)
                    ),
                    "p5_original_modified": False,
                    "interpretation_limit": (
                        "No delist date, liquidation date, interim path, "
                        "orders, or taxes were inferred."
                    ),
                }
            )
    return pd.DataFrame(rows)


def _monthly_diagnostics(
    daily: pd.DataFrame,
    orders: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = daily.loc[
        (daily["cost_scenario"] == BASELINE_SCENARIO)
        & daily["trade_date"].between("2022-01-01", "2025-12-31")
    ].copy()
    frame["month"] = frame["trade_date"].dt.to_period("M").astype(str)
    period_orders = orders.loc[
        (orders["cost_scenario"] == BASELINE_SCENARIO)
        & orders["trade_date"].between("2022-01-01", "2025-12-31")
    ].copy()
    period_orders["month"] = (
        period_orders["trade_date"].dt.to_period("M").astype(str)
    )
    monthly = (
        frame.groupby("month", as_index=False)
        .agg(
            start_date=("trade_date", "min"),
            end_date=("trade_date", "max"),
            trading_days=("trade_date", "size"),
            strategy_return=(
                "strategy_daily_return",
                lambda values: (1.0 + values).prod() - 1.0,
            ),
            benchmark_return=(
                "benchmark_daily_return",
                lambda values: (1.0 + values).prod() - 1.0,
            ),
            turnover=("turnover_ratio", "sum"),
            trading_cost=("total_trading_cost", "sum"),
            maximum_stale_weight=("stale_price_weight", "max"),
            average_cash_weight=("cash_weight", "mean"),
        )
    )
    monthly["excess_return"] = (
        monthly["strategy_return"] - monthly["benchmark_return"]
    )
    failed = period_orders.loc[period_orders["order_status"] == "FAILED"]
    failed_monthly = (
        failed.groupby("month", as_index=False)
        .agg(failed_orders=("order_id", "count"))
    )
    monthly = monthly.merge(failed_monthly, on="month", how="left")
    monthly["failed_orders"] = monthly["failed_orders"].fillna(0).astype(int)
    worst_strategy = monthly.nsmallest(10, "strategy_return").copy()
    worst_strategy.insert(0, "ranking_type", "WORST_STRATEGY_RETURN")
    worst_excess = monthly.nsmallest(10, "excess_return").copy()
    worst_excess.insert(0, "ranking_type", "WORST_EXCESS_RETURN")
    worst = pd.concat([worst_strategy, worst_excess], ignore_index=True)

    returns = frame.set_index("trade_date")["strategy_daily_return"]
    nav = (1.0 + returns).cumprod()
    peak = nav.cummax()
    drawdown = nav / peak - 1.0
    trough_date = drawdown.idxmin()
    peak_date = nav.loc[:trough_date].idxmax()
    recovery_candidates = nav.loc[trough_date:]
    recovered = recovery_candidates.loc[
        recovery_candidates >= nav.loc[peak_date]
    ]
    recovery_date = recovered.index[0] if not recovered.empty else pd.NaT
    drawdown_episode = pd.DataFrame(
        [
            {
                "episode_rank": 1,
                "peak_date": peak_date,
                "trough_date": trough_date,
                "recovery_date": recovery_date,
                "maximum_drawdown": float(drawdown.loc[trough_date]),
                "calendar_days_peak_to_trough": int(
                    (trough_date - peak_date).days
                ),
                "calendar_days_to_recovery": (
                    int((recovery_date - peak_date).days)
                    if pd.notna(recovery_date)
                    else None
                ),
                "recovered_by_oos_end": pd.notna(recovery_date),
            }
        ]
    )
    return monthly, worst, drawdown_episode


def _experiment_registry(
    control: dict[str, Any],
    metrics: pd.DataFrame,
    delisting: pd.DataFrame,
    output_hashes: dict[str, str],
    frozen_hash: str,
) -> pd.DataFrame:
    metric_lookup = metrics.set_index("experiment_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    for experiment in control["experiments"]:
        experiment_id = experiment["experiment_id"]
        metric = metric_lookup.get(experiment_id, {})
        if experiment_id == "DELIST_TERMINAL_RECOVERY":
            baseline = delisting.loc[
                delisting["is_baseline"]
                & (delisting["recovery_rate"] == 0.0)
            ].iloc[0]
            metric = {
                "status": "PASS",
                "source": "P6_TERMINAL_SENSITIVITY",
                "strategy_annualized_return": baseline[
                    "adjusted_strategy_annualized_return"
                ],
                "strategy_total_return": baseline[
                    "adjusted_strategy_total_return"
                ],
                "strategy_max_drawdown": None,
            }
        elif experiment_id == "FAILURE_PERIOD_ANALYSIS":
            metric = {
                "status": "PASS",
                "source": "P5_FROZEN_DIAGNOSTICS",
                "strategy_annualized_return": None,
                "strategy_total_return": None,
                "strategy_max_drawdown": None,
            }
        rows.append(
            {
                "experiment_id": experiment_id,
                "category": experiment["category"],
                "classification": "POST_OOS_ROBUSTNESS",
                "implementation": experiment["implementation"],
                "description": experiment["description"],
                "status": metric.get("status", "PASS"),
                "source": metric.get("source"),
                "strategy_total_return": metric.get("strategy_total_return"),
                "strategy_annualized_return": metric.get(
                    "strategy_annualized_return"
                ),
                "strategy_max_drawdown": metric.get(
                    "strategy_max_drawdown"
                ),
                "frozen_config_sha256": frozen_hash,
                "protocol_path": control["outputs"]["experiment_protocol"],
                "artifact_path": (
                    metric.get("artifact_prefix")
                    or control["outputs"]["delisting_sensitivity"]
                    if experiment_id == "DELIST_TERMINAL_RECOVERY"
                    else metric.get("artifact_prefix")
                    or control["outputs"]["monthly_diagnostics"]
                ),
                "artifact_sha256": (
                    output_hashes.get(
                        control["outputs"]["delisting_sensitivity"]
                    )
                    if experiment_id == "DELIST_TERMINAL_RECOVERY"
                    else None
                ),
                "p5_original_modified": False,
            }
        )
    return pd.DataFrame(rows)


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (9.2, 5.2),
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def _save_figure(figure: plt.Figure, relative: str) -> None:
    path = absolute(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _generate_charts(
    control: dict[str, Any],
    experiment_metrics: pd.DataFrame,
    delisting: pd.DataFrame,
) -> None:
    _set_plot_style()
    daily = pd.read_parquet(
        absolute(control["inputs"]["p5_oos_daily_portfolio"])
    )
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    base = daily.loc[daily["cost_scenario"] == BASELINE_SCENARIO].copy()
    base = base.sort_values("trade_date")
    oos = base.loc[base["trade_date"].between("2022-01-01", "2025-12-31")]
    strategy = (1.0 + oos["strategy_daily_return"]).cumprod()
    benchmark = (1.0 + oos["benchmark_daily_return"]).cumprod()

    figure, axis = plt.subplots()
    axis.plot(oos["trade_date"], strategy, label="Strategy", linewidth=2)
    axis.plot(oos["trade_date"], benchmark, label="CSI All Share", linewidth=1.7)
    axis.set_title("Frozen P5 final OOS cumulative NAV")
    axis.set_ylabel("NAV (start = 1)")
    axis.legend()
    _save_figure(figure, control["outputs"]["chart_cumulative_nav"])

    figure, axis = plt.subplots()
    strategy_dd = strategy / strategy.cummax() - 1.0
    benchmark_dd = benchmark / benchmark.cummax() - 1.0
    axis.fill_between(oos["trade_date"], strategy_dd, 0, alpha=0.35, label="Strategy")
    axis.plot(oos["trade_date"], benchmark_dd, label="CSI All Share", linewidth=1.2)
    axis.set_title("Frozen P5 final OOS drawdown")
    axis.set_ylabel("Drawdown")
    axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axis.legend()
    _save_figure(figure, control["outputs"]["chart_drawdown"])

    annual = pd.read_csv(absolute(control["inputs"]["p5_annual_performance"]))
    annual = annual.loc[
        (annual["cost_scenario"] == BASELINE_SCENARIO)
        & annual["year"].between(2022, 2025)
    ]
    positions = np.arange(len(annual))
    figure, axis = plt.subplots()
    axis.bar(positions - 0.18, annual["strategy_return"], 0.36, label="Strategy")
    axis.bar(positions + 0.18, annual["benchmark_return"], 0.36, label="Benchmark")
    axis.set_xticks(positions, annual["year"].astype(str))
    axis.set_title("Annual returns in final OOS")
    axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axis.legend()
    _save_figure(figure, control["outputs"]["chart_annual_returns"])

    ic_frames = []
    for period, key in (
        ("Research", "p2_ic_summary"),
        ("Validation", "p4_ic_summary"),
        ("Final OOS", "p5_ic_summary"),
    ):
        frame = pd.read_csv(absolute(control["inputs"][key]))
        frame["period"] = period
        ic_frames.append(frame)
    ic = pd.concat(ic_frames, ignore_index=True)
    factor_order = ["bm_proxy", "momentum_12_1", "lowvol_60"]
    period_order = ["Research", "Validation", "Final OOS"]
    figure, axis = plt.subplots()
    width = 0.24
    positions = np.arange(len(factor_order))
    for index, period in enumerate(period_order):
        values = (
            ic.loc[ic["period"] == period]
            .set_index("factor")
            .reindex(factor_order)["mean_rank_ic"]
        )
        axis.bar(
            positions + (index - 1) * width,
            values,
            width,
            label=period,
        )
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_xticks(positions, ["1/PB", "Momentum", "Low volatility"])
    axis.set_title("Mean monthly Rank IC by sample")
    axis.legend()
    _save_figure(figure, control["outputs"]["chart_factor_ic"])

    robust = experiment_metrics.loc[
        experiment_metrics["experiment_id"].isin(
            [
                "P5_BASELINE_REFERENCE",
                "MICROCAP_EXCLUDE_BOTTOM20",
                "MOMENTUM_9_1",
                "LOWVOL_120",
                "INDUSTRY_NEUTRAL",
                "LOG_SIZE_NEUTRAL",
                "COST_5BPS_REFERENCE",
                "COST_20BPS_REFERENCE",
            ]
        )
    ].copy()
    labels = {
        "P5_BASELINE_REFERENCE": "P5 base",
        "MICROCAP_EXCLUDE_BOTTOM20": "No bottom 20% size",
        "MOMENTUM_9_1": "Momentum 9-1",
        "LOWVOL_120": "Low-vol 120d",
        "INDUSTRY_NEUTRAL": "Industry neutral",
        "LOG_SIZE_NEUTRAL": "Log-size neutral",
        "COST_5BPS_REFERENCE": "Cost 5 bps",
        "COST_20BPS_REFERENCE": "Cost 20 bps",
    }
    robust["label"] = robust["experiment_id"].map(labels)
    robust = robust.sort_values("strategy_annualized_return")
    figure, axis = plt.subplots(figsize=(9.2, 6.0))
    colors = np.where(
        robust["strategy_annualized_return"] >= 0, "#3572A5", "#B55252"
    )
    axis.barh(robust["label"], robust["strategy_annualized_return"], color=colors)
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_title("Post-OOS robustness: annualized return")
    axis.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _save_figure(figure, control["outputs"]["chart_robustness"])

    sensitivity = delisting.loc[delisting["is_baseline"]].sort_values(
        "recovery_rate"
    )
    figure, axis = plt.subplots()
    axis.plot(
        sensitivity["recovery_rate"],
        sensitivity["adjusted_strategy_annualized_return"],
        marker="o",
        linewidth=2,
    )
    axis.set_title("Terminal delisting recovery sensitivity (not a path rerun)")
    axis.set_xlabel("Terminal recovery rate")
    axis.set_ylabel("Annualized return")
    axis.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _save_figure(figure, control["outputs"]["chart_delisting"])


def _fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}%}"


def _report_materials(
    control: dict[str, Any],
    frozen_hash: str,
    metrics: pd.DataFrame,
    delisting: pd.DataFrame,
    monthly: pd.DataFrame,
    worst: pd.DataFrame,
    failure: pd.DataFrame,
    p5_manifest: dict[str, Any],
) -> tuple[str, str, str, str, str]:
    baseline = metrics.loc[
        metrics["experiment_id"] == "P5_BASELINE_REFERENCE"
    ].iloc[0]
    reruns = metrics.loc[
        metrics["source"] == "P6_BASE_10BPS_RERUN"
    ].sort_values("experiment_id")
    zero_recovery = delisting.loc[
        delisting["is_baseline"] & (delisting["recovery_rate"] == 0.0)
    ].iloc[0]
    worst_strategy = worst.loc[
        worst["ranking_type"] == "WORST_STRATEGY_RETURN"
    ].iloc[0]
    best_variant = reruns.loc[
        reruns["strategy_annualized_return"].idxmax()
    ]
    worst_variant = reruns.loc[
        reruns["strategy_annualized_return"].idxmin()
    ]
    limitations = f"""# 研究限制与披露

## 最终 OOS 的不可变性

P5 原始结果保持不变，冻结配置 SHA-256 为
`{frozen_hash}`。P6 的所有实验均为 `POST_OOS_ROBUSTNESS`，
不能替代 P5 最终 OOS，不能被描述为新的预注册 OOS。

## 退市与陈旧估值

P5 对 `000413.SZ`、`000666.SZ`、`000671.SZ` 缺少可审计的精确退市日、
终止估值日和回收事件，因此沿用最后可得复权收盘价至期末。基准场景期末陈旧估值
为 {baseline['terminal_stale_price_weight']:.2%} 的组合净值。

P6 只对期末账面价值做 0%、25%、50%、75%、100% 回收率敏感性，不推断
退市日期，不重放交易路径，不生成订单或税费。0% 回收率下的年化收益为
{zero_recovery['adjusted_strategy_annualized_return']:.2%}；100% 等于 P5 原始
年化收益 {_fmt_pct(baseline['strategy_annualized_return'])}。

## PB 数据口径

使用供应商历史 PB 构造 1/PB 代理，未自行重建严格 point-in-time book equity，
供应商历史修订政策未完全核验，状态为 `NEEDS_MANUAL_CONFIRMATION`。
该限制可能带来历史修订和可得性偏差。

## 统计与选择偏差

- P6 是看到最终 OOS 后进行的稳健性检查，不用于调参或选择新的主模型。
- 样本期只有 2022 至 2025 四个自然年，宏观和风格环境覆盖有限。
- 组合使用分数等权、月末信号、下一交易日开盘成交；未建模盘口冲击和容量曲线。
- 停牌、涨跌停、历史印花税和换股吸收合并已显式处理，但数据源仍可能漏记其他
  公司行动。
- 行业与规模中性实验是诊断性变体，不应被宣传为原始 OOS 结果。
"""
    robustness_lines = [
        "# P6 稳健性研究结果",
        "",
        f"- 阶段状态：`{P6_STATUS}`",
        f"- 冻结配置 SHA-256：`{frozen_hash}`",
        "- P5 文件：运行前后逐项 SHA-256 一致。",
        "- 所有变体：`POST_OOS_ROBUSTNESS`，未用于替换主结果。",
        "",
        "## P5 原始锚点",
        "",
        f"- 年化收益：{baseline['strategy_annualized_return']:.2%}",
        f"- 累计收益：{baseline['strategy_total_return']:.2%}",
        f"- 最大回撤：{baseline['strategy_max_drawdown']:.2%}",
        f"- 信息比率：{baseline['information_ratio']:.3f}",
        "",
        "## 变体结果（10 bps）",
        "",
        "| 实验 | 年化收益 | 累计收益 | 最大回撤 | 换手率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in reruns.itertuples(index=False):
        robustness_lines.append(
            f"| {row.experiment_id} | {row.strategy_annualized_return:.2%} | "
            f"{row.strategy_total_return:.2%} | {row.strategy_max_drawdown:.2%} | "
            f"{row.two_way_turnover:.2f} |"
        )
    robustness_lines.extend(
        [
            "",
            "## 诊断结论",
            "",
            f"- 变体中年化收益最高：`{best_variant['experiment_id']}` "
            f"({_fmt_pct(best_variant['strategy_annualized_return'])})。",
            f"- 变体中年化收益最低：`{worst_variant['experiment_id']}` "
            f"({_fmt_pct(worst_variant['strategy_annualized_return'])})。",
            f"- 最差策略月份：{worst_strategy['month']}，收益 "
            f"{worst_strategy['strategy_return']:.2%}，超额 "
            f"{worst_strategy['excess_return']:.2%}。",
            f"- 共登记 {len(control['experiments'])} 个实验族；所有 P5 "
            "原始文件在实验后哈希未变。",
            "",
            "完整限制见 `reports/limitations.md`。",
        ]
    )
    robustness = "\n".join(robustness_lines)

    evidence = f"""# 简历证据映射

> 以下表述均可由项目内文件复核；P6 结果必须注明为事后稳健性分析。

| 可用简历表述 | 核心证据 | 限制/面试口径 |
|---|---|---|
| 构建 A 股三因子月频选股研究，覆盖研究、验证与一次性最终 OOS | `configs/frozen_config.yaml`; `reports/oos/p5_run_manifest.json` | 不声称无偏全市场生产策略 |
| 最终 OOS 2022 至 2025，10 bps 场景年化收益 {baseline['strategy_annualized_return']:.2%}、最大回撤 {baseline['strategy_max_drawdown']:.2%} | `results/p5_oos/oos_performance.csv`; `reports/oos/p5_result_report.md` | 同时披露退市陈旧估值和 PB 修订政策 |
| 实现停牌、涨跌停、历史印花税及换股吸收合并的事件驱动回测 | `configs/trading_costs.yaml`; `data/manual/corporate_actions.csv`; `reports/backtest/p3_audit_report.md` | 不声称覆盖所有公司行动 |
| 通过 P1 至 P6 分阶段审计、配置冻结与 SHA-256 数据血缘保证复现 | `reports/robustness/p6_run_manifest.json`; `results/experiment_registry.csv` | P6 是 post-OOS，不用于调参 |
| 完成微盘剔除、窗口、行业/规模中性、成本与退市回收敏感性 | `reports/robustness/p6_robustness_report.md`; `results/p6_robustness/experiment_metrics.csv` | 这些不是新的 OOS |

## 数字核对

- 冻结配置：`{frozen_hash}`
- P5 年化收益：{baseline['strategy_annualized_return']:.6%}
- P5 累计收益：{baseline['strategy_total_return']:.6%}
- P5 最大回撤：{baseline['strategy_max_drawdown']:.6%}
- P5 输出哈希项数：{len(p5_manifest['output_sha256'])}
- P6 稳健性重新运行变体数：{len(reruns)}
"""

    interview = f"""# A 股多因子项目面试材料

## 90 秒项目介绍

我完成了一个可审计的 A 股横截面多因子研究。股票池限定沪深 A 股，使用供应商历史
PB 的倒数、12-1 动量和 60 日低波三个因子，月末生成信号、下一交易日开盘成交，
组合 Top-100 等权。流程先在 2016 至 2019 研究，2020 至 2021 验证并冻结配置，
随后只运行一次 2022 至 2025 最终 OOS。10 bps 单边成本下，原始最终 OOS 年化
收益 {baseline['strategy_annualized_return']:.2%}、累计收益
{baseline['strategy_total_return']:.2%}、最大回撤
{baseline['strategy_max_drawdown']:.2%}。回测显式处理停牌、涨跌停、历史印花税和
600270.SH 换股吸收合并。最后我把所有稳健性实验标为 post-OOS，并保留 P5 原始
哈希不变。最重要的限制是三只退市证券缺少精确终止估值事件，以及 PB 历史修订政策
未完全核验；我没有掩盖或事后修正原始结果，而是单独给出敏感性区间。

## 高频追问

### 1. 为什么选择这三个因子？

价值、趋势和低风险来自不同经济直觉；等权避免验证期调参。单因子 IC 在研究、验证、
OOS 三段分别报告，动量在 OOS 转负，说明组合收益不能简单归因于所有因子持续有效。

### 2. 如何避免前视偏差？

月末收盘后形成信号，下一市场交易日开盘执行；上市年龄按交易日累计；历史 ST 名称按
有效区间判断；退市日期缺失时不从最后行情日倒推。

### 3. 为什么 P6 不是新的 OOS？

因为 P6 在观察 P5 后执行。所有实验登记为 `POST_OOS_ROBUSTNESS`，只检验敏感性，
不选择新主模型，也不覆盖 P5。

### 4. 交易成本如何处理？

买卖均计单边佣金与滑点；卖出按历史生效日读取印花税。P5 冻结 5、10、20 bps
三组场景，主结果采用 10 bps。

### 5. 如何处理停牌和涨跌停？

执行表在开盘逐单判断：停牌、无开盘价、涨停买入、跌停卖出均失败，并保留失败订单
原因。P5 OOS 基准场景买入失败 {int(baseline['failed_buy_orders'])} 笔、卖出失败
{int(baseline['failed_sell_orders'])} 笔。

### 6. 600270.SH 公司行动如何处理？

2018-12-13 起不可交易，2018-12-28 按 3.8225 比例转换为 601598.SH；不收佣金、
滑点或印花税，允许非整数股，并以 `CORPORATE_ACTION` 记录。

### 7. 最严重的数据限制是什么？

三只退市持仓缺少精确退市与回收事件，P5 期末仍按最后价格估值，占净值
{baseline['terminal_stale_price_weight']:.2%}。P6 的 0% 终值回收情景年化收益为
{zero_recovery['adjusted_strategy_annualized_return']:.2%}，但这不是路径重放。

### 8. PB 有什么问题？

1/PB 是供应商历史 PB 的代理，不是自行重建的 point-in-time 账面权益；供应商是否
回填修订尚未完全核验。

### 9. 为什么用复权价格？

采用后复权 `close * adj_factor` 形成连续收益，不以最新因子前复权归一化。代码变更
主体以 canonical code 连接，但跨代码边界先经过价格和复权因子审计。

### 10. 如何解释 2025 年跑输基准？

组合 2025 年仍为正收益，但中证全指更强；这说明策略是风格暴露与选股的组合，不保证
每年跑赢。应展示年度分解，而不是只报四年累计值。

### 11. 最差月份发生了什么？

最差策略月份为 {worst_strategy['month']}，收益
{worst_strategy['strategy_return']:.2%}。我同时检查当月换手、失败订单、现金和陈旧
估值，避免把单一原因强行解释为因果。

### 12. 行业中性化如何做？

每月对三个标准化因子分别按行业去均值，再做横截面样本标准化；这是 P6 诊断变体。

### 13. 规模中性化如何做？

每月将各因子对 `log(total_mv_cny)` 做带截距 OLS，取残差后再样本标准化。

### 14. 怎样证明没有覆盖原始结果？

P5 manifest 中 {len(p5_manifest['output_sha256'])} 个输出逐项计算 SHA-256；P6 运行
前后都必须匹配，P6 数据只写入 `data/processed/p6_robustness/`。

### 15. 如果继续升级会做什么？

优先补齐可审计退市事件与回收现金流、验证 PB point-in-time 政策，再加入容量模型和
更完整公司行动；不会先根据 P6 表现挑选参数。

## 不应声称

- 不说“无幸存者偏差”或“严格 point-in-time 基本面”。
- 不把 P6 最佳变体包装成最终 OOS。
- 不忽略退市陈旧估值和 2025 年相对落后。
- 不说策略已可直接实盘；当前是研究与工程证据项目。
"""

    notes = f"""# Learning Notes

## 项目闭环

本项目按 P1 数据标准化、P2 因子研究、P3 执行回测、P4 验证与冻结、P5 一次性
最终 OOS、P6 事后稳健性与报告完成。最终状态为 `{P6_STATUS}`。

## 最重要的工程经验

1. 配置冻结必须同时绑定输入 SHA-256，否则“未改参数”不足以保证复现。
2. 证券代码和上市主体不是同一概念；代码变更必须保留当日真实代码与 canonical code。
3. 公司行动是持仓事件，不是普通交易；数量、价值、成本和审计类型应分别验证。
4. 缺失退市事件不能用最后行情日静默推断；应保留原结果并做明确敏感性。
5. 看过 OOS 后的任何实验都只能叫 post-OOS robustness，不能重新命名为验证。

## 结果阅读

P5 主结果年化收益 {baseline['strategy_annualized_return']:.2%}，最大回撤
{baseline['strategy_max_drawdown']:.2%}。P6 变体区间从
{reruns['strategy_annualized_return'].min():.2%} 到
{reruns['strategy_annualized_return'].max():.2%}，说明结论对建模选择有敏感性。
这些数字用于理解脆弱性，不用于改写主模型。
"""
    return limitations, robustness, evidence, interview, notes


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _research_report_tex(
    control: dict[str, Any],
    frozen_hash: str,
    metrics: pd.DataFrame,
    delisting: pd.DataFrame,
    worst: pd.DataFrame,
    failure: pd.DataFrame,
    p5_manifest: dict[str, Any],
) -> str:
    baseline = metrics.loc[
        metrics["experiment_id"] == "P5_BASELINE_REFERENCE"
    ].iloc[0]
    robust = metrics.loc[
        metrics["experiment_id"].isin(
            [
                "MICROCAP_EXCLUDE_BOTTOM20",
                "MOMENTUM_9_1",
                "LOWVOL_120",
                "INDUSTRY_NEUTRAL",
                "LOG_SIZE_NEUTRAL",
            ]
        )
    ].sort_values("experiment_id")
    annual = pd.read_csv(absolute(control["inputs"]["p5_annual_performance"]))
    annual = annual.loc[
        (annual["cost_scenario"] == BASELINE_SCENARIO)
        & annual["year"].between(2022, 2025)
    ]
    ic_rows = []
    for label, key in (
        ("研究期", "p2_ic_summary"),
        ("验证期", "p4_ic_summary"),
        ("最终 OOS", "p5_ic_summary"),
    ):
        frame = pd.read_csv(absolute(control["inputs"][key]))
        for row in frame.itertuples(index=False):
            ic_rows.append(
                f"{label} & {_latex_escape(row.factor)} & "
                f"{row.mean_rank_ic:.4f} & {row.rank_icir_annualized:.3f} \\\\"
            )
    robust_rows = [
        f"{_latex_escape(row.experiment_id)} & "
        f"{row.strategy_annualized_return:.2%} & "
        f"{row.strategy_total_return:.2%} & "
        f"{row.strategy_max_drawdown:.2%} & {row.two_way_turnover:.2f} \\\\"
        for row in robust.itertuples(index=False)
    ]
    annual_rows = [
        f"{int(row.year)} & {row.strategy_return:.2%} & "
        f"{row.benchmark_return:.2%} & {row.return_difference:.2%} \\\\"
        for row in annual.itertuples(index=False)
    ]
    sensitivity = delisting.loc[delisting["is_baseline"]].sort_values(
        "recovery_rate"
    )
    sensitivity_rows = [
        f"{row.recovery_rate:.0%} & "
        f"{row.adjusted_strategy_total_return:.2%} & "
        f"{row.adjusted_strategy_annualized_return:.2%} & "
        f"{row.annualized_return_change_vs_p5:.2%} \\\\"
        for row in sensitivity.itertuples(index=False)
    ]
    worst_rows = [
        f"{_latex_escape(row.month)} & {row.strategy_return:.2%} & "
        f"{row.benchmark_return:.2%} & {row.excess_return:.2%} & "
        f"{int(row.failed_orders)} \\\\"
        for row in worst.loc[
            worst["ranking_type"] == "WORST_STRATEGY_RETURN"
        ].head(5).itertuples(index=False)
    ]
    baseline_failure = failure.loc[
        failure["cost_scenario"] == BASELINE_SCENARIO
    ]
    failure_rows = [
        f"{_latex_escape(str(row.side))} & "
        f"{_latex_escape(str(row.failure_reason))} & "
        f"{int(row.failed_orders)} \\\\"
        for row in baseline_failure.head(12).itertuples(index=False)
    ]
    figures = {
        key: Path(value).as_posix()
        for key, value in control["outputs"].items()
        if key.startswith("chart_")
    }
    return rf"""\documentclass[11pt,a4paper]{{ctexart}}
\usepackage[margin=2.25cm]{{geometry}}
\usepackage{{booktabs,longtable,array,graphicx,float,xcolor,hyperref,fancyhdr}}
\usepackage{{enumitem}}
\definecolor{{navy}}{{HTML}}{{17365D}}
\definecolor{{softblue}}{{HTML}}{{EAF1F8}}
\hypersetup{{colorlinks=true,linkcolor=navy,urlcolor=navy}}
\setlength{{\parindent}}{{2em}}
\setlength{{\parskip}}{{0.35em}}
\setlist{{nosep}}
\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{A 股多因子研究}}
\fancyhead[R]{{P6 正式报告}}
\fancyfoot[C]{{\thepage}}
\renewcommand{{\headrulewidth}}{{0.4pt}}
\newcommand{{\pct}}[1]{{\textbf{{#1}}}}
\title{{\textbf{{A 股三因子选股研究}}\\[0.4em]
\large 从数据审计、冻结验证到一次性最终 OOS}}
\author{{可复现量化研究项目}}
\date{{2026 年 7 月 27 日}}
\begin{{document}}
\maketitle
\begin{{abstract}}
本文记录一个以 1/PB、12-1 动量和 60 日低波为核心的 A 股月频横截面研究。
项目将 2016 至 2019 定义为研究期、2020 至 2021 定义为验证期，并在参数冻结后
只执行一次 2022 至 2025 最终 OOS。10 bps 单边成本下，最终 OOS 年化收益为
\pct{{{baseline['strategy_annualized_return']:.2%}}}，累计收益为
\pct{{{baseline['strategy_total_return']:.2%}}}，最大回撤为
\pct{{{baseline['strategy_max_drawdown']:.2%}}}。P6 稳健性不改写原始结果，
全部标记为事后研究，并显式披露退市陈旧估值及 PB 历史修订政策限制。
\end{{abstract}}

\section{{研究设计与阶段治理}}
研究对象限定为沪深 A 股。P1 形成标准化日频、月末、执行状态和基准面板；
P2 在研究期检验单因子；P3 构建带交易约束的事件驱动回测；P4 使用独立验证期并冻结
配置；P5 通过冻结研究闸门后执行一次最终 OOS；P6 只做事后稳健性、报告和证据整理。
冻结配置 SHA-256 为：
\begin{{quote}}\small\ttfamily {_latex_escape(frozen_hash)}\end{{quote}}
P5 manifest 记录 {len(p5_manifest['output_sha256'])} 个输出哈希，P6 前后逐项一致。

\section{{数据、口径与审计}}
日行情成交量由手换算为股，成交额由千元换算为人民币元；总股本、流通股本、自由流通
股本和市值按已确认单位标准化。复权收盘价采用后复权
\texttt{{close\_hfq = close * adj\_factor}}，收益由连续复权价格计算。证券代码变更
同时保留当日真实 \texttt{{ts\_code}} 与连续主体
\texttt{{canonical\_ts\_code}}。基准为中证全指
\texttt{{000985.CSI}}。

历史 ST、上市区间、停牌、开盘涨跌停均采用当日可得状态。历史卖出印花税按生效日配置。
600270.SH 换股吸收合并按每股转换 3.8225 股 601598.SH 处理，记录为公司行动而非
交易，并保持组合价值连续。

\section{{因子与组合构建}}
价值因子为供应商历史 PB 的倒数，仅在 PB 大于零时计算；动量为跳过最近约 21 个
交易日后的约 252 日价格变化；低波因子为 60 日收益样本标准差的相反数。每月横截面
1\% 和 99\% 缩尾后做样本标准化，三因子等权合成。股票池排除历史 ST、上市不足
120 个交易日、流动性最低 20\% 及缺少任一因子的股票。组合选取 Top-100 等权，
月末信号在下一市场交易日开盘执行。

\section{{单因子稳定性}}
\begin{{table}}[H]\centering\small
\begin{{tabular}}{{llrr}}\toprule
样本 & 因子 & 平均 Rank IC & 年化 ICIR \\\midrule
{chr(10).join(ic_rows)}
\bottomrule\end{{tabular}}
\caption{{三个样本阶段的单因子月度 Rank IC}}
\end{{table}}
低波因子在三段样本均保持正 IC，价值因子也为正；动量在最终 OOS 转负。这一结果
支持对组合收益来源保持克制：不能声称三个因子在所有阶段都稳定有效。

\section{{最终 OOS 结果}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.94\textwidth]{{{figures['chart_cumulative_nav']}}}
\caption{{P5 原始最终 OOS 累计净值}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.94\textwidth]{{{figures['chart_drawdown']}}}
\caption{{P5 原始最终 OOS 回撤}}
\end{{figure}}

最终 OOS 共 {int(baseline['trading_days'])} 个交易日。策略年化波动率
{baseline['strategy_annualized_volatility']:.2%}，零无风险利率夏普
{baseline['strategy_sharpe_zero_rf']:.3f}，信息比率
{baseline['information_ratio']:.3f}，双边换手
{baseline['two_way_turnover']:.2f}。同期中证全指累计收益
{baseline['benchmark_total_return']:.2%}。

\begin{{table}}[H]\centering
\begin{{tabular}}{{rrrr}}\toprule
年份 & 策略 & 中证全指 & 差值 \\\midrule
{chr(10).join(annual_rows)}
\bottomrule\end{{tabular}}
\caption{{最终 OOS 年度收益}}
\end{{table}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.92\textwidth]{{{figures['chart_annual_returns']}}}
\caption{{最终 OOS 年度收益对比}}
\end{{figure}}

\section{{事后稳健性研究}}
以下所有结果均为 \texttt{{POST\_OOS\_ROBUSTNESS}}，不替代 P5。
\begin{{table}}[H]\centering\small
\begin{{tabular}}{{lrrrr}}\toprule
实验 & 年化收益 & 累计收益 & 最大回撤 & 双边换手 \\\midrule
{chr(10).join(robust_rows)}
\bottomrule\end{{tabular}}
\caption{{10 bps 场景稳健性变体}}
\end{{table}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.94\textwidth]{{{figures['chart_robustness']}}}
\caption{{事后稳健性变体年化收益}}
\end{{figure}}

微盘剔除检验结果是否依赖最小市值证券；两组窗口变体检验动量与低波定义；
行业中性化先按月在行业内去均值再标准化；规模中性化将因子对总市值对数回归取残差。
成本压力直接引用 P5 已冻结的 5、10、20 bps 结果，不重新运行或选择主情景。

\section{{退市回收率敏感性}}
P5 的三只退市持仓缺少精确退市日、回收日和现金流，因此原始回测使用最后可得价格估值。
P6 不推断缺失事件，只调整期末账面价值。
\begin{{table}}[H]\centering
\begin{{tabular}}{{rrrr}}\toprule
期末回收率 & 调整后累计收益 & 调整后年化收益 & 相对 P5 年化变化 \\\midrule
{chr(10).join(sensitivity_rows)}
\bottomrule\end{{tabular}}
\caption{{10 bps 场景期末退市回收率敏感性}}
\end{{table}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.92\textwidth]{{{figures['chart_delisting']}}}
\caption{{期末退市回收率敏感性，不是交易路径重放}}
\end{{figure}}

\section{{失败时期与执行诊断}}
\begin{{table}}[H]\centering\small
\begin{{tabular}}{{lrrrr}}\toprule
月份 & 策略收益 & 基准收益 & 超额收益 & 失败订单 \\\midrule
{chr(10).join(worst_rows)}
\bottomrule\end{{tabular}}
\caption{{最终 OOS 最差五个策略月份}}
\end{{table}}
\begin{{table}}[H]\centering\small
\begin{{tabular}}{{llr}}\toprule
方向 & 原因 & 失败订单 \\\midrule
{chr(10).join(failure_rows)}
\bottomrule\end{{tabular}}
\caption{{最终 OOS 失败订单原因摘要}}
\end{{table}}
失败时期分析同时检查收益、超额、换手、交易成本、现金和陈旧估值；这些变量是诊断线索，
不被直接解释为因果。

\section{{限制与可信表述}}
\begin{{enumerate}}
\item 供应商历史 PB 的修订政策未完全核验，1/PB 不是自行重建的严格 point-in-time
账面权益。
\item 三只退市证券缺少可审计终止估值事件，P5 期末陈旧估值权重为
{baseline['terminal_stale_price_weight']:.2%}。
\item P6 在看到最终 OOS 后执行，只能用于稳健性解释，不可用于调参或重新挑选策略。
\item 四年 OOS 覆盖的市场状态有限，且未建模容量曲线与盘口冲击。
\end{{enumerate}}

\section{{结论}}
项目的主要价值不只是一个收益数字，而是可追踪的研究治理：输入单位和语义明确，
样本边界固定，交易约束可审计，公司行动可复核，验证后参数冻结，最终 OOS 只运行
一次，事后实验与原始结果隔离。主结果在最终 OOS 为正并跑赢同期基准，但存在清晰的
数据与估值限制。因此最终状态记为
\texttt{{PROJECT\_COMPLETE\_WITH\_DISCLOSED\_LIMITATIONS}}。

\appendix
\section{{复现入口}}
\begin{{verbatim}}
python scripts/build_p6.py
python -m pytest -q
python scripts/audit_p6.py
\end{{verbatim}}
关键机器可读输出为 \texttt{{results/metrics.json}} 与
\texttt{{results/experiment\_registry.csv}}；审计清单为
\texttt{{reports/robustness/p6\_run\_manifest.json}}。
\end{{document}}
"""


def _research_report_html(
    control: dict[str, Any],
    frozen_hash: str,
    metrics: pd.DataFrame,
    delisting: pd.DataFrame,
    worst: pd.DataFrame,
    failure: pd.DataFrame,
    p5_manifest: dict[str, Any],
) -> str:
    baseline = metrics.loc[
        metrics["experiment_id"] == "P5_BASELINE_REFERENCE"
    ].iloc[0]
    robust = metrics.loc[
        metrics["source"] == "P6_BASE_10BPS_RERUN"
    ].sort_values("experiment_id")
    annual = pd.read_csv(absolute(control["inputs"]["p5_annual_performance"]))
    annual = annual.loc[
        (annual["cost_scenario"] == BASELINE_SCENARIO)
        & annual["year"].between(2022, 2025)
    ].copy()
    annual_table = annual[
        ["year", "strategy_return", "benchmark_return", "return_difference"]
    ].copy()
    annual_table.columns = ["年份", "策略收益", "中证全指", "收益差"]
    for column in ("策略收益", "中证全指", "收益差"):
        annual_table[column] = annual_table[column].map(lambda value: f"{value:.2%}")

    robust_table = robust[
        [
            "experiment_id",
            "strategy_annualized_return",
            "strategy_total_return",
            "strategy_max_drawdown",
            "two_way_turnover",
        ]
    ].copy()
    robust_table.columns = ["实验", "年化收益", "累计收益", "最大回撤", "双边换手"]
    for column in ("年化收益", "累计收益", "最大回撤"):
        robust_table[column] = robust_table[column].map(lambda value: f"{value:.2%}")
    robust_table["双边换手"] = robust_table["双边换手"].map(
        lambda value: f"{value:.2f}"
    )

    sensitivity = delisting.loc[delisting["is_baseline"]].sort_values(
        "recovery_rate"
    )[
        [
            "recovery_rate",
            "adjusted_strategy_total_return",
            "adjusted_strategy_annualized_return",
            "annualized_return_change_vs_p5",
        ]
    ].copy()
    sensitivity.columns = ["期末回收率", "调整后累计收益", "调整后年化收益", "年化变化"]
    for column in sensitivity.columns:
        sensitivity[column] = sensitivity[column].map(lambda value: f"{value:.2%}")

    ic_tables: list[pd.DataFrame] = []
    for period, key in (
        ("研究期 2016-2019", "p2_ic_summary"),
        ("验证期 2020-2021", "p4_ic_summary"),
        ("最终 OOS 2022-2025", "p5_ic_summary"),
    ):
        frame = pd.read_csv(absolute(control["inputs"][key]))[
            ["factor", "months", "mean_rank_ic", "rank_icir_annualized"]
        ].copy()
        frame.insert(0, "样本", period)
        ic_tables.append(frame)
    ic = pd.concat(ic_tables, ignore_index=True)
    ic.columns = ["样本", "因子", "月数", "平均 Rank IC", "年化 ICIR"]
    ic["平均 Rank IC"] = ic["平均 Rank IC"].map(lambda value: f"{value:.4f}")
    ic["年化 ICIR"] = ic["年化 ICIR"].map(lambda value: f"{value:.3f}")

    worst_table = worst.loc[
        worst["ranking_type"] == "WORST_STRATEGY_RETURN"
    ][
        [
            "month",
            "strategy_return",
            "benchmark_return",
            "excess_return",
            "turnover",
            "failed_orders",
        ]
    ].head(10).copy()
    worst_table.columns = ["月份", "策略收益", "基准收益", "超额收益", "换手", "失败订单"]
    for column in ("策略收益", "基准收益", "超额收益"):
        worst_table[column] = worst_table[column].map(lambda value: f"{value:.2%}")
    worst_table["换手"] = worst_table["换手"].map(lambda value: f"{value:.2f}")

    failure_table = failure.loc[
        failure["cost_scenario"] == BASELINE_SCENARIO,
        ["side", "failure_reason", "failed_orders"]
    ].head(16).copy()
    failure_table.columns = ["方向", "失败原因", "订单数"]

    def table(frame: pd.DataFrame) -> str:
        return frame.to_html(index=False, border=0, classes="data-table", escape=True)

    figure_uris = {
        key: absolute(value).resolve().as_uri()
        for key, value in control["outputs"].items()
        if key.startswith("chart_")
    }
    title_metrics = (
        f"<span>年化收益 <b>{baseline['strategy_annualized_return']:.2%}</b></span>"
        f"<span>累计收益 <b>{baseline['strategy_total_return']:.2%}</b></span>"
        f"<span>最大回撤 <b>{baseline['strategy_max_drawdown']:.2%}</b></span>"
        f"<span>信息比率 <b>{baseline['information_ratio']:.3f}</b></span>"
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>A 股三因子选股研究</title>
<style>
@page {{
  size: A4;
  margin: 18mm 18mm 19mm;
  @top-left {{ content: "A 股多因子研究"; color: #506176; font-size: 8.5pt; }}
  @top-right {{ content: "P6 正式报告"; color: #506176; font-size: 8.5pt; }}
  @bottom-center {{ content: "第 " counter(page) " 页"; color: #66768a; font-size: 8pt; }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; color: #172438; font-family: "Microsoft YaHei", "Noto Sans CJK SC",
  "SimSun", sans-serif; font-size: 10.3pt; line-height: 1.62;
}}
h1, h2, h3 {{ color: #17365d; margin: 0 0 10px; }}
h1 {{ font-size: 27pt; line-height: 1.28; letter-spacing: .03em; }}
h2 {{ font-size: 18pt; border-bottom: 2px solid #89a7c7; padding-bottom: 6px; }}
h3 {{ font-size: 12.5pt; margin-top: 14px; }}
p {{ margin: 7px 0; text-align: justify; }}
ul, ol {{ margin: 6px 0 8px 22px; padding: 0; }}
li {{ margin: 3px 0; }}
code {{ font-family: Consolas, monospace; background: #eef3f8; padding: 1px 4px; }}
.cover {{
  height: 242mm; display: flex; flex-direction: column; justify-content: center;
  position: relative; break-after: page;
}}
.cover:before {{
  content: ""; position: absolute; top: 22mm; left: 0; width: 46mm; height: 5px;
  background: #2f6fa6;
}}
.kicker {{ color: #2f6fa6; font-weight: 700; letter-spacing: .18em; margin-bottom: 16px; }}
.subtitle {{ font-size: 15pt; color: #52677d; margin: 13px 0 25px; }}
.metric-strip {{ display: grid; grid-template-columns: 1fr 1fr; gap: 9px; margin: 22px 0; }}
.metric-strip span {{ background: #edf3f8; border-left: 4px solid #2f6fa6; padding: 10px; }}
.meta {{ margin-top: 35px; color: #617186; }}
.section {{ break-before: page; min-height: 238mm; }}
.figure-page {{ break-before: page; min-height: 238mm; }}
.callout {{
  border-left: 4px solid #d18b2c; background: #fff7e8; padding: 10px 13px;
  margin: 12px 0;
}}
.success {{
  border-left: 4px solid #32836b; background: #edf8f4; padding: 10px 13px;
  margin: 12px 0;
}}
.hash {{ word-break: break-all; font: 8.8pt Consolas, monospace; color: #40546b; }}
.data-table {{ width: 100%; border-collapse: collapse; margin: 11px 0 15px; font-size: 8.8pt; }}
.data-table th {{ background: #17365d; color: white; text-align: left; padding: 6px; }}
.data-table td {{ border-bottom: 1px solid #d7e0e8; padding: 5px 6px; }}
.data-table tr:nth-child(even) td {{ background: #f6f8fa; }}
.chart {{ width: 100%; max-height: 174mm; object-fit: contain; margin-top: 15px; }}
.caption {{ color: #607187; font-size: 9pt; text-align: center; margin-top: 7px; }}
.two-col {{ columns: 2; column-gap: 24px; }}
.small {{ font-size: 9pt; color: #4f6075; }}
.status {{ display: inline-block; background: #17365d; color: white; padding: 4px 10px; }}
</style>
</head>
<body>
<article class="cover">
  <div class="kicker">可复现量化研究项目</div>
  <h1>A 股三因子选股研究</h1>
  <div class="subtitle">从数据审计、验证冻结到一次性最终 OOS</div>
  <div class="metric-strip">{title_metrics}</div>
  <p class="success"><b>最终状态：</b>{P6_STATUS}<br>
  P5 原始结果保持不变；P6 全部实验均标记为 POST_OOS_ROBUSTNESS。</p>
  <div class="meta">
    <p>基准：中证全指 000985.CSI</p>
    <p>最终 OOS：2022-01-01 至 2025-12-31</p>
    <p>报告日期：2026-07-27</p>
  </div>
</article>

<section class="section">
<h2>执行摘要</h2>
<p>本项目研究 1/PB、12-1 动量和 60 日低波三个 A 股横截面因子。研究期为
2016 至 2019，验证期为 2020 至 2021；配置冻结后只运行一次 2022 至 2025
最终 OOS。组合每月选择 Top-100 等权，月末收盘形成信号，下一市场交易日开盘执行。</p>
<p>10 bps 单边成本下，P5 原始最终 OOS 年化收益为
<b>{baseline['strategy_annualized_return']:.2%}</b>，累计收益
<b>{baseline['strategy_total_return']:.2%}</b>，最大回撤
<b>{baseline['strategy_max_drawdown']:.2%}</b>。同期中证全指累计收益
{baseline['benchmark_total_return']:.2%}。</p>
<div class="callout"><b>最重要的限制：</b>三只退市证券缺少精确终止估值和回收事件；
P5 将最后可得复权收盘价沿用至期末。P6 只做期末回收率敏感性，不推断路径。
此外，供应商历史 PB 修订政策未完全核验。</div>
<h3>治理证据</h3>
<ul>
<li>冻结配置 SHA-256：<span class="hash">{html.escape(frozen_hash)}</span></li>
<li>P5 manifest 哈希输出：{len(p5_manifest['output_sha256'])} 项，P6 前后逐项一致。</li>
<li>P6 变体与主结果分目录存储，不重新定义 OOS，不选择新主模型。</li>
</ul>
</section>

<section class="section">
<h2>1. 数据、口径与审计</h2>
<p>研究只纳入 .SH 和 .SZ。日行情成交量从手转换为股，成交额从千元转换为人民币元；
股本与市值按确认单位标准化。复权价采用后复权
<code>close_hfq = close * adj_factor</code>，收益由连续复权价计算。</p>
<p>证券代码变更同时保留当天真实 <code>ts_code</code> 与连续上市主体
<code>canonical_ts_code</code>。历史 ST 名称按有效区间判断，上市年龄以交易日累计，
首日计 1；北交所异常上市区间保留在原始审计中，但排除于正式沪深股票池。</p>
<h3>交易与事件语义</h3>
<ul>
<li>停牌、缺开盘价、开盘涨停买入、开盘跌停卖出均形成可审计失败订单。</li>
<li>卖出印花税按历史生效日配置，买入方为 0。</li>
<li>600270.SH 换股吸收合并按 3.8225 比例转为 601598.SH，记为公司行动。</li>
<li>基准标准化输出只含代码、日期、收盘价与收益。</li>
</ul>
<div class="callout">1/PB 使用供应商历史 PB 倒数，不是自行重建的严格 point-in-time
book equity。供应商历史修订政策仍为 NEEDS_MANUAL_CONFIRMATION。</div>
</section>

<section class="section">
<h2>2. 因子、股票池与组合</h2>
<h3>因子定义</h3>
<ol>
<li><b>价值：</b>仅 PB 大于 0 时计算 1/PB。</li>
<li><b>动量：</b>跳过最近约 21 个交易日后的约 252 日复权价格变化。</li>
<li><b>低波：</b>60 日收益样本标准差的相反数。</li>
</ol>
<p>因子每月按 1% 和 99% 分位缩尾，再按样本标准差做横截面标准化。三因子固定等权，
避免在验证期调权。股票池要求上市至少 120 个交易日，排除历史 ST、流动性最低
20% 以及任一因子缺失的证券。</p>
<h3>组合与成交</h3>
<p>复合分数降序选择 Top-100，目标等权 1%。卖出先于买入，现金不足时按包含受阻
买单需求的比例分配。交易价为复权开盘价，收盘估值为复权收盘价；缺失收盘价时仅用于
估值沿用最后可得价格，不允许据此成交。</p>
<div class="success">该设计优先保证可复现和可解释，而不是在验证或 OOS 后追求最佳参数。</div>
</section>

<section class="section">
<h2>3. 单因子跨样本表现</h2>
{table(ic)}
<p>低波因子在研究、验证和最终 OOS 均保持正平均 Rank IC，价值因子亦为正；
动量在最终 OOS 转负。因此主组合不能被描述为三个因子在所有阶段都稳定有效。</p>
<p>IC 是横截面排序诊断，不等同于可交易组合收益。正式组合还受到调仓时点、失败订单、
成本、现金分配、公司行动与陈旧估值影响。</p>
</section>

<section class="figure-page">
<h2>4. 最终 OOS 累计净值</h2>
<img class="chart" src="{figure_uris['chart_cumulative_nav']}">
<p class="caption">图 1：P5 原始最终 OOS 累计净值；P6 未覆盖该结果。</p>
<p>最终 OOS 共 {int(baseline['trading_days'])} 个交易日，信息比率
{baseline['information_ratio']:.3f}，零无风险利率夏普
{baseline['strategy_sharpe_zero_rf']:.3f}，双边换手
{baseline['two_way_turnover']:.2f}。</p>
</section>

<section class="figure-page">
<h2>5. 最终 OOS 回撤</h2>
<img class="chart" src="{figure_uris['chart_drawdown']}">
<p class="caption">图 2：策略与中证全指最终 OOS 回撤。</p>
<p>策略最大回撤 {baseline['strategy_max_drawdown']:.2%}。回撤是完整路径指标，
因此退市期末回收敏感性不重新计算或宣称修复历史最大回撤。</p>
</section>

<section class="section">
<h2>6. 年度结果</h2>
{table(annual_table)}
<img class="chart" style="max-height:125mm" src="{figure_uris['chart_annual_returns']}">
<p class="caption">图 3：2022 至 2025 策略与基准年度收益。</p>
<p>年度分解显示结果并非每年稳定跑赢。尤其 2025 年中证全指显著上涨时，策略虽为正收益，
但相对落后。报告保留该事实，避免只展示四年累计数字。</p>
</section>

<section class="figure-page">
<h2>7. 因子 IC 迁移</h2>
<img class="chart" src="{figure_uris['chart_factor_ic']}">
<p class="caption">图 4：研究、验证、最终 OOS 的平均月度 Rank IC。</p>
<p>因子跨阶段变化是模型风险的重要证据。P6 不因动量 OOS 表现转负而删除或重新加权它，
因为这会违反冻结规则并形成事后选择。</p>
</section>

<section class="section">
<h2>8. 事后稳健性实验</h2>
<p>以下五个完整回测变体均采用 10 bps 单边成本，运行相同 2016 至 2025 交易路径，
只报告 2022 至 2025 结果。它们全部是 <b>POST_OOS_ROBUSTNESS</b>。</p>
{table(robust_table)}
<p>微盘剔除检验收益是否依赖最小市值股票；9-1 动量与 120 日低波检验窗口敏感性；
行业中性化按月在行业内去均值后再标准化；规模中性化每月对
<code>log(total_mv_cny)</code> 回归取残差。</p>
</section>

<section class="figure-page">
<h2>9. 稳健性结果比较</h2>
<img class="chart" src="{figure_uris['chart_robustness']}">
<p class="caption">图 5：P5 锚点、五个重跑变体及冻结成本压力参考。</p>
<div class="callout">图中的最佳变体不能取代 P5。观察 OOS 后挑选表现最好的变体会产生
二次数据窥探，因此这里只讨论敏感性范围。</div>
</section>

<section class="section">
<h2>10. 退市期末回收率敏感性</h2>
<p>受影响证券为 000413.SZ、000666.SZ、000671.SZ。P6 只调整 P5 期末陈旧
账面价值，不推断退市日、现金到账日、订单、税费或中间净值路径。</p>
{table(sensitivity)}
<img class="chart" style="max-height:112mm" src="{figure_uris['chart_delisting']}">
<p class="caption">图 6：10 bps 场景期末回收率敏感性；100% 精确复现 P5。</p>
</section>

<section class="section">
<h2>11. 失败时期与订单诊断</h2>
<h3>最差十个策略月份</h3>
{table(worst_table)}
<h3>主要失败订单原因</h3>
{table(failure_table)}
<p class="small">月份收益、失败订单、换手、现金和陈旧估值是共同诊断线索，本报告不把
同月共现直接解释为因果。</p>
</section>

<section class="section">
<h2>12. 复现、限制与结论</h2>
<h3>复现入口</h3>
<pre><code>python scripts/build_p6.py
python -m pytest -q
python scripts/audit_p6.py</code></pre>
<p>机器可读结果位于 <code>results/metrics.json</code> 与
<code>results/experiment_registry.csv</code>；最终审计清单位于
<code>reports/robustness/p6_run_manifest.json</code>。</p>
<h3>必须同时披露的限制</h3>
<ul>
<li>三只退市证券缺少精确终止估值和回收事件。</li>
<li>历史 PB 修订政策未完全核验。</li>
<li>P6 是看到最终 OOS 后的稳健性研究，不构成新 OOS。</li>
<li>四年 OOS 市场环境覆盖有限，尚未建模容量曲线和盘口冲击。</li>
</ul>
<div class="success"><b>结论：</b>项目完成了数据语义、证券主体、交易约束、公司行动、
配置冻结、一次性最终 OOS、事后稳健性隔离和证据映射的完整闭环。最终状态为
<code>{P6_STATUS}</code>。</div>
</section>
</body>
</html>"""


def _compile_pdf(control: dict[str, Any]) -> dict[str, Any]:
    html_path = absolute("tmp/pdfs/research_report.html")
    pdf_path = absolute(control["outputs"]["research_report_pdf"])
    rendered_path = pdf_path.with_name("research_report.rendering.pdf")
    discovered_edge = shutil.which("msedge")
    program_roots = [
        value
        for value in (
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("ProgramFiles"),
        )
        if value
    ]
    edge_candidates = [
        Path(root) / "Microsoft" / "Edge" / "Application" / "msedge.exe"
        for root in program_roots
    ]
    edge = (
        Path(discovered_edge)
        if discovered_edge
        else next(
            (path for path in edge_candidates if path.is_file()),
            None,
        )
    )
    if edge is None:
        raise RuntimeError("Microsoft Edge is required to render the formal PDF")
    user_data = absolute(f".report_tmp/edge_pdf_profile_{os.getpid()}")
    user_data.mkdir(parents=True, exist_ok=True)
    if rendered_path.exists():
        rendered_path.unlink()
    command = [
        str(edge),
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-gpu-compositing",
        "--disable-software-rasterizer",
        "--disable-dev-shm-usage",
        "--disable-features=VizDisplayCompositor",
        "--allow-file-access-from-files",
        "--no-pdf-header-footer",
        f"--user-data-dir={user_data}",
        f"--print-to-pdf={rendered_path}",
        html_path.resolve().as_uri(),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=180,
        check=False,
    )
    if (
        completed.returncode != 0
        and (
            not rendered_path.is_file()
            or rendered_path.stat().st_size < 100_000
        )
    ):
        log_path = absolute("tmp/pdfs/research_report.build.log")
        log_path.write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        raise RuntimeError(
            f"Edge PDF rendering failed; inspect {log_path.relative_to(PROJECT_ROOT)}"
        )
    if not rendered_path.is_file() or rendered_path.stat().st_size < 100_000:
        raise RuntimeError("Compiled research report PDF is missing or too small")
    os.replace(rendered_path, pdf_path)
    copy_path = absolute(control["outputs"]["research_report_pdf_copy"])
    copy_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, copy_path)
    return {
        "renderer": "Microsoft Edge headless print",
        "renderer_path": str(edge),
        "renderer_return_code": completed.returncode,
        "html_source": str(html_path.relative_to(PROJECT_ROOT)),
        "tex_source_preserved": control["outputs"]["research_report_tex"],
        "pdf_size_bytes": pdf_path.stat().st_size,
        "pdf_sha256": _sha256(pdf_path),
        "copy_sha256": _sha256(copy_path),
    }


def build_p6() -> dict[str, Any]:
    started = datetime.now(UTC)
    start_clock = time.perf_counter()
    control = load_config()
    existing_manifest = absolute(control["outputs"]["p6_run_manifest"])
    if existing_manifest.exists():
        raise RuntimeError(
            "A completed P6 manifest already exists; refusing to overwrite the "
            "final project package."
        )
    _log("validating frozen P5 and protected inputs")
    frozen, runtime, p5_manifest, p5_hash_before, protected_before = _preflight(
        control
    )
    frozen_hash = control["project"]["expected_frozen_sha256"]
    _write_csv(p5_hash_before, control["outputs"]["p5_output_hash_check"])
    input_hashes = pd.DataFrame(
        [
            {"relative_path": path, **metadata}
            for path, metadata in protected_before.items()
        ]
    )
    _write_csv(input_hashes, control["outputs"]["p6_input_hashes"])

    baseline_panel = pd.read_parquet(
        absolute(control["inputs"]["p5_factor_panel"])
    )
    benchmark = pd.read_parquet(absolute(control["inputs"]["benchmark_daily"]))
    benchmark["trade_date"] = pd.to_datetime(benchmark["trade_date"])
    actions = _load_corporate_actions(runtime)
    stamp_policy = _load_stamp_policy(runtime)
    experiments = {
        item["experiment_id"]: item for item in control["experiments"]
    }
    metrics_rows = _reference_metrics(control, frozen_hash)
    annual_variants: list[pd.DataFrame] = []
    for experiment_id in (
        "MICROCAP_EXCLUDE_BOTTOM20",
        "MOMENTUM_9_1",
        "LOWVOL_120",
        "INDUSTRY_NEUTRAL",
        "LOG_SIZE_NEUTRAL",
    ):
        experiment = experiments[experiment_id]
        panel = _variant_factor_panel(
            runtime, experiment_id, experiment, baseline_panel
        )
        metric, _, annual = _run_variant(
            experiment_id,
            experiment,
            panel,
            runtime,
            frozen,
            benchmark,
            actions,
            stamp_policy,
        )
        metric["frozen_config_sha256"] = frozen_hash
        metrics_rows.append(metric)
        annual.insert(0, "experiment_id", experiment_id)
        annual_variants.append(annual)
    experiment_metrics = pd.DataFrame(metrics_rows)
    _write_csv(experiment_metrics, control["outputs"]["experiment_metrics"])
    _write_csv(
        pd.concat(annual_variants, ignore_index=True),
        "results/p6_robustness/variant_annual_performance.csv",
    )

    _log("building delisting and failure-period sensitivities")
    delisting = _delisting_sensitivity(control)
    _write_csv(delisting, control["outputs"]["delisting_sensitivity"])
    p5_daily = pd.read_parquet(
        absolute(control["inputs"]["p5_daily_portfolio"])
    )
    p5_orders = pd.read_parquet(absolute(control["inputs"]["p5_orders"]))
    p5_daily["trade_date"] = pd.to_datetime(p5_daily["trade_date"])
    p5_orders["trade_date"] = pd.to_datetime(p5_orders["trade_date"])
    monthly, worst, drawdowns = _monthly_diagnostics(p5_daily, p5_orders)
    failure = _failure_summary(
        p5_orders.loc[
            p5_orders["trade_date"].between("2022-01-01", "2025-12-31")
        ]
    )
    _write_csv(monthly, control["outputs"]["monthly_diagnostics"])
    _write_csv(worst, control["outputs"]["worst_months"])
    _write_csv(drawdowns, control["outputs"]["drawdown_episodes"])
    _write_csv(failure, control["outputs"]["failure_reason_summary"])

    _log("generating charts and written evidence package")
    _generate_charts(control, experiment_metrics, delisting)
    limitations, robustness, evidence, interview, notes = _report_materials(
        control,
        frozen_hash,
        experiment_metrics,
        delisting,
        monthly,
        worst,
        failure,
        p5_manifest,
    )
    _write_text(limitations, control["outputs"]["limitations"])
    _write_text(robustness, control["outputs"]["robustness_report"])
    _write_text(evidence, control["outputs"]["resume_evidence_map"])
    _write_text(interview, control["outputs"]["interview_notes"])
    _write_text(notes, "docs/LEARNING_NOTES.md")
    tex = _research_report_tex(
        control,
        frozen_hash,
        experiment_metrics,
        delisting,
        worst,
        failure,
        p5_manifest,
    )
    _write_text(tex, control["outputs"]["research_report_tex"])
    html_report = _research_report_html(
        control,
        frozen_hash,
        experiment_metrics,
        delisting,
        worst,
        failure,
        p5_manifest,
    )
    _write_text(html_report, "tmp/pdfs/research_report.html")
    _log("compiling the formal PDF report")
    pdf_metadata = _compile_pdf(control)

    primary = experiment_metrics.loc[
        experiment_metrics["experiment_id"] == "P5_BASELINE_REFERENCE"
    ].iloc[0]
    metrics_payload = {
        "project": {
            "name": "A股三因子横截面研究",
            "final_status": P6_STATUS,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "frozen_config_sha256": frozen_hash,
            "p5_original_preserved": True,
            "p6_classification": "POST_OOS_ROBUSTNESS",
        },
        "primary_p5_final_oos": _json_safe(primary.to_dict()),
        "factor_ic": {
            key: _json_safe(
                pd.read_csv(absolute(control["inputs"][path_key])).to_dict(
                    orient="records"
                )
            )
            for key, path_key in (
                ("research_2016_2019", "p2_ic_summary"),
                ("validation_2020_2021", "p4_ic_summary"),
                ("final_oos_2022_2025", "p5_ic_summary"),
            )
        },
        "robustness_experiments": _json_safe(
            experiment_metrics.to_dict(orient="records")
        ),
        "delisting_terminal_recovery_sensitivity": _json_safe(
            delisting.to_dict(orient="records")
        ),
        "worst_months": _json_safe(worst.to_dict(orient="records")),
        "limitations": [
            "Three delisted securities lack exact terminal recovery events.",
            "Historical PB revision policy is not fully verified.",
            "P6 is post-OOS robustness and cannot replace P5.",
        ],
        "resume_safe_claims_path": control["outputs"]["resume_evidence_map"],
        "formal_report_path": control["outputs"]["research_report_pdf"],
    }
    _write_json(metrics_payload, control["outputs"]["metrics_json"])

    preliminary_hashes = {
        relative: _sha256(absolute(relative))
        for relative in control["outputs"].values()
        if absolute(relative).is_file()
        and relative not in {
            control["outputs"]["p6_audit_summary"],
            control["outputs"]["p6_audit_report"],
            control["outputs"]["p6_run_manifest"],
        }
    }
    registry = _experiment_registry(
        control,
        experiment_metrics,
        delisting,
        preliminary_hashes,
        frozen_hash,
    )
    _write_csv(registry, control["outputs"]["experiment_registry"])

    _log("verifying that every protected P5 input remains unchanged")
    protected_after = _snapshot(control["protected_p6_inputs"])
    changed = [
        path for path in protected_before
        if protected_before[path] != protected_after[path]
    ]
    if changed:
        raise RuntimeError(f"P6 modified protected P5 inputs: {changed}")
    p5_hash_after, _, _ = _p5_output_hash_check(control)
    if not bool(p5_hash_after["matches"].all()):
        raise RuntimeError("P5 output hashes changed during P6")

    output_hashes = {
        relative: _sha256(absolute(relative))
        for relative in control["outputs"].values()
        if absolute(relative).is_file()
        and relative not in {
            control["outputs"]["p6_audit_summary"],
            control["outputs"]["p6_audit_report"],
            control["outputs"]["p6_run_manifest"],
        }
    }
    manifest = {
        "stage": "P6_ROBUSTNESS_REPORT_AND_CAREER",
        "builder_version": BUILDER_VERSION,
        "status": "P6_BUILT_PENDING_AUDIT",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - start_clock,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
            "duckdb": duckdb.__version__,
        },
        "authorization": control["project"]["authorization_reference"],
        "frozen_config_sha256": frozen_hash,
        "classification": "POST_OOS_ROBUSTNESS",
        "p5_status": p5_manifest["status"],
        "p5_original_preserved": True,
        "p5_output_hash_count": len(p5_hash_after),
        "p5_output_hashes_all_match": bool(p5_hash_after["matches"].all()),
        "protected_inputs_all_match": not changed,
        "experiment_count": len(registry),
        "rerun_variant_count": 5,
        "pdf": pdf_metadata,
        "counts": {
            "experiment_metric_rows": len(experiment_metrics),
            "delisting_sensitivity_rows": len(delisting),
            "monthly_diagnostic_rows": len(monthly),
            "worst_month_rows": len(worst),
            "failure_reason_rows": len(failure),
        },
        "output_sha256": output_hashes,
        "scope_guards": {
            "p5_overwritten": False,
            "p5_rerun": False,
            "p5_retuned": False,
            "delisting_path_rerun": False,
            "p6_results_used_as_new_oos": False,
            "other_project_started": False,
        },
        "disclosures": {
            "pb_revision_policy": "NEEDS_MANUAL_CONFIRMATION",
            "delisting_recovery": (
                "Terminal valuation sensitivity only; no missing event was inferred."
            ),
        },
        "audit": {
            "status": "PENDING",
            "pass_count": 0,
            "warn_count": 0,
            "fail_count": 0,
        },
    }
    _write_json(manifest, control["outputs"]["p6_run_manifest"])
    _log(
        "build complete; P5 preserved, "
        f"{len(registry)} experiment families registered"
    )
    return manifest
