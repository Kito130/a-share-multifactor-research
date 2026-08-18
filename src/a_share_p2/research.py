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

from .boundary import run_boundary_tests
from .config import CONFIG_PATH, PROJECT_ROOT, absolute, load_config, sql_path


BUILDER_VERSION = "p2.1"
FACTOR_COLUMNS = {
    "bm_proxy": {
        "raw": "bm_proxy",
        "winsorized": "bm_proxy_winsorized",
        "zscore": "bm_proxy_z",
    },
    "momentum_12_1": {
        "raw": "momentum_12_1",
        "winsorized": "momentum_12_1_winsorized",
        "zscore": "momentum_12_1_z",
    },
    "lowvol_60": {
        "raw": "lowvol_60",
        "winsorized": "lowvol_60_winsorized",
        "zscore": "lowvol_60_z",
    },
}


def _log(message: str) -> None:
    print(f"[P2] {message}", flush=True)


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


def _copy_query(
    connection: duckdb.DuckDBPyConnection, query: str, relative_path: str
) -> None:
    output = absolute(relative_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    temporary_relative = (
        temporary.resolve()
        .relative_to(PROJECT_ROOT.resolve())
        .as_posix()
        .replace("'", "''")
    )
    try:
        connection.execute(
            f"""
            COPY (
                {query}
            )
            TO '{temporary_relative}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
            """
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _input_snapshot(
    paths: Iterable[str],
) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for relative_path in paths:
        path = absolute(relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"P2受保护输入不存在：{relative_path}")
        snapshot[relative_path] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return snapshot


def _validate_stage_config(config: dict[str, Any]) -> None:
    project = config["project"]
    research_start = pd.Timestamp(project["research_start"])
    research_end = pd.Timestamp(project["research_end"])
    validation_start = pd.Timestamp(project["validation_start"])
    oos_start = pd.Timestamp(project["oos_start"])
    if not research_start <= research_end < validation_start < oos_start:
        raise ValueError("P2样本日期闸门无效")
    factors = config["factors"]
    if int(factors["momentum_recent_skip_observations"]) != 21:
        raise ValueError("P2冻结动量跳过窗口必须为21")
    if int(factors["momentum_lookback_observations"]) != 252:
        raise ValueError("P2冻结动量回看窗口必须为252")
    if int(factors["lowvol_window_observations"]) != 60:
        raise ValueError("P2冻结低波窗口必须为60")
    if factors["lowvol_standard_deviation"] != "sample":
        raise ValueError("P2低波标准差口径必须显式固定为sample")
    if factors["zscore_standard_deviation"] != "sample":
        raise ValueError("P2标准化标准差口径必须显式固定为sample")


def _factor_panel_query(config: dict[str, Any]) -> str:
    inputs = config["inputs"]
    project = config["project"]
    factor = config["factors"]
    universe = config["universe"]
    daily_path = sql_path(inputs["daily_panel"])
    monthly_path = sql_path(inputs["month_end_base_panel"])
    warmup_start = project["warmup_start"]
    research_start = project["research_start"]
    research_end = project["research_end"]
    recent_skip = int(factor["momentum_recent_skip_observations"])
    lookback = int(factor["momentum_lookback_observations"])
    lowvol_window = int(factor["lowvol_window_observations"])
    liquidity_window = int(universe["liquidity_window_observations"])
    listing_age = int(universe["listing_age_minimum_trading_days"])
    liquidity_quantile = float(
        universe["liquidity_excluded_lower_quantile"]
    )
    winsor_low = float(factor["winsor_lower_quantile"])
    winsor_high = float(factor["winsor_upper_quantile"])
    lowvol_preceding = lowvol_window - 1
    liquidity_preceding = liquidity_window - 1
    exclude_st = bool(universe["exclude_historical_st_name"])
    st_clause = "AND NOT coalesce(is_st_name_flag, false)" if exclude_st else ""

    return f"""
        WITH valid_daily AS (
            SELECT
                ts_code,
                canonical_ts_code,
                trade_date,
                adjusted_close,
                amount_cny
            FROM read_parquet('{daily_path}')
            WHERE trade_date BETWEEN DATE '{warmup_start}'
                                 AND DATE '{research_end}'
              AND is_sh_sz
              AND is_within_listing_window
              AND security_code_interval_valid
              AND adjusted_close IS NOT NULL
              AND adjusted_close > 0
        ),
        canonical_returns AS (
            SELECT
                *,
                adjusted_close
                    / lag(adjusted_close) OVER (
                        PARTITION BY canonical_ts_code
                        ORDER BY trade_date
                    ) - 1.0 AS canonical_daily_return
            FROM valid_daily
        ),
        daily_features AS (
            SELECT
                *,
                count(*) OVER (
                    PARTITION BY canonical_ts_code
                    ORDER BY trade_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS canonical_price_observations,
                lag(adjusted_close, {recent_skip}) OVER (
                    PARTITION BY canonical_ts_code
                    ORDER BY trade_date
                ) AS adjusted_price_t_minus_21,
                lag(adjusted_close, {lookback}) OVER (
                    PARTITION BY canonical_ts_code
                    ORDER BY trade_date
                ) AS adjusted_price_t_minus_252,
                count(canonical_daily_return) OVER (
                    PARTITION BY canonical_ts_code
                    ORDER BY trade_date
                    ROWS BETWEEN {lowvol_preceding} PRECEDING
                             AND CURRENT ROW
                ) AS lowvol_observations,
                stddev_samp(canonical_daily_return) OVER (
                    PARTITION BY canonical_ts_code
                    ORDER BY trade_date
                    ROWS BETWEEN {lowvol_preceding} PRECEDING
                             AND CURRENT ROW
                ) AS rolling_return_std_60,
                count(amount_cny) OVER (
                    PARTITION BY canonical_ts_code
                    ORDER BY trade_date
                    ROWS BETWEEN {liquidity_preceding} PRECEDING
                             AND CURRENT ROW
                ) AS canonical_liquidity_observations,
                avg(amount_cny) OVER (
                    PARTITION BY canonical_ts_code
                    ORDER BY trade_date
                    ROWS BETWEEN {liquidity_preceding} PRECEDING
                             AND CURRENT ROW
                ) AS canonical_liquidity_20d
            FROM canonical_returns
        ),
        monthly_joined AS (
            SELECT
                monthly.ts_code,
                monthly.canonical_ts_code,
                monthly.signal_date,
                daily.adjusted_close AS adjusted_price,
                daily.canonical_daily_return,
                daily.canonical_price_observations,
                daily.adjusted_price_t_minus_21,
                daily.adjusted_price_t_minus_252,
                daily.lowvol_observations,
                daily.rolling_return_std_60,
                daily.canonical_liquidity_observations,
                daily.canonical_liquidity_20d,
                monthly.pb,
                monthly.total_mv_cny,
                monthly.circ_mv_cny,
                monthly.industry_code,
                monthly.industry_name,
                monthly.is_st_name_flag,
                monthly.listing_age_trading_days,
                monthly.listing_age_is_lower_bound,
                monthly.listing_age_status,
                monthly.has_listing_reference,
                monthly.is_within_listing_window,
                monthly.security_code_interval_valid,
                monthly.is_sh_sz
            FROM read_parquet('{monthly_path}') AS monthly
            INNER JOIN daily_features AS daily
              ON monthly.ts_code = daily.ts_code
             AND monthly.canonical_ts_code = daily.canonical_ts_code
             AND monthly.signal_date = daily.trade_date
            WHERE monthly.signal_date BETWEEN DATE '{research_start}'
                                          AND DATE '{research_end}'
              AND monthly.is_sh_sz
              AND monthly.is_within_listing_window
              AND monthly.security_code_interval_valid
        ),
        monthly_with_next AS (
            SELECT
                *,
                lead(signal_date) OVER (
                    PARTITION BY canonical_ts_code
                    ORDER BY signal_date
                ) AS next_signal_date_candidate,
                lead(adjusted_price) OVER (
                    PARTITION BY canonical_ts_code
                    ORDER BY signal_date
                ) AS next_adjusted_price_candidate
            FROM monthly_joined
        ),
        raw_factors AS (
            SELECT
                *,
                CASE
                    WHEN date_diff(
                        'month', signal_date, next_signal_date_candidate
                    ) = 1
                    THEN next_signal_date_candidate
                END AS next_signal_date,
                CASE
                    WHEN date_diff(
                        'month', signal_date, next_signal_date_candidate
                    ) = 1
                     AND next_adjusted_price_candidate > 0
                    THEN next_adjusted_price_candidate
                END AS next_adjusted_price,
                CASE
                    WHEN date_diff(
                        'month', signal_date, next_signal_date_candidate
                    ) = 1
                     AND next_adjusted_price_candidate > 0
                    THEN next_adjusted_price_candidate
                         / adjusted_price - 1.0
                END AS next_month_return,
                CASE WHEN pb > 0 THEN 1.0 / pb END AS bm_proxy,
                CASE
                    WHEN adjusted_price_t_minus_21 > 0
                     AND adjusted_price_t_minus_252 > 0
                    THEN adjusted_price_t_minus_21
                         / adjusted_price_t_minus_252 - 1.0
                END AS momentum_12_1,
                CASE
                    WHEN lowvol_observations = {lowvol_window}
                    THEN -rolling_return_std_60
                END AS lowvol_60
            FROM monthly_with_next
        ),
        pre_universe AS (
            SELECT
                *,
                (
                    is_sh_sz
                    AND is_within_listing_window
                    AND security_code_interval_valid
                    AND has_listing_reference
                    AND listing_age_trading_days >= {listing_age}
                    AND adjusted_price > 0
                    AND pb > 0
                    AND bm_proxy IS NOT NULL
                    AND momentum_12_1 IS NOT NULL
                    AND lowvol_60 IS NOT NULL
                    AND canonical_liquidity_observations = {liquidity_window}
                    AND canonical_liquidity_20d IS NOT NULL
                    {st_clause}
                ) AS universe_pre_liquidity
            FROM raw_factors
        ),
        liquidity_cutoffs AS (
            SELECT
                *,
                quantile_cont(
                    canonical_liquidity_20d, {liquidity_quantile}
                ) FILTER (
                    WHERE universe_pre_liquidity
                ) OVER (
                    PARTITION BY signal_date
                ) AS liquidity_20pct_cutoff
            FROM pre_universe
        ),
        eligible AS (
            SELECT
                *,
                (
                    universe_pre_liquidity
                    AND canonical_liquidity_20d >= liquidity_20pct_cutoff
                ) AS universe_eligible
            FROM liquidity_cutoffs
        ),
        winsor_limits AS (
            SELECT
                *,
                quantile_cont(bm_proxy, {winsor_low}) FILTER (
                    WHERE universe_eligible
                ) OVER (PARTITION BY signal_date) AS bm_proxy_p01,
                quantile_cont(bm_proxy, {winsor_high}) FILTER (
                    WHERE universe_eligible
                ) OVER (PARTITION BY signal_date) AS bm_proxy_p99,
                quantile_cont(momentum_12_1, {winsor_low}) FILTER (
                    WHERE universe_eligible
                ) OVER (PARTITION BY signal_date) AS momentum_12_1_p01,
                quantile_cont(momentum_12_1, {winsor_high}) FILTER (
                    WHERE universe_eligible
                ) OVER (PARTITION BY signal_date) AS momentum_12_1_p99,
                quantile_cont(lowvol_60, {winsor_low}) FILTER (
                    WHERE universe_eligible
                ) OVER (PARTITION BY signal_date) AS lowvol_60_p01,
                quantile_cont(lowvol_60, {winsor_high}) FILTER (
                    WHERE universe_eligible
                ) OVER (PARTITION BY signal_date) AS lowvol_60_p99
            FROM eligible
        ),
        winsorized AS (
            SELECT
                *,
                CASE WHEN universe_eligible THEN greatest(
                    bm_proxy_p01, least(bm_proxy_p99, bm_proxy)
                ) END AS bm_proxy_winsorized,
                CASE WHEN universe_eligible THEN greatest(
                    momentum_12_1_p01,
                    least(momentum_12_1_p99, momentum_12_1)
                ) END AS momentum_12_1_winsorized,
                CASE WHEN universe_eligible THEN greatest(
                    lowvol_60_p01, least(lowvol_60_p99, lowvol_60)
                ) END AS lowvol_60_winsorized
            FROM winsor_limits
        ),
        standardization_parameters AS (
            SELECT
                *,
                avg(bm_proxy_winsorized) OVER (
                    PARTITION BY signal_date
                ) AS bm_proxy_cross_section_mean,
                stddev_samp(bm_proxy_winsorized) OVER (
                    PARTITION BY signal_date
                ) AS bm_proxy_cross_section_std,
                avg(momentum_12_1_winsorized) OVER (
                    PARTITION BY signal_date
                ) AS momentum_12_1_cross_section_mean,
                stddev_samp(momentum_12_1_winsorized) OVER (
                    PARTITION BY signal_date
                ) AS momentum_12_1_cross_section_std,
                avg(lowvol_60_winsorized) OVER (
                    PARTITION BY signal_date
                ) AS lowvol_60_cross_section_mean,
                stddev_samp(lowvol_60_winsorized) OVER (
                    PARTITION BY signal_date
                ) AS lowvol_60_cross_section_std
            FROM winsorized
        )
        SELECT
            ts_code,
            canonical_ts_code,
            signal_date,
            next_signal_date,
            adjusted_price,
            next_adjusted_price,
            next_month_return,
            canonical_daily_return,
            canonical_price_observations,
            adjusted_price_t_minus_21,
            adjusted_price_t_minus_252,
            lowvol_observations,
            rolling_return_std_60,
            canonical_liquidity_observations,
            canonical_liquidity_20d,
            liquidity_20pct_cutoff,
            pb,
            bm_proxy,
            momentum_12_1,
            lowvol_60,
            bm_proxy_p01,
            bm_proxy_p99,
            momentum_12_1_p01,
            momentum_12_1_p99,
            lowvol_60_p01,
            lowvol_60_p99,
            bm_proxy_winsorized,
            momentum_12_1_winsorized,
            lowvol_60_winsorized,
            CASE
                WHEN universe_eligible
                 AND bm_proxy_cross_section_std > 0
                THEN (
                    bm_proxy_winsorized - bm_proxy_cross_section_mean
                ) / bm_proxy_cross_section_std
            END AS bm_proxy_z,
            CASE
                WHEN universe_eligible
                 AND momentum_12_1_cross_section_std > 0
                THEN (
                    momentum_12_1_winsorized
                    - momentum_12_1_cross_section_mean
                ) / momentum_12_1_cross_section_std
            END AS momentum_12_1_z,
            CASE
                WHEN universe_eligible
                 AND lowvol_60_cross_section_std > 0
                THEN (
                    lowvol_60_winsorized
                    - lowvol_60_cross_section_mean
                ) / lowvol_60_cross_section_std
            END AS lowvol_60_z,
            bm_proxy_cross_section_mean,
            bm_proxy_cross_section_std,
            momentum_12_1_cross_section_mean,
            momentum_12_1_cross_section_std,
            lowvol_60_cross_section_mean,
            lowvol_60_cross_section_std,
            total_mv_cny,
            circ_mv_cny,
            industry_code,
            industry_name,
            coalesce(is_st_name_flag, false) AS is_st_name_flag,
            listing_age_trading_days,
            listing_age_is_lower_bound,
            listing_age_status,
            has_listing_reference,
            is_within_listing_window,
            security_code_interval_valid,
            is_sh_sz,
            universe_pre_liquidity,
            universe_eligible,
            (next_month_return IS NOT NULL) AS has_next_month_label
        FROM standardization_parameters
        ORDER BY signal_date, canonical_ts_code
    """


def _safe_corr(
    frame: pd.DataFrame,
    left: str,
    right: str,
    method: str,
    minimum: int,
) -> tuple[float, int]:
    valid = frame[[left, right]].dropna()
    observations = len(valid)
    if (
        observations < minimum
        or valid[left].nunique() < 2
        or valid[right].nunique() < 2
    ):
        return math.nan, observations
    return float(valid[left].corr(valid[right], method=method)), observations


def _assign_quintiles(
    frame: pd.DataFrame, z_column: str, quintiles: int
) -> pd.Series:
    ordered = frame.sort_values("canonical_ts_code")
    ranks = ordered[z_column].rank(method="average")
    labels = np.ceil(ranks / len(ordered) * quintiles)
    labels = labels.clip(1, quintiles).astype("int64")
    return labels.reindex(frame.index)


def _coverage(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal_date, month in panel.groupby("signal_date", sort=True):
        base_rows = len(month)
        eligible = month["universe_eligible"].fillna(False)
        labeled = eligible & month["next_month_return"].notna()
        for factor_name, columns in FACTOR_COLUMNS.items():
            raw_available = int(month[columns["raw"]].notna().sum())
            eligible_rows = int(eligible.sum())
            evaluation_rows = int(
                (labeled & month[columns["zscore"]].notna()).sum()
            )
            rows.append(
                {
                    "signal_date": signal_date,
                    "factor": factor_name,
                    "base_rows": base_rows,
                    "raw_factor_rows": raw_available,
                    "raw_factor_coverage_rate": (
                        raw_available / base_rows if base_rows else math.nan
                    ),
                    "universe_pre_liquidity_rows": int(
                        month["universe_pre_liquidity"].fillna(False).sum()
                    ),
                    "universe_eligible_rows": eligible_rows,
                    "universe_eligible_rate": (
                        eligible_rows / base_rows if base_rows else math.nan
                    ),
                    "evaluation_rows": evaluation_rows,
                    "evaluation_rate_of_eligible": (
                        evaluation_rows / eligible_rows
                        if eligible_rows
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _monthly_ic(
    panel: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    method = config["statistics"]["rank_ic_method"]
    minimum = int(
        config["statistics"]["minimum_cross_section_observations"]
    )
    rows: list[dict[str, Any]] = []
    for signal_date, month in panel.groupby("signal_date", sort=True):
        evaluation = month.loc[
            month["universe_eligible"].fillna(False)
            & month["next_month_return"].notna()
        ]
        for factor_name, columns in FACTOR_COLUMNS.items():
            value, observations = _safe_corr(
                evaluation,
                columns["zscore"],
                "next_month_return",
                method,
                minimum,
            )
            rows.append(
                {
                    "signal_date": signal_date,
                    "factor": factor_name,
                    "rank_ic": value,
                    "observations": observations,
                }
            )
    return pd.DataFrame(rows)


def _ic_summary(
    monthly_ic: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    annualizer = float(config["statistics"]["icir_annualization_factor"])
    rows: list[dict[str, Any]] = []
    for factor_name, group in monthly_ic.groupby("factor", sort=True):
        values = group["rank_ic"].dropna()
        standard_deviation = values.std(ddof=1)
        rows.append(
            {
                "factor": factor_name,
                "months": len(values),
                "mean_rank_ic": values.mean(),
                "rank_ic_std": standard_deviation,
                "rank_icir_annualized": (
                    values.mean() / standard_deviation * annualizer
                    if pd.notna(standard_deviation)
                    and standard_deviation > 0
                    else math.nan
                ),
                "positive_rank_ic_rate": (values > 0).mean(),
                "minimum_rank_ic": values.min(),
                "maximum_rank_ic": values.max(),
            }
        )
    return pd.DataFrame(rows)


def _quintile_returns(
    panel: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    quintile_count = int(config["statistics"]["quintiles"])
    minimum = int(
        config["statistics"]["minimum_cross_section_observations"]
    )
    rows: list[dict[str, Any]] = []
    for signal_date, month in panel.groupby("signal_date", sort=True):
        evaluation = month.loc[
            month["universe_eligible"].fillna(False)
            & month["next_month_return"].notna()
        ].copy()
        for factor_name, columns in FACTOR_COLUMNS.items():
            valid = evaluation.loc[
                evaluation[columns["zscore"]].notna()
            ].copy()
            if len(valid) < max(minimum, quintile_count):
                continue
            valid["quintile"] = _assign_quintiles(
                valid, columns["zscore"], quintile_count
            )
            month_rows: dict[int, dict[str, Any]] = {}
            for quintile, bucket in valid.groupby("quintile", sort=True):
                record = {
                    "signal_date": signal_date,
                    "factor": factor_name,
                    "quintile": f"Q{int(quintile)}",
                    "mean_next_month_return": bucket[
                        "next_month_return"
                    ].mean(),
                    "observations": len(bucket),
                    "return_type": "DIAGNOSTIC_FORWARD_RETURN_NO_COST",
                }
                rows.append(record)
                month_rows[int(quintile)] = record
            if 1 in month_rows and quintile_count in month_rows:
                rows.append(
                    {
                        "signal_date": signal_date,
                        "factor": factor_name,
                        "quintile": "TOP_MINUS_BOTTOM",
                        "mean_next_month_return": (
                            month_rows[quintile_count][
                                "mean_next_month_return"
                            ]
                            - month_rows[1]["mean_next_month_return"]
                        ),
                        "observations": min(
                            month_rows[quintile_count]["observations"],
                            month_rows[1]["observations"],
                        ),
                        "return_type": (
                            "DIAGNOSTIC_FORWARD_RETURN_NO_COST"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _annual_results(
    monthly_ic: pd.DataFrame,
    quintile_returns: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    annualizer = float(config["statistics"]["icir_annualization_factor"])
    ic = monthly_ic.copy()
    ic["year"] = pd.to_datetime(ic["signal_date"]).dt.year
    spread = quintile_returns.loc[
        quintile_returns["quintile"] == "TOP_MINUS_BOTTOM"
    ].copy()
    spread["year"] = pd.to_datetime(spread["signal_date"]).dt.year
    rows: list[dict[str, Any]] = []
    keys = sorted(
        set(zip(ic["factor"], ic["year"], strict=False))
        | set(zip(spread["factor"], spread["year"], strict=False))
    )
    for factor_name, year in keys:
        ic_values = ic.loc[
            (ic["factor"] == factor_name) & (ic["year"] == year),
            "rank_ic",
        ].dropna()
        spread_values = spread.loc[
            (spread["factor"] == factor_name)
            & (spread["year"] == year),
            "mean_next_month_return",
        ].dropna()
        ic_std = ic_values.std(ddof=1)
        rows.append(
            {
                "factor": factor_name,
                "year": int(year),
                "rank_ic_months": len(ic_values),
                "mean_rank_ic": ic_values.mean(),
                "rank_ic_std": ic_std,
                "rank_icir_annualized": (
                    ic_values.mean() / ic_std * annualizer
                    if pd.notna(ic_std) and ic_std > 0
                    else math.nan
                ),
                "positive_rank_ic_rate": (ic_values > 0).mean(),
                "spread_months": len(spread_values),
                "mean_top_minus_bottom": spread_values.mean(),
                "compounded_top_minus_bottom": (
                    (1.0 + spread_values).prod() - 1.0
                    if len(spread_values)
                    else math.nan
                ),
                "positive_top_minus_bottom_rate": (
                    (spread_values > 0).mean()
                ),
                "return_type": "DIAGNOSTIC_FORWARD_RETURN_NO_COST",
            }
        )
    return pd.DataFrame(rows)


def _factor_correlations(
    panel: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    minimum = int(
        config["statistics"]["minimum_cross_section_observations"]
    )
    factor_names = list(FACTOR_COLUMNS)
    monthly_rows: list[dict[str, Any]] = []
    for signal_date, month in panel.groupby("signal_date", sort=True):
        eligible = month.loc[month["universe_eligible"].fillna(False)]
        for left_index, left in enumerate(factor_names):
            for right in factor_names[left_index + 1 :]:
                value, observations = _safe_corr(
                    eligible,
                    FACTOR_COLUMNS[left]["zscore"],
                    FACTOR_COLUMNS[right]["zscore"],
                    "spearman",
                    minimum,
                )
                monthly_rows.append(
                    {
                        "signal_date": signal_date,
                        "factor_left": left,
                        "factor_right": right,
                        "spearman_correlation": value,
                        "observations": observations,
                    }
                )
    monthly = pd.DataFrame(monthly_rows)
    summary_rows: list[dict[str, Any]] = []
    for (left, right), group in monthly.groupby(
        ["factor_left", "factor_right"], sort=True
    ):
        values = group["spearman_correlation"].dropna()
        summary_rows.append(
            {
                "factor_left": left,
                "factor_right": right,
                "months": len(values),
                "mean_spearman_correlation": values.mean(),
                "std_spearman_correlation": values.std(ddof=1),
                "minimum_spearman_correlation": values.min(),
                "maximum_spearman_correlation": values.max(),
            }
        )
    return monthly, pd.DataFrame(summary_rows)


def _industry_exposure(
    panel: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    quintile_count = int(config["statistics"]["quintiles"])
    rows: list[dict[str, Any]] = []
    for signal_date, month in panel.groupby("signal_date", sort=True):
        eligible = month.loc[
            month["universe_eligible"].fillna(False)
        ].copy()
        eligible["industry_code_for_report"] = eligible[
            "industry_code"
        ].fillna("UNCLASSIFIED")
        eligible["industry_name_for_report"] = eligible[
            "industry_name"
        ].fillna("未分类")
        universe_count = len(eligible)
        if not universe_count:
            continue
        for factor_name, columns in FACTOR_COLUMNS.items():
            valid = eligible.loc[
                eligible[columns["zscore"]].notna()
            ].copy()
            if len(valid) < quintile_count:
                continue
            valid["quintile"] = _assign_quintiles(
                valid, columns["zscore"], quintile_count
            )
            top = valid["quintile"] == quintile_count
            top_count = int(top.sum())
            grouped = valid.groupby(
                ["industry_code_for_report", "industry_name_for_report"],
                dropna=False,
                sort=True,
            )
            for (industry_code, industry_name), industry in grouped:
                industry_top = int(
                    (industry["quintile"] == quintile_count).sum()
                )
                universe_weight = len(industry) / len(valid)
                top_weight = (
                    industry_top / top_count if top_count else math.nan
                )
                rows.append(
                    {
                        "signal_date": signal_date,
                        "factor": factor_name,
                        "industry_code": industry_code,
                        "industry_name": industry_name,
                        "universe_count": len(industry),
                        "top_quintile_count": industry_top,
                        "universe_weight": universe_weight,
                        "top_quintile_weight": top_weight,
                        "active_weight": top_weight - universe_weight,
                        "mean_factor_z": industry[
                            columns["zscore"]
                        ].mean(),
                    }
                )
    return pd.DataFrame(rows)


def _size_exposure(
    panel: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    minimum = int(
        config["statistics"]["minimum_cross_section_observations"]
    )
    rows: list[dict[str, Any]] = []
    for signal_date, month in panel.groupby("signal_date", sort=True):
        eligible = month.loc[
            month["universe_eligible"].fillna(False)
            & (month["total_mv_cny"] > 0)
        ].copy()
        eligible["log_total_mv_cny"] = np.log(
            eligible["total_mv_cny"]
        )
        for factor_name, columns in FACTOR_COLUMNS.items():
            pearson, observations = _safe_corr(
                eligible,
                columns["zscore"],
                "log_total_mv_cny",
                "pearson",
                minimum,
            )
            spearman, _ = _safe_corr(
                eligible,
                columns["zscore"],
                "log_total_mv_cny",
                "spearman",
                minimum,
            )
            rows.append(
                {
                    "signal_date": signal_date,
                    "factor": factor_name,
                    "pearson_corr_log_total_mv": pearson,
                    "spearman_corr_log_total_mv": spearman,
                    "observations": observations,
                }
            )
    return pd.DataFrame(rows)


def _worst_periods(
    monthly_ic: pd.DataFrame,
    quintile_returns: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    count = int(config["statistics"]["worst_periods_per_factor_metric"])
    rows: list[dict[str, Any]] = []
    spread = quintile_returns.loc[
        quintile_returns["quintile"] == "TOP_MINUS_BOTTOM"
    ]
    for factor_name in FACTOR_COLUMNS:
        for metric, source, value_column in (
            (
                "RANK_IC",
                monthly_ic.loc[monthly_ic["factor"] == factor_name],
                "rank_ic",
            ),
            (
                "TOP_MINUS_BOTTOM",
                spread.loc[spread["factor"] == factor_name],
                "mean_next_month_return",
            ),
        ):
            worst = source.dropna(subset=[value_column]).nsmallest(
                count, value_column
            )
            for rank, (_, record) in enumerate(
                worst.iterrows(), start=1
            ):
                rows.append(
                    {
                        "factor": factor_name,
                        "metric": metric,
                        "worst_rank": rank,
                        "signal_date": record["signal_date"],
                        "value": record[value_column],
                    }
                )
    return pd.DataFrame(rows)


def _build_statistics(
    panel: pd.DataFrame, config: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    coverage = _coverage(panel)
    monthly_ic = _monthly_ic(panel, config)
    ic_summary = _ic_summary(monthly_ic, config)
    quintile_returns = _quintile_returns(panel, config)
    annual_results = _annual_results(
        monthly_ic, quintile_returns, config
    )
    correlations_monthly, correlations_summary = _factor_correlations(
        panel, config
    )
    return {
        "factor_coverage": coverage,
        "monthly_rank_ic": monthly_ic,
        "ic_summary": ic_summary,
        "quintile_returns": quintile_returns,
        "annual_results": annual_results,
        "factor_correlations_monthly": correlations_monthly,
        "factor_correlations_summary": correlations_summary,
        "industry_exposure": _industry_exposure(panel, config),
        "size_exposure": _size_exposure(panel, config),
        "worst_periods": _worst_periods(
            monthly_ic, quintile_returns, config
        ),
    }


def _config_sha256() -> str:
    return _sha256(PROJECT_ROOT / CONFIG_PATH)


def build_p2() -> dict[str, Any]:
    started_at = datetime.now(UTC)
    config = load_config()
    _validate_stage_config(config)
    protected_paths = list(config["protected_p2_inputs"])
    _log("记录P1输入哈希")
    before = _input_snapshot(protected_paths)

    _log("执行跨代码价格边界闸门")
    boundary_summary, boundary_tests = run_boundary_tests()
    if not bool((boundary_tests["status"] == "PASS").all()):
        raise RuntimeError("P2边界闸门未全部通过")

    _log("构建研究期单因子面板（不含验证期与OOS）")
    with duckdb.connect() as connection:
        connection.execute("SET preserve_insertion_order = false")
        connection.execute("SET threads = 4")
        _copy_query(
            connection,
            _factor_panel_query(config),
            config["outputs"]["single_factor_panel"],
        )
        panel_path = sql_path(config["outputs"]["single_factor_panel"])
        panel_counts = connection.execute(
            f"""
            SELECT
                count(*) AS rows,
                count(DISTINCT signal_date) AS months,
                count(*) FILTER (WHERE universe_eligible)
                    AS eligible_rows,
                min(signal_date) AS minimum_signal_date,
                max(signal_date) AS maximum_signal_date,
                count(*) FILTER (
                    WHERE signal_date >= DATE
                        '{config["project"]["validation_start"]}'
                ) AS validation_or_later_rows,
                count(*) FILTER (
                    WHERE signal_date >= DATE
                        '{config["project"]["oos_start"]}'
                ) AS oos_rows
            FROM read_parquet('{panel_path}')
            """
        ).fetchone()

    _log("计算覆盖率、Rank IC、五分位及暴露统计")
    panel = pd.read_parquet(
        absolute(config["outputs"]["single_factor_panel"])
    )
    panel["signal_date"] = pd.to_datetime(panel["signal_date"])
    panel["next_signal_date"] = pd.to_datetime(panel["next_signal_date"])
    statistics = _build_statistics(panel, config)
    for output_key, frame in statistics.items():
        _write_csv_atomic(frame, config["outputs"][output_key])

    _log("复核P1输入哈希")
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
    _write_csv_atomic(
        comparison, config["outputs"]["p2_input_hashes"]
    )
    if not bool(comparison["match"].all()):
        raise RuntimeError("P2构建期间P1输入发生变化")

    generated_keys = [
        "security_code_boundary_summary",
        "security_code_boundary_tests",
        "security_code_boundary_report",
        "single_factor_panel",
        *statistics.keys(),
        "p2_input_hashes",
    ]
    output_hashes = {
        key: _sha256(absolute(config["outputs"][key]))
        for key in generated_keys
    }
    completed_at = datetime.now(UTC)
    manifest: dict[str, Any] = {
        "stage": "P2_SINGLE_FACTOR_RESEARCH",
        "builder_version": BUILDER_VERSION,
        "status": "BUILT_PENDING_AUDIT",
        "project_root_policy": (
            "runtime_discovery_and_project_relative_config_only"
        ),
        "sample": {
            "warmup_start": config["project"]["warmup_start"],
            "research_start": config["project"]["research_start"],
            "research_end": config["project"]["research_end"],
            "validation_start_guard": config["project"][
                "validation_start"
            ],
            "oos_start_guard": config["project"]["oos_start"],
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
        "config_sha256": _config_sha256(),
        "frozen_methodology": {
            "bm_proxy": "1 / pb, only pb > 0",
            "momentum_12_1": (
                "adjusted_price(t-21) / adjusted_price(t-252) - 1"
            ),
            "lowvol_60": "-sample_std(canonical_daily_return[t-59:t])",
            "next_month_return": (
                "next_calendar_month_end_adjusted_price / "
                "signal_adjusted_price - 1; evaluation label only"
            ),
            "winsorization": (
                f"monthly cross-section "
                f"{config['factors']['winsor_lower_quantile']}/"
                f"{config['factors']['winsor_upper_quantile']}"
            ),
            "standardization": (
                "monthly cross-section sample-std z-score"
            ),
            "rank_ic": "monthly Spearman correlation",
            "icir": "mean(IC)/sample_std(IC)*sqrt(12)",
            "quintiles": (
                "equal-count rank buckets; diagnostic forward returns, "
                "not a tradable portfolio or backtest"
            ),
        },
        "boundary_gate": {
            "mapped_entities": len(boundary_summary),
            "pass_tests": int(
                (boundary_tests["status"] == "PASS").sum()
            ),
            "fail_tests": int(
                (boundary_tests["status"] == "FAIL").sum()
            ),
            "rule": (
                "only security_code_interval_valid rows are stitched "
                "within canonical_ts_code"
            ),
        },
        "protected_inputs_all_match": bool(comparison["match"].all()),
        "panel": {
            "rows": int(panel_counts[0]),
            "months": int(panel_counts[1]),
            "eligible_rows": int(panel_counts[2]),
            "minimum_signal_date": str(panel_counts[3]),
            "maximum_signal_date": str(panel_counts[4]),
            "validation_or_later_rows": int(panel_counts[5]),
            "oos_rows": int(panel_counts[6]),
        },
        "output_sha256": output_hashes,
        "scope_guards": {
            "validation_period_evaluated": False,
            "oos_period_read_or_evaluated": False,
            "composite_factor_computed": False,
            "tradable_portfolio_built": False,
            "orders_generated": False,
            "transaction_costs_applied": False,
            "backtest_run": False,
        },
        "disclosures": [
            (
                "使用供应商历史PB构造1/PB代理，未自行重建严格 "
                "point-in-time book equity，供应商历史修订政策未完全核验。"
            ),
            (
                "P2五分位结果是无成本的下一月收益诊断，不是P3交易组合、"
                "成交模拟或回测。"
            ),
        ],
    }
    _write_json_atomic(manifest, config["outputs"]["p2_run_manifest"])
    _log(
        "构建完成："
        f"panel={panel_counts[0]:,}，"
        f"eligible={panel_counts[2]:,}，"
        "validation/OOS=0；等待P2审计"
    )
    return manifest

