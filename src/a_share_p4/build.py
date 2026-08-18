from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd
import pyarrow
import yaml

from a_share_p2.research import (
    _build_statistics,
    _factor_panel_query,
)
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

from .config import CONFIG_PATH, PROJECT_ROOT, absolute, load_config


BUILDER_VERSION = "p4.1"
PB_DISCLOSURE = (
    "使用供应商历史PB构造1/PB代理，未自行重建严格 "
    "point-in-time book equity，供应商历史修订政策未完全核验。"
)


def _log(message: str) -> None:
    print(f"[P4] {message}", flush=True)


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


def _write_parquet_atomic(
    frame: pd.DataFrame, relative_path: str
) -> None:
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


def _write_text_atomic(text: str, relative_path: str) -> None:
    output = absolute(relative_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, output)


def _write_yaml_atomic(
    payload: dict[str, Any], relative_path: str
) -> None:
    output = absolute(relative_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            payload,
            handle,
            allow_unicode=True,
            sort_keys=False,
        )
    os.replace(temporary, output)


def _copy_query_atomic(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    relative_path: str,
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
            raise FileNotFoundError(
                f"P4受保护输入不存在：{relative_path}"
            )
        snapshot[relative_path] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return snapshot


def _read_yaml(relative_path: str) -> dict[str, Any]:
    with absolute(relative_path).open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle)
    return payload


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    project = config["project"]
    dates = [
        pd.Timestamp(project[key])
        for key in (
            "research_start",
            "research_end",
            "validation_start",
            "validation_end",
            "oos_start",
            "oos_end",
        )
    ]
    if not (
        dates[0] <= dates[1] < dates[2] <= dates[3] < dates[4] <= dates[5]
    ):
        raise ValueError("P4研究、验证和OOS日期闸门无效")
    if project["validation_end"] != "2021-12-31":
        raise ValueError("P4验证截止日必须冻结为2021-12-31")
    if project["oos_start"] != "2022-01-01":
        raise ValueError("P4最终OOS起始日必须冻结为2022-01-01")

    p2_config = _read_yaml(config["inputs"]["p2_config"])
    p3_config = _read_yaml(config["inputs"]["p3_config"])
    for section in ("factors", "universe", "statistics"):
        if config[section] != p2_config[section]:
            raise ValueError(f"P4未复用P2冻结参数：{section}")
    for section in (
        "portfolio",
        "composite",
        "cost_scenarios",
        "valuation",
        "corporate_actions",
        "metrics",
    ):
        if config[section] != p3_config[section]:
            raise ValueError(f"P4未复用P3冻结参数：{section}")
    if (
        config["freeze"]["status"] != "FROZEN_AFTER_VALIDATION"
        or bool(
            config["freeze"]["allow_parameter_changes_after_freeze"]
        )
        or not bool(
            config["freeze"][
                "oos_requires_new_explicit_authorization"
            ]
        )
    ):
        raise ValueError("P4冻结及OOS人工闸门配置无效")

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
    if not str(p2_manifest.get("status", "")).startswith(
        "P2_ACCEPTED"
    ):
        raise RuntimeError("P2未验收，不能开始P4")
    if not str(p3_manifest.get("status", "")).startswith(
        "P3_ACCEPTED"
    ):
        raise RuntimeError("P3未验收，不能开始P4")
    if p3_manifest.get("audit", {}).get("fail_count") != 0:
        raise RuntimeError("P3仍有审计FAIL，不能开始P4")
    if p3_manifest.get("scope_guards", {}).get(
        "oos_period_read_or_run"
    ):
        raise RuntimeError("P3已触碰OOS，不能开始P4")
    return {
        "p2_manifest": p2_manifest,
        "p3_manifest": p3_manifest,
        "p2_config": p2_config,
        "p3_config": p3_config,
    }


def _factor_config(config: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    result["project"] = {
        "warmup_start": config["project"]["warmup_start"],
        "research_start": config["project"]["research_start"],
        "research_end": config["project"]["validation_end"],
    }
    return result


def _simulation_config(config: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    result["project"] = {
        "research_start": config["project"]["research_start"],
        "research_end": config["project"]["validation_end"],
        "validation_start": config["project"]["oos_start"],
        "oos_start": config["project"]["oos_start"],
    }
    result["inputs"]["p2_single_factor_panel"] = config["outputs"][
        "factor_panel"
    ]
    return result


def _period_performance(
    daily: pd.DataFrame,
    orders: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    periods = (
        (
            "RESEARCH_2016_2019",
            pd.Timestamp(config["project"]["research_start"]),
            pd.Timestamp(config["project"]["research_end"]),
        ),
        (
            "VALIDATION_2020_2021",
            pd.Timestamp(config["project"]["validation_start"]),
            pd.Timestamp(config["project"]["validation_end"]),
        ),
    )
    trading_days = int(config["metrics"]["trading_days_per_year"])
    rows: list[dict[str, Any]] = []
    for scenario, scenario_daily in daily.groupby(
        "cost_scenario", sort=True
    ):
        scenario_daily = scenario_daily.sort_values("trade_date")
        for period_name, start, end in periods:
            group = scenario_daily.loc[
                scenario_daily["trade_date"].between(start, end)
            ].copy()
            if group.empty:
                raise RuntimeError(f"P4期间无组合记录：{period_name}")
            strategy_path = (
                1.0 + group["strategy_daily_return"]
            ).cumprod()
            benchmark_path = (
                1.0 + group["benchmark_daily_return"]
            ).cumprod()
            strategy_return = float(strategy_path.iloc[-1] - 1.0)
            benchmark_return = float(benchmark_path.iloc[-1] - 1.0)
            years = len(group) / trading_days
            period_orders = orders.loc[
                (orders["cost_scenario"] == scenario)
                & orders["trade_date"].between(start, end)
            ]
            rows.append(
                {
                    "cost_scenario": scenario,
                    "is_baseline": bool(group.iloc[0]["is_baseline"]),
                    "period": period_name,
                    "start_date": group["trade_date"].min(),
                    "end_date": group["trade_date"].max(),
                    "trading_days": len(group),
                    "strategy_total_return": strategy_return,
                    "strategy_annualized_return": (
                        (1.0 + strategy_return) ** (1.0 / years) - 1.0
                    ),
                    "strategy_max_drawdown_within_period": float(
                        (strategy_path / strategy_path.cummax() - 1.0).min()
                    ),
                    "benchmark_total_return": benchmark_return,
                    "benchmark_annualized_return": (
                        (1.0 + benchmark_return) ** (1.0 / years) - 1.0
                    ),
                    "benchmark_max_drawdown_within_period": float(
                        (
                            benchmark_path
                            / benchmark_path.cummax()
                            - 1.0
                        ).min()
                    ),
                    "annualized_return_difference": (
                        (1.0 + strategy_return) ** (1.0 / years)
                        - (1.0 + benchmark_return) ** (1.0 / years)
                    ),
                    "average_cash_weight": float(
                        group["cash_weight"].mean()
                    ),
                    "two_way_turnover": float(
                        group["turnover_ratio"].sum()
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


def _validation_comparison(
    period_performance: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "strategy_total_return",
        "strategy_annualized_return",
        "strategy_max_drawdown_within_period",
        "benchmark_total_return",
        "benchmark_annualized_return",
        "annualized_return_difference",
        "average_cash_weight",
        "two_way_turnover",
        "total_trading_cost",
        "failed_buy_orders",
        "failed_sell_orders",
    ]
    research = period_performance.loc[
        period_performance["period"] == "RESEARCH_2016_2019",
        ["cost_scenario", "is_baseline", *metrics],
    ].rename(columns={metric: f"{metric}_research" for metric in metrics})
    validation = period_performance.loc[
        period_performance["period"] == "VALIDATION_2020_2021",
        ["cost_scenario", "is_baseline", *metrics],
    ].rename(
        columns={metric: f"{metric}_validation" for metric in metrics}
    )
    result = research.merge(
        validation,
        on=["cost_scenario", "is_baseline"],
        validate="one_to_one",
    )
    for metric in (
        "strategy_annualized_return",
        "strategy_max_drawdown_within_period",
        "annualized_return_difference",
        "average_cash_weight",
    ):
        result[f"{metric}_difference"] = (
            result[f"{metric}_validation"]
            - result[f"{metric}_research"]
        )
    return result


def _research_reproduction(
    config: dict[str, Any],
) -> pd.DataFrame:
    paths = {
        key: absolute(value).as_posix().replace("'", "''")
        for key, value in {
            "old_factor": config["inputs"]["p2_panel"],
            "new_factor": config["outputs"]["factor_panel"],
            "old_signal": config["inputs"]["p3_composite_signals"],
            "new_signal": config["outputs"]["composite_signals"],
            "old_target": config["inputs"]["p3_target_holdings"],
            "new_target": config["outputs"]["target_holdings"],
            "old_daily": config["inputs"]["p3_daily_portfolio"],
            "new_daily": config["outputs"]["daily_portfolio"],
        }.items()
    }
    end = config["project"]["research_end"]
    rows: list[dict[str, Any]] = []
    with duckdb.connect() as connection:
        factor = connection.execute(
            f"""
            WITH old AS (
                SELECT * FROM read_parquet('{paths["old_factor"]}')
            ),
            new AS (
                SELECT * FROM read_parquet('{paths["new_factor"]}')
                WHERE signal_date <= DATE '{end}'
            ),
            joined AS (
                SELECT old.*, new.ts_code AS new_ts_code,
                       new.universe_eligible AS new_eligible,
                       new.bm_proxy AS new_bm,
                       new.momentum_12_1 AS new_momentum,
                       new.lowvol_60 AS new_lowvol,
                       new.bm_proxy_z AS new_bm_z,
                       new.momentum_12_1_z AS new_momentum_z,
                       new.lowvol_60_z AS new_lowvol_z
                FROM old
                FULL OUTER JOIN new
                  USING (canonical_ts_code, signal_date)
            )
            SELECT
                (SELECT count(*) FROM old),
                (SELECT count(*) FROM new),
                count(*) FILTER (
                    WHERE ts_code IS NULL OR new_ts_code IS NULL
                ),
                count(*) FILTER (
                    WHERE universe_eligible <> new_eligible
                ),
                max(abs(bm_proxy - new_bm)),
                max(abs(momentum_12_1 - new_momentum)),
                max(abs(lowvol_60 - new_lowvol)),
                max(abs(bm_proxy_z - new_bm_z)),
                max(abs(momentum_12_1_z - new_momentum_z)),
                max(abs(lowvol_60_z - new_lowvol_z))
            FROM joined
            """
        ).fetchone()
        factor_error = max(
            float(value or 0.0) for value in factor[4:]
        )
        rows.append(
            {
                "check": "P2_FACTOR_RESEARCH_PREFIX",
                "old_rows": factor[0],
                "new_rows": factor[1],
                "missing_or_extra_rows": factor[2],
                "categorical_mismatches": factor[3],
                "maximum_numeric_error": factor_error,
                "status": (
                    "PASS"
                    if factor[0] == factor[1]
                    and factor[2] == 0
                    and factor[3] == 0
                    and factor_error <= 1e-12
                    else "FAIL"
                ),
            }
        )
        for label, old_key, new_key, columns in (
            (
                "P3_COMPOSITE_RESEARCH_PREFIX",
                "old_signal",
                "new_signal",
                (
                    "composite_score",
                    "selection_rank",
                    "target_weight",
                ),
            ),
            (
                "P3_TARGET_RESEARCH_PREFIX",
                "old_target",
                "new_target",
                ("composite_score", "selection_rank", "target_weight"),
            ),
        ):
            numeric = ", ".join(
                f"max(abs(old.{column} - new.{column}))"
                for column in columns
            )
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
                        WHERE old.is_selected <> new.is_selected
                    ),
                    {numeric}
                FROM old
                FULL OUTER JOIN new
                  USING (canonical_ts_code, signal_date)
                """
            ).fetchone()
            numeric_error = max(
                float(value or 0.0) for value in result[4:]
            )
            rows.append(
                {
                    "check": label,
                    "old_rows": result[0],
                    "new_rows": result[1],
                    "missing_or_extra_rows": result[2],
                    "categorical_mismatches": result[3],
                    "maximum_numeric_error": numeric_error,
                    "status": (
                        "PASS"
                        if result[0] == result[1]
                        and result[2] == 0
                        and result[3] == 0
                        and numeric_error <= 1e-12
                        else "FAIL"
                    ),
                }
            )
        daily = connection.execute(
            f"""
            WITH old AS (
                SELECT * FROM read_parquet('{paths["old_daily"]}')
            ),
            new AS (
                SELECT * FROM read_parquet('{paths["new_daily"]}')
                WHERE trade_date <= DATE '{end}'
            ),
            joined AS (
                SELECT
                    old.cost_scenario AS old_scenario,
                    new.cost_scenario AS new_scenario,
                    old.total_nav_cny AS old_nav,
                    new.total_nav_cny AS new_nav,
                    old.cash_cny AS old_cash,
                    new.cash_cny AS new_cash,
                    old.strategy_daily_return AS old_return,
                    new.strategy_daily_return AS new_return
                FROM old
                FULL OUTER JOIN new
                  USING (cost_scenario, trade_date)
            )
            SELECT
                (SELECT count(*) FROM old),
                (SELECT count(*) FROM new),
                count(*) FILTER (
                    WHERE old_scenario IS NULL OR new_scenario IS NULL
                ),
                0,
                max(abs(old_nav - new_nav)),
                max(abs(old_cash - new_cash)),
                max(abs(old_return - new_return))
            FROM joined
            """
        ).fetchone()
        daily_error = max(float(value or 0.0) for value in daily[4:])
        rows.append(
            {
                "check": "P3_DAILY_RESEARCH_PREFIX",
                "old_rows": daily[0],
                "new_rows": daily[1],
                "missing_or_extra_rows": daily[2],
                "categorical_mismatches": daily[3],
                "maximum_numeric_error": daily_error,
                "status": (
                    "PASS"
                    if daily[0] == daily[1]
                    and daily[2] == 0
                    and daily_error <= 1e-6
                    else "FAIL"
                ),
            }
        )
    return pd.DataFrame(rows)


def _freeze_payload(
    config: dict[str, Any],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "freeze_version": "p4.1",
        "status": config["freeze"]["status"],
        "sample": {
            "warmup": [
                config["project"]["warmup_start"],
                "2015-12-31",
            ],
            "research": [
                config["project"]["research_start"],
                config["project"]["research_end"],
            ],
            "validation": [
                config["project"]["validation_start"],
                config["project"]["validation_end"],
            ],
            "final_oos": [
                config["project"]["oos_start"],
                config["project"]["oos_end"],
            ],
        },
        "factors": config["factors"],
        "universe": config["universe"],
        "portfolio": config["portfolio"],
        "composite": config["composite"],
        "cost_scenarios": config["cost_scenarios"],
        "valuation": config["valuation"],
        "corporate_actions": config["corporate_actions"],
        "metrics": config["metrics"],
        "data_policies": {
            "benchmark_code": "000985.CSI",
            "benchmark_name": "CSI All Share Index",
            "pb_disclosure": PB_DISCLOSURE,
            "historical_stamp_duty_config": (
                config["inputs"]["trading_costs"]
            ),
            "corporate_action_table": (
                config["inputs"]["corporate_actions"]
            ),
        },
        "source_sha256": source_hashes,
        "post_freeze_rules": {
            "allow_parameter_changes": False,
            "allow_validation_retuning": False,
            "final_oos_requires_new_explicit_authorization": True,
            "final_oos_has_been_run": False,
            "p5_implementation_generated": False,
        },
    }


def _protocol_text(
    config: dict[str, Any],
    freeze_hash: str,
    period_performance: pd.DataFrame,
    ic_summary: pd.DataFrame,
) -> str:
    baseline = period_performance.loc[
        period_performance["is_baseline"].astype(bool)
        & (
            period_performance["period"]
            == "VALIDATION_2020_2021"
        )
    ].iloc[0]
    lines = [
        "# P4 验证与冻结协议",
        "",
        "## 状态",
        "",
        "- P4 使用范围：2016-01-01 至 2021-12-31。",
        "- 验证期：2020-01-01 至 2021-12-31。",
        "- 最终 OOS：2022-01-01 至 2025-12-31，尚未运行。",
        "- 验证期间未调参，P2/P3冻结实现必须在研究期前缀精确复现。",
        "",
        "## 验证期基准情景",
        "",
        f"- 策略累计收益：{baseline['strategy_total_return']:.6%}",
        f"- 策略年化收益：{baseline['strategy_annualized_return']:.6%}",
        (
            "- 策略期内最大回撤："
            f"{baseline['strategy_max_drawdown_within_period']:.6%}"
        ),
        f"- 基准累计收益：{baseline['benchmark_total_return']:.6%}",
        f"- 基准年化收益：{baseline['benchmark_annualized_return']:.6%}",
        (
            "- 年化收益差："
            f"{baseline['annualized_return_difference']:.6%}"
        ),
        "",
        "## 验证期单因子 Rank IC",
        "",
        "| 因子 | 月数 | IC均值 | ICIR |",
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
            "## 冻结",
            "",
            f"- 冻结配置：`{config['outputs']['frozen_config']}`",
            f"- SHA-256：`{freeze_hash}`",
            "- 冻结后不允许根据验证期或最终OOS结果修改参数。",
            f"- {PB_DISCLOSURE}",
            "",
            "## P5 人工闸门",
            "",
            "P4 完成后必须停止。只有用户再次明确授权，才允许首次运行",
            "2022–2025 最终 OOS；不得预览结果后调参，也不得覆盖本冻结配置。",
            "",
        ]
    )
    return "\n".join(lines)


def build_p4() -> dict[str, Any]:
    started_at = datetime.now(UTC)
    config = load_config()
    gate = _validate_config(config)
    protected_paths = list(config["protected_p4_inputs"])
    _log("记录P4受保护输入SHA-256")
    before = _input_snapshot(protected_paths)

    factor_config = _factor_config(config)
    _log("按P2冻结方法构建2016–2021因子面板")
    with duckdb.connect() as connection:
        connection.execute("SET preserve_insertion_order = false")
        connection.execute("SET threads = 4")
        _copy_query_atomic(
            connection,
            _factor_panel_query(factor_config),
            config["outputs"]["factor_panel"],
        )
    panel = pd.read_parquet(absolute(config["outputs"]["factor_panel"]))
    panel["signal_date"] = pd.to_datetime(panel["signal_date"])
    panel["next_signal_date"] = pd.to_datetime(panel["next_signal_date"])
    if panel["signal_date"].max() >= pd.Timestamp(
        config["project"]["oos_start"]
    ):
        raise RuntimeError("P4因子面板包含最终OOS记录")

    validation_panel = panel.loc[
        panel["signal_date"].between(
            pd.Timestamp(config["project"]["validation_start"]),
            pd.Timestamp(config["project"]["validation_end"]),
        )
    ].copy()
    _log("计算2020–2021验证期单因子统计")
    validation_statistics = _build_statistics(
        validation_panel, factor_config
    )
    for key, frame in validation_statistics.items():
        _write_csv_atomic(
            frame, config["outputs"][f"validation_{key}"]
        )

    simulation_config = _simulation_config(config)
    _log("构建2016–2021冻结复合信号与调仓表")
    signals, targets = _build_composite_signals(simulation_config)
    benchmark = pd.read_parquet(
        absolute(config["inputs"]["benchmark_daily"])
    )
    benchmark["trade_date"] = pd.to_datetime(benchmark["trade_date"])
    benchmark = benchmark.loc[
        benchmark["trade_date"].between(
            signals["signal_date"].min(),
            pd.Timestamp(config["project"]["validation_end"]),
        )
    ].sort_values("trade_date")
    if set(benchmark["benchmark_code"]) != {"000985.CSI"}:
        raise RuntimeError("P4基准不是中证全指000985.CSI")
    schedule = _build_schedule(
        signals,
        benchmark,
        pd.Timestamp(config["project"]["validation_end"]),
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

    corporate_actions = _load_corporate_actions(simulation_config)
    stamp_policy = _load_stamp_policy(simulation_config)
    prices, executions, benchmark = _load_market_inputs(
        simulation_config,
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
    for scenario in config["cost_scenarios"]:
        _log(f"运行冻结成本情景 {scenario['scenario']}")
        daily, holdings, orders, rebalances, actions = (
            _simulate_scenario(
                simulation_config,
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

    daily_portfolio = pd.concat(daily_frames, ignore_index=True)
    actual_holdings = pd.concat(holding_frames, ignore_index=True)
    orders = pd.concat(order_frames, ignore_index=True)
    rebalances = pd.concat(rebalance_frames, ignore_index=True)
    action_events = pd.concat(action_frames, ignore_index=True)
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
        ("corporate_action_events", action_events),
    ):
        _write_parquet_atomic(frame, config["outputs"][key])
    _write_csv_atomic(rebalances, config["outputs"]["rebalance_summary"])

    performance = _performance_summary(
        daily_portfolio, orders, simulation_config
    )
    annual = _annual_performance(
        daily_portfolio, orders, simulation_config
    )
    period_performance = _period_performance(
        daily_portfolio, orders, config
    )
    validation_comparison = _validation_comparison(
        period_performance
    )
    failure_summary = _failure_summary(orders)
    stale_summary = _stale_position_summary(
        actual_holdings,
        pd.Timestamp(config["project"]["validation_end"]),
    )
    for key, frame in (
        ("performance_summary", performance),
        ("annual_performance", annual),
        ("period_performance", period_performance),
        ("validation_comparison", validation_comparison),
        ("failure_reason_summary", failure_summary),
        ("stale_position_summary", stale_summary),
    ):
        _write_csv_atomic(frame, config["outputs"][key])

    reproduction = _research_reproduction(config)
    _write_csv_atomic(
        reproduction, config["outputs"]["research_reproduction"]
    )
    if not bool((reproduction["status"] == "PASS").all()):
        raise RuntimeError(
            "P4未能精确复现P2/P3研究期冻结结果："
            f"{reproduction.to_dict(orient='records')}"
        )

    _log("生成冻结配置与OOS人工闸门")
    source_hashes = {
        relative_path: before[relative_path]["sha256"]
        for relative_path in protected_paths
    }
    freeze_payload = _freeze_payload(config, source_hashes)
    _write_yaml_atomic(
        freeze_payload, config["outputs"]["frozen_config"]
    )
    freeze_hash = _sha256(absolute(config["outputs"]["frozen_config"]))
    _write_text_atomic(
        (
            f"path={config['outputs']['frozen_config']}\n"
            f"sha256={freeze_hash}\n"
        ),
        config["outputs"]["config_sha256"],
    )
    _write_text_atomic(
        _protocol_text(
            config,
            freeze_hash,
            period_performance,
            validation_statistics["ic_summary"],
        ),
        config["outputs"]["frozen_protocol"],
    )

    _log("复核P4受保护输入SHA-256")
    after = _input_snapshot(protected_paths)
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
            for path in protected_paths
        ]
    )
    _write_csv_atomic(
        input_hashes, config["outputs"]["p4_input_hashes"]
    )
    if not bool(input_hashes["match"].all()):
        raise RuntimeError("P4构建期间受保护输入发生变化")

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
        *[f"validation_{key}" for key in validation_statistics],
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
    ]
    output_hashes = {
        key: _sha256(absolute(config["outputs"][key]))
        for key in generated_keys
    }
    baseline_validation = period_performance.loc[
        period_performance["is_baseline"].astype(bool)
        & (
            period_performance["period"]
            == "VALIDATION_2020_2021"
        )
    ].iloc[0]
    completed_at = datetime.now(UTC)
    manifest: dict[str, Any] = {
        "stage": "P4_VALIDATION_AND_FREEZE",
        "builder_version": BUILDER_VERSION,
        "status": "BUILT_PENDING_AUDIT",
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
        "sample": {
            **config["project"],
            "factor_panel_min": str(panel["signal_date"].min().date()),
            "factor_panel_max": str(panel["signal_date"].max().date()),
            "portfolio_min": str(
                daily_portfolio["trade_date"].min().date()
            ),
            "portfolio_max": str(
                daily_portfolio["trade_date"].max().date()
            ),
        },
        "gate": {
            "p2_status": gate["p2_manifest"]["status"],
            "p3_status": gate["p3_manifest"]["status"],
            "p3_fail_count": gate["p3_manifest"]["audit"][
                "fail_count"
            ],
        },
        "counts": {
            "factor_rows": len(panel),
            "validation_factor_rows": len(validation_panel),
            "signal_months": int(signals["signal_date"].nunique()),
            "scheduled_rebalances": int(
                schedule["scheduled_trade_date"].notna().sum()
            ),
            "out_of_scope_signals": int(
                schedule["scheduled_trade_date"].isna().sum()
            ),
            "daily_portfolio_rows": len(daily_portfolio),
            "orders": len(orders),
            "failed_orders": len(failed_orders),
            "corporate_action_events": len(action_events),
        },
        "research_reproduction_all_pass": bool(
            (reproduction["status"] == "PASS").all()
        ),
        "validation_baseline_result": {
            key: (
                int(baseline_validation[key])
                if key in {
                    "trading_days",
                    "failed_buy_orders",
                    "failed_sell_orders",
                }
                else float(baseline_validation[key])
            )
            for key in (
                "trading_days",
                "strategy_total_return",
                "strategy_annualized_return",
                "strategy_max_drawdown_within_period",
                "benchmark_total_return",
                "benchmark_annualized_return",
                "benchmark_max_drawdown_within_period",
                "annualized_return_difference",
                "average_cash_weight",
                "two_way_turnover",
                "total_trading_cost",
                "failed_buy_orders",
                "failed_sell_orders",
            )
        },
        "freeze": {
            "frozen_config_path": config["outputs"]["frozen_config"],
            "frozen_config_sha256": freeze_hash,
            "allow_parameter_changes": False,
        },
        "protected_inputs_all_match": bool(
            input_hashes["match"].all()
        ),
        "output_sha256": output_hashes,
        "scope_guards": {
            "validation_period_evaluated": True,
            "validation_parameters_retuned": False,
            "oos_rows_written": False,
            "oos_results_computed": False,
            "oos_results_previewed": False,
            "p5_code_generated": False,
            "p5_run": False,
            "p6_run": False,
        },
        "disclosures": [
            PB_DISCLOSURE,
            (
                "P4验证结果不是最终OOS；冻结后不得根据2020–2021"
                "验证表现或未来OOS表现修改参数。"
            ),
        ],
    }
    _write_json_atomic(manifest, config["outputs"]["p4_run_manifest"])
    _log(
        "构建完成："
        f"months={manifest['counts']['signal_months']}，"
        f"rebalances={manifest['counts']['scheduled_rebalances']}，"
        f"validation_return="
        f"{manifest['validation_baseline_result']['strategy_total_return']:.2%}；"
        "等待P4审计"
    )
    return manifest
