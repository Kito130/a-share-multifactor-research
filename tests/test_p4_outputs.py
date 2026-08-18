from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from a_share_p4.config import absolute, load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_yaml(path: str) -> dict:
    with absolute(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def connection(config: dict) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for key in (
        "factor_panel",
        "composite_signals",
        "target_holdings",
        "daily_portfolio",
        "actual_holdings",
        "orders",
        "failed_orders",
        "corporate_action_events",
    ):
        path = absolute(config["outputs"][key])
        if not path.is_file():
            pytest.fail(f"缺少P4输出：{path}")
        sql_path = path.as_posix().replace("'", "''")
        con.execute(
            f"""
            CREATE VIEW {key} AS
            SELECT * FROM read_parquet('{sql_path}')
            """
        )
    yield con
    con.close()


def test_p4_paths_are_relative(config: dict) -> None:
    values = (
        list(config["inputs"].values())
        + list(config["outputs"].values())
        + list(config["protected_p4_inputs"])
    )
    assert all(not Path(str(value)).is_absolute() for value in values)


def test_p4_reuses_p2_and_p3_frozen_parameters(config: dict) -> None:
    p2 = _read_yaml(config["inputs"]["p2_config"])
    p3 = _read_yaml(config["inputs"]["p3_config"])
    for section in ("factors", "universe", "statistics"):
        assert config[section] == p2[section]
    for section in (
        "portfolio",
        "composite",
        "cost_scenarios",
        "valuation",
        "corporate_actions",
        "metrics",
    ):
        assert config[section] == p3[section]


def test_p4_factor_and_signal_scope(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    factor = connection.execute(
        """
        SELECT
            count(DISTINCT signal_date),
            min(signal_date),
            max(signal_date),
            count(*) FILTER (
                WHERE signal_date >= DATE '2022-01-01'
            )
        FROM factor_panel
        """
    ).fetchone()
    signal = connection.execute(
        """
        SELECT
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
    assert factor[0] == signal[0] == 72
    assert (
        pd.Timestamp(factor[1]).date().isoformat()
        == pd.Timestamp(signal[1]).date().isoformat()
        == "2016-01-29"
    )
    assert (
        pd.Timestamp(factor[2]).date().isoformat()
        == pd.Timestamp(signal[2]).date().isoformat()
        == "2021-12-31"
    )
    assert factor[3] == signal[3] == 0
    assert signal[4] <= 1e-12


def test_p4_target_portfolios_and_schedule(config: dict) -> None:
    with duckdb.connect() as con:
        path = absolute(config["outputs"]["target_holdings"])
        target = con.execute(
            f"""
            WITH monthly AS (
                SELECT
                    signal_date,
                    count(*) AS names,
                    sum(target_weight) AS gross,
                    min(selection_rank) AS min_rank,
                    max(selection_rank) AS max_rank,
                    max(target_weight) AS max_weight
                FROM read_parquet('{path.as_posix()}')
                GROUP BY signal_date
            )
            SELECT
                count(*),
                min(names), max(names),
                min(gross), max(gross),
                min(min_rank), max(max_rank),
                max(max_weight)
            FROM monthly
            """
        ).fetchone()
    assert target[0] == 72
    assert target[1:3] == (100, 100)
    assert abs(target[3] - 1.0) <= 1e-12
    assert abs(target[4] - 1.0) <= 1e-12
    assert target[5:7] == (1, 100)
    assert target[7] <= 0.02

    schedule = pd.read_csv(
        absolute(config["outputs"]["rebalance_schedule"])
    )
    assert len(schedule) == 72
    assert schedule["scheduled_trade_date"].notna().sum() == 71
    assert (
        schedule.iloc[-1]["schedule_status"]
        == "OUT_OF_SCOPE_NOT_EXECUTED"
    )


def test_p4_research_prefix_is_exact(config: dict) -> None:
    reproduction = pd.read_csv(
        absolute(config["outputs"]["research_reproduction"])
    )
    assert len(reproduction) == 4
    assert (reproduction["status"] == "PASS").all()
    assert (reproduction["missing_or_extra_rows"] == 0).all()
    assert (reproduction["categorical_mismatches"] == 0).all()


def test_p4_daily_scope_and_accounting(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    result = connection.execute(
        """
        SELECT
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
    assert result[0] == 3
    assert pd.Timestamp(result[1]).date().isoformat() == "2016-01-29"
    assert pd.Timestamp(result[2]).date().isoformat() == "2021-12-31"
    assert result[3] == 0
    assert result[4] >= -1e-8
    assert result[5] > 0
    assert result[6] <= 1e-6
    assert result[7] <= 1e-12


def test_p4_orders_respect_timing_prices_and_limits(
    connection: duckdb.DuckDBPyConnection,
    config: dict,
) -> None:
    basic = connection.execute(
        """
        SELECT count(*)
        FROM orders
        WHERE executed_notional > 0
          AND (
            trade_date <= signal_date
            OR trade_date >= DATE '2022-01-01'
            OR execution_price_source <> 'ADJUSTED_OPEN'
            OR adjusted_open IS NULL
            OR adjusted_open <= 0
          )
        """
    ).fetchone()[0]
    execution_path = absolute(
        config["inputs"]["execution_status"]
    ).as_posix()
    limits = connection.execute(
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
    assert basic == 0
    assert limits == 0


def test_p4_cost_formulas(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    result = connection.execute(
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
    assert result[0] <= 1e-12
    assert result[1] <= 1e-12
    assert result[2:4] == (0, 0)
    assert result[4] <= 1e-12


def test_p4_corporate_action_remains_valid(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    event = connection.execute(
        """
        SELECT
            count(*),
            count(DISTINCT cost_scenario),
            count(*) FILTER (
                WHERE old_ts_code <> '600270.SH'
                   OR successor_ts_code <> '601598.SH'
                   OR activity_type <> 'CORPORATE_ACTION'
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
    assert event[0:3] == (3, 3, 0)
    assert event[3] <= 1e-8
    assert event[4] <= 1e-6
    assert event[5] <= 1e-12
    assert terminal_old == 0


def test_p4_validation_ic_does_not_read_next_oos_month(
    config: dict,
) -> None:
    summary = pd.read_csv(
        absolute(config["outputs"]["validation_ic_summary"])
    )
    monthly = pd.read_csv(
        absolute(config["outputs"]["validation_monthly_rank_ic"]),
        parse_dates=["signal_date"],
    )
    assert set(summary["factor"]) == {
        "bm_proxy",
        "momentum_12_1",
        "lowvol_60",
    }
    assert set(summary["months"]) == {23}
    december = monthly.loc[
        monthly["signal_date"] == pd.Timestamp("2021-12-31")
    ]
    assert len(december) == 3
    assert december["rank_ic"].isna().all()
    assert (december["observations"] == 0).all()
    assert (
        monthly.loc[monthly["rank_ic"].notna(), "signal_date"].max()
        <= pd.Timestamp("2021-11-30")
    )


def test_p4_freeze_hash_sources_and_rules(config: dict) -> None:
    frozen = _read_yaml(config["outputs"]["frozen_config"])
    text = absolute(config["outputs"]["config_sha256"]).read_text(
        encoding="utf-8"
    )
    recorded = re.search(r"sha256=([0-9a-f]{64})", text)
    assert recorded is not None
    current = _sha256(absolute(config["outputs"]["frozen_config"]))
    assert recorded.group(1) == current
    assert set(frozen["source_sha256"]) == set(
        config["protected_p4_inputs"]
    )
    for path in config["protected_p4_inputs"]:
        assert frozen["source_sha256"][path] == _sha256(absolute(path))
    assert frozen["status"] == "FROZEN_AFTER_VALIDATION"
    rules = frozen["post_freeze_rules"]
    assert rules["allow_parameter_changes"] is False
    assert rules["allow_validation_retuning"] is False
    assert rules["final_oos_requires_new_explicit_authorization"] is True
    assert rules["final_oos_has_been_run"] is False
    assert rules["p5_implementation_generated"] is False


def test_p4_manifest_declares_oos_untouched(config: dict) -> None:
    manifest = json.loads(
        absolute(config["outputs"]["p4_run_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    scope = manifest["scope_guards"]
    assert scope["validation_period_evaluated"] is True
    assert scope["validation_parameters_retuned"] is False
    for key in (
        "oos_rows_written",
        "oos_results_computed",
        "oos_results_previewed",
        "p5_code_generated",
        "p5_run",
        "p6_run",
    ):
        assert scope[key] is False
