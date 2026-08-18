from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from a_share_p4.build import _sha256
from a_share_p5.config import absolute, load_config


CONFIG = load_config()
if not absolute(CONFIG["outputs"]["p5_run_manifest"]).is_file():
    pytest.skip(
        "P5最终OOS尚未获准或尚未运行",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def config() -> dict:
    return CONFIG


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
        "oos_daily_portfolio",
        "oos_actual_holdings",
    ):
        path = absolute(config["outputs"][key]).as_posix()
        con.execute(
            f"""
            CREATE VIEW {key} AS
            SELECT * FROM read_parquet('{path}')
            """
        )
    yield con
    con.close()


def test_p5_is_first_authorized_run(config: dict) -> None:
    intent = json.loads(
        absolute(config["outputs"]["run_intent"]).read_text(
            encoding="utf-8"
        )
    )
    assert intent["attempt_number"] == 1
    assert intent["status"] == "COMPLETED"
    assert intent["authorization_obtained_before_run"] is True
    assert intent["parameters_retuned"] is False
    assert intent["frozen_config_modified"] is False


def test_p5_reproduces_all_p4_prefixes(config: dict) -> None:
    reproduction = pd.read_csv(
        absolute(config["outputs"]["p4_reproduction"])
    )
    assert len(reproduction) == 7
    assert (reproduction["status"] == "PASS").all()
    assert (reproduction["missing_or_extra_rows"] == 0).all()
    assert (reproduction["categorical_mismatches"] == 0).all()


def test_p5_factor_signal_and_target_scope(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    factor = connection.execute(
        """
        SELECT
            count(DISTINCT signal_date),
            min(signal_date),
            max(signal_date),
            count(*) FILTER (
                WHERE signal_date > DATE '2025-12-31'
            )
        FROM factor_panel
        """
    ).fetchone()
    signal = connection.execute(
        """
        SELECT
            count(DISTINCT signal_date),
            max(abs(
                composite_score
                - (bm_proxy_z + momentum_12_1_z + lowvol_60_z)
                  / 3.0
            ))
        FROM composite_signals
        """
    ).fetchone()
    target = connection.execute(
        """
        WITH monthly AS (
            SELECT
                signal_date,
                count(*) AS names,
                sum(target_weight) AS gross,
                min(selection_rank) AS min_rank,
                max(selection_rank) AS max_rank
            FROM target_holdings
            GROUP BY signal_date
        )
        SELECT
            count(*), min(names), max(names),
            min(gross), max(gross),
            min(min_rank), max(max_rank)
        FROM monthly
        """
    ).fetchone()
    assert factor[0] == signal[0] == 120
    assert pd.Timestamp(factor[1]).date().isoformat() == "2016-01-29"
    assert pd.Timestamp(factor[2]).date().isoformat() == "2025-12-31"
    assert factor[3] == 0
    assert signal[1] <= 1e-12
    assert target[0] == 120
    assert target[1:3] == (100, 100)
    assert abs(target[3] - 1.0) <= 1e-12
    assert abs(target[4] - 1.0) <= 1e-12
    assert target[5:7] == (1, 100)


def test_p5_daily_scope_and_accounting(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    result = connection.execute(
        """
        SELECT
            count(DISTINCT cost_scenario),
            min(trade_date),
            max(trade_date),
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
    assert result[0] == 3
    assert pd.Timestamp(result[1]).date().isoformat() == "2016-01-29"
    assert pd.Timestamp(result[2]).date().isoformat() == "2025-12-31"
    assert result[3] == 0
    assert result[4] >= -1e-8
    assert result[5] > 0
    assert result[6] <= 1e-6
    assert result[7] <= 1e-12


def test_p5_oos_slice_is_strict(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    result = connection.execute(
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
    assert result[0] > 0
    assert result[1] == 3
    assert pd.Timestamp(result[2]).date().isoformat() == "2022-01-04"
    assert pd.Timestamp(result[3]).date().isoformat() == "2025-12-31"
    assert result[4] == 0


def test_p5_orders_and_stamp_duty(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    result = connection.execute(
        """
        SELECT
            count(*) FILTER (
                WHERE executed_notional > 0
                  AND (
                    trade_date <= signal_date
                    OR execution_price_source <> 'ADJUSTED_OPEN'
                    OR adjusted_open IS NULL
                    OR adjusted_open <= 0
                  )
            ),
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
    assert result[0] == 0
    assert result[1] <= 1e-12
    assert result[2] <= 1e-12
    assert result[3:] == (0, 0, 0)


def test_p5_failed_orders_do_not_fill(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    result = connection.execute(
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
    assert result[0] > 0
    assert result[1:] == (0, 0)


def test_p5_corporate_action_prefix_is_unchanged(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    result = connection.execute(
        """
        SELECT
            count(*),
            count(DISTINCT cost_scenario),
            count(*) FILTER (
                WHERE old_ts_code <> '600270.SH'
                   OR successor_ts_code <> '601598.SH'
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
    assert result[0:3] == (3, 3, 0)
    assert result[3] <= 1e-8
    assert result[4] <= 1e-6
    assert result[5] <= 1e-12


def test_p5_oos_ic_stops_before_outside_label(config: dict) -> None:
    summary = pd.read_csv(
        absolute(config["outputs"]["oos_ic_summary"])
    )
    monthly = pd.read_csv(
        absolute(config["outputs"]["oos_monthly_rank_ic"]),
        parse_dates=["signal_date"],
    )
    assert set(summary["months"]) == {47}
    december = monthly.loc[
        monthly["signal_date"] == pd.Timestamp("2025-12-31")
    ]
    assert len(december) == 3
    assert december["rank_ic"].isna().all()
    assert (december["observations"] == 0).all()


def test_p5_performance_is_final_oos_only(config: dict) -> None:
    performance = pd.read_csv(
        absolute(config["outputs"]["oos_performance"])
    )
    assert len(performance) == 3
    assert set(performance["period"]) == {"FINAL_OOS_2022_2025"}
    assert set(performance["start_date"]) == {"2022-01-04"}
    assert set(performance["end_date"]) == {"2025-12-31"}
    cost = performance.set_index("cost_scenario")
    assert (
        cost.loc["STRESS_5BPS", "strategy_total_return"]
        > cost.loc["BASE_10BPS", "strategy_total_return"]
        > cost.loc["STRESS_20BPS", "strategy_total_return"]
    )


def test_p5_frozen_config_remains_unchanged(config: dict) -> None:
    expected = config["project"]["expected_frozen_sha256"]
    assert (
        _sha256(absolute(config["inputs"]["frozen_config"]))
        == expected
    )


def test_p5_manifest_stops_before_p6(config: dict) -> None:
    manifest = json.loads(
        absolute(config["outputs"]["p5_run_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    scope = manifest["scope_guards"]
    assert scope["final_oos_authorized_before_run"] is True
    assert scope["final_oos_results_computed"] is True
    assert scope["parameters_retuned_after_validation"] is False
    assert scope["frozen_config_modified"] is False
    assert scope["p6_code_generated"] is False
    assert scope["p6_run"] is False
