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

from a_share_p3.audit import (
    CORPORATE_ACTION_DISCLOSURE,
    PB_DISCLOSURE,
)
from a_share_p3.config import absolute, load_config


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def connection(config: dict) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for name in (
        "composite_signals",
        "target_holdings",
        "daily_portfolio",
        "actual_holdings",
        "orders",
        "failed_orders",
        "corporate_action_events",
    ):
        path = config["outputs"][name]
        if not absolute(path).is_file():
            pytest.fail(f"缺少P3输出：{path}")
        con.execute(
            f"""
            CREATE VIEW {name} AS
            SELECT * FROM read_parquet('{path}')
            """
        )
    yield con
    con.close()


def test_p3_paths_are_relative(config: dict) -> None:
    values = (
        list(config["inputs"].values())
        + list(config["outputs"].values())
        + list(config["protected_p3_inputs"])
    )
    assert all(not Path(value).is_absolute() for value in values)


def test_p2_gate_was_accepted(config: dict) -> None:
    manifest = json.loads(
        absolute(config["inputs"]["p2_run_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"].startswith("P2_ACCEPTED")


def test_equal_weight_composite_formula(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    result = connection.execute(
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
                WHERE signal_date >= DATE '2020-01-01'
            )
        FROM composite_signals
        """
    ).fetchone()
    assert result[0] == 107292
    assert result[1] == 48
    assert result[2] <= 1e-12
    assert result[3] == 0


def test_target_portfolios_are_top_100_equal_weight(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    errors = connection.execute(
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
            count(*),
            min(names), max(names),
            min(gross), max(gross),
            min(min_rank), max(max_rank),
            max(max_weight)
        FROM monthly
        """
    ).fetchone()
    assert errors[0] == 48
    assert errors[1:3] == (100, 100)
    assert abs(errors[3] - 1.0) <= 1e-12
    assert abs(errors[4] - 1.0) <= 1e-12
    assert errors[5:7] == (1, 100)
    assert errors[7] <= 0.02


def test_rebalance_schedule_uses_next_market_day(config: dict) -> None:
    schedule = pd.read_csv(
        absolute(config["outputs"]["rebalance_schedule"]),
        parse_dates=["signal_date", "scheduled_trade_date"],
    )
    benchmark = pd.read_parquet(
        absolute(config["inputs"]["benchmark_daily"])
    )
    benchmark["trade_date"] = pd.to_datetime(benchmark["trade_date"])
    assert len(schedule) == 48
    assert schedule["scheduled_trade_date"].notna().sum() == 47
    for row in schedule.dropna(
        subset=["scheduled_trade_date"]
    ).itertuples(index=False):
        expected = benchmark.loc[
            (benchmark["trade_date"] > row.signal_date)
            & (benchmark["trade_date"] <= pd.Timestamp("2019-12-31")),
            "trade_date",
        ].min()
        assert row.scheduled_trade_date == expected
    assert (
        schedule.iloc[-1]["schedule_status"]
        == "OUT_OF_SCOPE_NOT_EXECUTED"
    )


def test_daily_portfolio_scope_and_accounting(
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
    assert result[0] == 2868
    assert result[1] == 3
    assert pd.Timestamp(result[2]).date().isoformat() == "2016-01-29"
    assert pd.Timestamp(result[3]).date().isoformat() == "2019-12-31"
    assert result[4] == 0
    assert result[5] >= -1e-8
    assert result[6] > 0
    assert result[7] <= 1e-6
    assert result[8] <= 1e-12


def test_orders_use_open_and_respect_execution_flags(
    connection: duckdb.DuckDBPyConnection,
    config: dict,
) -> None:
    basic_violations = connection.execute(
        """
        SELECT count(*)
        FROM orders
        WHERE executed_notional > 0
          AND (
            execution_price_source <> 'ADJUSTED_OPEN'
            OR adjusted_open IS NULL
            OR adjusted_open <= 0
            OR trade_date <= signal_date
            OR trade_date >= DATE '2020-01-01'
          )
        """
    ).fetchone()[0]
    execution_violations = connection.execute(
        f"""
        SELECT count(*)
        FROM orders
        LEFT JOIN read_parquet(
            '{config["inputs"]["execution_status"]}'
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
    assert basic_violations == 0
    assert execution_violations == 0


def test_cost_and_stamp_duty_formulas(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    result = connection.execute(
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
    assert result[0] <= 1e-12
    assert result[1] == 0
    assert result[2] <= 1e-12
    assert result[3] <= 1e-12


def test_failed_orders_are_not_filled_and_failed_sells_remain(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    counts = connection.execute(
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
    assert counts[0] > 0
    assert counts[1:] == (0, 0)
    assert preservation == 0


def test_stale_prices_are_valuation_only(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    executed_stale = connection.execute(
        """
        SELECT count(*)
        FROM orders
        WHERE executed_notional > 0
          AND execution_price_source
                = 'LAST_AVAILABLE_ADJUSTED_CLOSE'
        """
    ).fetchone()[0]
    stale = connection.execute(
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
    assert executed_stale == 0
    assert stale[0] > 0
    assert stale[1] == 0


def test_cost_stress_is_monotonic(config: dict) -> None:
    performance = pd.read_csv(
        absolute(config["outputs"]["performance_summary"])
    ).set_index("cost_scenario")
    assert (
        performance.loc["STRESS_5BPS", "strategy_total_return"]
        > performance.loc["BASE_10BPS", "strategy_total_return"]
        > performance.loc["STRESS_20BPS", "strategy_total_return"]
    )
    assert (
        performance.loc["STRESS_5BPS", "total_trading_cost"]
        < performance.loc["BASE_10BPS", "total_trading_cost"]
        < performance.loc["STRESS_20BPS", "total_trading_cost"]
    )


def test_600270_stock_swap_is_auditable(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    event = connection.execute(
        """
        SELECT
            count(*),
            count(DISTINCT cost_scenario),
            max(abs(
                successor_share_quantity_after
                - old_share_quantity_before * exchange_ratio
            )),
            max(abs(position_value_difference_cny)),
            max(abs(portfolio_value_difference_cny)),
            max(abs(total_action_cost_cny)),
            min(effective_date),
            max(effective_date)
        FROM corporate_action_events
        WHERE activity_type = 'CORPORATE_ACTION'
          AND old_ts_code = '600270.SH'
          AND successor_ts_code = '601598.SH'
          AND exchange_ratio = 3.8225
        """
    ).fetchone()
    old_terminal = connection.execute(
        """
        SELECT count(*)
        FROM actual_holdings
        WHERE canonical_ts_code = '600270.SH'
          AND trade_date = DATE '2019-12-31'
        """
    ).fetchone()[0]
    successor_on_event = connection.execute(
        """
        SELECT count(*)
        FROM actual_holdings
        WHERE canonical_ts_code = '601598.SH'
          AND trade_date = DATE '2018-12-28'
          AND position_origin_activity_type = 'CORPORATE_ACTION'
        """
    ).fetchone()[0]
    event_in_orders = connection.execute(
        """
        SELECT count(*)
        FROM orders
        WHERE activity_type <> 'TRADE'
        """
    ).fetchone()[0]
    assert event[0:2] == (3, 3)
    assert event[2] <= 1e-8
    assert event[3] <= 1e-6
    assert event[4] <= 1e-6
    assert event[5] <= 1e-12
    assert pd.Timestamp(event[6]).date().isoformat() == "2018-12-28"
    assert pd.Timestamp(event[7]).date().isoformat() == "2018-12-28"
    assert old_terminal == 0
    assert successor_on_event == 3
    assert event_in_orders == 0


def test_successor_uses_market_price_then_normal_rebalance(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    current_price_rows = connection.execute(
        """
        SELECT count(*)
        FROM actual_holdings
        WHERE canonical_ts_code = '601598.SH'
          AND trade_date = DATE '2019-01-18'
          AND valuation_price_source = 'CURRENT_ADJUSTED_CLOSE'
        """
    ).fetchone()[0]
    sell = connection.execute(
        """
        SELECT
            count(*),
            min(trade_date),
            max(trade_date),
            min(signal_date),
            max(signal_date)
        FROM orders
        WHERE canonical_ts_code = '601598.SH'
          AND side = 'SELL'
          AND executed_notional > 0
          AND activity_type = 'TRADE'
        """
    ).fetchone()
    assert current_price_rows == 3
    assert sell[0] == 3
    assert pd.Timestamp(sell[1]).date().isoformat() == "2019-02-01"
    assert pd.Timestamp(sell[2]).date().isoformat() == "2019-02-01"
    assert pd.Timestamp(sell[3]).date().isoformat() == "2019-01-31"
    assert pd.Timestamp(sell[4]).date().isoformat() == "2019-01-31"


def test_original_data_hashes_are_unchanged(config: dict) -> None:
    hashes = pd.read_csv(
        absolute(config["outputs"]["original_data_hash_check"])
    )
    assert len(hashes) > 17000
    assert hashes["matches_p1_before"].astype(bool).all()
    assert hashes["matches_p1_after"].astype(bool).all()
    assert hashes["unchanged_during_build"].astype(bool).all()
    assert (hashes["sha256_before"] == hashes["sha256_after"]).all()


def test_manifest_scope_and_disclosures(config: dict) -> None:
    path = absolute(config["outputs"]["p3_run_manifest"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    text = path.read_text(encoding="utf-8")
    assert not any(manifest["scope_guards"].values())
    assert manifest["counts"]["scheduled_rebalances"] == 47
    assert manifest["counts"]["out_of_scope_signals"] == 1
    assert PB_DISCLOSURE in text
    assert "上一可得复权收盘价仅用于估值" in text
    assert "fractional adjusted-price units" in text
    assert CORPORATE_ACTION_DISCLOSURE in text
    assert manifest["counts"]["corporate_action_events"] == 3
    assert (
        manifest["original_data_integrity"][
            "changed_during_build_count"
        ]
        == 0
    )
