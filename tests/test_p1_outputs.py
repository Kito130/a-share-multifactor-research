from __future__ import annotations

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

from a_share_p1.config import absolute, load_config


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def connection(config: dict) -> duckdb.DuckDBPyConnection:
    required = [
        config["outputs"][name]
        for name in (
            "daily_panel",
            "execution_status",
            "industry_history",
            "name_history",
            "month_end_base_panel",
            "benchmark_daily",
        )
    ]
    missing = [path for path in required if not absolute(path).is_file()]
    if missing:
        pytest.fail(f"缺少P1输出：{missing}")
    con = duckdb.connect()
    for name in (
        "daily_panel",
        "execution_status",
        "industry_history",
        "name_history",
        "month_end_base_panel",
        "benchmark_daily",
    ):
        con.execute(
            f"""
            CREATE VIEW {name} AS
            SELECT * FROM read_parquet('{config["outputs"][name]}')
            """
        )
    yield con
    con.close()


def test_config_paths_are_relative(config: dict) -> None:
    values = (
        list(config["inputs"].values())
        + list(config["outputs"].values())
        + list(config["policies"].values())
        + list(config["protected_inputs"])
    )
    assert all(not Path(value).is_absolute() for value in values)


def test_dictionary_statuses_and_confirmations() -> None:
    dictionary = pd.read_csv(absolute("data/data_dictionary.csv"))
    assert set(dictionary["status"]) <= {
        "CONFIRMED",
        "INFERRED_FROM_DATA",
        "NEEDS_MANUAL_CONFIRMATION",
    }
    confirmed = dictionary.loc[
        dictionary["status"] == "CONFIRMED", "source_field"
    ]
    for field in (
        "vol",
        "amount",
        "adj_factor",
        "pb",
        "total_share",
        "float_share",
        "free_share",
        "total_mv",
        "circ_mv",
    ):
        assert field in set(confirmed)


def test_daily_panel_is_unique_and_positive(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    duplicates, invalid_adjusted, invalid_age = connection.execute(
        """
        SELECT
            (
                SELECT COUNT(*) FROM (
                    SELECT ts_code, trade_date, COUNT(*) AS n
                    FROM daily_panel GROUP BY ALL HAVING n > 1
                )
            ),
            (
                SELECT COUNT(*) FROM daily_panel
                WHERE adjusted_close IS NULL OR adjusted_close <= 0
            ),
            (
                SELECT COUNT(*) FROM daily_panel
                WHERE is_within_listing_window
                  AND (
                    listing_age_trading_days IS NULL
                    OR listing_age_trading_days < 1
                  )
            )
        """
    ).fetchone()
    assert duplicates == 0
    assert invalid_adjusted == 0
    assert invalid_age == 0


def test_security_code_history_and_original_listing_age(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    mapping = pd.read_csv(
        absolute("data/manual/security_code_history.csv"), dtype=str
    )
    expected = {
        ("000022.SZ", "001872.SZ", "20181226", "001872.SZ", "19930505"),
        ("000043.SZ", "001914.SZ", "20191216", "001914.SZ", "19940928"),
        ("300114.SZ", "302132.SZ", "20250217", "302132.SZ", "20100827"),
    }
    observed = set(
        mapping[
            [
                "old_ts_code",
                "new_ts_code",
                "effective_date",
                "canonical_ts_code",
                "original_list_date",
            ]
        ].itertuples(index=False, name=None)
    )
    assert observed == expected

    mismatches = connection.execute(
        """
        SELECT COUNT(*)
        FROM daily_panel AS daily
        JOIN read_csv_auto(
            'data/manual/security_code_history.csv',
            header = true,
            all_varchar = true
        ) AS mapping
          ON daily.ts_code IN (mapping.old_ts_code, mapping.new_ts_code)
        WHERE daily.canonical_ts_code <> mapping.canonical_ts_code
           OR strftime(daily.list_date, '%Y%m%d')
                <> mapping.original_list_date
           OR NOT daily.has_manual_code_history
        """
    ).fetchone()[0]
    assert mismatches == 0

    canonical_duplicate_dates = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT canonical_ts_code, trade_date
            FROM daily_panel
            WHERE has_manual_code_history
            GROUP BY ALL
            HAVING COUNT(DISTINCT ts_code) > 1
        )
        """
    ).fetchone()[0]
    assert canonical_duplicate_dates > 0


def test_bj_invalid_listing_intervals_are_documented_exclusions(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    invalid_rows, non_bj_rows, sh_sz_rows = connection.execute(
        """
        SELECT
            COUNT(*) FILTER (WHERE invalid_listing_interval),
            COUNT(*) FILTER (
                WHERE invalid_listing_interval AND exchange <> 'BJ'
            ),
            COUNT(*) FILTER (
                WHERE invalid_listing_interval AND is_sh_sz
            )
        FROM daily_panel
        """
    ).fetchone()
    assert invalid_rows == 75958
    assert non_bj_rows == 0
    assert sh_sz_rows == 0


def test_execution_blocks_missing_prices_and_open_suspensions(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    invalid = connection.execute(
        """
        SELECT COUNT(*) FROM execution_status
        WHERE (
                NOT has_daily_price
                AND (NOT cannot_buy_at_open OR NOT cannot_sell_at_open)
              )
           OR (
                suspended_at_open
                AND (NOT cannot_buy_at_open OR NOT cannot_sell_at_open)
              )
           OR (
                has_resume_r AND NOT has_suspend_s AND suspended_at_open
              )
        """
    ).fetchone()[0]
    assert invalid == 0


@pytest.mark.parametrize("view", ["industry_history", "name_history"])
def test_history_intervals_do_not_overlap(
    connection: duckdb.DuckDBPyConnection, view: str
) -> None:
    overlaps = connection.execute(
        f"""
        WITH numbered AS (
            SELECT row_number() OVER () AS row_id, *
            FROM {view}
        )
        SELECT COUNT(*)
        FROM numbered AS left_row
        JOIN numbered AS right_row
          ON left_row.ts_code = right_row.ts_code
         AND left_row.row_id < right_row.row_id
         AND left_row.effective_from
             <= coalesce(right_row.effective_to, DATE '9999-12-31')
         AND right_row.effective_from
             <= coalesce(left_row.effective_to, DATE '9999-12-31')
        """
    ).fetchone()[0]
    assert overlaps == 0


def test_month_end_panel_scope_and_uniqueness(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    columns = {
        row[0]
        for row in connection.execute(
            "DESCRIBE SELECT * FROM month_end_base_panel"
        ).fetchall()
    }
    assert "listing_age_trading_days" in columns
    assert "canonical_ts_code" in columns
    assert "invalid_listing_interval" in columns
    assert "listing_age_days" not in columns
    assert not columns & {
        "bm_proxy",
        "momentum_12_1",
        "lowvol_60",
        "rank_ic",
        "factor_score",
        "next_month_return",
    }
    duplicates = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT ts_code, signal_date, COUNT(*) AS n
            FROM month_end_base_panel GROUP BY ALL HAVING n > 1
        )
        """
    ).fetchone()[0]
    assert duplicates == 0


def test_benchmark_code_dates_and_fields(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    rows, wrong_code, duplicate_dates = connection.execute(
        """
        SELECT
            COUNT(*),
            COUNT(*) FILTER (WHERE benchmark_code <> '000985.CSI'),
            COUNT(*) - COUNT(DISTINCT trade_date)
        FROM benchmark_daily
        """
    ).fetchone()
    columns = {
        row[0]
        for row in connection.execute(
            "DESCRIBE SELECT * FROM benchmark_daily"
        ).fetchall()
    }
    assert rows > 0
    assert wrong_code == 0
    assert duplicate_dates == 0
    assert columns == {
        "benchmark_code",
        "trade_date",
        "close",
        "benchmark_return",
    }


def test_confirmed_stamp_duty_configuration() -> None:
    policy = yaml.safe_load(
        absolute("configs/trading_costs.yaml").read_text(encoding="utf-8")
    )
    assert policy["policy_status"] == "CONFIRMED"
    assert policy["stamp_duty"]["buy_rate"] == 0.0
    assert [
        (
            str(item["effective_from"]),
            str(item["effective_to"]),
            item["sell_rate"],
        )
        for item in policy["stamp_duty"]["periods"]
    ] == [
        ("2016-01-01", "2023-08-27", 0.001),
        ("2023-08-28", "2025-12-31", 0.0005),
    ]
