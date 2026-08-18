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

from a_share_p2.audit import PB_DISCLOSURE
from a_share_p2.config import absolute, load_config


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def connection(config: dict) -> duckdb.DuckDBPyConnection:
    panel = absolute(config["outputs"]["single_factor_panel"])
    if not panel.is_file():
        pytest.fail(f"缺少P2输出：{panel}")
    con = duckdb.connect()
    con.execute(
        f"""
        CREATE VIEW panel AS
        SELECT * FROM read_parquet(
            '{config["outputs"]["single_factor_panel"]}'
        )
        """
    )
    yield con
    con.close()


def test_p2_config_paths_are_relative(config: dict) -> None:
    paths = (
        list(config["inputs"].values())
        + list(config["outputs"].values())
        + list(config["protected_p2_inputs"])
    )
    assert all(not Path(value).is_absolute() for value in paths)


def test_boundary_gate_all_pass(config: dict) -> None:
    tests = pd.read_csv(
        absolute(config["outputs"]["security_code_boundary_tests"])
    )
    summary = pd.read_csv(
        absolute(config["outputs"]["security_code_boundary_summary"])
    )
    assert len(tests) == 18
    assert (tests["status"] == "PASS").all()
    assert len(summary) == 3
    assert (summary["boundary_status"] == "PASS").all()
    assert summary["valid_canonical_duplicate_rows"].sum() == 0
    assert summary["raw_canonical_duplicate_rows"].sum() == 4333


def test_panel_is_research_only_and_unique(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    result = connection.execute(
        """
        SELECT
            min(signal_date),
            max(signal_date),
            count(DISTINCT signal_date),
            count(*) - count(DISTINCT (
                canonical_ts_code, signal_date
            )),
            count(*) FILTER (
                WHERE signal_date >= DATE '2020-01-01'
            ),
            count(*) FILTER (
                WHERE signal_date >= DATE '2022-01-01'
            )
        FROM panel
        """
    ).fetchone()
    assert str(result[0]) == "2016-01-29"
    assert str(result[1]) == "2019-12-31"
    assert result[2] == 48
    assert result[3:] == (0, 0, 0)


def test_source_code_respects_effective_interval(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    violations = connection.execute(
        """
        WITH mapping(old_code, new_code, effective_date) AS (
            VALUES
                ('000022.SZ','001872.SZ',DATE '2018-12-26'),
                ('000043.SZ','001914.SZ',DATE '2019-12-16'),
                ('300114.SZ','302132.SZ',DATE '2025-02-17')
        )
        SELECT count(*)
        FROM panel
        INNER JOIN mapping
          ON panel.ts_code IN (mapping.old_code, mapping.new_code)
        WHERE (
            panel.ts_code = mapping.old_code
            AND panel.signal_date >= mapping.effective_date
        ) OR (
            panel.ts_code = mapping.new_code
            AND panel.signal_date < mapping.effective_date
        )
        """
    ).fetchone()[0]
    assert violations == 0


def test_frozen_factor_formulas(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    errors = connection.execute(
        """
        SELECT
            max(abs(bm_proxy - 1.0 / pb)) FILTER (
                WHERE bm_proxy IS NOT NULL
            ),
            max(abs(
                momentum_12_1
                - adjusted_price_t_minus_21
                    / adjusted_price_t_minus_252 + 1.0
            )) FILTER (WHERE momentum_12_1 IS NOT NULL),
            max(abs(
                lowvol_60 + rolling_return_std_60
            )) FILTER (WHERE lowvol_60 IS NOT NULL),
            max(abs(
                next_month_return
                - next_adjusted_price / adjusted_price + 1.0
            )) FILTER (WHERE next_month_return IS NOT NULL)
        FROM panel
        """
    ).fetchone()
    assert all(error is not None and error <= 1e-12 for error in errors)


def test_forward_label_is_next_month_only(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    violations = connection.execute(
        """
        SELECT count(*)
        FROM panel
        WHERE next_month_return IS NOT NULL
          AND (
            next_signal_date <= signal_date
            OR date_diff('month', signal_date, next_signal_date) <> 1
            OR next_signal_date > DATE '2019-12-31'
          )
        """
    ).fetchone()[0]
    assert violations == 0


def test_universe_rules_are_exact(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    violations = connection.execute(
        """
        SELECT
            count(*) FILTER (
                WHERE universe_eligible
                  AND (
                    listing_age_trading_days < 120
                    OR pb <= 0
                    OR bm_proxy IS NULL
                    OR momentum_12_1 IS NULL
                    OR lowvol_60 IS NULL
                    OR canonical_liquidity_observations <> 20
                    OR canonical_liquidity_20d
                        < liquidity_20pct_cutoff
                    OR is_st_name_flag
                    OR NOT is_sh_sz
                    OR NOT is_within_listing_window
                    OR NOT security_code_interval_valid
                    OR NOT has_listing_reference
                  )
            ),
            count(*) FILTER (
                WHERE universe_pre_liquidity
                  AND canonical_liquidity_20d
                        >= liquidity_20pct_cutoff
                  AND NOT universe_eligible
            )
        FROM panel
        """
    ).fetchone()
    assert violations == (0, 0)


def test_cross_sectional_zscores(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    errors = connection.execute(
        """
        WITH monthly AS (
            SELECT
                signal_date,
                avg(bm_proxy_z) FILTER (
                    WHERE universe_eligible
                ) AS bm_mean,
                stddev_samp(bm_proxy_z) FILTER (
                    WHERE universe_eligible
                ) AS bm_std,
                avg(momentum_12_1_z) FILTER (
                    WHERE universe_eligible
                ) AS momentum_mean,
                stddev_samp(momentum_12_1_z) FILTER (
                    WHERE universe_eligible
                ) AS momentum_std,
                avg(lowvol_60_z) FILTER (
                    WHERE universe_eligible
                ) AS lowvol_mean,
                stddev_samp(lowvol_60_z) FILTER (
                    WHERE universe_eligible
                ) AS lowvol_std
            FROM panel
            GROUP BY signal_date
        )
        SELECT
            greatest(
                max(abs(bm_mean)),
                max(abs(momentum_mean)),
                max(abs(lowvol_mean))
            ),
            greatest(
                max(abs(bm_std - 1.0)),
                max(abs(momentum_std - 1.0)),
                max(abs(lowvol_std - 1.0))
            )
        FROM monthly
        """
    ).fetchone()
    assert errors[0] <= 1e-12
    assert errors[1] <= 1e-12


def test_universe_does_not_depend_on_label(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    eligible, labeled = connection.execute(
        """
        SELECT
            count(*) FILTER (WHERE universe_eligible),
            count(*) FILTER (
                WHERE universe_eligible
                  AND next_month_return IS NOT NULL
            )
        FROM panel
        WHERE signal_date = DATE '2019-12-31'
        """
    ).fetchone()
    assert eligible > 0
    assert labeled == 0


def test_statistical_output_shapes(config: dict) -> None:
    expected = {
        "factor_coverage": 144,
        "monthly_rank_ic": 144,
        "ic_summary": 3,
        "quintile_returns": 846,
        "annual_results": 12,
        "factor_correlations_monthly": 144,
        "factor_correlations_summary": 3,
        "size_exposure": 144,
        "worst_periods": 30,
    }
    for key, rows in expected.items():
        assert len(pd.read_csv(absolute(config["outputs"][key]))) == rows
    assert len(
        pd.read_csv(absolute(config["outputs"]["industry_exposure"]))
    ) > 0


def test_quintile_spread_is_q5_minus_q1(config: dict) -> None:
    frame = pd.read_csv(
        absolute(config["outputs"]["quintile_returns"])
    )
    pivot = frame.pivot(
        index=["signal_date", "factor"],
        columns="quintile",
        values="mean_next_month_return",
    )
    error = (
        pivot["TOP_MINUS_BOTTOM"] - (pivot["Q5"] - pivot["Q1"])
    ).abs().max()
    assert error <= 1e-12
    assert set(frame["return_type"]) == {
        "DIAGNOSTIC_FORWARD_RETURN_NO_COST"
    }


def test_manifest_scope_and_pb_disclosure(config: dict) -> None:
    path = absolute(config["outputs"]["p2_run_manifest"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert not any(manifest["scope_guards"].values())
    assert manifest["panel"]["validation_or_later_rows"] == 0
    assert manifest["panel"]["oos_rows"] == 0
    assert PB_DISCLOSURE in path.read_text(encoding="utf-8")


def test_no_p3_columns(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    columns = {
        row[0] for row in connection.execute("DESCRIBE panel").fetchall()
    }
    assert not columns & {
        "composite",
        "composite_z",
        "target_weight",
        "actual_weight",
        "order_quantity",
        "transaction_cost",
        "portfolio_return",
        "nav",
    }
