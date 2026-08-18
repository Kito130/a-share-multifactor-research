from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .config import absolute, load_config, sql_path


BOUNDARY_TEST_NAMES = (
    "OLD_LAST_TRADE_DATE",
    "NEW_FIRST_TRADE_DATE",
    "NEW_FIRST_PRE_CLOSE_VS_OLD_CLOSE",
    "OLD_LAST_CLOSE",
    "ADJ_FACTOR_BOUNDARY",
    "CANONICAL_DUPLICATE_DATES",
)


def _require_inputs(config: dict[str, Any]) -> None:
    missing = [
        relative_path
        for relative_path in config["inputs"].values()
        if not absolute(relative_path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"缺少P2边界测试输入：{missing}")


def _boundary_summary(
    connection: duckdb.DuckDBPyConnection, config: dict[str, Any]
) -> pd.DataFrame:
    daily_path = sql_path(config["inputs"]["daily_panel"])
    mapping_path = sql_path(config["inputs"]["security_code_history"])
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW p2_daily AS
        SELECT * FROM read_parquet('{daily_path}')
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW p2_code_history AS
        SELECT
            old_ts_code,
            new_ts_code,
            CAST(strptime(effective_date, '%Y%m%d') AS DATE)
                AS effective_date,
            canonical_ts_code
        FROM read_csv(
            '{mapping_path}',
            header = true,
            all_varchar = true
        )
        """
    )
    return connection.execute(
        """
        WITH old_ranked AS (
            SELECT
                mapping.*,
                daily.trade_date AS old_last_trade_date,
                daily.close AS old_last_close,
                daily.adj_factor AS old_last_adj_factor,
                row_number() OVER (
                    PARTITION BY mapping.old_ts_code
                    ORDER BY daily.trade_date DESC
                ) AS row_number
            FROM p2_code_history AS mapping
            LEFT JOIN p2_daily AS daily
              ON daily.ts_code = mapping.old_ts_code
             AND daily.trade_date < mapping.effective_date
             AND daily.security_code_interval_valid
        ),
        new_ranked AS (
            SELECT
                mapping.old_ts_code,
                mapping.new_ts_code,
                daily.trade_date AS new_first_trade_date,
                daily.pre_close AS new_first_pre_close,
                daily.close AS new_first_close,
                daily.adj_factor AS new_first_adj_factor,
                row_number() OVER (
                    PARTITION BY mapping.new_ts_code
                    ORDER BY daily.trade_date
                ) AS row_number
            FROM p2_code_history AS mapping
            LEFT JOIN p2_daily AS daily
              ON daily.ts_code = mapping.new_ts_code
             AND daily.trade_date >= mapping.effective_date
             AND daily.security_code_interval_valid
        ),
        ranges AS (
            SELECT
                mapping.old_ts_code,
                mapping.new_ts_code,
                min(daily.trade_date) FILTER (
                    WHERE daily.ts_code = mapping.old_ts_code
                ) AS old_raw_first_date,
                max(daily.trade_date) FILTER (
                    WHERE daily.ts_code = mapping.old_ts_code
                ) AS old_raw_last_date,
                min(daily.trade_date) FILTER (
                    WHERE daily.ts_code = mapping.new_ts_code
                ) AS new_raw_first_date,
                max(daily.trade_date) FILTER (
                    WHERE daily.ts_code = mapping.new_ts_code
                ) AS new_raw_last_date,
                count(*) FILTER (
                    WHERE daily.ts_code = mapping.new_ts_code
                      AND daily.trade_date < mapping.effective_date
                ) AS new_code_backfill_rows
            FROM p2_code_history AS mapping
            LEFT JOIN p2_daily AS daily
              ON daily.ts_code IN (
                    mapping.old_ts_code, mapping.new_ts_code
                 )
            GROUP BY mapping.old_ts_code, mapping.new_ts_code
        ),
        duplicate_counts AS (
            SELECT
                mapping.old_ts_code,
                mapping.new_ts_code,
                count(*) - count(DISTINCT daily.trade_date)
                    AS raw_canonical_duplicate_rows,
                count(*) FILTER (
                    WHERE daily.security_code_interval_valid
                )
                - count(DISTINCT daily.trade_date) FILTER (
                    WHERE daily.security_code_interval_valid
                ) AS valid_canonical_duplicate_rows
            FROM p2_code_history AS mapping
            LEFT JOIN p2_daily AS daily
              ON daily.canonical_ts_code = mapping.canonical_ts_code
            GROUP BY mapping.old_ts_code, mapping.new_ts_code
        )
        SELECT
            old.old_ts_code,
            old.new_ts_code,
            old.canonical_ts_code,
            old.effective_date,
            old.old_last_trade_date,
            new.new_first_trade_date,
            old.old_last_close,
            new.new_first_pre_close,
            new.new_first_close,
            old.old_last_adj_factor,
            new.new_first_adj_factor,
            new.new_first_pre_close - old.old_last_close
                AS pre_close_minus_old_close,
            abs(
                new.new_first_adj_factor / old.old_last_adj_factor - 1.0
            ) AS adj_factor_relative_difference,
            new.new_first_close * new.new_first_adj_factor
                / (old.old_last_close * old.old_last_adj_factor) - 1.0
                AS adjusted_boundary_return,
            ranges.old_raw_first_date,
            ranges.old_raw_last_date,
            ranges.new_raw_first_date,
            ranges.new_raw_last_date,
            ranges.new_code_backfill_rows,
            duplicates.raw_canonical_duplicate_rows,
            duplicates.valid_canonical_duplicate_rows
        FROM old_ranked AS old
        INNER JOIN new_ranked AS new
          ON old.old_ts_code = new.old_ts_code
         AND old.new_ts_code = new.new_ts_code
        INNER JOIN ranges
          ON old.old_ts_code = ranges.old_ts_code
         AND old.new_ts_code = ranges.new_ts_code
        INNER JOIN duplicate_counts AS duplicates
          ON old.old_ts_code = duplicates.old_ts_code
         AND old.new_ts_code = duplicates.new_ts_code
        WHERE old.row_number = 1
          AND new.row_number = 1
        ORDER BY old.effective_date
        """
    ).fetchdf()


def _test_rows(
    summary: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    settings = config["boundary_tests"]
    price_tolerance = float(settings["price_absolute_tolerance"])
    adj_tolerance = float(settings["adj_factor_relative_tolerance"])
    duplicate_limit = int(settings["maximum_valid_canonical_duplicate_rows"])
    require_exact_date = bool(
        settings["require_new_first_date_equals_effective_date"]
    )
    rows: list[dict[str, Any]] = []

    def add(
        record: pd.Series,
        test_name: str,
        passed: bool,
        observed: Any,
        criterion: str,
    ) -> None:
        rows.append(
            {
                "old_ts_code": record["old_ts_code"],
                "new_ts_code": record["new_ts_code"],
                "canonical_ts_code": record["canonical_ts_code"],
                "effective_date": record["effective_date"],
                "test_name": test_name,
                "status": "PASS" if passed else "FAIL",
                "observed": observed,
                "criterion": criterion,
            }
        )

    for _, record in summary.iterrows():
        old_date = record["old_last_trade_date"]
        new_date = record["new_first_trade_date"]
        effective_date = record["effective_date"]
        add(
            record,
            "OLD_LAST_TRADE_DATE",
            pd.notna(old_date) and old_date < effective_date,
            old_date,
            "存在，且早于 effective_date",
        )
        new_date_ok = pd.notna(new_date) and new_date >= effective_date
        if require_exact_date:
            new_date_ok = new_date_ok and new_date == effective_date
        add(
            record,
            "NEW_FIRST_TRADE_DATE",
            new_date_ok,
            new_date,
            (
                "存在，且等于 effective_date"
                if require_exact_date
                else "存在，且不早于 effective_date"
            ),
        )
        price_difference = record["pre_close_minus_old_close"]
        add(
            record,
            "NEW_FIRST_PRE_CLOSE_VS_OLD_CLOSE",
            pd.notna(price_difference)
            and abs(float(price_difference)) <= price_tolerance,
            price_difference,
            f"|new_first_pre_close-old_last_close| <= {price_tolerance}",
        )
        old_close = record["old_last_close"]
        add(
            record,
            "OLD_LAST_CLOSE",
            pd.notna(old_close) and float(old_close) > 0,
            old_close,
            "存在且大于0",
        )
        adj_difference = record["adj_factor_relative_difference"]
        add(
            record,
            "ADJ_FACTOR_BOUNDARY",
            pd.notna(adj_difference)
            and float(adj_difference) <= adj_tolerance,
            adj_difference,
            f"相对差异 <= {adj_tolerance}",
        )
        duplicate_rows = record["valid_canonical_duplicate_rows"]
        add(
            record,
            "CANONICAL_DUPLICATE_DATES",
            pd.notna(duplicate_rows)
            and int(duplicate_rows) <= duplicate_limit,
            duplicate_rows,
            f"有效代码区间 canonical 重复行 <= {duplicate_limit}",
        )
    tests = pd.DataFrame(rows)
    if set(tests["test_name"]) != set(BOUNDARY_TEST_NAMES):
        raise RuntimeError("代码边界测试集不完整")
    return tests


def _write_report(
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    output = absolute(config["outputs"]["security_code_boundary_report"])
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = config["boundary_tests"]
    pass_count = int((tests["status"] == "PASS").sum())
    fail_count = int((tests["status"] == "FAIL").sum())
    mapping_pass = (
        tests.groupby("canonical_ts_code")["status"]
        .apply(lambda values: bool((values == "PASS").all()))
        .sum()
    )
    lines = [
        "# P2 历史证券代码边界测试",
        "",
        f"- 运行日期：{date.today().isoformat()}",
        f"- 映射主体：{len(summary)}",
        f"- 主体全部通过：{int(mapping_pass)}",
        f"- 测试结果：{pass_count} PASS / {fail_count} FAIL",
        (
            "- 价格衔接容差："
            f"{settings['price_absolute_tolerance']} 元/股"
        ),
        (
            "- 复权因子相对差异容差："
            f"{settings['adj_factor_relative_tolerance']}"
        ),
        "",
        "## 结论",
        "",
    ]
    if fail_count:
        lines.append(
            "边界闸门 **FAIL**。不得在 canonical_ts_code 下跨代码拼接复权价格。"
        )
    else:
        lines.extend(
            [
                (
                    "边界闸门 **PASS**。仅在 `security_code_interval_valid = True` "
                    "后，允许按 `canonical_ts_code` 形成连续复权价格历史。"
                ),
                (
                    "供应商对新代码回填了旧代码历史；这些回填行继续保留在 P1，"
                    "P2 不将其纳入连续价格历史。"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## 主体摘要",
            "",
            "| old | new | effective | old last | new first | "
            "pre_close差 | adj相对差 | 原始重复 | 有效重复 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            "| {old} | {new} | {effective} | {old_date} | {new_date} | "
            "{price_diff:.6g} | {adj_diff:.6g} | {raw_dup} | {valid_dup} |".format(
                old=row["old_ts_code"],
                new=row["new_ts_code"],
                effective=row["effective_date"],
                old_date=row["old_last_trade_date"],
                new_date=row["new_first_trade_date"],
                price_diff=row["pre_close_minus_old_close"],
                adj_diff=row["adj_factor_relative_difference"],
                raw_dup=int(row["raw_canonical_duplicate_rows"]),
                valid_dup=int(row["valid_canonical_duplicate_rows"]),
            )
        )
    lines.extend(
        [
            "",
            "## 固定处理规则",
            "",
            "- `ts_code` 始终保留当天真实交易代码。",
            "- `canonical_ts_code` 只表示连续上市主体。",
            "- 旧代码只取生效日前记录，新代码只取生效日及以后记录。",
            "- 不使用新代码在生效日前的供应商回填行。",
            "- 只有本报告六类测试全部通过，才允许跨代码计算复权价格收益。",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def run_boundary_tests() -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config()
    _require_inputs(config)
    with duckdb.connect() as connection:
        summary = _boundary_summary(connection, config)
    tests = _test_rows(summary, config)

    summary_status = (
        tests.groupby(
            ["old_ts_code", "new_ts_code", "canonical_ts_code"],
            as_index=False,
        )
        .agg(
            boundary_status=(
                "status",
                lambda values: (
                    "PASS" if bool((values == "PASS").all()) else "FAIL"
                ),
            )
        )
    )
    summary = summary.merge(
        summary_status,
        on=["old_ts_code", "new_ts_code", "canonical_ts_code"],
        how="left",
        validate="one_to_one",
    )
    for key, frame in (
        ("security_code_boundary_summary", summary),
        ("security_code_boundary_tests", tests),
    ):
        output = absolute(config["outputs"][key])
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False, encoding="utf-8-sig")
    _write_report(summary, tests, config)

    if bool((tests["status"] == "FAIL").any()):
        failed = tests.loc[tests["status"] == "FAIL"]
        raise RuntimeError(
            "P2代码边界闸门失败："
            + failed[
                ["canonical_ts_code", "test_name", "observed"]
            ].to_dict(orient="records").__repr__()
        )
    return summary, tests

