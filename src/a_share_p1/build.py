from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pandas as pd
import pyarrow

from .config import PROJECT_ROOT, absolute, load_config, relative_posix


BUILDER_VERSION = "p1.2"


def _log(message: str) -> None:
    print(f"[P1] {message}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_files(config: dict[str, Any]) -> list[Path]:
    files: set[Path] = set()
    for value in config["protected_inputs"]:
        target = absolute(value)
        if target.is_file():
            files.add(target)
        elif target.is_dir():
            files.update(path for path in target.rglob("*") if path.is_file())
        else:
            raise FileNotFoundError(f"受保护输入不存在：{value}")
    return sorted(files, key=relative_posix)


def _hash_snapshot(files: Iterable[Path]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for path in files:
        snapshot[relative_posix(path)] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return snapshot


def _tree_sha256(
    snapshot: dict[str, dict[str, Any]], selected_paths: Iterable[str]
) -> str:
    digest = hashlib.sha256()
    for relative_path in sorted(selected_paths):
        item = snapshot[relative_path]
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["size_bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\n")
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
    temporary_relative = relative_posix(temporary)
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


def _glob_files(pattern: str) -> list[Path]:
    return sorted(
        (path for path in PROJECT_ROOT.glob(pattern) if path.is_file()),
        key=relative_posix,
    )


def _iso_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text[:10]


def _create_source_views(
    connection: duckdb.DuckDBPyConnection, config: dict[str, Any]
) -> None:
    inputs = config["inputs"]
    parquet_views = {
        "src_trade_cal": "trade_cal",
        "src_stock_basic": "stock_basic",
        "src_namechange": "namechange",
        "src_sw_classify": "sw_index_classify",
        "src_sw_member": "sw_index_member_history",
        "src_daily": "daily",
        "src_adj": "adj_factor",
        "src_daily_basic": "daily_basic_month_end",
        "src_limit": "stk_limit",
        "src_suspend": "suspend_d",
    }
    for view_name, input_key in parquet_views.items():
        connection.execute(
            f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT *
            FROM read_parquet('{inputs[input_key]}', union_by_name = true)
            """
        )
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW src_benchmark AS
        SELECT *
        FROM read_csv_auto(
            '{inputs["benchmark_daily"]}',
            header = true,
            all_varchar = false
        )
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW src_security_code_history AS
        SELECT *
        FROM read_csv_auto(
            '{inputs["security_code_history"]}',
            header = true,
            all_varchar = true
        )
        """
    )


def _preflight(
    connection: duckdb.DuckDBPyConnection, config: dict[str, Any]
) -> None:
    cutoff = config["project"]["research_cutoff"]
    duplicate_queries = {
        "daily": (
            "SELECT COUNT(*) FROM (SELECT ts_code, trade_date, COUNT(*) AS n "
            "FROM src_daily GROUP BY ALL HAVING n > 1)"
        ),
        "adj_factor": (
            "SELECT COUNT(*) FROM (SELECT ts_code, trade_date, COUNT(*) AS n "
            "FROM src_adj GROUP BY ALL HAVING n > 1)"
        ),
        "daily_basic_month_end": (
            "SELECT COUNT(*) FROM (SELECT ts_code, trade_date, COUNT(*) AS n "
            "FROM src_daily_basic GROUP BY ALL HAVING n > 1)"
        ),
        "stk_limit": (
            "SELECT COUNT(*) FROM (SELECT ts_code, trade_date, COUNT(*) AS n "
            "FROM src_limit GROUP BY ALL HAVING n > 1)"
        ),
        "stock_basic": (
            "SELECT COUNT(*) FROM (SELECT ts_code, COUNT(*) AS n "
            "FROM src_stock_basic GROUP BY ALL HAVING n > 1)"
        ),
        "trade_cal": (
            "SELECT COUNT(*) FROM (SELECT cal_date, COUNT(*) AS n "
            "FROM src_trade_cal GROUP BY ALL HAVING n > 1)"
        ),
    }
    failures = {
        name: int(connection.execute(query).fetchone()[0])
        for name, query in duplicate_queries.items()
    }
    failures = {name: count for name, count in failures.items() if count}
    if failures:
        raise RuntimeError(f"输入主键不唯一，停止构建：{failures}")

    benchmark = connection.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT ts_code) AS codes,
            MIN(ts_code) AS minimum_code,
            MAX(ts_code) AS maximum_code,
            COUNT(*) - COUNT(DISTINCT trade_date) AS duplicate_dates,
            MIN(trade_date) AS minimum_date,
            MAX(trade_date) AS maximum_date,
            COUNT(*) FILTER (WHERE close IS NULL OR close <= 0) AS invalid_close
        FROM src_benchmark
        """
    ).fetchone()
    if (
        benchmark[0] == 0
        or benchmark[1] != 1
        or benchmark[2] != "000985.CSI"
        or benchmark[3] != "000985.CSI"
        or benchmark[4] != 0
        or benchmark[7] != 0
        or _iso_date(benchmark[6]) > cutoff
    ):
        raise RuntimeError(f"宽基 CSV 预检查失败：{benchmark}")

    missing_adj = connection.execute(
        """
        SELECT COUNT(*)
        FROM src_daily AS d
        ANTI JOIN src_adj AS a USING (ts_code, trade_date)
        """
    ).fetchone()[0]
    if missing_adj:
        raise RuntimeError(f"daily 有 {missing_adj} 行缺少 adj_factor，停止构建")

    manual_mapping = connection.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT old_ts_code) AS old_codes,
            COUNT(DISTINCT new_ts_code) AS new_codes,
            COUNT(DISTINCT canonical_ts_code) AS canonical_codes,
            COUNT(*) FILTER (
                WHERE old_ts_code = new_ts_code
                   OR canonical_ts_code <> new_ts_code
                   OR status <> 'CONFIRMED'
                   OR relationship <> 'SAME_LISTED_ENTITY_CODE_CHANGE'
                   OR strptime(original_list_date, '%Y%m%d')
                        >= strptime(effective_date, '%Y%m%d')
            ) AS invalid_rows
        FROM src_security_code_history
        """
    ).fetchone()
    manual_code_collisions = connection.execute(
        """
        WITH codes AS (
            SELECT old_ts_code AS ts_code FROM src_security_code_history
            UNION ALL
            SELECT new_ts_code AS ts_code FROM src_security_code_history
        )
        SELECT COUNT(*)
        FROM (
            SELECT ts_code, COUNT(*) AS n
            FROM codes GROUP BY ts_code HAVING n > 1
        )
        """
    ).fetchone()[0]
    manual_missing_daily_codes = connection.execute(
        """
        WITH codes AS (
            SELECT old_ts_code AS ts_code FROM src_security_code_history
            UNION
            SELECT new_ts_code AS ts_code FROM src_security_code_history
        )
        SELECT COUNT(*)
        FROM codes
        ANTI JOIN (SELECT DISTINCT ts_code FROM src_daily) USING (ts_code)
        """
    ).fetchone()[0]
    if (
        manual_mapping[:4] != (3, 3, 3, 3)
        or manual_mapping[4] != 0
        or manual_code_collisions != 0
        or manual_missing_daily_codes != 0
    ):
        raise RuntimeError(
            "人工证券代码历史预检查失败："
            f"summary={manual_mapping}, collisions={manual_code_collisions}, "
            f"missing_daily_codes={manual_missing_daily_codes}"
        )


def _input_inventory(
    connection: duckdb.DuckDBPyConnection,
    config: dict[str, Any],
    protected_snapshot: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    definitions = [
        ("trade_cal", "trade_cal", "src_trade_cal", "cal_date"),
        ("stock_basic_all_status", "stock_basic", "src_stock_basic", "list_date"),
        ("namechange", "namechange", "src_namechange", "start_date"),
        ("sw_index_classify", "sw_index_classify", "src_sw_classify", None),
        (
            "sw_index_member_history",
            "sw_index_member_history",
            "src_sw_member",
            "in_date",
        ),
        ("daily", "daily", "src_daily", "trade_date"),
        ("adj_factor", "adj_factor", "src_adj", "trade_date"),
        (
            "daily_basic_month_end",
            "daily_basic_month_end",
            "src_daily_basic",
            "trade_date",
        ),
        ("stk_limit", "stk_limit", "src_limit", "trade_date"),
        ("suspend_d", "suspend_d", "src_suspend", "trade_date"),
        ("benchmark_daily", "benchmark_daily", "src_benchmark", "trade_date"),
        (
            "security_code_history",
            "security_code_history",
            "src_security_code_history",
            "effective_date",
        ),
    ]
    rows: list[dict[str, Any]] = []
    cutoff = config["project"]["research_cutoff"]
    for dataset, input_key, view, date_field in definitions:
        pattern = config["inputs"][input_key]
        files = _glob_files(pattern)
        if date_field:
            row_count, minimum, maximum = connection.execute(
                f"SELECT COUNT(*), MIN({date_field}), MAX({date_field}) FROM {view}"
            ).fetchone()
        else:
            row_count = connection.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
            minimum, maximum = None, None
        selected = [
            relative_posix(path)
            for path in files
            if relative_posix(path) in protected_snapshot
        ]
        maximum_iso = _iso_date(maximum)
        notes: list[str] = []
        if maximum_iso and maximum_iso > cutoff:
            notes.append("静态源含截止日后版本；P1按研究截止日过滤")
        if dataset == "benchmark_daily":
            notes.append("实际代码、日期和重复值已在预检查中验证")
        if dataset == "security_code_history":
            notes.append("人工确认的同一上市主体证券代码变更；P1仅建立映射")
        rows.append(
            {
                "dataset": dataset,
                "path": pattern,
                "exists": bool(files),
                "file_count": len(files),
                "row_count": int(row_count),
                "minimum_date": _iso_date(minimum),
                "maximum_date": maximum_iso,
                "aggregate_sha256": (
                    _tree_sha256(protected_snapshot, selected) if selected else ""
                ),
                "status": "PASS" if files and row_count else "FAIL",
                "notes": "；".join(notes),
            }
        )

    parts_files = [
        path
        for path in _protected_files(config)
        if relative_posix(path).startswith("data/_parts/")
    ]
    part_relatives = [relative_posix(path) for path in parts_files]
    rows.append(
        {
            "dataset": "_parts",
            "path": "data/_parts",
            "exists": bool(parts_files),
            "file_count": len(parts_files),
            "row_count": pd.NA,
            "minimum_date": "",
            "maximum_date": "",
            "aggregate_sha256": _tree_sha256(protected_snapshot, part_relatives),
            "status": "PASS" if parts_files else "FAIL",
            "notes": "受保护的采集分片；不作为P1规范表的直接输入",
        }
    )
    return pd.DataFrame(rows)


def _prepare_calendar_and_stock(
    connection: duckdb.DuckDBPyConnection, config: dict[str, Any]
) -> None:
    cutoff = config["project"]["research_cutoff"]
    calendar_start = config["project"]["calendar_start"]
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW open_calendar AS
        SELECT
            CAST(strptime(cal_date, '%Y%m%d') AS DATE) AS cal_date,
            row_number() OVER (ORDER BY cal_date) AS open_day_number
        FROM src_trade_cal
        WHERE is_open = 1
          AND CAST(strptime(cal_date, '%Y%m%d') AS DATE)
              BETWEEN DATE '{calendar_start}' AND DATE '{cutoff}'
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE code_identity_map AS
        WITH typed AS (
            SELECT
                old_ts_code,
                new_ts_code,
                CAST(strptime(effective_date, '%Y%m%d') AS DATE)
                    AS effective_date,
                canonical_ts_code,
                CAST(strptime(original_list_date, '%Y%m%d') AS DATE)
                    AS original_list_date
            FROM src_security_code_history
        )
        SELECT
            old_ts_code AS ts_code,
            canonical_ts_code,
            old_ts_code,
            new_ts_code,
            effective_date,
            original_list_date,
            'OLD_CODE' AS security_code_role
        FROM typed
        UNION ALL
        SELECT
            new_ts_code AS ts_code,
            canonical_ts_code,
            old_ts_code,
            new_ts_code,
            effective_date,
            original_list_date,
            'NEW_CODE' AS security_code_role
        FROM typed
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE stock_meta AS
        WITH stock_basic_typed AS (
            SELECT
                ts_code,
                CAST(strptime(list_date, '%Y%m%d') AS DATE) AS list_date
            FROM src_stock_basic
        ),
        code_universe AS (
            SELECT ts_code FROM stock_basic_typed
            UNION
            SELECT ts_code FROM code_identity_map
        ),
        typed AS (
            SELECT
                universe.ts_code,
                stock.list_date AS source_list_date,
                coalesce(mapping.original_list_date, stock.list_date)
                    AS list_date,
                coalesce(mapping.canonical_ts_code, universe.ts_code)
                    AS canonical_ts_code,
                mapping.old_ts_code,
                mapping.new_ts_code,
                mapping.effective_date AS security_code_effective_date,
                mapping.security_code_role,
                stock.ts_code IS NOT NULL AS has_stock_basic_record,
                mapping.ts_code IS NOT NULL AS has_manual_code_history
            FROM code_universe AS universe
            LEFT JOIN stock_basic_typed AS stock USING (ts_code)
            LEFT JOIN code_identity_map AS mapping USING (ts_code)
        )
        SELECT
            typed.ts_code,
            typed.canonical_ts_code,
            typed.source_list_date,
            typed.list_date,
            CAST(NULL AS DATE) AS delist_date,
            typed.old_ts_code,
            typed.new_ts_code,
            typed.security_code_effective_date,
            typed.security_code_role,
            typed.has_stock_basic_record,
            typed.has_manual_code_history,
            typed.list_date IS NOT NULL AS has_listing_reference,
            typed.list_date < DATE '{calendar_start}' AS listing_age_is_lower_bound,
            (
                SELECT MIN(open_day_number)
                FROM open_calendar AS calendar
                WHERE calendar.cal_date >=
                    CASE
                        WHEN typed.list_date < DATE '{calendar_start}'
                            THEN DATE '{calendar_start}'
                        ELSE typed.list_date
                    END
            ) AS listing_start_open_day
        FROM typed
        """
    )


def _build_histories(
    connection: duckdb.DuckDBPyConnection, config: dict[str, Any]
) -> None:
    cutoff = config["project"]["research_cutoff"]
    outputs = config["outputs"]
    _log("构建历史行业表")
    _copy_query(
        connection,
        f"""
        SELECT DISTINCT
            source.ts_code,
            coalesce(mapping.canonical_ts_code, source.ts_code)
                AS canonical_ts_code,
            source.l3_code AS industry_code,
            source.l3_name AS industry_name,
            CAST(strptime(source.in_date, '%Y%m%d') AS DATE) AS effective_from,
            CASE
                WHEN source.out_date IS NULL
                  OR trim(source.out_date) = ''
                  OR CAST(strptime(source.out_date, '%Y%m%d') AS DATE)
                        > DATE '{cutoff}'
                    THEN CAST(NULL AS DATE)
                ELSE CAST(strptime(source.out_date, '%Y%m%d') AS DATE)
            END AS effective_to
        FROM src_sw_member AS source
        LEFT JOIN code_identity_map AS mapping USING (ts_code)
        WHERE source.in_date IS NOT NULL
          AND trim(source.in_date) <> ''
          AND CAST(strptime(source.in_date, '%Y%m%d') AS DATE)
                <= DATE '{cutoff}'
        ORDER BY source.ts_code, effective_from, industry_code
        """,
        outputs["industry_history"],
    )

    _log("构建历史名称表")
    _copy_query(
        connection,
        f"""
        SELECT DISTINCT
            source.ts_code,
            coalesce(mapping.canonical_ts_code, source.ts_code)
                AS canonical_ts_code,
            source.name AS stock_name,
            CAST(strptime(source.start_date, '%Y%m%d') AS DATE)
                AS effective_from,
            CASE
                WHEN source.end_date IS NULL
                  OR trim(source.end_date) = ''
                  OR CAST(strptime(source.end_date, '%Y%m%d') AS DATE)
                        > DATE '{cutoff}'
                    THEN CAST(NULL AS DATE)
                ELSE CAST(strptime(source.end_date, '%Y%m%d') AS DATE)
            END AS effective_to,
            upper(coalesce(source.name, '')) LIKE '%ST%' AS is_st_name_flag
        FROM src_namechange AS source
        LEFT JOIN code_identity_map AS mapping USING (ts_code)
        WHERE source.start_date IS NOT NULL
          AND trim(source.start_date) <> ''
          AND CAST(strptime(source.start_date, '%Y%m%d') AS DATE)
                <= DATE '{cutoff}'
        ORDER BY source.ts_code, effective_from, stock_name
        """,
        outputs["name_history"],
    )


def _build_daily_panel(
    connection: duckdb.DuckDBPyConnection, config: dict[str, Any]
) -> None:
    cutoff = config["project"]["research_cutoff"]
    output = config["outputs"]["daily_panel"]
    _log("构建日频标准化面板")
    _copy_query(
        connection,
        f"""
        WITH joined AS (
            SELECT
                daily.ts_code,
                stock.canonical_ts_code,
                CAST(strptime(daily.trade_date, '%Y%m%d') AS DATE) AS trade_date,
                split_part(daily.ts_code, '.', 2) AS exchange,
                daily.open,
                daily.high,
                daily.low,
                daily.close,
                daily.pre_close,
                daily.change,
                daily.pct_chg,
                daily.vol AS volume_raw,
                daily.amount AS amount_raw,
                daily.vol * 100.0 AS volume_share,
                daily.amount * 1000.0 AS amount_cny,
                adj.adj_factor,
                daily.close * adj.adj_factor AS close_hfq,
                stock.source_list_date,
                stock.list_date,
                stock.delist_date,
                stock.security_code_effective_date,
                stock.security_code_role,
                stock.has_manual_code_history,
                stock.listing_age_is_lower_bound,
                stock.listing_start_open_day,
                calendar.open_day_number,
                coalesce(stock.has_stock_basic_record, false)
                    AS has_stock_basic_record,
                coalesce(stock.has_listing_reference, false)
                    AS has_listing_reference,
                CASE
                    WHEN NOT coalesce(stock.has_listing_reference, false)
                        THEN false
                    ELSE
                        CAST(strptime(daily.trade_date, '%Y%m%d') AS DATE)
                            >= stock.list_date
                        AND (
                            stock.delist_date IS NULL
                            OR CAST(
                                strptime(daily.trade_date, '%Y%m%d') AS DATE
                            ) <= stock.delist_date
                        )
                END AS is_within_listing_window,
                CASE
                    WHEN stock.security_code_role = 'OLD_CODE'
                        THEN CAST(strptime(daily.trade_date, '%Y%m%d') AS DATE)
                            < stock.security_code_effective_date
                    WHEN stock.security_code_role = 'NEW_CODE'
                        THEN CAST(strptime(daily.trade_date, '%Y%m%d') AS DATE)
                            >= stock.security_code_effective_date
                    ELSE true
                END AS security_code_interval_valid
            FROM src_daily AS daily
            LEFT JOIN src_adj AS adj USING (ts_code, trade_date)
            LEFT JOIN stock_meta AS stock USING (ts_code)
            LEFT JOIN open_calendar AS calendar
              ON calendar.cal_date =
                 CAST(strptime(daily.trade_date, '%Y%m%d') AS DATE)
            WHERE CAST(strptime(daily.trade_date, '%Y%m%d') AS DATE)
                  <= DATE '{cutoff}'
        ),
        calculated AS (
            SELECT
                *,
                close_hfq AS adjusted_close,
                CASE
                    WHEN security_code_interval_valid
                     AND coalesce(
                        lag(security_code_interval_valid) OVER (
                            PARTITION BY ts_code ORDER BY trade_date
                        ),
                        false
                     )
                    THEN close_hfq
                        / lag(close_hfq) OVER (
                            PARTITION BY ts_code ORDER BY trade_date
                        ) - 1.0
                    ELSE CAST(NULL AS DOUBLE)
                END AS daily_return,
                CASE
                    WHEN is_within_listing_window
                      AND open_day_number IS NOT NULL
                      AND listing_start_open_day IS NOT NULL
                    THEN CAST(
                        open_day_number - listing_start_open_day + 1
                        AS BIGINT
                    )
                    ELSE CAST(NULL AS BIGINT)
                END AS listing_age_trading_days,
                CASE
                    WHEN NOT has_listing_reference
                        THEN 'MISSING_LISTING_REFERENCE'
                    WHEN NOT is_within_listing_window
                        THEN 'OUTSIDE_CONFIRMED_LISTING_WINDOW'
                    WHEN has_manual_code_history AND listing_age_is_lower_bound
                        THEN 'MANUAL_ORIGINAL_LIST_DATE_LOWER_BOUND'
                    WHEN has_manual_code_history
                        THEN 'MANUAL_ORIGINAL_LIST_DATE_EXACT'
                    WHEN listing_age_is_lower_bound
                        THEN 'LOWER_BOUND_CALENDAR_START'
                    ELSE 'EXACT_WITHIN_AVAILABLE_CALENDAR'
                END AS listing_age_status,
                NOT is_within_listing_window AS invalid_listing_interval,
                NOT security_code_interval_valid
                    AS invalid_security_code_interval,
                split_part(ts_code, '.', 2) IN ('SH', 'SZ') AS is_sh_sz
            FROM joined
        )
        SELECT
            ts_code,
            canonical_ts_code,
            trade_date,
            exchange,
            open,
            high,
            low,
            close,
            pre_close,
            change,
            pct_chg,
            volume_raw,
            amount_raw,
            volume_share,
            amount_cny,
            adj_factor,
            close_hfq,
            adjusted_close,
            daily_return,
            source_list_date,
            list_date,
            delist_date,
            security_code_effective_date,
            security_code_role,
            listing_age_trading_days,
            listing_age_is_lower_bound,
            listing_age_status,
            has_stock_basic_record,
            has_manual_code_history,
            has_listing_reference,
            is_within_listing_window,
            invalid_listing_interval,
            security_code_interval_valid,
            invalid_security_code_interval,
            is_sh_sz
        FROM calculated
        ORDER BY ts_code, trade_date
        """,
        output,
    )


def _prepare_suspend_aggregate(
    connection: duckdb.DuckDBPyConnection, config: dict[str, Any]
) -> None:
    cutoff = config["project"]["research_cutoff"]
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE suspend_aggregate AS
        WITH base AS (
            SELECT
                ts_code,
                CAST(strptime(trade_date, '%Y%m%d') AS DATE) AS trade_date,
                upper(trim(suspend_type)) AS suspend_type,
                trim(coalesce(suspend_timing, '')) AS suspend_timing,
                upper(trim(suspend_type)) || ':' ||
                    CASE
                        WHEN trim(coalesce(suspend_timing, '')) = ''
                          AND upper(trim(suspend_type)) = 'S'
                            THEN 'FULL_DAY'
                        WHEN trim(coalesce(suspend_timing, '')) = ''
                            THEN 'UNSPECIFIED'
                        ELSE trim(suspend_timing)
                    END AS event_text
            FROM src_suspend
            WHERE CAST(strptime(trade_date, '%Y%m%d') AS DATE)
                  <= DATE '{cutoff}'
        ),
        expanded AS (
            SELECT
                base.*,
                trim(segment) AS segment
            FROM base
            LEFT JOIN LATERAL unnest(
                CASE
                    WHEN suspend_timing = '' THEN ['']
                    ELSE string_split(suspend_timing, ',')
                END
            ) AS fragments(segment) ON true
        )
        SELECT
            ts_code,
            trade_date,
            true AS has_suspend_record,
            bool_or(suspend_type = 'S') AS has_suspend_s,
            bool_or(suspend_type = 'R') AS has_resume_r,
            string_agg(
                DISTINCT event_text,
                '|' ORDER BY event_text
            ) AS suspend_intervals,
            bool_or(
                suspend_type = 'S'
                AND (
                    segment = ''
                    OR (
                        length(segment) = 11
                        AND split_part(segment, '-', 1) <= '09:30'
                        AND split_part(segment, '-', 2) >= '09:30'
                    )
                )
            ) AS suspended_at_open
        FROM expanded
        GROUP BY ts_code, trade_date
        """
    )


def _build_execution_status(
    connection: duckdb.DuckDBPyConnection, config: dict[str, Any]
) -> None:
    cutoff = config["project"]["research_cutoff"]
    daily_panel = config["outputs"]["daily_panel"]
    output = config["outputs"]["execution_status"]
    _prepare_suspend_aggregate(connection, config)
    _log("构建成交状态面板")
    _copy_query(
        connection,
        f"""
        WITH keys AS (
            SELECT
                ts_code,
                CAST(strptime(trade_date, '%Y%m%d') AS DATE) AS trade_date
            FROM src_daily
            WHERE CAST(strptime(trade_date, '%Y%m%d') AS DATE)
                  <= DATE '{cutoff}'
            UNION
            SELECT
                ts_code,
                CAST(strptime(trade_date, '%Y%m%d') AS DATE) AS trade_date
            FROM src_limit
            WHERE CAST(strptime(trade_date, '%Y%m%d') AS DATE)
                  <= DATE '{cutoff}'
            UNION
            SELECT ts_code, trade_date
            FROM suspend_aggregate
        ),
        joined AS (
            SELECT
                keys.ts_code,
                coalesce(
                    daily.canonical_ts_code,
                    mapping.canonical_ts_code,
                    keys.ts_code
                ) AS canonical_ts_code,
                keys.trade_date,
                daily.open AS open_price,
                daily.ts_code IS NOT NULL
                    AND daily.open IS NOT NULL
                    AND daily.close IS NOT NULL AS has_daily_price,
                coalesce(suspend.has_suspend_record, false)
                    AS has_suspend_record,
                coalesce(suspend.has_suspend_s, false) AS has_suspend_s,
                coalesce(suspend.has_resume_r, false) AS has_resume_r,
                suspend.suspend_intervals,
                coalesce(suspend.suspended_at_open, false)
                    AS suspended_at_open,
                limits.up_limit,
                limits.down_limit,
                CASE
                    WHEN mapping.security_code_role = 'OLD_CODE'
                        THEN keys.trade_date < mapping.effective_date
                    WHEN mapping.security_code_role = 'NEW_CODE'
                        THEN keys.trade_date >= mapping.effective_date
                    ELSE true
                END AS security_code_interval_valid
            FROM keys
            LEFT JOIN read_parquet('{daily_panel}') AS daily
              USING (ts_code, trade_date)
            LEFT JOIN code_identity_map AS mapping USING (ts_code)
            LEFT JOIN (
                SELECT
                    ts_code,
                    CAST(strptime(trade_date, '%Y%m%d') AS DATE) AS trade_date,
                    up_limit,
                    down_limit
                FROM src_limit
            ) AS limits USING (ts_code, trade_date)
            LEFT JOIN suspend_aggregate AS suspend USING (ts_code, trade_date)
        ),
        flags AS (
            SELECT
                *,
                coalesce(
                    has_daily_price
                    AND up_limit IS NOT NULL
                    AND (
                        round(open_price, 2) >= round(up_limit, 2)
                        OR abs(open_price - up_limit) <= 0.005
                    ),
                    false
                ) AS open_at_up_limit,
                coalesce(
                    has_daily_price
                    AND down_limit IS NOT NULL
                    AND (
                        round(open_price, 2) <= round(down_limit, 2)
                        OR abs(open_price - down_limit) <= 0.005
                    ),
                    false
                ) AS open_at_down_limit
            FROM joined
        ),
        decisions AS (
            SELECT
                *,
                (NOT has_daily_price)
                    OR suspended_at_open
                    OR open_at_up_limit AS cannot_buy_at_open,
                (NOT has_daily_price)
                    OR suspended_at_open
                    OR open_at_down_limit AS cannot_sell_at_open
            FROM flags
        )
        SELECT
            ts_code,
            canonical_ts_code,
            trade_date,
            open_price,
            has_daily_price,
            has_suspend_record,
            has_suspend_s,
            has_resume_r,
            suspend_intervals,
            suspended_at_open,
            up_limit,
            down_limit,
            open_at_up_limit,
            open_at_down_limit,
            security_code_interval_valid,
            NOT security_code_interval_valid
                AS invalid_security_code_interval,
            cannot_buy_at_open,
            cannot_sell_at_open,
            nullif(
                concat_ws(
                    '|',
                    CASE WHEN NOT has_daily_price
                        THEN 'NO_DAILY_PRICE' END,
                    CASE WHEN suspended_at_open
                        THEN 'SUSPENDED_AT_OPEN' END,
                    CASE WHEN open_at_up_limit
                        THEN 'OPEN_AT_UP_LIMIT' END,
                    CASE WHEN open_at_down_limit
                        THEN 'OPEN_AT_DOWN_LIMIT' END
                ),
                ''
            ) AS execution_block_reason
        FROM decisions
        ORDER BY ts_code, trade_date
        """,
        output,
    )


def _build_month_end_panel(
    connection: duckdb.DuckDBPyConnection, config: dict[str, Any]
) -> None:
    cutoff = config["project"]["research_cutoff"]
    history_threshold = int(
        config["project"]["history_threshold_observations"]
    )
    liquidity_window = int(
        config["project"]["liquidity_window_observations"]
    )
    daily_panel = config["outputs"]["daily_panel"]
    industry_history = config["outputs"]["industry_history"]
    name_history = config["outputs"]["name_history"]
    output = config["outputs"]["month_end_base_panel"]
    preceding = liquidity_window - 1
    _log("构建月末基础面板")
    _copy_query(
        connection,
        f"""
        WITH daily_metrics AS (
            SELECT
                ts_code,
                canonical_ts_code,
                trade_date,
                listing_age_trading_days,
                listing_age_is_lower_bound,
                listing_age_status,
                has_stock_basic_record,
                has_manual_code_history,
                has_listing_reference,
                is_within_listing_window,
                invalid_listing_interval,
                security_code_interval_valid,
                invalid_security_code_interval,
                is_sh_sz,
                count(*) FILTER (
                    WHERE is_within_listing_window
                      AND security_code_interval_valid
                ) OVER (
                    PARTITION BY ts_code ORDER BY trade_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS available_history_observations,
                count(amount_cny) FILTER (
                    WHERE is_within_listing_window
                      AND security_code_interval_valid
                ) OVER (
                    PARTITION BY ts_code ORDER BY trade_date
                    ROWS BETWEEN {preceding} PRECEDING AND CURRENT ROW
                ) AS liquidity_observations,
                avg(
                    CASE
                        WHEN is_within_listing_window
                         AND security_code_interval_valid
                        THEN amount_cny
                    END
                ) OVER (
                    PARTITION BY ts_code ORDER BY trade_date
                    ROWS BETWEEN {preceding} PRECEDING AND CURRENT ROW
                ) AS liquidity_mean
            FROM read_parquet('{daily_panel}')
        ),
        basic AS (
            SELECT
                ts_code,
                CAST(strptime(trade_date, '%Y%m%d') AS DATE) AS signal_date,
                close AS daily_basic_close,
                turnover_rate,
                turnover_rate_f,
                volume_ratio,
                pe,
                pe_ttm,
                pb,
                ps,
                ps_ttm,
                dv_ratio,
                dv_ttm,
                total_share,
                float_share,
                free_share,
                total_mv,
                circ_mv,
                limit_status
            FROM src_daily_basic
            WHERE CAST(strptime(trade_date, '%Y%m%d') AS DATE)
                  <= DATE '{cutoff}'
        )
        SELECT
            basic.ts_code,
            metrics.canonical_ts_code,
            basic.signal_date,
            basic.daily_basic_close,
            basic.turnover_rate,
            basic.turnover_rate_f,
            basic.volume_ratio,
            basic.pe,
            basic.pe_ttm,
            basic.pb,
            basic.ps,
            basic.ps_ttm,
            basic.dv_ratio,
            basic.dv_ttm,
            basic.total_share,
            basic.float_share,
            basic.free_share,
            basic.total_share * 10000.0 AS total_share_shares,
            basic.float_share * 10000.0 AS float_share_shares,
            basic.free_share * 10000.0 AS free_share_shares,
            basic.total_mv,
            basic.circ_mv,
            basic.total_mv * 10000.0 AS total_mv_cny,
            basic.circ_mv * 10000.0 AS circ_mv_cny,
            basic.limit_status,
            metrics.listing_age_trading_days,
            metrics.listing_age_is_lower_bound,
            metrics.listing_age_status,
            metrics.has_stock_basic_record,
            metrics.has_manual_code_history,
            metrics.has_listing_reference,
            metrics.is_within_listing_window,
            metrics.invalid_listing_interval,
            metrics.security_code_interval_valid,
            metrics.invalid_security_code_interval,
            industry.industry_code,
            industry.industry_name,
            industry.effective_from AS industry_effective_from,
            industry.effective_to AS industry_effective_to,
            industry.ts_code IS NOT NULL AS has_industry_history,
            names.stock_name,
            names.effective_from AS name_effective_from,
            names.effective_to AS name_effective_to,
            names.is_st_name_flag,
            names.ts_code IS NOT NULL AS has_name_history,
            metrics.is_sh_sz,
            metrics.available_history_observations,
            metrics.available_history_observations >= {history_threshold}
                AS has_enough_252d_history,
            CASE
                WHEN metrics.liquidity_observations = {liquidity_window}
                    THEN metrics.liquidity_mean
                ELSE CAST(NULL AS DOUBLE)
            END AS liquidity_20d
        FROM basic
        INNER JOIN daily_metrics AS metrics
          ON basic.ts_code = metrics.ts_code
         AND basic.signal_date = metrics.trade_date
        LEFT JOIN read_parquet('{industry_history}') AS industry
          ON basic.ts_code = industry.ts_code
         AND industry.effective_from <= basic.signal_date
         AND (
            industry.effective_to IS NULL
            OR basic.signal_date <= industry.effective_to
         )
        LEFT JOIN read_parquet('{name_history}') AS names
          ON basic.ts_code = names.ts_code
         AND names.effective_from <= basic.signal_date
         AND (
            names.effective_to IS NULL
            OR basic.signal_date <= names.effective_to
         )
        ORDER BY basic.ts_code, basic.signal_date
        """,
        output,
    )


def _build_benchmark(
    connection: duckdb.DuckDBPyConnection, config: dict[str, Any]
) -> None:
    cutoff = config["project"]["research_cutoff"]
    output = config["outputs"]["benchmark_daily"]
    _log("构建宽基标准化面板")
    _copy_query(
        connection,
        f"""
        WITH typed AS (
            SELECT
                ts_code AS benchmark_code,
                CAST(strptime(CAST(trade_date AS VARCHAR), '%Y%m%d') AS DATE)
                    AS trade_date,
                CAST(close AS DOUBLE) AS close
            FROM src_benchmark
            WHERE CAST(
                strptime(CAST(trade_date AS VARCHAR), '%Y%m%d') AS DATE
            ) <= DATE '{cutoff}'
        )
        SELECT
            benchmark_code,
            trade_date,
            close,
            close / lag(close) OVER (ORDER BY trade_date) - 1.0
                AS benchmark_return
        FROM typed
        ORDER BY trade_date
        """,
        output,
    )


def _sample_join_audit(
    connection: duckdb.DuckDBPyConnection, config: dict[str, Any]
) -> pd.DataFrame:
    outputs = config["outputs"]
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW out_daily AS
        SELECT * FROM read_parquet('{outputs["daily_panel"]}');
        CREATE OR REPLACE TEMP VIEW out_execution AS
        SELECT * FROM read_parquet('{outputs["execution_status"]}');
        CREATE OR REPLACE TEMP VIEW out_industry AS
        SELECT * FROM read_parquet('{outputs["industry_history"]}');
        CREATE OR REPLACE TEMP VIEW out_name AS
        SELECT * FROM read_parquet('{outputs["name_history"]}');
        """
    )
    frames: list[pd.DataFrame] = []
    categories = [
        (
            "NORMAL",
            "base.has_daily_price "
            "AND NOT base.cannot_buy_at_open "
            "AND NOT base.cannot_sell_at_open",
            "base.ts_code",
        ),
        (
            "SUSPENDED_OR_NO_PRICE",
            "(NOT base.has_daily_price OR base.suspended_at_open)",
            "base.suspended_at_open DESC, base.has_suspend_record DESC, "
            "base.ts_code",
        ),
        (
            "NAME_CHANGED_STOCK",
            "base.has_daily_price AND base.name_history_record_count > 1 "
            "AND base.stock_name IS NOT NULL",
            "base.name_history_record_count DESC, base.ts_code",
        ),
        (
            "INDUSTRY_CHANGED_STOCK",
            "base.has_daily_price AND base.industry_history_record_count > 1 "
            "AND base.industry_code IS NOT NULL",
            "base.industry_history_record_count DESC, base.ts_code",
        ),
    ]
    for sample_date in config["project"]["sample_dates"]:
        for category, condition, order_by in categories:
            frame = connection.execute(
                f"""
                WITH name_counts AS (
                    SELECT
                        ts_code,
                        COUNT(DISTINCT stock_name)
                            AS name_history_record_count
                    FROM out_name
                    WHERE effective_from <= DATE '{sample_date}'
                    GROUP BY ts_code
                ),
                industry_counts AS (
                    SELECT
                        ts_code,
                        COUNT(DISTINCT industry_code)
                            AS industry_history_record_count
                    FROM out_industry
                    WHERE effective_from <= DATE '{sample_date}'
                    GROUP BY ts_code
                ),
                base AS (
                    SELECT
                        execution.*,
                        daily.close,
                        names.stock_name,
                        names.effective_from AS name_effective_from,
                        names.effective_to AS name_effective_to,
                        names.is_st_name_flag,
                        coalesce(name_counts.name_history_record_count, 0)
                            AS name_history_record_count,
                        industry.industry_code,
                        industry.industry_name,
                        industry.effective_from AS industry_effective_from,
                        industry.effective_to AS industry_effective_to,
                        coalesce(
                            industry_counts.industry_history_record_count, 0
                        ) AS industry_history_record_count
                    FROM out_execution AS execution
                    LEFT JOIN out_daily AS daily
                      USING (ts_code, trade_date)
                    LEFT JOIN out_name AS names
                      ON execution.ts_code = names.ts_code
                     AND names.effective_from <= execution.trade_date
                     AND (
                        names.effective_to IS NULL
                        OR execution.trade_date <= names.effective_to
                     )
                    LEFT JOIN out_industry AS industry
                      ON execution.ts_code = industry.ts_code
                     AND industry.effective_from <= execution.trade_date
                     AND (
                        industry.effective_to IS NULL
                        OR execution.trade_date <= industry.effective_to
                     )
                    LEFT JOIN name_counts USING (ts_code)
                    LEFT JOIN industry_counts USING (ts_code)
                    WHERE execution.trade_date = DATE '{sample_date}'
                )
                SELECT
                    DATE '{sample_date}' AS audit_date,
                    '{category}' AS sample_category,
                    base.*,
                    concat_ws(
                        '|',
                        CASE WHEN stock_name IS NOT NULL
                            THEN 'NAME_JOINED' ELSE 'NAME_MISSING' END,
                        CASE WHEN industry_code IS NOT NULL
                            THEN 'INDUSTRY_JOINED' ELSE 'INDUSTRY_MISSING' END
                    ) AS join_status
                FROM base
                WHERE {condition}
                ORDER BY {order_by}
                LIMIT 1
                """
            ).fetchdf()
            if frame.empty:
                frame = pd.DataFrame(
                    [
                        {
                            "audit_date": sample_date,
                            "sample_category": category,
                            "ts_code": pd.NA,
                            "join_status": "NO_ELIGIBLE_SAMPLE",
                        }
                    ]
                )
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def _row_counts(
    connection: duckdb.DuckDBPyConnection, config: dict[str, Any]
) -> dict[str, int]:
    result: dict[str, int] = {}
    for key in (
        "daily_panel",
        "execution_status",
        "industry_history",
        "name_history",
        "month_end_base_panel",
        "benchmark_daily",
    ):
        path = config["outputs"][key]
        result[key] = int(
            connection.execute(
                f"SELECT COUNT(*) FROM read_parquet('{path}')"
            ).fetchone()[0]
        )
    return result


def build_p1() -> None:
    started_at = datetime.now(UTC)
    os.chdir(PROJECT_ROOT)
    config = load_config()
    for relative_path in config["outputs"].values():
        absolute(relative_path).parent.mkdir(parents=True, exist_ok=True)

    _log("计算受保护输入的构建前 SHA-256")
    protected_files_before = _protected_files(config)
    before = _hash_snapshot(protected_files_before)

    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    _create_source_views(connection, config)
    _log("执行输入主键、复权覆盖和宽基 CSV 预检查")
    _preflight(connection, config)

    inventory = _input_inventory(connection, config, before)
    _write_csv_atomic(inventory, config["outputs"]["input_inventory"])
    if (inventory["status"] != "PASS").any():
        raise RuntimeError("输入库存存在 FAIL，停止构建")

    _prepare_calendar_and_stock(connection, config)
    _build_histories(connection, config)
    _build_daily_panel(connection, config)
    _build_execution_status(connection, config)
    _build_month_end_panel(connection, config)
    _build_benchmark(connection, config)

    _log("生成人工连接抽查样例")
    sample_frame = _sample_join_audit(connection, config)
    _write_csv_atomic(sample_frame, config["outputs"]["sample_join_audit"])

    _log("计算受保护输入的构建后 SHA-256")
    protected_files_after = _protected_files(config)
    after = _hash_snapshot(protected_files_after)
    all_paths = sorted(set(before) | set(after))
    comparison = pd.DataFrame(
        [
            {
                "path": path,
                "size_bytes_before": before.get(path, {}).get("size_bytes"),
                "size_bytes_after": after.get(path, {}).get("size_bytes"),
                "sha256_before": before.get(path, {}).get("sha256"),
                "sha256_after": after.get(path, {}).get("sha256"),
                "match": before.get(path) == after.get(path),
            }
            for path in all_paths
        ]
    )
    _write_csv_atomic(comparison, config["outputs"]["protected_input_hashes"])
    if not comparison["match"].all():
        changed = comparison.loc[~comparison["match"], "path"].tolist()
        raise RuntimeError(f"受保护输入在构建期间发生变化：{changed[:10]}")

    counts = _row_counts(connection, config)
    output_hashes = {
        key: _sha256(absolute(config["outputs"][key]))
        for key in (
            "daily_panel",
            "execution_status",
            "industry_history",
            "name_history",
            "month_end_base_panel",
            "benchmark_daily",
        )
    }
    completed_at = datetime.now(UTC)
    manifest = {
        "stage": "P1_STANDARDIZED_RESEARCH_PANEL",
        "builder_version": BUILDER_VERSION,
        "status": "BUILT_PENDING_AUDIT",
        "project_root_policy": "runtime_discovery_and_project_relative_config_only",
        "research_cutoff": config["project"]["research_cutoff"],
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "elapsed_seconds": round((completed_at - started_at).total_seconds(), 3),
        "runtime": {
            "python": platform.python_version(),
            "duckdb": duckdb.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "protected_input_files": len(comparison),
        "protected_inputs_all_match": bool(comparison["match"].all()),
        "output_row_counts": counts,
        "output_sha256": output_hashes,
        "scope_guards": {
            "formal_factors_computed": False,
            "portfolio_built": False,
            "backtest_run": False,
            "oos_run": False,
            "cross_code_adjusted_price_stitched": False,
        },
    }
    _write_json_atomic(manifest, config["outputs"]["p1_run_manifest"])
    connection.close()
    _log(
        "构建完成，等待 audit_p1.py 验收；"
        + ", ".join(f"{key}={value}" for key, value in counts.items())
    )


if __name__ == "__main__":
    build_p1()
