from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml

from .build import (
    _create_source_views,
    _hash_snapshot,
    _prepare_calendar_and_stock,
    _prepare_suspend_aggregate,
    _protected_files,
    _write_csv_atomic,
    _write_json_atomic,
)
from .config import PROJECT_ROOT, absolute, load_config, relative_posix


class Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []

    def add(
        self,
        check_id: str,
        category: str,
        description: str,
        passed: bool,
        observed: Any,
        expected: Any,
        notes: str = "",
        warning: bool = False,
        status_override: str | None = None,
    ) -> None:
        status = (
            status_override
            if passed and status_override
            else ("PASS" if passed else ("WARN" if warning else "FAIL"))
        )
        self.rows.append(
            {
                "check_id": check_id,
                "category": category,
                "description": description,
                "status": status,
                "observed": str(observed),
                "expected": str(expected),
                "notes": notes,
            }
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _scalar(connection: duckdb.DuckDBPyConnection, query: str) -> Any:
    return connection.execute(query).fetchone()[0]


def _columns(connection: duckdb.DuckDBPyConnection, view: str) -> set[str]:
    return {
        row[0]
        for row in connection.execute(f"DESCRIBE SELECT * FROM {view}").fetchall()
    }


def _markdown_report(
    frame: pd.DataFrame, overall_status: str, audited_at: str
) -> str:
    counts = frame["status"].value_counts().to_dict()
    lines = [
        "# P1 标准化研究面板验收报告",
        "",
        f"- 验收时间（UTC）：`{audited_at}`",
        f"- 总体状态：`{overall_status}`",
        f"- PASS：`{counts.get('PASS', 0)}`",
        "- PASS_WITH_DOCUMENTED_EXCLUSION："
        f"`{counts.get('PASS_WITH_DOCUMENTED_EXCLUSION', 0)}`",
        f"- WARN：`{counts.get('WARN', 0)}`",
        f"- FAIL：`{counts.get('FAIL', 0)}`",
        "",
        "## 验收明细",
        "",
        "| ID | 类别 | 状态 | 检查 | 观测值 | 期望值 | 说明 |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in frame.itertuples(index=False):
        values = [
            row.check_id,
            row.category,
            row.status,
            row.description,
            row.observed,
            row.expected,
            row.notes,
        ]
        escaped = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        lines.append("| " + " | ".join(escaped) + " |")
    lines.extend(
        [
            "",
            "## 非阻断披露",
            "",
            "- 交易日历从 2014-12-01 开始；更早上市股票的"
            " `listing_age_trading_days` 是显式标记的可验证下界。",
            "- 供应商历史 PB 修订政策、宽基原始成交字段单位仍待人工确认。",
            "- 历史卖出印花税率及官方来源已冻结到"
            " `configs/trading_costs.yaml`。",
            "- 代码变更主体仅建立映射；P1 没有跨代码拼接复权价格。"
            " P2 使用前必须通过边界测试。",
            "- 本阶段没有计算正式因子、构建组合、回测或运行 OOS。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_text_atomic(text: str, relative_path: str) -> None:
    output = absolute(relative_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, output)


def run_audit() -> str:
    os.chdir(PROJECT_ROOT)
    config = load_config()
    outputs = config["outputs"]
    cutoff = config["project"]["research_cutoff"]
    audit = Audit()

    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    _create_source_views(connection, config)
    _prepare_calendar_and_stock(connection, config)

    required_files = [
        "configs/paths.yaml",
        "data/data_dictionary.csv",
        "reports/data_audit/field_semantics.md",
        config["inputs"]["security_code_history"],
        config["policies"]["trading_costs"],
        config["policies"]["stamp_duty_sources"],
        outputs["input_inventory"],
        outputs["protected_input_hashes"],
        outputs["sample_join_audit"],
        outputs["p1_run_manifest"],
        outputs["daily_panel"],
        outputs["execution_status"],
        outputs["industry_history"],
        outputs["name_history"],
        outputs["month_end_base_panel"],
        outputs["benchmark_daily"],
    ]
    missing_files = [path for path in required_files if not absolute(path).is_file()]
    audit.add(
        "P1-001",
        "路径与输入",
        "P1 必需文件均存在",
        not missing_files,
        len(missing_files),
        0,
        "；".join(missing_files),
    )

    all_config_paths = list(config["inputs"].values()) + list(
        config["outputs"].values()
    ) + list(config["policies"].values()) + list(config["protected_inputs"])
    absolute_config_paths = [
        value for value in all_config_paths if Path(value).is_absolute()
    ]
    audit.add(
        "P1-002",
        "路径与输入",
        "路径配置全部为项目相对路径",
        not absolute_config_paths,
        len(absolute_config_paths),
        0,
        "；".join(absolute_config_paths),
    )

    source_literals: list[str] = []
    drive_pattern = re.compile(r"(?i)(?<![a-z])[a-z]:[\\/]")
    for directory in ("src", "scripts", "tests", "configs"):
        root = absolute(directory)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".py", ".yaml", ".yml", ".json"}:
                continue
            text = path.read_text(encoding="utf-8")
            if drive_pattern.search(text):
                source_literals.append(relative_posix(path))
    audit.add(
        "P1-003",
        "路径与输入",
        "P1 代码与配置不存在硬编码 Windows 绝对路径",
        not source_literals,
        len(source_literals),
        0,
        "；".join(source_literals),
    )

    inventory = pd.read_csv(absolute(outputs["input_inventory"]))
    inventory_failures = int((inventory["status"] != "PASS").sum())
    audit.add(
        "P1-004",
        "路径与输入",
        "输入库存全部通过",
        inventory_failures == 0 and len(inventory) == 13,
        f"rows={len(inventory)}, failures={inventory_failures}",
        "rows=13, failures=0",
    )

    dictionary = pd.read_csv(absolute("data/data_dictionary.csv"))
    allowed_status = {
        "CONFIRMED",
        "INFERRED_FROM_DATA",
        "NEEDS_MANUAL_CONFIRMATION",
    }
    invalid_dictionary_status = sorted(
        set(dictionary["status"].dropna()) - allowed_status
    )
    audit.add(
        "P1-005",
        "字段语义",
        "数据字典状态值合法",
        not invalid_dictionary_status,
        invalid_dictionary_status,
        "only allowed statuses",
    )
    required_semantic_fields = {
        "vol",
        "amount",
        "adj_factor",
        "pb",
        "total_share",
        "float_share",
        "free_share",
        "total_mv",
        "circ_mv",
        "up_limit",
        "down_limit",
        "suspend_timing",
    }
    observed_semantic_fields = set(dictionary["source_field"].astype(str))
    missing_semantics = sorted(required_semantic_fields - observed_semantic_fields)
    audit.add(
        "P1-006",
        "字段语义",
        "人工确认的关键字段已进入数据字典",
        not missing_semantics,
        missing_semantics,
        "none missing",
    )

    trading_costs = yaml.safe_load(
        absolute(config["policies"]["trading_costs"]).read_text(encoding="utf-8")
    )
    expected_stamp_periods = [
        {
            "effective_from": "2016-01-01",
            "effective_to": "2023-08-27",
            "sell_rate": 0.001,
        },
        {
            "effective_from": "2023-08-28",
            "effective_to": "2025-12-31",
            "sell_rate": 0.0005,
        },
    ]
    observed_stamp_periods = [
        {
            "effective_from": str(period["effective_from"]),
            "effective_to": str(period["effective_to"]),
            "sell_rate": float(period["sell_rate"]),
        }
        for period in trading_costs["stamp_duty"]["periods"]
    ]
    source_ids = {
        source["source_id"] for source in trading_costs["source_registry"]
    }
    stamp_policy_valid = (
        trading_costs["policy_status"] == "CONFIRMED"
        and float(trading_costs["stamp_duty"]["buy_rate"]) == 0.0
        and trading_costs["stamp_duty"]["taxable_side"] == "sell"
        and observed_stamp_periods == expected_stamp_periods
        and source_ids
        == {"MOF_2008_SINGLE_SIDE", "STAMP_TAX_LAW_2022", "MOF_STA_2023_39"}
    )
    audit.add(
        "P1-008",
        "政策配置",
        "历史卖出印花税区间、税率和官方来源已冻结",
        stamp_policy_valid,
        observed_stamp_periods,
        expected_stamp_periods,
        "买入方税率必须为0；P1不运行回测",
    )

    semantics_text = absolute(
        "reports/data_audit/field_semantics.md"
    ).read_text(encoding="utf-8")
    required_pb_disclosure = (
        "使用供应商历史PB构造1/PB代理，未自行重建严格 "
        "point-in-time book equity，供应商历史修订政策未完全核验。"
    )
    normalized_semantics = " ".join(semantics_text.split())
    audit.add(
        "P1-009",
        "字段语义",
        "PB代理限制披露已按人工确认写入",
        required_pb_disclosure in normalized_semantics,
        required_pb_disclosure in normalized_semantics,
        True,
    )

    hash_frame = pd.read_csv(absolute(outputs["protected_input_hashes"]))
    recorded_mismatches = int((~hash_frame["match"].astype(bool)).sum())
    current_snapshot = _hash_snapshot(_protected_files(config))
    recorded_paths = set(hash_frame["path"])
    current_paths = set(current_snapshot)
    current_mismatches = 0
    for row in hash_frame.itertuples(index=False):
        current = current_snapshot.get(row.path)
        if (
            current is None
            or current["size_bytes"] != row.size_bytes_after
            or current["sha256"] != row.sha256_after
        ):
            current_mismatches += 1
    path_set_match = recorded_paths == current_paths
    audit.add(
        "P1-007",
        "输入保护",
        "raw/static/_parts/benchmark 构建前后及当前 SHA-256 一致",
        recorded_mismatches == 0 and current_mismatches == 0 and path_set_match,
        (
            f"recorded={recorded_mismatches}, current={current_mismatches}, "
            f"path_set_match={path_set_match}, files={len(current_paths)}"
        ),
        "all match",
    )

    connection.execute(
        f"""
        CREATE OR REPLACE VIEW out_daily AS
        SELECT * FROM read_parquet('{outputs["daily_panel"]}');
        CREATE OR REPLACE VIEW out_execution AS
        SELECT * FROM read_parquet('{outputs["execution_status"]}');
        CREATE OR REPLACE VIEW out_industry AS
        SELECT * FROM read_parquet('{outputs["industry_history"]}');
        CREATE OR REPLACE VIEW out_name AS
        SELECT * FROM read_parquet('{outputs["name_history"]}');
        CREATE OR REPLACE VIEW out_month AS
        SELECT * FROM read_parquet('{outputs["month_end_base_panel"]}');
        CREATE OR REPLACE VIEW out_benchmark AS
        SELECT * FROM read_parquet('{outputs["benchmark_daily"]}');
        """
    )

    source_daily_rows = int(
        _scalar(
            connection,
            f"""
            SELECT COUNT(*) FROM src_daily
            WHERE CAST(strptime(trade_date, '%Y%m%d') AS DATE)
                  <= DATE '{cutoff}'
            """,
        )
    )
    output_daily_rows = int(_scalar(connection, "SELECT COUNT(*) FROM out_daily"))
    audit.add(
        "P1-010",
        "日频面板",
        "日频面板行数与 daily 锚表完全一致",
        source_daily_rows == output_daily_rows,
        output_daily_rows,
        source_daily_rows,
    )
    daily_duplicates = int(
        _scalar(
            connection,
            """
            SELECT COUNT(*) FROM (
                SELECT ts_code, trade_date, COUNT(*) AS n
                FROM out_daily GROUP BY ALL HAVING n > 1
            )
            """,
        )
    )
    audit.add(
        "P1-011",
        "日频面板",
        "ts_code + trade_date 主键唯一",
        daily_duplicates == 0,
        daily_duplicates,
        0,
    )
    created_daily = int(
        _scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM out_daily AS output
            ANTI JOIN (
                SELECT
                    ts_code,
                    CAST(strptime(trade_date, '%Y%m%d') AS DATE) AS trade_date
                FROM src_daily
            ) AS source USING (ts_code, trade_date)
            """,
        )
    )
    missing_daily = int(
        _scalar(
            connection,
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT
                    ts_code,
                    CAST(strptime(trade_date, '%Y%m%d') AS DATE) AS trade_date
                FROM src_daily
                WHERE CAST(strptime(trade_date, '%Y%m%d') AS DATE)
                      <= DATE '{cutoff}'
            ) AS source
            ANTI JOIN out_daily AS output USING (ts_code, trade_date)
            """,
        )
    )
    audit.add(
        "P1-012",
        "日频面板",
        "不创造也不丢失行情键",
        created_daily == 0 and missing_daily == 0,
        f"created={created_daily}, missing={missing_daily}",
        "created=0, missing=0",
    )
    invalid_adjusted = int(
        _scalar(
            connection,
            """
            SELECT COUNT(*) FROM out_daily
            WHERE adjusted_close IS NULL OR adjusted_close <= 0
            """,
        )
    )
    audit.add(
        "P1-013",
        "日频面板",
        "adjusted_close 全部为正",
        invalid_adjusted == 0,
        invalid_adjusted,
        0,
    )
    adjusted_formula_mismatch = int(
        _scalar(
            connection,
            """
            SELECT COUNT(*) FROM out_daily
            WHERE abs(close_hfq - close * adj_factor)
                    > 1e-10 * greatest(1.0, abs(close_hfq))
               OR abs(adjusted_close - close_hfq)
                    > 1e-12 * greatest(1.0, abs(close_hfq))
            """,
        )
    )
    audit.add(
        "P1-014",
        "日频面板",
        "后复权公式为 close × adj_factor",
        adjusted_formula_mismatch == 0,
        adjusted_formula_mismatch,
        0,
    )
    return_mismatch = int(
        _scalar(
            connection,
            """
            WITH expected AS (
                SELECT
                    ts_code,
                    trade_date,
                    daily_return,
                    CASE
                        WHEN security_code_interval_valid
                         AND coalesce(
                            lag(security_code_interval_valid) OVER (
                                PARTITION BY ts_code ORDER BY trade_date
                            ),
                            false
                         )
                        THEN adjusted_close
                            / lag(adjusted_close) OVER (
                                PARTITION BY ts_code ORDER BY trade_date
                            ) - 1.0
                        ELSE CAST(NULL AS DOUBLE)
                    END AS expected_return
                FROM out_daily
            )
            SELECT COUNT(*) FROM expected
            WHERE (daily_return IS NULL) <> (expected_return IS NULL)
               OR (
                    daily_return IS NOT NULL
                    AND abs(daily_return - expected_return)
                        > 1e-12 * greatest(1.0, abs(expected_return))
               )
            """,
        )
    )
    audit.add(
        "P1-015",
        "日频面板",
        "daily_return仅在同一有效真实代码区间内由close_hfq计算",
        return_mismatch == 0,
        return_mismatch,
        0,
    )
    daily_future = int(
        _scalar(
            connection,
            f"SELECT COUNT(*) FROM out_daily WHERE trade_date > DATE '{cutoff}'",
        )
    )
    audit.add(
        "P1-016",
        "日频面板",
        "日频面板无截止日后记录",
        daily_future == 0,
        daily_future,
        0,
    )
    invalid_listing_semantics = int(
        _scalar(
            connection,
            """
            SELECT COUNT(*) FROM out_daily
            WHERE (
                    is_within_listing_window
                    AND (
                        list_date IS NULL
                        OR trade_date < list_date
                        OR listing_age_trading_days IS NULL
                        OR listing_age_trading_days < 1
                    )
                  )
               OR (
                    NOT is_within_listing_window
                    AND listing_age_trading_days IS NOT NULL
                  )
            """,
        )
    )
    audit.add(
        "P1-017",
        "日频面板",
        "上市有效标记与交易日年龄遵守冻结语义",
        invalid_listing_semantics == 0,
        invalid_listing_semantics,
        0,
    )
    listing_age_mismatch = int(
        _scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM out_daily AS daily
            JOIN open_calendar AS calendar
              ON daily.trade_date = calendar.cal_date
            JOIN stock_meta AS stock USING (ts_code)
            WHERE daily.is_within_listing_window
              AND daily.listing_age_trading_days
                  <> calendar.open_day_number
                     - stock.listing_start_open_day + 1
            """,
        )
    )
    audit.add(
        "P1-018",
        "日频面板",
        "listing_age_trading_days 按开市日累计且首日为1",
        listing_age_mismatch == 0,
        listing_age_mismatch,
        0,
    )
    lower_bound_rows = int(
        _scalar(
            connection,
            """
            SELECT COUNT(*) FROM out_daily
            WHERE listing_age_is_lower_bound
            """,
        )
    )
    audit.add(
        "P1-019",
        "日频面板",
        "日历起点前上市股票已显式标记年龄下界",
        lower_bound_rows > 0,
        lower_bound_rows,
        "> 0 and explicitly flagged",
        "这是输入时间范围限制，不虚构更早交易日",
    )
    missing_listing_reference_codes = int(
        _scalar(
            connection,
            """
            SELECT COUNT(DISTINCT ts_code) FROM out_daily
            WHERE NOT has_listing_reference
            """,
        )
    )
    manually_resolved_codes = int(
        _scalar(
            connection,
            """
            SELECT COUNT(DISTINCT ts_code) FROM out_daily
            WHERE NOT has_stock_basic_record
              AND has_manual_code_history
              AND has_listing_reference
            """,
        )
    )
    audit.add(
        "P1-019A",
        "日频面板",
        "缺失stock_basic的旧代码已由人工代码历史补足上市参考",
        missing_listing_reference_codes == 0 and manually_resolved_codes == 3,
        (
            f"missing_listing_reference={missing_listing_reference_codes}, "
            f"manual_resolved_codes={manually_resolved_codes}"
        ),
        "missing_listing_reference=0, manual_resolved_codes=3",
        "000022.SZ、000043.SZ、300114.SZ不视为普通退市",
    )
    invalid_listing_rows, invalid_non_bj_rows, invalid_sh_sz_rows = (
        connection.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE invalid_listing_interval),
                COUNT(*) FILTER (
                    WHERE invalid_listing_interval AND exchange <> 'BJ'
                ),
                COUNT(*) FILTER (
                    WHERE invalid_listing_interval AND is_sh_sz
                )
            FROM out_daily
            """
        ).fetchone()
    )
    documented_bj_exclusion = (
        invalid_listing_rows == 75958
        and invalid_non_bj_rows == 0
        and invalid_sh_sz_rows == 0
    )
    audit.add(
        "P1-019B",
        "日频面板",
        "北交所上市区间异常已保留并从沪深研究范围排除",
        documented_bj_exclusion,
        (
            f"invalid_rows={invalid_listing_rows}, "
            f"non_bj={invalid_non_bj_rows}, is_sh_sz={invalid_sh_sz_rows}"
        ),
        "invalid_rows=75958, non_bj=0, is_sh_sz=0",
        "raw与processed行情均保留；invalid_listing_interval=True",
        status_override="PASS_WITH_DOCUMENTED_EXCLUSION",
    )

    manual_mapping_mismatch = int(
        _scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM out_daily AS daily
            JOIN code_identity_map AS mapping USING (ts_code)
            WHERE daily.canonical_ts_code <> mapping.canonical_ts_code
               OR daily.list_date <> mapping.original_list_date
               OR NOT daily.has_manual_code_history
               OR NOT daily.has_listing_reference
            """,
        )
    )
    canonical_duplicate_dates = int(
        _scalar(
            connection,
            """
            SELECT COUNT(*) FROM (
                SELECT canonical_ts_code, trade_date, COUNT(*) AS n
                FROM out_daily
                WHERE has_manual_code_history
                GROUP BY ALL
                HAVING COUNT(DISTINCT ts_code) > 1
            )
            """,
        )
    )
    invalid_code_interval_rows = int(
        _scalar(
            connection,
            """
            SELECT COUNT(*) FROM out_daily
            WHERE invalid_security_code_interval
            """,
        )
    )
    audit.add(
        "P1-019C",
        "日频面板",
        "真实代码与连续主体代码映射已建立但未跨代码拼接价格",
        manual_mapping_mismatch == 0
        and canonical_duplicate_dates > 0
        and invalid_code_interval_rows > 0,
        (
            f"mapping_mismatch={manual_mapping_mismatch}, "
            f"canonical_duplicate_dates={canonical_duplicate_dates}, "
            f"invalid_code_interval_rows={invalid_code_interval_rows}"
        ),
        "mapping_mismatch=0, duplicates and interval conflicts documented",
        "canonical重复日期与代码边界留待P2边界测试；不得无条件拼接复权价格",
    )

    execution_duplicates = int(
        _scalar(
            connection,
            """
            SELECT COUNT(*) FROM (
                SELECT ts_code, trade_date, COUNT(*) AS n
                FROM out_execution GROUP BY ALL HAVING n > 1
            )
            """,
        )
    )
    audit.add(
        "P1-020",
        "成交状态",
        "成交状态主键唯一",
        execution_duplicates == 0,
        execution_duplicates,
        0,
    )
    execution_key_difference = int(
        _scalar(
            connection,
            f"""
            WITH source_keys AS (
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
                SELECT
                    ts_code,
                    CAST(strptime(trade_date, '%Y%m%d') AS DATE) AS trade_date
                FROM src_suspend
                WHERE CAST(strptime(trade_date, '%Y%m%d') AS DATE)
                      <= DATE '{cutoff}'
            ),
            differences AS (
                (SELECT ts_code, trade_date FROM source_keys
                 EXCEPT
                 SELECT ts_code, trade_date FROM out_execution)
                UNION ALL
                (SELECT ts_code, trade_date FROM out_execution
                 EXCEPT
                 SELECT ts_code, trade_date FROM source_keys)
            )
            SELECT COUNT(*) FROM differences
            """,
        )
    )
    audit.add(
        "P1-021",
        "成交状态",
        "成交状态恰好覆盖 daily/stk_limit/suspend_d 键并不创造事件",
        execution_key_difference == 0,
        execution_key_difference,
        0,
    )
    no_price_not_blocked = int(
        _scalar(
            connection,
            """
            SELECT COUNT(*) FROM out_execution
            WHERE NOT has_daily_price
              AND (NOT cannot_buy_at_open OR NOT cannot_sell_at_open)
            """,
        )
    )
    audit.add(
        "P1-022",
        "成交状态",
        "无日行情时买卖两侧均阻断",
        no_price_not_blocked == 0,
        no_price_not_blocked,
        0,
    )
    suspended_not_blocked = int(
        _scalar(
            connection,
            """
            SELECT COUNT(*) FROM out_execution
            WHERE suspended_at_open
              AND (NOT cannot_buy_at_open OR NOT cannot_sell_at_open)
            """,
        )
    )
    audit.add(
        "P1-023",
        "成交状态",
        "开盘停牌时买卖两侧均阻断",
        suspended_not_blocked == 0,
        suspended_not_blocked,
        0,
    )
    _prepare_suspend_aggregate(connection, config)
    suspend_mismatch = int(
        _scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM suspend_aggregate AS source
            JOIN out_execution AS output USING (ts_code, trade_date)
            WHERE source.has_suspend_s <> output.has_suspend_s
               OR source.has_resume_r <> output.has_resume_r
               OR source.suspended_at_open <> output.suspended_at_open
               OR source.suspend_intervals <> output.suspend_intervals
            """,
        )
    )
    audit.add(
        "P1-024",
        "成交状态",
        "同股票日期停复牌记录完整聚合且09:30语义一致",
        suspend_mismatch == 0,
        suspend_mismatch,
        0,
    )
    resume_only_wrong = int(
        _scalar(
            connection,
            """
            SELECT COUNT(*) FROM out_execution
            WHERE has_resume_r AND NOT has_suspend_s AND suspended_at_open
            """,
        )
    )
    audit.add(
        "P1-025",
        "成交状态",
        "只有R记录不判定为开盘停牌",
        resume_only_wrong == 0,
        resume_only_wrong,
        0,
    )
    limit_mismatch = int(
        _scalar(
            connection,
            """
            SELECT COUNT(*) FROM out_execution
            WHERE open_at_up_limit <> coalesce(
                has_daily_price AND up_limit IS NOT NULL
                AND (
                    round(open_price, 2) >= round(up_limit, 2)
                    OR abs(open_price - up_limit) <= 0.005
                ), false
            )
               OR open_at_down_limit <> coalesce(
                has_daily_price AND down_limit IS NOT NULL
                AND (
                    round(open_price, 2) <= round(down_limit, 2)
                    OR abs(open_price - down_limit) <= 0.005
                ), false
            )
            """,
        )
    )
    audit.add(
        "P1-026",
        "成交状态",
        "涨跌停比较使用2位小数或0.005绝对误差",
        limit_mismatch == 0,
        limit_mismatch,
        0,
    )

    for prefix, view in (("industry", "out_industry"), ("name", "out_name")):
        invalid_interval = int(
            _scalar(
                connection,
                f"""
                SELECT COUNT(*) FROM {view}
                WHERE effective_from > DATE '{cutoff}'
                   OR effective_to > DATE '{cutoff}'
                   OR effective_from > effective_to
                """,
            )
        )
        audit.add(
            f"P1-03{0 if prefix == 'industry' else 3}",
            "历史区间",
            f"{prefix} 历史区间有效且不含截止日后版本",
            invalid_interval == 0,
            invalid_interval,
            0,
        )
        overlaps = int(
            _scalar(
                connection,
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
                """,
            )
        )
        audit.add(
            f"P1-03{1 if prefix == 'industry' else 4}",
            "历史区间",
            f"{prefix} 一个股票日期最多匹配一个有效区间",
            overlaps == 0,
            overlaps,
            0,
        )
    st_flag_mismatch = int(
        _scalar(
            connection,
            """
            SELECT COUNT(*) FROM out_name
            WHERE is_st_name_flag
                  <> (upper(coalesce(stock_name, '')) LIKE '%ST%')
            """,
        )
    )
    audit.add(
        "P1-035",
        "历史区间",
        "历史ST名称标记与当时有效名称一致",
        st_flag_mismatch == 0,
        st_flag_mismatch,
        0,
    )
    equity_views = (
        "out_daily",
        "out_execution",
        "out_industry",
        "out_name",
        "out_month",
    )
    missing_identity_columns = [
        view
        for view in equity_views
        if not {"ts_code", "canonical_ts_code"} <= _columns(connection, view)
    ]
    null_canonical_rows = sum(
        int(
            _scalar(
                connection,
                f"SELECT COUNT(*) FROM {view} "
                "WHERE canonical_ts_code IS NULL",
            )
        )
        for view in equity_views
        if view not in missing_identity_columns
    )
    audit.add(
        "P1-036",
        "证券代码",
        "全部股票类processed表同时保留ts_code与canonical_ts_code",
        not missing_identity_columns and null_canonical_rows == 0,
        (
            f"missing_columns={missing_identity_columns}, "
            f"null_canonical_rows={null_canonical_rows}"
        ),
        "missing_columns=[], null_canonical_rows=0",
    )

    month_duplicates = int(
        _scalar(
            connection,
            """
            SELECT COUNT(*) FROM (
                SELECT ts_code, signal_date, COUNT(*) AS n
                FROM out_month GROUP BY ALL HAVING n > 1
            )
            """,
        )
    )
    audit.add(
        "P1-040",
        "月末面板",
        "月末每只股票每个信号日只有一行",
        month_duplicates == 0,
        month_duplicates,
        0,
    )
    source_month_rows = int(
        _scalar(
            connection,
            f"""
            SELECT COUNT(*) FROM src_daily_basic
            WHERE CAST(strptime(trade_date, '%Y%m%d') AS DATE)
                  <= DATE '{cutoff}'
            """,
        )
    )
    output_month_rows = int(_scalar(connection, "SELECT COUNT(*) FROM out_month"))
    audit.add(
        "P1-041",
        "月末面板",
        "月末基础表完整锚定 daily_basic_month_end",
        source_month_rows == output_month_rows,
        output_month_rows,
        source_month_rows,
    )
    month_missing_daily = int(
        _scalar(
            connection,
            """
            SELECT COUNT(*)
            FROM out_month AS month
            ANTI JOIN out_daily AS daily
              ON month.ts_code = daily.ts_code
             AND month.signal_date = daily.trade_date
            """,
        )
    )
    audit.add(
        "P1-042",
        "月末面板",
        "月末基础行均有真实日行情锚点",
        month_missing_daily == 0,
        month_missing_daily,
        0,
    )
    month_columns = _columns(connection, "out_month")
    prohibited_factor_columns = {
        "bm_proxy",
        "momentum_12_1",
        "lowvol_60",
        "rank_ic",
        "factor_score",
        "next_month_return",
    }
    present_factor_columns = sorted(month_columns & prohibited_factor_columns)
    required_override_columns = {
        "listing_age_trading_days",
        "listing_age_is_lower_bound",
    }
    audit.add(
        "P1-043",
        "范围闸门",
        "月末面板未计算正式因子或前瞻收益",
        not present_factor_columns,
        present_factor_columns,
        "none",
    )
    audit.add(
        "P1-044",
        "月末面板",
        "上市年龄采用用户冻结的交易日字段名",
        required_override_columns <= month_columns
        and "listing_age_days" not in month_columns,
        sorted(required_override_columns & month_columns),
        sorted(required_override_columns),
    )
    unit_mismatch = int(
        _scalar(
            connection,
            """
            SELECT COUNT(*) FROM out_month
            WHERE abs(total_share_shares - total_share * 10000.0)
                    > 1e-8 * greatest(1.0, abs(total_share_shares))
               OR abs(float_share_shares - float_share * 10000.0)
                    > 1e-8 * greatest(1.0, abs(float_share_shares))
               OR (
                    free_share IS NOT NULL
                    AND abs(free_share_shares - free_share * 10000.0)
                        > 1e-8 * greatest(1.0, abs(free_share_shares))
               )
               OR abs(total_mv_cny - total_mv * 10000.0)
                    > 1e-8 * greatest(1.0, abs(total_mv_cny))
               OR abs(circ_mv_cny - circ_mv * 10000.0)
                    > 1e-8 * greatest(1.0, abs(circ_mv_cny))
            """,
        )
    )
    audit.add(
        "P1-045",
        "月末面板",
        "股本与市值标准单位换算正确",
        unit_mismatch == 0,
        unit_mismatch,
        0,
    )
    future_join = int(
        _scalar(
            connection,
            f"""
            SELECT COUNT(*) FROM out_month
            WHERE signal_date > DATE '{cutoff}'
               OR (
                    industry_code IS NOT NULL
                    AND (
                        industry_effective_from > signal_date
                        OR (
                            industry_effective_to IS NOT NULL
                            AND signal_date > industry_effective_to
                        )
                    )
               )
               OR (
                    stock_name IS NOT NULL
                    AND (
                        name_effective_from > signal_date
                        OR (
                            name_effective_to IS NOT NULL
                            AND signal_date > name_effective_to
                        )
                    )
               )
            """,
        )
    )
    audit.add(
        "P1-046",
        "月末面板",
        "历史名称与行业连接无未来信息",
        future_join == 0,
        future_join,
        0,
    )
    history_flag_mismatch = int(
        _scalar(
            connection,
            f"""
            SELECT COUNT(*) FROM out_month
            WHERE has_enough_252d_history
                  <> (available_history_observations
                      >= {int(config["project"]["history_threshold_observations"])})
            """,
        )
    )
    audit.add(
        "P1-047",
        "月末面板",
        "252日历史标记由可得行情观测数生成",
        history_flag_mismatch == 0,
        history_flag_mismatch,
        0,
    )
    invalid_liquidity = int(
        _scalar(
            connection,
            f"""
            SELECT COUNT(*) FROM out_month
            WHERE (
                    available_history_observations
                        < {int(config["project"]["liquidity_window_observations"])}
                    AND liquidity_20d IS NOT NULL
                  )
               OR (liquidity_20d IS NOT NULL AND liquidity_20d < 0)
            """,
        )
    )
    audit.add(
        "P1-048",
        "月末面板",
        "20日流动性仅在窗口充足时生成且非负",
        invalid_liquidity == 0,
        invalid_liquidity,
        0,
    )

    benchmark_rows = int(_scalar(connection, "SELECT COUNT(*) FROM out_benchmark"))
    source_benchmark_rows = int(
        _scalar(
            connection,
            f"""
            SELECT COUNT(*) FROM src_benchmark
            WHERE CAST(
                strptime(CAST(trade_date AS VARCHAR), '%Y%m%d') AS DATE
            ) <= DATE '{cutoff}'
            """,
        )
    )
    benchmark_duplicates = int(
        _scalar(
            connection,
            """
            SELECT COUNT(*) FROM (
                SELECT trade_date, COUNT(*) AS n
                FROM out_benchmark GROUP BY trade_date HAVING n > 1
            )
            """,
        )
    )
    benchmark_wrong_code = int(
        _scalar(
            connection,
            """
            SELECT COUNT(*) FROM out_benchmark
            WHERE benchmark_code <> '000985.CSI'
            """,
        )
    )
    audit.add(
        "P1-050",
        "宽基",
        "CSV实际代码、日期唯一性与行数通过",
        benchmark_rows == source_benchmark_rows
        and benchmark_duplicates == 0
        and benchmark_wrong_code == 0,
        (
            f"rows={benchmark_rows}, duplicate_dates={benchmark_duplicates}, "
            f"wrong_code={benchmark_wrong_code}"
        ),
        f"rows={source_benchmark_rows}, duplicate_dates=0, wrong_code=0",
    )
    benchmark_return_mismatch = int(
        _scalar(
            connection,
            """
            WITH expected AS (
                SELECT
                    benchmark_return,
                    close / lag(close) OVER (ORDER BY trade_date) - 1.0
                        AS expected_return
                FROM out_benchmark
            )
            SELECT COUNT(*) FROM expected
            WHERE (benchmark_return IS NULL) <> (expected_return IS NULL)
               OR (
                    benchmark_return IS NOT NULL
                    AND abs(benchmark_return - expected_return)
                        > 1e-12 * greatest(1.0, abs(expected_return))
               )
            """,
        )
    )
    audit.add(
        "P1-051",
        "宽基",
        "宽基收益由收盘点位计算",
        benchmark_return_mismatch == 0,
        benchmark_return_mismatch,
        0,
    )
    benchmark_columns = _columns(connection, "out_benchmark")
    expected_benchmark_columns = {
        "benchmark_code",
        "trade_date",
        "close",
        "benchmark_return",
    }
    audit.add(
        "P1-052",
        "宽基",
        "宽基P1输出严格只含代码、日期、收盘点位和收益",
        benchmark_columns == expected_benchmark_columns,
        sorted(benchmark_columns),
        sorted(expected_benchmark_columns),
        "原始vol/amount保留在CSV且单位继续待确认",
    )

    samples = pd.read_csv(absolute(outputs["sample_join_audit"]))
    expected_dates = set(config["project"]["sample_dates"])
    expected_categories = {
        "NORMAL",
        "SUSPENDED_OR_NO_PRICE",
        "NAME_CHANGED_STOCK",
        "INDUSTRY_CHANGED_STOCK",
    }
    observed_pairs = {
        (str(row.audit_date)[:10], row.sample_category)
        for row in samples.itertuples(index=False)
    }
    expected_pairs = {
        (sample_date, category)
        for sample_date in expected_dates
        for category in expected_categories
    }
    no_sample = int((samples["join_status"] == "NO_ELIGIBLE_SAMPLE").sum())
    audit.add(
        "P1-060",
        "人工抽查",
        "四个指定日期均含四类可复核连接样例",
        observed_pairs == expected_pairs and no_sample == 0 and len(samples) == 16,
        f"rows={len(samples)}, missing_sample={no_sample}",
        "rows=16, missing_sample=0",
    )

    processed_files = {
        relative_posix(path)
        for path in absolute("data/processed").glob("*.parquet")
        if path.is_file()
    }
    expected_processed = {
        outputs[key]
        for key in (
            "daily_panel",
            "execution_status",
            "industry_history",
            "name_history",
            "month_end_base_panel",
            "benchmark_daily",
        )
    }
    unexpected_processed = sorted(processed_files - expected_processed)
    audit.add(
        "P1-070",
        "范围闸门",
        "processed目录仅包含六个P1规范表",
        processed_files == expected_processed,
        f"unexpected={unexpected_processed}, total={len(processed_files)}",
        "unexpected=[], total=6",
    )

    manifest_path = absolute(outputs["p1_run_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scope_guards = manifest.get("scope_guards", {})
    scope_violation = [
        key for key, value in scope_guards.items() if bool(value)
    ]
    audit.add(
        "P1-071",
        "范围闸门",
        "未计算正式因子、组合、回测、OOS或跨代码连续复权价格",
        not scope_violation,
        scope_violation,
        "none",
    )

    frame = audit.frame()
    fail_count = int((frame["status"] == "FAIL").sum())
    warn_count = int((frame["status"] == "WARN").sum())
    documented_exclusion_count = int(
        (frame["status"] == "PASS_WITH_DOCUMENTED_EXCLUSION").sum()
    )
    overall_status = (
        "FAIL"
        if fail_count
        else (
            "PASS_WITH_WARNINGS"
            if warn_count
            else (
                "PASS_WITH_DOCUMENTED_EXCLUSION"
                if documented_exclusion_count
                else "PASS"
            )
        )
    )
    audited_at = datetime.now(UTC).isoformat()
    _write_csv_atomic(frame, outputs["p1_audit_summary"])
    report = _markdown_report(frame, overall_status, audited_at)
    _write_text_atomic(report, outputs["p1_audit_report"])

    manifest["status"] = (
        "P1_AUDIT_FAILED"
        if overall_status == "FAIL"
        else (
            "P1_ACCEPTED_WITH_DOCUMENTED_EXCLUSION"
            if overall_status == "PASS_WITH_DOCUMENTED_EXCLUSION"
            else "P1_ACCEPTED"
        )
    )
    manifest["audit"] = {
        "audited_at_utc": audited_at,
        "overall_status": overall_status,
        "pass_count": int((frame["status"] == "PASS").sum()),
        "documented_exclusion_count": documented_exclusion_count,
        "warn_count": warn_count,
        "fail_count": fail_count,
    }
    _write_json_atomic(manifest, outputs["p1_run_manifest"])
    connection.close()

    print(
        f"[P1 AUDIT] {overall_status}: "
        f"PASS={manifest['audit']['pass_count']}, "
        f"DOCUMENTED_EXCLUSION={documented_exclusion_count}, "
        f"WARN={warn_count}, FAIL={fail_count}",
        flush=True,
    )
    if fail_count:
        failed_ids = frame.loc[frame["status"] == "FAIL", "check_id"].tolist()
        raise RuntimeError(f"P1 验收失败：{failed_ids}")
    return overall_status


if __name__ == "__main__":
    run_audit()
