from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .config import CONFIG_PATH, PROJECT_ROOT, absolute, load_config, sql_path
from .research import FACTOR_COLUMNS


PB_DISCLOSURE = (
    "使用供应商历史PB构造1/PB代理，未自行重建严格 "
    "point-in-time book equity，供应商历史修订政策未完全核验。"
)


@dataclass
class AuditResult:
    check_id: str
    category: str
    status: str
    observed: str
    expected: str
    details: str


class AuditCollector:
    def __init__(self) -> None:
        self.results: list[AuditResult] = []

    def add(
        self,
        check_id: str,
        category: str,
        passed: bool,
        observed: Any,
        expected: Any,
        details: str,
        *,
        warning: bool = False,
    ) -> None:
        if warning:
            status = "WARN"
        else:
            status = "PASS" if passed else "FAIL"
        self.results.append(
            AuditResult(
                check_id=check_id,
                category=category,
                status=status,
                observed=str(observed),
                expected=str(expected),
                details=details,
            )
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(result) for result in self.results)


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


def _write_text_atomic(text: str, relative_path: str) -> None:
    output = absolute(relative_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_text(text, encoding="utf-8")
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


def _all_config_paths_relative(config: dict[str, Any]) -> bool:
    path_values = (
        list(config["inputs"].values())
        + list(config["outputs"].values())
        + list(config["protected_p2_inputs"])
    )
    return all(not Path(str(value)).is_absolute() for value in path_values)


def _result_frames(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    keys = (
        "factor_coverage",
        "monthly_rank_ic",
        "ic_summary",
        "quintile_returns",
        "annual_results",
        "factor_correlations_monthly",
        "factor_correlations_summary",
        "industry_exposure",
        "size_exposure",
        "worst_periods",
    )
    return {
        key: pd.read_csv(absolute(config["outputs"][key]))
        for key in keys
    }


def _direct_rank_ic(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    panel["signal_date"] = pd.to_datetime(panel["signal_date"])
    for signal_date, month in panel.groupby("signal_date", sort=True):
        evaluation = month.loc[
            month["universe_eligible"].fillna(False)
            & month["next_month_return"].notna()
        ]
        for factor_name, columns in FACTOR_COLUMNS.items():
            valid = evaluation[
                [columns["zscore"], "next_month_return"]
            ].dropna()
            value = (
                valid[columns["zscore"]].corr(
                    valid["next_month_return"], method="spearman"
                )
                if len(valid) >= 5
                else math.nan
            )
            rows.append(
                {
                    "signal_date": signal_date,
                    "factor": factor_name,
                    "rank_ic_recomputed": value,
                    "observations_recomputed": len(valid),
                }
            )
    return pd.DataFrame(rows)


def _markdown_report(
    audit_frame: pd.DataFrame,
    overall_status: str,
    panel_summary: dict[str, Any],
    boundary_summary: pd.DataFrame,
    ic_summary: pd.DataFrame,
    audited_at: str,
) -> str:
    pass_count = int((audit_frame["status"] == "PASS").sum())
    warn_count = int((audit_frame["status"] == "WARN").sum())
    fail_count = int((audit_frame["status"] == "FAIL").sum())
    lines = [
        "# P2 单因子研究审计报告",
        "",
        f"- 审计时间（UTC）：{audited_at}",
        f"- 总体状态：**{overall_status}**",
        (
            f"- 检查结果：{pass_count} PASS / {warn_count} WARN / "
            f"{fail_count} FAIL"
        ),
        "- 阶段范围：仅 2016-01-01 至 2019-12-31 研究期",
        "- 未运行：验证期、OOS、复合因子、可交易组合、成交模拟、回测",
        "",
        "## 数据与样本",
        "",
        f"- 面板行数：{panel_summary['rows']:,}",
        f"- 月份数：{panel_summary['months']}",
        f"- 股票-月主键重复：{panel_summary['duplicate_keys']}",
        f"- 股票池合格行数：{panel_summary['eligible_rows']:,}",
        f"- 有下一月评价标签的合格行数：{panel_summary['evaluated_rows']:,}",
        f"- 验证期或更晚行数：{panel_summary['validation_or_later_rows']}",
        f"- OOS 行数：{panel_summary['oos_rows']}",
        "",
        "## 跨代码价格边界",
        "",
        (
            f"- 映射主体：{len(boundary_summary)}；"
            f"全部状态："
            f"{', '.join(sorted(boundary_summary['boundary_status'].unique()))}"
        ),
        (
            "- 仅 `security_code_interval_valid = True` 的行按 "
            "`canonical_ts_code` 形成连续历史；供应商新代码生效日前回填"
            "不进入 P2。"
        ),
        "",
        "## 固定研究口径",
        "",
        "- `bm_proxy = 1 / pb`，仅 `pb > 0`。",
        (
            "- `momentum_12_1 = adjusted_price(t-21) / "
            "adjusted_price(t-252) - 1`。"
        ),
        "- `lowvol_60 = -Std_samp(daily_return[t-59:t])`。",
        "- 月度横截面 1%/99% 分位去极值。",
        "- 月度横截面样本标准差 z-score。",
        "- Rank IC 为 Spearman 相关；ICIR 按 `mean/std*sqrt(12)`。",
        (
            "- 五分位为无成本下一月收益诊断，不是P3组合、交易或回测结果。"
        ),
        "",
        "## 单因子研究摘要",
        "",
        "| factor | months | mean Rank IC | ICIR(ann.) | positive IC |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in ic_summary.itertuples(index=False):
        lines.append(
            f"| {row.factor} | {int(row.months)} | "
            f"{row.mean_rank_ic:.6f} | "
            f"{row.rank_icir_annualized:.6f} | "
            f"{row.positive_rank_ic_rate:.2%} |"
        )
    lines.extend(
        [
            "",
            "这些统计只描述冻结研究期内的样本关系，不代表可交易策略表现。",
            "",
            "## 已披露限制",
            "",
            f"- {PB_DISCLOSURE}",
            "",
            "## 检查明细",
            "",
            "| ID | 类别 | 状态 | 观测 | 期望 |",
            "|---|---|---|---|---|",
        ]
    )
    for row in audit_frame.itertuples(index=False):
        lines.append(
            f"| {row.check_id} | {row.category} | {row.status} | "
            f"{row.observed} | {row.expected} |"
        )
    lines.append("")
    return "\n".join(lines)


def audit_p2() -> pd.DataFrame:
    audited_at = datetime.now(UTC).isoformat()
    config = load_config()
    outputs = config["outputs"]
    audit = AuditCollector()

    required_keys = (
        "security_code_boundary_summary",
        "security_code_boundary_tests",
        "security_code_boundary_report",
        "single_factor_panel",
        "factor_coverage",
        "monthly_rank_ic",
        "ic_summary",
        "quintile_returns",
        "annual_results",
        "factor_correlations_monthly",
        "factor_correlations_summary",
        "industry_exposure",
        "size_exposure",
        "worst_periods",
        "p2_input_hashes",
        "p2_run_manifest",
    )
    missing = [
        outputs[key]
        for key in required_keys
        if not absolute(outputs[key]).is_file()
    ]
    audit.add(
        "P2-001",
        "文件",
        not missing,
        f"missing={missing}",
        "missing=[]",
        "P2规定输出必须全部存在。",
    )
    if missing:
        frame = audit.frame()
        _write_csv_atomic(frame, outputs["p2_audit_summary"])
        raise FileNotFoundError(f"缺少P2输出：{missing}")

    audit.add(
        "P2-002",
        "路径",
        _all_config_paths_relative(config),
        _all_config_paths_relative(config),
        True,
        "P2配置中的输入、输出和受保护输入全部使用项目相对路径。",
    )

    boundary_tests = pd.read_csv(
        absolute(outputs["security_code_boundary_tests"])
    )
    boundary_summary = pd.read_csv(
        absolute(outputs["security_code_boundary_summary"])
    )
    boundary_failures = int((boundary_tests["status"] != "PASS").sum())
    audit.add(
        "P2-003",
        "代码边界",
        len(boundary_tests) == 18 and boundary_failures == 0,
        f"rows={len(boundary_tests)}, failures={boundary_failures}",
        "rows=18, failures=0",
        "三组映射各执行六类冻结边界测试。",
    )
    valid_duplicates = int(
        boundary_summary["valid_canonical_duplicate_rows"].sum()
    )
    raw_duplicates = int(
        boundary_summary["raw_canonical_duplicate_rows"].sum()
    )
    audit.add(
        "P2-004",
        "代码边界",
        valid_duplicates == 0 and raw_duplicates == 4333,
        f"raw={raw_duplicates}, valid={valid_duplicates}",
        "raw=4333, valid=0",
        "原始重复来自供应商回填；有效代码区间必须唯一。",
    )

    hashes = pd.read_csv(absolute(outputs["p2_input_hashes"]))
    current_mismatches = 0
    for row in hashes.itertuples(index=False):
        path = absolute(row.path)
        if (
            not path.is_file()
            or path.stat().st_size != row.size_bytes_after
            or _sha256(path) != row.sha256_after
        ):
            current_mismatches += 1
    hash_matches = bool(hashes["match"].all()) and current_mismatches == 0
    audit.add(
        "P2-005",
        "输入完整性",
        hash_matches,
        (
            f"recorded_match={bool(hashes['match'].all())}, "
            f"current_mismatches={current_mismatches}"
        ),
        "recorded_match=True, current_mismatches=0",
        "P1日频面板、月末面板和人工映射在P2前后不得变化。",
    )

    panel_path = sql_path(outputs["single_factor_panel"])
    with duckdb.connect() as connection:
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW panel AS
            SELECT * FROM read_parquet('{panel_path}')
            """
        )
        (
            row_count,
            month_count,
            minimum_date,
            maximum_date,
            eligible_rows,
            evaluated_rows,
            duplicate_keys,
            validation_rows,
            oos_rows,
        ) = connection.execute(
            """
            SELECT
                count(*),
                count(DISTINCT signal_date),
                min(signal_date),
                max(signal_date),
                count(*) FILTER (WHERE universe_eligible),
                count(*) FILTER (
                    WHERE universe_eligible
                      AND next_month_return IS NOT NULL
                ),
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
        panel_summary = {
            "rows": int(row_count),
            "months": int(month_count),
            "minimum_date": str(minimum_date),
            "maximum_date": str(maximum_date),
            "eligible_rows": int(eligible_rows),
            "evaluated_rows": int(evaluated_rows),
            "duplicate_keys": int(duplicate_keys),
            "validation_or_later_rows": int(validation_rows),
            "oos_rows": int(oos_rows),
        }
        audit.add(
            "P2-006",
            "样本",
            (
                minimum_date >= pd.Timestamp("2016-01-01").date()
                and maximum_date <= pd.Timestamp("2019-12-31").date()
                and month_count == 48
            ),
            (
                f"{minimum_date}..{maximum_date}, "
                f"months={month_count}"
            ),
            "research only, months=48",
            "P2面板只包含冻结研究期。",
        )
        audit.add(
            "P2-007",
            "样本闸门",
            validation_rows == 0 and oos_rows == 0,
            f"validation_or_later={validation_rows}, oos={oos_rows}",
            "validation_or_later=0, oos=0",
            "P2不得运行验证期或最终OOS。",
        )
        audit.add(
            "P2-008",
            "主键",
            duplicate_keys == 0,
            duplicate_keys,
            0,
            "canonical_ts_code + signal_date 必须唯一。",
        )
        code_interval_violations = connection.execute(
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
        audit.add(
            "P2-009",
            "代码映射",
            code_interval_violations == 0,
            code_interval_violations,
            0,
            "`ts_code` 必须是信号日真实交易代码。",
        )

        formula = connection.execute(
            """
            SELECT
                count(*) FILTER (
                    WHERE pb <= 0 AND bm_proxy IS NOT NULL
                ) AS invalid_bm_rows,
                max(abs(bm_proxy - 1.0 / pb)) FILTER (
                    WHERE pb > 0
                ) AS bm_error,
                max(abs(
                    momentum_12_1
                    - (
                        adjusted_price_t_minus_21
                        / adjusted_price_t_minus_252 - 1.0
                    )
                )) FILTER (
                    WHERE momentum_12_1 IS NOT NULL
                ) AS momentum_error,
                max(abs(
                    lowvol_60 + rolling_return_std_60
                )) FILTER (
                    WHERE lowvol_60 IS NOT NULL
                ) AS lowvol_error,
                max(abs(
                    next_month_return
                    - (next_adjusted_price / adjusted_price - 1.0)
                )) FILTER (
                    WHERE next_month_return IS NOT NULL
                ) AS forward_error,
                count(*) FILTER (
                    WHERE next_month_return IS NOT NULL
                      AND (
                        next_signal_date <= signal_date
                        OR date_diff(
                            'month', signal_date, next_signal_date
                        ) <> 1
                      )
                ) AS label_date_violations
            FROM panel
            """
        ).fetchone()
        factor_names = ("BM", "MOMENTUM", "LOWVOL", "FORWARD_LABEL")
        errors = (formula[1], formula[2], formula[3], formula[4])
        for index, (name, error) in enumerate(
            zip(factor_names, errors, strict=True), start=10
        ):
            passed = error is not None and float(error) <= 1e-12
            if name == "BM":
                passed = passed and formula[0] == 0
            audit.add(
                f"P2-{index:03d}",
                "公式",
                passed,
                (
                    f"max_error={error}"
                    + (
                        f", invalid_pb_rows={formula[0]}"
                        if name == "BM"
                        else ""
                    )
                ),
                "max_error<=1e-12",
                f"{name} 冻结公式全量复算。",
            )
        audit.add(
            "P2-014",
            "标签时序",
            formula[5] == 0,
            formula[5],
            0,
            "下一月标签必须严格晚于信号日且为下一自然月。",
        )

        universe_violations, converse_violations = connection.execute(
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
        audit.add(
            "P2-015",
            "股票池",
            universe_violations == 0 and converse_violations == 0,
            (
                f"eligible_violations={universe_violations}, "
                f"converse_violations={converse_violations}"
            ),
            "both=0",
            "冻结股票池与流动性前80%规则全量复算。",
        )
        min_ratio, average_ratio, max_ratio = connection.execute(
            """
            SELECT min(ratio), avg(ratio), max(ratio)
            FROM (
                SELECT
                    signal_date,
                    count(*) FILTER (
                        WHERE universe_eligible
                    )::DOUBLE
                    / nullif(
                        count(*) FILTER (
                            WHERE universe_pre_liquidity
                        ), 0
                    ) AS ratio
                FROM panel
                GROUP BY signal_date
            )
            """
        ).fetchone()
        audit.add(
            "P2-016",
            "股票池",
            min_ratio >= 0.799 and max_ratio <= 0.801,
            (
                f"min={min_ratio:.6f}, avg={average_ratio:.6f}, "
                f"max={max_ratio:.6f}"
            ),
            "monthly ratio approximately 0.80",
            "分位阈值含并列值，允许极小离散误差。",
        )
        z_errors = connection.execute(
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
        audit.add(
            "P2-017",
            "横截面处理",
            z_errors[0] <= 1e-12 and z_errors[1] <= 1e-12,
            f"max_mean_error={z_errors[0]}, max_std_error={z_errors[1]}",
            "both<=1e-12",
            "月度横截面z-score均值为0、样本标准差为1。",
        )
        winsor_violations = connection.execute(
            """
            SELECT count(*)
            FROM panel
            WHERE universe_eligible
              AND (
                bm_proxy_winsorized NOT BETWEEN bm_proxy_p01
                                                AND bm_proxy_p99
                OR momentum_12_1_winsorized
                    NOT BETWEEN momentum_12_1_p01
                            AND momentum_12_1_p99
                OR lowvol_60_winsorized NOT BETWEEN lowvol_60_p01
                                                 AND lowvol_60_p99
              )
            """
        ).fetchone()[0]
        audit.add(
            "P2-018",
            "横截面处理",
            winsor_violations == 0,
            winsor_violations,
            0,
            "合格样本的去极值结果必须位于1%/99%月度边界内。",
        )
        null_eligible_factors, ineligible_z = connection.execute(
            """
            SELECT
                count(*) FILTER (
                    WHERE universe_eligible
                      AND (
                        bm_proxy_z IS NULL
                        OR momentum_12_1_z IS NULL
                        OR lowvol_60_z IS NULL
                      )
                ),
                count(*) FILTER (
                    WHERE NOT universe_eligible
                      AND (
                        bm_proxy_z IS NOT NULL
                        OR momentum_12_1_z IS NOT NULL
                        OR lowvol_60_z IS NOT NULL
                      )
                )
            FROM panel
            """
        ).fetchone()
        audit.add(
            "P2-019",
            "横截面处理",
            null_eligible_factors == 0 and ineligible_z == 0,
            (
                f"eligible_null={null_eligible_factors}, "
                f"ineligible_z={ineligible_z}"
            ),
            "both=0",
            "标准化因子只对最终股票池输出。",
        )
        december_eligible, december_labels = connection.execute(
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
        audit.add(
            "P2-020",
            "无前视",
            december_eligible > 0 and december_labels == 0,
            (
                f"december_eligible={december_eligible}, "
                f"december_labels={december_labels}"
            ),
            "eligible>0, labels=0",
            "股票池形成不依赖未来标签；研究期末仍可形成信号但不评价。",
        )
        panel_columns = {
            row[0]
            for row in connection.execute("DESCRIBE panel").fetchall()
        }
        forbidden_columns = {
            "composite",
            "composite_z",
            "target_weight",
            "actual_weight",
            "order_quantity",
            "transaction_cost",
            "portfolio_return",
            "nav",
        }
        present_forbidden = sorted(panel_columns & forbidden_columns)
        audit.add(
            "P2-021",
            "范围",
            not present_forbidden,
            present_forbidden,
            [],
            "P2面板不得包含P3复合因子、交易或回测字段。",
        )

    frames = _result_frames(config)
    expected_rows = {
        "factor_coverage": 48 * 3,
        "monthly_rank_ic": 48 * 3,
        "ic_summary": 3,
        "quintile_returns": 47 * 3 * 6,
        "annual_results": 4 * 3,
        "factor_correlations_monthly": 48 * 3,
        "factor_correlations_summary": 3,
        "size_exposure": 48 * 3,
        "worst_periods": 3 * 2 * 5,
    }
    row_mismatches = {
        key: (len(frames[key]), expected)
        for key, expected in expected_rows.items()
        if len(frames[key]) != expected
    }
    audit.add(
        "P2-022",
        "统计输出",
        not row_mismatches and len(frames["industry_exposure"]) > 0,
        (
            f"mismatches={row_mismatches}, "
            f"industry_rows={len(frames['industry_exposure'])}"
        ),
        "mismatches={}, industry_rows>0",
        "覆盖率、IC、分组、年度、相关性及暴露输出形状完整。",
    )

    panel = pd.read_parquet(absolute(outputs["single_factor_panel"]))
    recomputed_ic = _direct_rank_ic(panel)
    recorded_ic = frames["monthly_rank_ic"].copy()
    recorded_ic["signal_date"] = pd.to_datetime(
        recorded_ic["signal_date"]
    )
    ic_comparison = recorded_ic.merge(
        recomputed_ic,
        on=["signal_date", "factor"],
        validate="one_to_one",
    )
    ic_error = (
        ic_comparison["rank_ic"]
        - ic_comparison["rank_ic_recomputed"]
    ).abs().max()
    observation_mismatches = int(
        (
            ic_comparison["observations"]
            != ic_comparison["observations_recomputed"]
        ).sum()
    )
    audit.add(
        "P2-023",
        "Rank IC",
        ic_error <= 1e-12 and observation_mismatches == 0,
        (
            f"max_error={ic_error}, "
            f"observation_mismatches={observation_mismatches}"
        ),
        "max_error<=1e-12, observation_mismatches=0",
        "从P2面板独立重算全部月度Spearman Rank IC。",
    )

    quintiles = frames["quintile_returns"]
    pivot = quintiles.pivot(
        index=["signal_date", "factor"],
        columns="quintile",
        values="mean_next_month_return",
    )
    spread_error = (
        pivot["TOP_MINUS_BOTTOM"] - (pivot["Q5"] - pivot["Q1"])
    ).abs().max()
    return_types = set(quintiles["return_type"].dropna())
    audit.add(
        "P2-024",
        "五分位",
        spread_error <= 1e-12
        and return_types == {"DIAGNOSTIC_FORWARD_RETURN_NO_COST"},
        f"max_spread_error={spread_error}, types={sorted(return_types)}",
        (
            "max_spread_error<=1e-12, "
            "type=DIAGNOSTIC_FORWARD_RETURN_NO_COST"
        ),
        "Top-minus-bottom必须等于Q5-Q1并明确不是回测。",
    )

    manifest_path = absolute(outputs["p2_run_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scope_violations = [
        key
        for key, value in manifest["scope_guards"].items()
        if bool(value)
    ]
    audit.add(
        "P2-025",
        "范围",
        not scope_violations,
        scope_violations,
        [],
        "验证期、OOS、复合因子、组合、订单、成本和回测闸门均应为False。",
    )
    hash_mismatches: list[str] = []
    for key, expected_hash in manifest["output_sha256"].items():
        path = absolute(outputs[key])
        if not path.is_file() or _sha256(path) != expected_hash:
            hash_mismatches.append(key)
    audit.add(
        "P2-026",
        "输出完整性",
        not hash_mismatches,
        hash_mismatches,
        [],
        "构建后输出必须与manifest记录的SHA-256一致。",
    )

    config_text = (PROJECT_ROOT / CONFIG_PATH).read_text(encoding="utf-8")
    manifest_text = manifest_path.read_text(encoding="utf-8")
    disclosure_present = PB_DISCLOSURE in manifest_text
    audit.add(
        "P2-027",
        "已知限制",
        disclosure_present,
        f"disclosure_present={disclosure_present}",
        "True",
        PB_DISCLOSURE,
        warning=True,
    )
    audit.add(
        "P2-028",
        "配置",
        "oos_start: \"2022-01-01\"" in config_text
        and manifest["panel"]["oos_rows"] == 0,
        (
            f"oos_guard_configured="
            f"{'oos_start: \"2022-01-01\"' in config_text}, "
            f"manifest_oos_rows={manifest['panel']['oos_rows']}"
        ),
        "configured=True, manifest_oos_rows=0",
        "最终OOS仍处于未授权状态。",
    )

    frame = audit.frame()
    fail_count = int((frame["status"] == "FAIL").sum())
    warn_count = int((frame["status"] == "WARN").sum())
    overall_status = (
        "FAIL"
        if fail_count
        else (
            "PASS_WITH_DISCLOSED_LIMITATION"
            if warn_count
            else "PASS"
        )
    )
    _write_csv_atomic(frame, outputs["p2_audit_summary"])
    report = _markdown_report(
        frame,
        overall_status,
        panel_summary,
        boundary_summary,
        frames["ic_summary"],
        audited_at,
    )
    _write_text_atomic(report, outputs["p2_audit_report"])

    manifest["status"] = (
        "P2_AUDIT_FAILED"
        if fail_count
        else (
            "P2_ACCEPTED_WITH_DISCLOSED_LIMITATION"
            if warn_count
            else "P2_ACCEPTED"
        )
    )
    manifest["audit"] = {
        "audited_at_utc": audited_at,
        "overall_status": overall_status,
        "pass_count": int((frame["status"] == "PASS").sum()),
        "warn_count": warn_count,
        "fail_count": fail_count,
        "audit_summary_sha256": _sha256(
            absolute(outputs["p2_audit_summary"])
        ),
        "audit_report_sha256": _sha256(
            absolute(outputs["p2_audit_report"])
        ),
    }
    _write_json_atomic(manifest, outputs["p2_run_manifest"])

    print(
        f"[P2 AUDIT] {overall_status}: "
        f"PASS={manifest['audit']['pass_count']}, "
        f"WARN={warn_count}, FAIL={fail_count}",
        flush=True,
    )
    if fail_count:
        failed = frame.loc[frame["status"] == "FAIL"]
        raise RuntimeError(
            "P2审计失败："
            + failed[
                ["check_id", "observed", "expected"]
            ].to_dict(orient="records").__repr__()
        )
    return frame

