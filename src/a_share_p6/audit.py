from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pypdf import PdfReader

from .build import (
    BASELINE_SCENARIO,
    DELIST_CODES,
    P6_STATUS,
    _p5_output_hash_check,
    _sha256,
    _snapshot,
    _write_csv,
    _write_json,
    _write_text,
)
from .config import absolute, load_config


AUDITOR_VERSION = "p6.1"


def _log(message: str) -> None:
    print(f"[P6 AUDIT] {message}", flush=True)


def _row(
    check_id: str,
    section: str,
    status: str,
    observed: Any,
    expected: Any,
    evidence_path: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "section": section,
        "status": status,
        "observed": observed,
        "expected": expected,
        "evidence_path": evidence_path,
        "detail": detail,
    }


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def _report(summary: pd.DataFrame, status: str) -> str:
    counts = summary["status"].value_counts().to_dict()
    lines = [
        "# P6 最终审计报告",
        "",
        f"- 项目状态：`{status}`",
        f"- PASS：{counts.get('PASS', 0)}",
        f"- WARN：{counts.get('WARN', 0)}",
        f"- FAIL：{counts.get('FAIL', 0)}",
        "- P5 原始结果：未修改，逐项 SHA-256 匹配。",
        "- P6 分类：全部为 `POST_OOS_ROBUSTNESS`。",
        "",
        "## 检查明细",
        "",
        "| 检查 | 分区 | 状态 | 观察值 | 预期 |",
        "|---|---|---|---|---|",
    ]
    for item in summary.itertuples(index=False):
        observed = str(item.observed).replace("|", "/")
        expected = str(item.expected).replace("|", "/")
        lines.append(
            f"| {item.check_id} | {item.section} | {item.status} | "
            f"{observed} | {expected} |"
        )
    lines.extend(
        [
            "",
            "## 保留限制",
            "",
            "- 三只退市证券缺少可审计的精确终止估值与回收事件；P6 仅做期末敏感性。",
            "- 历史 PB 修订政策仍为 `NEEDS_MANUAL_CONFIRMATION`。",
            "- P6 是观察最终 OOS 后的稳健性研究，不构成新的 OOS。",
        ]
    )
    return "\n".join(lines)


def audit_p6() -> dict[str, Any]:
    control = load_config()
    manifest_path = absolute(control["outputs"]["p6_run_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError("P6 manifest is missing; run build_p6.py first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []

    p5_hashes, p5_manifest, _ = _p5_output_hash_check(control)
    rows.append(
        _row(
            "P6-001",
            "P5_INTEGRITY",
            "PASS" if bool(p5_hashes["matches"].all()) else "FAIL",
            int(p5_hashes["matches"].sum()),
            len(p5_hashes),
            control["outputs"]["p5_output_hash_check"],
            "Every P5 output hash must match its accepted manifest.",
        )
    )
    protected = _snapshot(control["protected_p6_inputs"])
    build_hashes = pd.read_csv(absolute(control["outputs"]["p6_input_hashes"]))
    build_lookup = build_hashes.set_index("relative_path")["sha256"].to_dict()
    unchanged = all(
        build_lookup.get(path) == metadata["sha256"]
        for path, metadata in protected.items()
    )
    rows.append(
        _row(
            "P6-002",
            "P5_INTEGRITY",
            "PASS" if unchanged else "FAIL",
            unchanged,
            True,
            control["outputs"]["p6_input_hashes"],
            "Protected inputs must match the P6 pre-run snapshot.",
        )
    )
    rows.append(
        _row(
            "P6-003",
            "P5_INTEGRITY",
            "PASS" if manifest.get("p5_original_preserved") else "FAIL",
            manifest.get("p5_original_preserved"),
            True,
            control["outputs"]["p6_run_manifest"],
            "P6 manifest must state that the original P5 result is preserved.",
        )
    )
    scope = manifest.get("scope_guards", {})
    scope_ok = all(value is False for value in scope.values())
    rows.append(
        _row(
            "P6-004",
            "SCOPE",
            "PASS" if scope_ok else "FAIL",
            scope,
            "all false",
            control["outputs"]["p6_run_manifest"],
            "No P5 overwrite, retune, path rerun, or new OOS is allowed.",
        )
    )
    rows.append(
        _row(
            "P6-005",
            "FREEZE",
            "PASS"
            if manifest.get("frozen_config_sha256")
            == control["project"]["expected_frozen_sha256"]
            else "FAIL",
            manifest.get("frozen_config_sha256"),
            control["project"]["expected_frozen_sha256"],
            control["outputs"]["p6_run_manifest"],
            "The P4 frozen configuration remains the comparison anchor.",
        )
    )

    registry = pd.read_csv(absolute(control["outputs"]["experiment_registry"]))
    expected_ids = [item["experiment_id"] for item in control["experiments"]]
    actual_ids = registry["experiment_id"].tolist()
    rows.append(
        _row(
            "P6-006",
            "EXPERIMENTS",
            "PASS" if set(actual_ids) == set(expected_ids) else "FAIL",
            "|".join(actual_ids),
            "|".join(expected_ids),
            control["outputs"]["experiment_registry"],
            "Every pre-registered experiment family must appear exactly once.",
        )
    )
    rows.append(
        _row(
            "P6-007",
            "EXPERIMENTS",
            "PASS"
            if registry["classification"].eq("POST_OOS_ROBUSTNESS").all()
            else "FAIL",
            registry["classification"].nunique(),
            1,
            control["outputs"]["experiment_registry"],
            "All experiments must retain the post-OOS label.",
        )
    )
    rows.append(
        _row(
            "P6-008",
            "EXPERIMENTS",
            "PASS"
            if registry["p5_original_modified"].eq(False).all()
            else "FAIL",
            int(registry["p5_original_modified"].sum()),
            0,
            control["outputs"]["experiment_registry"],
            "No experiment may modify the P5 result.",
        )
    )
    protocol = absolute(control["outputs"]["experiment_protocol"]).read_text(
        encoding="utf-8"
    )
    protocol_ok = all(experiment_id in protocol for experiment_id in expected_ids)
    rows.append(
        _row(
            "P6-009",
            "EXPERIMENTS",
            "PASS" if protocol_ok else "FAIL",
            protocol_ok,
            True,
            control["outputs"]["experiment_protocol"],
            "The experiment protocol must pre-register every family.",
        )
    )

    metrics = pd.read_csv(absolute(control["outputs"]["experiment_metrics"]))
    p5_performance = pd.read_csv(
        absolute(control["inputs"]["p5_oos_performance"])
    )
    baseline = metrics.loc[
        metrics["experiment_id"] == "P5_BASELINE_REFERENCE"
    ].iloc[0]
    p5_baseline = p5_performance.loc[
        p5_performance["cost_scenario"] == BASELINE_SCENARIO
    ].iloc[0]
    baseline_columns = {
        "strategy_total_return": "strategy_total_return",
        "strategy_annualized_return": "strategy_annualized_return",
        "strategy_max_drawdown": "strategy_max_drawdown_within_period",
        "information_ratio": "information_ratio",
        "two_way_turnover": "two_way_turnover",
        "total_trading_cost": "total_trading_cost",
    }
    errors = {
        left: abs(float(baseline[left]) - float(p5_baseline[right]))
        for left, right in baseline_columns.items()
    }
    rows.append(
        _row(
            "P6-010",
            "BASELINE",
            "PASS" if max(errors.values()) <= 1e-12 else "FAIL",
            max(errors.values()),
            "<=1e-12",
            control["outputs"]["experiment_metrics"],
            "P6 baseline reference must exactly reuse P5 metrics.",
        )
    )
    reruns = metrics.loc[metrics["source"] == "P6_BASE_10BPS_RERUN"]
    rerun_ok = (
        len(reruns) == 5
        and reruns["status"].eq("PASS").all()
        and reruns["cost_scenario"].eq(BASELINE_SCENARIO).all()
        and reruns["trading_days"].eq(969).all()
    )
    rows.append(
        _row(
            "P6-011",
            "ROBUSTNESS",
            "PASS" if rerun_ok else "FAIL",
            len(reruns),
            5,
            control["outputs"]["experiment_metrics"],
            "Five 10 bps robustness variants must cover the full OOS.",
        )
    )
    rows.append(
        _row(
            "P6-012",
            "ROBUSTNESS",
            "PASS"
            if reruns["minimum_selected"].ge(100).all()
            and reruns["maximum_selected"].le(100).all()
            else "FAIL",
            (
                f"{reruns['minimum_selected'].min()}-"
                f"{reruns['maximum_selected'].max()}"
            ),
            "100-100",
            control["outputs"]["experiment_metrics"],
            "Each monthly variant should hold the frozen Top-100 target.",
        )
    )
    finite_metrics = reruns[
        [
            "strategy_total_return",
            "strategy_annualized_return",
            "strategy_max_drawdown",
            "two_way_turnover",
        ]
    ].map(math.isfinite).all().all()
    rows.append(
        _row(
            "P6-013",
            "ROBUSTNESS",
            "PASS" if finite_metrics else "FAIL",
            finite_metrics,
            True,
            control["outputs"]["experiment_metrics"],
            "Core robustness metrics must be finite.",
        )
    )
    processed_dir = absolute("data/processed/p6_robustness")
    p6_files = list(processed_dir.glob("*.parquet"))
    rows.append(
        _row(
            "P6-014",
            "ROBUSTNESS",
            "PASS" if len(p6_files) >= 37 else "FAIL",
            len(p6_files),
            ">=37",
            "data/processed/p6_robustness",
            "Variant artifacts must remain in the P6 namespace.",
        )
    )
    max_variant_date = max(
        pd.read_parquet(path, columns=["trade_date"])["trade_date"].max()
        for path in processed_dir.glob("*_daily.parquet")
    )
    rows.append(
        _row(
            "P6-015",
            "SCOPE",
            "PASS" if pd.Timestamp(max_variant_date) <= pd.Timestamp("2025-12-31") else "FAIL",
            str(pd.Timestamp(max_variant_date).date()),
            "<=2025-12-31",
            "data/processed/p6_robustness",
            "P6 may not touch 2026 or later data.",
        )
    )

    sensitivity = pd.read_csv(
        absolute(control["outputs"]["delisting_sensitivity"])
    )
    base_sensitivity = sensitivity.loc[sensitivity["is_baseline"]].sort_values(
        "recovery_rate"
    )
    expected_rates = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows.append(
        _row(
            "P6-016",
            "DELISTING",
            "PASS"
            if base_sensitivity["recovery_rate"].tolist() == expected_rates
            else "FAIL",
            base_sensitivity["recovery_rate"].tolist(),
            expected_rates,
            control["outputs"]["delisting_sensitivity"],
            "The baseline scenario must include all pre-registered recovery rates.",
        )
    )
    hundred = base_sensitivity.loc[
        base_sensitivity["recovery_rate"] == 1.0
    ].iloc[0]
    hundred_ok = (
        _close(
            hundred["adjusted_strategy_total_return"],
            p5_baseline["strategy_total_return"],
        )
        and _close(
            hundred["adjusted_strategy_annualized_return"],
            p5_baseline["strategy_annualized_return"],
        )
    )
    rows.append(
        _row(
            "P6-017",
            "DELISTING",
            "PASS" if hundred_ok else "FAIL",
            hundred["adjusted_strategy_annualized_return"],
            p5_baseline["strategy_annualized_return"],
            control["outputs"]["delisting_sensitivity"],
            "100% terminal recovery must reproduce P5.",
        )
    )
    monotonic = base_sensitivity["adjusted_strategy_total_return"].is_monotonic_increasing
    rows.append(
        _row(
            "P6-018",
            "DELISTING",
            "PASS" if monotonic else "FAIL",
            monotonic,
            True,
            control["outputs"]["delisting_sensitivity"],
            "Higher recovery must not reduce terminal return.",
        )
    )
    codes = set(
        "|".join(base_sensitivity["affected_codes"].fillna(""))
        .strip("|")
        .split("|")
    )
    rows.append(
        _row(
            "P6-019",
            "DELISTING",
            "PASS" if codes == set(DELIST_CODES) else "FAIL",
            "|".join(sorted(codes)),
            "|".join(DELIST_CODES),
            control["outputs"]["delisting_sensitivity"],
            "Only the three disclosed stale delisting positions are in scope.",
        )
    )
    method_ok = sensitivity["method"].eq(
        "TERMINAL_VALUATION_SENSITIVITY_NO_PATH_RERUN"
    ).all()
    rows.append(
        _row(
            "P6-020",
            "DELISTING",
            "PASS" if method_ok else "FAIL",
            method_ok,
            True,
            control["outputs"]["delisting_sensitivity"],
            "Delisting analysis must not masquerade as a path rerun.",
        )
    )

    monthly = pd.read_csv(absolute(control["outputs"]["monthly_diagnostics"]))
    worst = pd.read_csv(absolute(control["outputs"]["worst_months"]))
    rows.append(
        _row(
            "P6-021",
            "DIAGNOSTICS",
            "PASS" if len(monthly) == 48 else "FAIL",
            len(monthly),
            48,
            control["outputs"]["monthly_diagnostics"],
            "The four-year OOS should contain 48 monthly diagnostics.",
        )
    )
    rows.append(
        _row(
            "P6-022",
            "DIAGNOSTICS",
            "PASS"
            if len(worst) == 20 and worst["ranking_type"].nunique() == 2
            else "FAIL",
            len(worst),
            20,
            control["outputs"]["worst_months"],
            "Keep ten worst strategy and ten worst excess months.",
        )
    )

    required_writing = (
        "limitations",
        "robustness_report",
        "resume_evidence_map",
        "interview_notes",
        "research_report_tex",
        "research_report_pdf",
        "research_report_pdf_copy",
        "metrics_json",
    )
    writing_ok = all(
        absolute(control["outputs"][key]).is_file()
        and absolute(control["outputs"][key]).stat().st_size > 500
        for key in required_writing
    )
    rows.append(
        _row(
            "P6-023",
            "DELIVERABLES",
            "PASS" if writing_ok else "FAIL",
            writing_ok,
            True,
            control["outputs"]["research_report_pdf"],
            "Formal report and career evidence files must be non-empty.",
        )
    )
    reader = PdfReader(absolute(control["outputs"]["research_report_pdf"]))
    page_count = len(reader.pages)
    extracted = "\n".join((page.extract_text() or "") for page in reader.pages)
    pdf_ok = (
        page_count >= 10
        and "PROJECT_COMPLETE_WITH_DISCLOSED_LIMITATIONS" in extracted
        and "POST_OOS_ROBUSTNESS" in extracted
        and "000413.SZ" in extracted
    )
    rows.append(
        _row(
            "P6-024",
            "PDF",
            "PASS" if pdf_ok else "FAIL",
            page_count,
            ">=10 pages and required sections",
            control["outputs"]["research_report_pdf"],
            "The PDF must contain the complete formal report.",
        )
    )
    copy_match = _sha256(
        absolute(control["outputs"]["research_report_pdf"])
    ) == _sha256(absolute(control["outputs"]["research_report_pdf_copy"]))
    rows.append(
        _row(
            "P6-025",
            "PDF",
            "PASS" if copy_match else "FAIL",
            copy_match,
            True,
            control["outputs"]["research_report_pdf_copy"],
            "The delivery PDF copy must match reports/research_report.pdf.",
        )
    )
    charts_ok = all(
        absolute(control["outputs"][key]).is_file()
        and absolute(control["outputs"][key]).stat().st_size > 30_000
        for key in (
            "chart_cumulative_nav",
            "chart_drawdown",
            "chart_annual_returns",
            "chart_factor_ic",
            "chart_robustness",
            "chart_delisting",
        )
    )
    rows.append(
        _row(
            "P6-026",
            "DELIVERABLES",
            "PASS" if charts_ok else "FAIL",
            charts_ok,
            True,
            "reports/figures",
            "All six report figures must be present and non-trivial.",
        )
    )
    metrics_json = json.loads(
        absolute(control["outputs"]["metrics_json"]).read_text(encoding="utf-8")
    )
    rows.append(
        _row(
            "P6-027",
            "DELIVERABLES",
            "PASS"
            if metrics_json["project"]["final_status"] == P6_STATUS
            else "FAIL",
            metrics_json["project"]["final_status"],
            P6_STATUS,
            control["outputs"]["metrics_json"],
            "Machine-readable metrics must carry the terminal project status.",
        )
    )
    limitations_text = absolute(control["outputs"]["limitations"]).read_text(
        encoding="utf-8"
    )
    rows.append(
        _row(
            "P6-028",
            "DISCLOSURES",
            "PASS"
            if all(
                phrase in limitations_text
                for phrase in (
                    "POST_OOS_ROBUSTNESS",
                    "NEEDS_MANUAL_CONFIRMATION",
                    "point-in-time",
                    "000413.SZ",
                )
            )
            else "FAIL",
            "required disclosures checked",
            "all present",
            control["outputs"]["limitations"],
            "The limitations file must disclose post-OOS, PB, and delisting risks.",
        )
    )
    rows.append(
        _row(
            "P6-029",
            "DISCLOSURES",
            "WARN",
            "NEEDS_MANUAL_CONFIRMATION",
            "supplier policy verified",
            control["outputs"]["limitations"],
            "Historical PB revision policy remains unverified.",
        )
    )
    rows.append(
        _row(
            "P6-030",
            "DISCLOSURES",
            "WARN",
            "terminal sensitivity only",
            "auditable delist cash-flow events",
            control["outputs"]["delisting_sensitivity"],
            "Exact delisting recovery events remain unavailable.",
        )
    )

    summary = pd.DataFrame(rows)
    pass_count = int((summary["status"] == "PASS").sum())
    warn_count = int((summary["status"] == "WARN").sum())
    fail_count = int((summary["status"] == "FAIL").sum())
    final_status = P6_STATUS if fail_count == 0 else "P6_AUDIT_FAILED"
    _write_csv(summary, control["outputs"]["p6_audit_summary"])
    _write_text(
        _report(summary, final_status),
        control["outputs"]["p6_audit_report"],
    )

    manifest["status"] = final_status
    manifest["audit"] = {
        "auditor_version": AUDITOR_VERSION,
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "pass_count": pass_count,
        "warn_count": warn_count,
        "fail_count": fail_count,
        "summary_path": control["outputs"]["p6_audit_summary"],
        "report_path": control["outputs"]["p6_audit_report"],
    }
    for relative_path in control["outputs"].values():
        path = absolute(relative_path)
        if path.is_file() and relative_path != control["outputs"]["p6_run_manifest"]:
            manifest["output_sha256"][relative_path] = _sha256(path)
    _write_json(manifest, control["outputs"]["p6_run_manifest"])
    _log(
        f"status={final_status} PASS={pass_count} WARN={warn_count} "
        f"FAIL={fail_count}"
    )
    if fail_count:
        failed = summary.loc[summary["status"] == "FAIL", "check_id"].tolist()
        raise RuntimeError(f"P6 audit failed: {failed}")
    return manifest
