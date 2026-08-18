from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from a_share_p6.build import BASELINE_SCENARIO, P6_STATUS, _sha256
from a_share_p6.config import absolute, load_config


CONFIG = load_config()
if not absolute(CONFIG["outputs"]["p6_run_manifest"]).is_file():
    pytest.skip("P6 has not been built yet", allow_module_level=True)


def _completed_manifest() -> tuple[dict, dict]:
    control = CONFIG
    path = absolute(control["outputs"]["p6_run_manifest"])
    return control, json.loads(path.read_text(encoding="utf-8"))


def test_p6_manifest_and_scope_guards() -> None:
    control, manifest = _completed_manifest()
    assert manifest["p5_original_preserved"] is True
    assert manifest["classification"] == "POST_OOS_ROBUSTNESS"
    assert all(value is False for value in manifest["scope_guards"].values())
    assert manifest["p5_output_hashes_all_match"] is True


def test_p6_experiment_registry_is_complete() -> None:
    control, manifest = _completed_manifest()
    registry = pd.read_csv(absolute(control["outputs"]["experiment_registry"]))
    expected = {item["experiment_id"] for item in control["experiments"]}
    assert set(registry["experiment_id"]) == expected
    assert registry["classification"].eq("POST_OOS_ROBUSTNESS").all()
    assert registry["p5_original_modified"].eq(False).all()


def test_p6_baseline_exactly_matches_p5() -> None:
    control, manifest = _completed_manifest()
    metrics = pd.read_csv(absolute(control["outputs"]["experiment_metrics"]))
    p5 = pd.read_csv(absolute(control["inputs"]["p5_oos_performance"]))
    baseline = metrics.loc[
        metrics["experiment_id"] == "P5_BASELINE_REFERENCE"
    ].iloc[0]
    source = p5.loc[p5["cost_scenario"] == BASELINE_SCENARIO].iloc[0]
    assert math.isclose(
        baseline["strategy_total_return"],
        source["strategy_total_return"],
        abs_tol=1e-12,
    )
    assert math.isclose(
        baseline["strategy_annualized_return"],
        source["strategy_annualized_return"],
        abs_tol=1e-12,
    )


def test_delisting_sensitivity_is_monotonic_and_reproduces_p5() -> None:
    control, manifest = _completed_manifest()
    sensitivity = pd.read_csv(
        absolute(control["outputs"]["delisting_sensitivity"])
    )
    baseline = sensitivity.loc[sensitivity["is_baseline"]].sort_values(
        "recovery_rate"
    )
    p5 = pd.read_csv(absolute(control["inputs"]["p5_oos_performance"]))
    source = p5.loc[p5["cost_scenario"] == BASELINE_SCENARIO].iloc[0]
    assert baseline["recovery_rate"].tolist() == [0, 0.25, 0.5, 0.75, 1]
    assert baseline["adjusted_strategy_total_return"].is_monotonic_increasing
    hundred = baseline.iloc[-1]
    assert math.isclose(
        hundred["adjusted_strategy_total_return"],
        source["strategy_total_return"],
        abs_tol=1e-12,
    )


def test_formal_report_is_valid_and_copied() -> None:
    control, manifest = _completed_manifest()
    report = absolute(control["outputs"]["research_report_pdf"])
    copy = absolute(control["outputs"]["research_report_pdf_copy"])
    assert len(PdfReader(report).pages) >= 10
    assert _sha256(report) == _sha256(copy)


def test_completed_project_metrics_status() -> None:
    control, manifest = _completed_manifest()
    metrics = json.loads(
        absolute(control["outputs"]["metrics_json"]).read_text(encoding="utf-8")
    )
    assert metrics["project"]["final_status"] == P6_STATUS
