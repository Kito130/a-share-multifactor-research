from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from a_share_p4.build import _sha256
from a_share_p5.build import validate_p5_preflight
from a_share_p5.config import absolute, load_config


def test_p5_paths_are_relative() -> None:
    config = load_config()
    values = (
        list(config["inputs"].values())
        + list(config["outputs"].values())
        + list(config["protected_p5_inputs"])
    )
    assert all(not Path(str(value)).is_absolute() for value in values)


def test_p5_frozen_hash_anchor_is_unchanged() -> None:
    gate = validate_p5_preflight(require_fresh_run=False)
    config = gate["control"]
    expected = config["project"]["expected_frozen_sha256"]
    assert gate["frozen_hash"] == expected
    assert (
        _sha256(absolute(config["inputs"]["frozen_config"]))
        == expected
    )


def test_p5_has_explicit_authorization_record() -> None:
    config = load_config()
    assert (
        config["project"]["authorization_reference"]
        == "FROZEN_OOS_RELEASE_GATE"
    )
    assert config["project"]["one_shot_final_oos"] is True


def test_p4_gate_is_accepted_and_oos_was_untouched() -> None:
    config = load_config()
    manifest = json.loads(
        absolute(config["inputs"]["p4_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"].startswith(
        "P4_ACCEPTED_AND_FROZEN"
    )
    assert manifest["audit"]["fail_count"] == 0
    for key in (
        "oos_rows_written",
        "oos_results_computed",
        "oos_results_previewed",
        "p5_code_generated",
        "p5_run",
        "p6_run",
    ):
        assert manifest["scope_guards"][key] is False


def test_frozen_parameters_match_protected_p2_p3_configs() -> None:
    gate = validate_p5_preflight(require_fresh_run=False)
    frozen = gate["frozen"]
    p2 = gate["p2_config"]
    p3 = gate["p3_config"]
    for section in ("factors", "universe"):
        assert frozen[section] == p2[section]
    for section in (
        "portfolio",
        "composite",
        "cost_scenarios",
        "valuation",
        "corporate_actions",
        "metrics",
    ):
        assert frozen[section] == p3[section]
    with absolute(
        gate["control"]["inputs"]["frozen_config"]
    ).open("r", encoding="utf-8") as handle:
        reloaded = yaml.safe_load(handle)
    assert reloaded == frozen
