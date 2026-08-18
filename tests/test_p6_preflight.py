from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from a_share_p6.build import _p5_output_hash_check, _sha256
from a_share_p6.config import absolute, load_config


def test_p6_configuration_is_post_oos_and_relative() -> None:
    control = load_config()
    assert control["project"]["experiment_classification"] == "POST_OOS_ROBUSTNESS"
    assert control["project"]["preserve_p5_original_results"] is True
    assert len(control["experiments"]) == 10


def test_p5_accepted_outputs_are_unchanged() -> None:
    control = load_config()
    hashes, manifest, _ = _p5_output_hash_check(control)
    assert manifest["status"] == "P5_ACCEPTED_FINAL_OOS_WITH_DISCLOSED_LIMITATIONS"
    assert hashes["matches"].all()
    assert len(hashes) == len(manifest["output_sha256"])


def test_frozen_configuration_hash_matches_protocol() -> None:
    control = load_config()
    assert (
        _sha256(absolute(control["inputs"]["frozen_config"]))
        == control["project"]["expected_frozen_sha256"]
    )
    manifest = json.loads(
        absolute(control["inputs"]["p5_manifest"]).read_text(encoding="utf-8")
    )
    assert (
        manifest["frozen_config"]["sha256"]
        == control["project"]["expected_frozen_sha256"]
    )
