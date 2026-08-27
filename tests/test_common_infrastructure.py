from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from a_share_common.hashing import file_sha256
from a_share_common.io import atomic_write_csv, atomic_write_json, atomic_write_text
from a_share_common.paths import PROJECT_ROOT, absolute, sql_path
from a_share_p1 import config as p1_config
from a_share_p2 import config as p2_config
from a_share_p3 import config as p3_config
from a_share_p4 import config as p4_config
from a_share_p5 import config as p5_config
from a_share_p6 import config as p6_config


def test_stage_configs_share_the_same_project_root() -> None:
    modules = (
        p1_config,
        p2_config,
        p3_config,
        p4_config,
        p5_config,
        p6_config,
    )
    assert all(module.PROJECT_ROOT == PROJECT_ROOT for module in modules)
    assert all(module.load_config()["project"] for module in modules)


@pytest.mark.parametrize("value", ["../outside.txt", "../../outside.txt"])
def test_project_paths_reject_parent_traversal(value: str) -> None:
    with pytest.raises(ValueError, match="escapes the project root"):
        absolute(value)


def test_sql_path_normalizes_and_escapes_quotes() -> None:
    assert sql_path("data/demo's/file.parquet") == "data/demo''s/file.parquet"


def test_file_sha256_matches_standard_library(tmp_path: Path) -> None:
    payload = b"public-infrastructure-contract\r\n"
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)
    assert file_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_atomic_public_audit_writers_preserve_formats(tmp_path: Path) -> None:
    paths = (
        PROJECT_ROOT / "tests/_io_contract.txt",
        PROJECT_ROOT / "tests/_io_contract.json",
        PROJECT_ROOT / "tests/_io_contract.csv",
    )
    try:
        atomic_write_text("中文报告", "tests/_io_contract.txt")
        assert paths[0].read_text(encoding="utf-8") == "中文报告"

        atomic_write_json({"status": "PASS"}, "tests/_io_contract.json")
        assert json.loads(paths[1].read_text(encoding="utf-8")) == {
            "status": "PASS"
        }
        assert paths[1].read_bytes().endswith(b"\n")

        atomic_write_csv(pd.DataFrame({"value": [1]}), "tests/_io_contract.csv")
        assert paths[2].read_bytes().startswith(b"\xef\xbb\xbf")
    finally:
        for path in paths:
            path.unlink(missing_ok=True)
