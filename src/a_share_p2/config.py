from __future__ import annotations

from pathlib import Path
from typing import Any

from a_share_common.config import load_stage_config
from a_share_common.paths import (
    PROJECT_ROOT,
    absolute,
    assert_project_relative as _assert_relative_path,
    sql_path,
)


CONFIG_PATH = Path("configs/p2_factor_research.yaml")


def load_config() -> dict[str, Any]:
    return load_stage_config(CONFIG_PATH, protected_key="protected_p2_inputs")
