from __future__ import annotations

from pathlib import Path
from typing import Any

from a_share_common.config import load_stage_config
from a_share_common.paths import (
    PROJECT_ROOT,
    absolute,
    assert_project_relative as _assert_relative_path,
    relative_posix,
)


CONFIG_PATH = Path("configs/paths.yaml")


def load_config() -> dict[str, Any]:
    return load_stage_config(
        CONFIG_PATH,
        protected_key="protected_inputs",
        path_sections=("inputs", "outputs", "policies"),
    )
