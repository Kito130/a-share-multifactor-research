from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from .paths import PROJECT_ROOT, assert_project_relative


def load_stage_config(
    config_path: Path,
    *,
    protected_key: str,
    path_sections: Iterable[str] = ("inputs", "outputs"),
) -> dict[str, Any]:
    with (PROJECT_ROOT / config_path).open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)

    for section in path_sections:
        for key, value in config[section].items():
            assert_project_relative(str(value), f"{section}.{key}")
    for index, value in enumerate(config[protected_key]):
        assert_project_relative(str(value), f"{protected_key}[{index}]")
    return config

