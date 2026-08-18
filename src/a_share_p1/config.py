from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/paths.yaml")


def _assert_relative_path(value: str, label: str) -> None:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{label} 必须是项目相对路径：{value}")
    candidate = (PROJECT_ROOT / path).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} 不能越出项目根目录：{value}") from exc


def load_config() -> dict[str, Any]:
    config_file = PROJECT_ROOT / CONFIG_PATH
    with config_file.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)

    for section in ("inputs", "outputs", "policies"):
        for key, value in config[section].items():
            _assert_relative_path(str(value), f"{section}.{key}")
    for index, value in enumerate(config["protected_inputs"]):
        _assert_relative_path(str(value), f"protected_inputs[{index}]")
    return config


def absolute(relative_path: str) -> Path:
    _assert_relative_path(relative_path, "path")
    return PROJECT_ROOT / relative_path


def relative_posix(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
