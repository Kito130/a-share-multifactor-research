"""Small atomic output helpers shared by public audit stages."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import absolute


def _temporary_path(output: Path) -> Path:
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    return temporary


def atomic_write_text(
    text: str,
    relative_path: str,
    *,
    encoding: str = "utf-8",
) -> None:
    output = absolute(relative_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(output)
    temporary.write_text(text, encoding=encoding)
    os.replace(temporary, output)


def atomic_write_csv(
    frame: pd.DataFrame,
    relative_path: str,
    *,
    encoding: str = "utf-8-sig",
) -> None:
    output = absolute(relative_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(output)
    frame.to_csv(temporary, index=False, encoding=encoding)
    os.replace(temporary, output)


def atomic_write_json(
    payload: dict[str, Any],
    relative_path: str,
) -> None:
    output = absolute(relative_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(output)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, output)
