from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def assert_project_relative(value: str, label: str) -> None:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{label} must be project-relative: {value}")
    candidate = (PROJECT_ROOT / path).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the project root: {value}") from exc


def absolute(relative_path: str) -> Path:
    assert_project_relative(relative_path, "path")
    return PROJECT_ROOT / relative_path


def relative_posix(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def sql_path(relative_path: str) -> str:
    assert_project_relative(relative_path, "SQL path")
    return Path(relative_path).as_posix().replace("'", "''")

