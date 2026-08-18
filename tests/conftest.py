"""Keep public CI separate from private frozen-artifact verification."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


PRIVATE_RESEARCH_TESTS = {
    "test_p1_outputs.py",
    "test_p2_outputs.py",
    "test_p3_outputs.py",
    "test_p4_outputs.py",
    "test_p5_outputs.py",
    "test_p5_preflight.py",
    "test_p6_outputs.py",
    "test_p6_preflight.py",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip frozen research checks unless a private maintainer opts in."""
    run_private = os.environ.get("RUN_PRIVATE_RESEARCH_TESTS") == "1"
    skip_private = pytest.mark.skip(
        reason=(
            "requires licensed data and frozen private research artifacts; "
            "set RUN_PRIVATE_RESEARCH_TESTS=1 only in the complete private workspace"
        )
    )
    for item in items:
        if Path(str(item.fspath)).name not in PRIVATE_RESEARCH_TESTS:
            continue
        item.add_marker(pytest.mark.private_research_artifacts)
        if not run_private:
            item.add_marker(skip_private)

