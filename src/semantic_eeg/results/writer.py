"""Writers for the detailed publication result schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from semantic_eeg.results.schemas import DETAILED_RESULT_COLUMNS, DETAILED_RESULT_KEYS
from semantic_eeg.utils.io import upsert_csv


def write_detailed_results(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Persist result rows idempotently using the frozen experimental key."""

    upsert_csv(path, rows, DETAILED_RESULT_KEYS, DETAILED_RESULT_COLUMNS)
