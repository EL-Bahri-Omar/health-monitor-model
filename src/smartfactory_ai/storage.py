"""
SmartFactory AI — CSV storage helpers.

Appends sensor readings and alerts to local CSV files for debugging
and offline analysis.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def append_row_csv(path: str | Path, row: dict[str, Any], fieldnames: list[str] | None = None) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or list(row.keys())
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_last_lines(path: str | Path, max_lines: int = 5000) -> list[str]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    return lines[-max_lines:]
