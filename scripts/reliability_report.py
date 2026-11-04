#!/usr/bin/env python
"""
SmartFactory AI — Data reliability report.

Analyses collected sensor data to assess MQTT delivery reliability,
per-machine coverage, and potential data quality issues.

Usage:
  python scripts/reliability_report.py --input data/training_data.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def compute_reliability(df: pd.DataFrame) -> dict[str, object]:
    """Compute per-machine data coverage and quality stats."""
    if "machine_code" not in df.columns:
        raise ValueError("CSV must include a machine_code column")

    required_sensors = ["temperature", "vibration", "current", "rpm"]
    missing_cols = [col for col in required_sensors if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing sensor columns: {missing_cols}")

    machines: list[dict] = []
    total_rows = 0
    total_null = 0

    for machine_code, group in df.groupby("machine_code"):
        row_count = len(group)
        null_count = int(group[required_sensors].isnull().sum().sum())
        total_rows += row_count
        total_null += null_count

        stats: dict[str, object] = {
            "machine_code": str(machine_code),
            "readings": row_count,
            "null_values": null_count,
            "completeness": round(1.0 - (null_count / (row_count * len(required_sensors))), 4),
        }

        # Per-sensor stats
        for sensor in required_sensors:
            col = group[sensor].dropna()
            if len(col) > 0:
                stats[f"{sensor}_min"] = round(float(col.min()), 2)
                stats[f"{sensor}_max"] = round(float(col.max()), 2)
                stats[f"{sensor}_mean"] = round(float(col.mean()), 2)

        # Status distribution
        if "status" in group.columns:
            stats["status_distribution"] = group["status"].value_counts().to_dict()

        machines.append(stats)

    total_cells = total_rows * len(required_sensors)
    return {
        "total_readings": total_rows,
        "total_null_values": total_null,
        "overall_completeness": round(1.0 - (total_null / total_cells), 4) if total_cells else 0.0,
        "machines": machines,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmartFactory data reliability report")
    parser.add_argument("--input", default="data/training_data.csv")
    parser.add_argument("--output", default="reports/reliability_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    report = compute_reliability(df)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
