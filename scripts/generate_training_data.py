#!/usr/bin/env python
"""
SmartFactory AI — Synthetic training data generator.

Generates labelled sensor telemetry matching SmartFactory's 4-sensor schema
(temperature, vibration, current, rpm) with realistic fault signatures for
model training.

Statuses match SmartFactory backend conventions:
  normal | degradation | anomaly | failure

Usage:
  python scripts/generate_training_data.py --output data/training_data.csv --rows 12000
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartfactory_ai.config import load_config


FIELDNAMES = ["machine_code", "timestamp", "temperature", "vibration", "current", "rpm", "status"]

# SmartFactory scenario targets (matching iot-simulator conventions)
SCENARIOS = {
    "normal":      {"temperature": 45.0, "vibration": 2.1,  "current": 8.0,  "rpm": 1500.0},
    "degradation": {"temperature": 62.0, "vibration": 4.8,  "current": 10.0, "rpm": 1450.0},
    "anomaly":     {"temperature": 85.0, "vibration": 9.2,  "current": 14.0, "rpm": 1300.0},
    "failure":     {"temperature": 95.0, "vibration": 12.0, "current": 16.0, "rpm": 200.0},
}

NOISE = {
    "normal":      {"temperature": 0.4, "vibration": 0.15, "current": 0.2, "rpm": 6.0},
    "degradation": {"temperature": 0.6, "vibration": 0.22, "current": 0.3, "rpm": 9.0},
    "anomaly":     {"temperature": 0.8, "vibration": 0.30, "current": 0.4, "rpm": 12.0},
    "failure":     {"temperature": 1.0, "vibration": 0.45, "current": 0.6, "rpm": 18.0},
}

STATUSES = list(SCENARIOS.keys())


def status_for_index(index: int, block_size: int = 1500) -> str:
    """Cycle through scenarios in blocks to create varied training data."""
    schedule = ["normal", "normal", "degradation", "normal", "anomaly", "normal", "failure", "normal"]
    return schedule[(index // block_size) % len(schedule)]


def generate_row(index: int, status: str, machine_code: str, rng: random.Random) -> dict[str, object]:
    """Generate one synthetic sensor reading for the given status."""
    targets = SCENARIOS[status]
    noise = NOISE[status]

    t = index * 3.0  # 3-second intervals (SmartFactory default)
    temp_val = targets["temperature"] + rng.gauss(0, noise["temperature"])
    vib_val = targets["vibration"] + rng.gauss(0, noise["vibration"])
    curr_val = targets["current"] + rng.gauss(0, noise["current"])
    rpm_val = targets["rpm"] + rng.gauss(0, noise["rpm"])

    # Add slow sinusoidal drift for realism
    temp_val += 1.5 * math.sin(2 * math.pi * 0.005 * t)
    rpm_val += 12.0 * math.sin(2 * math.pi * 0.04 * t)
    curr_val += 0.15 * math.sin(2 * math.pi * 0.02 * t)

    # Clamp to physical bounds
    temp_val = max(15.0, temp_val)
    vib_val = max(0.0, vib_val)
    curr_val = max(0.0, curr_val)
    rpm_val = max(0.0, rpm_val)

    start = datetime(2026, 5, 13, 8, 0, tzinfo=timezone.utc)
    timestamp = (start + timedelta(seconds=t)).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    return {
        "machine_code": machine_code,
        "timestamp": timestamp,
        "temperature": round(temp_val, 1),
        "vibration": round(vib_val, 2),
        "current": round(curr_val, 2),
        "rpm": round(rpm_val, 0),
        "status": status,
    }


def generate_rows(
    count: int,
    machine_code: str,
    seed: int,
    drop_rate: float,
) -> Iterable[dict[str, object]]:
    rng = random.Random(seed)
    for index in range(count):
        if drop_rate > 0 and rng.random() < drop_rate:
            continue
        status = status_for_index(index)
        yield generate_row(index, status, machine_code, rng)


def write_csv(rows: Iterable[dict[str, object]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SmartFactory training data")
    parser.add_argument("--output", default="data/training_data.csv")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--machine-code", default="CNC-001")
    parser.add_argument("--drop-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    rows = args.rows or cfg.simulation.default_rows
    drop_rate = cfg.simulation.drop_rate if args.drop_rate is None else args.drop_rate
    seed = cfg.simulation.seed if args.seed is None else args.seed
    machine_code = args.machine_code

    generated = generate_rows(rows, machine_code, seed, drop_rate)
    count = write_csv(generated, Path(args.output))
    print(f"Generated {count} training rows -> {args.output}")


if __name__ == "__main__":
    main()

