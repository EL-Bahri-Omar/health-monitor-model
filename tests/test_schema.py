"""Tests for SmartFactory AI sensor reading validation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest
from smartfactory_ai.schemas import validate_reading, SENSOR_COLUMNS, VALID_STATUSES


def test_validate_reading_coerces_types():
    """validate_reading should convert string values to proper types."""
    row = validate_reading({
        "machine_code": "CNC-024",
        "timestamp": "2026-05-13T08:00:00.000Z",
        "temperature": "45.2",
        "vibration": "2.1",
        "current": "8.0",
        "rpm": "1500",
        "status": "normal",
    })
    assert row["machine_code"] == "CNC-024"
    assert row["temperature"] == 45.2
    assert row["vibration"] == 2.1
    assert row["current"] == 8.0
    assert row["rpm"] == 1500.0
    assert row["status"] == "normal"


def test_validate_reading_missing_field():
    """validate_reading should raise ValueError when required fields are missing."""
    with pytest.raises(ValueError, match="Missing fields"):
        validate_reading({
            "machine_code": "CNC-024",
            "timestamp": "2026-05-13T08:00:00.000Z",
            "temperature": "45.2",
            # missing vibration, current, rpm
        })


def test_validate_reading_unknown_status():
    """Unknown status values should be normalised to 'unknown'."""
    row = validate_reading({
        "machine_code": "CNC-024",
        "timestamp": "2026-05-13T08:00:00.000Z",
        "temperature": 45.2,
        "vibration": 2.1,
        "current": 8.0,
        "rpm": 1500,
        "status": "broken",
    })
    assert row["status"] == "unknown"


def test_sensor_columns_match():
    """SENSOR_COLUMNS should list the 4 SmartFactory sensor types."""
    assert set(SENSOR_COLUMNS) == {"temperature", "vibration", "current", "rpm"}


def test_valid_statuses():
    """VALID_STATUSES should include all SmartFactory lifecycle states."""
    expected = {"normal", "degradation", "anomaly", "failure", "idle", "unknown"}
    assert expected.issubset(VALID_STATUSES)


def test_validate_reading_without_status():
    """validate_reading without status should default to 'unknown'."""
    row = validate_reading({
        "machine_code": "PUMP-001",
        "timestamp": "2026-05-13T08:00:00.000Z",
        "temperature": 50.0,
        "vibration": 3.0,
        "current": 9.5,
        "rpm": 1480,
    })
    assert row["status"] == "unknown"
