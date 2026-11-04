"""Tests for SmartFactory AI feature extraction."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from smartfactory_ai.features import extract_window_features, make_feature_table
from generate_training_data import generate_row

import random


def _make_window(status: str, count: int, machine_code: str = "CNC-001") -> pd.DataFrame:
    """Generate a DataFrame of synthetic readings for testing."""
    rng = random.Random(42)
    rows = [generate_row(i, status, machine_code, rng) for i in range(count)]
    return pd.DataFrame(rows)


def test_feature_table_builds_windows():
    """make_feature_table should produce at least 3 windows from 180 readings."""
    df = _make_window("normal", 180)
    table = make_feature_table(df, window_size=64, step_size=32, sampling_interval_s=3.0)
    assert len(table) >= 3
    assert "vibration_rms" in table.columns
    assert "temperature_mean" in table.columns
    assert "current_mean" in table.columns
    assert "rpm_mean" in table.columns
    assert table["status"].iloc[0] == "normal"


def test_extract_window_features_shape():
    """extract_window_features should return expected keys for a 128-reading window."""
    df = _make_window("degradation", 128)
    features = extract_window_features(df, sampling_interval_s=3.0)
    assert features["sample_count"] == 128
    assert features["vibration_rms"] > 0
    assert features["temperature_mean"] > 0
    assert features["current_mean"] > 0
    assert features["rpm_mean"] > 0


def test_feature_table_multiple_statuses():
    """Feature table should handle mixed statuses."""
    rng = random.Random(99)
    rows = []
    for status in ["normal", "degradation", "anomaly", "failure"]:
        rows.extend(generate_row(i, status, "CNC-002", rng) for i in range(80))
    df = pd.DataFrame(rows)
    table = make_feature_table(df, window_size=64, step_size=32, sampling_interval_s=3.0)
    assert len(table) > 0
    assert set(table["status"].unique()).issubset({"normal", "degradation", "anomaly", "failure"})


def test_feature_values_differ_across_statuses():
    """Features from 'normal' and 'failure' should be distinguishable."""
    normal_df = _make_window("normal", 128)
    failure_df = _make_window("failure", 128)
    normal_features = extract_window_features(normal_df, sampling_interval_s=3.0)
    failure_features = extract_window_features(failure_df, sampling_interval_s=3.0)
    # Failure should have higher temperature and vibration
    assert failure_features["temperature_mean"] > normal_features["temperature_mean"]
    assert failure_features["vibration_rms"] > normal_features["vibration_rms"]
