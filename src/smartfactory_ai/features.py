"""
SmartFactory AI — Feature extraction from sensor time windows.

Converts raw 4-sensor time-series windows (temperature, vibration, current, RPM)
into a feature vector suitable for the RandomForest fault classifier and the
IsolationForest anomaly detector.

Each window of N readings produces one feature row with statistical, slope,
and spectral features computed per sensor channel.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

FEATURE_VERSION = "2026.10-smartfactory"
SENSOR_COLUMNS = ["temperature", "vibration", "current", "rpm"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else 0.0


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=0)) if len(values) else 0.0


def _safe_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    slope, _ = np.polyfit(x, values.astype(float), 1)
    return float(slope)


def _safe_skew(values: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    centered = values - np.mean(values)
    std = np.std(values)
    if std == 0:
        return 0.0
    return float(np.mean((centered / std) ** 3))


def _safe_kurtosis(values: np.ndarray) -> float:
    if len(values) < 4:
        return 0.0
    centered = values - np.mean(values)
    std = np.std(values)
    if std == 0:
        return 0.0
    return float(np.mean((centered / std) ** 4) - 3.0)


def _zero_crossings(values: np.ndarray) -> int:
    if len(values) < 2:
        return 0
    centered = values - np.mean(values)
    return int(np.sum(np.diff(np.signbit(centered)) != 0))


def _mode(values: Iterable[str]) -> str:
    cleaned = [str(v) for v in values if str(v)]
    if not cleaned:
        return "unknown"
    return Counter(cleaned).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Core feature extraction
# ---------------------------------------------------------------------------

def extract_window_features(window: pd.DataFrame, sampling_interval_s: float = 3.0) -> dict[str, float | str | int]:
    """Extract features from a time window of sensor readings.

    Parameters
    ----------
    window : DataFrame with columns temperature, vibration, current, rpm
    sampling_interval_s : seconds between readings (default 3s for SmartFactory simulator)

    Returns
    -------
    Dictionary of named features (floats, ints, and optional string labels).
    """
    missing = [col for col in SENSOR_COLUMNS if col not in window.columns]
    if missing:
        raise ValueError(f"Missing sensor columns: {missing}")

    temp = window["temperature"].to_numpy(dtype=float)
    vib = window["vibration"].to_numpy(dtype=float)
    current = window["current"].to_numpy(dtype=float)
    rpm = window["rpm"].to_numpy(dtype=float)

    vib_rms = _rms(vib)
    vib_max_abs = float(np.max(np.abs(vib))) if len(vib) else 0.0
    crest = float(vib_max_abs / vib_rms) if vib_rms > 0 else 0.0

    features: dict[str, float | str | int] = {
        "sample_count": int(len(window)),

        # Temperature features
        "temperature_mean": float(np.mean(temp)),
        "temperature_std": _safe_std(temp),
        "temperature_max": float(np.max(temp)),
        "temperature_min": float(np.min(temp)),
        "temperature_slope": _safe_slope(temp),
        "temperature_range": float(np.max(temp) - np.min(temp)),

        # Vibration features
        "vibration_mean": float(np.mean(vib)),
        "vibration_std": _safe_std(vib),
        "vibration_rms": vib_rms,
        "vibration_max": float(np.max(vib)),
        "vibration_min": float(np.min(vib)),
        "vibration_peak_to_peak": float(np.max(vib) - np.min(vib)),
        "vibration_crest_factor": crest,
        "vibration_skew": _safe_skew(vib),
        "vibration_kurtosis": _safe_kurtosis(vib),
        "vibration_zero_crossings": _zero_crossings(vib),

        # Current features
        "current_mean": float(np.mean(current)),
        "current_std": _safe_std(current),
        "current_max": float(np.max(current)),
        "current_slope": _safe_slope(current),

        # RPM features
        "rpm_mean": float(np.mean(rpm)),
        "rpm_std": _safe_std(rpm),
        "rpm_min": float(np.min(rpm)),
        "rpm_range": float(np.max(rpm) - np.min(rpm)),
    }

    # Label and metadata columns (preserved for training, ignored at inference)
    if "status" in window.columns:
        features["status"] = _mode(window["status"].astype(str).tolist())
    if "machine_code" in window.columns:
        features["machine_code"] = _mode(window["machine_code"].astype(str).tolist())
    if "timestamp" in window.columns:
        features["window_start"] = str(window["timestamp"].iloc[0])
        features["window_end"] = str(window["timestamp"].iloc[-1])

    # Sanitise NaN / Inf
    for key, value in list(features.items()):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            features[key] = 0.0
    return features


def make_feature_table(
    raw: pd.DataFrame,
    window_size: int = 64,
    step_size: int = 32,
    sampling_interval_s: float = 3.0,
) -> pd.DataFrame:
    """Slide windows over raw readings and extract feature rows.

    Groups by machine_code to keep machines separate.
    """
    if raw.empty:
        raise ValueError("Raw data is empty")
    missing = [col for col in SENSOR_COLUMNS if col not in raw.columns]
    if missing:
        raise ValueError(f"Missing sensor columns: {missing}")

    df = raw.copy()
    if "machine_code" not in df.columns:
        df["machine_code"] = "CNC-001"
    if "timestamp" in df.columns:
        df = df.sort_values(["machine_code", "timestamp"])

    records: list[dict[str, float | str | int]] = []
    for _, group in df.groupby("machine_code", sort=False):
        group = group.reset_index(drop=True)
        if len(group) < window_size:
            continue
        for start in range(0, len(group) - window_size + 1, step_size):
            window = group.iloc[start : start + window_size]
            records.append(extract_window_features(window, sampling_interval_s=sampling_interval_s))

    if not records:
        raise ValueError("Not enough rows to build any time windows")
    return pd.DataFrame(records)


def feature_columns(feature_table: pd.DataFrame) -> list[str]:
    """Return only the numeric feature column names (exclude labels/metadata)."""
    excluded = {"status", "machine_code", "window_start", "window_end"}
    return [col for col in feature_table.columns if col not in excluded]
