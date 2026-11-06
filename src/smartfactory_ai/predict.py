"""
SmartFactory AI — Fault prediction and anomaly scoring.

Loads the trained RandomForest classifier and IsolationForest anomaly model,
runs inference on a feature-extracted window, and returns a structured result
including estimated time to event for predictive maintenance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from smartfactory_ai.features import extract_window_features


# Operational thresholds: when a sensor's mean value crosses these levels, the
# machine is approaching or in a fault state.  Values are set BELOW the
# simulator's scenario targets so the AI model flags a problem BEFORE the
# machine actually reaches the scenario value — giving operators lead-time.
#
# Simulator targets for reference:
#   NORMAL      temperature 45, vibration 2.1, current 8,  rpm 1500
#   DEGRADATION temperature 62, vibration 4.8, current 10, rpm 1450
#   ANOMALY     temperature 85, vibration 9.2, current 14, rpm 1300
#
# "warning" is set around the DEGRADATION target so we alert while TRENDING
# toward degradation (not after it's fully reached). "critical" is set at
# the midpoint between degradation and anomaly targets.
SENSOR_THRESHOLDS = {
    "temperature": {"normal_max": 52.0, "warning": 58.0, "critical": 72.0},
    "vibration":   {"normal_max": 2.8,  "warning": 3.5,  "critical": 6.5},
    "current":     {"normal_max": 9.0,  "warning": 9.5,  "critical": 12.0},
    "rpm":         {"normal_max": 1480, "warning": 1460, "critical": 1380},
}


@dataclass
class PredictionResult:
    label: str           # "normal", "degradation", "anomaly", "failure"
    confidence: float    # 0.0 to 1.0
    anomaly_score: float # 0.0 to 1.0 (higher = more anomalous)
    features: dict[str, Any]
    estimated_hours_to_event: float | None = None  # hours until degradation/anomaly predicted
    driving_sensors: list[dict[str, Any]] = field(default_factory=list)  # sensors driving the prediction
    recommended_fix: str | None = None  # actionable repair suggestion for technicians


def _estimate_time_to_event(features: dict, label: str, sampling_interval_s: float) -> tuple[float | None, list[dict]]:
    """Estimate hours until threshold breach based on feature slopes.

    For each sensor, we look at:
      - current mean value
      - slope (rate of change per reading)
      - distance to the warning/critical threshold

    RPM is special: it FALLS during degradation (1500 → 1450 → 1300), so we
    check for negative slopes and compare against thresholds from above.

    Returns (estimated_hours, driving_sensors).
    """
    if label in ("failure",):
        # Already in failure — event is NOW
        return 0.0, [{"sensor": "all", "status": "failure", "urgency": "immediate"}]

    # RPM thresholds are "falling" — the machine degrades by LOSING RPM.
    FALLING_SENSORS = {"rpm"}

    driving = []
    min_hours = float("inf")

    for sensor, thresholds in SENSOR_THRESHOLDS.items():
        mean_val = features.get(f"{sensor}_mean", 0.0)
        slope = features.get(f"{sensor}_slope", 0.0)
        is_falling = sensor in FALLING_SENSORS

        if is_falling:
            # RPM: slope should be negative (falling) to indicate degradation
            if slope >= -0.001:
                continue  # Not falling — skip
            abs_slope = abs(slope)
            # For RPM, warning < critical < normal (values get LOWER)
            if mean_val > thresholds["warning"]:
                target = thresholds["warning"]
                target_label = "degradation"
                gap = mean_val - target
            elif mean_val > thresholds["critical"]:
                target = thresholds["critical"]
                target_label = "anomaly"
                gap = mean_val - target
            else:
                # Already past critical (below it)
                driving.append({
                    "sensor": sensor,
                    "currentValue": round(mean_val, 2),
                    "threshold": thresholds["critical"],
                    "status": "exceeded",
                    "trendDirection": "falling",
                })
                min_hours = 0.0
                continue
            trend_dir = "falling"
        else:
            # Temperature, vibration, current: slope should be positive (rising)
            if slope <= 0.001:
                continue
            abs_slope = slope
            if mean_val < thresholds["warning"]:
                target = thresholds["warning"]
                target_label = "degradation"
                gap = target - mean_val
            elif mean_val < thresholds["critical"]:
                target = thresholds["critical"]
                target_label = "anomaly"
                gap = target - mean_val
            else:
                driving.append({
                    "sensor": sensor,
                    "currentValue": round(mean_val, 2),
                    "threshold": thresholds["critical"],
                    "status": "exceeded",
                    "trendDirection": "rising",
                })
                min_hours = 0.0
                continue
            trend_dir = "rising"

        readings_to_breach = gap / abs_slope
        hours = (readings_to_breach * sampling_interval_s) / 3600.0

        if hours < 48:  # Only report if predicted within 48 hours
            driving.append({
                "sensor": sensor,
                "currentValue": round(mean_val, 2),
                "threshold": target,
                "predictedEvent": target_label,
                "estimatedHours": round(hours, 1),
                "trendDirection": trend_dir,
                "slopePerReading": round(slope, 4),
            })
            min_hours = min(min_hours, hours)

    if not driving:
        # No sensor trending toward a breach — check anomaly score
        if label in ("degradation", "anomaly"):
            # Model says degradation/anomaly but no clear slope trend:
            # estimate based on confidence and current state
            base_hours = 4.0 if label == "degradation" else 1.5
            return round(base_hours, 1), [{"sensor": "model", "status": label, "note": "pattern-based estimate"}]
        return None, []

    return round(min_hours, 1) if min_hours < float("inf") else None, driving


def _generate_recommended_fix(features: dict, label: str, driving_sensors: list[dict]) -> str | None:
    """Generate an actionable repair suggestion based on sensor analysis.

    The fix is specific to whichever sensors are trending abnormally and
    includes concrete actions a technician can take immediately.
    """
    if label == "normal":
        return None

    # Identify which sensors are driving the issue
    problem_sensors = set()
    for ds in driving_sensors:
        sensor = ds.get("sensor", "")
        if sensor in ("temperature", "vibration", "current", "rpm", "all", "model"):
            problem_sensors.add(sensor)

    # Also check raw feature values if driving_sensors is sparse
    temp = features.get("temperature_mean", 0.0)
    vib = features.get("vibration_rms", 0.0)
    curr = features.get("current_mean", 0.0)
    rpm = features.get("rpm_mean", 0.0)

    if temp > SENSOR_THRESHOLDS["temperature"]["warning"]:
        problem_sensors.add("temperature")
    if vib > SENSOR_THRESHOLDS["vibration"]["warning"]:
        problem_sensors.add("vibration")
    if curr > SENSOR_THRESHOLDS["current"]["warning"]:
        problem_sensors.add("current")
    if rpm < SENSOR_THRESHOLDS["rpm"]["warning"]:
        problem_sensors.add("rpm")

    fixes = []

    if label == "failure":
        return "⛔ Stop the machine immediately. Cut power and contact a senior technician. Do not restart before full inspection."

    # High temperature scenarios
    if "temperature" in problem_sensors:
        if temp > SENSOR_THRESHOLDS["temperature"]["critical"]:
            fixes.append("🌡️ Stop the machine for 15-20 min to cool down. Check the ventilation system and coolant levels.")
        else:
            fixes.append("🌡️ Reduce workload. Check ventilation and clean air filters. If the trend continues, stop for 10 min.")

    # High vibration scenarios
    if "vibration" in problem_sensors:
        if vib > SENSOR_THRESHOLDS["vibration"]["critical"]:
            fixes.append("📳 Critical vibrations: stop and check bearings, motor shaft alignment, and mounting bolt tightness.")
        else:
            fixes.append("📳 Check mounting bolt tightness and rotating component balance. Lubricate friction points.")

    # High current scenarios
    if "current" in problem_sensors:
        if curr > SENSOR_THRESHOLDS["current"]["critical"]:
            fixes.append("⚡ Excessive current: reduce load immediately. Inspect electrical connections and variable frequency drive.")
        else:
            fixes.append("⚡ Check electrical connections, variable frequency drive, and mechanical load.")

    # RPM drop scenarios
    if "rpm" in problem_sensors:
        if rpm < SENSOR_THRESHOLDS["rpm"]["critical"]:
            fixes.append("🔄 Critical RPM: check drive belt, clutch, and motor. Possible mechanical blockage.")
        else:
            fixes.append("🔄 Speed drop detected: check belt, power supply voltage, and mechanical load.")

    # Combined sensor patterns
    if "temperature" in problem_sensors and "vibration" in problem_sensors:
        fixes.append("⚠️ Worn bearing pattern: replace bearings and check shaft alignment.")
    elif "temperature" in problem_sensors and "current" in problem_sensors:
        fixes.append("⚠️ Possible motor overload: reduce load and check electrical insulation.")
    elif "vibration" in problem_sensors and "rpm" in problem_sensors:
        fixes.append("⚠️ Mechanical imbalance: rebalance rotating components and retighten mountings.")

    # Model-based pattern without clear sensor
    if "model" in problem_sensors or (not fixes and label in ("degradation", "anomaly")):
        if label == "anomaly":
            fixes.append("🤖 AI-detected anomaly: stop the machine for preventive inspection (15 min) and check critical components.")
        else:
            fixes.append("🤖 Progressive degradation: schedule preventive maintenance within the next few hours. Monitor sensor trends.")

    return " | ".join(fixes) if fixes else None


class FaultPredictor:
    """Loads models and runs fault classification + anomaly detection."""

    def __init__(self, model_dir: str | Path = "models"):
        self.model_dir = Path(model_dir)
        self.classifier = joblib.load(self.model_dir / "fault_classifier.joblib")
        self.anomaly_model = joblib.load(self.model_dir / "anomaly_detector.joblib")
        metadata = json.loads((self.model_dir / "model_metadata.json").read_text(encoding="utf-8"))
        self.feature_columns = metadata["feature_columns"]
        self.sampling_interval_s = float(metadata.get("sampling_interval_s", 3.0))

    def predict_window(self, window: pd.DataFrame) -> PredictionResult:
        """Run fault classification and anomaly scoring on a sensor window."""
        features = extract_window_features(window, sampling_interval_s=self.sampling_interval_s)
        row = pd.DataFrame([{col: features.get(col, 0.0) for col in self.feature_columns}])

        # Fault classification
        label = str(self.classifier.predict(row)[0])
        confidence = 0.0
        if hasattr(self.classifier, "predict_proba"):
            classes = list(self.classifier.classes_)
            probabilities = self.classifier.predict_proba(row)[0]
            confidence = float(probabilities[classes.index(label)])

        # Anomaly scoring (Isolation Forest)
        raw_score = float(self.anomaly_model.decision_function(row)[0])
        anomaly_score = max(0.0, min(1.0, 0.5 - raw_score))

        # Estimate time to event based on feature slopes
        estimated_hours, driving_sensors = _estimate_time_to_event(
            features, label, self.sampling_interval_s
        )

        # Generate actionable repair suggestion for technicians
        recommended_fix = _generate_recommended_fix(features, label, driving_sensors)

        return PredictionResult(
            label=label,
            confidence=confidence,
            anomaly_score=anomaly_score,
            features=features,
            estimated_hours_to_event=estimated_hours,
            driving_sensors=driving_sensors,
            recommended_fix=recommended_fix,
        )
