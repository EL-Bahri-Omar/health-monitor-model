"""
SmartFactory AI — Telemetry validation schemas.

Defines the 4-sensor data contract that matches the SmartFactory backend:
  TEMPERATURE (°C), VIBRATION (mm/s), CURRENT (A), RPM (RPM).

Every incoming reading is validated and type-coerced before entering
the feature-extraction pipeline.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# The four sensor columns expected by the SmartFactory AI module.
SENSOR_COLUMNS = ["temperature", "vibration", "current", "rpm"]

REQUIRED_FIELDS = {"machine_code", "timestamp", "temperature", "vibration", "current", "rpm"}

VALID_STATUSES = {"normal", "degradation", "anomaly", "failure", "idle", "unknown"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def validate_reading(payload: dict[str, Any], require_status: bool = False) -> dict[str, Any]:
    """Validate and coerce a unified sensor reading from the SmartFactory platform."""
    missing = REQUIRED_FIELDS.difference(payload)
    if require_status and "status" not in payload:
        missing.add("status")
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")

    cleaned = dict(payload)
    cleaned["machine_code"] = str(cleaned["machine_code"])
    cleaned["timestamp"] = str(cleaned["timestamp"])
    for name in SENSOR_COLUMNS:
        cleaned[name] = float(cleaned[name])
    cleaned["status"] = str(cleaned.get("status", "unknown")).lower()
    if cleaned["status"] not in VALID_STATUSES:
        cleaned["status"] = "unknown"
    return cleaned
