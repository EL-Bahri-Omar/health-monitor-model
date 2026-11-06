#!/usr/bin/env python
"""
SmartFactory AI — Real-time inference subscriber.

Subscribes to the SmartFactory MQTT topics (factory/machines/#), buffers
per-sensor messages into unified 4-sensor snapshots per machine, extracts
features from sliding windows, and runs fault classification + anomaly
detection.

Alerts are published back to MQTT and optionally forwarded to the
SmartFactory Spring Boot backend via REST API.

Usage:
  python scripts/run_inference_subscriber.py --config config/config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import requests

from smartfactory_ai.config import load_config
from smartfactory_ai.predict import FaultPredictor
from smartfactory_ai.schemas import SENSOR_COLUMNS, utc_now_iso, validate_reading
from smartfactory_ai.storage import append_row_csv

READING_FIELDS = ["machine_code", "timestamp", "temperature", "vibration", "current", "rpm", "status"]

ALERT_FIELDS = [
    "timestamp",
    "machine_code",
    "predicted_status",
    "confidence",
    "anomaly_score",
    "temperature",
    "vibration_rms",
    "reason",
]

# Severity mapping to SmartFactory backend conventions
SEVERITY_MAP = {
    "normal": "INFO",
    "degradation": "WARNING",
    "anomaly": "HIGH",
    "failure": "CRITICAL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmartFactory real-time fault inference")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--post-backend", action="store_true", help="Forward predictions to Spring Boot backend")
    return parser.parse_args()


def alert_reason(features: dict, result, thresholds) -> str | None:
    """Build a comma-separated reason string for why an alert was raised."""
    reasons: list[str] = []
    if result.label != "normal" and result.confidence >= thresholds.fault_confidence_threshold:
        reasons.append(f"fault_prediction:{result.label}")
    if result.anomaly_score >= thresholds.anomaly_score_threshold:
        reasons.append("anomaly_score")
    if features.get("temperature_mean", 0.0) >= thresholds.temperature_alert_c:
        reasons.append("high_temperature")
    if features.get("vibration_rms", 0.0) >= thresholds.vibration_rms_alert:
        reasons.append("high_vibration")
    return ",".join(reasons) if reasons else None


def post_to_backend(cfg, machine_code: str, result, features: dict) -> None:
    """Forward prediction result to Spring Boot backend via REST API."""
    payload = {
        "machineCode": machine_code,
        "predictedStatus": result.label,
        "confidence": round(result.confidence, 4),
        "anomalyScore": round(result.anomaly_score, 4),
        "severity": SEVERITY_MAP.get(result.label, "INFO"),
        "temperatureMean": round(features.get("temperature_mean", 0.0), 2),
        "vibrationRms": round(features.get("vibration_rms", 0.0), 4),
        "currentMean": round(features.get("current_mean", 0.0), 2),
        "rpmMean": round(features.get("rpm_mean", 0.0), 1),
        "timestamp": utc_now_iso(),
        # Sprint 9 — predictive time estimation
        "estimatedHoursToEvent": result.estimated_hours_to_event,
        "drivingSensors": result.driving_sensors,
        # Sprint 10 — actionable repair suggestion for technicians
        "recommendedFix": result.recommended_fix,
    }
    # Clé API interne pour l'authentification machine-to-machine
    api_key = cfg.backend.api_key if hasattr(cfg.backend, "api_key") and cfg.backend.api_key else "smartfactory-ai-internal-2024"
    headers = {
        "Content-Type": "application/json",
        "X-AI-Api-Key": api_key,
    }
    try:
        url = f"{cfg.backend.api_url}/api/anomalies/ai-prediction"
        resp = requests.post(url, json=payload, headers=headers, timeout=5)
        if resp.ok:
            data = resp.json()
            print(f"  → Backend accepted prediction for {machine_code} (anomaly={data.get('anomalyId', 'N/A')})")
        else:
            print(f"  → Backend rejected: {resp.status_code} {resp.text[:200]}")
    except requests.RequestException as exc:
        print(f"  → Backend unreachable: {exc}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    predictor = FaultPredictor(args.model_dir)

    try:
        import paho.mqtt.client as mqtt
        from paho.mqtt.enums import CallbackAPIVersion
    except ImportError as exc:
        raise RuntimeError("Install paho-mqtt first: pip install paho-mqtt") from exc

    # Per-machine buffers: each buffer holds unified 4-sensor readings
    window_buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=cfg.features.window_size))

    # Per-machine step counter: predictions run every step_size readings
    # after the first full window, so the model re-evaluates with fresh data
    step_counters: dict[str, int] = defaultdict(int)

    # Per-machine partial snapshot: accumulates per-sensor MQTT messages
    # until all 4 sensors are received, then merges into one reading
    partial_snapshots: dict[str, dict] = defaultdict(dict)

    SENSOR_TYPES = {"TEMPERATURE", "VIBRATION", "CURRENT", "RPM"}
    SENSOR_KEY_MAP = {
        "TEMPERATURE": "temperature",
        "VIBRATION": "vibration",
        "CURRENT": "current",
        "RPM": "rpm",
    }

    def on_connect(client, userdata, flags, reason_code, properties=None):
        topic = f"{cfg.mqtt.topic_prefix}/#"
        print(f"Connected to MQTT broker (code {reason_code}), subscribing to {topic}")
        client.subscribe(topic)

    def on_message(client, userdata, message):
        try:
            topic = message.topic
            payload = json.loads(message.payload.decode("utf-8"))

            # Parse topic: factory/machines/{machineCode}/{sensorType}
            parts = topic.split("/")
            if len(parts) < 4:
                return
            machine_code = parts[2]
            sensor_type = parts[3].upper()

            if sensor_type not in SENSOR_TYPES:
                return

            value = float(payload.get("value", 0.0))
            timestamp = payload.get("timestamp", utc_now_iso())

            # Accumulate into partial snapshot
            snapshot = partial_snapshots[machine_code]
            snapshot[SENSOR_KEY_MAP[sensor_type]] = value
            snapshot["timestamp"] = timestamp
            snapshot["machine_code"] = machine_code

            # Check if all 4 sensors are present
            if not all(k in snapshot for k in SENSOR_COLUMNS):
                return

            # Build unified reading and clear partial snapshot
            reading = validate_reading(dict(snapshot))
            partial_snapshots[machine_code] = {}

            append_row_csv(cfg.storage.live_readings_csv, reading, READING_FIELDS)
            window_buffers[machine_code].append(reading)

            buf_len = len(window_buffers[machine_code])
            if buf_len < cfg.features.window_size:
                # Print progress while filling the initial window
                if buf_len % 5 == 0:
                    print(f"  [{machine_code}] buffering {buf_len}/{cfg.features.window_size} readings...")
                return

            # Run prediction every step_size readings after the first full window.
            # This gives continuous early detection instead of waiting for the
            # buffer to completely refill.
            step_counters[machine_code] += 1
            if step_counters[machine_code] % cfg.features.step_size != 0:
                return

            # Run prediction on full window
            frame = pd.DataFrame(list(window_buffers[machine_code]))
            result = predictor.predict_window(frame)
            reason = alert_reason(result.features, result, cfg.thresholds)

            status_str = f"{result.label} (conf={result.confidence:.2f}, anomaly={result.anomaly_score:.2f})"
            if result.recommended_fix:
                status_str += f"\n  FIX: {result.recommended_fix[:80]}"
            print(f"[{machine_code}] {status_str}")

            # Always post to backend so the frontend gets real-time predictions
            # (even normal ones update the live prediction state).
            if args.post_backend:
                post_to_backend(cfg, machine_code, result, result.features)

            if reason:
                alert = {
                    "timestamp": utc_now_iso(),
                    "machine_code": machine_code,
                    "predicted_status": result.label,
                    "confidence": round(result.confidence, 4),
                    "anomaly_score": round(result.anomaly_score, 4),
                    "temperature": round(result.features.get("temperature_mean", 0.0), 2),
                    "vibration_rms": round(result.features.get("vibration_rms", 0.0), 4),
                    "reason": reason,
                }
                append_row_csv(cfg.storage.live_alerts_csv, alert, ALERT_FIELDS)
                client.publish(cfg.mqtt.alert_topic, json.dumps(alert), qos=0)
                print(f"  ALERT: {reason}")

                if args.post_backend:
                    post_to_backend(cfg, machine_code, result, result.features)

        except Exception as exc:
            print(f"Inference error: {exc}")

    client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2, client_id=f"{cfg.mqtt.client_id}-inference")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(cfg.mqtt.host, cfg.mqtt.port, cfg.mqtt.keepalive)
    print("SmartFactory AI inference subscriber started. Press Ctrl+C to stop.")
    client.loop_forever()


if __name__ == "__main__":
    main()
