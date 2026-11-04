#!/usr/bin/env python
"""
SmartFactory AI — MQTT data collector.

Subscribes to SmartFactory MQTT sensor topics, merges per-sensor messages
into unified 4-sensor readings per machine, and appends them to a CSV
file for offline analysis or later model retraining.

Usage:
  python scripts/collect_mqtt_to_csv.py --output data/collected_readings.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartfactory_ai.config import load_config
from smartfactory_ai.schemas import SENSOR_COLUMNS, utc_now_iso, validate_reading
from smartfactory_ai.storage import append_row_csv

FIELDNAMES = ["machine_code", "timestamp", "temperature", "vibration", "current", "rpm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect SmartFactory sensor data to CSV")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output = args.output or cfg.storage.live_readings_csv

    try:
        import paho.mqtt.client as mqtt
        from paho.mqtt.enums import CallbackAPIVersion
    except ImportError as exc:
        raise RuntimeError("Install paho-mqtt first: pip install paho-mqtt") from exc

    # Per-machine partial snapshots: accumulate until all 4 sensors arrive
    partial_snapshots: dict[str, dict] = defaultdict(dict)
    SENSOR_KEY_MAP = {
        "TEMPERATURE": "temperature",
        "VIBRATION": "vibration",
        "CURRENT": "current",
        "RPM": "rpm",
    }
    count = 0

    def on_connect(client, userdata, flags, reason_code, properties=None):
        topic = f"{cfg.mqtt.topic_prefix}/#"
        print(f"Connected (code {reason_code}), collecting from {topic} → {output}")
        client.subscribe(topic)

    def on_message(client, userdata, message):
        nonlocal count
        try:
            topic = message.topic
            payload = json.loads(message.payload.decode("utf-8"))

            parts = topic.split("/")
            if len(parts) < 4:
                return
            machine_code = parts[2]
            sensor_type = parts[3].upper()

            if sensor_type not in SENSOR_KEY_MAP:
                return

            snapshot = partial_snapshots[machine_code]
            snapshot[SENSOR_KEY_MAP[sensor_type]] = float(payload.get("value", 0.0))
            snapshot["timestamp"] = payload.get("timestamp", utc_now_iso())
            snapshot["machine_code"] = machine_code

            if not all(k in snapshot for k in SENSOR_COLUMNS):
                return

            reading = validate_reading(dict(snapshot))
            partial_snapshots[machine_code] = {}
            append_row_csv(output, reading, FIELDNAMES)
            count += 1
            print(f"  [{count}] {machine_code} temp={reading['temperature']:.1f} vib={reading['vibration']:.2f}")

        except Exception as exc:
            print(f"Skipped: {exc}")

    client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2, client_id=f"{cfg.mqtt.client_id}-collector")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(cfg.mqtt.host, cfg.mqtt.port, cfg.mqtt.keepalive)
    print("SmartFactory data collector started. Press Ctrl+C to stop.")
    client.loop_forever()


if __name__ == "__main__":
    main()
