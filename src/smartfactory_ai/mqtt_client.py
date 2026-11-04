"""
SmartFactory AI — MQTT client wrapper.

Connects to the Mosquitto broker and provides publish/subscribe helpers
for the SmartFactory MQTT topic structure:
  factory/machines/{machineCode}/{sensorType}
"""
from __future__ import annotations

import json
from typing import Any, Callable

from smartfactory_ai.config import MqttConfig


def build_client(client_id: str):
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise RuntimeError("Install paho-mqtt: pip install paho-mqtt>=2.0") from exc
    return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)


class SmartFactoryMqttClient:
    """MQTT client tailored to SmartFactory topic conventions."""

    def __init__(self, config: MqttConfig):
        self.config = config
        self.client = build_client(config.client_id)

    def connect(self) -> None:
        self.client.connect(self.config.host, self.config.port, self.config.keepalive)

    def publish_json(self, topic: str, payload: dict[str, Any], qos: int = 0) -> None:
        self.client.publish(topic, json.dumps(payload), qos=qos)

    def subscribe_json(self, topic: str, callback: Callable[[dict[str, Any]], None]) -> None:
        def on_message(client, userdata, message):
            try:
                payload = json.loads(message.payload.decode("utf-8"))
                callback(payload)
            except Exception as exc:
                print(f"Bad MQTT message: {exc}")

        self.client.on_message = on_message
        self.client.subscribe(topic)
        self.client.loop_forever()
