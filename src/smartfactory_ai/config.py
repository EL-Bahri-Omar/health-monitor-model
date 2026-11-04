"""
SmartFactory AI — Configuration loader.

Reads the YAML config that defines MQTT broker settings, feature-extraction
parameters (window size, step), alert thresholds, and simulation defaults.
Uses yaml.safe_load (safe against code-injection in YAML).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MqttConfig:
    host: str = "localhost"
    port: int = 1883
    keepalive: int = 60
    topic_prefix: str = "factory/machines"
    alert_topic: str = "factory/ai/alerts"
    client_id: str = "smartfactory-ai-service"


@dataclass(frozen=True)
class FeatureConfig:
    """Sliding-window parameters for feature extraction."""
    sampling_interval_s: float = 3.0
    window_size: int = 64
    step_size: int = 32


@dataclass(frozen=True)
class ThresholdConfig:
    fault_confidence_threshold: float = 0.72
    anomaly_score_threshold: float = 0.62
    temperature_alert_c: float = 75.0
    vibration_rms_alert: float = 8.5


@dataclass(frozen=True)
class StorageConfig:
    live_readings_csv: str = "data/live_readings.csv"
    live_alerts_csv: str = "data/live_alerts.csv"


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 42
    drop_rate: float = 0.05
    default_rows: int = 12000


@dataclass(frozen=True)
class BackendConfig:
    api_url: str = "http://localhost:8080"
    api_key: str = "smartfactory-ai-internal-2024"
    api_email: str = ""
    api_password: str = ""


@dataclass(frozen=True)
class AppConfig:
    mqtt: MqttConfig = MqttConfig()
    features: FeatureConfig = FeatureConfig()
    thresholds: ThresholdConfig = ThresholdConfig()
    storage: StorageConfig = StorageConfig()
    simulation: SimulationConfig = SimulationConfig()
    backend: BackendConfig = BackendConfig()


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{name}' must be a mapping")
    return section


def load_config(path: str | Path = "config/config.yaml") -> AppConfig:
    """Load configuration from YAML file. Missing keys get sane defaults."""
    config_path = Path(path)
    data: dict[str, Any] = {}
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return AppConfig(
        mqtt=MqttConfig(**_section(data, "mqtt")),
        features=FeatureConfig(**_section(data, "features")),
        thresholds=ThresholdConfig(**_section(data, "thresholds")),
        storage=StorageConfig(**_section(data, "storage")),
        simulation=SimulationConfig(**_section(data, "simulation")),
        backend=BackendConfig(**_section(data, "backend")),
    )
