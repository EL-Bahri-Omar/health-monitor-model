# SmartFactory AI — Training Data Generation

## Overview

The `generate_training_data.py` script creates synthetic sensor data that mimics real SmartFactory telemetry across all 4 health statuses. This data is used to train the ML models without requiring real hardware.

## Usage

```bash
python scripts/generate_training_data.py --output data/training_data.csv --rows 12000
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--output` | `data/training_data.csv` | Output CSV file path |
| `--rows` | 12000 (from config) | Number of readings to generate |
| `--machine-code` | `CNC-001` | Machine identifier |
| `--seed` | 42 (from config) | Random seed for reproducibility |
| `--drop-rate` | 0.05 (from config) | Simulated packet loss rate |
| `--config` | `config/config.yaml` | Config file for defaults |

## Data Generation Strategy

### Scenario Cycling

Readings cycle through status blocks (1500 readings each by default):

```
normal → normal → degradation → normal → anomaly → normal → failure → normal → repeat
```

This creates realistic transitions with ample "normal" baseline data.

### Sensor Targets

Each status has predefined target values with Gaussian noise:

| Status | Temperature (°C) | Vibration (mm/s) | Current (A) | RPM |
|---|---|---|---|---|
| normal | 45.0 ± 0.4 | 2.1 ± 0.15 | 8.0 ± 0.2 | 1500 ± 6 |
| degradation | 62.0 ± 0.6 | 4.8 ± 0.22 | 10.0 ± 0.3 | 1450 ± 9 |
| anomaly | 85.0 ± 0.8 | 9.2 ± 0.30 | 14.0 ± 0.4 | 1300 ± 12 |
| failure | 95.0 ± 1.0 | 12.0 ± 0.45 | 16.0 ± 0.6 | 200 ± 18 |

### Realism Enhancements

- **Sinusoidal drift** on temperature, RPM, and current to simulate environmental variation
- **Gaussian noise** on all sensors matching real-world measurement uncertainty
- **Physical clamping** — no negative values for any sensor
- **Packet drop simulation** — configurable rate to simulate MQTT delivery losses
- **3-second intervals** matching the SmartFactory IoT simulator

## Relationship to IoT Simulator

The SmartFactory project has two data sources:

| | `generate_training_data.py` | `iot-simulator/` |
|---|---|---|
| Purpose | Generate CSV for model training | Live MQTT publishing for integration testing |
| Output | CSV file | MQTT messages |
| Sensors | Same 4 types | Same 4 types |
| Scenarios | Same status labels | Same status labels |
| Targets | Matched values | Matched values |

Both use the same sensor ranges and status definitions, ensuring training data matches live telemetry.
