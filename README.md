# SmartFactory AI — Predictive Maintenance Module

Machine learning module for **real-time fault classification** and **anomaly detection** on industrial sensor data. Part of the SmartFactory IoT platform.

---

## What It Does

This module listens to live sensor data from industrial machines via MQTT, runs two machine learning models on sliding windows of readings, and produces health predictions that flow into the SmartFactory backend for alerting and visualization.

### Prediction Pipeline

```
MQTT sensor data
      │
      ▼
┌─────────────────┐
│ Per-sensor MQTT  │  factory/machines/{code}/TEMPERATURE
│ messages arrive  │  factory/machines/{code}/VIBRATION
│ (4 per machine)  │  factory/machines/{code}/CURRENT
│                  │  factory/machines/{code}/RPM
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Snapshot buffer  │  Waits for all 4 sensors before creating
│                  │  a unified reading row
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Sliding window   │  64-reading buffer per machine
│ (step = 32)      │  Runs feature extraction when full
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature engine   │  Extracts 22 statistical features
│ (features.py)    │  from the window
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│              Two ML Models                   │
│                                              │
│  RandomForest Classifier                     │
│  → Label: normal | degradation | anomaly |   │
│           failure                            │
│  → Confidence: 0.0 – 1.0                    │
│                                              │
│  IsolationForest Detector                    │
│  → Anomaly Score: 0.0 – 1.0                 │
└────────┬─────────────────────────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────────┐
│ MQTT Alert      │     │ REST API POST        │
│ factory/ai/     │     │ /api/anomalies/      │
│ alerts          │     │ ai-prediction        │
└─────────────────┘     │ (Spring Boot)        │
                        └──────────────────────┘
```

### Health Status Classes

| Status | Meaning | Backend Severity |
|---|---|---|
| `normal` | Machine operating within expected parameters | INFO |
| `degradation` | Early signs of wear, drift, or increased noise | WARNING |
| `anomaly` | Abnormal sensor patterns — investigation needed | HIGH |
| `failure` | Critical fault requiring immediate intervention | CRITICAL |

---

## Project Structure

```
health-monitor-model/
├── src/smartfactory_ai/           # Core Python package
│   ├── __init__.py
│   ├── config.py                  # YAML configuration loader (dataclasses)
│   ├── schemas.py                 # 4-sensor data validation & type coercion
│   ├── features.py                # Feature extraction engine (22 features)
│   ├── predict.py                 # Model loading, inference, result container
│   ├── mqtt_client.py             # MQTT connection helpers
│   └── storage.py                 # Thread-safe CSV append utilities
│
├── scripts/
│   ├── generate_training_data.py  # Synthetic training data generator
│   ├── train_models.py            # Model training pipeline (RF + IF)
│   ├── run_inference_subscriber.py# Real-time MQTT → prediction → backend
│   ├── collect_mqtt_to_csv.py     # MQTT data collector for CSV logging
│   └── reliability_report.py      # Data quality analysis reporter
│
├── config/
│   └── config.yaml                # All settings: MQTT, features, thresholds, backend
│
├── models/                        # Trained model artifacts
│   ├── fault_classifier.joblib    # RandomForest (259 KB)
│   ├── anomaly_detector.joblib    # IsolationForest (1.2 MB)
│   ├── model_metadata.json        # Feature columns & training config
│   └── metrics.json               # Accuracy, F1, confusion matrix
│
├── data/                          # Training and collected data
│   ├── training_data.csv          # Generated training data (11,409 rows)
│   ├── live_readings.csv          # Runtime: buffered sensor readings
│   └── live_alerts.csv            # Runtime: generated alerts
│
├── reports/
│   ├── model_report.md            # Human-readable training report
│   └── training_feature_windows.csv
│
├── app/
│   └── dashboard.py               # Streamlit monitoring dashboard (optional)
│
├── tests/                         # Unit tests (10 tests, pytest)
│   ├── test_features.py           # Feature extraction tests
│   └── test_schema.py             # Schema validation tests
│
├── docs/
│   ├── architecture.md            # System architecture & data flow
│   ├── data_schema.md             # Sensor data schema reference
│   ├── simulation.md              # Training data simulation docs
│   └── security_setup_log.md      # Security audit & setup log
│
├── requirements.txt               # Python dependencies (version ranges)
├── pyproject.toml                 # Package metadata & build config
├── NOTICE                         # Attribution notice
└── .gitignore
```

---

## How It Works — In Detail

### 1. Sensor Ingestion

The inference subscriber connects to the MQTT broker and subscribes to:

```
factory/machines/#
```

Each machine publishes 4 separate MQTT messages (one per sensor) every 3 seconds:

| Topic | Sensor | Unit | Payload Example |
|---|---|---|---|
| `factory/machines/CNC-024/TEMPERATURE` | Temperature | °C | `{"machineId":"CNC-024","sensorType":"TEMPERATURE","value":45.2,"unit":"°C","timestamp":"..."}` |
| `factory/machines/CNC-024/VIBRATION` | Vibration | mm/s | `{"machineId":"CNC-024","sensorType":"VIBRATION","value":2.1,"unit":"mm/s","timestamp":"..."}` |
| `factory/machines/CNC-024/CURRENT` | Current | A | `{"machineId":"CNC-024","sensorType":"CURRENT","value":8.0,"unit":"A","timestamp":"..."}` |
| `factory/machines/CNC-024/RPM` | RPM | RPM | `{"machineId":"CNC-024","sensorType":"RPM","value":1500,"unit":"RPM","timestamp":"..."}` |

### 2. Snapshot Aggregation

Since MQTT delivers one sensor at a time, the subscriber buffers partial readings per machine. When all 4 sensors have reported, it merges them into a single unified row:

```python
{ "machine_code": "CNC-024", "temperature": 45.2, "vibration": 2.1,
  "current": 8.0, "rpm": 1500, "timestamp": "2026-10-30T10:15:00Z" }
```

The `schemas.py` module validates types and coerces values (e.g. string → float).

### 3. Sliding Window Buffer

Each machine maintains a `deque(maxlen=64)` buffer. Prediction only runs when the buffer is full (64 complete readings ≈ 3.2 minutes of data at 3-second intervals).

### 4. Feature Extraction (22 Features)

The `features.py` module computes statistical features from each 64-reading window:

| Sensor | Features | Count |
|---|---|---|
| **Temperature** | mean, std, max, min, slope, range | 6 |
| **Vibration** | mean, std, rms, max, min, peak-to-peak, crest factor, skew, kurtosis, zero crossings | 10 |
| **Current** | mean, std, max, slope | 4 |
| **RPM** | mean, std, min, range | 4 → **22 total** |

Vibration features are more numerous because vibration signatures are the most informative indicator of mechanical health (bearing wear, imbalance, misalignment).

### 5. Dual-Model Prediction

Two scikit-learn models run on the 22-feature vector:

**RandomForest Classifier (fault_classifier.joblib)**
- 220 estimators, max depth 10, balanced class weights
- Outputs: predicted class (`normal`/`degradation`/`anomaly`/`failure`) + confidence probability
- Trained on all 4 status classes
- Current performance: **96.63% accuracy, 95.18% F1 score**

**IsolationForest Anomaly Detector (anomaly_detector.joblib)**
- 180 estimators, 8% contamination rate
- Trained exclusively on "normal" data
- Raw decision function mapped to a 0.0–1.0 anomaly score
- High score = more anomalous

### 6. Alert Logic

An alert is triggered when ANY of these conditions is met:
- Fault confidence ≥ 0.72 AND predicted status ≠ normal
- Anomaly score ≥ 0.62
- Temperature mean ≥ 75.0°C
- Vibration RMS ≥ 8.5 mm/s

Alerts are:
1. Saved to `data/live_alerts.csv`
2. Published to MQTT topic `factory/ai/alerts`
3. **If `--post-backend` is enabled:** POSTed to `http://localhost:8080/api/anomalies/ai-prediction` with API key authentication

### 7. Backend Integration

When `--post-backend` is used, predictions are forwarded via REST:

```
POST /api/anomalies/ai-prediction
Header: X-AI-Api-Key: smartfactory-ai-internal-2024
Content-Type: application/json

{
  "machineCode": "CNC-024",
  "predictedStatus": "degradation",
  "confidence": 0.8542,
  "anomalyScore": 0.3201,
  "severity": "WARNING",
  "temperatureMean": 62.5,
  "vibrationRms": 4.23,
  "currentMean": 10.1,
  "rpmMean": 1450.0,
  "timestamp": "2026-10-30T10:15:00Z"
}
```

The backend then:
- Creates an `Anomaly` document with `thresholdType: "AI_MODEL"`
- Creates a linked `Alert` document
- Broadcasts via WebSocket (STOMP) to the frontend
- Triggers FCM/in-app notifications

---

## Setup & Usage

### 1. Create Virtual Environment

```powershell
cd health-monitor-model
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Generate Training Data & Train Models

```bash
python scripts/generate_training_data.py --output data/training_data.csv --rows 12000
python scripts/train_models.py --input data/training_data.csv
```

Output:
- `models/fault_classifier.joblib` — RandomForest classifier
- `models/anomaly_detector.joblib` — IsolationForest detector
- `models/model_metadata.json` — feature column metadata
- `models/metrics.json` — accuracy, F1, confusion matrix
- `reports/model_report.md` — human-readable report

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

Expected: 10/10 passed.

### 5. Start Real-Time Inference

```bash
# Basic: prints predictions to console, publishes alerts to MQTT
python scripts/run_inference_subscriber.py --config config/config.yaml

# With backend integration: also POSTs to Spring Boot
python scripts/run_inference_subscriber.py --config config/config.yaml --post-backend
```

---

## Configuration

All settings are in `config/config.yaml`:

```yaml
mqtt:
  host: localhost              # MQTT broker address
  port: 1883                   # MQTT broker port
  keepalive: 60
  topic_prefix: "factory/machines"
  alert_topic: "factory/ai/alerts"
  client_id: "smartfactory-ai-service"

features:
  sampling_interval_s: 3.0     # Expected interval between readings
  window_size: 64              # Readings per analysis window
  step_size: 32                # Step between windows (overlap = 32)

thresholds:
  fault_confidence_threshold: 0.72   # Min confidence to trigger alert
  anomaly_score_threshold: 0.62      # Min anomaly score to trigger alert
  temperature_alert_c: 75.0          # Temperature alert limit (°C)
  vibration_rms_alert: 8.5           # Vibration RMS alert limit (mm/s)

storage:
  live_readings_csv: "data/live_readings.csv"
  live_alerts_csv: "data/live_alerts.csv"

backend:
  api_url: "http://localhost:8080"
  api_key: "smartfactory-ai-internal-2024"
```

---

## Sensor Reference

| Sensor | Unit | Normal Range | Degradation | Anomaly | Failure |
|---|---|---|---|---|---|
| Temperature | °C | 40–50 | 55–65 | 80–90 | 90–100+ |
| Vibration | mm/s | 1.5–3.0 | 3.5–6.0 | 8.0–12.0 | 10.0–15.0 |
| Current | A | 7.0–9.0 | 9.0–11.0 | 12.0–16.0 | 14.0–18.0 |
| RPM | RPM | 1450–1550 | 1350–1450 | 1000–1350 | 100–1000 |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| numpy | ≥2.1.0 | Numerical computation |
| pandas | ≥2.2.0 | DataFrame operations |
| scikit-learn | ≥1.6.0 | ML models (RandomForest, IsolationForest) |
| joblib | ≥1.4.0 | Model serialization |
| PyYAML | ≥6.0 | Configuration loading |
| paho-mqtt | ≥2.1.0 | MQTT client |
| requests | ≥2.32.0 | HTTP POST to backend |

Python 3.10+ required.
