# SmartFactory AI — Architecture

## System Overview

The SmartFactory AI module is a **Python sidecar service** that runs alongside the Spring Boot backend. It subscribes to the same MQTT broker, processes sensor data through ML models, and feeds predictions back into the platform.

```
┌──────────────────┐     MQTT      ┌─────────────────────┐
│  IoT Simulator   │──────────────▶│   Mosquitto Broker   │
│  (iot-simulator)  │              │   (port 1883)        │
└──────────────────┘              └──────────┬────────────┘
                                             │
                          ┌──────────────────┼──────────────────┐
                          │                  │                  │
                          ▼                  ▼                  ▼
                 ┌─────────────┐   ┌──────────────┐   ┌──────────────┐
                 │ Spring Boot │   │ SmartFactory  │   │    MQTT      │
                 │   Backend   │   │  AI Service   │   │  Collector   │
                 │ (port 8080) │   │  (inference)  │   │  (CSV log)   │
                 └─────────────┘   └──────┬───────┘   └──────────────┘
                       ▲                  │
                       │     REST API     │
                       └──────────────────┘
                                POST /api/anomalies/ai-prediction
```

## Data Flow

### 1. Sensor Ingestion

The IoT simulator publishes per-sensor MQTT messages:
- `factory/machines/{machineCode}/TEMPERATURE`
- `factory/machines/{machineCode}/VIBRATION`
- `factory/machines/{machineCode}/CURRENT`
- `factory/machines/{machineCode}/RPM`

### 2. Message Aggregation

The inference subscriber buffers per-sensor messages into **unified 4-sensor snapshots** per machine. A snapshot is complete when all 4 sensors have reported.

### 3. Sliding Window

Completed snapshots are added to a per-machine sliding window buffer (default: 64 readings, step: 32). When the buffer is full, feature extraction runs.

### 4. Feature Extraction

22 statistical features are computed from each window:

| Sensor | Features |
|---|---|
| Temperature | mean, std, max, min, slope, range |
| Vibration | mean, std, rms, max, min, peak-to-peak, crest factor, skew, kurtosis, zero crossings |
| Current | mean, std, max, slope |
| RPM | mean, std, min, range |

### 5. Prediction

Two models run on the feature vector:
- **RandomForest** → fault class (`normal`/`degradation`/`anomaly`/`failure`) + confidence
- **IsolationForest** → anomaly score (0.0–1.0)

### 6. Alert Generation

If the prediction meets threshold criteria (fault confidence ≥ 0.72, anomaly score ≥ 0.62, or sensor limits exceeded), an alert is:
1. Saved to `data/live_alerts.csv`
2. Published to MQTT topic `factory/ai/alerts`
3. Optionally posted to the Spring Boot backend REST API

## Model Architecture

### RandomForest Classifier
- 220 estimators, max depth 10
- Balanced class weights for handling class imbalance
- Trained on all 4 status classes

### IsolationForest Anomaly Detector
- 180 estimators, 8% contamination rate
- Trained exclusively on "normal" data
- Decision function mapped to 0.0–1.0 anomaly score

## File Organisation

```
src/smartfactory_ai/
├── config.py       # Configuration dataclasses + YAML loader
├── schemas.py      # Input validation and type coercion
├── features.py     # Feature extraction engine
├── predict.py      # Model loading and inference
├── mqtt_client.py  # MQTT connection helpers
└── storage.py      # CSV append utilities
```
