# SmartFactory AI — Data Schema

## Sensor Types

The SmartFactory platform uses 4 scalar sensor types, matching the backend `SensorType` enum:

| Sensor Type | Field Name | Unit | Description |
|---|---|---|---|
| `TEMPERATURE` | `temperature` | °C | Motor/component surface temperature |
| `VIBRATION` | `vibration` | mm/s | Vibration velocity (RMS) |
| `CURRENT` | `current` | A | Electrical current draw |
| `RPM` | `rpm` | RPM | Rotational speed |

## MQTT Message Format

Each sensor publishes separately to its own MQTT topic:

**Topic pattern:** `factory/machines/{machineCode}/{sensorType}`

**Payload:**
```json
{
  "machineCode": "CNC-024",
  "machineId": "CNC-024",
  "sensorType": "TEMPERATURE",
  "value": 45.2,
  "unit": "°C",
  "timestamp": "2026-05-13T08:00:00.000Z"
}
```

## Unified Reading (Internal)

The AI module merges 4 per-sensor messages into a single unified reading:

```json
{
  "machine_code": "CNC-024",
  "timestamp": "2026-05-13T08:00:00.000Z",
  "temperature": 45.2,
  "vibration": 2.1,
  "current": 8.0,
  "rpm": 1500.0,
  "status": "normal"
}
```

The `status` field is present only in training data; during inference, it is predicted by the model.

## Training Data CSV Format

```csv
machine_code,timestamp,temperature,vibration,current,rpm,status
CNC-001,2026-05-13T08:00:00.000Z,45.2,2.1,8.0,1500,normal
CNC-001,2026-05-13T08:00:03.000Z,62.1,4.9,10.2,1448,degradation
CNC-001,2026-05-13T08:00:06.000Z,86.3,9.4,14.1,1295,anomaly
CNC-001,2026-05-13T08:00:09.000Z,94.8,12.3,16.2,198,failure
```

## Status Labels

| Status | Description | Typical Sensor Ranges |
|---|---|---|
| `normal` | Operating within spec | Temp 40–50°C, Vib 1.5–3.0, Curr 7–9A, RPM 1450–1550 |
| `degradation` | Early wear signs | Temp 55–70°C, Vib 3.5–6.0, Curr 9–11A, RPM 1400–1480 |
| `anomaly` | Abnormal patterns | Temp 75–90°C, Vib 7–11, Curr 12–15A, RPM 1200–1400 |
| `failure` | Critical fault | Temp 90–100+°C, Vib 10–15, Curr 15–18A, RPM <300 |

## Feature Vector

The feature extraction pipeline converts a sliding window of raw readings into 22 numeric features. See `docs/architecture.md` for the full list.
