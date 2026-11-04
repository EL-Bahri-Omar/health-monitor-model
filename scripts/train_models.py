#!/usr/bin/env python
"""
SmartFactory AI — Model training script.

Trains a RandomForest fault classifier and an IsolationForest anomaly detector
from the SmartFactory 4-sensor training data (temperature, vibration, current, rpm).

Usage:
  python scripts/train_models.py --input data/training_data.csv --model-dir models
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from smartfactory_ai.config import load_config
from smartfactory_ai.features import FEATURE_VERSION, feature_columns, make_feature_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SmartFactory AI models")
    parser.add_argument("--input", default="data/training_data.csv")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    return parser.parse_args()


def train_models(raw: pd.DataFrame, model_dir: Path, reports_dir: Path, cfg) -> dict[str, object]:
    """Train the fault classifier and anomaly detector."""
    features = make_feature_table(
        raw,
        window_size=cfg.features.window_size,
        step_size=cfg.features.step_size,
        sampling_interval_s=cfg.features.sampling_interval_s,
    )
    if "status" not in features.columns:
        raise ValueError("Training data must include a status column")

    # Use only labelled rows
    usable = features[features["status"].isin(["normal", "degradation", "anomaly", "failure"])]
    columns = feature_columns(usable)
    x = usable[columns].fillna(0.0)
    y = usable["status"].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y,
    )

    # --- Fault Classifier (RandomForest) ---
    classifier = RandomForestClassifier(
        n_estimators=220,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)

    # --- Anomaly Detector (IsolationForest) ---
    normal_x = x_train[y_train == "normal"]
    anomaly_model = IsolationForest(
        n_estimators=180,
        contamination=0.08,
        random_state=42,
    )
    anomaly_model.fit(normal_x if len(normal_x) >= 20 else x_train)

    # --- Save models ---
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, model_dir / "fault_classifier.joblib")
    joblib.dump(anomaly_model, model_dir / "anomaly_detector.joblib")

    # --- Save metadata ---
    labels = sorted(y.unique().tolist())
    accuracy = float(accuracy_score(y_test, predictions))
    f1_macro = float(f1_score(y_test, predictions, average="macro"))
    report = classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=labels).tolist()

    metadata = {
        "project": "smartfactory-ai",
        "feature_version": FEATURE_VERSION,
        "sampling_interval_s": cfg.features.sampling_interval_s,
        "window_size": cfg.features.window_size,
        "step_size": cfg.features.step_size,
        "feature_columns": columns,
        "labels": labels,
        "source_data": str(Path("data") / Path("training_data.csv").name),
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": matrix,
        "test_windows": int(len(x_test)),
        "train_windows": int(len(x_train)),
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (reports_dir / "model_report.md").write_text(_build_report(metrics), encoding="utf-8")
    features.to_csv(reports_dir / "training_feature_windows.csv", index=False)

    return metrics


def _build_report(metrics: dict[str, object]) -> str:
    labels = metrics["labels"]
    matrix = metrics["confusion_matrix"]
    report = metrics["classification_report"]
    lines = [
        "# SmartFactory AI — Model Training Report",
        "",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Macro F1: {metrics['f1_macro']:.4f}",
        f"Train windows: {metrics['train_windows']}",
        f"Test windows: {metrics['test_windows']}",
        "",
        "## Per-class metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in labels:
        row = report[label]
        lines.append(f"| {label} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1-score']:.3f} | {int(row['support'])} |")
    lines.extend(["", "## Confusion matrix", ""])
    header = "| Actual \\ Predicted | " + " | ".join(labels) + " |"
    sep = "|---" + "|---:" * len(labels) + "|"
    lines.extend([header, sep])
    for label, row in zip(labels, matrix):
        lines.append("| " + label + " | " + " | ".join(str(v) for v in row) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    raw = pd.read_csv(args.input)
    model_dir = Path(args.model_dir)
    reports_dir = Path(args.reports_dir)
    metrics = train_models(raw, model_dir, reports_dir, cfg)
    print(json.dumps({"accuracy": metrics["accuracy"], "f1_macro": metrics["f1_macro"]}, indent=2))


if __name__ == "__main__":
    main()
