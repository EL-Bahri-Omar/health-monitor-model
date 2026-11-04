# SmartFactory AI — Model Training Report

Accuracy: 0.9663
Macro F1: 0.9518
Train windows: 266
Test windows: 89

## Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| anomaly | 0.846 | 1.000 | 0.917 | 11 |
| degradation | 0.917 | 1.000 | 0.957 | 11 |
| failure | 1.000 | 0.909 | 0.952 | 11 |
| normal | 1.000 | 0.964 | 0.982 | 56 |

## Confusion matrix

| Actual \ Predicted | anomaly | degradation | failure | normal |
|---|---:|---:|---:|---:|
| anomaly | 11 | 0 | 0 | 0 |
| degradation | 0 | 11 | 0 | 0 |
| failure | 1 | 0 | 10 | 0 |
| normal | 1 | 1 | 0 | 54 |
