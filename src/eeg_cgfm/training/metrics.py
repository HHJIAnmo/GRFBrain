from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def best_f1_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(probability)
    scores = np.asarray([f1_score(y, probability > value, zero_division=0) for value in candidates])
    return float(candidates[int(scores.argmax())])


def classification_metrics(y: np.ndarray, logits: np.ndarray,
                           threshold: float | None = None) -> dict[str, float]:
    probability = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    threshold = best_f1_threshold(y, probability) if threshold is None else float(threshold)
    pred = probability > threshold
    return {"accuracy": float(accuracy_score(y, pred)),
            "auroc": float(roc_auc_score(y, probability)),
            "auprc": float(average_precision_score(y, probability)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "threshold": threshold, "examples": int(len(y))}
