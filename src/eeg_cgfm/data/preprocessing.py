from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def load_signal(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "signal" not in data:
                raise ValueError(f"NPZ lacks 'signal': {path}")
            value = data["signal"]
    else:
        raise ValueError("Public preprocessing accepts aligned .npy/.npz clips")
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (19, 2400) or not np.isfinite(value).all():
        raise ValueError(f"Signal must be finite (19,2400), got {value.shape}")
    return value


def log_fft_windows(signal: np.ndarray, sample_rate: int = 200) -> np.ndarray:
    if signal.shape != (19, 12 * sample_rate):
        raise ValueError("Expected a 19-channel twelve-second signal")
    windows = signal.reshape(19, 12, sample_rate).transpose(1, 0, 2)
    spectrum = np.fft.fft(windows, n=sample_rate, axis=-1)[..., : sample_rate // 2]
    return np.log(np.maximum(np.abs(spectrum), 1e-8)).astype(np.float32)


def fit_normalization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 4 or features.shape[-2:] != (19, 100):
        raise ValueError("Training features must be (samples,12,19,100)")
    mean = features.mean(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    std = features.std(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-6)


def absolute_spearman_topk(node_features: np.ndarray, top_k: int = 3) -> np.ndarray:
    if node_features.ndim != 2 or node_features.shape[0] != 19:
        raise ValueError("Node features must have shape (19,D)")
    ranks = np.stack([rankdata(row, method="average") for row in node_features])
    corr = np.nan_to_num(np.corrcoef(ranks), nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr).astype(np.float32)
    np.fill_diagonal(corr, 0.0)
    order = np.argsort(-corr, axis=1, kind="stable")[:, :top_k]
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.arange(19)[:, None], order] = True
    return np.where(mask, corr, 0.0).astype(np.float32)


def graph_sequence(features: np.ndarray, top_k: int = 3) -> np.ndarray:
    if features.shape != (12, 19, 100):
        raise ValueError("Features must be (12,19,100)")
    return np.stack([absolute_spearman_topk(x, top_k) for x in features])


def upper_edges(graph: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (graph + graph.T)
    rows, cols = np.triu_indices(graph.shape[0], k=1)
    return symmetric[rows, cols].astype(np.float32)


def build_sample(signal: np.ndarray, label: int, sample_id: str,
                 mean: np.ndarray, std: np.ndarray, top_k: int = 3) -> dict[str, np.ndarray]:
    if label not in (0, 1) or not sample_id or Path(sample_id).name != sample_id:
        raise ValueError("label must be binary and id a safe basename")
    raw = log_fft_windows(signal)
    features = ((raw - mean[None, None]) / std[None, None]).astype(np.float32)
    graphs = graph_sequence(features, top_k)
    return {
        "history_x": features[:11], "history_a": graphs[:11],
        "target_x": features[11], "target_a": graphs[11],
        "target_edges": upper_edges(graphs[11]),
        "label": np.asarray(label, dtype=np.float32),
        "id": np.asarray(sample_id),
    }


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = [row.get("id") for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Manifest must be nonempty with unique IDs")
    return rows
