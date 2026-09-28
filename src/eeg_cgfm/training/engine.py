from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .metrics import classification_metrics


def move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, threshold=None):
    model.eval(); logits, labels, ids = [], [], []
    for batch in loader:
        batch = move(batch, device)
        logits.append(model.no_transport_logits(batch["history_x"], batch["history_a"]).cpu().numpy())
        labels.append(batch["label"].cpu().numpy()); ids.extend(batch["id"])
    z, y = np.concatenate(logits), np.concatenate(labels)
    return classification_metrics(y, z, threshold), {"logits": z, "labels": y, "ids": np.asarray(ids)}


def graph_forecast_loss(model, batch):
    prediction = model.graph_forecaster(batch["history_x"], batch["history_a"])
    return F.smooth_l1_loss(prediction.edge_mean, batch["target_edges"])
