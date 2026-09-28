from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SHAPES = {
    "history_x": (11, 19, 100), "history_a": (11, 19, 19),
    "target_x": (19, 100), "target_a": (19, 19), "target_edges": (171,),
}


class ProcessedEEGDataset(Dataset):
    def __init__(self, manifest: str | Path):
        self.manifest = Path(manifest).resolve()
        with self.manifest.open(encoding="utf-8") as stream:
            self.rows = [json.loads(line) for line in stream if line.strip()]
        ids = [row["id"] for row in self.rows]
        if not self.rows or len(ids) != len(set(ids)):
            raise ValueError("Processed manifest must be nonempty with unique IDs")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        path = (self.manifest.parent / row["path"]).resolve()
        with np.load(path, allow_pickle=False) as data:
            values = {key: np.asarray(data[key], dtype=np.float32) for key in SHAPES}
            label = float(data["label"])
            sample_id = str(data["id"])
        for key, shape in SHAPES.items():
            if values[key].shape != shape or not np.isfinite(values[key]).all():
                raise ValueError(f"Invalid {key} in {path}")
        if label not in (0.0, 1.0) or sample_id != row["id"]:
            raise ValueError(f"Label/ID mismatch in {path}")
        return {**{k: torch.from_numpy(v.copy()) for k, v in values.items()},
                "label": torch.tensor(label), "id": sample_id}
