from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {"data", "ggrf", "model", "flow_pretraining", "downstream", "runtime"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError(f"Configuration must contain {sorted(required)}")
    data, model = config["data"], config["model"]
    if (data["num_nodes"], data["node_dim"], data["history_windows"]) != (19, 100, 11):
        raise ValueError("This released contract is fixed to (11,19,100)")
    if model["hidden_dim"] % model["num_heads"]:
        raise ValueError("hidden_dim must be divisible by num_heads")
    if model["interaction_mode"] != "bidirectional":
        raise ValueError("Released model requires bidirectional node-edge interaction")
    return config
