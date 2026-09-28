"""Convert the original split checkpoints into one portable released checkpoint."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch

from eeg_cgfm.config import load_config
from eeg_cgfm.models import EEGConditionalFlowModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--hidden-checkpoint", required=True,
                        help="Original node_hidden_no_transport best.pt")
    parser.add_argument("--edge-predictor-checkpoint", required=True,
                        help="Original deterministic graph forecaster checkpoint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    hidden = torch.load(args.hidden_checkpoint, map_location="cpu", weights_only=False)
    edge = torch.load(args.edge_predictor_checkpoint, map_location="cpu", weights_only=False)
    if hidden.get("arm") != "node_hidden_no_transport":
        raise ValueError("Expected a node_hidden_no_transport checkpoint")
    config = deepcopy(load_config(args.config))
    source = hidden.get("config", {}).get("source", {})
    graph_source = hidden.get("config", {}).get("graph_source_config", {})
    for key, target in (("sigma_X", "sigma_x"), ("sigma_A", "sigma_a")):
        if key in source: config["ggrf"][target] = float(source[key])
    for key in ("q_X", "q_A"):
        if key in source: config["flow_pretraining"][key.lower()] = float(source[key])
    for key in ("alpha", "nu"):
        if key in graph_source: config["ggrf"][key] = float(graph_source[key])
    model = EEGConditionalFlowModel(config)
    model.graph_forecaster.load_state_dict(edge["model_state"], strict=True)
    model.temporal.load_state_dict(hidden["temporal_state"], strict=True)
    model.field.load_state_dict(hidden["field_state"], strict=True)
    model.readout.load_state_dict(hidden["classifier_state"], strict=True)
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): raise FileExistsError(output)
    torch.save({"stage": "node_hidden_no_transport", "epoch": int(hidden["epoch"]),
                "config": config, "model": model.state_dict(), "dev": hidden["dev"],
                "threshold": float(hidden["dev"]["threshold"]),
                "provenance": {"hidden_checkpoint": str(Path(args.hidden_checkpoint).resolve()),
                               "edge_predictor_checkpoint": str(Path(args.edge_predictor_checkpoint).resolve())}}, output)
    print(f"PORTABLE_CHECKPOINT_OK {output}")


if __name__ == "__main__": main()
