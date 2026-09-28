from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eeg_cgfm.config import load_config
from eeg_cgfm.data import ProcessedEEGDataset
from eeg_cgfm.models import EEGConditionalFlowModel
from eeg_cgfm.training.engine import evaluate
from eeg_cgfm.utils import deterministic_policy, save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-threshold downstream evaluation")
    parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); config = load_config(args.config)
    deterministic_policy(); seed_everything(config["runtime"]["seed"])
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if state.get("stage") != "node_hidden_no_transport" or "threshold" not in state:
        raise ValueError("Evaluation requires a dev-selected no-transport checkpoint")
    model = EEGConditionalFlowModel(config).to(device); model.load_state_dict(state["model"], strict=True)
    loader = DataLoader(ProcessedEEGDataset(args.manifest), config["downstream"]["batch_size"],
                        shuffle=False, num_workers=config["runtime"]["num_workers"])
    metrics, predictions = evaluate(model, loader, device, state["threshold"])
    save_json(output / "metrics.json", {"metrics": metrics, "threshold_source": "frozen dev",
                                        "test_selection_or_tuning": False})
    np.savez_compressed(output / "predictions.npz", **predictions)


if __name__ == "__main__": main()
