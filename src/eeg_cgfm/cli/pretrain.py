from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eeg_cgfm.config import load_config
from eeg_cgfm.data import ProcessedEEGDataset
from eeg_cgfm.models import EEGConditionalFlowModel
from eeg_cgfm.training.engine import graph_forecast_loss, move
from eeg_cgfm.utils import deterministic_policy, save_json, seed_everything


@torch.no_grad()
def dev_loss(model, loader, device):
    model.eval(); values = []
    for batch in loader:
        result = model.flow_matching_loss(move(batch, device)); values.append(float(result["loss"]))
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain conditional GGRF Joint Flow")
    parser.add_argument("--config", required=True); parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--dev-manifest", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); config = load_config(args.config)
    deterministic_policy(); seed_everything(int(config["runtime"]["seed"]))
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    train = ProcessedEEGDataset(args.train_manifest); dev = ProcessedEEGDataset(args.dev_manifest)
    settings = config["flow_pretraining"]
    train_loader = DataLoader(train, settings["batch_size"], shuffle=True,
                              num_workers=config["runtime"]["num_workers"])
    dev_loader = DataLoader(dev, settings["batch_size"], shuffle=False,
                            num_workers=config["runtime"]["num_workers"])
    model = EEGConditionalFlowModel(config).to(device)

    # Supervise the past-only conditional edge mean first, then freeze it.
    graph_opt = torch.optim.AdamW(model.graph_forecaster.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    for epoch in range(int(settings.get("graph_forecaster_epochs", 3))):
        model.train()
        for batch in tqdm(train_loader, desc=f"graph {epoch+1}"):
            batch = move(batch, device); loss = graph_forecast_loss(model, batch)
            graph_opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.graph_forecaster.parameters(), settings["grad_clip"])
            graph_opt.step()
    model.graph_forecaster.eval().requires_grad_(False)

    # Re-estimate velocity normalization on train only. These values are part of
    # the checkpoint/config contract and must never be fitted from dev or test.
    totals = {"node_sum": 0.0, "node_count": 0, "edge_sum": 0.0, "edge_count": 0}
    model.eval()
    for batch in tqdm(train_loader, desc="calibrate q"):
        values = model.calibration_statistics(move(batch, device))
        for key, value in values.items(): totals[key] += value
    model.q_x = totals["node_sum"] / totals["node_count"]
    model.q_a = totals["edge_sum"] / totals["edge_count"]
    config["flow_pretraining"]["q_x"] = model.q_x
    config["flow_pretraining"]["q_a"] = model.q_a
    save_json(output / "calibration.json", {"split": "train-only", "q_x": model.q_x,
                                             "q_a": model.q_a,
                                             "sigma_x": model.ggrf_config["sigma_x"],
                                             "sigma_a": model.ggrf_config["sigma_a"]})

    parameters = list(model.temporal.parameters()) + list(model.field.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    best = float("inf"); history = []
    for epoch in range(int(settings["epochs"])):
        model.train(); model.graph_forecaster.eval(); total = 0.0
        for batch in tqdm(train_loader, desc=f"flow {epoch+1}"):
            batch = move(batch, device); result = model.flow_matching_loss(batch)
            optimizer.zero_grad(set_to_none=True); result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, settings["grad_clip"]); optimizer.step()
            total += float(result["loss"].detach())
        value = dev_loss(model, dev_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": total / len(train_loader), "dev_loss": value})
        state = {"stage": "flow_pretrain", "epoch": epoch + 1, "config": config,
                 "model": model.state_dict(), "dev_loss": value}
        torch.save(state, output / "last.pt")
        if value < best:
            best = value; torch.save(state, output / "best.pt")
        save_json(output / "history.json", {"history": history, "best_dev_loss": best})


if __name__ == "__main__": main()
