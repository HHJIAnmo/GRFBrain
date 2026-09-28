from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from eeg_cgfm.config import load_config
from eeg_cgfm.data import ProcessedEEGDataset
from eeg_cgfm.models import EEGConditionalFlowModel
from eeg_cgfm.training.engine import evaluate, move
from eeg_cgfm.utils import deterministic_policy, save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train node_hidden_no_transport readout")
    parser.add_argument("--config", required=True); parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True); parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--output", required=True); args = parser.parse_args()
    config = load_config(args.config); deterministic_policy(); seed_everything(config["runtime"]["seed"])
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    train, dev = ProcessedEEGDataset(args.train_manifest), ProcessedEEGDataset(args.dev_manifest)
    setting = config["downstream"]
    train_loader = DataLoader(train, setting["batch_size"], shuffle=True,
                              num_workers=config["runtime"]["num_workers"])
    dev_loader = DataLoader(dev, setting["batch_size"], shuffle=False,
                            num_workers=config["runtime"]["num_workers"])
    model = EEGConditionalFlowModel(config).to(device)
    state = torch.load(args.flow_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.graph_forecaster.eval().requires_grad_(False)
    optimizer = torch.optim.AdamW([
        {"params": model.readout.parameters(), "lr": setting["classifier_lr"]},
        {"params": model.temporal.parameters(), "lr": setting["temporal_lr"]},
        {"params": model.field.parameters(), "lr": setting["field_lr"]}],
        weight_decay=setting["weight_decay"])
    best, stale, history = -1.0, 0, []
    for epoch in range(1, setting["epochs"] + 1):
        model.train(); model.graph_forecaster.eval()
        model.field.requires_grad_(epoch > setting["field_warmup_epochs"])
        total = 0.0
        for batch in tqdm(train_loader, desc=f"downstream {epoch}"):
            batch = move(batch, device)
            logits = model.no_transport_logits(batch["history_x"], batch["history_a"])
            cls = F.binary_cross_entropy_with_logits(logits, batch["label"])
            fm = (model.flow_matching_loss(batch)["loss"]
                  if epoch > setting["field_warmup_epochs"] else cls.new_zeros(()))
            loss = cls + setting["fm_weight"] * fm
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), setting["grad_clip"]); optimizer.step()
            total += float(loss.detach())
        metrics, _ = evaluate(model, dev_loader, device)
        history.append({"epoch": epoch, "train_loss": total / len(train_loader), "dev": metrics})
        checkpoint = {"stage": "node_hidden_no_transport", "epoch": epoch,
                      "config": config, "model": model.state_dict(), "dev": metrics,
                      "threshold": metrics["threshold"]}
        torch.save(checkpoint, output / "last.pt")
        if metrics["auprc"] > best:
            best, stale = metrics["auprc"], 0; torch.save(checkpoint, output / "best.pt")
        else:
            stale += 1
        save_json(output / "history.json", {"history": history, "best_dev_auprc": best})
        if stale >= setting["patience"]: break


if __name__ == "__main__": main()
