from __future__ import annotations

from pathlib import Path

import torch

from eeg_cgfm.config import load_config
from eeg_cgfm.models import EEGConditionalFlowModel
from eeg_cgfm.utils import seed_everything


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    config = load_config(root / "configs/default.yaml")
    config["runtime"]["device"] = "cpu"; seed_everything(123)
    model = EEGConditionalFlowModel(config)
    b = 2
    x = torch.randn(b, 12, 19, 100)
    # Synthetic positive directed history; real graphs come from preprocessing.
    a = torch.rand(b, 12, 19, 19)
    diagonal = torch.arange(19); a[:, :, diagonal, diagonal] = 0
    edges = 0.5 * (a[:, 11] + a[:, 11].transpose(-1, -2))
    ij = torch.triu_indices(19, 19, 1)
    batch = {"history_x": x[:, :11], "history_a": a[:, :11], "target_x": x[:, 11],
             "target_edges": edges[:, ij[0], ij[1]], "label": torch.tensor([0., 1.])}
    losses = model.flow_matching_loss(batch); losses["loss"].backward()
    if not torch.isfinite(losses["loss"]) or model.field.node_output_head[-1].weight.grad is None:
        raise RuntimeError("Flow smoke failed")
    model.zero_grad(set_to_none=True)
    logits, hidden = model.no_transport_logits(batch["history_x"], batch["history_a"], True)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["label"]).backward()
    if logits.shape != (b,) or hidden.shape != (b, 19, 64):
        raise RuntimeError("No-transport contract failed")
    if model.readout.output.weight.grad is None:
        raise RuntimeError("Readout gradient missing")
    print("EEG_CGFM_SMOKE_OK")


if __name__ == "__main__": main()
