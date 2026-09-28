from __future__ import annotations

import numpy as np
import torch

from eeg_cgfm.config import load_config
from eeg_cgfm.data.preprocessing import absolute_spearman_topk, build_sample
from eeg_cgfm.models.ggrf import history_ggrf_operators, sample_edge_source, sample_node_source
from eeg_cgfm.models.system import EEGConditionalFlowModel


def config():
    from pathlib import Path
    return load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")


def synthetic_batch(batch=2):
    torch.manual_seed(7)
    x = torch.randn(batch, 12, 19, 100)
    a = torch.rand(batch, 12, 19, 19)
    diagonal = torch.arange(19); a[:, :, diagonal, diagonal] = 0
    target = 0.5 * (a[:, 11] + a[:, 11].transpose(-1, -2))
    ij = torch.triu_indices(19, 19, 1)
    return {"history_x": x[:, :11], "history_a": a[:, :11], "target_x": x[:, 11],
            "target_edges": target[:, ij[0], ij[1]], "label": torch.arange(batch).float() % 2}


def test_preprocessing_contract():
    rng = np.random.default_rng(3)
    signal = rng.normal(size=(19, 2400)).astype(np.float32)
    sample = build_sample(signal, 1, "sample", np.zeros(100, np.float32), np.ones(100, np.float32))
    assert sample["history_x"].shape == (11, 19, 100)
    assert sample["history_a"].shape == (11, 19, 19)
    assert np.all((sample["history_a"] > 0).sum(-1) == 3)
    assert sample["target_edges"].shape == (171,)


def test_topk_is_directed_and_zero_diagonal():
    graph = absolute_spearman_topk(np.random.default_rng(9).normal(size=(19, 100)).astype(np.float32))
    assert np.all(np.diag(graph) == 0)
    assert np.all((graph > 0).sum(-1) == 3)


def test_ggrf_psd_trace_and_shapes():
    batch = synthetic_batch()
    ops = history_ggrf_operators(batch["history_a"])
    assert ops.covariance.shape == (2, 19, 19)
    assert torch.allclose(ops.covariance.diagonal(dim1=-2, dim2=-1).sum(-1),
                          torch.full((2,), 19.0), atol=1e-4)
    assert torch.linalg.eigvalsh(ops.covariance).min() > -1e-5
    node, _ = sample_node_source(batch["history_a"], 100, .25)
    edge, _ = sample_edge_source(batch["history_a"], .25, operators=ops)
    assert node.shape == (2, 19, 100) and edge.shape == (2, 171)


def test_joint_field_and_no_transport_readout():
    model = EEGConditionalFlowModel(config())
    assert sum(p.numel() for p in model.graph_forecaster.parameters()) == 125803
    assert sum(p.numel() for p in model.temporal.parameters()) == 36032
    assert sum(p.numel() for p in model.field.parameters()) == 393581
    assert sum(p.numel() for p in model.readout.parameters()) == 65
    batch = synthetic_batch()
    losses = model.flow_matching_loss(batch)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()
    assert model.field.node_output_head[-1].weight.grad is not None
    model.zero_grad(set_to_none=True)
    logits, hidden = model.no_transport_logits(batch["history_x"], batch["history_a"], True)
    assert logits.shape == (2,) and hidden.shape == (2, 19, 64)
    logits.sum().backward()
    assert model.readout.output.weight.grad is not None
    # The classifier reads pre-linear node hidden tokens. Final velocity Linear
    # layers receive gradients only from the separate FM objective.
    assert model.field.node_output_head[-1].weight.grad is None
    assert model.field.edge_output_head[-1].weight.grad is None


def test_target_isolation_for_no_transport():
    model = EEGConditionalFlowModel(config()).eval()
    batch = synthetic_batch()
    with torch.no_grad():
        first = model.no_transport_logits(batch["history_x"], batch["history_a"])
        batch["target_x"] += 1000
        batch["target_edges"] *= -3
        second = model.no_transport_logits(batch["history_x"], batch["history_a"])
    assert torch.equal(first, second)
