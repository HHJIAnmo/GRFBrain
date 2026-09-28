from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .ggrf import history_ggrf_operators, sample_edge_source, sample_node_source
from .graph_forecaster import GraphForecaster
from .joint_transformer import JointNodeEdgeTransformer


class TemporalConditionEncoder(nn.Module):
    def __init__(self, node_dim: int = 100, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(node_dim, hidden_dim, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.output_projection.weight); nn.init.zeros_(self.output_projection.bias)

    def forward(self, history_x: torch.Tensor, node_mean: torch.Tensor):
        b = len(history_x)
        sequence = history_x.permute(0, 2, 1, 3).reshape(b * 19, 11, 100)
        _, state = self.gru(sequence)
        delta = self.output_projection(state[-1]).reshape(b, 19, 64)
        base = F.adaptive_avg_pool1d(node_mean.reshape(b * 19, 1, 100), 64).reshape(b, 19, 64)
        context = base + delta
        return context, context.mean(1)


class NodeHiddenReadout(nn.Module):
    """Shared node score followed by max-node pooling; exactly 65 parameters."""
    def __init__(self, hidden_dim: int = 64):
        super().__init__(); self.output = nn.Linear(hidden_dim, 1)

    def forward(self, node_hidden: torch.Tensor) -> torch.Tensor:
        if node_hidden.ndim != 3 or node_hidden.shape[1:] != (19, 64):
            raise ValueError("node_hidden must be (B,19,64)")
        return self.output(torch.relu(node_hidden)).squeeze(-1).max(1).values


@dataclass
class Conditions:
    node_mean: torch.Tensor
    edge_mean: torch.Tensor
    node_context: torch.Tensor
    node_global: torch.Tensor
    edge_global: torch.Tensor
    laplacian: torch.Tensor
    covariance: torch.Tensor


class EEGConditionalFlowModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model = config["model"]
        self.ggrf_config = dict(config["ggrf"])
        self.q_x = float(config["flow_pretraining"]["q_x"])
        self.q_a = float(config["flow_pretraining"]["q_a"])
        self.graph_forecaster = GraphForecaster(
            hidden_dim=model["edge_context_dim"], heads=model["num_heads"],
            dropout=model["graph_forecaster_dropout"])
        self.temporal = TemporalConditionEncoder(100, model["node_context_dim"])
        self.field = JointNodeEdgeTransformer(
            hidden_dim=model["hidden_dim"], node_context_dim=model["node_context_dim"],
            edge_context_dim=model["edge_context_dim"], tau_dim=model["tau_dim"],
            num_heads=model["num_heads"], num_blocks=model["num_blocks"],
            dropout=model["dropout"])
        self.readout = NodeHiddenReadout(model["hidden_dim"])

    def conditions(self, history_x: torch.Tensor, history_a: torch.Tensor) -> Conditions:
        node_mean = history_x.mean(1)
        graph = self.graph_forecaster(history_x, history_a)
        node_context, node_global = self.temporal(history_x, node_mean)
        ops = history_ggrf_operators(history_a, self.ggrf_config["alpha"], self.ggrf_config["nu"])
        return Conditions(node_mean, graph.edge_mean, node_context, node_global,
                          graph.context, ops.laplacian, ops.covariance)

    def flow_matching_loss(self, batch: dict) -> dict[str, torch.Tensor]:
        history_x, history_a = batch["history_x"], batch["history_a"]
        c = self.conditions(history_x, history_a)
        rx0, ops = sample_node_source(history_a, 100, self.ggrf_config["sigma_x"],
                                      self.ggrf_config["alpha"], self.ggrf_config["nu"])
        ra0, _ = sample_edge_source(history_a, self.ggrf_config["sigma_a"],
                                    self.ggrf_config["alpha"], self.ggrf_config["nu"], operators=ops)
        rx1, ra1 = batch["target_x"] - c.node_mean, batch["target_edges"] - c.edge_mean
        tau = torch.rand(len(history_x), 1, device=history_x.device)
        rx = (1 - tau[:, :, None]) * rx0 + tau[:, :, None] * rx1
        ra = (1 - tau) * ra0 + tau * ra1
        output = self.field(rx, ra, tau, c.node_context, c.node_global, c.edge_global,
                            c.node_mean, c.edge_mean, c.laplacian, c.covariance)
        node = (output.node_velocity - (rx1 - rx0)).square().mean() / max(self.q_x, 1e-8)
        edge = (output.edge_velocity - (ra1 - ra0)).square().mean() / max(self.q_a, 1e-8)
        return {"loss": node + edge, "node": node, "edge": edge}

    @torch.no_grad()
    def calibration_statistics(self, batch: dict) -> dict[str, float]:
        """Unnormalized train-only sums/counts for q_X and q_A calibration."""
        history_x, history_a = batch["history_x"], batch["history_a"]
        c = self.conditions(history_x, history_a)
        rx0, ops = sample_node_source(history_a, 100, self.ggrf_config["sigma_x"],
                                      self.ggrf_config["alpha"], self.ggrf_config["nu"])
        ra0, _ = sample_edge_source(history_a, self.ggrf_config["sigma_a"],
                                    self.ggrf_config["alpha"], self.ggrf_config["nu"], operators=ops)
        ux = batch["target_x"] - c.node_mean - rx0
        ua = batch["target_edges"] - c.edge_mean - ra0
        return {"node_sum": float(ux.square().sum()), "node_count": ux.numel(),
                "edge_sum": float(ua.square().sum()), "edge_count": ua.numel()}

    def no_transport_logits(self, history_x: torch.Tensor, history_a: torch.Tensor,
                            return_hidden: bool = False):
        c = self.conditions(history_x, history_a)
        rx = history_x.new_zeros(len(history_x), 19, 100)
        ra = history_x.new_zeros(len(history_x), 171)
        tau = history_x.new_ones(len(history_x), 1)
        output = self.field(rx, ra, tau, c.node_context, c.node_global, c.edge_global,
                            c.node_mean, c.edge_mean, c.laplacian, c.covariance)
        logits = self.readout(output.node_hidden)
        return (logits, output.node_hidden) if return_hidden else logits
