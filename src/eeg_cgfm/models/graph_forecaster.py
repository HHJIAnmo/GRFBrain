from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .ggrf import graphs_to_upper_edges, upper_edges_to_symmetric


@dataclass
class GraphForecast:
    edge_mean: torch.Tensor
    context: torch.Tensor


class GraphWindowEncoder(nn.Module):
    def __init__(self, node_dim: int = 100, hidden_dim: int = 64,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.node_projection = nn.Linear(node_dim, hidden_dim)
        self.message_projection = nn.Linear(hidden_dim, hidden_dim)
        self.message_norm = nn.LayerNorm(hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, 2 * hidden_dim,
                                           dropout, "gelu", batch_first=True,
                                           norm_first=True)
        self.spatial_attention = nn.TransformerEncoder(layer, 1, nn.LayerNorm(hidden_dim))
        self.edge_projection = nn.Sequential(nn.Linear(171, hidden_dim), nn.GELU(),
                                             nn.Linear(hidden_dim, hidden_dim))
        self.window_norm = nn.LayerNorm(hidden_dim)

    def forward(self, history_x, history_a):
        adjacency = 0.5 * (history_a + history_a.transpose(-1, -2))
        diagonal = torch.arange(19, device=history_a.device)
        adjacency = adjacency.clone()
        adjacency[..., diagonal, diagonal] = 0.0
        eye = torch.eye(19, device=history_a.device, dtype=history_a.dtype)[None, None]
        adjacency = adjacency + eye
        degree = adjacency.sum(-1).clamp_min(1e-6)
        normalized = degree.rsqrt()[..., None] * adjacency * degree.rsqrt()[..., None, :]
        h = self.node_projection(history_x)
        flat = h.reshape(-1, 19, 64)
        flat = self.message_norm(flat + self.message_projection(
            torch.bmm(normalized.reshape(-1, 19, 19), flat)))
        node = self.spatial_attention(flat).mean(1).reshape(len(h), 11, 64)
        edge = self.edge_projection(graphs_to_upper_edges(adjacency - eye))
        return self.window_norm(node + edge)


class GraphForecaster(nn.Module):
    """Past-only graph-conditioned GRU used to define M_A and edge context."""

    def __init__(self, node_dim: int = 100, hidden_dim: int = 64,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.window_encoder = GraphWindowEncoder(node_dim, hidden_dim, heads, dropout)
        self.position_embedding = nn.Parameter(torch.zeros(1, 11, hidden_dim))
        nn.init.normal_(self.position_embedding, std=0.02)
        self.temporal_encoder = nn.GRU(hidden_dim, hidden_dim, num_layers=2,
                                       dropout=dropout, batch_first=True)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.edge_delta_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
                                             nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 171))
        nn.init.zeros_(self.edge_delta_head[-1].weight); nn.init.zeros_(self.edge_delta_head[-1].bias)

    def forward(self, history_x: torch.Tensor, history_a: torch.Tensor) -> GraphForecast:
        if history_x.shape[1:] != (11, 19, 100) or history_a.shape[1:] != (11, 19, 19):
            raise ValueError("Expected history_x/history_a (B,11,19,100)/(B,11,19,19)")
        tokens = self.window_encoder(history_x, history_a) + self.position_embedding
        _, hidden = self.temporal_encoder(tokens)
        context = self.context_norm(hidden[-1])
        symmetric_history = 0.5 * (history_a + history_a.transpose(-1, -2))
        diagonal = torch.arange(19, device=history_a.device)
        symmetric_history = symmetric_history.clone()
        symmetric_history[..., diagonal, diagonal] = 0.0
        anchor = graphs_to_upper_edges(symmetric_history.mean(1))
        edge_mean = torch.sigmoid(torch.logit(anchor.clamp(.005, .995)) + self.edge_delta_head(context))
        return GraphForecast(edge_mean=edge_mean, context=context)
