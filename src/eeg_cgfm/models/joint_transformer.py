from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .ggrf import upper_edges_to_symmetric


class TauEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.dim = dim
        half = dim // 2
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(half) / max(half - 1, 1))
        self.register_buffer("frequencies", frequencies, persistent=True)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.reshape(-1, 1)
        value = tau * self.frequencies.to(dtype=tau.dtype)[None]
        return torch.cat((value.sin(), value.cos()), -1)


def edge_token_matrix(tokens: torch.Tensor, n: int = 19) -> torch.Tensor:
    matrix = tokens.new_zeros(len(tokens), n, n, tokens.shape[-1])
    ij = torch.triu_indices(n, n, 1, device=tokens.device)
    matrix[:, ij[0], ij[1]] = tokens; matrix[:, ij[1], ij[0]] = tokens
    return matrix


class JointBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, condition_dim: int, dropout: float):
        super().__init__(); self.hidden, self.heads = hidden, heads; self.head_dim = hidden // heads
        self.node_norm, self.edge_norm = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.film = nn.Linear(condition_dim, 4 * hidden)
        self.query, self.key, self.value = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.attention_output = nn.Linear(hidden, hidden); self.edge_attention_bias = nn.Linear(hidden, heads)
        self.edge_to_node_message = nn.Linear(hidden, hidden)
        self.node_to_edge_message = nn.Sequential(nn.Linear(3 * hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))
        self.edge_independent = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * hidden, hidden))
        self.node_ffn_norm, self.edge_ffn_norm = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.node_ffn = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * hidden, hidden))
        self.edge_ffn = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * hidden, hidden))

    def forward(self, nodes, edges, condition, rows, cols):
        ns, nshift, es, eshift = self.film(condition).chunk(4, -1)
        nh = self.node_norm(nodes) * (1 + ns[:, None]) + nshift[:, None]
        eh = self.edge_norm(edges) * (1 + es[:, None]) + eshift[:, None]
        b, n, _ = nodes.shape
        reshape = lambda x: x.reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = reshape(self.query(nh)), reshape(self.key(nh)), reshape(self.value(nh))
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores + edge_token_matrix(self.edge_attention_bias(eh)).permute(0, 3, 1, 2)
        attended = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, n, self.hidden)
        incident = nodes.new_zeros(b, n, self.hidden)
        message = self.edge_to_node_message(eh)
        incident.index_add_(1, rows, message); incident.index_add_(1, cols, message)
        nodes = nodes + self.attention_output(attended) + incident / (n - 1)
        nodes = nodes + self.node_ffn(self.node_ffn_norm(nodes))
        edges = edges + self.edge_independent(eh)
        left, right = nodes[:, rows], nodes[:, cols]
        edges = edges + self.node_to_edge_message(torch.cat((left + right, (left - right).abs(), left * right), -1))
        edges = edges + self.edge_ffn(self.edge_ffn_norm(edges))
        return nodes, edges


@dataclass
class VelocityOutput:
    node_velocity: torch.Tensor
    edge_velocity: torch.Tensor
    node_hidden: torch.Tensor
    edge_hidden: torch.Tensor


class JointNodeEdgeTransformer(nn.Module):
    def __init__(self, node_dim=100, hidden_dim=64, node_context_dim=64,
                 edge_context_dim=64, tau_dim=32, num_heads=4, num_blocks=2,
                 dropout=0.0):
        super().__init__()
        self.node_dim, self.hidden_dim = node_dim, hidden_dim
        ij = torch.triu_indices(19, 19, 1)
        self.register_buffer("edge_rows", ij[0]); self.register_buffer("edge_cols", ij[1])
        self.tau_embedding = TauEmbedding(tau_dim)
        self.node_state_projection = nn.Linear(2 * node_dim, hidden_dim)
        self.node_context_projection = nn.Linear(node_context_dim, hidden_dim)
        self.edge_state_projection = nn.Linear(2, hidden_dim)
        self.edge_pair_context_projection = nn.Linear(3 * node_context_dim, hidden_dim)
        self.edge_operator_projection = nn.Linear(2, hidden_dim)
        condition_dim = node_context_dim + edge_context_dim + tau_dim
        self.blocks = nn.ModuleList([JointBlock(hidden_dim, num_heads, condition_dim, dropout)
                                     for _ in range(num_blocks)])
        self.node_output_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, node_dim))
        self.edge_output_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, rx, ra, tau, node_context, node_global, edge_global,
                node_mean, edge_mean, laplacian, covariance) -> VelocityOutput:
        b = len(rx); tau = tau.reshape(b, 1)
        nodes = self.node_state_projection(torch.cat((rx, node_mean), -1)) + self.node_context_projection(node_context)
        left, right = node_context[:, self.edge_rows], node_context[:, self.edge_cols]
        pair = torch.cat((left + right, (left - right).abs(), left * right), -1)
        edges = self.edge_state_projection(torch.stack((ra, edge_mean), -1)) + self.edge_pair_context_projection(pair)
        operators = torch.stack((laplacian[:, self.edge_rows, self.edge_cols],
                                 covariance[:, self.edge_rows, self.edge_cols]), -1)
        edges = edges + self.edge_operator_projection(operators)
        condition = torch.cat((node_global, edge_global, self.tau_embedding(tau)), -1)
        for block in self.blocks:
            nodes, edges = block(nodes, edges, condition, self.edge_rows, self.edge_cols)
        node_hidden, edge_hidden = self.node_output_head[0](nodes), self.edge_output_head[0](edges)
        return VelocityOutput(self.node_output_head[1](node_hidden),
                              self.edge_output_head[1](edge_hidden).squeeze(-1),
                              node_hidden, edge_hidden)
