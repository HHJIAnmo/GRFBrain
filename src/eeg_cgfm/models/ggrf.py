from __future__ import annotations

from dataclasses import dataclass

import torch


def upper_edges_to_symmetric(edges: torch.Tensor, num_nodes: int = 19) -> torch.Tensor:
    expected = num_nodes * (num_nodes - 1) // 2
    if edges.shape[-1] != expected:
        raise ValueError(f"Expected {expected} edges")
    out = edges.new_zeros(*edges.shape[:-1], num_nodes, num_nodes)
    ij = torch.triu_indices(num_nodes, num_nodes, 1, device=edges.device)
    out[..., ij[0], ij[1]] = edges
    out[..., ij[1], ij[0]] = edges
    return out


def graphs_to_upper_edges(graphs: torch.Tensor) -> torch.Tensor:
    if graphs.shape[-1] != graphs.shape[-2]:
        raise ValueError("Graph must be square")
    ij = torch.triu_indices(graphs.shape[-1], graphs.shape[-1], 1, device=graphs.device)
    return graphs[..., ij[0], ij[1]]


@dataclass(frozen=True)
class GGRFOperators:
    laplacian: torch.Tensor
    covariance: torch.Tensor
    covariance_sqrt: torch.Tensor
    frequencies: torch.Tensor
    frequency_weights: torch.Tensor
    edge_trace_scale: torch.Tensor


def history_ggrf_operators(history_a: torch.Tensor, alpha: float = 1.0,
                           nu: float = 1.0) -> GGRFOperators:
    if history_a.ndim != 4 or history_a.shape[-2:] != (19, 19):
        raise ValueError("history_a must be (B,T,19,19)")
    if alpha < 0 or nu <= 0 or not torch.isfinite(history_a).all():
        raise ValueError("Invalid GGRF input")
    adjacency = history_a.mean(1)
    adjacency = 0.5 * (adjacency + adjacency.transpose(-1, -2))
    diagonal = torch.arange(19, device=adjacency.device)
    adjacency = adjacency.clone(); adjacency[:, diagonal, diagonal] = 0
    degree = adjacency.sum(-1).clamp_min(1e-8)
    inv = degree.rsqrt()
    eye = torch.eye(19, device=adjacency.device, dtype=adjacency.dtype)[None]
    laplacian = eye - inv[:, :, None] * adjacency * inv[:, None, :]
    laplacian = 0.5 * (laplacian + laplacian.transpose(-1, -2))
    frequencies, vectors = torch.linalg.eigh(laplacian)
    frequencies = frequencies.clamp_min(0)
    raw = (1 + alpha * frequencies).pow(-nu)
    weights = 19 * raw / raw.sum(-1, keepdim=True).clamp_min(1e-12)
    covariance = (vectors * weights[:, None]) @ vectors.transpose(-1, -2)
    covariance_sqrt = (vectors * weights.sqrt()[:, None]) @ vectors.transpose(-1, -2)
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    covariance_sqrt = 0.5 * (covariance_sqrt + covariance_sqrt.transpose(-1, -2))

    # Exact represented edge-space trace, matching E independent edge variances.
    ij = torch.triu_indices(19, 19, 1, device=history_a.device)
    diag = covariance.diagonal(dim1=-2, dim2=-1)
    squared_gram = covariance_sqrt.square().transpose(-1, -2) @ covariance_sqrt.square()
    raw_trace = (diag[:, ij[0]] * diag[:, ij[1]]
                 + covariance[:, ij[0], ij[1]].square()
                 - 2 * squared_gram[:, ij[0], ij[1]]).sum(-1)
    scale = torch.sqrt(torch.tensor(171.0, device=raw_trace.device) / raw_trace.clamp_min(1e-12))
    return GGRFOperators(laplacian, covariance, covariance_sqrt,
                         frequencies, weights, scale)


def sample_node_source(history_a: torch.Tensor, node_dim: int, sigma: float,
                       alpha: float = 1.0, nu: float = 1.0,
                       generator: torch.Generator | None = None) -> tuple[torch.Tensor, GGRFOperators]:
    ops = history_ggrf_operators(history_a, alpha, nu)
    eps = torch.randn(len(history_a), 19, node_dim, device=history_a.device,
                      dtype=history_a.dtype, generator=generator)
    return sigma * torch.einsum("bij,bjd->bid", ops.covariance_sqrt, eps), ops


def sample_edge_source(history_a: torch.Tensor, sigma: float,
                       alpha: float = 1.0, nu: float = 1.0,
                       generator: torch.Generator | None = None,
                       operators: GGRFOperators | None = None) -> tuple[torch.Tensor, GGRFOperators]:
    ops = operators or history_ggrf_operators(history_a, alpha, nu)
    eps = torch.randn(len(history_a), 171, device=history_a.device,
                      dtype=history_a.dtype, generator=generator)
    matrix = upper_edges_to_symmetric(eps)
    root = ops.covariance_sqrt
    transformed = root @ matrix @ root
    edges = graphs_to_upper_edges(transformed)
    return sigma * ops.edge_trace_scale[:, None] * edges, ops
