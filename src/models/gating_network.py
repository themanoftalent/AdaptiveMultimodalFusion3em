from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveGatingNetwork(nn.Module):
    """MLP gate for adaptive decision-level fusion.

    Input:
        Concatenated probability vector [p_r ; p_h]
        Dimension: 6

    Output:
        Sample-specific radiograph-branch weight alpha_hat in (0, 1)
    """

    def __init__(self, input_dim: int = 6, hidden_dims: list[int] | None = None):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

    def forward(self, p_r: torch.Tensor, p_h: torch.Tensor) -> torch.Tensor:
        x = torch.cat([p_r, p_h], dim=1)
        return self.mlp(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
