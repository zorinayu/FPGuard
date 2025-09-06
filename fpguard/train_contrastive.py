from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .config import FPGuardConfig
from .fingerprints import ProjectionHeads, concat_fingerprints


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    z_i = nn.functional.normalize(z_i, dim=-1)
    z_j = nn.functional.normalize(z_j, dim=-1)
    logits = z_i @ z_j.T / temperature
    labels = torch.arange(z_i.size(0), device=z_i.device)
    return nn.functional.cross_entropy(logits, labels)


class ContrastiveTrainer:
    def __init__(self, layer_dims: Dict[int, int], config: FPGuardConfig, device: Optional[str] = None):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ProjectionHeads(layer_dims, projection_dim=config.projection_dim).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    def step(self, hidden_a: Dict[int, torch.Tensor], hidden_b: Dict[int, torch.Tensor]) -> float:
        proj_a = self.model({k: v.to(self.device) for k, v in hidden_a.items()})
        proj_b = self.model({k: v.to(self.device) for k, v in hidden_b.items()})
        z_a = concat_fingerprints(proj_a, layer_order=sorted(proj_a.keys()))
        z_b = concat_fingerprints(proj_b, layer_order=sorted(proj_b.keys()))
        loss = nt_xent_loss(z_a, z_b, temperature=self.config.temperature)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())


