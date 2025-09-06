from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.LayerNorm(prev))
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.LayerNorm(prev))
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(prev, hidden_dims[-1])

        self.skip = nn.Linear(input_dim, hidden_dims[-1]) if input_dim != hidden_dims[-1] else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        y = self.out(h)
        return y + self.skip(x)


class ProjectionHeads(nn.Module):
    def __init__(self, layer_dims: Dict[int, int], projection_dim: int):
        super().__init__()
        self.heads = nn.ModuleDict()
        for layer_id, dim in layer_dims.items():
            self.heads[str(layer_id)] = ResidualMLP(dim, [512, 256, projection_dim])

    def forward(self, hidden_by_layer: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        outputs: Dict[int, torch.Tensor] = {}
        for layer_id, h in hidden_by_layer.items():
            head = self.heads[str(layer_id)]
            outputs[layer_id] = head(h)
        return outputs


def concat_fingerprints(fp_by_layer: Dict[int, torch.Tensor], layer_order: List[int]) -> torch.Tensor:
    ordered: List[torch.Tensor] = [fp_by_layer[l] for l in layer_order]
    return torch.cat(ordered, dim=-1)


@torch.no_grad()
def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = nn.functional.normalize(a, dim=-1)
    b = nn.functional.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


