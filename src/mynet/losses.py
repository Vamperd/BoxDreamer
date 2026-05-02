from __future__ import annotations

from typing import Dict

import torch


def heatmap_mse(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Heatmap MSE for direct 8-corner supervision."""
    pred = torch.sigmoid(logits.float())
    return ((pred - target.float()) ** 2).mean()


def corner_loss(logits: torch.Tensor, target_heatmaps: torch.Tensor) -> Dict[str, torch.Tensor]:
    loss = heatmap_mse(logits, target_heatmaps)
    return {"loss": loss}
