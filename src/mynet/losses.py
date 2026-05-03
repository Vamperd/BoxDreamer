from __future__ import annotations

from typing import Dict, Optional

import torch


def heatmap_mse(logits: torch.Tensor, target: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Heatmap MSE for direct 8-corner supervision."""
    pred = torch.sigmoid(logits.float())
    loss = (pred - target.float()) ** 2
    if valid is None:
        return loss.mean()

    channel_mask = valid.bool().view(valid.shape[0], valid.shape[1], 1, 1)
    if not bool(channel_mask.any()):
        return loss.sum() * 0.0
    return (loss * channel_mask).sum() / (channel_mask.sum() * logits.shape[-2] * logits.shape[-1]).clamp_min(1)


def corner_loss(
    logits: torch.Tensor,
    target_heatmaps: torch.Tensor,
    corner_valid: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    if logits.shape != target_heatmaps.shape:
        raise ValueError(f"Logit/target heatmap shape mismatch: {tuple(logits.shape)} != {tuple(target_heatmaps.shape)}")
    loss = heatmap_mse(logits, target_heatmaps, valid=corner_valid)
    return {"loss": loss}
