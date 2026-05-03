from __future__ import annotations

from typing import Dict, Optional

import torch


def heatmap_mse(logits: torch.Tensor, target: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Heatmap MSE for direct 8-corner supervision."""
    pred = torch.sigmoid(logits.float())
    loss = (pred - target.float()) ** 2
    if valid is None:
        return loss.flatten(1).mean(dim=1).mean()

    channel_mask = valid.bool().view(valid.shape[0], valid.shape[1], 1, 1)
    valid_channels = channel_mask.sum(dim=(1, 2, 3)).float()
    valid_samples = valid_channels > 0
    if not bool(valid_samples.any()):
        return loss.sum() * 0.0

    per_sample_loss = (loss * channel_mask).sum(dim=(1, 2, 3))
    per_sample_denom = (valid_channels * logits.shape[-2] * logits.shape[-1]).clamp_min(1.0)
    return (per_sample_loss / per_sample_denom)[valid_samples].mean()


def corner_loss(
    logits: torch.Tensor,
    target_heatmaps: torch.Tensor,
    corner_valid: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    if logits.shape != target_heatmaps.shape:
        raise ValueError(f"Logit/target heatmap shape mismatch: {tuple(logits.shape)} != {tuple(target_heatmaps.shape)}")
    loss = heatmap_mse(logits, target_heatmaps, valid=corner_valid)
    return {"loss": loss}
