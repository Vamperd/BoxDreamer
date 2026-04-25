from __future__ import annotations

from typing import Dict

import torch


def decode_heatmap_argmax(logits: torch.Tensor, crop_size: int = 256) -> torch.Tensor:
    """Decode heatmap logits into crop-space corner coordinates.

    Args:
        logits: [B, 8, H, W]
        crop_size: crop image size in pixels.

    Returns:
        Tensor [B, 8, 2] in crop pixel coordinates.
    """
    probs = torch.sigmoid(logits)
    bsz, channels, height, width = probs.shape
    flat = probs.flatten(2)
    idx = flat.argmax(dim=2)
    xs = (idx % width).float()
    ys = (idx // width).float()
    coords = torch.stack([xs, ys], dim=-1)
    scale = torch.tensor([crop_size / width, crop_size / height], device=logits.device, dtype=logits.dtype)
    return coords * scale


@torch.no_grad()
def corner_metrics(
    logits: torch.Tensor,
    target_corners: torch.Tensor,
    valid: torch.Tensor,
    crop_size: int = 256,
) -> Dict[str, float]:
    pred = decode_heatmap_argmax(logits, crop_size=crop_size)
    mask = valid.bool()
    if mask.sum().item() == 0:
        return {"corner_px": 0.0, "pck_2": 0.0, "pck_5": 0.0, "pck_10": 0.0}

    dist = torch.linalg.norm(pred - target_corners, dim=-1)
    dist_valid = dist[mask]
    return {
        "corner_px": float(dist_valid.mean().item()),
        "pck_2": float((dist_valid <= 2.0).float().mean().item()),
        "pck_5": float((dist_valid <= 5.0).float().mean().item()),
        "pck_10": float((dist_valid <= 10.0).float().mean().item()),
    }

