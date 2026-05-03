from __future__ import annotations

from typing import Dict, Optional

import torch


def _crop_hw_tensor(
    logits: torch.Tensor,
    crop_size: int = 256,
    crop_hw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    bsz = logits.shape[0]
    if crop_hw is None:
        return torch.full((bsz, 2), float(crop_size), device=logits.device, dtype=logits.dtype)
    crop_hw = crop_hw.to(device=logits.device, dtype=logits.dtype)
    if crop_hw.ndim == 1:
        crop_hw = crop_hw.view(1, 2).expand(bsz, 2)
    if crop_hw.shape != (bsz, 2):
        raise ValueError(f"crop_hw must be [B, 2] as [height, width], got {tuple(crop_hw.shape)}")
    return crop_hw


def _heatmap_to_crop_scale(
    logits: torch.Tensor,
    crop_size: int = 256,
    crop_hw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _, _, height, width = logits.shape
    crop_hw = _crop_hw_tensor(logits, crop_size=crop_size, crop_hw=crop_hw)
    crop_h = crop_hw[:, 0].clamp_min(1.0)
    crop_w = crop_hw[:, 1].clamp_min(1.0)
    return torch.stack([crop_w / float(width), crop_h / float(height)], dim=-1).view(-1, 1, 2)


def decode_heatmap_argmax(
    logits: torch.Tensor,
    crop_size: int = 256,
    crop_hw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Decode heatmap logits into crop-space corner coordinates.

    Args:
        logits: [B, 8, H, W]
        crop_size: fixed square crop size in pixels.
        crop_hw: optional dynamic crop size tensor [B, 2] as [height, width].

    Returns:
        Tensor [B, 8, 2] in crop pixel coordinates.
    """
    probs = torch.sigmoid(logits.float())
    bsz, channels, height, width = probs.shape
    flat = probs.flatten(2)
    idx = flat.argmax(dim=2)
    xs = (idx % width).float()
    ys = (idx // width).float()
    coords = torch.stack([xs, ys], dim=-1)
    scale = _heatmap_to_crop_scale(logits, crop_size=crop_size, crop_hw=crop_hw)
    return coords * scale


def decode_heatmap_subpixel(
    logits: torch.Tensor,
    crop_size: int = 256,
    crop_hw: Optional[torch.Tensor] = None,
    window: int = 5,
) -> torch.Tensor:
    """Decode heatmaps with a local weighted centroid around the argmax peak.

    The coordinate convention intentionally matches the existing dataset
    generation: crop coordinates are mapped to heatmap coordinates with
    `x_hm = x_crop * heatmap_width / crop_size`.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError("subpixel window must be a positive odd integer.")

    probs = torch.sigmoid(logits.float())
    bsz, channels, height, width = probs.shape
    flat = probs.flatten(2)
    idx = flat.argmax(dim=2)
    xs = (idx % width).float()
    ys = (idx // width).float()

    radius = window // 2
    coords = torch.empty((bsz, channels, 2), device=logits.device, dtype=probs.dtype)
    eps = 1e-6
    for batch_idx in range(bsz):
        for channel_idx in range(channels):
            x0 = int(xs[batch_idx, channel_idx].item())
            y0 = int(ys[batch_idx, channel_idx].item())
            x1 = max(0, x0 - radius)
            x2 = min(width, x0 + radius + 1)
            y1 = max(0, y0 - radius)
            y2 = min(height, y0 + radius + 1)
            patch = probs[batch_idx, channel_idx, y1:y2, x1:x2]
            denom = patch.sum()
            if denom <= eps:
                coords[batch_idx, channel_idx] = torch.stack([xs[batch_idx, channel_idx], ys[batch_idx, channel_idx]])
                continue
            yy, xx = torch.meshgrid(
                torch.arange(y1, y2, device=logits.device, dtype=probs.dtype),
                torch.arange(x1, x2, device=logits.device, dtype=probs.dtype),
                indexing="ij",
            )
            coords[batch_idx, channel_idx, 0] = (patch * xx).sum() / denom
            coords[batch_idx, channel_idx, 1] = (patch * yy).sum() / denom

    scale = _heatmap_to_crop_scale(logits, crop_size=crop_size, crop_hw=crop_hw).to(dtype=probs.dtype)
    return coords * scale


def decode_heatmap(
    logits: torch.Tensor,
    crop_size: int = 256,
    crop_hw: Optional[torch.Tensor] = None,
    decode_method: str = "subpixel",
    subpixel_window: int = 5,
) -> torch.Tensor:
    if decode_method == "argmax":
        return decode_heatmap_argmax(logits, crop_size=crop_size, crop_hw=crop_hw)
    if decode_method == "subpixel":
        return decode_heatmap_subpixel(logits, crop_size=crop_size, crop_hw=crop_hw, window=subpixel_window)
    raise ValueError(f"Unsupported decode method: {decode_method}")


@torch.no_grad()
def corner_metrics(
    logits: torch.Tensor,
    target_corners: torch.Tensor,
    valid: torch.Tensor,
    crop_size: int = 256,
    crop_hw: Optional[torch.Tensor] = None,
    decode_method: str = "subpixel",
    subpixel_window: int = 5,
    invisible_peak_threshold: float = 0.3,
) -> Dict[str, float]:
    pred = decode_heatmap(logits, crop_size=crop_size, crop_hw=crop_hw, decode_method=decode_method, subpixel_window=subpixel_window)
    mask = valid.bool()
    probs = torch.sigmoid(logits.float())
    peak_scores = probs.flatten(2).max(dim=2).values
    invisible_mask = ~mask
    visible_peak = peak_scores[mask]
    invisible_peak = peak_scores[invisible_mask]
    valid_ratio = float(mask.float().mean().item()) if mask.numel() else 0.0
    visibility_metrics = {
        "valid_corner_ratio": valid_ratio,
        "invisible_corner_ratio": 1.0 - valid_ratio,
        "visible_peak_mean": float(visible_peak.mean().item()) if visible_peak.numel() else 0.0,
        "invisible_peak_mean": float(invisible_peak.mean().item()) if invisible_peak.numel() else 0.0,
        "invisible_false_peak_rate": float((invisible_peak > invisible_peak_threshold).float().mean().item()) if invisible_peak.numel() else 0.0,
    }
    if mask.sum().item() == 0:
        return {"corner_px": 0.0, "pck_2": 0.0, "pck_5": 0.0, "pck_10": 0.0, **visibility_metrics}

    dist = torch.linalg.norm(pred - target_corners, dim=-1)
    dist_valid = dist[mask]
    return {
        "corner_px": float(dist_valid.mean().item()),
        "pck_2": float((dist_valid <= 2.0).float().mean().item()),
        "pck_5": float((dist_valid <= 5.0).float().mean().item()),
        "pck_10": float((dist_valid <= 10.0).float().mean().item()),
        **visibility_metrics,
    }
