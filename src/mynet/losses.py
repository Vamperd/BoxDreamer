from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def heatmap_coarse_mse(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Coarse heatmap loss over all 8 corner channels.

    Invisible corners are represented by all-zero target heatmaps, so they must
    stay in this loss to teach the model to suppress false peaks.
    """
    pred = torch.sigmoid(logits.float())
    return ((pred - target.float()) ** 2).mean()


def soft_argmax_2d(logits: torch.Tensor, crop_size: int = 256, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("soft-argmax temperature must be positive.")
    logits_f = logits.float()
    bsz, channels, height, width = logits_f.shape
    probs = torch.softmax(logits_f.flatten(2) / temperature, dim=2)

    ys, xs = torch.meshgrid(
        torch.arange(height, device=logits.device, dtype=probs.dtype),
        torch.arange(width, device=logits.device, dtype=probs.dtype),
        indexing="ij",
    )
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)
    coords_hm = torch.stack([(probs * xs).sum(dim=2), (probs * ys).sum(dim=2)], dim=-1)
    scale = torch.tensor([crop_size / width, crop_size / height], device=logits.device, dtype=probs.dtype)
    return coords_hm * scale


def visible_coordinate_smooth_l1(
    pred_corners: torch.Tensor,
    target_corners: torch.Tensor,
    valid: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    mask = valid.bool()
    if mask.sum().item() == 0:
        return pred_corners.sum() * 0.0
    return F.smooth_l1_loss(pred_corners[mask], target_corners.float()[mask], beta=beta)


def corner_loss(
    logits: torch.Tensor,
    target_heatmaps: torch.Tensor,
    target_corners: torch.Tensor,
    valid: torch.Tensor,
    crop_size: int = 256,
    fine_loss_weight: float = 2.0,
    fine_softargmax_temperature: float = 1.0,
    fine_smooth_l1_beta: float = 1.0,
) -> Dict[str, torch.Tensor]:
    coarse = heatmap_coarse_mse(logits, target_heatmaps)
    pred_corners = soft_argmax_2d(logits, crop_size=crop_size, temperature=fine_softargmax_temperature)
    fine = visible_coordinate_smooth_l1(pred_corners / float(crop_size), target_corners / float(crop_size), valid, beta=fine_smooth_l1_beta)
    total = coarse + fine_loss_weight * fine
    return {"loss": total, "loss_coarse": coarse, "loss_fine": fine}
