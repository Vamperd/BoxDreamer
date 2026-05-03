import json

import numpy as np
import torch
from PIL import Image

from src.mynet.dataset import BOPCornerDataset, SCALE_AUG_NONE, SCALE_AUG_REAL_VIDEO_COVERAGE, collate_corner_batch
from src.mynet.decode import decode_heatmap
from src.mynet.losses import corner_loss


def test_rect_dynamic_dataset_uses_tight_bbox(tmp_path):
    image_path = tmp_path / "frame.png"
    Image.fromarray(np.zeros((80, 120, 3), dtype=np.uint8)).save(image_path)
    index_path = tmp_path / "train.json"
    sample = {
        "sample_id": "sample_0",
        "input_mode": "rect_dynamic",
        "rgb_path": str(image_path),
        "bbox_xyxy_full": [10, 12, 70, 52],
        "crop_box_xyxy_full": [0, 0, 120, 80],
        "corners_2d_full": [[15, 17], [65, 17], [65, 47], [15, 47], [20, 22], [60, 22], [60, 42], [20, 42]],
        "corner_visible": [1] * 8,
        "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "t": [0, 0, 1],
    }
    index_path.write_text(json.dumps([sample]), encoding="utf-8")

    item = BOPCornerDataset(index_path, input_mode="rect_dynamic", scale_aug_mode=SCALE_AUG_NONE)[0]

    assert tuple(item["image"].shape) == (3, 40, 60)
    assert tuple(item["heatmap"].shape) == (8, 10, 15)
    assert item["crop_hw"].tolist() == [40.0, 60.0]
    assert item["corners_2d_crop"][0].tolist() == [5.0, 5.0]


def test_rect_dynamic_loss_and_decode_accept_crop_hw(tmp_path):
    image_path = tmp_path / "frame.png"
    Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8)).save(image_path)
    index_path = tmp_path / "train.json"
    sample = {
        "sample_id": "sample_0",
        "input_mode": "rect_dynamic",
        "rgb_path": str(image_path),
        "bbox_xyxy_full": [0, 0, 60, 40],
        "corners_2d_full": [[5, 5], [55, 5], [55, 35], [5, 35], [10, 10], [50, 10], [50, 30], [10, 30]],
        "corner_visible": [1] * 8,
        "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "t": [0, 0, 1],
    }
    index_path.write_text(json.dumps([sample]), encoding="utf-8")

    batch = collate_corner_batch([BOPCornerDataset(index_path, input_mode="rect_dynamic", scale_aug_mode=SCALE_AUG_NONE)[0]])
    logits = torch.zeros_like(batch["heatmap"])

    loss = corner_loss(logits, batch["heatmap"], corner_valid=batch["corner_valid"])["loss"]
    decoded = decode_heatmap(logits, crop_hw=batch["crop_hw"], decode_method="argmax")

    assert loss.ndim == 0
    assert decoded.shape == (1, 8, 2)


def test_rect_dynamic_scale_aug_resizes_tight_bbox_without_padding(tmp_path):
    image_path = tmp_path / "frame.png"
    Image.fromarray(np.zeros((80, 120, 3), dtype=np.uint8)).save(image_path)
    index_path = tmp_path / "train.json"
    sample = {
        "sample_id": "sample_0",
        "input_mode": "rect_dynamic",
        "rgb_path": str(image_path),
        "bbox_xyxy_full": [10, 12, 70, 52],
        "crop_box_xyxy_full": [0, 0, 120, 80],
        "corners_2d_full": [[15, 17], [65, 17], [65, 47], [15, 47], [20, 22], [60, 22], [60, 42], [20, 42]],
        "corner_visible": [1] * 8,
        "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "t": [0, 0, 1],
    }
    index_path.write_text(json.dumps([sample]), encoding="utf-8")

    dataset = BOPCornerDataset(
        index_path,
        input_mode="rect_dynamic",
        scale_aug_mode=SCALE_AUG_REAL_VIDEO_COVERAGE,
        scale_long_edge_min=320,
        scale_long_edge_max=768,
        scale_short_edge_min=180,
    )
    items = [dataset[0] for _ in range(8)]
    sizes = {tuple(int(v) for v in item["crop_hw"].tolist()) for item in items}

    assert len(sizes) > 1
    for item in items:
        crop_h, crop_w = [int(v) for v in item["crop_hw"].tolist()]
        assert 320 <= max(crop_h, crop_w) <= 768
        assert min(crop_h, crop_w) >= 180
        assert item["original_crop_hw"].tolist() == [40.0, 60.0]
        assert item["scale_factor"].item() > 1.0
