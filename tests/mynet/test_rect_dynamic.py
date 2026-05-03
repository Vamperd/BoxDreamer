import json

import numpy as np
import torch
from PIL import Image

from src.mynet.dataset import BOPCornerDataset, RECT_BATCH_MODE_ASPECT_BUCKET, SCALE_AUG_NONE, SCALE_AUG_REAL_VIDEO_COVERAGE, collate_corner_batch
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


def test_rect_dynamic_batch_scale_aug_uses_one_size_without_padding(tmp_path):
    image_path = tmp_path / "frame.png"
    Image.fromarray(np.zeros((160, 240, 3), dtype=np.uint8)).save(image_path)
    index_path = tmp_path / "train.json"
    sample_a = {
        "sample_id": "sample_0",
        "input_mode": "rect_dynamic",
        "rgb_path": str(image_path),
        "bbox_xyxy_full": [10, 12, 70, 52],
        "crop_box_xyxy_full": [0, 0, 240, 160],
        "corners_2d_full": [[15, 17], [65, 17], [65, 47], [15, 47], [20, 22], [60, 22], [60, 42], [20, 42]],
        "corner_visible": [1] * 8,
        "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "t": [0, 0, 1],
    }
    sample_b = {
        **sample_a,
        "sample_id": "sample_1",
        "bbox_xyxy_full": [20, 24, 140, 104],
        "corners_2d_full": [[30, 34], [130, 34], [130, 94], [30, 94], [40, 44], [120, 44], [120, 84], [40, 84]],
    }
    index_path.write_text(json.dumps([sample_a, sample_b]), encoding="utf-8")

    dataset = BOPCornerDataset(
        index_path,
        input_mode="rect_dynamic",
        scale_aug_mode=SCALE_AUG_NONE,
    )
    batch = collate_corner_batch(
        [dataset[0], dataset[1]],
        rect_batch_mode=RECT_BATCH_MODE_ASPECT_BUCKET,
        scale_aug_mode=SCALE_AUG_REAL_VIDEO_COVERAGE,
        scale_long_edge_min=320,
        scale_long_edge_max=320,
        scale_short_edge_min=180,
    )

    assert tuple(batch["image"].shape) == (2, 3, 213, 320)
    assert tuple(batch["heatmap"].shape) == (2, 8, 54, 80)
    assert batch["crop_hw"].tolist() == [[213.0, 320.0], [213.0, 320.0]]
    assert batch["original_crop_hw"].tolist() == [[40.0, 60.0], [80.0, 120.0]]
    assert torch.allclose(batch["corners_2d_crop"][0, 0], torch.tensor([26.6667, 26.6250]), atol=1e-4)
    assert torch.allclose(batch["scale_xy"][0], torch.tensor([5.3333, 5.3250]), atol=1e-4)


def test_corner_loss_uses_per_sample_normalization():
    logits = torch.zeros((2, 8, 2, 2), dtype=torch.float32)
    target = torch.empty_like(logits)
    target[0].fill_(1.0)
    target[1].fill_(0.5)
    valid = torch.zeros((2, 8), dtype=torch.bool)
    valid[0, :] = True
    valid[1, 0] = True

    loss = corner_loss(logits, target, corner_valid=valid)["loss"]

    assert torch.isclose(loss, torch.tensor(0.125))
