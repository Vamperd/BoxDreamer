from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.mynet.dataset import IMAGENET_MEAN, IMAGENET_STD, INPUT_MODE_FIXED, INPUT_MODE_RECT_DYNAMIC, clip_bbox_to_image
from src.mynet.decode import decode_heatmap


BBox = Tuple[float, float, float, float]


COLORS = [
    (255, 0, 0),
    (255, 128, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 255, 255),
    (0, 128, 255),
    (0, 0, 255),
    (255, 0, 255),
]
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MyNet 8-corner heatmap inference on one image.")
    parser.add_argument("--image", type=Path, required=True, help="Input RGB image, for example image.png.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained checkpoint, usually best.pt.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mynet_infer"))
    parser.add_argument(
        "--bbox",
        action="append",
        default=[],
        help="Object bbox as x1,y1,x2,y2 in full-image pixels. Repeat once or twice for DJI Action4 instances.",
    )
    parser.add_argument("--bbox-json", type=Path, default=None, help="Optional JSON containing a list of bboxes or {'bboxes': [...]}.")
    parser.add_argument("--min-bboxes", type=int, default=1, help="Minimum number of bboxes to run.")
    parser.add_argument("--max-bboxes", type=int, default=2, help="Maximum number of bboxes to run.")
    parser.add_argument("--input-mode", choices=["auto", "fixed", "rect_dynamic"], default="auto")
    parser.add_argument("--bbox-padding", type=float, default=0.10, help="Padding ratio applied before square ROI crop.")
    parser.add_argument(
        "--bbox-padding-pixels",
        type=float,
        default=None,
        help="Absolute padding pixels added to each side before square ROI crop. Overrides --bbox-padding when set.",
    )
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--decode-method", choices=["argmax", "subpixel"], default="subpixel")
    parser.add_argument("--subpixel-window", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-crops", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_bbox(text: str) -> BBox:
    parts = [float(item.strip()) for item in text.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected bbox x1,y1,x2,y2, got: {text}")
    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox with non-positive size: {text}")
    return x1, y1, x2, y2


def normalize_bbox_item(item: Sequence[float]) -> BBox:
    if len(item) != 4:
        raise ValueError(f"Expected bbox with 4 values, got: {item}")
    x1, y1, x2, y2 = (float(v) for v in item)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox with non-positive size: {item}")
    return x1, y1, x2, y2


def load_bboxes(args: argparse.Namespace) -> List[BBox]:
    bboxes = [parse_bbox(item) for item in args.bbox]
    if args.bbox_json is not None:
        with args.bbox_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
        items = data["bboxes"] if isinstance(data, dict) else data
        bboxes.extend(normalize_bbox_item(item) for item in items)
    if not (args.min_bboxes <= len(bboxes) <= args.max_bboxes):
        raise ValueError(
            f"Expected {args.min_bboxes} to {args.max_bboxes} bbox(es), got {len(bboxes)}. "
            "MyNet expects object ROI crops, not a full image."
        )
    return bboxes


def make_square_crop_box(box: BBox, padding_ratio: float, padding_pixels: Optional[float] = None) -> BBox:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if padding_pixels is not None:
        side = max(w, h) + 2.0 * padding_pixels
    else:
        side = max(w, h) * (1.0 + 2.0 * padding_ratio)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    half = side * 0.5
    return cx - half, cy - half, cx + half, cy + half


def crop_with_padding(image: Image.Image, crop_box: BBox, crop_size: int) -> Image.Image:
    src = np.asarray(image.convert("RGB"))
    x1, y1, x2, y2 = crop_box
    side = x2 - x1
    if side <= 0:
        raise ValueError(f"Invalid crop box: {crop_box}")
    return Image.fromarray(src).transform(
        (crop_size, crop_size),
        Image.Transform.AFFINE,
        (side / crop_size, 0.0, x1, 0.0, side / crop_size, y1),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )


def crop_rect(image: Image.Image, bbox: BBox) -> Tuple[Image.Image, BBox]:
    crop_box = clip_bbox_to_image(bbox, image.size)
    if crop_box is None:
        raise ValueError(f"Invalid bbox after clipping to image bounds: {bbox}")
    left, top, right, bottom = [int(v) for v in crop_box]
    return image.crop((left, top, right, bottom)), crop_box


def crop_to_full(points: np.ndarray, crop_box: BBox, crop_size: Optional[int] = None) -> np.ndarray:
    x1, y1, x2, y2 = crop_box
    out = np.empty_like(points, dtype=np.float32)
    if crop_size is None:
        out[:, 0] = x1 + points[:, 0]
        out[:, 1] = y1 + points[:, 1]
        return out

    side = x2 - x1
    out[:, 0] = x1 + points[:, 0] * side / crop_size
    out[:, 1] = y1 + points[:, 1] * side / crop_size
    return out


def preprocess(crop: Image.Image) -> torch.Tensor:
    image_np = np.asarray(crop.convert("RGB"), dtype=np.float32) / 255.0
    image = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image.unsqueeze(0)


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    from src.mynet.model import CornerResNet34

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model = CornerResNet34(out_channels=8, pretrained=False)
    model.load_state_dict(state_dict)
    args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    model.input_mode = str(args.get("input_mode", INPUT_MODE_FIXED))
    model.to(device)
    model.eval()
    return model


def draw_prediction(canvas: Image.Image, bbox: BBox, crop_box: BBox, corners: np.ndarray, scores: Sequence[float], instance_id: int) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(bbox, outline=(255, 255, 255), width=2)
    draw.rectangle(crop_box, outline=(80, 180, 255), width=2)
    draw.text((bbox[0] + 3, bbox[1] + 3), f"roi {instance_id}", fill=(255, 255, 255))

    for a, b in EDGES:
        draw.line([tuple(corners[a]), tuple(corners[b])], fill=(255, 255, 255), width=2)

    for idx, (x, y) in enumerate(corners):
        color = COLORS[idx]
        r = 4
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        draw.text((x + 5, y + 5), f"{idx}:{scores[idx]:.2f}", fill=color)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.bbox_padding < 0:
        raise ValueError("--bbox-padding must be non-negative.")
    if args.bbox_padding_pixels is not None and args.bbox_padding_pixels < 0:
        raise ValueError("--bbox-padding-pixels must be non-negative.")
    device = torch.device(args.device)
    bboxes = load_bboxes(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert("RGB")
    model = load_model(args.checkpoint, device)
    input_mode = getattr(model, "input_mode", INPUT_MODE_FIXED) if args.input_mode == "auto" else args.input_mode

    canvas = image.copy()
    predictions: List[Dict[str, Any]] = []
    for inst_idx, bbox in enumerate(bboxes):
        if input_mode == INPUT_MODE_RECT_DYNAMIC:
            crop, crop_box = crop_rect(image, bbox)
        else:
            crop_box = make_square_crop_box(bbox, args.bbox_padding, args.bbox_padding_pixels)
            crop = crop_with_padding(image, crop_box, args.crop_size)
        tensor = preprocess(crop).to(device)

        logits = model(tensor)
        crop_hw = torch.tensor([[crop.height, crop.width]], device=device, dtype=torch.float32) if input_mode == INPUT_MODE_RECT_DYNAMIC else None
        corners_crop = decode_heatmap(
            logits,
            crop_size=args.crop_size,
            crop_hw=crop_hw,
            decode_method=args.decode_method,
            subpixel_window=args.subpixel_window,
        )[0].cpu().numpy()
        probs = torch.sigmoid(logits).flatten(2)
        scores = probs.max(dim=2).values[0].cpu().numpy().astype(float)
        corners_full = crop_to_full(corners_crop, crop_box, None if input_mode == INPUT_MODE_RECT_DYNAMIC else args.crop_size)

        draw_prediction(canvas, bbox, crop_box, corners_full, scores, inst_idx)
        if args.save_crops:
            crop.save(args.output_dir / f"roi_{inst_idx:02d}.png")

        predictions.append(
            {
                "instance_id": inst_idx,
                "bbox_xyxy_full": [float(v) for v in bbox],
                "crop_box_xyxy_full": [float(v) for v in crop_box],
                "corners_2d_crop": corners_crop.astype(float).tolist(),
                "corners_2d_full": corners_full.astype(float).tolist(),
                "corner_scores": [float(v) for v in scores],
            }
        )

    canvas.save(args.output_dir / "prediction_overlay.png")
    with (args.output_dir / "predictions.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "image": str(args.image),
                "checkpoint": str(args.checkpoint),
                "input_mode": input_mode,
                "bbox_padding": args.bbox_padding,
                "bbox_padding_pixels": args.bbox_padding_pixels,
                "crop_size": args.crop_size,
                "decode_method": args.decode_method,
                "subpixel_window": args.subpixel_window,
                "predictions": predictions,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"saved overlay: {args.output_dir / 'prediction_overlay.png'}")
    print(f"saved predictions: {args.output_dir / 'predictions.json'}")


if __name__ == "__main__":
    main()
