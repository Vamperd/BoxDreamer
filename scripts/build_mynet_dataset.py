"""Build MyNet 8-corner heatmap dataset from a BOP-style DJI Action4 dataset.

This script reads the existing `dji-action4` layout:

    dji-action4/
      models/models_info.json
      train_pbr/000000/
        rgb/000000.jpg
        mask/000000_000000.png
        mask_visib/000000_000000.png
        scene_camera.json
        scene_gt.json
        scene_gt_info.json

and writes the MyNet training layout described in plan.md:

    data/dji_action4_mynet/
      meta.json
      train.json
      val.json
      crops/{train,val}/*.png
      heatmaps/{train,val}/*.npy
      debug_vis/{train,val}/*.jpg

Each object instance becomes one sample, so frames containing multiple DJI
Action4 instances naturally produce multiple ROI crops.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


CornerArray = np.ndarray
BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class BuildConfig:
    bop_root: Path
    output_root: Path
    obj_id: int
    crop_size: int
    heatmap_size: int
    sigma: float
    padding_ratio: float
    val_ratio: float
    seed: int
    min_visib_fract: float
    bbox_source: str
    debug_vis_limit: int
    max_samples: Optional[int]
    overwrite: bool


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def path_for_json(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def find_rgb_path(rgb_dir: Path, image_id: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = rgb_dir / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No RGB file found for image id {image_id} in {rgb_dir}")


def bbox_xywh_to_xyxy(box: Sequence[float]) -> Optional[BBox]:
    if len(box) != 4:
        return None
    x, y, w, h = [float(v) for v in box]
    if w <= 0 or h <= 0:
        return None
    return x, y, x + w, y + h


def bbox_from_mask(mask_path: Path) -> Optional[BBox]:
    if not mask_path.exists():
        return None
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def valid_bbox(box: Optional[BBox]) -> bool:
    if box is None:
        return False
    x1, y1, x2, y2 = box
    return x2 > x1 and y2 > y1 and all(math.isfinite(v) for v in box)


def bbox_from_points(points: np.ndarray) -> Optional[BBox]:
    if points.size == 0 or not np.isfinite(points).all():
        return None
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def make_corners_from_models_info(models_info: Dict[str, Any], obj_id: int) -> np.ndarray:
    info = models_info[str(obj_id)]
    min_x, max_x = float(info["min_x"]), float(info["max_x"])
    min_y, max_y = float(info["min_y"]), float(info["max_y"])
    min_z, max_z = float(info["min_z"]), float(info["max_z"])

    # Same stable order as plan.md and BoxDreamer-style bbox consistency.
    corners = np.array(
        [
            [min_x, min_y, min_z],
            [min_x, max_y, min_z],
            [max_x, max_y, min_z],
            [max_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, max_z],
            [max_x, max_y, max_z],
            [max_x, min_y, max_z],
        ],
        dtype=np.float32,
    )
    return corners


def project_points(corners_3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    pts_cam = (R @ corners_3d.T).T + t.reshape(1, 3)
    z = pts_cam[:, 2:3]
    if np.any(np.abs(z) < 1e-6):
        z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    pts_norm = pts_cam[:, :2] / z
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    uv = np.empty((corners_3d.shape[0], 2), dtype=np.float32)
    uv[:, 0] = fx * pts_norm[:, 0] + cx
    uv[:, 1] = fy * pts_norm[:, 1] + cy
    return uv


def choose_base_bbox(
    cfg: BuildConfig,
    scene_dir: Path,
    image_id: str,
    inst_idx: int,
    info: Dict[str, Any],
    projected_corners: np.ndarray,
) -> Tuple[Optional[BBox], str]:
    mask_path = scene_dir / "mask" / f"{image_id}_{inst_idx:06d}.png"
    mask_visib_path = scene_dir / "mask_visib" / f"{image_id}_{inst_idx:06d}.png"

    candidates: List[Tuple[str, Optional[BBox]]] = []
    if cfg.bbox_source == "auto":
        candidates = [
            ("mask_visib", bbox_from_mask(mask_visib_path)),
            ("bbox_visib", bbox_xywh_to_xyxy(info.get("bbox_visib", []))),
            ("mask", bbox_from_mask(mask_path)),
            ("bbox_obj", bbox_xywh_to_xyxy(info.get("bbox_obj", []))),
            ("projected_corners", bbox_from_points(projected_corners)),
        ]
    elif cfg.bbox_source == "mask_visib":
        candidates = [("mask_visib", bbox_from_mask(mask_visib_path))]
    elif cfg.bbox_source == "mask":
        candidates = [("mask", bbox_from_mask(mask_path))]
    elif cfg.bbox_source == "bbox_visib":
        candidates = [("bbox_visib", bbox_xywh_to_xyxy(info.get("bbox_visib", [])))]
    elif cfg.bbox_source == "bbox_obj":
        candidates = [("bbox_obj", bbox_xywh_to_xyxy(info.get("bbox_obj", [])))]
    elif cfg.bbox_source == "projected_corners":
        candidates = [("projected_corners", bbox_from_points(projected_corners))]
    else:
        raise ValueError(f"Unsupported bbox source: {cfg.bbox_source}")

    for source, box in candidates:
        if valid_bbox(box):
            return box, source
    return None, "none"


def make_square_crop_box(box: BBox, image_size: Tuple[int, int], padding_ratio: float) -> BBox:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    side = max(w, h) * (1.0 + 2.0 * padding_ratio)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    half = side * 0.5
    return cx - half, cy - half, cx + half, cy + half


def crop_with_padding(image: Image.Image, crop_box: BBox, crop_size: int) -> Image.Image:
    """Crop a possibly out-of-image square box and resize to crop_size."""
    src = np.asarray(image.convert("RGB"))
    x1, y1, x2, y2 = crop_box
    side = x2 - x1
    if side <= 0:
        raise ValueError(f"Invalid crop box: {crop_box}")

    # Pillow's transform maps output coordinates to source coordinates.
    # This preserves the exact floating-point crop_box used for coordinate mapping.
    return Image.fromarray(src).transform(
        (crop_size, crop_size),
        Image.Transform.AFFINE,
        (side / crop_size, 0.0, x1, 0.0, side / crop_size, y1),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )


def full_to_crop(points: np.ndarray, crop_box: BBox, crop_size: int) -> np.ndarray:
    x1, y1, x2, y2 = crop_box
    side = x2 - x1
    out = np.empty_like(points, dtype=np.float32)
    out[:, 0] = (points[:, 0] - x1) * crop_size / side
    out[:, 1] = (points[:, 1] - y1) * crop_size / side
    return out


def make_heatmaps(
    corners_crop: np.ndarray,
    crop_size: int,
    heatmap_size: int,
    sigma: float,
) -> Tuple[np.ndarray, List[int]]:
    heatmaps = np.zeros((8, heatmap_size, heatmap_size), dtype=np.float32)
    valid: List[int] = []
    yy, xx = np.mgrid[0:heatmap_size, 0:heatmap_size].astype(np.float32)
    scale = heatmap_size / float(crop_size)

    for idx, point in enumerate(corners_crop):
        x_hm = float(point[0] * scale)
        y_hm = float(point[1] * scale)
        is_valid = 0 <= x_hm < heatmap_size and 0 <= y_hm < heatmap_size
        valid.append(1 if is_valid else 0)
        if not is_valid:
            continue
        heatmaps[idx] = np.exp(-((xx - x_hm) ** 2 + (yy - y_hm) ** 2) / (2.0 * sigma**2))

    return heatmaps, valid


def draw_debug_crop(
    crop_img: Image.Image,
    corners_crop: np.ndarray,
    heatmaps: np.ndarray,
    valid: Sequence[int],
) -> Image.Image:
    canvas = crop_img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    colors = [
        (255, 0, 0),
        (255, 128, 0),
        (255, 255, 0),
        (0, 255, 0),
        (0, 255, 255),
        (0, 128, 255),
        (0, 0, 255),
        (255, 0, 255),
    ]

    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        if valid[a] and valid[b]:
            draw.line([tuple(corners_crop[a]), tuple(corners_crop[b])], fill=(255, 255, 255), width=2)

    for idx, (x, y) in enumerate(corners_crop):
        if not valid[idx]:
            continue
        r = 3
        draw.ellipse((x - r, y - r, x + r, y + r), fill=colors[idx])
        draw.text((x + 4, y + 4), str(idx), fill=colors[idx])

    # Add a small heatmap strip on the right side for fast label inspection.
    merged = heatmaps.max(axis=0)
    if merged.max() > 0:
        merged = (merged / merged.max() * 255).astype(np.uint8)
    heat_img = Image.fromarray(merged).resize(canvas.size, Image.Resampling.BILINEAR).convert("RGB")
    overlay = Image.blend(canvas, heat_img, 0.25)
    return overlay


def collect_scene_dirs(train_pbr_root: Path) -> List[Path]:
    return sorted([p for p in train_pbr_root.iterdir() if p.is_dir() and (p / "scene_gt.json").exists()])


def split_key(scene_id: str, image_id: str, val_image_ids: Dict[str, set[str]]) -> str:
    return "val" if image_id in val_image_ids.get(scene_id, set()) else "train"


def build_val_image_ids(scene_dirs: Sequence[Path], val_ratio: float, seed: int) -> Dict[str, set[str]]:
    rng = random.Random(seed)
    result: Dict[str, set[str]] = {}
    for scene_dir in scene_dirs:
        scene_gt = load_json(scene_dir / "scene_gt.json")
        image_ids = sorted(scene_gt.keys(), key=lambda x: int(x))
        shuffled = image_ids[:]
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * val_ratio))) if shuffled and val_ratio > 0 else 0
        result[scene_dir.name] = set(shuffled[:val_count])
    return result


def guard_outputs(cfg: BuildConfig) -> None:
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    protected = [cfg.output_root / "train.json", cfg.output_root / "val.json", cfg.output_root / "meta.json"]
    existing = [p for p in protected if p.exists()]
    if existing and not cfg.overwrite:
        names = ", ".join(str(p) for p in existing)
        raise FileExistsError(f"Output index already exists: {names}. Pass --overwrite to replace index/files.")


def build_dataset(cfg: BuildConfig) -> Dict[str, int]:
    guard_outputs(cfg)

    models_info = load_json(cfg.bop_root / "models" / "models_info.json")
    corners_3d = make_corners_from_models_info(models_info, cfg.obj_id)
    scene_dirs = collect_scene_dirs(cfg.bop_root / "train_pbr")
    val_image_ids = build_val_image_ids(scene_dirs, cfg.val_ratio, cfg.seed)

    repo_root = Path.cwd()
    samples_by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}
    debug_counts = {"train": 0, "val": 0}
    skipped = 0
    written = 0

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        scene_camera = load_json(scene_dir / "scene_camera.json")
        scene_gt = load_json(scene_dir / "scene_gt.json")
        scene_info = load_json(scene_dir / "scene_gt_info.json")
        rgb_dir = scene_dir / "rgb"

        image_ids = sorted(scene_gt.keys(), key=lambda x: int(x))
        for image_id in image_ids:
            rgb_path = find_rgb_path(rgb_dir, f"{int(image_id):06d}")
            image = Image.open(rgb_path).convert("RGB")
            image_w, image_h = image.size
            K = np.array(scene_camera[image_id]["cam_K"], dtype=np.float32).reshape(3, 3)
            split = split_key(scene_id, image_id, val_image_ids)

            gt_entries = scene_gt[image_id]
            info_entries = scene_info[image_id]
            for inst_idx, (gt, info) in enumerate(zip(gt_entries, info_entries)):
                if int(gt["obj_id"]) != cfg.obj_id:
                    continue
                if float(info.get("visib_fract", 1.0)) < cfg.min_visib_fract:
                    skipped += 1
                    continue

                R = np.array(gt["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
                t = np.array(gt["cam_t_m2c"], dtype=np.float32).reshape(3)
                corners_2d_full = project_points(corners_3d, K, R, t)

                base_bbox, bbox_source = choose_base_bbox(cfg, scene_dir, f"{int(image_id):06d}", inst_idx, info, corners_2d_full)
                if not valid_bbox(base_bbox):
                    skipped += 1
                    continue

                crop_box = make_square_crop_box(base_bbox, (image_w, image_h), cfg.padding_ratio)
                crop_img = crop_with_padding(image, crop_box, cfg.crop_size)
                corners_2d_crop = full_to_crop(corners_2d_full, crop_box, cfg.crop_size)
                heatmaps, corner_valid = make_heatmaps(corners_2d_crop, cfg.crop_size, cfg.heatmap_size, cfg.sigma)

                sample_id = f"{scene_id}_{int(image_id):06d}_{inst_idx:06d}"
                crop_path = cfg.output_root / "crops" / split / f"{sample_id}.png"
                heatmap_path = cfg.output_root / "heatmaps" / split / f"{sample_id}.npy"
                debug_path = cfg.output_root / "debug_vis" / split / f"{sample_id}.jpg"

                crop_path.parent.mkdir(parents=True, exist_ok=True)
                heatmap_path.parent.mkdir(parents=True, exist_ok=True)
                crop_img.save(crop_path)
                np.save(heatmap_path, heatmaps)

                if debug_counts[split] < cfg.debug_vis_limit:
                    debug_path.parent.mkdir(parents=True, exist_ok=True)
                    draw_debug_crop(crop_img, corners_2d_crop, heatmaps, corner_valid).save(debug_path, quality=92)
                    debug_counts[split] += 1

                mask_path = scene_dir / "mask_visib" / f"{int(image_id):06d}_{inst_idx:06d}.png"
                if not mask_path.exists():
                    mask_path = scene_dir / "mask" / f"{int(image_id):06d}_{inst_idx:06d}.png"

                sample = {
                    "sample_id": sample_id,
                    "scene_id": scene_id,
                    "image_id": f"{int(image_id):06d}",
                    "instance_idx": inst_idx,
                    "rgb_path": path_for_json(rgb_path, repo_root),
                    "mask_path": path_for_json(mask_path, repo_root) if mask_path.exists() else None,
                    "crop_path": path_for_json(crop_path, repo_root),
                    "heatmap_path": path_for_json(heatmap_path, repo_root),
                    "obj_id": cfg.obj_id,
                    "K": K.tolist(),
                    "R": R.tolist(),
                    "t": t.tolist(),
                    "bbox_source": bbox_source,
                    "bbox_xyxy_full": [float(v) for v in base_bbox],
                    "crop_box_xyxy_full": [float(v) for v in crop_box],
                    "corners_3d": corners_3d.tolist(),
                    "corners_2d_full": corners_2d_full.tolist(),
                    "corners_2d_crop": corners_2d_crop.tolist(),
                    "corner_valid": corner_valid,
                    "visib_fract": float(info.get("visib_fract", 1.0)),
                    "px_count_visib": int(info.get("px_count_visib", 0)),
                    "image_size_full": [image_w, image_h],
                    "crop_size": cfg.crop_size,
                    "heatmap_size": cfg.heatmap_size,
                }
                samples_by_split[split].append(sample)
                written += 1
                if cfg.max_samples is not None and written >= cfg.max_samples:
                    break
            if cfg.max_samples is not None and written >= cfg.max_samples:
                break
        if cfg.max_samples is not None and written >= cfg.max_samples:
            break

    meta = {
        "object_name": "DJI Action 4",
        "obj_id": cfg.obj_id,
        "source_bop_root": path_for_json(cfg.bop_root, repo_root),
        "corner_order": [
            "min_x,min_y,min_z",
            "min_x,max_y,min_z",
            "max_x,max_y,min_z",
            "max_x,min_y,min_z",
            "min_x,min_y,max_z",
            "min_x,max_y,max_z",
            "max_x,max_y,max_z",
            "max_x,min_y,max_z",
        ],
        "corners_3d": corners_3d.tolist(),
        "crop_size": cfg.crop_size,
        "heatmap_size": cfg.heatmap_size,
        "sigma": cfg.sigma,
        "crop_padding_ratio": cfg.padding_ratio,
        "bbox_source": cfg.bbox_source,
        "min_visib_fract": cfg.min_visib_fract,
        "val_ratio": cfg.val_ratio,
        "seed": cfg.seed,
    }

    save_json(cfg.output_root / "meta.json", meta)
    save_json(cfg.output_root / "train.json", samples_by_split["train"])
    save_json(cfg.output_root / "val.json", samples_by_split["val"])

    return {
        "train": len(samples_by_split["train"]),
        "val": len(samples_by_split["val"]),
        "skipped": skipped,
        "debug_train": debug_counts["train"],
        "debug_val": debug_counts["val"],
    }


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(description="Build MyNet DJI Action4 corner heatmap dataset from BOP train_pbr.")
    parser.add_argument("--bop-root", type=Path, default=Path("dji-action4"), help="Input BOP-style DJI Action4 root.")
    parser.add_argument("--output-root", type=Path, default=Path("data/dji_action4_mynet"), help="Output MyNet dataset root.")
    parser.add_argument("--obj-id", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--padding-ratio", type=float, default=0.25)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-visib-fract", type=float, default=0.05)
    parser.add_argument(
        "--bbox-source",
        choices=["auto", "mask_visib", "mask", "bbox_visib", "bbox_obj", "projected_corners"],
        default="auto",
    )
    parser.add_argument("--debug-vis-limit", type=int, default=100, help="Max debug visualizations per split.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing index/crop/heatmap files.")
    args = parser.parse_args()

    return BuildConfig(
        bop_root=args.bop_root,
        output_root=args.output_root,
        obj_id=args.obj_id,
        crop_size=args.crop_size,
        heatmap_size=args.heatmap_size,
        sigma=args.sigma,
        padding_ratio=args.padding_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        min_visib_fract=args.min_visib_fract,
        bbox_source=args.bbox_source,
        debug_vis_limit=args.debug_vis_limit,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
    )


def main() -> None:
    cfg = parse_args()
    stats = build_dataset(cfg)
    print("MyNet dataset build complete:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"  output_root: {cfg.output_root}")


if __name__ == "__main__":
    main()
