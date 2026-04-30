"""Generate per-corner visibility labels for BOP-style DJI Action4 data.

The output is one `scene_gt_corners.json` file per scene. It stores projected
2D box corners, camera-space corner depth, a binary visibility flag, and a
coarse occlusion type for each object instance.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VisibilityConfig:
    bop_root: Path
    corners_meta: Optional[Path]
    obj_id: int
    self_mode: str
    depth_radius: int
    depth_percentile: float
    depth_margin_mm: float
    output_name: str
    max_scenes: Optional[int]
    max_images_per_scene: Optional[int]
    dry_run: bool


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_corners_from_models_info(models_info: Dict[str, Any], obj_id: int) -> np.ndarray:
    info = models_info[str(obj_id)]
    min_x, max_x = float(info["min_x"]), float(info["max_x"])
    min_y, max_y = float(info["min_y"]), float(info["max_y"])
    min_z, max_z = float(info["min_z"]), float(info["max_z"])
    return np.array(
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


def load_corners_3d(cfg: VisibilityConfig) -> np.ndarray:
    if cfg.corners_meta is not None:
        meta = load_json(cfg.corners_meta)
        if "corners_3d" not in meta:
            raise KeyError(f"`corners_3d` not found in corners meta: {cfg.corners_meta}")
        corners = np.array(meta["corners_3d"], dtype=np.float32)
        if corners.shape != (8, 3):
            raise ValueError(f"`corners_3d` must be [8, 3], got {corners.shape}: {cfg.corners_meta}")
        return corners

    models_info_path = cfg.bop_root / "models" / "models_info.json"
    if not models_info_path.exists():
        raise FileNotFoundError(
            f"Could not find {models_info_path}. Pass --corners-meta pointing to a MyNet meta.json with corners_3d."
        )
    return make_corners_from_models_info(load_json(models_info_path), cfg.obj_id)


def transform_corners(corners_3d: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ corners_3d.T).T + t.reshape(1, 3)


def project_camera_points(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = points_cam[:, 2:3]
    safe_z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    pts_norm = points_cam[:, :2] / safe_z
    uv = np.empty((points_cam.shape[0], 2), dtype=np.float32)
    uv[:, 0] = K[0, 0] * pts_norm[:, 0] + K[0, 2]
    uv[:, 1] = K[1, 1] * pts_norm[:, 1] + K[1, 2]
    return uv


def cuboid_self_visible_corners(corners_3d: np.ndarray, R: np.ndarray, t: np.ndarray) -> List[bool]:
    """Approximate self visibility with front-facing cuboid faces.

    A corner is considered self-visible if it belongs to at least one cuboid
    face whose outward normal points toward the camera.
    """
    visible = [False] * len(corners_3d)
    eps = 1e-5
    for dim in range(3):
        for side_value, sign in ((float(corners_3d[:, dim].min()), -1.0), (float(corners_3d[:, dim].max()), 1.0)):
            face_indices = np.where(np.abs(corners_3d[:, dim] - side_value) <= eps)[0]
            if face_indices.size == 0:
                continue
            normal_obj = np.zeros(3, dtype=np.float32)
            normal_obj[dim] = sign
            normal_cam = R @ normal_obj
            face_center_cam = transform_corners(corners_3d[face_indices], R, t).mean(axis=0)
            points_toward_camera = float(np.dot(normal_cam, -face_center_cam)) > 0.0
            if points_toward_camera:
                for idx in face_indices:
                    visible[int(idx)] = True
    return visible


def load_depth(scene_dir: Path, image_id: str, depth_scale: float) -> Optional[np.ndarray]:
    depth_path = scene_dir / "depth" / f"{int(image_id):06d}.png"
    if not depth_path.exists():
        return None
    return np.array(Image.open(depth_path), dtype=np.float32) * float(depth_scale)


def depth_has_closer_surface(
    depth: Optional[np.ndarray],
    uv: Sequence[float],
    z_cam: float,
    radius: int,
    percentile: float,
    margin_mm: float,
) -> bool:
    if depth is None:
        return False
    height, width = depth.shape
    x = int(round(float(uv[0])))
    y = int(round(float(uv[1])))
    if x < 0 or x >= width or y < 0 or y >= height:
        return False

    x1 = max(0, x - radius)
    x2 = min(width, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(height, y + radius + 1)
    values = depth[y1:y2, x1:x2]
    values = values[values > 0]
    if values.size == 0:
        return False
    observed = float(np.percentile(values, percentile))
    return observed < float(z_cam) - float(margin_mm)


def classify_corners(
    corners_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_size: Tuple[int, int],
    depth: Optional[np.ndarray],
    cfg: VisibilityConfig,
) -> Dict[str, Any]:
    points_cam = transform_corners(corners_3d, R, t)
    corners_2d = project_camera_points(points_cam, K)
    width, height = image_size

    if cfg.self_mode == "none":
        self_visible = [True] * len(corners_3d)
    elif cfg.self_mode == "bbox":
        self_visible = cuboid_self_visible_corners(corners_3d, R, t)
    else:
        raise ValueError(f"Unsupported self-mode: {cfg.self_mode}")

    occ_type: List[str] = []
    visible: List[int] = []
    for idx, (uv, point_cam) in enumerate(zip(corners_2d, points_cam)):
        u, v = float(uv[0]), float(uv[1])
        z_cam = float(point_cam[2])
        if z_cam <= 0 or u < 0 or u >= width or v < 0 or v >= height:
            kind = "outside"
        elif not self_visible[idx]:
            kind = "self"
        elif depth_has_closer_surface(depth, uv, z_cam, cfg.depth_radius, cfg.depth_percentile, cfg.depth_margin_mm):
            kind = "external"
        else:
            kind = "visible"
        occ_type.append(kind)
        visible.append(1 if kind == "visible" else 0)

    return {
        "corners_2d": [[float(x), float(y)] for x, y in corners_2d],
        "corners_depth": [float(v) for v in points_cam[:, 2]],
        "corners_visib": visible,
        "corners_occ_type": occ_type,
    }


def collect_scene_dirs(train_pbr_root: Path) -> List[Path]:
    return sorted([p for p in train_pbr_root.iterdir() if p.is_dir() and (p / "scene_gt.json").exists()])


def infer_image_size(scene_dir: Path, image_id: str) -> Tuple[int, int]:
    rgb_dir = scene_dir / "rgb"
    for suffix in (".png", ".jpg", ".jpeg"):
        path = rgb_dir / f"{int(image_id):06d}{suffix}"
        if path.exists():
            with Image.open(path) as image:
                return image.size
    depth_path = scene_dir / "depth" / f"{int(image_id):06d}.png"
    if depth_path.exists():
        with Image.open(depth_path) as depth:
            width, height = depth.size
            return width, height
    raise FileNotFoundError(f"No RGB or depth image found for image {image_id} in {scene_dir}")


def process_scene(scene_dir: Path, corners_3d: np.ndarray, cfg: VisibilityConfig) -> Counter:
    scene_camera = load_json(scene_dir / "scene_camera.json")
    scene_gt = load_json(scene_dir / "scene_gt.json")
    frames: Dict[str, List[Dict[str, Any]]] = {}
    stats: Counter = Counter()

    image_ids = sorted(scene_gt.keys(), key=lambda item: int(item))
    if cfg.max_images_per_scene is not None:
        image_ids = image_ids[: cfg.max_images_per_scene]

    for image_id in image_ids:
        camera = scene_camera[image_id]
        K = np.array(camera["cam_K"], dtype=np.float32).reshape(3, 3)
        depth = load_depth(scene_dir, image_id, float(camera.get("depth_scale", 1.0)))
        image_size = infer_image_size(scene_dir, image_id)
        entries: List[Dict[str, Any]] = []

        for inst_idx, gt in enumerate(scene_gt[image_id]):
            obj_id = int(gt["obj_id"])
            if obj_id != cfg.obj_id:
                entries.append({"obj_id": obj_id, "instance_idx": int(inst_idx), "unsupported": True})
                continue
            R = np.array(gt["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
            t = np.array(gt["cam_t_m2c"], dtype=np.float32).reshape(3)
            corner_record = classify_corners(corners_3d, K, R, t, image_size, depth, cfg)
            corner_record["obj_id"] = obj_id
            corner_record["instance_idx"] = int(inst_idx)
            entries.append(corner_record)
            stats["instances"] += 1
            for kind in corner_record["corners_occ_type"]:
                stats[f"corner_{kind}"] += 1
        frames[image_id] = entries
        stats["frames"] += 1

    output = {
        "meta": {
            "generator": "scripts/gen_dji_corner_visibility.py",
            "obj_id": cfg.obj_id,
            "self_mode": cfg.self_mode,
            "depth_radius": cfg.depth_radius,
            "depth_percentile": cfg.depth_percentile,
            "depth_margin_mm": cfg.depth_margin_mm,
        },
        "frames": frames,
    }
    if not cfg.dry_run:
        save_json(scene_dir / cfg.output_name, output)
    stats["scenes"] += 1
    return stats


def merge_counters(counters: Iterable[Counter]) -> Counter:
    total: Counter = Counter()
    for counter in counters:
        total.update(counter)
    return total


def parse_args() -> VisibilityConfig:
    parser = argparse.ArgumentParser(description="Generate DJI Action4 per-corner visibility labels from BOP depth and poses.")
    parser.add_argument("--bop-root", type=Path, required=True)
    parser.add_argument("--corners-meta", type=Path, default=None, help="MyNet meta.json containing corners_3d. Required if bop-root/models is absent.")
    parser.add_argument("--obj-id", type=int, default=1)
    parser.add_argument("--self-mode", choices=["bbox", "none"], default="bbox")
    parser.add_argument("--depth-radius", type=int, default=5)
    parser.add_argument("--depth-percentile", type=float, default=10.0)
    parser.add_argument("--depth-margin-mm", type=float, default=15.0)
    parser.add_argument("--output-name", type=str, default="scene_gt_corners.json")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-images-per-scene", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Compute and print stats without writing scene_gt_corners.json.")
    args = parser.parse_args()
    if args.depth_radius < 0:
        raise ValueError("--depth-radius must be non-negative.")
    if not 0 <= args.depth_percentile <= 100:
        raise ValueError("--depth-percentile must be between 0 and 100.")
    return VisibilityConfig(
        bop_root=args.bop_root,
        corners_meta=args.corners_meta,
        obj_id=args.obj_id,
        self_mode=args.self_mode,
        depth_radius=args.depth_radius,
        depth_percentile=args.depth_percentile,
        depth_margin_mm=args.depth_margin_mm,
        output_name=args.output_name,
        max_scenes=args.max_scenes,
        max_images_per_scene=args.max_images_per_scene,
        dry_run=args.dry_run,
    )


def main() -> None:
    cfg = parse_args()
    corners_3d = load_corners_3d(cfg)
    scene_dirs = collect_scene_dirs(cfg.bop_root / "train_pbr")
    if cfg.max_scenes is not None:
        scene_dirs = scene_dirs[: cfg.max_scenes]
    stats = merge_counters(process_scene(scene_dir, corners_3d, cfg) for scene_dir in scene_dirs)

    print("Corner visibility generation complete:")
    for key in sorted(stats.keys()):
        print(f"  {key}: {stats[key]}")
    if cfg.dry_run:
        print("  dry_run: true")


if __name__ == "__main__":
    main()
