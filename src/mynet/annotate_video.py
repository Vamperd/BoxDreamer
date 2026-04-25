from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


BBox = Tuple[float, float, float, float]

CORNER_ORDER = [
    "min_x,min_y,min_z",
    "min_x,max_y,min_z",
    "max_x,max_y,min_z",
    "max_x,min_y,min_z",
    "min_x,min_y,max_z",
    "min_x,max_y,max_z",
    "max_x,max_y,max_z",
    "max_x,min_y,max_z",
]

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
    parser = argparse.ArgumentParser(description="Annotate real video frames for YOLO detection and MyNet corner fine-tuning.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-every-sec", type=float, default=1.0)
    parser.add_argument("--start-sec", type=float, default=0.0, help="Start annotation from this timestamp in seconds.")
    parser.add_argument("--end-sec", type=float, default=None, help="Stop annotation before this timestamp in seconds.")
    parser.add_argument("--min-bboxes", type=int, default=1)
    parser.add_argument("--max-bboxes", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--bbox-padding", type=float, default=0.25)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-display-side", type=int, default=1400)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional smoke-test limit on sampled candidate frames.")
    parser.add_argument("--overwrite", action="store_true", help="Start a fresh annotations.json instead of resuming.")
    return parser.parse_args()


def load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install it with: uv pip install opencv-python") from exc
    return cv2


def ensure_args(args: argparse.Namespace) -> None:
    if args.sample_every_sec <= 0:
        raise ValueError("--sample-every-sec must be positive.")
    if args.start_sec < 0:
        raise ValueError("--start-sec must be >= 0.")
    if args.end_sec is not None and args.end_sec <= args.start_sec:
        raise ValueError("--end-sec must be greater than --start-sec.")
    if args.min_bboxes < 0 or args.max_bboxes < args.min_bboxes:
        raise ValueError("Invalid --min-bboxes/--max-bboxes.")
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("--train-ratio must be > 0, --val-ratio must be >= 0, and their sum must be < 1.")
    if args.crop_size <= 0 or args.heatmap_size <= 0:
        raise ValueError("--crop-size and --heatmap-size must be positive.")


def relpath(path: Path, root: Path) -> str:
    return os.path.relpath(path, root).replace("\\", "/")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scale_image_for_display(cv2: Any, image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    side = max(height, width)
    if side <= max_side:
        return image.copy(), 1.0
    scale = max_side / float(side)
    display = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    return display, scale


def scaled_bbox(box: Sequence[float], scale: float) -> tuple[tuple[int, int], tuple[int, int]]:
    x1, y1, x2, y2 = box
    return (int(round(x1 * scale)), int(round(y1 * scale))), (int(round(x2 * scale)), int(round(y2 * scale)))


def xywh_to_xyxy(box: Sequence[float], scale: float) -> BBox:
    x, y, w, h = box
    x1 = x / scale
    y1 = y / scale
    x2 = (x + w) / scale
    y2 = (y + h) / scale
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox: {box}")
    return round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)


def draw_text_panel(cv2: Any, canvas: np.ndarray, lines: Sequence[str]) -> None:
    if not lines:
        return
    line_h = 24
    panel_h = 12 + line_h * len(lines)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], panel_h), (0, 0, 0), -1)
    for idx, line in enumerate(lines):
        cv2.putText(canvas, line, (10, 26 + idx * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def wait_action(cv2: Any, image: np.ndarray, lines: Sequence[str], keys: Sequence[str], window_name: str) -> str:
    while True:
        canvas = image.copy()
        draw_text_panel(cv2, canvas, lines)
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(0) & 0xFF
        if key == 27 and "s" in keys:
            return "s"
        char = chr(key).lower() if 0 <= key < 256 else ""
        if char in keys:
            return char


def draw_boxes(cv2: Any, image: np.ndarray, bboxes: Sequence[BBox], scale: float = 1.0) -> np.ndarray:
    canvas = image.copy()
    for idx, box in enumerate(bboxes):
        p1, p2 = scaled_bbox(box, scale)
        cv2.rectangle(canvas, p1, p2, (255, 255, 255), 2)
        cv2.putText(canvas, f"obj {idx}", (p1[0] + 4, p1[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return canvas


def select_bboxes_for_frame(cv2: Any, image: np.ndarray, args: argparse.Namespace, frame_idx: int) -> Optional[List[BBox]]:
    display, scale = scale_image_for_display(cv2, image, args.max_display_side)
    while True:
        window_name = f"frame {frame_idx}: select {args.min_bboxes}-{args.max_bboxes} bboxes"
        print(f"Frame {frame_idx}: draw one bbox per DJI Action4. Enter/Space confirms each ROI; Esc finishes selection.")
        rois = cv2.selectROIs(window_name, display, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window_name)

        bboxes = [xywh_to_xyxy([float(v) for v in roi], scale) for roi in rois.tolist() if roi[2] > 0 and roi[3] > 0]
        bboxes = sorted(bboxes, key=lambda item: item[0])
        if args.min_bboxes <= len(bboxes) <= args.max_bboxes:
            return bboxes

        preview = draw_boxes(cv2, display, bboxes, scale=scale)
        action = wait_action(
            cv2,
            preview,
            [
                f"Selected {len(bboxes)} bbox(es), expected {args.min_bboxes}-{args.max_bboxes}.",
                "r: retry bbox selection | s/Esc: skip frame | q: save and quit",
            ],
            ["r", "s", "q"],
            "bbox selection action",
        )
        cv2.destroyWindow("bbox selection action")
        if action == "r":
            continue
        if action == "s":
            return []
        if action == "q":
            return None


def draw_corner_state(
    cv2: Any,
    display: np.ndarray,
    scale: float,
    bbox: BBox,
    corners: Sequence[Optional[List[float]]],
    valid: Sequence[int],
    current_idx: int,
    instance_idx: int,
) -> np.ndarray:
    canvas = display.copy()
    p1, p2 = scaled_bbox(bbox, scale)
    cv2.rectangle(canvas, p1, p2, (255, 255, 255), 2)

    for a, b in EDGES:
        if valid[a] and valid[b] and corners[a] is not None and corners[b] is not None:
            pa = (int(round(corners[a][0] * scale)), int(round(corners[a][1] * scale)))
            pb = (int(round(corners[b][0] * scale)), int(round(corners[b][1] * scale)))
            cv2.line(canvas, pa, pb, (255, 255, 255), 2)

    for idx, point in enumerate(corners):
        if point is None or not valid[idx]:
            continue
        color = COLORS[idx]
        bgr = color[2], color[1], color[0]
        x, y = int(round(point[0] * scale)), int(round(point[1] * scale))
        cv2.circle(canvas, (x, y), 5, bgr, -1)
        cv2.putText(canvas, str(idx), (x + 6, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)

    if current_idx < 8:
        current = f"current corner {current_idx}: {CORNER_ORDER[current_idx]}"
    else:
        current = "all 8 corners are marked; press Enter to finish this instance"
    draw_text_panel(
        cv2,
        canvas,
        [
            f"instance {instance_idx} | {current}",
            "left click: mark corner | i: invalid | u: undo | r: restart instance | Enter: finish | Esc: skip frame | q: quit",
        ],
    )
    return canvas


def annotate_instance_corners(
    cv2: Any,
    image: np.ndarray,
    bbox: BBox,
    instance_idx: int,
    max_display_side: int,
) -> tuple[str, Optional[List[List[float]]], Optional[List[int]]]:
    display, scale = scale_image_for_display(cv2, image, max_display_side)
    corners: List[Optional[List[float]]] = [None] * 8
    valid: List[int] = [0] * 8
    current_idx = 0
    window_name = f"instance {instance_idx}: annotate 8 corners"

    def redraw() -> None:
        canvas = draw_corner_state(cv2, display, scale, bbox, corners, valid, current_idx, instance_idx)
        cv2.imshow(window_name, canvas)

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal current_idx
        if event != cv2.EVENT_LBUTTONDOWN or current_idx >= 8:
            return
        corners[current_idx] = [round(float(x / scale), 2), round(float(y / scale), 2)]
        valid[current_idx] = 1
        current_idx += 1
        redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (10, 13):
            if current_idx == 8:
                cv2.destroyWindow(window_name)
                completed = [[0.0, 0.0] if point is None else point for point in corners]
                return "done", completed, valid
            print("Please mark or invalidate all 8 corners before pressing Enter.")
        elif key == ord("u"):
            if current_idx > 0:
                current_idx -= 1
                corners[current_idx] = None
                valid[current_idx] = 0
                redraw()
        elif key == ord("i"):
            if current_idx < 8:
                corners[current_idx] = [0.0, 0.0]
                valid[current_idx] = 0
                current_idx += 1
                redraw()
        elif key == ord("r"):
            corners = [None] * 8
            valid = [0] * 8
            current_idx = 0
            redraw()
        elif key == 27:
            cv2.destroyWindow(window_name)
            return "skip", None, None
        elif key == ord("q"):
            cv2.destroyWindow(window_name)
            return "quit", None, None


def get_video_info(cv2: Any, video_path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Could not read valid video metadata from: {video_path}")
    return fps, frame_count, width, height


def sampled_frame_indices(
    fps: float,
    frame_count: int,
    sample_every_sec: float,
    max_frames: Optional[int],
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
) -> List[int]:
    step = max(1, int(round(fps * sample_every_sec)))
    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = frame_count if end_sec is None else min(frame_count, max(start_frame + 1, int(np.ceil(end_sec * fps))))
    indices = list(range(start_frame, end_frame, step))
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def split_for_position(position: int, total: int, train_ratio: float, val_ratio: float) -> str:
    train_cut = int(round(total * train_ratio))
    val_cut = int(round(total * (train_ratio + val_ratio)))
    train_cut = min(max(1, train_cut), max(1, total - 1))
    val_cut = min(max(train_cut, val_cut), total)
    if position < train_cut:
        return "train"
    if position < val_cut:
        return "val"
    return "test"


def extract_frame(cv2: Any, video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def init_annotations(args: argparse.Namespace, frame_size: Sequence[int]) -> Dict[str, Any]:
    path = args.output_root / "annotations.json"
    if path.exists() and not args.overwrite:
        return load_json(path)
    return {
        "video_path": str(args.video),
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
        "corner_order": CORNER_ORDER,
        "frames": [],
    }


def upsert_frame_record(data: Dict[str, Any], record: Dict[str, Any]) -> None:
    frame_idx = int(record["frame_idx"])
    frames = [item for item in data.get("frames", []) if int(item["frame_idx"]) != frame_idx]
    frames.append(record)
    frames.sort(key=lambda item: int(item["frame_idx"]))
    data["frames"] = frames


def append_frame_record(data: Dict[str, Any], record: Dict[str, Any], overwrite: bool) -> None:
    if overwrite:
        upsert_frame_record(data, record)
        return
    frames = list(data.get("frames", []))
    frames.append(record)
    data["frames"] = frames


def next_record_id(data: Dict[str, Any], frame_idx: int) -> str:
    return f"frame_{frame_idx:06d}_ann_{len(data.get('frames', [])):06d}"


def make_square_crop_box(box: BBox, padding_ratio: float) -> BBox:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    side = max(w, h) * (1.0 + 2.0 * padding_ratio)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    half = side * 0.5
    return cx - half, cy - half, cx + half, cy + half


def crop_with_padding(image: Image.Image, crop_box: BBox, crop_size: int) -> Image.Image:
    src = np.asarray(image.convert("RGB"))
    x1, y1, x2, _ = crop_box
    side = x2 - x1
    if side <= 0:
        raise ValueError(f"Invalid crop box: {crop_box}")
    return Image.fromarray(src).transform(
        (crop_size, crop_size),
        Image.Transform.AFFINE,
        (side / crop_size, 0.0, x1, 0.0, side / crop_size, crop_box[1]),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )


def full_to_crop(points: np.ndarray, crop_box: BBox, crop_size: int) -> np.ndarray:
    x1, y1, x2, _ = crop_box
    side = x2 - x1
    out = np.empty_like(points, dtype=np.float32)
    out[:, 0] = (points[:, 0] - x1) * crop_size / side
    out[:, 1] = (points[:, 1] - y1) * crop_size / side
    return out


def make_heatmaps(
    corners_crop: np.ndarray,
    corner_valid: Sequence[int],
    crop_size: int,
    heatmap_size: int,
    sigma: float,
) -> tuple[np.ndarray, List[int]]:
    heatmaps = np.zeros((8, heatmap_size, heatmap_size), dtype=np.float32)
    effective_valid: List[int] = []
    yy, xx = np.mgrid[0:heatmap_size, 0:heatmap_size].astype(np.float32)
    scale = heatmap_size / float(crop_size)

    for idx, point in enumerate(corners_crop):
        x_hm = float(point[0] * scale)
        y_hm = float(point[1] * scale)
        is_valid = bool(corner_valid[idx]) and 0 <= x_hm < heatmap_size and 0 <= y_hm < heatmap_size
        effective_valid.append(1 if is_valid else 0)
        if not is_valid:
            continue
        heatmaps[idx] = np.exp(-((xx - x_hm) ** 2 + (yy - y_hm) ** 2) / (2.0 * sigma**2))
    return heatmaps, effective_valid


def draw_debug_crop(crop_img: Image.Image, corners_crop: np.ndarray, heatmaps: np.ndarray, valid: Sequence[int]) -> Image.Image:
    canvas = crop_img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for a, b in EDGES:
        if valid[a] and valid[b]:
            draw.line([tuple(corners_crop[a]), tuple(corners_crop[b])], fill=(255, 255, 255), width=2)
    for idx, (x, y) in enumerate(corners_crop):
        if not valid[idx]:
            continue
        color = COLORS[idx]
        r = 3
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        draw.text((x + 4, y + 4), str(idx), fill=color)

    merged = heatmaps.max(axis=0)
    if merged.max() > 0:
        merged = (merged / merged.max() * 255).astype(np.uint8)
    heat_img = Image.fromarray(merged).resize(canvas.size, Image.Resampling.BILINEAR).convert("RGB")
    return Image.blend(canvas, heat_img, 0.25)


def bbox_to_yolo_line(bbox: BBox, width: int, height: int) -> Optional[str]:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    xc = ((x1 + x2) * 0.5) / width
    yc = ((y1 + y2) * 0.5) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}"


def export_yolo(data: Dict[str, Any], output_root: Path) -> None:
    yolo_root = output_root / "yolo"
    width, height = data["frame_size"]
    for split in ["train", "val", "test"]:
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    data_yaml = 'names: ["dji_action4"]\nnc: 1\ntrain: images/train\nval: images/val\ntest: images/test\n'
    (yolo_root / "data.yaml").write_text(data_yaml, encoding="utf-8")

    for frame in data.get("frames", []):
        if frame.get("status") not in {"bbox_only", "full"}:
            continue
        split = frame["split"]
        src_img = output_root / frame["image_path"]
        if not src_img.exists():
            continue
        dst_img = yolo_root / "images" / split / src_img.name
        shutil.copy2(src_img, dst_img)
        label_path = yolo_root / "labels" / split / f"{src_img.stem}.txt"
        lines = []
        for inst in frame.get("instances", []):
            line = bbox_to_yolo_line(tuple(inst["bbox_xyxy_full"]), int(width), int(height))
            if line is not None:
                lines.append(line)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def export_mynet(data: Dict[str, Any], output_root: Path, args: argparse.Namespace) -> None:
    mynet_root = output_root / "mynet"
    samples_by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    image_w, image_h = data["frame_size"]

    for split in ["train", "val", "test"]:
        for folder in ["crops", "heatmaps", "debug_vis"]:
            (mynet_root / folder / split).mkdir(parents=True, exist_ok=True)

    for frame in data.get("frames", []):
        if frame.get("status") != "full":
            continue
        split = frame["split"]
        frame_path = output_root / frame["image_path"]
        if not frame_path.exists():
            continue
        image = Image.open(frame_path).convert("RGB")
        for inst_idx, inst in enumerate(frame.get("instances", [])):
            if "corners_2d_full" not in inst:
                continue
            bbox = tuple(float(v) for v in inst["bbox_xyxy_full"])
            crop_box = make_square_crop_box(bbox, args.bbox_padding)
            crop_img = crop_with_padding(image, crop_box, args.crop_size)
            corners_full = np.array(inst["corners_2d_full"], dtype=np.float32).reshape(8, 2)
            corner_valid = [int(v) for v in inst.get("corner_valid", [1] * 8)]
            corners_crop = full_to_crop(corners_full, crop_box, args.crop_size)
            heatmaps, effective_valid = make_heatmaps(corners_crop, corner_valid, args.crop_size, args.heatmap_size, args.sigma)

            record_id = str(frame.get("record_id") or Path(frame["image_path"]).stem)
            sample_id = f"{record_id}_{inst_idx:06d}"
            crop_rel = Path("crops") / split / f"{sample_id}.png"
            heatmap_rel = Path("heatmaps") / split / f"{sample_id}.npy"
            debug_rel = Path("debug_vis") / split / f"{sample_id}.jpg"

            crop_path = mynet_root / crop_rel
            heatmap_path = mynet_root / heatmap_rel
            debug_path = mynet_root / debug_rel
            crop_img.save(crop_path)
            np.save(heatmap_path, heatmaps)
            draw_debug_crop(crop_img, corners_crop, heatmaps, effective_valid).save(debug_path, quality=92)

            sample = {
                "sample_id": sample_id,
                "scene_id": "real_video",
                "image_id": f"{int(frame['frame_idx']):06d}",
                "instance_idx": inst_idx,
                "rgb_path": relpath(frame_path, mynet_root),
                "mask_path": None,
                "crop_path": crop_rel.as_posix(),
                "heatmap_path": heatmap_rel.as_posix(),
                "obj_id": 1,
                "K": np.eye(3, dtype=np.float32).tolist(),
                "R": np.eye(3, dtype=np.float32).tolist(),
                "t": [0.0, 0.0, 0.0],
                "bbox_source": "manual_video",
                "bbox_xyxy_full": [float(v) for v in bbox],
                "crop_box_xyxy_full": [float(v) for v in crop_box],
                "corners_3d": [[0.0, 0.0, 0.0] for _ in range(8)],
                "corners_2d_full": corners_full.astype(float).tolist(),
                "corners_2d_crop": corners_crop.astype(float).tolist(),
                "corner_valid": effective_valid,
                "visib_fract": 1.0,
                "px_count_visib": 0,
                "image_size_full": [int(image_w), int(image_h)],
                "crop_size": args.crop_size,
                "heatmap_size": args.heatmap_size,
            }
            samples_by_split[split].append(sample)

    meta = {
        "object_name": "DJI Action 4",
        "obj_id": 1,
        "source_video": data.get("video_path"),
        "corner_order": CORNER_ORDER,
        "crop_size": args.crop_size,
        "heatmap_size": args.heatmap_size,
        "sigma": args.sigma,
        "crop_padding_ratio": args.bbox_padding,
        "bbox_source": "manual_video",
        "K_placeholder": "identity",
        "R_placeholder": "identity",
        "t_placeholder": [0.0, 0.0, 0.0],
    }
    save_json(mynet_root / "meta.json", meta)
    for split, samples in samples_by_split.items():
        save_json(mynet_root / f"{split}.json", samples)


def export_all(data: Dict[str, Any], output_root: Path, args: argparse.Namespace) -> None:
    export_yolo(data, output_root)
    export_mynet(data, output_root, args)


def draw_full_preview(cv2: Any, image: np.ndarray, bboxes: Sequence[BBox], instances: Sequence[Dict[str, Any]]) -> np.ndarray:
    canvas = image.copy()
    for idx, bbox in enumerate(bboxes):
        p1 = int(round(bbox[0])), int(round(bbox[1]))
        p2 = int(round(bbox[2])), int(round(bbox[3]))
        cv2.rectangle(canvas, p1, p2, (255, 255, 255), 2)
        cv2.putText(canvas, f"obj {idx}", (p1[0] + 4, p1[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for inst in instances:
        corners = inst.get("corners_2d_full")
        valid = inst.get("corner_valid", [0] * 8)
        if not corners:
            continue
        for a, b in EDGES:
            if valid[a] and valid[b]:
                pa = tuple(int(round(v)) for v in corners[a])
                pb = tuple(int(round(v)) for v in corners[b])
                cv2.line(canvas, pa, pb, (255, 255, 255), 2)
        for corner_idx, point in enumerate(corners):
            if not valid[corner_idx]:
                continue
            color = COLORS[corner_idx]
            bgr = color[2], color[1], color[0]
            x, y = int(round(point[0])), int(round(point[1]))
            cv2.circle(canvas, (x, y), 5, bgr, -1)
            cv2.putText(canvas, str(corner_idx), (x + 6, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)
    return canvas


def make_frame_record(
    output_root: Path,
    frame_idx: int,
    timestamp_sec: float,
    frame_path: Path,
    split: str,
    status: str,
    instances: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "record_id": frame_path.stem,
        "frame_idx": int(frame_idx),
        "timestamp_sec": round(float(timestamp_sec), 4),
        "image_path": relpath(frame_path, output_root),
        "split": split,
        "status": status,
        "instances": list(instances),
    }


def annotate_video(args: argparse.Namespace) -> None:
    ensure_args(args)
    cv2 = load_cv2()
    fps, frame_count, width, height = get_video_info(cv2, args.video)
    frame_indices = sampled_frame_indices(fps, frame_count, args.sample_every_sec, args.max_frames, args.start_sec, args.end_sec)
    data = init_annotations(args, frame_size=[width, height])

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "frames" / "all").mkdir(parents=True, exist_ok=True)
    (args.output_root / "previews").mkdir(parents=True, exist_ok=True)

    print(f"Video: {args.video}")
    print(f"FPS={fps:.3f}, frames={frame_count}, sampled candidates={len(frame_indices)}, size={width}x{height}")
    print(f"Annotation time range: {args.start_sec:.3f}s to {args.end_sec if args.end_sec is not None else 'video_end'}s")
    print("Without --overwrite, new records are appended with unique names and existing annotations are preserved.")

    for position, frame_idx in enumerate(frame_indices):
        split = split_for_position(position, len(frame_indices), args.train_ratio, args.val_ratio)
        frame = extract_frame(cv2, args.video, frame_idx)
        record_id = next_record_id(data, frame_idx)
        frame_path = args.output_root / "frames" / "all" / f"{record_id}.png"
        cv2.imwrite(str(frame_path), frame)
        timestamp_sec = frame_idx / fps

        while True:
            bboxes = select_bboxes_for_frame(cv2, frame, args, frame_idx)
            if bboxes is None:
                save_json(args.output_root / "annotations.json", data)
                export_all(data, args.output_root, args)
                print("Saved current progress. Quit requested.")
                return
            if not bboxes:
                record = make_frame_record(args.output_root, frame_idx, timestamp_sec, frame_path, split, "skipped", [])
                append_frame_record(data, record, args.overwrite)
                save_json(args.output_root / "annotations.json", data)
                print(f"Skipped frame {frame_idx}.")
                break

            display, display_scale = scale_image_for_display(cv2, frame, args.max_display_side)
            boxed_preview = draw_boxes(cv2, display, bboxes, scale=display_scale)
            action = wait_action(
                cv2,
                boxed_preview,
                [
                    f"frame {frame_idx} split={split}; selected {len(bboxes)} bbox(es)",
                    "c: annotate 8 corners | b: save bbox-only for YOLO | r: retry bboxes | s/Esc: skip | q: save and quit",
                ],
                ["c", "b", "r", "s", "q"],
                "after bbox action",
            )
            cv2.destroyWindow("after bbox action")
            if action == "r":
                continue
            break

        if not bboxes:
            continue
        if action == "q":
            save_json(args.output_root / "annotations.json", data)
            export_all(data, args.output_root, args)
            print("Saved current progress. Quit requested.")
            return
        if action == "s":
            record = make_frame_record(args.output_root, frame_idx, timestamp_sec, frame_path, split, "skipped", [])
            append_frame_record(data, record, args.overwrite)
            save_json(args.output_root / "annotations.json", data)
            print(f"Skipped frame {frame_idx}.")
            continue

        instances: List[Dict[str, Any]] = [{"bbox_xyxy_full": [float(v) for v in bbox]} for bbox in bboxes]
        status = "bbox_only"
        if action == "c":
            status = "full"
            for inst_idx, bbox in enumerate(bboxes):
                result, corners, valid = annotate_instance_corners(cv2, frame, bbox, inst_idx, args.max_display_side)
                if result == "quit":
                    save_json(args.output_root / "annotations.json", data)
                    export_all(data, args.output_root, args)
                    print("Saved current progress. Quit requested.")
                    return
                if result == "skip":
                    record = make_frame_record(args.output_root, frame_idx, timestamp_sec, frame_path, split, "skipped", [])
                    append_frame_record(data, record, args.overwrite)
                    save_json(args.output_root / "annotations.json", data)
                    print(f"Skipped frame {frame_idx}.")
                    break
                assert corners is not None and valid is not None
                instances[inst_idx]["corners_2d_full"] = corners
                instances[inst_idx]["corner_valid"] = valid
            else:
                preview = draw_full_preview(cv2, frame, bboxes, instances)
                preview_path = args.output_root / "previews" / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(preview_path), preview)
                display_preview, _ = scale_image_for_display(cv2, preview, args.max_display_side)
                final_action = wait_action(
                    cv2,
                    display_preview,
                    [
                        f"frame {frame_idx}: preview after corner annotation",
                        "y: save bbox+corners | b: save bbox-only | s/Esc: skip | q: save progress and quit",
                    ],
                    ["y", "b", "s", "q"],
                    "final frame action",
                )
                cv2.destroyWindow("final frame action")
                if final_action == "q":
                    save_json(args.output_root / "annotations.json", data)
                    export_all(data, args.output_root, args)
                    print("Saved current progress. Quit requested.")
                    return
                if final_action == "s":
                    record = make_frame_record(args.output_root, frame_idx, timestamp_sec, frame_path, split, "skipped", [])
                    append_frame_record(data, record, args.overwrite)
                    save_json(args.output_root / "annotations.json", data)
                    print(f"Skipped frame {frame_idx}.")
                    continue
                if final_action == "b":
                    status = "bbox_only"
                    instances = [{"bbox_xyxy_full": [float(v) for v in bbox]} for bbox in bboxes]
                else:
                    status = "full"
            if data.get("frames") and int(data["frames"][-1]["frame_idx"]) == frame_idx and data["frames"][-1].get("status") == "skipped":
                continue

        record = make_frame_record(args.output_root, frame_idx, timestamp_sec, frame_path, split, status, instances)
        append_frame_record(data, record, args.overwrite)
        save_json(args.output_root / "annotations.json", data)
        export_all(data, args.output_root, args)
        print(f"Saved frame {frame_idx} as {status}; split={split}; instances={len(instances)}")

    save_json(args.output_root / "annotations.json", data)
    export_all(data, args.output_root, args)
    cv2.destroyAllWindows()
    print(f"Done. Saved annotations to {args.output_root / 'annotations.json'}")
    print(f"YOLO data: {args.output_root / 'yolo'}")
    print(f"MyNet data: {args.output_root / 'mynet'}")


def main() -> None:
    annotate_video(parse_args())


if __name__ == "__main__":
    main()
