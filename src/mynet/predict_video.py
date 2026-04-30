from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.mynet.decode import decode_heatmap_argmax
from src.mynet.infer_image import crop_to_full, crop_with_padding, load_model, make_square_crop_box, preprocess


BBox = Tuple[float, float, float, float]

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
CORNER_COLORS_RGB = [
    (255, 0, 0),
    (255, 128, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 255, 255),
    (0, 128, 255),
    (0, 0, 255),
    (255, 0, 255),
]
MODEL_COLOR_BGR = (0, 255, 0)
ANNOTATION_COLOR_BGR = (0, 180, 255)
NO_DETECTION_COLOR_BGR = (0, 0, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO detector + MyNet corner prediction on a video.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--mynet-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/video_mynet_pipeline"))
    parser.add_argument("--name", type=str, default="action4_result")
    parser.add_argument("--detector-device", type=str, default="0")
    parser.add_argument("--mynet-device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=2)
    parser.add_argument("--bbox-padding", type=float, default=0.25)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--annotations-json", type=Path, default=None)
    parser.add_argument("--fallback-to-annotations", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--line-width", type=int, default=2)
    return parser.parse_args()


def load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Run: uv pip install opencv-python") from exc
    return cv2


def load_detector(weights: Path) -> Any:
    if not weights.exists():
        raise FileNotFoundError(f"Detector weights not found: {weights}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed. Run: uv pip install ultralytics") from exc
    return YOLO(str(weights))


def load_annotation_lookup(path: Optional[Path]) -> Dict[int, List[BBox]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Annotation JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    lookup: Dict[int, List[BBox]] = {}
    for frame in data.get("frames", []):
        if frame.get("status") not in {"bbox_only", "full"}:
            continue
        bboxes: List[BBox] = []
        for inst in frame.get("instances", []):
            bbox = inst.get("bbox_xyxy_full")
            if bbox is not None and len(bbox) == 4:
                bboxes.append(tuple(float(v) for v in bbox))
        if bboxes:
            lookup[int(frame["frame_idx"])] = bboxes
    return lookup


def result_to_detections(result: Any, conf_threshold: float, max_detections: int) -> List[Dict[str, Any]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    clss = result.boxes.cls.detach().cpu().numpy() if result.boxes.cls is not None else np.zeros(len(boxes), dtype=np.float32)
    detections: List[Dict[str, Any]] = []
    for bbox, conf, cls_id in zip(boxes, confs, clss):
        if float(conf) < conf_threshold:
            continue
        detections.append(
            {
                "bbox_xyxy_full": [float(v) for v in bbox],
                "conf": float(conf),
                "cls": int(cls_id),
                "source": "model",
            }
        )
    detections.sort(key=lambda item: item["conf"], reverse=True)
    detections = detections[:max_detections]
    detections.sort(key=lambda item: item["bbox_xyxy_full"][0])
    for idx, det in enumerate(detections):
        det["instance_id"] = idx
    return detections


def annotation_detections(annotation_lookup: Dict[int, List[BBox]], frame_idx: int, max_detections: int) -> List[Dict[str, Any]]:
    bboxes = annotation_lookup.get(frame_idx, [])[:max_detections]
    detections = [
        {
            "instance_id": idx,
            "bbox_xyxy_full": [float(v) for v in bbox],
            "conf": 1.0,
            "cls": 0,
            "source": "annotation",
        }
        for idx, bbox in enumerate(sorted(bboxes, key=lambda item: item[0]))
    ]
    return detections


def detect_frame(detector: Any, frame_bgr: np.ndarray, args: argparse.Namespace, annotation_lookup: Dict[int, List[BBox]], frame_idx: int) -> List[Dict[str, Any]]:
    results = detector.predict(
        source=frame_bgr,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.detector_device,
        max_det=max(1, args.max_detections),
        verbose=False,
    )
    detections = result_to_detections(results[0], args.conf, args.max_detections)
    if detections or not args.fallback_to_annotations:
        return detections
    return annotation_detections(annotation_lookup, frame_idx, args.max_detections)


def frame_to_pil_rgb(cv2: Any, frame_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def run_mynet_on_detections(
    cv2: Any,
    mynet: torch.nn.Module,
    frame_bgr: np.ndarray,
    detections: Sequence[Dict[str, Any]],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[List[Dict[str, Any]], List[Image.Image], np.ndarray]:
    if not detections:
        return [], [], np.zeros((0, 8, 1, 1), dtype=np.float32)

    frame_rgb = frame_to_pil_rgb(cv2, frame_bgr)
    crops: List[Image.Image] = []
    crop_boxes: List[BBox] = []
    tensors = []
    for det in detections:
        bbox = tuple(float(v) for v in det["bbox_xyxy_full"])
        crop_box = make_square_crop_box(bbox, args.bbox_padding)
        crop = crop_with_padding(frame_rgb, crop_box, args.crop_size)
        crops.append(crop)
        crop_boxes.append(crop_box)
        tensors.append(preprocess(crop))

    batch = torch.cat(tensors, dim=0).to(device)
    use_amp = args.amp and device.type == "cuda"
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        logits = mynet(batch)

    corners_crop_batch = decode_heatmap_argmax(logits, crop_size=args.crop_size).detach().cpu().numpy()
    heatmaps_batch = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    scores_batch = heatmaps_batch.reshape(heatmaps_batch.shape[0], heatmaps_batch.shape[1], -1).max(axis=2)

    predictions: List[Dict[str, Any]] = []
    for idx, (corners_crop, scores, crop_box) in enumerate(zip(corners_crop_batch, scores_batch, crop_boxes)):
        corners_full = crop_to_full(corners_crop.astype(np.float32), crop_box, args.crop_size)
        predictions.append(
            {
                "instance_id": int(detections[idx]["instance_id"]),
                "crop_box_xyxy_full": [float(v) for v in crop_box],
                "corners_2d_crop": corners_crop.astype(float).tolist(),
                "corners_2d_full": corners_full.astype(float).tolist(),
                "corner_scores": [float(v) for v in scores],
            }
        )
    return predictions, crops, heatmaps_batch


def detection_color(det: Dict[str, Any]) -> Tuple[int, int, int]:
    return ANNOTATION_COLOR_BGR if det.get("source") == "annotation" else MODEL_COLOR_BGR


def draw_bbox_frame(cv2: Any, frame_bgr: np.ndarray, detections: Sequence[Dict[str, Any]], line_width: int) -> np.ndarray:
    canvas = frame_bgr.copy()
    if not detections:
        cv2.putText(canvas, "no detections", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, NO_DETECTION_COLOR_BGR, 2)
        return canvas
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy_full"]]
        color = detection_color(det)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_width)
        label = f"{det['instance_id']} {det['source']} {det['conf']:.2f}"
        cv2.putText(canvas, label, (x1 + 4, max(20, y1 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return canvas


def draw_roi_corners(crop: Image.Image, corners_crop: Sequence[Sequence[float]], scores: Sequence[float]) -> Image.Image:
    canvas = crop.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for a, b in EDGES:
        draw.line([tuple(corners_crop[a]), tuple(corners_crop[b])], fill=(255, 255, 255), width=2)
    for idx, (x, y) in enumerate(corners_crop):
        color = CORNER_COLORS_RGB[idx]
        r = 4
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        draw.text((x + 5, y + 5), f"{idx}:{scores[idx]:.2f}", fill=color)
    return canvas


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    hm = np.asarray(heatmap, dtype=np.float32)
    hm = hm - float(hm.min())
    max_value = float(hm.max())
    if max_value > 1e-6:
        hm = hm / max_value
    return np.clip(hm * 255.0, 0, 255).astype(np.uint8)


def heatmap_to_bgr(cv2: Any, heatmap: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(normalize_heatmap(heatmap), size, interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(resized, cv2.COLORMAP_JET)


def draw_heatmap_grid(cv2: Any, heatmaps: np.ndarray, scores: Sequence[float], tile_size: int = 160) -> np.ndarray:
    tiles = []
    for idx, heatmap in enumerate(heatmaps):
        tile = heatmap_to_bgr(cv2, heatmap, (tile_size, tile_size))
        cv2.putText(tile, f"corner {idx} max={scores[idx]:.2f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
        cv2.putText(tile, f"corner {idx} max={scores[idx]:.2f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)
        tiles.append(tile)
    row1 = np.concatenate(tiles[:4], axis=1)
    row2 = np.concatenate(tiles[4:], axis=1)
    return np.concatenate([row1, row2], axis=0)


def draw_heatmap_overlay(cv2: Any, crop: Image.Image, heatmaps: np.ndarray) -> np.ndarray:
    crop_bgr = cv2.cvtColor(np.asarray(crop.convert("RGB")), cv2.COLOR_RGB2BGR)
    merged = heatmaps.max(axis=0)
    heat_bgr = heatmap_to_bgr(cv2, merged, (crop_bgr.shape[1], crop_bgr.shape[0]))
    return cv2.addWeighted(crop_bgr, 0.55, heat_bgr, 0.45, 0)


def draw_full_frame(
    cv2: Any,
    frame_bgr: np.ndarray,
    detections: Sequence[Dict[str, Any]],
    corner_predictions: Sequence[Dict[str, Any]],
    line_width: int,
) -> np.ndarray:
    canvas = draw_bbox_frame(cv2, frame_bgr, detections, line_width)
    corners_by_id = {int(item["instance_id"]): item for item in corner_predictions}
    for det in detections:
        pred = corners_by_id.get(int(det["instance_id"]))
        if pred is None:
            continue
        corners = pred["corners_2d_full"]
        for a, b in EDGES:
            pa = tuple(int(round(v)) for v in corners[a])
            pb = tuple(int(round(v)) for v in corners[b])
            cv2.line(canvas, pa, pb, (255, 255, 255), line_width)
        for idx, point in enumerate(corners):
            rgb = CORNER_COLORS_RGB[idx]
            bgr = rgb[2], rgb[1], rgb[0]
            x, y = int(round(point[0])), int(round(point[1]))
            cv2.circle(canvas, (x, y), 5, bgr, -1)
            cv2.putText(canvas, str(idx), (x + 6, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)
    return canvas


def save_debug_images(
    cv2: Any,
    out_dir: Path,
    frame_idx: int,
    bbox_frame: np.ndarray,
    crops: Sequence[Image.Image],
    corner_predictions: Sequence[Dict[str, Any]],
    heatmaps: np.ndarray,
    final_frame: np.ndarray,
) -> None:
    debug_root = out_dir / "debug"
    bbox_dir = debug_root / "detector_bbox_frames"
    roi_input_dir = debug_root / "roi_inputs"
    roi_corner_dir = debug_root / "roi_corners"
    roi_heatmap_dir = debug_root / "roi_heatmaps"
    roi_heatmap_overlay_dir = debug_root / "roi_heatmap_overlays"
    roi_heatmap_raw_dir = debug_root / "roi_heatmaps_raw"
    final_dir = debug_root / "final_frames"
    for folder in [bbox_dir, roi_input_dir, roi_corner_dir, roi_heatmap_dir, roi_heatmap_overlay_dir, roi_heatmap_raw_dir, final_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(bbox_dir / f"frame_{frame_idx:06d}.jpg"), bbox_frame)
    cv2.imwrite(str(final_dir / f"frame_{frame_idx:06d}.jpg"), final_frame)
    for idx, crop in enumerate(crops):
        crop.save(roi_input_dir / f"frame_{frame_idx:06d}_obj_{idx:02d}.jpg", quality=92)
        if idx < len(corner_predictions):
            roi = draw_roi_corners(crop, corner_predictions[idx]["corners_2d_crop"], corner_predictions[idx]["corner_scores"])
            roi.save(roi_corner_dir / f"frame_{frame_idx:06d}_obj_{idx:02d}.jpg", quality=92)
        if idx < len(heatmaps):
            scores = corner_predictions[idx]["corner_scores"] if idx < len(corner_predictions) else [0.0] * 8
            heatmap_grid = draw_heatmap_grid(cv2, heatmaps[idx], scores)
            heatmap_overlay = draw_heatmap_overlay(cv2, crop, heatmaps[idx])
            cv2.imwrite(str(roi_heatmap_dir / f"frame_{frame_idx:06d}_obj_{idx:02d}.jpg"), heatmap_grid)
            cv2.imwrite(str(roi_heatmap_overlay_dir / f"frame_{frame_idx:06d}_obj_{idx:02d}.jpg"), heatmap_overlay)
            np.save(roi_heatmap_raw_dir / f"frame_{frame_idx:06d}_obj_{idx:02d}.npy", heatmaps[idx])


def jsonl_record(frame_idx: int, fps: float, detections: Sequence[Dict[str, Any]], corners: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "frame_idx": frame_idx,
        "timestamp_sec": round(frame_idx / fps, 6),
        "detections": [
            {
                "instance_id": int(det["instance_id"]),
                "bbox_xyxy_full": [float(v) for v in det["bbox_xyxy_full"]],
                "conf": float(det["conf"]),
                "source": str(det["source"]),
            }
            for det in detections
        ],
        "corners": list(corners),
    }


def process_video(args: argparse.Namespace) -> None:
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.mynet_checkpoint.exists():
        raise FileNotFoundError(f"MyNet checkpoint not found: {args.mynet_checkpoint}")
    if args.max_detections <= 0:
        raise ValueError("--max-detections must be positive.")
    if args.debug_every <= 0:
        raise ValueError("--debug-every must be positive.")

    cv2 = load_cv2()
    detector = load_detector(args.detector_weights)
    mynet_device = torch.device(args.mynet_device)
    mynet = load_model(args.mynet_checkpoint, mynet_device)
    annotation_lookup = load_annotation_lookup(args.annotations_json) if args.fallback_to_annotations else {}

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Could not read video size from: {args.video}")

    out_dir = args.output_dir / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    video_out = out_dir / f"{args.name}.mp4"
    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer: {video_out}")

    jsonl_path = out_dir / "predictions.jsonl"
    counts = {"frames": 0, "frames_with_detections": 0, "model_detections": 0, "annotation_detections": 0, "corner_predictions": 0}
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        frame_idx = 0
        while True:
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break

            detections = detect_frame(detector, frame, args, annotation_lookup, frame_idx)
            corners, crops, heatmaps = run_mynet_on_detections(cv2, mynet, frame, detections, mynet_device, args)
            bbox_frame = draw_bbox_frame(cv2, frame, detections, args.line_width)
            final_frame = draw_full_frame(cv2, frame, detections, corners, args.line_width)
            writer.write(final_frame)

            if args.debug and frame_idx % args.debug_every == 0:
                save_debug_images(cv2, out_dir, frame_idx, bbox_frame, crops, corners, heatmaps, final_frame)

            record = jsonl_record(frame_idx, fps, detections, corners)
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

            counts["frames"] += 1
            counts["frames_with_detections"] += 1 if detections else 0
            counts["model_detections"] += sum(1 for det in detections if det["source"] == "model")
            counts["annotation_detections"] += sum(1 for det in detections if det["source"] == "annotation")
            counts["corner_predictions"] += len(corners)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"processed {frame_idx}/{total_frames or '?'} frames")

    cap.release()
    writer.release()

    summary = {
        "video": str(args.video),
        "output_video": str(video_out),
        "predictions_jsonl": str(jsonl_path),
        "detector_weights": str(args.detector_weights),
        "mynet_checkpoint": str(args.mynet_checkpoint),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames_reported": total_frames,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "counts": counts,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"saved video: {video_out}")
    print(f"saved predictions: {jsonl_path}")
    print(f"saved summary: {out_dir / 'summary.json'}")


def main() -> None:
    process_video(parse_args())


if __name__ == "__main__":
    main()
