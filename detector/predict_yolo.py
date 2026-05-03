from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fine-tuned DJI Action4 YOLO detector on images or videos.")
    parser.add_argument("--weights", type=Path, default=Path("detector/runs/dji_action4_yolo/weights/best.pt"))
    parser.add_argument("--source", type=Path, required=True, help="Image, video, or directory to predict.")
    parser.add_argument("--project", type=Path, default=Path("detector/predictions"))
    parser.add_argument("--name", type=str, default="dji_action4_predict")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--max-detections", type=int, default=2, help="Keep at most this many highest-confidence detections per frame.")
    parser.add_argument("--annotations-json", type=Path, default=None, help="Optional annotate_video annotations.json for fallback bboxes.")
    parser.add_argument("--fallback-to-annotations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    parser.add_argument("--save-crops", action="store_true", help="Save the original-sized bounding box crops.")
    return parser.parse_args()


def load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Run: uv pip install opencv-python") from exc
    return cv2


def load_annotation_lookup(path: Optional[Path]) -> Dict[str, Dict[Any, List[List[float]]]]:
    lookup: Dict[str, Dict[Any, List[List[float]]]] = {"frame_idx": {}, "image_name": {}, "image_stem": {}}
    if path is None:
        return lookup
    if not path.exists():
        raise FileNotFoundError(f"Annotation JSON not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for frame in data.get("frames", []):
        if frame.get("status") not in {"bbox_only", "full"}:
            continue
        bboxes = []
        for inst in frame.get("instances", []):
            bbox = inst.get("bbox_xyxy_full")
            if bbox is not None and len(bbox) == 4:
                bboxes.append([float(v) for v in bbox])
        if not bboxes:
            continue

        frame_idx = int(frame["frame_idx"])
        image_name = Path(frame.get("image_path", "")).name
        image_stem = Path(image_name).stem
        lookup["frame_idx"][frame_idx] = bboxes
        if image_name:
            lookup["image_name"][image_name] = bboxes
        if image_stem:
            lookup["image_stem"][image_stem] = bboxes
    return lookup


def annotation_detections(
    lookup: Dict[str, Dict[Any, List[List[float]]]],
    *,
    frame_idx: Optional[int],
    image_path: Optional[Path],
    max_detections: int,
) -> List[Dict[str, Any]]:
    bboxes: Optional[List[List[float]]] = None
    if frame_idx is not None:
        bboxes = lookup["frame_idx"].get(frame_idx)
    if bboxes is None and image_path is not None:
        bboxes = lookup["image_name"].get(image_path.name) or lookup["image_stem"].get(image_path.stem)
    if not bboxes:
        return []
    return [{"bbox": bbox, "conf": 1.0, "cls": 0, "source": "annotation"} for bbox in bboxes[:max_detections]]


def result_to_detections(result: Any, conf_threshold: float, max_detections: int) -> List[Dict[str, Any]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    clss = result.boxes.cls.detach().cpu().numpy() if result.boxes.cls is not None else [0] * len(boxes)
    detections = []
    for bbox, conf, cls_id in zip(boxes, confs, clss):
        if float(conf) < conf_threshold:
            continue
        detections.append({"bbox": [float(v) for v in bbox], "conf": float(conf), "cls": int(cls_id), "source": "model"})
    detections.sort(key=lambda item: item["conf"], reverse=True)
    return detections[:max_detections]


def detect_frame(model: Any, frame: Any, args: argparse.Namespace) -> List[Dict[str, Any]]:
    results = model.predict(
        source=frame,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_det=max(1, args.max_detections),
        verbose=False,
    )
    return result_to_detections(results[0], args.conf, args.max_detections)


def draw_detections(cv2: Any, frame: Any, detections: Sequence[Dict[str, Any]], line_width: int) -> Any:
    canvas = frame.copy()
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        color = (0, 255, 0) if det["source"] == "model" else (0, 180, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_width)
        label = f"{idx} {det['source']} {det['conf']:.2f}"
        cv2.putText(canvas, label, (x1 + 4, max(20, y1 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return canvas


def yolo_label_lines(detections: Sequence[Dict[str, Any]], width: int, height: int, save_conf: bool) -> List[str]:
    lines = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        xc = ((x1 + x2) * 0.5) / width
        yc = ((y1 + y2) * 0.5) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        line = f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}"
        if save_conf:
            line += f" {det['conf']:.8f}"
        lines.append(line)
    return lines


def maybe_apply_fallback(
    detections: List[Dict[str, Any]],
    lookup: Dict[str, Dict[Any, List[List[float]]]],
    args: argparse.Namespace,
    *,
    frame_idx: Optional[int],
    image_path: Optional[Path],
) -> List[Dict[str, Any]]:
    if detections or not args.fallback_to_annotations or args.annotations_json is None:
        return detections
    return annotation_detections(lookup, frame_idx=frame_idx, image_path=image_path, max_detections=args.max_detections)


def save_label_file(path: Path, detections: Sequence[Dict[str, Any]], width: int, height: int, save_conf: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = yolo_label_lines(detections, width, height, save_conf)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def process_image(cv2: Any, model: Any, image_path: Path, out_dir: Path, lookup: Dict[str, Dict[Any, List[List[float]]]], args: argparse.Namespace) -> Dict[str, Any]:
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = frame.shape[:2]
    detections = detect_frame(model, frame, args)
    detections = maybe_apply_fallback(detections, lookup, args, frame_idx=None, image_path=image_path)
    overlay = draw_detections(cv2, frame, detections, args.line_width)
    output_path = out_dir / f"{image_path.stem}_pred{image_path.suffix}"
    cv2.imwrite(str(output_path), overlay)
    
    if args.save_crops:
        crops_dir = out_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        for det in detections:
            bbox = det.get("bbox") or det.get("bbox_xyxy_full")
            if bbox:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    cv2.imwrite(str(crops_dir / f"{image_path.stem}_inst_{det.get('instance_id', 0)}.jpg"), crop)
                    
    if args.save_txt:
        save_label_file(out_dir / "labels" / f"{image_path.stem}.txt", detections, width, height, args.save_conf)
    return {"image": str(image_path), "output": str(output_path), "detections": detections}


def process_video(cv2: Any, model: Any, video_path: Path, out_dir: Path, lookup: Dict[str, Dict[Any, List[List[float]]]], args: argparse.Namespace) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    output_path = out_dir / f"{video_path.stem}_pred.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer: {output_path}")

    frame_records = []
    frame_idx = 0
    labels_dir = out_dir / "labels"
    crops_dir = out_dir / "crops"
    if args.save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)
        
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detect_frame(model, frame, args)
        detections = maybe_apply_fallback(detections, lookup, args, frame_idx=frame_idx, image_path=None)
        writer.write(draw_detections(cv2, frame, detections, args.line_width))
        
        if args.save_crops:
            for det in detections:
                # Fallback to "bbox_xyxy_full" if "bbox" is missing (result_to_detections output)
                bbox = det.get("bbox") or det.get("bbox_xyxy_full")
                if bbox:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        cv2.imwrite(str(crops_dir / f"frame_{frame_idx:06d}_inst_{det.get('instance_id', 0)}.jpg"), crop)

        if args.save_txt:
            save_label_file(labels_dir / f"frame_{frame_idx:06d}.txt", detections, width, height, args.save_conf)
        frame_records.append({"frame_idx": frame_idx, "detections": detections})
        frame_idx += 1

    cap.release()
    writer.release()
    return {"video": str(video_path), "output": str(output_path), "frames": frame_records}


def iter_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted([item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES])


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Detector weights not found: {args.weights}")
    if not args.source.exists():
        raise FileNotFoundError(f"Prediction source not found: {args.source}")
    if args.max_detections <= 0:
        raise ValueError("--max-detections must be positive.")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed. Run: uv pip install ultralytics") from exc

    cv2 = load_cv2()
    lookup = load_annotation_lookup(args.annotations_json)
    model = YOLO(str(args.weights))
    out_dir = args.project / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    source_suffix = args.source.suffix.lower()
    if args.source.is_file() and source_suffix in VIDEO_SUFFIXES:
        prediction = process_video(cv2, model, args.source, out_dir, lookup, args)
    else:
        images = iter_images(args.source)
        if not images:
            raise ValueError(f"No supported images found in: {args.source}")
        prediction = {"images": [process_image(cv2, model, image_path, out_dir, lookup, args) for image_path in images]}

    with (out_dir / "predictions.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "weights": str(args.weights),
                "source": str(args.source),
                "conf": args.conf,
                "iou": args.iou,
                "max_detections": args.max_detections,
                "annotations_json": str(args.annotations_json) if args.annotations_json else None,
                "prediction": prediction,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved predictions to: {out_dir}")


if __name__ == "__main__":
    main()
