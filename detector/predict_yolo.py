from __future__ import annotations

import argparse
from pathlib import Path


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
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Detector weights not found: {args.weights}")
    if not args.source.exists():
        raise FileNotFoundError(f"Prediction source not found: {args.source}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed. Run: uv pip install ultralytics") from exc

    model = YOLO(str(args.weights))
    model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=str(args.project),
        name=args.name,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
    )


if __name__ == "__main__":
    main()
