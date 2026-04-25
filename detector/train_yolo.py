from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an Ultralytics YOLO detector for DJI Action4 bbox detection.")
    parser.add_argument("--data", type=Path, default=Path("data/dji_action4_video_annot/yolo/data.yaml"))
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model or checkpoint path, e.g. yolo11n.pt or detector/runs/.../last.pt.")
    parser.add_argument("--project", type=Path, default=Path("detector/runs"))
    parser.add_argument("--name", type=str, default="dji_action4_yolo")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=str, default="16", help="Integer batch size, or 'auto' for Ultralytics auto batch.")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id such as '0', or 'cpu'.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cache", choices=["none", "ram", "disk"], default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_batch(value: str) -> int:
    if value.lower() == "auto":
        return -1
    return int(value)


def parse_cache(value: str) -> bool | str:
    if value == "none":
        return False
    return value


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"YOLO data config not found: {args.data}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed. Run: uv pip install ultralytics") from exc

    model = YOLO(args.model)
    train_kwargs: dict[str, Any] = {
        "data": str(args.data),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": parse_batch(args.batch),
        "device": args.device,
        "workers": args.workers,
        "project": str(args.project),
        "name": args.name,
        "patience": args.patience,
        "cache": parse_cache(args.cache),
        "seed": args.seed,
        "resume": args.resume,
        "exist_ok": args.exist_ok,
        "amp": args.amp,
        "plots": args.plots,
    }
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
