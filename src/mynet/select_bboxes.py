from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually select one or two DJI Action4 bboxes from an image.")
    parser.add_argument("--image", type=Path, required=True, help="Input image, for example image.png.")
    parser.add_argument("--output", type=Path, default=Path("bboxes.json"), help="Output bbox JSON path.")
    parser.add_argument("--preview", type=Path, default=None, help="Optional preview image with selected boxes drawn.")
    parser.add_argument("--min-bboxes", type=int, default=1)
    parser.add_argument("--max-bboxes", type=int, default=2)
    parser.add_argument("--max-display-side", type=int, default=1400, help="Resize large images for easier on-screen selection.")
    return parser.parse_args()


def load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for manual bbox selection. Install it with: uv pip install opencv-python") from exc
    return cv2


def scale_image_for_display(cv2: Any, image: Any, max_side: int) -> tuple[Any, float]:
    height, width = image.shape[:2]
    side = max(height, width)
    if side <= max_side:
        return image, 1.0
    scale = max_side / float(side)
    display = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    return display, scale


def xywh_to_xyxy(box: List[float], scale: float) -> List[float]:
    x, y, w, h = box
    x1 = x / scale
    y1 = y / scale
    x2 = (x + w) / scale
    y2 = (y + h) / scale
    return [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)]


def draw_preview(cv2: Any, image: Any, bboxes: List[List[float]], preview_path: Path) -> None:
    canvas = image.copy()
    for idx, (x1, y1, x2, y2) in enumerate(bboxes):
        p1 = int(round(x1)), int(round(y1))
        p2 = int(round(x2)), int(round(y2))
        cv2.rectangle(canvas, p1, p2, (255, 255, 255), 2)
        cv2.putText(canvas, f"roi {idx}", (p1[0] + 4, p1[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), canvas)


def main() -> None:
    args = parse_args()
    cv2 = load_cv2()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    display, scale = scale_image_for_display(cv2, image, args.max_display_side)
    window_name = "select 1 or 2 bboxes: draw box, press Enter/Space, press Esc when done"
    print("Draw one bbox per DJI Action4.")
    print("After each box, press Enter or Space. Press Esc when all boxes are selected.")
    rois = cv2.selectROIs(window_name, display, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    bboxes = [xywh_to_xyxy([float(v) for v in roi], scale) for roi in rois.tolist() if roi[2] > 0 and roi[3] > 0]
    if not (args.min_bboxes <= len(bboxes) <= args.max_bboxes):
        raise ValueError(f"Expected {args.min_bboxes} to {args.max_bboxes} bbox(es), got {len(bboxes)}.")

    output: Dict[str, object] = {
        "image": str(args.image),
        "format": "xyxy_full_image_pixels",
        "bboxes": bboxes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    preview_path = args.preview or args.output.with_name(args.output.stem + "_preview.png")
    draw_preview(cv2, image, bboxes, preview_path)

    print(f"saved bboxes: {args.output}")
    print(f"saved preview: {preview_path}")
    for idx, bbox in enumerate(bboxes):
        print(f"bbox {idx}: {bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")


if __name__ == "__main__":
    main()
