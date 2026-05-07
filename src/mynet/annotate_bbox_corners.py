from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from src.mynet.annotate_video import (
    BBox,
    annotate_instance_corners,
    crop_with_padding,
    draw_full_preview,
    export_all,
    load_cv2,
    load_json,
    make_square_crop_box,
    save_json,
    scale_image_for_display,
    wait_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate 8 corners from existing bbox-only video annotations.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root containing annotations.json and frames/all.")
    parser.add_argument("--annotations-json", type=Path, default=None)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--bbox-padding", type=float, default=0.25)
    parser.add_argument("--max-display-side", type=int, default=1200)
    parser.add_argument("--corner-display-side", type=int, default=768, help="Upscale ROI corner annotation window to this side length before display. Set 0 to disable upscaling.")
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--record-id", action="append", default=[], help="Only annotate matching record_id. Can be repeated.")
    parser.add_argument("--append-full-records", action="store_true", help="Append full records instead of upgrading bbox-only records in place.")
    parser.add_argument("--export-mode", choices=["final", "batch", "each"], default="final", help="When to rebuild YOLO/MyNet exports during annotation.")
    parser.add_argument("--export-every", type=int, default=20, help="Export every N completed records when --export-mode=batch.")
    parser.add_argument("--export-only", action="store_true", help="Only rebuild YOLO/MyNet exports from annotations.json and exit.")
    return parser.parse_args()


def crop_to_full(points: np.ndarray, crop_box: BBox, crop_size: int) -> np.ndarray:
    x1, y1, x2, _ = crop_box
    side = x2 - x1
    out = np.empty_like(points, dtype=np.float32)
    out[:, 0] = x1 + points[:, 0] * side / crop_size
    out[:, 1] = y1 + points[:, 1] * side / crop_size
    return out


def needs_corner_annotation(frame: Dict[str, Any]) -> bool:
    if frame.get("status") == "bbox_only":
        return True
    if frame.get("status") != "full":
        return False
    return any("corners_2d_full" not in inst for inst in frame.get("instances", []))


def should_visit(frame: Dict[str, Any], args: argparse.Namespace) -> bool:
    if args.record_id and str(frame.get("record_id", "")) not in set(args.record_id):
        return False
    timestamp = float(frame.get("timestamp_sec", 0.0))
    if args.start_sec is not None and timestamp < args.start_sec:
        return False
    if args.end_sec is not None and timestamp >= args.end_sec:
        return False
    return needs_corner_annotation(frame)


def next_appended_record_id(data: Dict[str, Any], base_id: str) -> str:
    used = {str(frame.get("record_id", "")) for frame in data.get("frames", [])}
    idx = 0
    while True:
        candidate = f"{base_id}_corners_{idx:03d}"
        if candidate not in used:
            return candidate
        idx += 1


def replace_frame_record(data: Dict[str, Any], index: int, record: Dict[str, Any], append: bool) -> None:
    if append:
        data["frames"].append(record)
    else:
        data["frames"][index] = record


def load_frame_image(cv2: Any, output_root: Path, frame: Dict[str, Any]) -> np.ndarray:
    image_path = output_root / frame["image_path"]
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read frame image: {image_path}")
    return image


def pil_crop_to_bgr(cv2: Any, crop: Any) -> np.ndarray:
    return cv2.cvtColor(np.asarray(crop.convert("RGB")), cv2.COLOR_RGB2BGR)


def annotate_record(cv2: Any, data: Dict[str, Any], frame: Dict[str, Any], args: argparse.Namespace) -> tuple[str, Optional[Dict[str, Any]]]:
    image_bgr = load_frame_image(cv2, args.output_root, frame)
    image_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    new_frame = copy.deepcopy(frame)
    instances = new_frame.get("instances", [])
    if not instances:
        return "skip", None

    for inst_idx, inst in enumerate(instances):
        if "corners_2d_full" in inst:
            continue
        bbox = tuple(float(v) for v in inst["bbox_xyxy_full"])
        crop_box = make_square_crop_box(bbox, args.bbox_padding)
        crop_img = crop_with_padding(image_rgb, crop_box, args.crop_size)
        crop_bgr = pil_crop_to_bgr(cv2, crop_img)
        crop_bbox = (0.0, 0.0, float(args.crop_size), float(args.crop_size))

        while True:
            result, corners_crop_list, valid = annotate_instance_corners(
                cv2,
                crop_bgr,
                crop_bbox,
                inst_idx,
                args.max_display_side,
                min_display_side=args.corner_display_side,
            )
            if result in {"quit", "skip"}:
                return result, None
            if corners_crop_list is None or valid is None:
                return "skip", None
            corners_crop = np.array(corners_crop_list, dtype=np.float32).reshape(8, 2)
            corners_full = crop_to_full(corners_crop, crop_box, args.crop_size)
            inst["corners_2d_full"] = corners_full.astype(float).tolist()
            inst["corner_valid"] = [int(v) for v in valid]

            preview = draw_full_preview(cv2, image_bgr, [tuple(float(v) for v in item["bbox_xyxy_full"]) for item in instances], instances)
            preview, _ = scale_image_for_display(cv2, preview, args.max_display_side)
            action = wait_action(
                cv2,
                preview,
                [
                    f"record {new_frame.get('record_id')} frame={new_frame.get('frame_idx')} obj={inst_idx}",
                    "y: accept corners | r: retry this object | s/Esc: skip record | q: save and quit",
                ],
                ["y", "r", "s", "q"],
                "bbox-corner preview",
            )
            cv2.destroyWindow("bbox-corner preview")
            if action == "y":
                break
            if action == "r":
                inst.pop("corners_2d_full", None)
                inst.pop("corner_valid", None)
                continue
            if action == "s":
                inst.pop("corners_2d_full", None)
                inst.pop("corner_valid", None)
                return "skip", None
            if action == "q":
                inst.pop("corners_2d_full", None)
                inst.pop("corner_valid", None)
                return "quit", None

    if all("corners_2d_full" in inst for inst in instances):
        new_frame["status"] = "full"
    if args.append_full_records:
        base_id = str(new_frame.get("record_id") or Path(new_frame["image_path"]).stem)
        new_frame["record_id"] = next_appended_record_id(data, base_id)
    return "done", new_frame


def validate_args(args: argparse.Namespace) -> None:
    if args.crop_size <= 0 or args.heatmap_size <= 0:
        raise ValueError("--crop-size and --heatmap-size must be positive.")
    if args.bbox_padding < 0:
        raise ValueError("--bbox-padding must be >= 0.")
    if args.corner_display_side < 0:
        raise ValueError("--corner-display-side must be >= 0.")
    if args.start_sec is not None and args.start_sec < 0:
        raise ValueError("--start-sec must be >= 0.")
    if args.end_sec is not None and args.start_sec is not None and args.end_sec <= args.start_sec:
        raise ValueError("--end-sec must be greater than --start-sec.")
    if args.max_records is not None and args.max_records <= 0:
        raise ValueError("--max-records must be positive.")
    if args.export_every <= 0:
        raise ValueError("--export-every must be positive.")


def maybe_export(data: Dict[str, Any], args: argparse.Namespace, completed: int) -> None:
    if args.export_mode == "each":
        export_all(data, args.output_root, args)
        print("Exported YOLO/MyNet data.")
    elif args.export_mode == "batch" and completed > 0 and completed % args.export_every == 0:
        export_all(data, args.output_root, args)
        print(f"Exported YOLO/MyNet data after {completed} completed records.")


def final_export(data: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.export_mode in {"final", "batch"}:
        print("Exporting YOLO/MyNet data...")
        export_all(data, args.output_root, args)
        print("Export complete.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.annotations_json = args.annotations_json or (args.output_root / "annotations.json")
    if not args.annotations_json.exists():
        raise FileNotFoundError(f"annotations.json not found: {args.annotations_json}")

    data = load_json(args.annotations_json)
    if args.export_only:
        print("Exporting YOLO/MyNet data from annotations.json...")
        export_all(data, args.output_root, args)
        print(f"Updated MyNet data: {args.output_root / 'mynet'}")
        print(f"Updated YOLO data: {args.output_root / 'yolo'}")
        return

    cv2 = load_cv2()
    frames: List[Dict[str, Any]] = data.get("frames", [])
    targets = [(idx, frame) for idx, frame in enumerate(frames) if should_visit(frame, args)]
    if args.max_records is not None:
        targets = targets[: args.max_records]

    print(f"Found {len(targets)} bbox/corner records to annotate.")
    completed = 0
    for index, frame in targets:
        print(f"Annotating record={frame.get('record_id')} frame={frame.get('frame_idx')} split={frame.get('split')}")
        result, new_frame = annotate_record(cv2, data, frame, args)
        if result == "quit":
            save_json(args.annotations_json, data)
            final_export(data, args)
            print("Saved current progress. Quit requested.")
            return
        if result == "skip" or new_frame is None:
            print(f"Skipped record={frame.get('record_id')}")
            continue
        replace_frame_record(data, index, new_frame, args.append_full_records)
        save_json(args.annotations_json, data)
        completed += 1
        maybe_export(data, args, completed)
        print(f"Saved record={new_frame.get('record_id')} as {new_frame.get('status')}.")

    save_json(args.annotations_json, data)
    final_export(data, args)
    cv2.destroyAllWindows()
    print(f"Done. Completed {completed}/{len(targets)} records.")
    print(f"Updated annotations: {args.annotations_json}")
    print(f"MyNet data: {args.output_root / 'mynet'}")


if __name__ == "__main__":
    main()
