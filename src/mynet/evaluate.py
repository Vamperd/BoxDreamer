from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import torch
from torch.utils.data import DataLoader

from src.mynet.dataset import BOPCornerDataset, collate_corner_batch
from src.mynet.decode import corner_metrics
from src.mynet.infer_image import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a MyNet checkpoint on a corner heatmap index.")
    parser.add_argument("--data-root", type=Path, default=Path("data/dji_action4_mynet"))
    parser.add_argument("--val-index", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--decode-method", choices=["argmax", "subpixel"], default="subpixel")
    parser.add_argument("--subpixel-window", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def aggregate(records: Iterable[Dict[str, float]]) -> Dict[str, float]:
    records = list(records)
    if not records:
        return {}
    keys = records[0].keys()
    return {key: sum(item[key] for item in records) / len(records) for key in keys}


def masked_heatmap_mse(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    loss_map = (pred - target) ** 2
    channel_mask = valid.float().unsqueeze(-1).unsqueeze(-1)
    denom = channel_mask.sum() * target.shape[-1] * target.shape[-2]
    if denom.item() <= 0:
        return loss_map.mean()
    return (loss_map * channel_mask).sum() / denom


def move_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    args.val_index = args.val_index or (args.data_root / "val.json")
    if not args.val_index.exists():
        raise FileNotFoundError(f"Validation index not found: {args.val_index}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device)
    dataset = BOPCornerDataset(args.val_index, dataset_root=args.data_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_corner_batch,
    )
    model = load_model(args.checkpoint, device)
    model.eval()

    records = []
    use_amp = args.amp and device.type == "cuda"
    for batch in loader:
        batch = move_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(batch["image"])
            loss = masked_heatmap_mse(logits, batch["heatmap"], batch["corner_valid"])
        metrics = corner_metrics(
            logits,
            batch["corners_2d_crop"],
            batch["corner_valid"],
            crop_size=args.crop_size,
            decode_method=args.decode_method,
            subpixel_window=args.subpixel_window,
        )
        metrics["loss"] = float(loss.item())
        records.append(metrics)

    metrics = aggregate(records)
    result: Dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "val_index": str(args.val_index),
        "samples": len(dataset),
        "crop_size": args.crop_size,
        "decode_method": args.decode_method,
        "subpixel_window": args.subpixel_window,
        "metrics": metrics,
    }
    return result


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
