from __future__ import annotations

import argparse
import json
import math
import random
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.mynet.dataset import (
    BOPCornerDataset,
    INPUT_MODE_RECT_DYNAMIC,
    RECT_BATCH_MODE_ASPECT_BUCKET,
    RECT_BATCH_MODE_STRICT,
    SCALE_AUG_NONE,
    SCALE_AUG_REAL_VIDEO_COVERAGE,
    collate_corner_batch,
)
from src.mynet.decode import corner_metrics
from src.mynet.losses import corner_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MyNet ResNet34 8-corner heatmap model.")
    parser.add_argument("--data-root", type=Path, default=Path("data/dji_action4_mynet"))
    parser.add_argument("--train-index", type=Path, default=None)
    parser.add_argument("--val-index", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("models/checkpoints/mynet_resnet34"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-mode", choices=["auto", "fixed", "rect_dynamic"], default="auto")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--rect-batch-mode", choices=[RECT_BATCH_MODE_STRICT, RECT_BATCH_MODE_ASPECT_BUCKET], default=RECT_BATCH_MODE_ASPECT_BUCKET)
    parser.add_argument("--aspect-buckets", type=int, default=5)
    parser.add_argument("--scale-aug-mode", choices=[SCALE_AUG_NONE, SCALE_AUG_REAL_VIDEO_COVERAGE], default=SCALE_AUG_REAL_VIDEO_COVERAGE)
    parser.add_argument("--scale-long-edge-min", type=int, default=320)
    parser.add_argument("--scale-long-edge-max", type=int, default=768)
    parser.add_argument("--scale-short-edge-min", type=int, default=180)
    parser.add_argument("--val-scale-long-edge", type=int, default=None, help="If set, resize rect_dynamic validation crops to this fixed long edge.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--decode-method", choices=["argmax", "subpixel"], default="subpixel")
    parser.add_argument("--subpixel-window", type=int, default=5)
    parser.add_argument("--invisible-peak-threshold", type=float, default=0.3)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-bn", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None, help="Optional training-step cap per epoch for quick iteration.")
    parser.add_argument("--max-val-steps", type=int, default=None, help="Optional validation-step cap per epoch for quick iteration.")
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-dir", type=Path, default=None)
    return parser.parse_args()


class AspectRatioBatchSampler:
    def __init__(self, dataset: BOPCornerDataset, batch_size: int, bucket_count: int, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.bucket_count = max(1, int(bucket_count))
        self.shuffle = shuffle
        self.buckets = self._build_buckets()

    def _ratio_for_sample(self, idx: int) -> float:
        sample = self.dataset.samples[idx]
        box = sample.get("bbox_xyxy_full")
        if not isinstance(box, list) or len(box) != 4:
            return 1.0
        width = max(1e-6, float(box[2]) - float(box[0]))
        height = max(1e-6, float(box[3]) - float(box[1]))
        return width / height

    def _build_buckets(self) -> List[List[int]]:
        indices = list(range(len(self.dataset)))
        indices.sort(key=self._ratio_for_sample)
        bucket_size = max(1, int(math.ceil(len(indices) / float(self.bucket_count))))
        return [indices[start : start + bucket_size] for start in range(0, len(indices), bucket_size)]

    def __iter__(self) -> Iterator[List[int]]:
        batches: List[List[int]] = []
        for bucket in self.buckets:
            indices = list(bucket)
            if self.shuffle:
                random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if batch:
                    batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return sum(int(math.ceil(len(bucket) / float(self.batch_size))) for bucket in self.buckets)


def make_loader(
    dataset: BOPCornerDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
    *,
    rect_batch_mode: str = RECT_BATCH_MODE_STRICT,
    scale_aug_mode: str = SCALE_AUG_NONE,
    scale_long_edge_min: int = 320,
    scale_long_edge_max: int = 768,
    scale_short_edge_min: int = 180,
    heatmap_sigma: float = 2.0,
    aspect_buckets: int = 5,
) -> DataLoader:
    collate_fn = partial(
        collate_corner_batch,
        rect_batch_mode=rect_batch_mode,
        scale_aug_mode=scale_aug_mode,
        scale_long_edge_min=scale_long_edge_min,
        scale_long_edge_max=scale_long_edge_max,
        scale_short_edge_min=scale_short_edge_min,
        heatmap_sigma=heatmap_sigma,
    )
    if dataset.input_mode == INPUT_MODE_RECT_DYNAMIC and rect_batch_mode == RECT_BATCH_MODE_ASPECT_BUCKET and shuffle:
        batch_sampler = AspectRatioBatchSampler(dataset, batch_size=batch_size, bucket_count=aspect_buckets, shuffle=True)
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_fn,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
    )


def move_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def aggregate(values: Iterable[Dict[str, float]]) -> Dict[str, float]:
    values = list(values)
    if not values:
        return {}
    keys = values[0].keys()
    return {key: sum(item[key] for item in values) / len(values) for key in keys}


def set_batchnorm_eval(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def compute_loss(logits: torch.Tensor, batch: Dict[str, object], args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    corner_valid = batch["corner_valid"] if args.input_mode == INPUT_MODE_RECT_DYNAMIC else None
    return corner_loss(logits, batch["heatmap"], corner_valid=corner_valid)


def is_finite_scalar(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).item())


def _format_tensor_values(value: object, *, max_items: int = 8) -> str:
    if not torch.is_tensor(value):
        return str(value)
    flat = value.detach().cpu().flatten()
    items = flat[:max_items].tolist()
    suffix = ", ..." if flat.numel() > max_items else ""
    return f"{items}{suffix}"


def describe_nonfinite_batch(batch: Dict[str, object], logits: torch.Tensor, loss: torch.Tensor) -> str:
    sample_ids = batch.get("sample_id", ["?"])
    parts = [
        f"sample_id={sample_ids[0] if isinstance(sample_ids, list) and sample_ids else sample_ids}",
        f"loss={float(loss.detach().cpu().item())}",
    ]
    for key in ("crop_hw", "original_crop_hw", "scale_factor"):
        if key in batch:
            parts.append(f"{key}={_format_tensor_values(batch[key])}")
    for key in ("image", "heatmap", "corner_valid"):
        value = batch.get(key)
        if torch.is_tensor(value):
            finite = bool(torch.isfinite(value).all().item())
            parts.append(f"{key}_finite={finite}")
    parts.append(f"logits_finite={bool(torch.isfinite(logits).all().item())}")
    return " ".join(parts)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    writer: Optional[object] = None,
    global_step: int = 0,
) -> tuple[Dict[str, float], int]:
    model.train()
    if args.freeze_bn:
        set_batchnorm_eval(model)
    losses = []
    valid_corner_ratios = []
    use_amp = args.amp and device.type == "cuda"
    accumulation_steps = max(1, int(args.gradient_accumulation_steps))
    accumulated_batches = 0
    fp32_fallbacks = 0
    skipped_nonfinite = 0
    max_steps = len(loader)
    if args.max_steps_per_epoch is not None:
        max_steps = min(max_steps, max(1, int(args.max_steps_per_epoch)))
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        if step > max_steps:
            break
        batch = move_to_device(batch, device)
        valid_ratio = float(batch["corner_valid"].float().mean().item())
        valid_corner_ratios.append(valid_ratio)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(batch["image"])
            loss_parts = compute_loss(logits, batch, args)
            loss = loss_parts["loss"]

        used_fp32_fallback = False
        if not is_finite_scalar(loss) and use_amp:
            with torch.autocast(device_type=device.type, enabled=False):
                logits = model(batch["image"].float())
                loss_parts = compute_loss(logits, batch, args)
                loss = loss_parts["loss"]
            if is_finite_scalar(loss):
                fp32_fallbacks += 1
                used_fp32_fallback = True
                if fp32_fallbacks <= 5:
                    print(
                        f"warning: AMP produced non-finite loss at epoch {epoch:03d} step {step:05d}; "
                        "recovered this batch with FP32 forward."
                    )

        if not is_finite_scalar(loss):
            skipped_nonfinite += 1
            if skipped_nonfinite <= 10:
                print(
                    f"warning: skipping non-finite train batch at epoch {epoch:03d} step {step:05d}; "
                    f"{describe_nonfinite_batch(batch, logits, loss)}"
                )
            elif skipped_nonfinite == 11:
                print("warning: suppressing further non-finite batch details for this epoch.")
            if step % args.log_every == 0:
                print(
                    f"epoch {epoch:03d} step {step:05d}/{max_steps:05d}"
                    f" train_loss=nan valid_ratio={valid_ratio:.3f}"
                    f" skipped_nonfinite={skipped_nonfinite}"
                )
            continue

        scaler.scale(loss / accumulation_steps).backward()
        accumulated_batches += 1
        if accumulated_batches >= accumulation_steps:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accumulated_batches = 0

        loss_value = float(loss.item())
        losses.append(loss_value)

        if writer is not None:
            writer.add_scalar("train/loss_step", loss_value, global_step)
            writer.add_scalar("train/valid_corner_ratio_step", valid_ratio, global_step)
            writer.add_scalar("train/fp32_fallback_step", 1.0 if used_fp32_fallback else 0.0, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
        global_step += 1

        if step % args.log_every == 0:
            print(
                f"epoch {epoch:03d} step {step:05d}/{max_steps:05d}"
                f" train_loss={loss_value:.6f}"
                f" valid_ratio={valid_ratio:.3f}"
            )

    if accumulated_batches > 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return {
        "loss": sum(losses) / max(1, len(losses)),
        "valid_corner_ratio": sum(valid_corner_ratios) / max(1, len(valid_corner_ratios)),
        "fp32_fallbacks": float(fp32_fallbacks),
        "skipped_nonfinite": float(skipped_nonfinite),
    }, global_step


@torch.no_grad()
def evaluate(model: nn.Module, loader: Optional[DataLoader], device: torch.device, args: argparse.Namespace) -> Dict[str, float]:
    if loader is None:
        return {}
    model.eval()
    records = []
    max_steps = len(loader)
    if args.max_val_steps is not None:
        max_steps = min(max_steps, max(1, int(args.max_val_steps)))
    for step, batch in enumerate(loader, start=1):
        if step > max_steps:
            break
        batch = move_to_device(batch, device)
        logits = model(batch["image"])
        loss_parts = compute_loss(logits, batch, args)
        metrics = corner_metrics(
            logits,
            batch["corners_2d_crop"],
            batch["corner_valid"],
            crop_size=args.crop_size,
            crop_hw=batch.get("crop_hw") if args.input_mode == INPUT_MODE_RECT_DYNAMIC else None,
            decode_method=args.decode_method,
            subpixel_window=args.subpixel_window,
            invisible_peak_threshold=args.invisible_peak_threshold,
        )
        metrics.update({key: float(value.item()) for key, value in loss_parts.items()})
        records.append(metrics)
    return aggregate(records)


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: Dict[str, float], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def make_summary_writer(args: argparse.Namespace) -> Optional[object]:
    if not args.tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("TensorBoard is enabled, but the tensorboard package is not installed. Run: uv pip install tensorboard") from exc

    log_dir = args.log_dir or (args.output_dir / "tensorboard")
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def append_metrics_jsonl(path: Path, record: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    args.train_index = args.train_index or (args.data_root / "train.json")
    args.val_index = args.val_index or (args.data_root / "val.json")
    device = torch.device(args.device)

    train_dataset = BOPCornerDataset(
        args.train_index,
        dataset_root=args.data_root,
        input_mode=args.input_mode,
        heatmap_sigma=args.sigma,
        scale_aug_mode=SCALE_AUG_NONE,
    )
    args.input_mode = train_dataset.input_mode if args.input_mode == "auto" else args.input_mode
    if args.gradient_accumulation_steps is None:
        if args.input_mode == INPUT_MODE_RECT_DYNAMIC and args.rect_batch_mode == RECT_BATCH_MODE_STRICT:
            args.gradient_accumulation_steps = 32
        elif args.input_mode == INPUT_MODE_RECT_DYNAMIC:
            args.gradient_accumulation_steps = max(1, int(math.ceil(128 / float(max(1, args.batch_size)))))
        else:
            args.gradient_accumulation_steps = 1
    if args.freeze_bn is None:
        args.freeze_bn = args.input_mode == INPUT_MODE_RECT_DYNAMIC
    if args.input_mode == INPUT_MODE_RECT_DYNAMIC and args.rect_batch_mode == RECT_BATCH_MODE_STRICT and args.batch_size != 1:
        print("rect_dynamic uses strict variable-size tensors; forcing physical batch_size=1.")
        args.batch_size = 1

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        device=device,
        rect_batch_mode=args.rect_batch_mode if args.input_mode == INPUT_MODE_RECT_DYNAMIC else RECT_BATCH_MODE_STRICT,
        scale_aug_mode=args.scale_aug_mode if args.input_mode == INPUT_MODE_RECT_DYNAMIC else SCALE_AUG_NONE,
        scale_long_edge_min=args.scale_long_edge_min,
        scale_long_edge_max=args.scale_long_edge_max,
        scale_short_edge_min=args.scale_short_edge_min,
        heatmap_sigma=args.sigma,
        aspect_buckets=args.aspect_buckets,
    )
    val_loader = None
    if args.val_index.exists():
        val_dataset = BOPCornerDataset(
            args.val_index,
            dataset_root=args.data_root,
            input_mode=args.input_mode,
            heatmap_sigma=args.sigma,
            scale_aug_mode=SCALE_AUG_NONE,
        )
        if len(val_dataset) > 0:
            val_resized = args.input_mode == INPUT_MODE_RECT_DYNAMIC and args.val_scale_long_edge is not None
            val_batch_size = 1 if args.input_mode == INPUT_MODE_RECT_DYNAMIC else args.batch_size
            val_loader = make_loader(
                val_dataset,
                val_batch_size,
                args.num_workers,
                shuffle=False,
                device=device,
                rect_batch_mode=RECT_BATCH_MODE_ASPECT_BUCKET if val_resized else RECT_BATCH_MODE_STRICT,
                scale_aug_mode=SCALE_AUG_REAL_VIDEO_COVERAGE if val_resized else SCALE_AUG_NONE,
                scale_long_edge_min=args.val_scale_long_edge or args.scale_long_edge_min,
                scale_long_edge_max=args.val_scale_long_edge or args.scale_long_edge_max,
                scale_short_edge_min=args.scale_short_edge_min,
                heatmap_sigma=args.sigma,
            )

    from src.mynet.model import CornerResNet34

    model = CornerResNet34(out_channels=8, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    writer = make_summary_writer(args)
    metrics_path = args.output_dir / "metrics.jsonl"
    best_val = float("inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics, global_step = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args, writer, global_step)
        val_metrics = evaluate(model, val_loader, device, args)
        if writer is not None:
            writer.add_scalar("train/loss_epoch", train_metrics["loss"], epoch)
            writer.add_scalar("train/valid_corner_ratio_epoch", train_metrics["valid_corner_ratio"], epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(f"val/{key}", value, epoch)
            writer.flush()

        append_metrics_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "global_step": global_step,
                "train": train_metrics,
                "val": val_metrics,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

        msg = (
            f"epoch {epoch:03d} train_loss={train_metrics['loss']:.6f}"
            f" valid_ratio={train_metrics['valid_corner_ratio']:.3f}"
        )
        if val_metrics:
            msg += (
                f" val_loss={val_metrics['loss']:.6f}"
                f" corner_px={val_metrics['corner_px']:.3f}"
                f" pck10={val_metrics['pck_10']:.3f}"
            )
        print(msg)

        current_val = val_metrics.get("loss", train_metrics["loss"])
        if current_val < best_val:
            best_val = current_val
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, val_metrics or train_metrics, args)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(args.output_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, val_metrics or train_metrics, args)

    save_checkpoint(args.output_dir / "last.pt", model, optimizer, args.epochs, {"best_val_loss": best_val}, args)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
