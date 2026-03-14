import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from config import DEPConfig
from dataset import MVSECDepthDataset, MVSECECDDPDepthDataset
from dep import DepthEstimator, compute_depth_metrics
from torch.utils.data import Dataset


# Hand-crafted color ramp that mimics the magenta → orange → yellow palette
# used in the provided visualization. Values are in RGB and interpolated to
# produce a smooth gradient.
DEPTH_COLOR_STOPS: Sequence[Tuple[float, Tuple[int, int, int]]] = (
    (0.0, (255, 255, 210)),  # near: pale yellow
    (0.25, (255, 180, 100)),  # warm orange
    (0.5, (255, 80, 120)),   # vivid pink
    (0.7, (190, 0, 210)),    # magenta
    (0.85, (70, 0, 120)),    # deep purple
    (1.0, (10, 0, 40)),      # far: dark violet
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVSEC depth inference & visualization helper.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data/storage/jianwen/cache/ckpts/2025-11-19-14:03_dep/checkpoint_003000.pt",
        help="Path to a depth checkpoint (.pt).",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="outdoor_day1",
        help="Comma separated MVSEC sequences or 'all' to use default split.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        choices=("train", "valid", "test"),
        help="Dataset split for MVSEC.",
    )
    parser.add_argument("--frame-dir", type=str, default="depth_frames", help="Directory for per-frame visualizations.")
    parser.add_argument("--output-video", type=str, default="depth_inference.mp4", help="Optional MP4 output path.")
    parser.add_argument("--fps", type=int, default=24, help="Video frame rate.")
    parser.add_argument("--no-video", action="store_true", help="Disable video writing.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N samples.")
    parser.add_argument("--stride", type=int, default=10, help="Process every N-th frame (default: 10).")
    parser.add_argument("--device", type=str, default=None, help="Override DEPConfig.device (e.g. cuda:0).")
    parser.add_argument("--min-depth", type=float, default=0.1, help="Override min depth bound for decoding/coloring.")
    parser.add_argument("--max-depth", type=float, default=30, help="Override max depth bound for decoding/coloring.")
    parser.add_argument("--no-gt", action="store_true", help="Skip GT visualization/metrics (if annotations missing).")
    parser.add_argument("--show-error", action="store_true", help="Visualize |pred - gt| heatmap.")
    parser.add_argument("--cmap-size", type=int, default=512, help="Number of bins in the depth color lookup table.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict=True when loading checkpoints (default: non-strict to allow missing EMA buffers, etc.).",
    )
    return parser.parse_args()


def build_colormap(
    stops: Sequence[Tuple[float, Tuple[int, int, int]]],
    size: int,
) -> np.ndarray:
    if size <= 1:
        raise ValueError("Colormap size must be >= 2")
    sorted_stops = sorted(stops, key=lambda item: item[0])
    lut = np.zeros((size, 3), dtype=np.float32)
    last_idx = 0
    last_color = np.array(sorted_stops[0][1], dtype=np.float32)
    for t, color in sorted_stops[1:]:
        idx = int(round(t * (size - 1)))
        idx = np.clip(idx, last_idx + 1, size - 1)
        target = np.array(color, dtype=np.float32)
        span = idx - last_idx
        for c in range(3):
            lut[last_idx: idx + 1, c] = np.linspace(last_color[c], target[c], span + 1)
        last_idx = idx
        last_color = target
    lut[-1] = last_color
    return lut.astype(np.uint8)


def tensor_to_rgb_image(
    tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    img = tensor.detach().cpu()
    if img.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got {img.shape}")
    mean_t = torch.as_tensor(mean, dtype=img.dtype)
    std_t = torch.as_tensor(std, dtype=img.dtype)
    if mean_t.numel() == 1:
        mean_t = mean_t.repeat(img.shape[0])
    if std_t.numel() == 1:
        std_t = std_t.repeat(img.shape[0])
    if mean_t.numel() != img.shape[0]:
        repeats = math.ceil(img.shape[0] / mean_t.numel())
        mean_t = mean_t.repeat(repeats)[: img.shape[0]]
    if std_t.numel() != img.shape[0]:
        repeats = math.ceil(img.shape[0] / std_t.numel())
        std_t = std_t.repeat(repeats)[: img.shape[0]]
    mean_t = mean_t.view(-1, 1, 1)
    std_t = std_t.view(-1, 1, 1)
    vis = img * std_t + mean_t
    vis = vis.clamp(0.0, 1.0)
    if vis.shape[0] == 1:
        vis = vis.repeat(3, 1, 1)
    elif vis.shape[0] != 3:
        vis = vis.mean(dim=0, keepdim=True).repeat(3, 1, 1)
    vis = vis.permute(1, 2, 0).numpy()
    return (vis * 255.0).astype(np.uint8)


def prepare_intrinsics_for_inference(intrinsics: Optional[Dict[str, float]], device: torch.device):
    if not intrinsics:
        return None
    result = {}
    for key, value in intrinsics.items():
        if value is None:
            continue
        result[key] = torch.tensor([float(value)], device=device, dtype=torch.float32)
    return result


def colorize_depth(
    depth: torch.Tensor,
    mask: torch.Tensor,
    colormap: np.ndarray,
    vmin: Optional[float],
    vmax: Optional[float],
    normalize_per_map: bool = True,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    depth_np = depth.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy() > 0.5
    valid_values = depth_np[mask_np]
    if normalize_per_map and valid_values.size > 0:
        vmin_local = float(valid_values.min())
        vmax_local = float(valid_values.max())
    else:
        vmin_local = float(vmin) if vmin is not None else float(depth_np.min())
        vmax_local = float(vmax) if vmax is not None else float(depth_np.max())
    if vmax_local <= vmin_local:
        vmax_local = vmin_local + 1e-3
    norm = np.zeros_like(depth_np, dtype=np.float32)
    if valid_values.size > 0:
        norm[mask_np] = (depth_np[mask_np] - vmin_local) / (vmax_local - vmin_local)
    norm = np.clip(norm, 0.0, 1.0)
    idx = (norm * (len(colormap) - 1)).round().astype(np.int32)
    idx = np.clip(idx, 0, len(colormap) - 1)
    color = colormap[idx]
    color[~mask_np] = 0
    return color.astype(np.uint8), (vmin_local, vmax_local)


def colorize_error_map(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> np.ndarray:
    diff = (pred - target).abs().detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy() > 0.5
    valid = diff[mask_np]
    err = np.zeros_like(diff, dtype=np.float32)
    if valid.size > 0:
        vmax = valid.max()
        if vmax < 1e-6:
            vmax = 1e-6
        err[mask_np] = np.clip(diff[mask_np] / vmax, 0.0, 1.0)
    err_u8 = (err * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(err_u8, cv2.COLORMAP_TURBO)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    heat_rgb[~mask_np] = 0
    return heat_rgb


def load_model(cfg: DEPConfig, checkpoint_path: str, strict: bool) -> DepthEstimator:
    model = DepthEstimator(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    load_result = model.load_state_dict(state, strict=strict)
    if isinstance(load_result, tuple):
        missing, unexpected = load_result
        if missing:
            print("[depth] missing keys:", missing)
        if unexpected:
            print("[depth] unexpected keys:", unexpected)
    print(f"Loaded checkpoint from {checkpoint_path}")
    model.to(cfg.device).eval()
    return model


def _split_sequence_argument(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    if value.lower() == "all":
        return None
    items = [segment.strip() for segment in value.split(",")]
    return [item for item in items if item]


def build_dataset(
    cfg: DEPConfig,
    args: argparse.Namespace,
) -> Tuple[Dataset, Dict[str, Sequence[float]], float | None, float | None]:
    encoder_mode = getattr(cfg, "encoder_mode", "ours")
    sequences = _split_sequence_argument(args.sequence)
    if sequences is None:
        if args.split == "train":
            sequences = cfg.mvsec_train_sequences
        else:
            sequences = cfg.mvsec_valid_sequences
    stats: Dict[str, Sequence[float]] = {
        "event_mean": cfg.MVSEC_EVENT_ME,
        "event_std": cfg.MVSEC_EVENT_SE,
        "image_mean": cfg.MVSEC_IMAGE_ME,
        "image_std": cfg.MVSEC_IMAGE_SE,
    }
    effective_min = args.min_depth if args.min_depth is not None else cfg.mvsec_eval_min_depth
    effective_max = args.max_depth if args.max_depth is not None else cfg.mvsec_eval_max_depth
    dataset_kwargs: Dict[str, object] = {}
    if encoder_mode == "ecddp":
        dataset_cls = MVSECECDDPDepthDataset
        dataset_kwargs.update(
            tensor_root=cfg.mvsec_root,
            tensor_subdir=getattr(cfg, "ecddp_tensor_subdir", "eventTensor_ecddp"),
            tensor_exts=getattr(cfg, "ecddp_tensor_exts", (".pt", ".npz", ".npy")),
        )
    else:
        dataset_cls = MVSECDepthDataset
    dataset = dataset_cls(
        root_dir=cfg.mvsec_root,
        split=args.split,
        sequences=sequences,
        transform=cfg.mvsec_valid_transform,
        min_depth=effective_min,
        max_depth=effective_max,
        depth_scale=cfg.mvsec_depth_scale,
        calibration_root=cfg.mvsec_calibration_root,
        **dataset_kwargs,
    )
    return dataset, stats, effective_min, effective_max


def frame_name_from_meta(meta: dict, index: int) -> str:
    seq = meta.get("sequence") or meta.get("cluster") or meta.get("scene") or "sample"
    frame_id = (
        meta.get("frame_id") if "frame_id" in meta else meta.get("frame_idx", meta.get("index", index))
    )
    if isinstance(frame_id, int):
        frame_tag = f"{frame_id:06d}"
    else:
        frame_tag = str(frame_id)
    safe_seq = str(seq).replace("/", "_")
    return f"{safe_seq}_{frame_tag}"


def ensure_video_writer(path: str, fps: int, height: int, width: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer for {path}")
    return writer


def main():
    args = parse_args()
    cfg = DEPConfig()
    if args.device is not None:
        cfg.device = args.device
    if args.stride <= 0:
        raise ValueError("--stride must be >= 1")

    dataset, stats, dataset_min, dataset_max = build_dataset(cfg, args)
    print(f"Loaded MVSEC/{args.split} dataset with {len(dataset)} samples.")

    model = load_model(cfg, args.checkpoint, strict=args.strict)
    color_lut = build_colormap(DEPTH_COLOR_STOPS, args.cmap_size)
    frame_root = Path(args.frame_dir)
    if getattr(cfg, "encoder_mode", "ours") == "ecddp":
        frame_root = frame_root.with_name(frame_root.name + "_ecddp")
    frame_root.mkdir(parents=True, exist_ok=True)

    total_samples = len(dataset)
    max_samples = total_samples if args.limit is None else min(args.limit, total_samples)
    writer = None
    maybe_video_path = Path(args.output_video)

    accum_metrics = {k: 0.0 for k in cfg.depth_metrics}
    metric_count = 0
    global_min = args.min_depth if args.min_depth is not None else dataset_min
    global_max = args.max_depth if args.max_depth is not None else dataset_max
    sequence_dirs: Dict[str, Path] = {}
    indices = list(range(0, max_samples, max(1, args.stride)))

    with torch.no_grad():
        for idx in tqdm(indices, desc="DepthInfer", unit="frame"):
            sample = dataset[idx]
            event_tensor = sample["event"]
            image_tensor = sample["image"]
            depth_tensor = sample["depth"]
            mask_tensor = sample["mask"]
            meta = sample.get("meta", {})

            event_batch = event_tensor.unsqueeze(0).to(cfg.device) if cfg.use_events else None
            image_batch = image_tensor.unsqueeze(0).to(cfg.device) if cfg.use_rgb else None
            depth_batch = depth_tensor.unsqueeze(0).unsqueeze(0).to(cfg.device)
            mask_batch = mask_tensor.unsqueeze(0).unsqueeze(0).to(cfg.device)
            intr = prepare_intrinsics_for_inference(sample.get("intrinsics"), cfg.device)

            depth_bounds = (global_min, global_max)

            pred = model.forward(
                event=event_batch,
                image=image_batch,
                intrinsics=intr,
                depth_bounds=depth_bounds,
            )
            pred = pred.squeeze(0)

            panels: List[np.ndarray] = []
            event_vis = None
            if cfg.use_events and getattr(cfg, "encoder_mode", "ours") != "ecddp":
                event_vis = tensor_to_rgb_image(event_tensor, stats["event_mean"], stats["event_std"])
                panels.append(event_vis)
            else:
                event_vis = None
            if cfg.use_rgb and getattr(cfg, "encoder_mode", "ours") != "ecddp":
                rgb_vis = tensor_to_rgb_image(image_tensor, stats["image_mean"], stats["image_std"])
                panels.append(rgb_vis)
            else:
                rgb_vis = None

            pred_color, (vis_min, vis_max) = colorize_depth(
                pred.squeeze(0),
                mask_tensor,
                color_lut,
                global_min,
                global_max,
                normalize_per_map=False,
            )
            panels.append(pred_color)

            err_panel = None
            if not args.no_gt:
                gt_color, _ = colorize_depth(
                    depth_tensor,
                    mask_tensor,
                    color_lut,
                    vis_min,
                    vis_max,
                    normalize_per_map=False,
                )
                panels.append(gt_color)
                if args.show_error:
                    err_panel = colorize_error_map(pred.squeeze(0), depth_tensor, mask_tensor)
                    panels.append(err_panel)

                metrics = compute_depth_metrics(pred.unsqueeze(0), depth_batch, mask_batch, global_min, global_max)
                for key, value in metrics.items():
                    if np.isfinite(value):
                        accum_metrics[key] += value
                metric_count += 1

            row = np.concatenate(panels, axis=1)
            if writer is None and not args.no_video:
                writer = ensure_video_writer(str(maybe_video_path), args.fps, row.shape[0], row.shape[1])
                print(f"Writing video to {maybe_video_path} @ {args.fps} FPS.")
            if writer is not None:
                writer.write(cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

            frame_id = frame_name_from_meta(meta, idx)
            seq_name = meta.get("sequence") or (args.sequence if args.sequence.lower() != "all" else "mvsec")
            seq_dir = sequence_dirs.get(seq_name)
            if seq_dir is None:
                seq_dir = frame_root / seq_name
                seq_dir.mkdir(parents=True, exist_ok=True)
                sequence_dirs[seq_name] = seq_dir

            Image.fromarray(pred_color).save(seq_dir / f"{frame_id}_pred.png")
            if not args.no_gt:
                Image.fromarray(gt_color).save(seq_dir / f"{frame_id}_gt.png")
                if args.show_error:
                    Image.fromarray(err_panel if err_panel is not None else np.zeros_like(pred_color)).save(
                        seq_dir / f"{frame_id}_err.png"
                    )
            if cfg.use_events and getattr(cfg, "encoder_mode", "ours") != "ecddp" and event_vis is not None:
                Image.fromarray(event_vis).save(seq_dir / f"{frame_id}_event.png")
            if cfg.use_rgb and getattr(cfg, "encoder_mode", "ours") != "ecddp" and rgb_vis is not None:
                Image.fromarray(rgb_vis).save(seq_dir / f"{frame_id}_rgb.png")

    if writer is not None:
        writer.release()
        print(f"✅ Saved video to {maybe_video_path}")
    print(f"✅ Saved per-frame results to {frame_root.resolve()} (per-sequence subfolders).")

    if metric_count > 0:
        averaged = {k: v / metric_count for k, v in accum_metrics.items()}
        readable = ", ".join(f"{k}={averaged[k]:.4f}" for k in cfg.depth_metrics)
        print(f"Depth metrics over {metric_count} samples: {readable}")


if __name__ == "__main__":
    main()
