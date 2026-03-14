#!/usr/bin/env python3
"""
Pre-process DSEC sequences into the event-tensor format expected by the ECDDP model.

The original pipeline in ``src/pre_dse.py`` accumulates events into RGB images so
they can be paired with warped RGB frames.  However, the ECDDP Swin encoder
expects multi-channel event tensors (e.g. histogram/voxel stacks) similar to the
representations used in Event-Camera-Data-Dense-Pre-training.  This script
recreates that processing locally so that we can plug ECDDP checkpoints into the
segmentation codebase without relying on the upstream data loader.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
ECDDP_DATA_ROOT = REPO_ROOT / "Event-Camera-Data-Dense-Pre-training" / "data"
if not ECDDP_DATA_ROOT.exists():
    raise FileNotFoundError(
        f"Cannot locate ECDDP data helpers at {ECDDP_DATA_ROOT}. "
        "Please ensure Event-Camera-Data-Dense-Pre-training is available."
    )
if str(ECDDP_DATA_ROOT) not in sys.path:
    sys.path.append(str(ECDDP_DATA_ROOT))

from seg_utils import EventSlicer as DSECEventSlicer  # type: ignore  # noqa: E402
from seg_utils import generate_input_representation  # type: ignore  # noqa: E402


def _load_rectify_map(rectify_path: Path) -> Optional[np.ndarray]:
    if not rectify_path.exists():
        print(f"[WARN] Missing rectify map: {rectify_path}")
        return None
    with h5py.File(rectify_path, "r") as rect_file:
        return rect_file["rectify_map"][()]


def _load_timestamps(timestamp_path: Path) -> np.ndarray:
    if not timestamp_path.exists():
        raise FileNotFoundError(f"Missing timestamp file: {timestamp_path}")
    timestamps = np.loadtxt(timestamp_path, dtype=np.int64)
    if timestamps.ndim == 0:
        timestamps = np.asarray([timestamps], dtype=np.int64)
    return timestamps


def _list_images(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    return sorted(image_dir.glob("*.png"))


def _build_sample_pairs(image_dir: Path, timestamps_path: Path) -> List[Tuple[Path, int]]:
    images = _list_images(image_dir)
    timestamps = _load_timestamps(timestamps_path)
    if len(images) != len(timestamps):
        raise RuntimeError(
            f"Timestamp count ({len(timestamps)}) does not match image count ({len(images)}) "
            f"under {image_dir}"
        )
    return [(img, int(ts)) for img, ts in zip(images, timestamps)]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_tensor(
    tensor: torch.Tensor,
    timestamp: int,
    out_path: Path,
    fmt: str,
) -> None:
    if fmt == "pt":
        torch.save({"tensor": tensor, "timestamp": int(timestamp)}, out_path)
    elif fmt == "npz":
        np.savez_compressed(out_path, tensor=tensor.numpy(), timestamp=np.array(timestamp, dtype=np.int64))
    elif fmt == "npy":
        np.save(out_path, tensor.numpy())
    else:
        raise ValueError(f"Unsupported save format: {fmt}")


@dataclass
class ProcessorConfig:
    root: Path
    split: str
    output_subdir: str
    nr_events_data: int
    nr_events_per_data: int
    nr_bins_per_data: int
    representation: str
    separate_pol: bool
    normalize_event: bool
    trim_bottom: int
    resize_shape: Optional[Tuple[int, int]]
    resize_mode: str
    output_format: str
    overwrite: bool
    min_events: int


class ECDDPPreprocessor:
    def __init__(self, cfg: ProcessorConfig):
        self.cfg = cfg
        split = cfg.split
        root = cfg.root
        self.image_root = root / f"{split}_images"
        self.events_root = root / f"{split}_events"
        self.calib_root = root / f"{split}_calibration"
        self.semantic_root = root / f"{split}_semantic_segmentation" / split

    def run(self, sequences: Optional[Sequence[str]] = None) -> None:
        candidates = sorted(p.name for p in self.image_root.iterdir() if p.is_dir())
        if sequences:
            missing = [seq for seq in sequences if seq not in candidates]
            if missing:
                raise ValueError(f"Unknown sequences: {missing}")
            target = [seq for seq in candidates if seq in sequences]
        else:
            target = candidates

        for seq in target:
            print(f"[INFO] Processing {seq} ({self.cfg.split})")
            self._process_sequence(seq)

    def _sequence_paths(self, sequence: str) -> Dict[str, Path]:
        image_dir = self.image_root / sequence / "images" / "left" / "rectified"
        timestamp_path = self.image_root / sequence / "images" / "timestamps.txt"
        event_path = self.events_root / sequence / "events" / "left" / "events.h5"
        rectify_path = self.events_root / sequence / "events" / "left" / "rectify_map.h5"
        calib_dir = self.calib_root / sequence / "calibration" / "cam_to_cam.yaml"
        output_dir = self.image_root / sequence / "images" / "left" / self.cfg.output_subdir
        return {
            "images": image_dir,
            "timestamps": timestamp_path,
            "events": event_path,
            "rectify": rectify_path,
            "calib": calib_dir,
            "output": output_dir,
        }

    def _process_sequence(self, sequence: str) -> None:
        paths = self._sequence_paths(sequence)
        sample_pairs = _build_sample_pairs(paths["images"], paths["timestamps"])
        if not sample_pairs:
            print(f"[WARN] No samples found for {sequence}")
            return

        if not paths["events"].exists():
            print(f"[WARN] Missing events file for {sequence}, skipping.")
            return

        rectify_map = _load_rectify_map(paths["rectify"])
        if rectify_map is None:
            print(f"[WARN] Rectify map missing for {sequence}; events will stay in raw coordinates.")
        event_shape = rectify_map.shape[:2] if rectify_map is not None else (480, 640)
        _ensure_dir(paths["output"])
        self._write_metadata(paths["output"])

        with h5py.File(paths["events"], "r") as event_file:
            slicer = DSECEventSlicer(event_file)
            total_events = self.cfg.nr_events_data * self.cfg.nr_events_per_data

            for image_path, timestamp in tqdm(sample_pairs, desc=f"{sequence}", unit="frame"):
                out_path = paths["output"] / f"{Path(image_path).stem}.{self.cfg.output_format}"
                if out_path.exists() and not self.cfg.overwrite:
                    continue

                event_dict = slicer.get_events_fixed_num(int(timestamp), nr_events=total_events)
                if event_dict is None or event_dict["t"].size < self.cfg.min_events:
                    continue

                stacked = self._rectify_and_stack(event_dict, rectify_map, event_shape)
                if stacked is None:
                    continue

                tensor = self._build_tensor(stacked, event_shape)
                if tensor is None:
                    continue

                tensor = self._post_process(tensor)
                _save_tensor(tensor, timestamp, out_path, self.cfg.output_format)

    def _rectify_and_stack(
        self,
        event_dict: Dict[str, np.ndarray],
        rectify_map: Optional[np.ndarray],
        event_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        x = event_dict["x"].astype(np.int64)
        y = event_dict["y"].astype(np.int64)
        t = event_dict["t"].astype(np.float32)
        p = event_dict["p"].astype(np.int8)

        h, w = event_shape
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        if not np.any(valid):
            return None
        x = x[valid]
        y = y[valid]
        t = t[valid]
        p = p[valid]

        if rectify_map is not None:
            coords = rectify_map[y, x]
            x_rect = coords[:, 0]
            y_rect = coords[:, 1]
        else:
            x_rect = x
            y_rect = y

        events = np.stack([x_rect, y_rect, t, p], axis=1)
        return events

    def _build_tensor(self, events: np.ndarray, event_shape: Tuple[int, int]) -> Optional[torch.Tensor]:
        num_events = events.shape[0]
        if num_events < self.cfg.min_events:
            return None

        channels = self.cfg.nr_events_data * self.cfg.nr_bins_per_data
        tensor = np.zeros((channels, event_shape[0], event_shape[1]), dtype=np.float32)
        chunk_size = num_events // self.cfg.nr_events_data
        if chunk_size == 0:
            return None

        for idx in range(self.cfg.nr_events_data):
            start = idx * chunk_size
            end = start + chunk_size
            if idx == self.cfg.nr_events_data - 1:
                end = num_events
            chunk = events[start:end].copy()
            if chunk.shape[0] == 0:
                continue
            repr_slice = generate_input_representation(
                chunk,
                self.cfg.representation,
                event_shape,
                nr_temporal_bins=self.cfg.nr_bins_per_data,
                separate_pol=self.cfg.separate_pol,
                normalize_event=self.cfg.normalize_event,
            )
            c_start = idx * self.cfg.nr_bins_per_data
            c_end = c_start + self.cfg.nr_bins_per_data
            tensor[c_start:c_end] = repr_slice

        return torch.from_numpy(tensor)

    def _post_process(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.cfg.trim_bottom > 0:
            tensor = tensor[:, :-self.cfg.trim_bottom, :]
        if self.cfg.resize_shape is not None:
            target_h, target_w = self.cfg.resize_shape
            interp_kwargs = {}
            if self.cfg.resize_mode in {"bilinear", "bicubic"}:
                interp_kwargs["align_corners"] = False
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(target_h, target_w),
                mode=self.cfg.resize_mode,
                **interp_kwargs,
            ).squeeze(0)
        return tensor.contiguous()

    def _write_metadata(self, output_dir: Path) -> None:
        meta_path = output_dir / "ecddp_config.json"
        meta = {
            "nr_events_data": self.cfg.nr_events_data,
            "nr_events_per_data": self.cfg.nr_events_per_data,
            "nr_bins_per_data": self.cfg.nr_bins_per_data,
            "representation": self.cfg.representation,
            "separate_pol": self.cfg.separate_pol,
            "normalize_event": self.cfg.normalize_event,
            "trim_bottom": self.cfg.trim_bottom,
            "resize_shape": self.cfg.resize_shape,
            "resize_mode": self.cfg.resize_mode,
            "output_format": self.cfg.output_format,
        }
        if meta_path.exists():
            return
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ECDDP-style event tensors for DSEC.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/storage/jianwen/DSEC"),
        help="Path to the DSEC root directory.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Dataset split to process (default runs both train and test).",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of sequence names to process (defaults to all).",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="eventTensor_ecddp",
        help="Sub-directory (under images/left) to store the tensors.",
    )
    parser.add_argument("--nr-events-data", type=int, default=10, help="Number of temporal chunks per frame.")
    parser.add_argument(
        "--nr-events-per-data",
        type=int,
        default=50000,
        help="Number of events per temporal chunk (before building bins).",
    )
    parser.add_argument(
        "--nr-bins-per-data",
        type=int,
        default=2,
        help="Number of channels produced per temporal chunk (e.g., histogram => 2).",
    )
    parser.add_argument(
        "--representation",
        choices=["histogram", "voxel_grid"],
        default="histogram",
        help="Event representation to use inside each chunk.",
    )
    parser.add_argument("--separate-pol", action="store_true", help="Keep positive/negative events in separate bins.")
    parser.add_argument("--normalize-event", action="store_true", help="Apply per-bin normalization.")
    parser.add_argument("--trim-bottom", type=int, default=40, help="Rows to drop from the bottom (DSEC default).")
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=(448, 640),
        help="Output spatial size after trimming.",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Disable resizing and keep the trimmed resolution.",
    )
    parser.add_argument(
        "--resize-mode",
        choices=["nearest", "bilinear", "bicubic"],
        default="bilinear",
        help="Interpolation mode used during resizing.",
    )
    parser.add_argument(
        "--output-format",
        choices=["pt", "npz", "npy"],
        default="pt",
        help="File format used when persisting tensors.",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing tensor files (default).",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Skip processing frames that already have tensors.",
    )
    parser.set_defaults(overwrite=True)
    parser.add_argument(
        "--min-events",
        type=int,
        default=None,
        help="Skip frames with fewer than this number of valid events (defaults to nr_events_data).",
    )
    args = parser.parse_args()
    if args.no_resize:
        args.resize = None
    return args


def main() -> None:
    args = parse_args()
    if args.split == "both":
        splits: Iterable[str] = ("train", "test")
    else:
        splits = (args.split,)

    root = args.root.expanduser().resolve()
    sequences = args.sequences
    resize_shape = tuple(args.resize) if args.resize is not None else None
    min_events = args.min_events or args.nr_events_data

    for split in splits:
        cfg = ProcessorConfig(
            root=root,
            split=split,
            output_subdir=args.output_subdir,
            nr_events_data=args.nr_events_data,
            nr_events_per_data=args.nr_events_per_data,
            nr_bins_per_data=args.nr_bins_per_data,
            representation=args.representation,
            separate_pol=args.separate_pol,
            normalize_event=args.normalize_event,
            trim_bottom=args.trim_bottom,
            resize_shape=resize_shape,
            resize_mode=args.resize_mode,
            output_format=args.output_format,
            overwrite=args.overwrite,
            min_events=min_events,
        )
        processor = ECDDPPreprocessor(cfg)
        processor.run(sequences)


if __name__ == "__main__":
    main()
