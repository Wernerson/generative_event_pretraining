#!/usr/bin/env python3
"""
Generate ECDDP-style event tensors for MVSEC sequences so that the depth
training pipeline can consume the same Swin encoder checkpoints as our DSEC
segmentation models.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
ECDDP_DATA_ROOT = REPO_ROOT / "Event-Camera-Data-Dense-Pre-training" / "data"
if not ECDDP_DATA_ROOT.exists():
    raise FileNotFoundError(
        f"Missing ECDDP helpers at {ECDDP_DATA_ROOT}. "
        "Please ensure Event-Camera-Data-Dense-Pre-training is available."
    )
import sys

if str(ECDDP_DATA_ROOT) not in sys.path:
    sys.path.append(str(ECDDP_DATA_ROOT))

from seg_utils import generate_input_representation  # type: ignore  # noqa: E402

from rosbags.highlevel import AnyReader
from rosbags.typesys import get_types_from_msg


@dataclass
class ProcessorConfig:
    output_root: Path
    output_subdir: str
    sequences: Sequence[str]
    nr_events_data: int
    nr_bins_per_data: int
    min_events: int
    representation: str
    separate_pol: bool
    normalize_event: bool
    resize_shape: Optional[Tuple[int, int]]
    resize_mode: str
    output_format: str
    overwrite: bool


@dataclass
class FrameSample:
    index: int
    depth_ts: float
    depth: np.ndarray
    events_x: np.ndarray
    events_y: np.ndarray
    events_t: np.ndarray
    events_p: np.ndarray
    event_start_time: float
    event_end_time: float


def _ensure_output_root(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    allowed_root = Path("/data/storage/jianwen").resolve()
    resolved = out_dir.resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"Output directory must be inside {allowed_root}, got {resolved}") from exc


def _compute_event_bounds(
    depth_ts: np.ndarray,
    event_ts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth_ts = np.asarray(depth_ts, dtype=np.float64)
    event_ts = np.asarray(event_ts, dtype=np.float64)
    if depth_ts.ndim != 1 or event_ts.ndim != 1:
        raise ValueError("timestamps must be 1-D arrays")
    if depth_ts.size == 0:
        raise ValueError("no depth timestamps found")
    if not np.all(np.diff(depth_ts) > 0):
        raise ValueError("depth timestamps must be strictly increasing")
    if not np.all(np.diff(event_ts) >= 0):
        raise ValueError("event timestamps must be non-decreasing")

    n_depth = depth_ts.size
    start_ts = np.empty_like(depth_ts, dtype=np.float64)
    end_ts = np.empty_like(depth_ts, dtype=np.float64)

    if n_depth == 1:
        start_ts[0] = event_ts[0]
        end_ts[0] = event_ts[-1]
    else:
        first_gap = depth_ts[1] - depth_ts[0]
        last_gap = depth_ts[-1] - depth_ts[-2]
        start_ts[0] = depth_ts[0] - first_gap
        end_ts[0] = depth_ts[0]
        start_ts[1:] = depth_ts[:-1]
        end_ts[1:] = depth_ts[1:]
        end_ts[-1] = depth_ts[-1] + last_gap

    start_ts = np.clip(start_ts, event_ts[0], event_ts[-1])
    end_ts = np.clip(end_ts, event_ts[0], event_ts[-1])
    mask = end_ts < start_ts
    if np.any(mask):
        end_ts[mask] = start_ts[mask]

    start_idx = np.searchsorted(event_ts, start_ts, side="left")
    end_idx = np.searchsorted(event_ts, end_ts, side="left")

    total_events = event_ts.shape[0]
    start_idx = np.clip(start_idx, 0, total_events)
    end_idx = np.clip(end_idx, 0, total_events)

    start_idx[0] = min(start_idx[0], end_idx[0])
    for i in range(1, len(start_idx)):
        start_idx[i] = max(start_idx[i], end_idx[i - 1])
        end_idx[i] = max(end_idx[i], start_idx[i])

    return start_idx, end_idx, start_ts, end_ts


class Hdf5FrameSource:
    def __init__(
        self,
        data_path: Path,
        gt_path: Path,
        camera: str,
    ) -> None:
        self.data_path = data_path
        self.gt_path = gt_path
        self.camera = camera

    def iter_samples(self, start: int | None, end: int | None) -> Iterator[FrameSample]:
        with h5py.File(self.data_path, "r") as data_fp, h5py.File(self.gt_path, "r") as gt_fp:
            ev_path = f"{self.camera}/events"
            depth_ts_path = f"{self.camera}/depth_image_rect_ts"
            depth_path = f"{self.camera}/depth_image_rect"

            if ev_path not in data_fp:
                raise KeyError(f"events dataset '{ev_path}' not found in {self.data_path}")
            if depth_path not in gt_fp:
                raise KeyError(f"depth dataset '{depth_path}' not found in {self.gt_path}")

            events_ds = data_fp[ev_path]
            depth_ds = gt_fp[depth_path]
            depth_ts = gt_fp[depth_ts_path][:]
            event_ts = events_ds[:, 2]

            start_idx, end_idx, window_start_ts, window_end_ts = _compute_event_bounds(
                depth_ts, event_ts
            )

            total = len(depth_ts)
            lo = max(start, 0) if start is not None else 0
            hi = total if end is None else min(end, total)

            for idx in range(lo, hi):
                depth = depth_ds[idx]
                events = events_ds[start_idx[idx]:end_idx[idx]]
                if events.size == 0:
                    h, w = depth.shape
                    x = np.empty(0, dtype=np.int32)
                    y = np.empty(0, dtype=np.int32)
                    t = np.empty(0, dtype=np.float32)
                    p = np.empty(0, dtype=bool)
                else:
                    x = events[:, 0].astype(np.int32)
                    y = events[:, 1].astype(np.int32)
                    t = events[:, 2].astype(np.float32)
                    p = events[:, 3] > 0

                yield FrameSample(
                    index=idx,
                    depth_ts=float(depth_ts[idx]),
                    depth=np.asarray(depth, dtype=np.float32),
                    events_x=x,
                    events_y=y,
                    events_t=t,
                    events_p=p,
                    event_start_time=float(window_start_ts[idx]),
                    event_end_time=float(window_end_ts[idx]),
                )


class RosbagEventStream:
    def __init__(self, bag_path: Path, topic: str) -> None:
        self.bag_path = bag_path
        self.topic = topic
        self.reader_ctx: Optional[AnyReader] = None
        self.reader: Optional[AnyReader] = None
        self.iterator: Optional[Iterator] = None
        self.chunks: Deque[Dict[str, np.ndarray]] = deque()
        self.buffer_end_time = float("-inf")
        self.exhausted = False

    def __enter__(self) -> "RosbagEventStream":
        self.reader_ctx = AnyReader([self.bag_path])
        self.reader = self.reader_ctx.__enter__()
        store = self.reader.typestore
        for conn in self.reader.connections:
            if conn.msgdef and conn.msgdef.data:
                store.register(get_types_from_msg(conn.msgdef.data, conn.msgtype))
        self.iterator = self.reader.messages()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.reader_ctx is not None:
            self.reader_ctx.__exit__(exc_type, exc, tb)
        self.reader_ctx = None
        self.reader = None
        self.iterator = None
        self.chunks.clear()
        self.buffer_end_time = float("-inf")
        self.exhausted = False

    def _append_chunk(self, ts: np.ndarray, xs: np.ndarray, ys: np.ndarray, polarities: np.ndarray) -> None:
        if ts.size == 0:
            return
        self.chunks.append(
            {
                "ts": ts,
                "x": xs,
                "y": ys,
                "p": polarities,
                "cursor": 0,
            }
        )
        if ts[-1] > self.buffer_end_time:
            self.buffer_end_time = float(ts[-1])

    def _fill_until(self, end_time: float) -> None:
        if self.iterator is None or self.reader is None:
            raise RuntimeError("Event stream not initialised")
        while not self.exhausted and self.buffer_end_time < end_time:
            try:
                conn, _, raw = next(self.iterator)
            except StopIteration:
                self.exhausted = True
                break
            if conn.topic != self.topic:
                continue
            msg = self.reader.deserialize(raw, conn.msgtype)
            if not msg.events:
                continue
            xs = np.fromiter((evt.x for evt in msg.events), dtype=np.int32, count=len(msg.events))
            ys = np.fromiter((evt.y for evt in msg.events), dtype=np.int32, count=len(msg.events))
            ts = np.fromiter(
                (evt.ts.sec + evt.ts.nanosec * 1e-9 for evt in msg.events),
                dtype=np.float64,
                count=len(msg.events),
            )
            ps = np.fromiter((bool(evt.polarity) for evt in msg.events), dtype=np.bool_, count=len(msg.events))
            self._append_chunk(ts, xs, ys, ps)

    def pop_range(self, start: float, end: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._fill_until(end)
        collected_x: List[np.ndarray] = []
        collected_y: List[np.ndarray] = []
        collected_t: List[np.ndarray] = []
        collected_p: List[np.ndarray] = []

        for chunk in list(self.chunks):
            ts = chunk["ts"]
            cursor = chunk["cursor"]
            if ts.size == 0:
                chunk["cursor"] = ts.size
                continue
            if ts[-1] < start:
                chunk["cursor"] = ts.size
                continue
            start_idx = max(cursor, int(np.searchsorted(ts, start, side="left")))
            end_idx = int(np.searchsorted(ts, end, side="left"))
            if end_idx > start_idx:
                collected_x.append(chunk["x"][start_idx:end_idx])
                collected_y.append(chunk["y"][start_idx:end_idx])
                collected_t.append(chunk["ts"][start_idx:end_idx])
                collected_p.append(chunk["p"][start_idx:end_idx])
            chunk["cursor"] = end_idx

        while self.chunks and self.chunks[0]["cursor"] >= self.chunks[0]["ts"].size:
            self.chunks.popleft()

        if collected_x:
            return (
                np.concatenate(collected_x),
                np.concatenate(collected_y),
                np.concatenate(collected_t),
                np.concatenate(collected_p),
            )
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=bool),
        )


def _build_tensor(
    events: np.ndarray,
    cfg: ProcessorConfig,
    frame_shape: Tuple[int, int],
) -> torch.Tensor:
    channels = cfg.nr_events_data * cfg.nr_bins_per_data
    tensor = np.zeros((channels, frame_shape[0], frame_shape[1]), dtype=np.float32)
    num_events = events.shape[0]
    if num_events < cfg.min_events:
        return torch.from_numpy(tensor)
    chunk_size = max(1, num_events // cfg.nr_events_data)
    for idx in range(cfg.nr_events_data):
        start = idx * chunk_size
        end = start + chunk_size
        if idx == cfg.nr_events_data - 1:
            end = num_events
        chunk = events[start:end]
        if chunk.shape[0] == 0:
            continue
        chunk = chunk.copy()
        t_min = chunk[:, 2].min()
        chunk[:, 2] -= t_min
        repr_slice = generate_input_representation(
            chunk,
            cfg.representation,
            frame_shape,
            nr_temporal_bins=cfg.nr_bins_per_data,
            separate_pol=cfg.separate_pol,
            normalize_event=cfg.normalize_event,
        )
        c_start = idx * cfg.nr_bins_per_data
        c_end = c_start + cfg.nr_bins_per_data
        tensor[c_start:c_end] = repr_slice
    return torch.from_numpy(tensor)


def _post_process(tensor: torch.Tensor, cfg: ProcessorConfig) -> torch.Tensor:
    if cfg.resize_shape is None:
        return tensor.contiguous()
    target_h, target_w = cfg.resize_shape
    interp_kwargs = {}
    if cfg.resize_mode in {"bilinear", "bicubic"}:
        interp_kwargs["align_corners"] = False
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(target_h, target_w),
        mode=cfg.resize_mode,
        **interp_kwargs,
    ).squeeze(0)
    return tensor.contiguous()


def _save_tensor(tensor: torch.Tensor, path: Path, fmt: str) -> None:
    if fmt == "pt":
        torch.save({"tensor": tensor}, path)
    elif fmt == "npz":
        np.savez_compressed(path, tensor=tensor.numpy())
    elif fmt == "npy":
        np.save(path, tensor.numpy())
    else:
        raise ValueError(f"Unsupported format '{fmt}'")


def _write_metadata(cfg: ProcessorConfig, out_dir: Path) -> None:
    meta = {
        "nr_events_data": cfg.nr_events_data,
        "nr_bins_per_data": cfg.nr_bins_per_data,
        "representation": cfg.representation,
        "separate_pol": cfg.separate_pol,
        "normalize_event": cfg.normalize_event,
        "resize_shape": cfg.resize_shape,
        "resize_mode": cfg.resize_mode,
        "output_format": cfg.output_format,
    }

    meta_path = out_dir / "ecddp_config.json"
    if meta_path.exists() and not cfg.overwrite:
        return
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


DATASETS: dict[str, dict[str, object]] = {
    "outdoor_day1": {
        "data_path": Path("/data/storage/jianwen/MVSEC/outdoor_day/outdoor_day1_data.hdf5"),
        "gt_path": Path("/data/storage/datasets/MVSEC/hdf5/outdoor_day/outdoor_day1_gt.hdf5"),
        "camera": "davis/left",
    },
    "outdoor_day2": {
        "data_path": Path("/data/storage/jianwen/MVSEC/outdoor_day/outdoor_day2_data.bag"),
        "gt_path": Path("/data/storage/jianwen/MVSEC/outdoor_day/outdoor_day2_gt.bag"),
        "camera": "davis/left",
    },
    "outdoor_night1": {
        "data_path": Path("/data/storage/jianwen/MVSEC/outdoor_night/outdoor_night1_data.hdf5"),
        "gt_path": Path("/data/storage/jianwen/MVSEC/outdoor_night/outdoor_night1_gt.hdf5"),
        "camera": "davis/left",
    },
    "outdoor_night2": {
        "data_path": Path("/data/storage/jianwen/MVSEC/outdoor_night/outdoor_night2_data.bag"),
        "gt_path": Path("/data/storage/jianwen/MVSEC/outdoor_night/outdoor_night2_gt.bag"),
        "camera": "davis/left",
    },
    "outdoor_night3": {
        "data_path": Path("/data/storage/jianwen/MVSEC/outdoor_night/outdoor_night3_data.bag"),
        "gt_path": Path("/data/storage/jianwen/MVSEC/outdoor_night/outdoor_night3_gt.bag"),
        "camera": "davis/left",
    },
}


def _iterate_frames(cfg_entry: dict, start: Optional[int], end: Optional[int]) -> Iterator[FrameSample]:
    data_path = Path(cfg_entry["data_path"])
    gt_path = Path(cfg_entry["gt_path"])
    camera = cfg_entry.get("camera", "davis/left")
    suffix = data_path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return Hdf5FrameSource(data_path, gt_path, camera).iter_samples(start, end)
    if suffix == ".bag":
        return RosbagFrameSource(data_path, gt_path, camera).iter_samples(start, end)
    raise ValueError(f"Unsupported input format: {data_path}")


class RosbagFrameSource:
    def __init__(self, data_path: Path, gt_path: Path, camera: str) -> None:
        self.data_path = data_path
        self.gt_path = gt_path
        self.camera = camera

    def iter_samples(self, start: int | None, end: int | None) -> Iterator[FrameSample]:
        topic_prefix = f"/{self.camera.strip('/')}"
        depth_topic = f"{topic_prefix}/depth_image_rect"

        with AnyReader([self.gt_path]) as depth_reader, RosbagEventStream(
            self.data_path, f"{topic_prefix}/events"
        ) as event_stream:
            store = depth_reader.typestore
            for conn in depth_reader.connections:
                if conn.msgdef and conn.msgdef.data:
                    store.register(get_types_from_msg(conn.msgdef.data, conn.msgtype))

            depth_iter = depth_reader.messages()
            prev_prev_entry: Optional[Dict[str, object]] = None
            prev_entry: Optional[Dict[str, object]] = None
            frame_index = -1
            lo = 0 if start is None else max(start, 0)
            hi = float("inf") if end is None else end

            for conn, ts_ns, raw in depth_iter:
                if conn.topic != depth_topic:
                    continue
                frame_index += 1
                msg = depth_reader.deserialize(raw, conn.msgtype)
                depth_arr = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
                ts = ts_ns * 1e-9
                entry = {
                    "index": frame_index,
                    "timestamp": float(ts),
                    "depth": depth_arr.copy(),
                }

                if prev_entry is not None:
                    if prev_prev_entry is None:
                        gap = max(entry["timestamp"] - prev_entry["timestamp"], 1e-6)
                        start_time = prev_entry["timestamp"] - gap
                    else:
                        start_time = prev_prev_entry["timestamp"]
                    end_time = prev_entry["timestamp"]
                    x, y, t, p = event_stream.pop_range(start_time, end_time)
                    if lo <= prev_entry["index"] < hi:
                        yield FrameSample(
                            index=prev_entry["index"],
                            depth_ts=prev_entry["timestamp"],
                            depth=prev_entry["depth"],
                            events_x=x.astype(np.int32),
                            events_y=y.astype(np.int32),
                            events_t=t.astype(np.float32),
                            events_p=p.astype(bool),
                            event_start_time=start_time,
                            event_end_time=end_time,
                        )
                prev_prev_entry = prev_entry
                prev_entry = entry

            if prev_entry is not None:
                if prev_prev_entry is not None:
                    gap = max(prev_entry["timestamp"] - prev_prev_entry["timestamp"], 1e-6)
                    start_time = prev_prev_entry["timestamp"]
                    end_time = prev_entry["timestamp"] + gap
                else:
                    start_time = prev_entry["timestamp"] - 0.025
                    end_time = prev_entry["timestamp"] + 0.025

                x, y, t, p = event_stream.pop_range(start_time, end_time)
                if lo <= prev_entry["index"] < hi:
                    yield FrameSample(
                        index=prev_entry["index"],
                        depth_ts=prev_entry["timestamp"],
                        depth=prev_entry["depth"],
                        events_x=x.astype(np.int32),
                        events_y=y.astype(np.int32),
                        events_t=t.astype(np.float32),
                        events_p=p.astype(bool),
                        event_start_time=start_time,
                        event_end_time=end_time,
                    )


def process_sequence(name: str, cfg_entry: dict, proc_cfg: ProcessorConfig) -> None:
    seq_root = proc_cfg.output_root / name
    out_dir = seq_root / proc_cfg.output_subdir
    _ensure_output_root(seq_root)
    if out_dir.exists() and proc_cfg.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(proc_cfg, out_dir)

    frame_iter = _iterate_frames(cfg_entry, cfg_entry.get("start"), cfg_entry.get("end"))

    num_written = 0
    for sample in tqdm(frame_iter, desc=f"{name}", unit="frame"):
        out_path = out_dir / f"{sample.index:06d}.{proc_cfg.output_format}"
        if out_path.exists() and not proc_cfg.overwrite:
            continue
        h, w = sample.depth.shape
        coords_valid = (
            (sample.events_x >= 0)
            & (sample.events_x < w)
            & (sample.events_y >= 0)
            & (sample.events_y < h)
        )
        if not np.any(coords_valid):
            events = np.zeros((0, 4), dtype=np.float32)
        else:
            events = np.stack(
                [
                    sample.events_x[coords_valid],
                    sample.events_y[coords_valid],
                    sample.events_t[coords_valid],
                    sample.events_p[coords_valid].astype(np.int8),
                ],
                axis=1,
            )
        tensor = _build_tensor(events, proc_cfg, (h, w))
        tensor = _post_process(tensor, proc_cfg)
        _save_tensor(tensor, out_path, proc_cfg.output_format)
        num_written += 1

    print(f"[{name}] wrote {num_written} tensors to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ECDDP tensors for MVSEC sequences.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/storage/jianwen/mvsec"),
        help="Directory containing per-sequence folders (default matches training config).",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="eventTensor_ecddp",
        help="Sub-directory under each sequence folder to store tensors.",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="*",
        default=list(DATASETS.keys()),
        help="Subset of MVSEC sequences to process.",
    )
    parser.add_argument("--nr-events-data", type=int, default=10, help="Temporal chunks per frame.")
    parser.add_argument("--nr-bins-per-data", type=int, default=2, help="Bins per chunk.")
    parser.add_argument("--min-events", type=int, default=5000, help="Discard frames with fewer events.")
    parser.add_argument(
        "--representation",
        choices=["histogram", "voxel_grid"],
        default="histogram",
        help="Event representation fed into ECDDP.",
    )
    parser.add_argument("--separate-pol", action="store_true", help="Split polarities into separate channels.")
    parser.add_argument("--normalize-event", action="store_true", help="Normalize channel magnitudes.")
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Optional resize target (skip to keep native resolution).",
    )
    parser.add_argument(
        "--resize-mode",
        choices=["nearest", "bilinear", "bicubic"],
        default="bilinear",
        help="Interpolation mode for resizing.",
    )
    parser.add_argument(
        "--format",
        choices=["pt", "npz", "npy"],
        default="pt",
        help="How tensors are persisted to disk.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove existing tensors before writing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = args.sequences or list(DATASETS.keys())
    missing = [seq for seq in sequences if seq not in DATASETS]
    if missing:
        raise ValueError(f"Unknown sequences: {missing}")

    resize_shape = tuple(args.resize) if args.resize is not None else None
    proc_cfg = ProcessorConfig(
        output_root=args.output_root,
        output_subdir=args.output_subdir,
        sequences=sequences,
        nr_events_data=args.nr_events_data,
        nr_bins_per_data=args.nr_bins_per_data,
        min_events=args.min_events,
        representation=args.representation,
        separate_pol=args.separate_pol,
        normalize_event=args.normalize_event,
        resize_shape=resize_shape,
        resize_mode=args.resize_mode,
        output_format=args.format,
        overwrite=args.overwrite,
    )

    for seq in sequences:
        process_sequence(seq, DATASETS[seq], proc_cfg)


if __name__ == "__main__":
    main()
