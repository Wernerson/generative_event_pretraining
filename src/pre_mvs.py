from __future__ import annotations

# NOTE:
#   This file now contains a streamlined MVSEC pre-processing pipeline tailored
#   for the five canonical sequences (outdoor_day1/2, outdoor_night1/2/3).
#   Experimental ROS utilities are commented out below; the active code path
#   relies exclusively on the existing HDF5 data that was validated earlier.

import csv
import shutil
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterator, List, Optional, Tuple

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import accumulate_to_rgb

from rosbags.highlevel import AnyReader
from rosbags.typesys import get_types_from_msg


@dataclass
class FrameSample:
    index: int
    depth_ts: float
    depth: np.ndarray
    image: np.ndarray
    image_index: int
    image_ts: float
    events_x: np.ndarray
    events_y: np.ndarray
    events_p: np.ndarray
    event_start_time: float
    event_end_time: float


def _print_structure(h5_path: Path) -> None:
    """Recursively print groups and datasets inside an HDF5 file."""
    with h5py.File(h5_path, "r") as fp:
        print(f"\n=== HDF5 file: {h5_path} ===\n")

        def _printer(name: str, obj):
            if isinstance(obj, h5py.Group):
                print(f"[Group] {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"[Dataset] {name} shape={obj.shape}, dtype={obj.dtype}")

        fp.visititems(_printer)


def _ensure_output_root(out_dir: Path) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - surface real filesystem error
        raise RuntimeError(f"failed to create output directory {out_dir}") from exc

    allowed_root = Path("/data/storage/jianwen").resolve()
    resolved = out_dir.resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(
            f"Output directory must be inside {allowed_root}, "
            f"got {resolved}"
        ) from exc


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
        # Only one depth frame: consume the entire event stream.
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
        if start_idx[i] < end_idx[i - 1]:
            start_idx[i] = end_idx[i - 1]
        if end_idx[i] < start_idx[i]:
            end_idx[i] = start_idx[i]

    return start_idx, end_idx, start_ts, end_ts


def _depth_to_uint16(depth: np.ndarray, scale: float) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth_scaled = np.clip(depth * scale, 0, np.iinfo(np.uint16).max)
    return depth_scaled.astype(np.uint16)


def _accumulate_events(
    events_slice: np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray],
    frame_shape: Tuple[int, int],
    percentile: float,
) -> np.ndarray:
    if isinstance(events_slice, tuple):
        x_arr, y_arr, p_arr = events_slice
    else:
        events_arr = np.asarray(events_slice)
        if events_arr.size:
            x_arr = events_arr[:, 0].astype(np.int32)
            y_arr = events_arr[:, 1].astype(np.int32)
            p_arr = events_arr[:, 3] > 0
        else:
            x_arr = y_arr = np.empty((0,), dtype=np.int32)
            p_arr = np.empty((0,), dtype=bool)
    if x_arr.size == 0:
        h, w = frame_shape
        return np.full((h, w, 3), 255, dtype=np.uint8)
    h, w = frame_shape
    x = x_arr.astype(np.int32)
    y = y_arr.astype(np.int32)
    p = p_arr.astype(bool)

    in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    x = x[in_bounds]
    y = y[in_bounds]
    p = p[in_bounds]

    return accumulate_to_rgb(x, y, p, frame_shape, pct=percentile)


def _match_image_indices(depth_ts: np.ndarray, image_ts: np.ndarray) -> np.ndarray:
    image_ts = np.asarray(image_ts, dtype=np.float64)
    depth_ts = np.asarray(depth_ts, dtype=np.float64)
    if image_ts.ndim != 1:
        raise ValueError("image timestamps must be 1-D")
    if not np.all(np.diff(image_ts) >= 0):
        raise ValueError("image timestamps must be non-decreasing")

    idxs = np.searchsorted(image_ts, depth_ts, side="left")
    max_idx = len(image_ts) - 1
    for i, idx in enumerate(idxs):
        if idx == 0:
            continue
        if idx > max_idx:
            idxs[i] = max_idx
            continue
        prev_idx = idx - 1
        after = image_ts[idx]
        before = image_ts[prev_idx]
        if depth_ts[i] - before <= after - depth_ts[i]:
            idxs[i] = prev_idx
    idxs = np.clip(idxs, 0, max_idx)
    return idxs.astype(np.int64)


class Hdf5FrameSource:
    def __init__(
        self,
        data_path: Path,
        gt_path: Path,
        camera: str,
        use_blended: bool,
    ) -> None:
        self.data_path = data_path
        self.gt_path = gt_path
        self.camera = camera
        self.use_blended = use_blended

    def iter_samples(self, start: int | None, end: int | None) -> Iterator[FrameSample]:
        with h5py.File(self.data_path, "r") as data_fp, h5py.File(self.gt_path, "r") as gt_fp:
            ev_path = f"{self.camera}/events"
            depth_ts_path = f"{self.camera}/depth_image_rect_ts"
            depth_path = f"{self.camera}/depth_image_rect"
            img_ts_path = f"{self.camera}/image_raw_ts"
            raw_img_path = f"{self.camera}/image_raw"

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

            if self.use_blended:
                blended_path = f"{self.camera}/blended_image_rect"
                if blended_path not in gt_fp:
                    raise KeyError(f"blended dataset '{blended_path}' not found in {self.gt_path}")
                image_ds = gt_fp[blended_path]
                image_ts_used = depth_ts
                image_indices = np.arange(len(depth_ts), dtype=np.int64)
            else:
                if raw_img_path not in data_fp:
                    raise KeyError(f"raw image dataset '{raw_img_path}' not found in {self.data_path}")
                image_ds = data_fp[raw_img_path]
                image_ts = data_fp[img_ts_path][:]
                image_indices = _match_image_indices(depth_ts, image_ts)
                image_ts_used = image_ts[image_indices]

            total = len(depth_ts)
            lo = max(start, 0) if start is not None else 0
            hi = total if end is None else min(end, total)

            for idx in range(lo, hi):
                depth = depth_ds[idx]
                image = image_ds[image_indices[idx]]
                events = events_ds[start_idx[idx]:end_idx[idx]]
                if events.size:
                    x = events[:, 0].astype(np.int32)
                    y = events[:, 1].astype(np.int32)
                    p = events[:, 3] > 0
                else:
                    h, w = depth.shape
                    x = np.empty((0,), dtype=np.int32)
                    y = np.empty((0,), dtype=np.int32)
                    p = np.empty((0,), dtype=bool)

                yield FrameSample(
                    index=idx,
                    depth_ts=float(depth_ts[idx]),
                    depth=np.asarray(depth, dtype=np.float32),
                    image=np.asarray(image),
                    image_index=int(image_indices[idx]),
                    image_ts=float(image_ts_used[idx]),
                    events_x=x,
                    events_y=y,
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

    def _append_chunk(
        self, ts: np.ndarray, xs: np.ndarray, ys: np.ndarray, polarities: np.ndarray
    ) -> None:
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

    def pop_range(self, start: float, end: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._fill_until(end)

        collected_x: List[np.ndarray] = []
        collected_y: List[np.ndarray] = []
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
                collected_p.append(chunk["p"][start_idx:end_idx])
            chunk["cursor"] = end_idx

        while self.chunks and self.chunks[0]["cursor"] >= self.chunks[0]["ts"].size:
            self.chunks.popleft()

        if collected_x:
            return (
                np.concatenate(collected_x),
                np.concatenate(collected_y),
                np.concatenate(collected_p),
            )
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=bool),
        )

    def peek_last_timestamp(self) -> float:
        if self.chunks:
            return float(self.chunks[-1]["ts"][-1])
        return float("-inf")


class RosbagImageStream:
    def __init__(self, bag_path: Path, topic: str) -> None:
        self.bag_path = bag_path
        self.topic = topic
        self.reader_ctx: Optional[AnyReader] = None
        self.reader: Optional[AnyReader] = None
        self.iterator: Optional[Iterator] = None
        self.prev_entry: Optional[Dict[str, object]] = None
        self.next_entry: Optional[Dict[str, object]] = None
        self.index_counter = -1
        self.exhausted = False

    def __enter__(self) -> "RosbagImageStream":
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
        self.prev_entry = None
        self.next_entry = None
        self.index_counter = -1
        self.exhausted = False

    def _decode_image(self, msg) -> np.ndarray:
        height = msg.height
        width = msg.width
        encoding = msg.encoding.lower()
        data = memoryview(msg.data)
        if encoding in ("mono8", "8uc1"):
            arr = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
        elif encoding in ("mono16", "16uc1"):
            arr = np.frombuffer(data, dtype=np.uint16).reshape((height, width))
        elif encoding in ("rgb8", "bgr8"):
            arr = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
            if encoding == "bgr8":
                arr = arr[..., ::-1]
        else:
            raise ValueError(f"Unsupported image encoding '{msg.encoding}'")
        return np.asarray(arr)

    def _fetch_next(self) -> Optional[Dict[str, object]]:
        if self.iterator is None or self.reader is None:
            raise RuntimeError("Image stream not initialised")
        for conn, _, raw in self.iterator:
            if conn.topic != self.topic:
                continue
            msg = self.reader.deserialize(raw, conn.msgtype)
            self.index_counter += 1
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            return {
                "index": self.index_counter,
                "timestamp": float(timestamp),
                "image": self._decode_image(msg),
            }
        self.exhausted = True
        return None

    def _ensure_next(self) -> None:
        if self.next_entry is None and not self.exhausted:
            self.next_entry = self._fetch_next()

    def get_nearest(self, target_time: float) -> Tuple[int, float, np.ndarray]:
        self._ensure_next()
        while self.next_entry and self.next_entry["timestamp"] < target_time and not self.exhausted:
            self.prev_entry = self.next_entry
            self.next_entry = self._fetch_next()

        candidates: List[Dict[str, object]] = []
        if self.prev_entry is not None:
            candidates.append(self.prev_entry)
        if self.next_entry is not None:
            candidates.append(self.next_entry)
        if not candidates:
            raise RuntimeError("No images available for selection")

        best = min(candidates, key=lambda entry: abs(entry["timestamp"] - target_time))
        if self.prev_entry is None or best["timestamp"] >= target_time:
            self.prev_entry = best
        return int(best["index"]), float(best["timestamp"]), np.asarray(best["image"])


class RosbagFrameSource:
    def __init__(
        self,
        data_path: Path,
        gt_path: Path,
        camera: str,
        use_blended: bool,
    ) -> None:
        if AnyReader is None or get_types_from_msg is None:
            raise ImportError("Package 'rosbags' is required to process ROS bag datasets.")
        self.data_path = data_path
        self.gt_path = gt_path
        self.camera = camera
        self.use_blended = use_blended

    def iter_samples(self, start: int | None, end: int | None) -> Iterator[FrameSample]:
        topic_prefix = f"/{self.camera.strip('/')}"
        depth_topic = f"{topic_prefix}/depth_image_rect"
        if self.use_blended:
            raise NotImplementedError("Blended images are not supported for ROS bag sources.")

        image_topic = f"{topic_prefix}/image_raw"

        with AnyReader([self.gt_path]) as depth_reader, RosbagEventStream(
            self.data_path, f"{topic_prefix}/events"
        ) as event_stream, RosbagImageStream(self.data_path, image_topic) as image_stream:
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
                    x, y, p = event_stream.pop_range(start_time, end_time)
                    if lo <= prev_entry["index"] < hi:
                        image_idx, image_ts, image_arr = image_stream.get_nearest(prev_entry["timestamp"])
                        yield FrameSample(
                            index=prev_entry["index"],
                            depth_ts=prev_entry["timestamp"],
                            depth=prev_entry["depth"],
                            image=image_arr,
                            image_index=image_idx,
                            image_ts=image_ts,
                            events_x=x,
                            events_y=y,
                            events_p=p,
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

                x, y, p = event_stream.pop_range(start_time, end_time)
                if lo <= prev_entry["index"] < hi:
                    image_idx, image_ts, image_arr = image_stream.get_nearest(prev_entry["timestamp"])
                    yield FrameSample(
                        index=prev_entry["index"],
                        depth_ts=prev_entry["timestamp"],
                        depth=prev_entry["depth"],
                        image=image_arr,
                        image_index=image_idx,
                        image_ts=image_ts,
                        events_x=x,
                        events_y=y,
                        events_p=p,
                        event_start_time=start_time,
                        event_end_time=end_time,
                    )


def convert_sequence(
    data_path: Path,
    gt_path: Path,
    out_dir: Path,
    *,
    camera: str = "davis/left",
    percentile: float = 99.0,
    depth_scale: float = 100.0,
    start: int | None = None,
    end: int | None = None,
    use_blended: bool = False,
) -> None:
    _ensure_output_root(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    rgb_dir = out_dir / "rgb"
    event_dir = out_dir / "events"
    depth_dir = out_dir / "depth"
    for path in (rgb_dir, event_dir, depth_dir):
        path.mkdir(parents=True, exist_ok=True)

    data_path = Path(data_path)
    gt_path = Path(gt_path)

    suffix = data_path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        source: Iterator[FrameSample] = Hdf5FrameSource(
            data_path, gt_path, camera, use_blended
        ).iter_samples(start, end)
    elif suffix == ".bag":
        source = RosbagFrameSource(
            data_path, gt_path, camera, use_blended=False
        ).iter_samples(start, end)
    else:
        raise ValueError(f"Unsupported data file format: {data_path}")

    records: list[tuple] = []
    for sample in tqdm(source, desc="Converting MVSEC", unit="frame"):
        frame_shape = sample.depth.shape
        event_img = _accumulate_events(
            (sample.events_x, sample.events_y, sample.events_p), frame_shape, percentile
        )

        image_arr = sample.image
        if image_arr.ndim == 2:
            gray_np = image_arr.astype(np.uint8)
            gray_pil = Image.fromarray(gray_np, mode="L")
        elif image_arr.ndim == 3 and image_arr.shape[-1] in {3, 4}:
            rgb = image_arr[..., :3].astype(np.uint8)
            gray_np = np.asarray(Image.fromarray(rgb).convert("L"))
            gray_pil = Image.fromarray(gray_np, mode="L")
        else:
            raise ValueError(f"Unsupported image shape {image_arr.shape} for frame {sample.index}")

        depth_uint16 = _depth_to_uint16(sample.depth, depth_scale)
        depth_img = Image.fromarray(depth_uint16, mode="I;16")

        stem = f"{sample.index:06d}"
        rgb_path_out = rgb_dir / f"{stem}.png"
        event_path_out = event_dir / f"{stem}.png"
        depth_path_out = depth_dir / f"{stem}.png"

        gray_pil.save(rgb_path_out)
        Image.fromarray(event_img).save(event_path_out)
        depth_img.save(depth_path_out)

        records.append(
            (
                sample.index,
                sample.depth_ts,
                sample.image_index,
                sample.image_ts,
                sample.event_start_time,
                sample.event_end_time,
                int(sample.events_x.size),
                rgb_path_out.relative_to(out_dir),
                event_path_out.relative_to(out_dir),
                depth_path_out.relative_to(out_dir),
            )
        )

    if records:
        index_path = out_dir / "index.csv"
        header = [
            "frame_id",
            "depth_timestamp",
            "image_index",
            "image_timestamp",
            "event_start_time",
            "event_end_time",
            "event_count",
            "rgb_path",
            "event_path",
            "depth_path",
        ]
        with index_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(records)


OUTPUT_ROOT = Path("/data/storage/jianwen/mvsec")

DATASETS: dict[str, dict[str, object]] = {
    "outdoor_day1": {
        "data_path": Path("/data/storage/jianwen/MVSEC/outdoor_day/outdoor_day1_data.hdf5"),
        "gt_path": Path("/data/storage/datasets/MVSEC/hdf5/outdoor_day/outdoor_day1_gt.hdf5"),
        "camera": "davis/left",
        "use_blended": False,
        "frame_slice": None,
    },
    "outdoor_day2": {
        "data_path": Path("/data/storage/jianwen/MVSEC/outdoor_day/outdoor_day2_data.bag"),
        "gt_path": Path("/data/storage/jianwen/MVSEC/outdoor_day/outdoor_day2_gt.bag"),
        "camera": "davis/left",
        "use_blended": False,
        "frame_slice": None,
    },
    "outdoor_night1": {
        "data_path": Path("/data/storage/jianwen/MVSEC/outdoor_night/outdoor_night1_data.hdf5"),
        "gt_path": Path("/data/storage/jianwen/MVSEC/outdoor_night/outdoor_night1_gt.hdf5"),
        "camera": "davis/left",
        "use_blended": False,
        "frame_slice": None,
    },
    "outdoor_night2": {
        "data_path": Path("/data/storage/jianwen/MVSEC/outdoor_night/outdoor_night2_data.bag"),
        "gt_path": Path("/data/storage/jianwen/MVSEC/outdoor_night/outdoor_night2_gt.bag"),
        "camera": "davis/left",
        "use_blended": False,
        "frame_slice": None,
    },
    "outdoor_night3": {
        "data_path": Path("/data/storage/jianwen/MVSEC/outdoor_night/outdoor_night3_data.bag"),
        "gt_path": Path("/data/storage/jianwen/MVSEC/outdoor_night/outdoor_night3_gt.bag"),
        "camera": "davis/left",
        "use_blended": False,
        "frame_slice": None,
    },
}

# Update this list before calling main() to choose which sequences to export.
RUN_DATASETS: list[str] = [
    "outdoor_day1",
    "outdoor_day2",
    "outdoor_night1",
    "outdoor_night2",
    "outdoor_night3",
]


def main() -> None:
    if not RUN_DATASETS:
        raise ValueError("RUN_DATASETS is empty; please specify at least one dataset name.")

    available = set(DATASETS)
    for name in RUN_DATASETS:
        if name not in available:
            raise ValueError(
                f"Unknown dataset '{name}'. Available options: {sorted(available)}"
            )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for name in RUN_DATASETS:
        cfg = DATASETS[name]
        slice_cfg = cfg.get("frame_slice")
        if slice_cfg is None:
            start = end = None
        else:
            start, end = slice_cfg
        out_dir = OUTPUT_ROOT / name
        print(f"[MVSEC] Processing {name} -> {out_dir}")
        convert_sequence(
            data_path=cfg["data_path"],
            gt_path=cfg["gt_path"],
            out_dir=out_dir,
            camera=cfg.get("camera", "davis/left"),
            percentile=cfg.get("percentile", 95.0),
            depth_scale=cfg.get("depth_scale", 100.0),
            start=start,
            end=end,
            use_blended=cfg.get("use_blended", False),
        )


if __name__ == "__main__":
    main()
