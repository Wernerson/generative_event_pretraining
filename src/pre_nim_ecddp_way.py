#!/usr/bin/env python3
"""
Pre-process N-ImageNet sequences into the event-tensor format (20 channels) expected by the ECDDP model.
This script adapts logic from src/pre_dse_ecddp.py but iterates over N-ImageNet structure found in src/pre_nim.py.
"""
from __future__ import annotations

import argparse
import sys
import os
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Ensure ECDDP data helpers are available
REPO_ROOT = Path(__file__).resolve().parents[1]
ECDDP_DATA_ROOT = REPO_ROOT / "Event-Camera-Data-Dense-Pre-training" / "data"
if not ECDDP_DATA_ROOT.exists():
    raise FileNotFoundError(
        f"Cannot locate ECDDP data helpers at {ECDDP_DATA_ROOT}. "
        "Please ensure Event-Camera-Data-Dense-Pre-training is available."
    )
if str(ECDDP_DATA_ROOT) not in sys.path:
    sys.path.append(str(ECDDP_DATA_ROOT))

from seg_utils import generate_input_representation, normalize_voxel_grid_numpy  # type: ignore

def _extract_xypt(data) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Robustly extract x,y,t,p from a structured event array.
    Copied from src/pre_nim.py
    """
    names = list(data.dtype.names or [])
    if not names:
        raise ValueError("event array has no named fields")
    name_map = {n.lower(): n for n in names}
    try:
        x = data[name_map.get('x', names[0])]
        y = data[name_map.get('y', names[1 if len(names) > 1 else 0])]
        t = data[name_map.get('t', names[2 if len(names) > 2 else 0])]
        p = data[name_map.get('p', names[3 if len(names) > 3 else 0])]
    except Exception:
        x = data[names[0]]
        y = data[names[1]] if len(names) > 1 else np.zeros_like(x)
        t = data[names[2]] if len(names) > 2 else np.zeros_like(x)
        p = data[names[3]] if len(names) > 3 else np.zeros_like(x)
    return x, y, t, p

def generate_event_histogram_fast(events, shape):
    """
    Optimized version of generate_event_histogram using np.bincount.
    """
    height, width = shape
    # events is (N, 4) -> x, y, t, p
    # avoiding full transpose if possible, but slicing columns is cheap
    x = events[:, 0].astype(np.int64)
    y = events[:, 1].astype(np.int64)
    p = events[:, 3]

    mask = (x < width) & (x >= 0) & (y < height) & (height >= 0)
    x = x[mask]
    y = y[mask]
    p = p[mask]

    # Handle polarity
    # In original: p[p == 0] = -1
    # We do logical indexing directly
    
    flat_indices = x + width * y
    min_len = height * width

    # Positive events (p == 1)
    mask_pos = (p == 1)
    img_pos = np.bincount(flat_indices[mask_pos], minlength=min_len).astype(np.float32)

    # Negative events (p == -1 or p == 0)
    # Original logic: p[p==0] = -1. So negative is p!=1
    mask_neg = (p != 1)
    img_neg = np.bincount(flat_indices[mask_neg], minlength=min_len).astype(np.float32)

    return np.stack([img_neg, img_pos], 0).reshape((2, height, width))

def _build_tensor(
    events: np.ndarray,
    event_shape: Tuple[int, int],
    nr_events_data: int,
    nr_bins_per_data: int,
    representation: str,
    separate_pol: bool,
    normalize_event: bool
) -> Optional[torch.Tensor]:
    """
    Builds a multi-channel tensor from events by splitting them into temporal chunks.
    Adapted from ECDDPPreprocessor._build_tensor in src/pre_dse_ecddp.py
    """
    num_events = events.shape[0]
    if num_events == 0:
        return None

    # Total channels = chunks * bins_per_chunk
    channels = nr_events_data * nr_bins_per_data
    tensor = np.zeros((channels, event_shape[0], event_shape[1]), dtype=np.float32)
    
    chunk_size = num_events // nr_events_data
    if chunk_size == 0:
        # Not enough events to split into requested chunks
        return None

    for idx in range(nr_events_data):
        start = idx * chunk_size
        end = start + chunk_size
        if idx == nr_events_data - 1:
            end = num_events
        chunk = events[start:end]
        
        if chunk.shape[0] == 0:
            continue
        
        if representation == "histogram":
            repr_slice = generate_event_histogram_fast(chunk, event_shape)
        else:
            repr_slice = generate_input_representation(
                chunk,
                representation,
                event_shape,
                nr_temporal_bins=nr_bins_per_data,
                separate_pol=separate_pol,
                normalize_event=normalize_event,
            )
        
        # seg_utils.generate_event_histogram does not normalize even if normalize_event=True
        # So we apply it manually here if requested.
        if representation == "histogram" and normalize_event:
            repr_slice = normalize_voxel_grid_numpy(repr_slice)

        c_start = idx * nr_bins_per_data
        c_end = c_start + nr_bins_per_data
        tensor[c_start:c_end] = repr_slice

    return torch.from_numpy(tensor)

def _post_process(
    tensor: torch.Tensor,
    resize_shape: Optional[Tuple[int, int]],
    resize_mode: str
) -> torch.Tensor:
    if resize_shape is not None:
        target_h, target_w = resize_shape
        interp_kwargs = {}
        if resize_mode in {"bilinear", "bicubic"}:
            interp_kwargs["align_corners"] = False
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(target_h, target_w),
            mode=resize_mode,
            **interp_kwargs,
        ).squeeze(0)
    return tensor.contiguous()

def list_event_files(root: str) -> List[str]:
    out = []
    if not os.path.isdir(root):
        return out
    for cls in sorted(os.listdir(root)):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.endswith('.npz'):
                out.append(os.path.join(cls_dir, fname))
    return out

import traceback

def process_one_file(
    event_path: str,
    args
) -> Tuple[str, bool, str]:
    try:
        # Determine output path
        event_path_obj = Path(event_path)
        if args.output_subdir:
            # If output_subdir is specified, create a mirror structure or subfolder
            # Here we follow simple logic: if subdir is relative, put it inside the class folder?
            # Or just save alongside with specific extension.
            # To match pre_dse_ecddp.py: images/left/eventTensor_ecddp
            # But N-ImageNet is extracted_train/class/xxx.npz
            # Let's save as extracted_train/class/xxx.pt if no subdir, or extracted_train/class/subdir/xxx.pt?
            # User request: "NImageNet dataset also into 20 channels".
            # Simplest is same folder, different extension.
            out_path = event_path_obj.with_suffix(f'.{args.output_ext}')
        else:
             out_path = event_path_obj.with_suffix(f'.{args.output_ext}')

        if out_path.exists() and not args.overwrite:
            return (event_path, True, 'skip_exists')

        # Load events
        with np.load(event_path) as npz:
            if 'event_data' in npz:
                data = npz['event_data']
            else:
                # Fallback for some npz structures
                data = list(npz.values())[0]
        
        x, y, t, p = _extract_xypt(data)
        
        # ECDDP expects events as (x, y, t, p) array
        # Note: x, y, p are int/float.
        # generate_input_representation expects structured array or specific columns?
        # pre_dse_ecddp.py: events = np.stack([x_rect, y_rect, t, p], axis=1)
        # So it passes a (N, 4) float array.
        
        # Valid mask (N-ImageNet should be clean but good to check)
        h_sensor, w_sensor = args.sensor_size
        valid = (x >= 0) & (x < w_sensor) & (y >= 0) & (y < h_sensor)
        if not np.any(valid):
            return (event_path, False, 'no_valid_events')
        
        x = x[valid].astype(np.float32)
        y = y[valid].astype(np.float32)
        t = t[valid].astype(np.float32)
        p = p[valid].astype(np.float32)
        
        events_stacked = np.stack([x, y, t, p], axis=1)

        # Build tensor
        tensor = _build_tensor(
            events_stacked,
            (h_sensor, w_sensor),
            args.nr_events_data,
            args.nr_bins_per_data,
            args.representation,
            args.separate_pol,
            args.normalize_event
        )

        if tensor is None:
             return (event_path, False, 'tensor_build_failed')

        # Post process (Resize)
        if args.resize:
             tensor = _post_process(tensor, tuple(args.resize), args.resize_mode)

        # Save
        if args.output_format == 'pt':
             torch.save(tensor, out_path)
        elif args.output_format == 'npz':
             np.savez_compressed(out_path, tensor=tensor.numpy())
        elif args.output_format == 'npy':
             np.save(out_path, tensor.numpy())
        
        return (event_path, True, 'ok')

    except Exception as e:
        traceback.print_exc()
        return (event_path, False, f'err:{e.__class__.__name__}')

def main():
    parser = argparse.ArgumentParser(description="Generate 20-channel event tensors for N-ImageNet.")
    
    # Dataset paths
    parser.add_argument('--train_root', type=str, default='/data/storage/jianwen/N_ImageNet/extracted_train')
    parser.add_argument('--valid_root', type=str, default='/data/storage/jianwen/N_ImageNet/extracted_val')
    
    # Processing Config
    parser.add_argument("--nr-events-data", type=int, default=10, help="Number of temporal chunks per sample.")
    parser.add_argument("--nr-bins-per-data", type=int, default=2, help="Number of channels produced per temporal chunk (e.g., histogram => 2).")
    parser.add_argument("--representation", choices=["histogram", "voxel_grid"], default="histogram", help="Event representation.")
    parser.add_argument("--separate-pol", action="store_true", help="Keep positive/negative events in separate bins.")
    parser.add_argument("--normalize-event", action="store_true", default=True, help="Apply per-bin normalization (default: True).")
    
    # Spatial Config
    parser.add_argument("--sensor-size", type=int, nargs=2, default=(480, 640), metavar=("H", "W"), help="Sensor resolution of events.")
    parser.add_argument("--resize", type=int, nargs=2, default=(224, 224), metavar=("H", "W"), help="Output spatial size (default 224x224 for ViT).")
    parser.add_argument("--resize-mode", choices=["nearest", "bilinear", "bicubic"], default="bilinear")
    
    # Output Config
    parser.add_argument("--output-subdir", type=str, default=None, help="Not used currently (saves alongside).")
    parser.add_argument("--output-ext", type=str, default="pt", help="Extension for output file (pt, npz).")
    parser.add_argument("--output-format", choices=["pt", "npz", "npy"], default="pt")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    
    # Execution
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--val-only", action="store_true")
    
    args = parser.parse_args()
    
    # Collect files
    files = []
    if args.val_only:
        files.extend(list_event_files(args.valid_root))
    else:
        files.extend(list_event_files(args.train_root))
        files.extend(list_event_files(args.valid_root))
        
    print(f"Found {len(files)} event files.")
    if not files:
        return

    # Process
    worker = partial(process_one_file, args=args)
    
    ok = 0
    err = 0
    skip = 0
    
    if args.num_workers <= 1:
        for f in tqdm(files):
            _, success, status = worker(f)
            if success:
                if status == 'skip_exists': skip += 1
                else: ok += 1
            else:
                err += 1
    else:
        with mp.Pool(args.num_workers) as pool:
            for _, success, status in tqdm(pool.imap_unordered(worker, files, chunksize=1), total=len(files)):
                if success:
                    if status == 'skip_exists': skip += 1
                    else: ok += 1
                else:
                    err += 1
                    
    print(f"Done. OK: {ok}, Skip: {skip}, Err: {err}")

if __name__ == "__main__":
    main()
