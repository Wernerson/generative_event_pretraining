#!/usr/bin/env python3
"""
Pre-process DSEC sequences into the event-tensor format (20 channels) expected by the ECDDP model.
Adapts DSEC loading from src/pre_dse.py and event tensor building from src/pre_nim_ecddp_way.py.
Refactored to process folders sequentially but parallelize image processing internally.
"""
from __future__ import annotations

import argparse
import sys
import os
import shutil
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import traceback

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

# Ensure ECDDP data helpers are available
REPO_ROOT = Path(__file__).resolve().parents[1]
ECDDP_DATA_ROOT = REPO_ROOT / "Event-Camera-Data-Dense-Pre-training" / "data"
if not ECDDP_DATA_ROOT.exists():
    pass
if str(ECDDP_DATA_ROOT) not in sys.path:
    sys.path.append(str(ECDDP_DATA_ROOT))

try:
    from seg_utils import generate_input_representation, normalize_voxel_grid_numpy  # type: ignore
except ImportError:
    pass

from utils import EventSlicer

# --- Global variables for worker processes ---
_global_x = None
_global_y = None
_global_t = None
_global_p = None

def _worker_init(x, y, t, p):
    """Initialize global shared arrays in worker process."""
    global _global_x, _global_y, _global_t, _global_p
    _global_x = x
    _global_y = y
    _global_t = t
    _global_p = p

def generate_event_histogram_fast(events, shape):
    """
    Optimized version of generate_event_histogram using np.bincount.
    """
    height, width = shape
    x = events[:, 0].astype(np.int64)
    y = events[:, 1].astype(np.int64)
    p = events[:, 3]

    mask = (x < width) & (x >= 0) & (y < height) & (height >= 0)
    x = x[mask]
    y = y[mask]
    p = p[mask]

    flat_indices = x + width * y
    min_len = height * width

    mask_pos = (p == 1)
    img_pos = np.bincount(flat_indices[mask_pos], minlength=min_len).astype(np.float32)

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
    num_events = events.shape[0]
    if num_events == 0:
        return None

    channels = nr_events_data * nr_bins_per_data
    tensor = np.zeros((channels, event_shape[0], event_shape[1]), dtype=np.float32)
    
    chunk_size = num_events // nr_events_data
    if chunk_size == 0:
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

def _get_camera(calib_dir) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def create_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T
    def invert_transform(T: np.ndarray) -> np.ndarray:
        R = T[:3, :3]
        t = T[:3, 3]
        R_inv = np.linalg.inv(R)
        t_inv = - R_inv @ t
        return create_transform(R_inv, t_inv)
    def cretate_K(intrinsic):
        fx_e, fy_e, cx_e, cy_e = intrinsic
        return np.array([[fx_e,    0,  cx_e],
                        [   0,  fy_e, cy_e],
                        [   0,     0,    1]])
    with open(calib_dir, 'r') as file:
        data = yaml.safe_load(file)
    
    intrin_event_dist = np.array(data["intrinsics"]["cam0"]["camera_matrix"])
    dist_coeffs = np.array(data["intrinsics"]["cam0"]["distortion_coeffs"])
    intrin_event_rect = np.array(data["intrinsics"]["camRect0"]["camera_matrix"])
    resolution = np.array(data["intrinsics"]["camRect0"]["resolution"])
    intrin_image = np.array(data["intrinsics"]["camRect1"]["camera_matrix"])
    T_i2e = np.array(data["extrinsics"]["T_10"])
    Re = np.array(data['extrinsics']['R_rect0'])
    Ri = np.array(data['extrinsics']['R_rect1'])
    Te = create_transform(Re, np.zeros(3))
    Ti = create_transform(Ri, np.zeros(3))

    K_event = cretate_K(intrin_event_rect)
    K_dist  = cretate_K(intrin_event_dist)
    K_image = cretate_K(intrin_image)
    
    T_i2e = Ti @ T_i2e @ invert_transform(Te)
    R_i2e = T_i2e[:3, :3]

    H_homography = K_image @ R_i2e @ np.linalg.inv(K_event)
    H_homography = np.linalg.inv(H_homography)

    return H_homography, K_event, K_dist, dist_coeffs, resolution, Re

def _process_image_task(args_tuple):
    """
    Worker function to process a single image event chunk.
    """
    (indices_list, image_ts, event_shape, nr_events_data, nr_bins, 
     representation, separate_pol, normalize_event, resize, resize_mode, out_dir) = args_tuple

    ev_x_list = []
    ev_y_list = []
    ev_t_list = []
    ev_p_list = []
    
    for (start, end) in indices_list:
        ev_x_list.append(_global_x[start:end])
        ev_y_list.append(_global_y[start:end])
        ev_t_list.append(_global_t[start:end])
        ev_p_list.append(_global_p[start:end])
        
    if not ev_x_list:
        return
        
    ev_x = np.concatenate(ev_x_list)
    ev_y = np.concatenate(ev_y_list)
    ev_t = np.concatenate(ev_t_list)
    ev_p = np.concatenate(ev_p_list)
    
    if len(ev_x) == 0:
        return

    events_stacked = np.stack([
        ev_x.astype(np.float32), 
        ev_y.astype(np.float32), 
        ev_t.astype(np.float32), 
        ev_p.astype(np.float32)
    ], axis=1)

    try:
        tensor = _build_tensor(
            events_stacked,
            event_shape,
            nr_events_data,
            nr_bins,
            representation,
            separate_pol,
            normalize_event
        )

        if tensor is not None:
            if resize:
                tensor = _post_process(tensor, tuple(resize), resize_mode)
            
            out_path = os.path.join(out_dir, f"{image_ts}.pt")
            torch.save(tensor, out_path)
    except Exception as e:
        # print(f"Error processing {image_ts}: {e}")
        # traceback.print_exc()
        pass

def process_one_folder(
    subfolder: str,
    args,
    force_overwrite: bool = False
) -> Tuple[str, bool, str]:
    try:
        # Construct paths
        events_root = os.path.join(args.root, f"{args.split}_events")
        image_root = os.path.join(args.root, f"{args.split}_images")
        calib_root = os.path.join(args.root, f"{args.split}_calibration")

        event_dir = os.path.join(events_root, subfolder, "events", "left", "events.h5")
        image_timestamp_dir = os.path.join(image_root, subfolder, "images", "timestamps.txt")
        calib_dir = os.path.join(calib_root, subfolder, "calibration", "cam_to_cam.yaml")
        
        save_root = os.path.join(image_root, subfolder, "images", "left")
        out_dir = os.path.join(save_root, "eventTensor_ecddp_way")
        
        if not args.overwrite and not force_overwrite and os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
            return (subfolder, True, 'skip_exists')
        
        if force_overwrite and os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(image_timestamp_dir):
            return (subfolder, False, 'no_timestamps')
        image_timestamp = np.loadtxt(image_timestamp_dir, dtype=np.int64)

        if not os.path.exists(event_dir):
             return (subfolder, False, 'no_events_h5')

        _, _, _, _, resolution, _ = _get_camera(calib_dir)
        W_sensor, H_sensor = resolution 
        
        with h5py.File(event_dir, "r") as f_event:
            event_slicer = EventSlicer(f_event)
            start_timestamp = event_slicer.get_start_time_us()
            end_timestamp = event_slicer.get_final_time_us()
            
            event_dict = event_slicer.get_events(start_timestamp, end_timestamp)
            if event_dict is None:
                 return (subfolder, False, 'event_load_failed')
            
            t_all = event_dict['t']
            x_all = event_dict['x']
            y_all = event_dict['y']
            p_all = event_dict['p']
            
            unique_moments, unique_index = np.unique(t_all, return_index=True)
            
            # --- 1. Sequential Scan to define tasks ---
            tasks = []
            
            j = 0 # image index
            current_accumulated_indices = [] # list of (start, end)
            
            num_images = len(image_timestamp)
            num_unique = len(unique_index)
            
            for i in range(num_unique - 1):
                if j >= num_images:
                    break
                
                idx_start = unique_index[i]
                idx_end = unique_index[i+1]
                
                current_accumulated_indices.append((idx_start, idx_end))
                
                curr_img_ts = image_timestamp[j]
                next_img_ts = image_timestamp[j+1] if (j+1) < num_images else curr_img_ts
                middle_ts = (curr_img_ts + next_img_ts) / 2
                
                curr_ev_ts = t_all[idx_start]
                
                if curr_ev_ts > middle_ts:
                    if current_accumulated_indices:
                        task_args = (
                            current_accumulated_indices,
                            curr_img_ts,
                            (H_sensor, W_sensor),
                            args.nr_events_data,
                            args.nr_bins_per_data,
                            args.representation,
                            args.separate_pol,
                            args.normalize_event,
                            args.resize,
                            args.resize_mode,
                            out_dir
                        )
                        tasks.append(task_args)
                    
                    current_accumulated_indices = []
                    j += 1
            
            # --- 2. Parallel Processing ---
            if tasks:
                n_workers = args.num_workers if args.num_workers > 0 else 1
                
                if n_workers > 1:
                    with mp.Pool(processes=n_workers, initializer=_worker_init, initargs=(x_all, y_all, t_all, p_all)) as pool:
                        list(tqdm(pool.imap_unordered(_process_image_task, tasks, chunksize=10), 
                                  total=len(tasks), desc=f"Proc {subfolder}", leave=False))
                else:
                    _worker_init(x_all, y_all, t_all, p_all)
                    for t in tqdm(tasks, desc=f"Proc {subfolder}", leave=False):
                        _process_image_task(t)

        return (subfolder, True, 'ok')

    except Exception as e:
        traceback.print_exc()
        return (subfolder, False, f'err:{e.__class__.__name__}')

def main():
    parser = argparse.ArgumentParser(description="Generate 20-channel event tensors for DSEC.")
    
    parser.add_argument("--root", default="/data/storage/jianwen/DSEC", type=str)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    
    # Processing Config
    parser.add_argument("--nr-events-data", type=int, default=10, help="Number of temporal chunks per sample.")
    parser.add_argument("--nr-bins-per-data", type=int, default=2, help="Number of channels produced per temporal chunk.")
    parser.add_argument("--representation", choices=["histogram", "voxel_grid"], default="histogram")
    parser.add_argument("--separate-pol", action="store_true", help="Keep positive/negative events in separate bins.")
    parser.add_argument("--normalize-event", action="store_true", default=True)
    
    # Spatial Config
    parser.add_argument("--resize", type=int, nargs=2, default=(224, 224), metavar=("H", "W"))
    parser.add_argument("--resize-mode", choices=["nearest", "bilinear", "bicubic"], default="bilinear")
    
    # Execution
    parser.add_argument("--num_workers", type=int, default=1, help="Number of intra-folder workers.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start-from", type=str, default=None, help="Start processing from this folder (inclusive). Previous folders are skipped. This folder is forced to restart.")
    
    args = parser.parse_args()
    
    # List subfolders
    split_img_root = os.path.join(args.root, f"{args.split}_images")
    if not os.path.isdir(split_img_root):
        print(f"Root dir {split_img_root} not found.")
        return

    subfolders = sorted(os.listdir(split_img_root))
    print(f"Found {len(subfolders)} subfolders in {args.split} split.")
    print(f"Processing sequentially folder-by-folder, with {args.num_workers} parallel workers inside each folder.")

    ok = 0
    err = 0
    skip = 0
    
    total_folders = len(subfolders)
    
    processing_started = True
    if args.start_from:
        processing_started = False
        print(f"Will skip folders until: {args.start_from}")

    for idx, sf in enumerate(subfolders):
        force_for_this = False
        if not processing_started:
            if sf == args.start_from:
                processing_started = True
                force_for_this = True
            else:
                continue
        
        print(f"[{idx+1}/{total_folders}] Processing folder: {sf}")
        _, success, status = process_one_folder(sf, args, force_overwrite=force_for_this)
        if success:
            if status == 'skip_exists': 
                skip += 1
                print(f"  -> Skipped (exists)")
            else: 
                ok += 1
        else:
            err += 1
            print(f"  -> Failed: {status}")
                    
    print(f"Done. OK: {ok}, Skip: {skip}, Err: {err}")

if __name__ == "__main__":
    main()