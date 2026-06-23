import sys
sys.path.append("dinov2")
import os
import glob
import argparse
import shutil
import h5py
import torch
import torchvision 
from data_utils_eventslicer import EventSlicer
import numpy as np
import cv2
import yaml
from tqdm import tqdm
from dinov2.models.vision_transformer import vit_small

def load_event_image(args):
    """
        Args:
            event_root       str: Event data file, should be a h5 file.
            image_dir       str: Image data folder.
        Returns:
            event_dict      dict: {"t": np.ndarray, "p": np.ndarray, "x": np.ndarray, "y": np.ndarray}
            unique_index    list: The index of unique moments in the event data t.
            image_names:    list(dict): [{"image_name": image_name str, "image_timestamp": image_timestamp np.ndarray}]
    """

    image_names = glob.glob(os.path.join(args.image_dir, "*.png"))
    image_names.sort()
    image_timestamp = np.loadtxt(args.image_timestamp_dir, dtype=np.int64)
    image_dict = [{"image_name": image_names[i], "image_timestamp": image_timestamp[i].item()} for i in range(len(image_names))]

    event = h5py.File(args.event_dir, "r")
    event_slicer = EventSlicer(event)
    start_timestamp = event_slicer.get_start_time_us()
    end_timestamp = event_slicer.get_final_time_us()
    duration = (end_timestamp - start_timestamp)
    event_dict = event_slicer.get_events(start_timestamp, start_timestamp + duration)
    t = event_dict["t"]
    unique_moments, unique_index = np.unique(t, return_index=True)
    event_dict["unique_index"] = unique_index
    
    print(f"num of events: {t.shape[0]}")
    print(f"num of unique moments: {unique_moments.shape[0]}")
    print(f"event start timestamp: {t[0]}, end_timestamp: {t[-1]}, duration in sec: {(t[-1] - t[0]) / 1e6:.4f}s, in microseconds: {t[-1] - t[0]}us")
    print(f"image start timestamp: {image_dict[0]['image_timestamp']}, end_timestamp: {image_dict[-1]['image_timestamp']}")
    print(f"event fps: {len(unique_moments) / (end_timestamp - start_timestamp) * 1e6}")
    print(f"image fps: {len(image_dict) / (end_timestamp - start_timestamp) * 1e6}")
    print(f"num of images: {len(image_dict)}")
    return event_dict, image_dict

def save_warpped_event_image_pair(event_dict, image_dict, args):    
    def get_camera(args) -> np.ndarray:
        def create_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
            """
            Creates a 4x4 homogeneous transformation matrix from a rotation matrix R (3x3)
            and a translation vector t (3,).
            """
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()
            return T
        def invert_transform(T: np.ndarray) -> np.ndarray:
            """
            Inverts a homogeneous transformation matrix T.
            For a matrix T = [R, t; 0, 1], its inverse is [R.T, -R.T @ t; 0, 1].
            """
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
        with open(args.calib_dir, 'r') as file:
            data = yaml.safe_load(file)
        file.close()
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

        # Construct 3x3 intrinsic matrices from the parameters.
        K_event = cretate_K(intrin_event_rect)
        K_dist  = cretate_K(intrin_event_dist)
        K_image = cretate_K(intrin_image)
        
        # Compute the inverse transformation: from event camera to image camera.
        T_i2e = Ti @ T_i2e @ invert_transform(Te)
        R_i2e = T_i2e[:3, :3]

        # Compute the homography from event frame (assumed plane) to the image.
        H_homography = K_image @ R_i2e @ np.linalg.inv(K_event)
        H_homography = np.linalg.inv(H_homography)

        return H_homography, K_event, K_dist, dist_coeffs, resolution, Re

    def accumulate_to_rgb(x, y, p, shape, G=None, pct=99):
        def percentile_normalise(arr, pct=99):
            """Scale |arr| into 0…255 by clipping the pct-th percentile."""
            clip_val = np.percentile(arr, pct)
            clip_val = max(clip_val, 1)               # avoid divide-by-0
            arr = np.clip(arr, 0, clip_val)
            return (arr / clip_val * 255).astype(np.uint8)
        H, W = shape
        pos = np.zeros((H, W), np.uint32)
        neg = np.zeros((H, W), np.uint32)

        np.add.at(pos, (y[p > 0], x[p > 0]), 1)
        np.add.at(neg, (y[p <= 0], x[p <= 0]), 1)

        # independent percentile scaling
        R = percentile_normalise(pos, pct)
        B = percentile_normalise(neg, pct)
        G = np.zeros_like(R) if G is None else G

        rgb = np.dstack([R, G, B])

        rgb[(pos + neg) == 0] = [255, 255, 255]

        return rgb.astype(np.uint8)
    dinov2 = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6)
    t, x, y, p = event_dict["t"], event_dict["x"], event_dict["y"], event_dict["p"]
    accu_x, accu_y, accu_p = [], [], []
    j = 0
    H_homography, K_event, K_dist, dist_coeffs, resolution, Re = get_camera(args)
    W, H = resolution
    unique_index = event_dict["unique_index"]
    for i in tqdm(range(len(unique_index) - 1)):
        if j == len(image_dict):
            break
        accu_x.extend(x[unique_index[i]: unique_index[i+1]])
        accu_y.extend(y[unique_index[i]: unique_index[i+1]])
        accu_p.extend(p[unique_index[i]: unique_index[i+1]])
        current_image_timestamp = image_dict[j]["image_timestamp"]
        next_image_timestamp = image_dict[j + 1]["image_timestamp"] if (j + 1) < len(image_dict) else image_dict[j]["image_timestamp"]
        middle_timestamp = (current_image_timestamp + next_image_timestamp) / 2
        current_event_timestamp = t[unique_index[i]]
        if current_event_timestamp > middle_timestamp:
            # 1. warp the intensity image FIRST so we can pass its gray frame to G
            image          = cv2.imread(image_dict[j]["image_name"])
            warped         = cv2.warpPerspective(image, H_homography, (W, H),
                                                flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT)

            # 2. build RGB event frame
            gray = np.ones(args.event_size) * 0.1
            gray[accu_y, accu_x] = accu_p
            event_image = np.ones((H, W, 3), dtype=np.uint8) * 255
            event_image[gray == 1] = np.array([255, 0, 0])
            event_image[gray == 0] = np.array([0, 0, 255])

            # 3. build accumulated RGB event frame
            event_rgb = accumulate_to_rgb(
                np.array(accu_x), np.array(accu_y), np.array(accu_p),
                args.event_size,
                G = cv2.cvtColor(event_image, cv2.COLOR_RGB2GRAY),
                pct=99                         # 99th-percentile clip
            )

            mapping = cv2.initUndistortRectifyMap(K_dist, dist_coeffs, Re, K_event, resolution, cv2.CV_32FC2)[0]
            event_rgb = cv2.remap(event_rgb, mapping, None, interpolation=cv2.INTER_CUBIC)

            # 4. save both outputs
            cv2.imwrite(os.path.join(args.event_image_dir, f"{current_image_timestamp}.png"), event_rgb)
            cv2.imwrite(os.path.join(args.warpped_dir, f"{current_image_timestamp}.png"), warped)
            label = os.path.join("/data/storage/jianwen/DSEC/train_semantic_segmentation/train/zurich_city_00_a/11classes_renamed", f"{current_image_timestamp}.png")
            label = cv2.imread(label)
            label = ((label - label.min()) / (label.max() - label.min())) * 255
            cv2.imwrite(os.path.join(args.vis_dir, f"{current_image_timestamp}.png"), np.concat([event_rgb, warped, label]))

            # 5. reset
            accu_x, accu_y, accu_p = [], [], []
            j += 1

def save_event_tokens(event_dict, image_dict, args):
    t, x, y, p = event_dict["t"], event_dict["x"], event_dict["y"], event_dict["p"]
    unique_index = event_dict["unique_index"]
    j = 0
    for i in tqdm(range(len(unique_index) - 1)):
        if j == len(image_dict):
            break
        current_image_timestamp = image_dict[j]["image_timestamp"]
        current_event_timestamp = t[unique_index[i]]


def main(args):
    event_dict, image_info = load_event_image(args)
    save_warpped_event_image_pair(event_dict, image_info, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/storage/jianwen/DSEC", type=str)
    parser.add_argument("--mode", default="train", type=str) # test
    parser.add_argument("--event_size", default=(480, 640), type=tuple)
    args = parser.parse_args()
    
    args.image_root = os.path.join(args.root, f"{args.mode}_images")
    args.events_root = os.path.join(args.root, f"{args.mode}_events")
    args.calib_root = os.path.join(args.root, f"{args.mode}_calibration")
    sementatic_subfolders = os.listdir("/scratch/brunns/data/dsec/train_semantic_segmentation/train")
    for folder in sorted(os.listdir(args.image_root)):
        if folder not in sementatic_subfolders:
            continue
        folder_path = os.path.join(args.image_root, folder)
        print("processing folder:", folder)
        args.calib_dir = os.path.join(args.calib_root, folder, "calibration", "cam_to_cam.yaml")
        args.event_dir = os.path.join(args.events_root, folder, "events", "left", "events.h5")
        args.image_dir = os.path.join(args.image_root, folder, "images", "left", "rectified")
        args.image_timestamp_dir = os.path.join(args.image_root, folder, "images", "timestamps.txt")
        args.save_dir = os.path.join(args.image_root, folder, "images", "left")
        args.warpped_dir = os.path.join(args.save_dir, "warpped")
        args.vis_dir = os.path.join(args.save_dir, "vis")
        args.event_image_dir = os.path.join(args.save_dir, "event_image")
        os.makedirs(args.warpped_dir, exist_ok=True)
        os.makedirs(args.event_image_dir, exist_ok=True)
        os.makedirs(args.vis_dir, exist_ok=True)
        # if len(os.listdir(args.warpped_dir)) == len(os.listdir(args.image_dir)):
        #     print(f"warpped images already exists in {args.warpped_dir}, skip this folder")
        #     continue
        # else:
        print("---------------------------------- removing both event images and warpped images to have a clean start")
        shutil.rmtree(args.warpped_dir)
        shutil.rmtree(args.event_image_dir)
        os.makedirs(args.warpped_dir, exist_ok=True)
        os.makedirs(args.event_image_dir, exist_ok=True)
        print(f"warpped images will be saved in:        {args.warpped_dir}")
        print(f"accumulated events will be saved in:    {args.event_image_dir}")
        main(args)
        
    