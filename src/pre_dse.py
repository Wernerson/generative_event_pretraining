import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

sys.path.append("dinov2")
sys.path.append("segmentation")
import glob
import os
import shutil

import cv2
import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
import yaml
from PIL import Image

from utils import EventSlicer
from dinov2.models.vision_transformer import vit_small
from dataset import PadToMinSide, PairedProcessor, Normalize, RandomCrop, RandomHorizontalFlip, RandomSwapEventRedBlue, ResizeKeepRatio, ToTensor, CenterCrop
from utils import accumulate_to_rgb

def _compute_stats_for_subfolder(image_root: str, subfolder: str):
    """Compute RGB sums, squared sums, and pixel counts for one subfolder.
    Returns tuple: (ev_sum, ev_sq_sum, ev_count, img_sum, img_sq_sum, img_count)
    """
    ev_sum = np.zeros(3, dtype=np.float64)
    ev_sq_sum = np.zeros(3, dtype=np.float64)
    ev_count = 0

    img_sum = np.zeros(3, dtype=np.float64)
    img_sq_sum = np.zeros(3, dtype=np.float64)
    img_count = 0

    save_root = os.path.join(image_root, subfolder, "images", "left")
    warpped_dir = os.path.join(save_root, "warpped")
    eventImage_dir = os.path.join(save_root, "eventImage")
    if not (os.path.isdir(warpped_dir) and os.path.isdir(eventImage_dir)):
        return ev_sum, ev_sq_sum, ev_count, img_sum, img_sq_sum, img_count

    warpped_files = {f for f in os.listdir(warpped_dir) if f.lower().endswith(".png")}
    event_files = {f for f in os.listdir(eventImage_dir) if f.lower().endswith(".png")}
    common = sorted(warpped_files & event_files)
    if not common:
        return ev_sum, ev_sq_sum, ev_count, img_sum, img_sq_sum, img_count

    for name in common:
        # event
        ev_path = os.path.join(eventImage_dir, name)
        ev = cv2.imread(ev_path, cv2.IMREAD_COLOR)
        if ev is not None:
            ev = cv2.cvtColor(ev, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            h, w, _ = ev.shape
            ev_reshaped = ev.reshape(-1, 3)
            ev_sum += ev_reshaped.sum(axis=0)
            ev_sq_sum += (ev_reshaped * ev_reshaped).sum(axis=0)
            ev_count += h * w

        # image
        img_path = os.path.join(warpped_dir, name)
        im = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if im is not None:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            h, w, _ = im.shape
            im_reshaped = im.reshape(-1, 3)
            img_sum += im_reshaped.sum(axis=0)
            img_sq_sum += (im_reshaped * im_reshaped).sum(axis=0)
            img_count += h * w

    return ev_sum, ev_sq_sum, ev_count, img_sum, img_sq_sum, img_count

class Processor():
    def __init__(self, args):
        super().__init__()

        self.root               = args.root
        self.split              = args.split
        self.device             = args.device

        self.image_root             = os.path.join(args.root, f"{args.split}_images")
        self.events_root            = os.path.join(args.root, f"{args.split}_events")
        self.calib_root             = os.path.join(args.root, f"{args.split}_calibration")
        self.sementatic_root        = os.path.join(args.root, f"{args.split}_semantic_segmentation/{self.split}")
        self.DSEC_ME = [0.8993729784963826, 0.7969581014619264, 0.8928228776286392]
        self.DSEC_SE = [0.2726268701553345, 0.2706460952758789, 0.2812058925628662]
        self.DSEC_MI = [0.23862534365611895, 0.24712072838418375, 0.2574492927024542]
        self.DSEC_SI = [0.2195923498258092, 0.2361895432347472, 0.2633142601113288]
        self.origi_H, self.origi_W  = None, None
        self.type = 'EI'
        self.DSEC_H, self.DSEC_W         = 224, 224
        self.scale_range = (0.5, 2)
    
    def load_event_image(self, image_dir, event_dir, image_timestamp_dir):
        """
            Args:
                event_root       str: Event data file, should be a h5 file.
                image_dir       str: Image data folder.
            Returns:
                event_dict      dict: {"t": np.ndarray, "p": np.ndarray, "x": np.ndarray, "y": np.ndarray}
                unique_index    list: The index of unique moments in the event data t.
                image_names:    list(dict): [{"image_name": image_name str, "image_timestamp": image_timestamp np.ndarray}]
        """
        image_names     = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        image_timestamp = np.loadtxt(image_timestamp_dir, dtype=np.int64)
        image_dict = [{"image_name": image_names[i], "image_timestamp": image_timestamp[i].item()} for i in range(len(image_names))]

        event = h5py.File(event_dir, "r")
        event_slicer = EventSlicer(event)
        start_timestamp = event_slicer.get_start_time_us()
        end_timestamp = event_slicer.get_final_time_us()
        duration = (end_timestamp - start_timestamp)
        duration = min(end_timestamp - start_timestamp, 5 * 1_000_000)  # cap at 5s (in microseconds)
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
    
    def _get_camera(self, calib_dir) -> np.ndarray:
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
        with open(calib_dir, 'r') as file:
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
    
    @torch.no_grad()
    def save_warpped_event_pair(self, event_dict, image_dict, calib_dir, eventImage_dir, warpped_dir, vis_dir, label_dir):
        t, x, y, p = event_dict["t"], event_dict["x"], event_dict["y"], event_dict["p"]
        accu_x, accu_y, accu_p = [], [], []
        j = 0
        H_homography, K_event, K_dist, dist_coeffs, resolution, Re = self._get_camera(calib_dir)
        W, H = resolution
        unique_index = event_dict["unique_index"]
        for i in tqdm(range(len(unique_index) - 1)):
            if j == len(image_dict):
                break
            accu_x.extend(x[unique_index[i]: unique_index[i+1]])
            accu_y.extend(y[unique_index[i]: unique_index[i+1]])
            accu_p.extend(p[unique_index[i]: unique_index[i+1]])
            # event_tokens = []
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
                mapping = cv2.initUndistortRectifyMap(K_dist, dist_coeffs, Re, K_event, resolution, cv2.CV_32FC2)[0]
                event_frame = accumulate_to_rgb(
                    np.array(accu_x), np.array(accu_y), np.array(accu_p),
                    (H, W),
                    pct=90,                         # 99th-percentile clip
                )

                event_frame = cv2.remap(event_frame, mapping, None, interpolation=cv2.INTER_CUBIC)
                
                # 4. save both outputs
                cv2.imwrite(os.path.join(eventImage_dir, f"{current_image_timestamp}.png"), event_frame)
                cv2.imwrite(os.path.join(warpped_dir, f"{current_image_timestamp}.png"), warped)

                # save additional label vis
                label = os.path.join(label_dir, f"{current_image_timestamp}.png")
                if os.path.exists(label):
                    label = cv2.imread(label)
                    label = ((label - label.min()) / (label.max() - label.min())) * 255
                    # concatenate side-by-side for quick visual check
                    try:
                        vis_img = np.concatenate([event_frame, warped, label], axis=1)
                        cv2.imwrite(os.path.join(vis_dir, f"{current_image_timestamp}.png"), vis_img)
                    except Exception:
                        pass  # ignore visualization errors

                # 5. reset
                accu_x, accu_y, accu_p = [], [], []
                j += 1
        
    def _process_subfolder(self, subfolder):
        print("---------------------------------- processing folder:", subfolder)
        event_dir = os.path.join(self.events_root, subfolder, "events", "left", "events.h5")
        image_dir = os.path.join(self.image_root, subfolder, "images", "left", "rectified")
        image_timestamp_dir = os.path.join(self.image_root, subfolder, "images", "timestamps.txt")
        calib_dir = os.path.join(self.calib_root, subfolder, "calibration", "cam_to_cam.yaml")
        save_root = os.path.join(self.image_root, subfolder, "images", "left")
        warpped_dir = os.path.join(save_root, "warpped")
        vis_dir = os.path.join(save_root, "vis")
        eventImage_dir = os.path.join(save_root, "eventImage")
        label_dir = os.path.join(self.sementatic_root, subfolder, "11classes_renamed")

        # completeness check (skip if already fully processed)
        # try:
        #     # number of rectified source images defines expected frame count
        #     expected = len(glob.glob(os.path.join(image_dir, '*.png')))
        #     warpped_cnt = len(glob.glob(os.path.join(warpped_dir, '*.png'))) if os.path.isdir(warpped_dir) else 0
        #     event_cnt = len(glob.glob(os.path.join(eventImage_dir, '*.png'))) if os.path.isdir(eventImage_dir) else 0
        #     if expected > 0 and warpped_cnt == expected and event_cnt == expected:
        #         print(f"Skip (already complete) {subfolder}: {warpped_cnt}/{expected} frames")
        #         return subfolder
        # except Exception as e:
        #     print(f"Completeness check failed for {subfolder}, will reprocess. Error: {e}")

        print("removing both event images and warpped images to have a clean start")
        shutil.rmtree(warpped_dir, ignore_errors=True)
        shutil.rmtree(eventImage_dir, ignore_errors=True)
        shutil.rmtree(vis_dir, ignore_errors=True)

        os.makedirs(warpped_dir, exist_ok=True)
        os.makedirs(eventImage_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(warpped_dir, exist_ok=True)
        os.makedirs(eventImage_dir, exist_ok=True)
        print(f"warpped images will be saved in:        {warpped_dir}")
        print(f"accumulated events will be saved in:    {eventImage_dir}")

        event_dict, image_dict = self.load_event_image(image_dir, event_dir, image_timestamp_dir)
        self.save_warpped_event_pair(event_dict, image_dict, calib_dir, eventImage_dir, warpped_dir, vis_dir, label_dir)
        return subfolder

    def run(self, subfolders=None, n_workers=1):
        target_subfolders = [s for s in sorted(os.listdir(self.image_root)) if (subfolders is None or s in subfolders)]
        # Filter out already complete before launching workers to avoid spawning tasks unnecessarily
        filtered = []
        for sf in target_subfolders:
            image_timestamp_dir = os.path.join(self.image_root, sf, "images", "left", "rectified", "..", "timestamps.txt")
            # path above is odd (..), rely on _process_subfolder internal skip instead; just collect all
            filtered.append(sf)
        target_subfolders = filtered
        if n_workers is None or n_workers <= 1:
            for sf in target_subfolders:
                self._process_subfolder(sf)
        else:
            print(f"Parallel processing with {n_workers} workers over {len(target_subfolders)} folders")
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(self._process_subfolder, sf): sf for sf in target_subfolders}
                for f in as_completed(futures):
                    sf = futures[f]
                    try:
                        _ = f.result()
                        print(f"Finished: {sf}")
                    except Exception as e:
                        print(f"Failed: {sf} with error: {e}")

    def compute_rgb_stats(self, subfolders=None, save_to: str | None = None, workers: int = 1):
        """
        Compute dataset-wide RGB mean and std for event (eventImage) and image (warpped) modalities.
        - Iterates over subfolders under self.image_root.
        - Expects generated pairs from run(): left/eventImage and left/warpped.

        Args:
            subfolders: optional list of subfolder names to restrict computation.
            save_to: optional path to a YAML file to save the results.

        Prints the 12 numbers and optionally saves them.
        """
        # running totals in float64 for numerical stability
        ev_sum = np.zeros(3, dtype=np.float64)
        ev_sq_sum = np.zeros(3, dtype=np.float64)
        ev_count = 0

        img_sum = np.zeros(3, dtype=np.float64)
        img_sq_sum = np.zeros(3, dtype=np.float64)
        img_count = 0

        folders = [s for s in sorted(os.listdir(self.image_root)) if (subfolders is None or s in subfolders)]

        if workers is None or workers <= 1:
            for subfolder in tqdm(folders, desc="stats folders"):
                ev_s, ev_ss, ev_c, im_s, im_ss, im_c = _compute_stats_for_subfolder(self.image_root, subfolder)
                ev_sum += ev_s; ev_sq_sum += ev_ss; ev_count += ev_c
                img_sum += im_s; img_sq_sum += im_ss; img_count += im_c
        else:
            print(f"Parallel stats with {workers} workers over {len(folders)} folders")
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_compute_stats_for_subfolder, self.image_root, sf): sf for sf in folders}
                for f in tqdm(as_completed(futures), total=len(futures), desc="stats futures"):
                    try:
                        ev_s, ev_ss, ev_c, im_s, im_ss, im_c = f.result()
                        ev_sum += ev_s; ev_sq_sum += ev_ss; ev_count += ev_c
                        img_sum += im_s; img_sq_sum += im_ss; img_count += im_c
                    except Exception as e:
                        sf = futures[f]
                        print(f"Failed stats for {sf}: {e}")

        def finalize(sum_, sq_sum_, count_):
            if count_ == 0:
                return [float('nan')] * 3, [float('nan')] * 3
            mean = sum_ / count_
            var = np.maximum(sq_sum_ / count_ - mean * mean, 0.0)
            std = np.sqrt(var)
            return mean.tolist(), std.tolist()

        ev_mean, ev_std = finalize(ev_sum, ev_sq_sum, ev_count)
        img_mean, img_std = finalize(img_sum, img_sq_sum, img_count)

        print("==== Dataset RGB stats (0..1 range) ====")
        print(f"Event  mean: {ev_mean}")
        print(f"Event  std : {ev_std}")
        print(f"Image  mean: {img_mean}")
        print(f"Image  std : {img_std}")

        if save_to is not None:
            stats = {
                "event": {"mean": ev_mean, "std": ev_std},
                "image": {"mean": img_mean, "std": img_std},
            }
            out_dir = os.path.dirname(save_to)
            os.makedirs(out_dir if out_dir != "" else ".", exist_ok=True)
            with open(save_to, "w") as f:
                yaml.safe_dump(stats, f)
            print(f"Saved stats to {save_to}")

    @torch.no_grad()
    def process_tokens(self, subfolders=None, workers: int = 1):
        self.train_preprocessor     = PairedProcessor([
                                            ToTensor(type=self.type),
                                            Normalize(self.DSEC_ME, self.DSEC_SE, self.DSEC_MI, self.DSEC_SI, type=self.type),
                                            # RandomSwapEventRedBlue(type=self.type),
                                            # RandomHorizontalFlip(p=0.5),
                                            PadToMinSide(target=(self.DSEC_H, self.DSEC_W), pad_x1=0, pad_x2=0),
                                            CenterCrop((self.DSEC_H, self.DSEC_W)),
                                            ])
        self.valid_preprocessor     = PairedProcessor([
                                                    ToTensor(type=self.type),
                                                    Normalize(self.DSEC_ME, self.DSEC_SE, self.DSEC_MI, self.DSEC_SI, type=self.type),
                                                    PadToMinSide(target=(self.DSEC_H, self.DSEC_W), pad_x1=0, pad_x2=0),
                                                    CenterCrop((self.DSEC_H, self.DSEC_W)),
                                                    ])
        self.image_encoder = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6).to(self.device)
        self.event_encoder = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6).to(self.device)
        self.image_encoder.load_state_dict(torch.load("./ckpts/dinov2_vits14_pretrain.pth", weights_only=True, map_location=self.device), strict=True)
        self.event_encoder.load_state_dict(torch.load("./ckpts/small.pt", weights_only=True, map_location=self.device)["event_encoder"], strict=True)
        self.image_encoder.eval()
        self.event_encoder.eval()

        for subfolder in sorted(os.listdir(self.image_root)):
            if subfolders is not None and subfolder not in subfolders:
                continue
            print("---------------------------------- merging tokens in folder:", subfolder)
            save_root = os.path.join(self.image_root, subfolder, "images", "left")
            eventToken_dir = os.path.join(save_root, "eventToken")
            imageToken_dir = os.path.join(save_root, "imageToken")
            warpped_dir = os.path.join(save_root, "warpped")
            eventImage_dir = os.path.join(save_root, "eventImage")
            print("removing to have a clean start")
            shutil.rmtree(eventToken_dir, ignore_errors=True)
            shutil.rmtree(imageToken_dir, ignore_errors=True)
            os.makedirs(eventToken_dir, exist_ok=True)
            os.makedirs(imageToken_dir, exist_ok=True)

            names = sorted(os.listdir(warpped_dir))

            def _process_one(name: str):
                timestamp = name.split(".")[0]
                warped = Image.open(os.path.join(warpped_dir, name))
                event_rgb = Image.open(os.path.join(eventImage_dir, name))

                if self.split == "train":
                    event_rgb_t, warped_t = self.train_preprocessor(event_rgb, warped)
                else:
                    event_rgb_t, warped_t = self.valid_preprocessor(event_rgb, warped)

                image_tokens = self.image_encoder.forward_features(warped_t.unsqueeze(0).to(self.device))["x_norm_patchtokens"].squeeze(0).cpu()
                event_tokens = self.event_encoder.forward_features(event_rgb_t.unsqueeze(0).to(self.device))["x_norm_patchtokens"].squeeze(0).cpu()
                torch.save(image_tokens, os.path.join(imageToken_dir, f"{timestamp}.pt"))
                torch.save(event_tokens, os.path.join(eventToken_dir, f"{timestamp}.pt"))

            if workers is None or workers <= 1:
                for name in tqdm(names):
                    _process_one(name)
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    list(tqdm(ex.map(_process_one, names), total=len(names)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/storage/jianwen/DSEC", type=str)
    parser.add_argument("--device", default="cuda:3", type=str)
    parser.add_argument("--workers", default=1, type=int, help="number of parallel processes for run()")
    parser.add_argument("--stats_out", default=None, type=str, help="optional path to save stats as YAML")
    args = parser.parse_args()

    for split in ["train", "test"]:
        args.split = split

        processor = Processor(args)
        subfolders = os.listdir(processor.sementatic_root)
        # processor.run(n_workers=args.workers)
        processor.process_tokens(workers=args.workers)
        # processor.compute_rgb_stats(save_to=args.stats_out, workers=args.workers)

