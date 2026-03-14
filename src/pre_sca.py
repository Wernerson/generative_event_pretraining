import sys

from utils import accumulate_to_rgb
sys.path.append("dinov2")

import os

import PIL
import numpy as np
import torch
from torchvision.transforms import transforms
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from dinov2.models.vision_transformer import vit_small
from dataset import PadToMinSide, PairedProcessor, Normalize, RandomCrop, RandomHorizontalFlip, RandomSwapEventRedBlue, ResizeKeepRatio, ToTensor, CenterCrop

def _process_sequence(sequence_path: str):
    os.makedirs(os.path.join(sequence_path, "event_images"), exist_ok=True)
    event_path = os.path.join(sequence_path, "events/data")
    image_path = os.path.join(sequence_path, "rgb/data")
    event_files = [i for i in sorted(os.listdir(event_path)) if i.endswith(".npz")]
    image_files = [i for i in sorted(os.listdir(image_path)) if i.endswith(".png")]
    for i in range(len(event_files)):
        event_data = np.load(os.path.join(event_path, event_files[i]), allow_pickle=True)
        image_data = PIL.Image.open(os.path.join(image_path, image_files[i]))
        shape = (image_data.size[1], image_data.size[0])
        x = event_data["x"]; y = event_data["y"]; p = event_data["p"]; t = event_data["t"]
        event_rgb = Image.fromarray(accumulate_to_rgb(x, y, p, shape))
        event_rgb.save(os.path.join(sequence_path, "event_images", f"{i:06d}.png"))

def load_data(path: str, num_workers: int = 1):
    for cluster in sorted(os.listdir(path)):
        cluster_path = os.path.join(path, cluster)
        sequences = sorted(os.listdir(cluster_path))
        sequence_paths = [os.path.join(cluster_path, s) for s in sequences]

        if num_workers and num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                for _ in tqdm(ex.map(_process_sequence, sequence_paths), total=len(sequence_paths)):
                    pass
        else:
            for sequence_path in tqdm(sequence_paths):
                _process_sequence(sequence_path)

def get_feature(path: str):
    H, W = 224, 224
    ME, SE            = [0.9888106104297153, 0.9747728761936781, 0.9859595939484498], [0.06632214632296055, 0.09744895769725151, 0.0735958979181792]
    MI, SI            = [0.3691855686450492, 0.372362750445305, 0.38055244521714615], [0.21236170116948833, 0.20754919382931097, 0.21394057988512902]
    DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
    type = "EI"
    train_processor     = PairedProcessor([
                                            ToTensor(type=type),
                                            Normalize(ME, SE, MI, SI, type=type),
                                            RandomSwapEventRedBlue(type=type),
                                            RandomHorizontalFlip(p=0.5),
                                            PadToMinSide(target=(H, W), pad_x1=0, pad_x2=0),
                                            RandomCrop(crop_size=(H, W), type=type),
                                            ])
    valid_processor     = PairedProcessor([
                                                    ToTensor(type=type),
                                                    Normalize(ME, SE, MI, SI, type=type),
                                                    PadToMinSide(target=(H, W), pad_x1=0, pad_x2=0),
                                                    CenterCrop((H, W)),
                                                    ])

    image_encoder = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6).to(DEVICE)
    event_encoder = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6).to(DEVICE)
    image_encoder.load_state_dict(torch.load("/data/storage/jianwen/cache/dinov2/dinov2_vits14_pretrain.pth", weights_only=True), strict=True)
    # event_encoder.load_state_dict(torch.load("/data/storage/jianwen/cache/dinov2/dinov2_vits14_pretrain.pth", weights_only=True), strict=True)
    event_encoder.load_state_dict(torch.load("/data/storage/jianwen/cache/ckpt_matters/gra_mixture_4x.pt", weights_only=True)["event_encoder"], strict=True)
    image_encoder.eval()
    event_encoder.eval()
    for cluster in sorted(os.listdir(path)):
        cluster_path = os.path.join(path, cluster)
        for sequence in tqdm(sorted(os.listdir(cluster_path))):
            sequence_path = os.path.join(cluster_path, sequence)
            os.makedirs(os.path.join(sequence_path, "event_features"), exist_ok=True)
            os.makedirs(os.path.join(sequence_path, "image_features"), exist_ok=True)
            event_path = os.path.join(sequence_path, "event_images")
            image_path = os.path.join(sequence_path, "rgb/data")
            event_files = [i for i in sorted(os.listdir(event_path)) if i.endswith(".png")]
            image_files = [i for i in sorted(os.listdir(image_path)) if i.endswith(".png")]
            for i in range(len(event_files)):
                event_image = Image.open(os.path.join(event_path, event_files[i]))
                image = Image.open(os.path.join(image_path, image_files[i]))
                
                if "train" in path:
                    event_tensor, image_tensor = train_processor(event_image, image)
                else:
                    event_tensor, image_tensor = valid_processor(event_image, image)
                event_tensor, image_tensor = event_tensor.unsqueeze(0).to(DEVICE), image_tensor.unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    event_feature = event_encoder.forward_features(event_tensor)["x_norm_patchtokens"].squeeze(0)
                    image_feature = image_encoder.forward_features(image_tensor)["x_norm_patchtokens"].squeeze(0)

                torch.save(event_feature, os.path.join(sequence_path, "event_features", f"{i:06d}.pt"))
                torch.save(image_feature, os.path.join(sequence_path, "image_features", f"{i:06d}.pt"))

def compute_mean_std(path: str, num_workers: int = 8):
    """
    Traverse all clusters/sequences under `path` and compute per-channel mean and std
    for both event images and RGB images.

    Assumptions:
    - Event images are stored under {sequence_path}/event_images/*.png (created by load_data).
    - RGB images are stored under {sequence_path}/rgb/data/*.png.
    - Stats are computed on float pixels scaled to [0, 1] after converting to RGB.

    Returns:
    - dict with keys 'event' and 'rgb', each containing {'mean': list[3], 'std': list[3]}.
    """

    def gather_files(root: str):
        event_files, rgb_files = [], []
        for cluster in sorted(os.listdir(root)):
            cluster_path = os.path.join(root, cluster)
            if not os.path.isdir(cluster_path):
                continue
            for sequence in sorted(os.listdir(cluster_path)):
                sequence_path = os.path.join(cluster_path, sequence)
                if not os.path.isdir(sequence_path):
                    continue
                ev_dir = os.path.join(sequence_path, "event_images")
                rgb_dir = os.path.join(sequence_path, "rgb", "data")
                if os.path.isdir(ev_dir):
                    event_files.extend(
                        [os.path.join(ev_dir, f) for f in sorted(os.listdir(ev_dir)) if f.endswith(".png")]
                    )
                if os.path.isdir(rgb_dir):
                    rgb_files.extend(
                        [os.path.join(rgb_dir, f) for f in sorted(os.listdir(rgb_dir)) if f.endswith(".png")]
                    )
        return event_files, rgb_files

    def file_sums(path_: str):
        # Compute per-image sums and squared sums per channel, with pixel count
        with Image.open(path_) as im:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.float32) / 255.0  # H, W, 3 in [0,1]
        h, w, _ = arr.shape
        n = h * w
        flat = arr.reshape(-1, 3)
        s = flat.sum(axis=0)
        s2 = (flat * flat).sum(axis=0)
        return s, s2, n

    def accumulate(files):
        total_sum = np.zeros(3, dtype=np.float64)
        total_sumsq = np.zeros(3, dtype=np.float64)
        total_pixels = 0

        if len(files) == 0:
            return total_sum, total_sumsq, total_pixels

        if num_workers and num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                for s, s2, n in tqdm(ex.map(file_sums, files), total=len(files)):
                    total_sum += s
                    total_sumsq += s2
                    total_pixels += n
        else:
            for f in tqdm(files):
                s, s2, n = file_sums(f)
                total_sum += s
                total_sumsq += s2
                total_pixels += n

        return total_sum, total_sumsq, total_pixels

    # Gather file lists
    event_files, rgb_files = gather_files(path)

    # Accumulate in parallel
    ev_sum, ev_sumsq, ev_pixels = accumulate(event_files)
    rgb_sum, rgb_sumsq, rgb_pixels = accumulate(rgb_files)

    def finalize(total_sum, total_sumsq, total_pixels):
        if total_pixels == 0:
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        mean = (total_sum / total_pixels).astype(np.float64)
        var = (total_sumsq / total_pixels) - (mean * mean)
        var = np.clip(var, a_min=0.0, a_max=None)
        std = np.sqrt(var)
        return mean.tolist(), std.tolist()

    ev_mean, ev_std = finalize(ev_sum, ev_sumsq, ev_pixels)
    rgb_mean, rgb_std = finalize(rgb_sum, rgb_sumsq, rgb_pixels)

    # Quick printout for convenience
    print("Event mean:", ev_mean)
    print("Event std:", ev_std)
    print("RGB mean:", rgb_mean)
    print("RGB std:", rgb_std)

    return {
        "event": {"mean": ev_mean, "std": ev_std},
        "rgb": {"mean": rgb_mean, "std": rgb_std},
    }

# load_data("/data/storage/jianwen/EventScape/train", num_workers=8)        
# load_data("/data/storage/jianwen/EventScape/valid", num_workers=8)
get_feature("/data/storage/jianwen/EventScape/train")
get_feature("/data/storage/jianwen/EventScape/valid")
# compute_mean_std("/data/storage/jianwen/EventScape/train", num_workers=8)        