from bisect import bisect_left, bisect_right
import csv
import json
import os
import glob
import math
import os
import glob
from pathlib import Path
import random
from typing import Dict, Optional, Tuple, Union, List, Sequence
from PIL import Image
import PIL
import numpy as np
import torch
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset
import imageio.v2 as imageio
import cv2
import yaml
import zipfile

class DSECPairedDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 split = "train",
                 transform = None,
                 event_channels = 3,
                 event_subdir = "eventImage"):
        super().__init__()
        self.image_names = []
        self.event_names = []
        self.event_channels = event_channels
        root = os.path.join(root_dir, f"{split}_images")
        for subfolder in sorted(os.listdir(root)):
            self.image_names.extend(sorted(glob.glob(os.path.join(root, subfolder, "images", "left", "warpped", "*.png"))))
            if self.event_channels == 20:
                # self.event_names.extend(sorted(glob.glob(os.path.join(root, subfolder, "images", "left", event_subdir, "*.pt"))))
                self.event_names.extend(sorted(glob.glob(os.path.join(root, subfolder, "images", "left", "eventToken", "*.pt"))))
            else:
                self.event_names.extend(sorted(glob.glob(os.path.join(root, subfolder, "images", "left", event_subdir, "*.png"))))

        assert len(self.image_names) == len(self.event_names), f"Number of images ({len(self.image_names)}) and events ({len(self.event_names)}) must match in {root}"
        self.transform = transform
    
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx: int):
        if self.event_channels == 20:
            try:
                event = torch.load(self.event_names[idx], map_location="cpu", weights_only=True)
            except Exception as e:
                print(f"Error loading file: {self.event_names[idx]}")
                raise e
        else:
            event = Image.open(self.event_names[idx])
            
        image = Image.open(self.image_names[idx])
        if self.transform is not None:
            event, image = self.transform(event, image)
        return event, image

class SCAPEPairedDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 split = "train",
                 transform = None,):
        super().__init__()
        self.image_names = []
        self.event_names = []

        scape_root = f"{root_dir}/{split}"
        for cluster in sorted(os.listdir(scape_root)):
            cluster_path = os.path.join(scape_root, cluster)
            for sequence in sorted(os.listdir(cluster_path)):
                sequence_path = os.path.join(cluster_path, sequence)
                self.image_names.extend(sorted(glob.glob(os.path.join(sequence_path, "rgb", "data", "*.png"))))
                self.event_names.extend(sorted(glob.glob(os.path.join(sequence_path, "event_images", "*.png"))))
        assert len(self.image_names) == len(self.event_names), "Number of images and events must match"
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx: int):
        event = Image.open(self.event_names[idx])
        image = Image.open(self.image_names[idx])
        if self.transform is not None:
            event, image = self.transform(event, image)
        return event, image

class NIMAPairedDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 split = "train",
                 transform = None,
                 event_channels = 3):
        super().__init__()
        self.image_names = []
        self.event_names = []
        self.event_channels = event_channels
        split = "val" if split == "valid" else split
        root = os.path.join(root_dir, f"extracted_{split}")
        for cls in sorted(os.listdir(root)):
            cls_dir = os.path.join(root, cls)
            for file in sorted(os.listdir(cls_dir)):
                file_path = os.path.join(cls_dir, file)
                if file_path.endswith(".npz"):
                    continue
                
                if self.event_channels == 20:
                    if file_path.endswith(".pt"):
                         # Check if corresponding image exists (assumes .JPEG)
                         base_name = os.path.splitext(file)[0]
                         img_path = os.path.join(cls_dir, base_name + ".JPEG")
                         if os.path.exists(img_path):
                             self.event_names.append(file_path)
                             self.image_names.append(img_path)
                else:
                    if file_path.endswith(".png"):
                        self.event_names.append(file_path)
                    if file_path.endswith(".JPEG"):
                        self.image_names.append(file_path)
                        
        assert len(self.image_names) == len(self.event_names), f"Number of images ({len(self.image_names)}) and events ({len(self.event_names)}) must match"
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx: int):
        if self.event_channels == 20:
            try:
                event = torch.load(self.event_names[idx], map_location="cpu", weights_only=True)
            except Exception as e:
                print(f"Error loading file: {self.event_names[idx]}")
                raise e
            # event is (20, H, W) tensor
        else:
            event = Image.open(self.event_names[idx])
            
        image = Image.open(self.image_names[idx]).convert("RGB")
        if self.transform is not None:
            event, image = self.transform(event, image)
        return event, image

class BDDPairedDataset(Dataset):
    """Paired BDD dataset reading from `paired/{split}`.

    - .png files are event images
    - .jpg files are RGB images
    - .h5 files (raw events) are ignored
    Pairs are matched by basename (without extension).
    """
    def __init__(self, root_dir: str, split: str = "train", transform=None):
        super().__init__()
        split = "val" if split == "valid" else split
        assert split in ["train", "val"], "split must be 'train' or 'val'"
        split_dir = os.path.join(root_dir, split)

        # Collect by basename
        img_paths = glob.glob(os.path.join(split_dir, "*.jpg"))
        evt_paths = glob.glob(os.path.join(split_dir, "*.png"))

        img_map = {os.path.splitext(os.path.basename(p))[0]: p for p in img_paths}
        evt_map = {os.path.splitext(os.path.basename(p))[0]: p for p in evt_paths}

        keys = sorted(set(img_map.keys()) & set(evt_map.keys()))
        self.image_names = [img_map[k] for k in keys]
        self.event_names = [evt_map[k] for k in keys]

        assert len(self.image_names) == len(self.event_names), "Number of images and events must match"
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx: int):
        event = Image.open(self.event_names[idx])
        image = Image.open(self.image_names[idx]).convert("RGB")
        if self.transform is not None:
            event, image = self.transform(event, image)
        return event, image

class DDD17PairedDataset(Dataset):
    def __init__(self,
                 split = "train",
                 transform = None,):
        super().__init__()
        assert split in ["train", "valid"], "split must be either 'train' or 'test'"

        self.image_names = []
        self.event_names = []
        if split == "train":
            nums = [0, 1, 3, 4, 6, 7]
        else:
            nums = [5]
        
        root = "/data/storage/jianwen/ddd17_seg/data"
        for num in nums:
            input_root = os.path.join(root, f"dir{num}/imgs")
            for file in sorted(os.listdir(input_root)):
                name = os.path.splitext(file)[0].split(".")[0][-8:]
                if num in [0, 1]:
                    image_dir = os.path.join(input_root, f"img_{name}.png")
                else:
                    image_dir = os.path.join(input_root, f"00{name}.png")
                event_dir = os.path.join(input_root, f"img_{name}.jpg")
                if not os.path.exists(image_dir):
                    continue
                self.event_names.append(event_dir)
                self.image_names.append(image_dir)

        assert len(self.image_names) == len(self.event_names), "Number of images and events must match"
        self.transform = transform
    
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx: int):
        event = Image.open(self.event_names[idx])
        event_np = np.array(event)
        image = Image.open(self.image_names[idx])
        
        
        if self.transform is not None:
            event, image = self.transform(event, image)
        return event, image
    
class NIMAClsDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 split = "train",
                 modality = "image",
                 transform = None,):
        super().__init__()
        assert split in ["train", "valid"]
        split = "val" if split == "valid" else split
        self.image_names = []
        self.event_names = []
        self.cls = []
        self.modality = modality
        root = os.path.join(root_dir, f"extracted_{split}")
        if split == "train":
            i = 0
            for i, cls in enumerate(os.listdir(root)):
                cls_dir = os.path.join(root, cls)
                for file in sorted(os.listdir(cls_dir)):
                    file_path = os.path.join(cls_dir, file)
                    if file_path.endswith(".npz"):
                        continue
                    if file_path.endswith(".png"):
                        self.event_names.append(file_path)
                    if file_path.endswith(".JPEG"):
                        self.image_names.append(file_path)
                        self.cls.append(i)

        elif split == "val":
            for i, cls in enumerate(os.listdir(root)):
                cls_dir = os.path.join(root, cls)
                for file in sorted(os.listdir(cls_dir)):
                    file_path = os.path.join(cls_dir, file)
                    if file_path.endswith(".npz"):
                        continue
                    if file_path.endswith(".png"):
                        self.event_names.append(file_path)
                    if file_path.endswith(".JPEG"):
                        self.image_names.append(file_path)
                        self.cls.append(i)
        assert len(self.image_names) == len(self.event_names), "Number of images and events must match"
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx: int):
        if self.modality == "event":
            data = Image.open(self.event_names[idx])
        elif self.modality == "image":
            data = Image.open(self.image_names[idx]).convert("RGB")
        if self.transform is not None:
            data = self.transform(data)
        label = self.cls[idx]
        return data, label

class NCALClsDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 split = "train",
                 modality = "image",
                 transform = None,):
        super().__init__()
        assert split in ["train", "valid"], "split must be either 'train' or 'valid'"
        assert modality in ["event", "image"], "modality must be either 'event' or 'image'"
        self.modality = modality
        self.transform = transform
        self.image_names = []
        self.event_names = []
        self.cls = []

        split_dir = os.path.join(root_dir, split)
        classes = [d for d in sorted(os.listdir(split_dir)) if os.path.isdir(os.path.join(split_dir, d))]
        self.class_names = classes
        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(split_dir, class_name)
            image_map = {Path(p).stem: p for p in glob.glob(os.path.join(class_dir, "*.jpg"))}
            event_map = {Path(p).stem: p for p in glob.glob(os.path.join(class_dir, "*.png"))}
            common_keys = sorted(set(image_map.keys()) & set(event_map.keys()))
            for key in common_keys:
                self.image_names.append(image_map[key])
                self.event_names.append(event_map[key])
                self.cls.append(class_idx)

        assert len(self.image_names) == len(self.event_names), "Number of images and events must match"

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx: int):
        if self.modality == "event":
            data = Image.open(self.event_names[idx])
        elif self.modality == "image":
            data = Image.open(self.image_names[idx]).convert("RGB")
        if self.transform is not None:
            data = self.transform(data)
        return data, self.cls[idx]

class EventDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 split = "train",
                 transform = None,):
        super().__init__()
        self.events = []
        self.split = split
        root = os.path.join(root_dir, f"{split}_images")
        for subfolder in sorted(os.listdir(root)):
            self.events.extend(sorted(glob.glob(os.path.join(root, subfolder, "images", "left", "eventImage", "*.png"))))
        self.transform = transform
    
    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx: int):
        data = Image.open(self.events[idx])
        data = self.transform(data) if self.transform is not None else data
        return data

class ImageDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 split = "train",
                 transform = None,):
        super().__init__()
        self.images = []
        self.split = split
        root = os.path.join(root_dir, f"{split}_images")
        for subfolder in sorted(os.listdir(root)):
            self.images.extend(sorted(glob.glob(os.path.join(root, subfolder, "images", "left", "warpped", "*.png"))))
        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        data = Image.open(self.images[idx])
        data = self.transform(data) if self.transform is not None else data
        return data

class DSECSegmentDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 split = "train",
                 C = 11,
                 data = "event",
                 transform = None,):
        super().__init__()
        assert data in ["event", "image"], "data must be either 'event' or 'image'"
        assert split in ["train", "test"], "split must be either 'train' or 'test'"

        self.input_names = []
        self.label_names = []
        input_root = os.path.join(root_dir, f"{split}_images")
        seman_root = os.path.join(root_dir, f"{split}_semantic_segmentation/{split}")
        for subfolder in sorted(os.listdir(seman_root)):
            data_type = "eventImage" if data == "event" else "warpped"
            self.input_names.extend(sorted(glob.glob(os.path.join(input_root, subfolder, "images", "left", data_type, "*.png"))))
            self.label_names.extend(sorted(glob.glob(os.path.join(seman_root, subfolder, f"{C}classes", "*.png"))))

        if len(self.label_names) >= len(self.input_names): # hack to allow less data frames (due to cut videos)
            self.label_names = self.label_names[:len(self.input_names)]

        assert len(self.input_names) == len(self.label_names), f"Number of images ({len(self.input_names)}) and labels ({len(self.label_names)}) must match"
        self.transform = transform
    
    def __len__(self):
        return len(self.input_names)

    def __getitem__(self, idx: int):
        input = Image.open(self.input_names[idx])
        label = Image.open(self.label_names[idx])
        
        if self.transform is not None:
            input, label = self.transform(input, label)
        return input, label

class DSECSegmentSequenceDataset(Dataset):
    """
    DSEC segmentation dataset that returns consecutive event frames with a label.

    Each item yields a two-frame tensor: index 0 is the current event frame, index 1
    is the next event frame in the sequence. The segmentation label corresponds to
    the current frame.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        C: int = 11,
        frames_per_sample: int = 2,
        transform=None,
    ):
        super().__init__()
        assert split in ["train", "test"], "split must be either 'train' or 'test'"
        if frames_per_sample != 2:
            raise ValueError("frames_per_sample must be 2 (current + next event)")

        self.transform = transform
        self.frames_per_sample = frames_per_sample
        self.modalities = ("event", "event")

        self.event_curr_paths: List[str] = []
        self.event_next_paths: List[str] = []
        self.label_paths: List[str] = []

        input_root = os.path.join(root_dir, f"{split}_images")
        seman_root = os.path.join(root_dir, f"{split}_semantic_segmentation/{split}")

        for subfolder in sorted(os.listdir(seman_root)):
            event_dir = os.path.join(input_root, subfolder, "images", "left", "eventImage")
            label_dir = os.path.join(seman_root, subfolder, f"{C}classes_renamed")

            event_files = glob.glob(os.path.join(event_dir, "*.png"))
            label_files = glob.glob(os.path.join(label_dir, "*.png"))

            if not event_files or not label_files:
                continue

            event_map = {Path(path).stem: path for path in event_files}
            label_map = {Path(path).stem: path for path in label_files}

            common_keys = sorted(set(event_map) & set(label_map))
            if len(common_keys) < 2:
                continue

            for idx in range(len(common_keys) - 1):
                curr_key = common_keys[idx]
                next_key = common_keys[idx + 1]
                self.event_curr_paths.append(event_map[curr_key])
                self.event_next_paths.append(event_map[next_key])
                self.label_paths.append(label_map[curr_key])

        if len(self.label_paths) == 0:
            raise RuntimeError(f"No paired samples found in {root_dir} for split '{split}'")

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, idx: int):
        event_curr = Image.open(self.event_curr_paths[idx])
        event_next = Image.open(self.event_next_paths[idx])
        label = Image.open(self.label_paths[idx])

        frames = [event_curr, event_next]
        if self.transform is not None:
            frames, label = self.transform(frames, label)
        else:
            frames = [
                torch.tensor(np.array(frame)).permute(2, 0, 1).float() / 255.0
                for frame in frames
            ]
            frames = torch.stack(frames, dim=0)
            label = torch.tensor(np.array(label), dtype=torch.int64)
        return frames, label

class DSECECDDPEventDataset(Dataset):
    """DSEC segmentation dataset that reads pre-generated ECDDP event tensors.

    Each event tensor is produced by ``src/pre_dse_ecddp.py`` (or the original
    ECDDP preprocessing code) and stored under ``images/left/<subdir>`` next to
    the warped RGB frames.  This loader pairs those tensors with the
    {C}classes_renamed semantic labels so that we can fine-tune the Swin encoder
    inside Event-Camera-Data-Dense-Pre-training.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        C: int = 11,
        tensor_subdir: str = "eventTensor_ecddp",
        tensor_exts: Optional[Sequence[str]] = None,
        transform=None,
    ):
        super().__init__()
        assert split in ["train", "test"], "split must be 'train' or 'test'"
        self.root_dir = Path(root_dir)
        self.split = split
        self.C = C
        self.tensor_subdir = tensor_subdir
        if tensor_exts is None:
            tensor_exts = (".pt", ".npz", ".npy")
        self.tensor_exts = tuple(ext if ext.startswith(".") else f".{ext}" for ext in tensor_exts)
        self.tensor_exts_lower = tuple(ext.lower() for ext in self.tensor_exts)
        self.transform = transform
        self.samples: List[Tuple[Path, Path]] = []
        self.event_channels: Optional[int] = None
        self.event_hw: Optional[Tuple[int, int]] = None

        input_root = self.root_dir / f"{split}_images"
        seman_root = self.root_dir / f"{split}_semantic_segmentation" / split
        if not seman_root.exists():
            raise FileNotFoundError(f"Missing semantic directory: {seman_root}")

        for subfolder in sorted(p.name for p in seman_root.iterdir() if p.is_dir()):
            self._collect_sequence(input_root, seman_root, subfolder)

        if not self.samples:
            raise RuntimeError(
                f"No paired event tensors/labels found under {root_dir} for split '{split}'. "
                f"Expected tensors inside '{tensor_subdir}'."
            )

    def _collect_sequence(self, input_root: Path, seman_root: Path, sequence: str) -> None:
        event_dir = input_root / sequence / "images" / "left" / self.tensor_subdir
        label_dir = seman_root / sequence / f"{self.C}classes_renamed"
        timestamp_path = input_root / sequence / "images" / "timestamps.txt"
        if not event_dir.exists() or not label_dir.exists():
            return

        timestamp_map = self._load_timestamp_map(timestamp_path)
        if not timestamp_map:
            return

        event_index_map = self._index_event_files(event_dir)
        if not event_index_map:
            return

        label_map = {path.stem: path for path in label_dir.glob("*.png")}

        matched = 0
        for ts_key, label_path in label_map.items():
            idx = timestamp_map.get(ts_key)
            if idx is None:
                continue
            event_path = event_index_map.get(idx)
            if event_path is None:
                continue
            self.samples.append((event_path, label_path))
            matched += 1

        if matched == 0:
            return

        self._try_register_meta(event_dir, self.samples[-1][0])

    def _index_event_files(self, event_dir: Path) -> Dict[int, Path]:
        mapping: Dict[int, Path] = {}
        for path in event_dir.iterdir():
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in self.tensor_exts_lower:
                continue
            stem = path.stem
            try:
                idx = int(stem)
            except ValueError:
                continue
            mapping[idx] = path
        return mapping

    def _load_timestamp_map(self, timestamp_path: Path) -> Optional[Dict[str, int]]:
        if not timestamp_path.exists():
            return None
        timestamps = np.loadtxt(timestamp_path, dtype=np.int64)
        if timestamps.ndim == 0:
            timestamps = np.asarray([int(timestamps)], dtype=np.int64)
        return {str(int(ts)): idx for idx, ts in enumerate(timestamps.tolist())}

    def _try_register_meta(self, event_dir: Path, sample_path: Path) -> None:
        if self.event_channels is not None and self.event_hw is not None:
            return
        meta_path = event_dir / "ecddp_config.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                channels = meta.get("nr_events_data") * meta.get("nr_bins_per_data")
                if channels:
                    self.event_channels = int(channels)
                resize_shape = meta.get("resize_shape")
                if isinstance(resize_shape, (list, tuple)) and len(resize_shape) == 2:
                    self.event_hw = (int(resize_shape[0]), int(resize_shape[1]))
            except (OSError, ValueError, TypeError):
                pass
        if self.event_channels is None or self.event_hw is None:
            tensor = self._load_event_tensor(sample_path)
            self.event_channels = int(tensor.shape[0])
            self.event_hw = (int(tensor.shape[-2]), int(tensor.shape[-1]))

    def _load_event_tensor(self, path: Path) -> torch.Tensor:
        suffix = path.suffix.lower()
        if suffix == ".pt":
            payload = torch.load(path, map_location="cpu")
            if isinstance(payload, torch.Tensor):
                tensor = payload
            elif isinstance(payload, dict):
                tensor = None
                for key in ("tensor", "event_tensor", "events"):
                    if key in payload:
                        tensor = payload[key]
                        break
                if tensor is None:
                    raise KeyError(f"No tensor entry found in {path}")
            else:
                raise TypeError(f"Unsupported payload type in {path}: {type(payload)}")
        elif suffix == ".npz":
            data = np.load(path)
            array = data["tensor"] if "tensor" in data else data[data.files[0]]
            tensor = torch.from_numpy(array)
        elif suffix == ".npy":
            array = np.load(path)
            tensor = torch.from_numpy(array)
        else:
            raise ValueError(f"Unsupported tensor format: {path.suffix}")
        if tensor.ndim != 3:
            raise ValueError(f"Event tensor at {path} must be 3D (C,H,W), got {tensor.shape}")
        return tensor.float().contiguous()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        event_path, label_path = self.samples[idx]
        event = self._load_event_tensor(event_path)
        label = Image.open(label_path)
        label = torch.tensor(np.array(label), dtype=torch.int64)
        if label.ndim == 3:
            label = label.squeeze()
        target_hw = event.shape[-2:]
        if label.shape != target_hw:
            label = TF.resize(
                label.unsqueeze(0),
                target_hw,
                interpolation=InterpolationMode.NEAREST,
            ).squeeze(0).to(torch.int64)
        if self.transform is not None:
            event, label = self.transform(event, label)
        return event, label

class DDD17SegmentDataset(Dataset):
    def __init__(self,
                 split = "train",
                 C = 6,
                 data = "event",
                 transform = None,):
        super().__init__()
        assert data in ["event", "image"], "data must be either 'event' or 'image'"
        assert split in ["train", "valid"], "split must be either 'train' or 'test'"

        self.input_names = []
        self.label_names = []
        if split == "train":
            nums = [0, 3, 4, 6, 7]
        else:
            nums = [1]
        if data == "event":
            ending = "jpg"
        else:
            ending = "png"
        
        root = "/data/storage/jianwen/ddd17_seg/data"
        for num in nums:
            input_root = os.path.join(root, f"dir{num}/imgs")
            seman_root = os.path.join(root, f"dir{num}/segmentation_masks")
            for file in sorted(os.listdir(seman_root)):
                name = os.path.splitext(file)[0].split(".")[0][-8:]
                if data == "image":
                    if num in [0, 1]:
                        input_dir = os.path.join(input_root, f"img_{name}.{ending}")
                    else:
                        input_dir = os.path.join(input_root, f"00{name}.{ending}")
                else:
                    input_dir = os.path.join(input_root, f"img_{name}.{ending}") 
                if not os.path.exists(input_dir):
                    continue
                self.label_names.append(os.path.join(seman_root, file))
                self.input_names.append(input_dir)

        assert len(self.input_names) == len(self.label_names), "Number of images and events must match"
        self.transform = transform
    
    def __len__(self):
        return len(self.input_names)

    def __getitem__(self, idx: int):
        data = Image.open(self.input_names[idx])
        label = Image.open(self.label_names[idx])
        
        if self.transform is not None:
            data, label = self.transform(data, label)
        return data, label

def decode_dsec_flow(flow_png_path: str):
    flow_16bit_bgr = cv2.imread(flow_png_path, cv2.IMREAD_UNCHANGED)
    flow_16bit = cv2.cvtColor(flow_16bit_bgr, cv2.COLOR_BGR2RGB)

    assert flow_16bit.dtype == np.uint16, f"光流图像 {os.path.basename(flow_png_path)} 的数据类型不是 uint16, 而是 {flow_16bit.dtype}"

    flow_float = flow_16bit.astype(np.float32)

    valid_mask = flow_float[:, :, 2] > 0

    flow_u = (flow_float[:, :, 0] - 32768.0) / 128.0
    flow_v = (flow_float[:, :, 1] - 32768.0) / 128.0
    
    flow = np.stack([flow_u, flow_v, valid_mask], axis=-1)

    return flow

class DSECOpticalDataset(Dataset):
    def __init__(self,
                 root_dir = "/data/storage/jianwen/DSEC/",
                 valid_sequence = "zurich_city_11_c",
                 split = "train",
                 data = "event",
                 transform = None,):
        super().__init__()
        self.input_names = []
        self.label_names = []
        self.transform = transform

        optical_root = os.path.join(root_dir, "train_optical_flow")
        event_root = os.path.join(root_dir, "train_images")

        all_sequences = sorted(os.listdir(optical_root))
        if split == "train":
            sequences_to_load = [seq for seq in all_sequences if seq != valid_sequence]
        elif split == "valid":
            sequences_to_load = [valid_sequence]
        else: # test or all
            sequences_to_load = all_sequences

        for sequence in sequences_to_load:
            if split == "train" and sequence == valid_sequence:
                continue
            optical_sequence_path = os.path.join(optical_root, sequence, "flow", "forward")
            data_type = "eventImage" if data == "event" else "warpped"
            event_sequence_path = os.path.join(event_root, sequence, "images", "left", data_type)
            for i, frame in enumerate(sorted(os.listdir(optical_sequence_path))):
                self.label_names.append(os.path.join(optical_sequence_path, frame))
                self.input_names.append(os.path.join(event_sequence_path, frame))
        
        assert len(self.input_names) == len(self.label_names), "Number of images and events must match"
    
    def __len__(self):
        return len(self.input_names)
    
    def __getitem__(self, idx: int):
        input = Image.open(self.input_names[idx])
        label = decode_dsec_flow(self.label_names[idx])
        
        if self.transform is not None:
            input, label = self.transform(input, label)
        return input, label

def _infer_mvsec_scenario(sequence: str) -> Optional[str]:
    seq_lower = sequence.lower()
    if "outdoor_day" in seq_lower:
        return "outdoor_day"
    if "outdoor_night" in seq_lower:
        return "outdoor_night"
    return None


class MVSECDepthDataset(Dataset):
    """
    MVSEC depth dataset that provides paired event/RGB frames with aligned depth maps.

    Each sample returns a dictionary with:
        - event : HxW event frame (PIL.Image during loading, tensor after transform)
        - image : HxW RGB frame
        - depth : float32 depth map in meters
        - mask  : float32 validity mask (1 when depth is valid)
        - meta  : metadata (sequence name, frame id, timestamps)
    """

    DEFAULT_SPLITS: Dict[str, List[str]] = {
        "train": ["outdoor_day1", "outdoor_night1", "outdoor_night2"],
        "valid": ["outdoor_night3"],
        "test": ["outdoor_night3"],
    }

    def __init__(
        self,
        root_dir: str = "/data/storage/jianwen/mvsec",
        split: str = "train",
        sequences: Optional[Sequence[str]] = None,
        transform=None,
        min_depth: float | None = 0.1,
        max_depth: float | None = 30.0,
        depth_scale: float = 1.0 / 100.0,
        calibration_root: str | None = "/data/storage/jianwen/MVSEC",
    ):
        super().__init__()
        root = Path(root_dir)
        if sequences is None:
            split = split.lower()
            if split not in self.DEFAULT_SPLITS:
                available = sorted([p.name for p in root.iterdir() if p.is_dir()])
                sequences = available
            else:
                sequences = self.DEFAULT_SPLITS[split]
        self.sequences = list(sequences)
        self.transform = transform
        self.min_depth = float(min_depth) if min_depth is not None else None
        self.max_depth = float(max_depth) if max_depth is not None else None
        self.depth_scale = float(depth_scale)
        self.calibration_root = Path(calibration_root) if calibration_root else None
        self._calibration_cache: Dict[str, Dict[str, float]] = {}

        self.samples: List[Dict[str, Union[str, int, float]]] = []

        for seq in self.sequences:
            seq_root = root / seq
            if not seq_root.exists():
                raise FileNotFoundError(f"Sequence {seq} not found under {root}")
            csv_path = seq_root / "index.csv"
            rgb_root = seq_root / "rgb"
            event_root = seq_root / "events"
            depth_root = seq_root / "depth"

            if csv_path.exists():
                with csv_path.open("r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        frame_id = int(row.get("frame_id", len(self.samples)))
                        rgb_path = seq_root / row["rgb_path"]
                        event_path = seq_root / row["event_path"]
                        depth_path = seq_root / row["depth_path"]
                        if not (rgb_path.exists() and event_path.exists() and depth_path.exists()):
                            continue
                        self.samples.append(
                            {
                                "sequence": seq,
                                "frame_id": frame_id,
                                "rgb_path": str(rgb_path),
                                "event_path": str(event_path),
                                "depth_path": str(depth_path),
                                "depth_timestamp": float(row.get("depth_timestamp", 0.0)),
                                "image_timestamp": float(row.get("image_timestamp", 0.0)),
                                "event_start_time": float(row.get("event_start_time", 0.0)),
                                "event_end_time": float(row.get("event_end_time", 0.0)),
                            }
                        )
            else:
                # Fall back to matching filenames by stem
                rgb_files = sorted(rgb_root.glob("*.png"))
                for rgb_path in rgb_files:
                    stem = rgb_path.stem
                    event_path = event_root / f"{stem}.png"
                    depth_path = depth_root / f"{stem}.png"
                    if not (event_path.exists() and depth_path.exists()):
                        continue
                    self.samples.append(
                        {
                            "sequence": seq,
                            "frame_id": int(stem),
                            "rgb_path": str(rgb_path),
                            "event_path": str(event_path),
                            "depth_path": str(depth_path),
                            "depth_timestamp": 0.0,
                            "image_timestamp": 0.0,
                            "event_start_time": 0.0,
                            "event_end_time": 0.0,
                        }
                    )

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found for MVSEC depth dataset under {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample_meta = self.samples[idx]
        event_img = Image.open(sample_meta["event_path"]).convert("RGB")
        rgb_img = Image.open(sample_meta["rgb_path"]).convert("RGB")
        depth_map = imageio.imread(sample_meta["depth_path"]).astype(np.float32)

        valid_mask = depth_map > 0
        depth_map = depth_map * self.depth_scale
        if self.min_depth is not None:
            np.maximum(depth_map, self.min_depth, out=depth_map)
        if self.max_depth is not None:
            np.minimum(depth_map, self.max_depth, out=depth_map)

        sample = {
            "event": event_img,
            "image": rgb_img,
            "depth": depth_map,
            "mask": valid_mask.astype(np.float32),
            "meta": sample_meta,
        }

        intrinsics = self._intrinsics_for_sequence(sample_meta["sequence"])
        if intrinsics is not None:
            sample["intrinsics"] = intrinsics

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _intrinsics_for_sequence(self, sequence: str) -> Optional[Dict[str, float]]:
        scenario = _infer_mvsec_scenario(sequence)
        if scenario is None or self.calibration_root is None:
            return None
        if scenario not in self._calibration_cache:
            self._calibration_cache[scenario] = self._load_scenario_intrinsics(scenario)
        base = self._calibration_cache[scenario]
        return dict(base)

    def _load_scenario_intrinsics(self, scenario: str) -> Dict[str, float]:
        scenario_dir = self.calibration_root / scenario
        if not scenario_dir.exists():
            raise FileNotFoundError(f"Calibration directory '{scenario_dir}' not found for scenario '{scenario}'.")
        yaml_name = f"camchain-imucam-{scenario}.yaml"
        yaml_path = scenario_dir / "carlib" / yaml_name
        if yaml_path.exists():
            with yaml_path.open("r") as fp:
                data = yaml.safe_load(fp)
        else:
            zip_path = scenario_dir / f"{scenario}_calib.zip"
            if not zip_path.exists():
                raise FileNotFoundError(
                    f"Neither calibration YAML nor zip archive found for scenario '{scenario}' under {scenario_dir}."
                )
            with zipfile.ZipFile(zip_path, "r") as zf:
                matches = [name for name in zf.namelist() if name.endswith(yaml_name)]
                if not matches:
                    raise FileNotFoundError(f"Could not find '{yaml_name}' inside {zip_path}.")
                with zf.open(matches[0], "r") as fp:
                    data = yaml.safe_load(fp)
        cam0 = data.get("cam0")
        if cam0 is None:
            raise KeyError(f"'cam0' entry missing in calibration for scenario '{scenario}'.")
        intr = cam0.get("intrinsics")
        res = cam0.get("resolution")
        if intr is None or res is None or len(intr) < 4 or len(res) < 2:
            raise ValueError(f"Incomplete intrinsics for scenario '{scenario}'.")
        return {
            "fx": float(intr[0]),
            "fy": float(intr[1]),
            "cx": float(intr[2]),
            "cy": float(intr[3]),
            "width": float(res[0]),
            "height": float(res[1]),
        }


class MVSECECDDPDepthDataset(MVSECDepthDataset):
    """
    MVSEC depth dataset variant that loads precomputed ECDDP event tensors
    (e.g., generated via `pre_dse_ecddp.py`) instead of RGB-style event
    projections. Each tensor is expected to be stored under
    `<sequence>/<tensor_subdir>/<frame_id>.(pt|npz|npy)` and will replace
    the default PNG event frame.
    """

    def __init__(
        self,
        *,
        root_dir: str = "/data/storage/jianwen/mvsec",
        split: str = "train",
        sequences: Optional[Sequence[str]] = None,
        transform=None,
        min_depth: float | None = 0.1,
        max_depth: float | None = 30.0,
        depth_scale: float = 1.0 / 100.0,
        calibration_root: str | None = "/data/storage/jianwen/MVSEC",
        tensor_root: str | None = None,
        tensor_subdir: str = "eventTensor_ecddp",
        tensor_exts: Sequence[str] = (".pt", ".npz", ".npy"),
    ):
        self.ecddp_tensor_root = Path(tensor_root) if tensor_root is not None else Path(root_dir)
        self.ecddp_tensor_subdir = tensor_subdir
        self.ecddp_tensor_exts = tuple(
            ext if ext.startswith(".") else f".{ext}" for ext in tensor_exts
        )
        self.event_channels: Optional[int] = None
        self.event_hw: Optional[Tuple[int, int]] = None
        super().__init__(
            root_dir=root_dir,
            split=split,
            sequences=sequences,
            transform=transform,
            min_depth=min_depth,
            max_depth=max_depth,
            depth_scale=depth_scale,
            calibration_root=calibration_root,
        )
        self._ecddp_lookup: Dict[Tuple[str, int], Path] = {}
        self._build_ecddp_lookup()

    def _build_ecddp_lookup(self):
        for sample in self.samples:
            seq = sample["sequence"]
            frame_id = int(sample["frame_id"])
            path = self._resolve_tensor_path(seq, frame_id)
            if path is None:
                raise FileNotFoundError(
                    f"Missing ECDDP tensor for {seq}/{frame_id} under "
                    f"{self.ecddp_tensor_root}/{seq}/{self.ecddp_tensor_subdir}"
                )
            self._ecddp_lookup[(seq, frame_id)] = path
        if not self._ecddp_lookup:
            raise RuntimeError("No ECDDP tensors found for MVSEC dataset.")
        first_tensor = self._load_tensor(next(iter(self._ecddp_lookup.values())))
        self.event_channels = int(first_tensor.shape[0])
        self.event_hw = (int(first_tensor.shape[-2]), int(first_tensor.shape[-1]))

    def _resolve_tensor_path(self, sequence: str, frame_id: int) -> Path | None:
        seq_dir = self.ecddp_tensor_root / sequence / self.ecddp_tensor_subdir
        if not seq_dir.exists():
            return None
        candidates = [
            f"{frame_id}",
            f"{frame_id:06d}",
            f"{frame_id:010d}",
        ]
        for stem in candidates:
            for ext in self.ecddp_tensor_exts:
                candidate = seq_dir / f"{stem}{ext}"
                if candidate.exists():
                    return candidate
        return None

    def _load_tensor(self, path: Path) -> torch.Tensor:
        suffix = path.suffix.lower()
        if suffix == ".pt":
            payload = torch.load(path, map_location="cpu")
            if isinstance(payload, torch.Tensor):
                tensor = payload
            elif isinstance(payload, dict):
                tensor = None
                for key in ("tensor", "event_tensor", "events"):
                    if key in payload:
                        tensor = payload[key]
                        break
                if tensor is None:
                    raise KeyError(f"No tensor entry found in {path}")
            else:
                raise TypeError(f"Unsupported payload type in {path}: {type(payload)}")
        elif suffix == ".npz":
            data = np.load(path)
            key = "tensor" if "tensor" in data.files else data.files[0]
            tensor = torch.from_numpy(data[key])
        elif suffix == ".npy":
            tensor = torch.from_numpy(np.load(path))
        else:
            raise ValueError(f"Unsupported tensor format: {path.suffix}")
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError(f"ECDDP tensor must be 3D (C,H,W), got {tensor.shape}")
        return tensor.float().contiguous()

    def __getitem__(self, idx: int):
        sample_meta = self.samples[idx]
        seq = sample_meta["sequence"]
        frame_id = int(sample_meta["frame_id"])
        event_tensor = self._load_tensor(self._ecddp_lookup[(seq, frame_id)])

        rgb_img = Image.open(sample_meta["rgb_path"]).convert("RGB")
        depth_map = imageio.imread(sample_meta["depth_path"]).astype(np.float32)

        valid_mask = depth_map > 0
        depth_map = depth_map * self.depth_scale
        if self.min_depth is not None:
            np.maximum(depth_map, self.min_depth, out=depth_map)
        if self.max_depth is not None:
            np.minimum(depth_map, self.max_depth, out=depth_map)

        sample = {
            "event": event_tensor,
            "image": rgb_img,
            "depth": depth_map,
            "mask": valid_mask.astype(np.float32),
            "meta": sample_meta,
        }

        intrinsics = self._intrinsics_for_sequence(sample_meta["sequence"])
        if intrinsics is not None:
            sample["intrinsics"] = intrinsics

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _intrinsics_for_sequence(self, sequence: str) -> Optional[Dict[str, float]]:
        scenario = _infer_mvsec_scenario(sequence)
        if scenario is None or self.calibration_root is None:
            return None
        if scenario not in self._calibration_cache:
            self._calibration_cache[scenario] = self._load_scenario_intrinsics(scenario)
        base = self._calibration_cache[scenario]
        return dict(base)  # return a copy per sample

    def _load_scenario_intrinsics(self, scenario: str) -> Dict[str, float]:
        scenario_dir = self.calibration_root / scenario
        if not scenario_dir.exists():
            raise FileNotFoundError(f"Calibration directory '{scenario_dir}' not found for scenario '{scenario}'.")
        yaml_name = f"camchain-imucam-{scenario}.yaml"
        yaml_path = scenario_dir / "carlib" / yaml_name
        if yaml_path.exists():
            with yaml_path.open("r") as fp:
                data = yaml.safe_load(fp)
        else:
            zip_path = scenario_dir / f"{scenario}_calib.zip"
            if not zip_path.exists():
                raise FileNotFoundError(
                    f"Neither calibration YAML nor zip archive found for scenario '{scenario}' under {scenario_dir}."
                )
            with zipfile.ZipFile(zip_path, "r") as zf:
                matches = [name for name in zf.namelist() if name.endswith(yaml_name)]
                if not matches:
                    raise FileNotFoundError(f"Could not find '{yaml_name}' inside {zip_path}.")
                with zf.open(matches[0], "r") as fp:
                    data = yaml.safe_load(fp)
        cam0 = data.get("cam0")
        if cam0 is None:
            raise KeyError(f"'cam0' entry missing in calibration for scenario '{scenario}'.")
        intr = cam0.get("intrinsics")
        res = cam0.get("resolution")
        if intr is None or res is None or len(intr) < 4 or len(res) < 2:
            raise ValueError(f"Incomplete intrinsics for scenario '{scenario}'.")
        return {
            "fx": float(intr[0]),
            "fy": float(intr[1]),
            "cx": float(intr[2]),
            "cy": float(intr[3]),
            "width": float(res[0]),
            "height": float(res[1]),
        }


class EventScapeDepthDataset(Dataset):
    """
    EventScape depth dataset loading event/RGB frames with dense depth maps.

    Directory layout (per sequence):
        event_images/*.png         -- event representations (RGB)
        rgb/data/*.png             -- RGB frames
        depth/data/*.npy           -- depth maps in meters (float64)
        depth/frames/*.png         -- optional depth visualization fallback
    """

    def __init__(
        self,
        root_dir: str = "/data/storage/jianwen/EventScape",
        split: str = "train",
        transform=None,
        min_depth: float | None = 1.0,
        max_depth: float | None = 600.0,
        depth_scale: float = 1.0,
        invalid_depth_value: float = 1000.0,
    ):
        super().__init__()
        root = Path(root_dir) / split
        if not root.exists():
            raise FileNotFoundError(f"EventScape split '{split}' not found under {root_dir}")

        self.transform = transform
        self.min_depth = float(min_depth) if min_depth is not None else None
        self.max_depth = float(max_depth) if max_depth is not None else None
        self.depth_scale = float(depth_scale)
        self.invalid_depth_value = float(invalid_depth_value)

        self.samples: list[dict[str, Union[str, int]]] = []

        cluster_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
        for cluster_path in cluster_dirs:
            sequence_dirs = sorted([p for p in cluster_path.iterdir() if p.is_dir()])
            for sequence_path in sequence_dirs:
                event_dir = sequence_path / "event_images"
                rgb_dir = sequence_path / "rgb" / "data"
                depth_data_dir = sequence_path / "depth" / "data"
                depth_frame_dir = sequence_path / "depth" / "frames"

                if not (event_dir.exists() and rgb_dir.exists()):
                    continue

                depth_files = sorted(depth_data_dir.glob("*.npy"))
                depth_format = "npy"
                if not depth_files:
                    depth_files = sorted(depth_frame_dir.glob("*.png"))
                    depth_format = "png"
                if not depth_files:
                    continue

                event_files = sorted(event_dir.glob("*.png"))
                rgb_files = sorted(rgb_dir.glob("*.png"))

                if not (len(event_files) == len(rgb_files) == len(depth_files)):
                    raise RuntimeError(
                        f"Mismatch between modalities in {sequence_path}: "
                        f"{len(event_files)} events, {len(rgb_files)} rgb, {len(depth_files)} depth"
                    )

                for frame_idx, (event_path, rgb_path, depth_path) in enumerate(zip(event_files, rgb_files, depth_files)):
                    self.samples.append(
                        {
                            "cluster": cluster_path.name,
                            "sequence": sequence_path.name,
                            "frame_idx": frame_idx,
                            "event_path": str(event_path),
                            "rgb_path": str(rgb_path),
                            "depth_path": str(depth_path),
                            "depth_format": depth_format,
                        }
                    )

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found for EventScape depth dataset under {root_dir}/{split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        meta = self.samples[idx]
        event_img = Image.open(meta["event_path"]).convert("RGB")
        rgb_img = Image.open(meta["rgb_path"]).convert("RGB")

        depth_format: str = meta["depth_format"]
        depth_path = meta["depth_path"]
        if depth_format == "npy":
            depth_map = np.load(depth_path).astype(np.float32)
        else:
            depth_map = imageio.imread(depth_path).astype(np.float32)

        depth_map *= self.depth_scale

        mask = (depth_map < self.invalid_depth_value).astype(np.float32)
        if self.min_depth is not None:
            np.maximum(depth_map, self.min_depth, out=depth_map)
        if self.max_depth is not None:
            np.minimum(depth_map, self.max_depth, out=depth_map)
        depth_map = depth_map * mask

        sample = {
            "event": event_img,
            "image": rgb_img,
            "depth": depth_map,
            "mask": mask,
            "meta": meta,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class SequenceDataset(Dataset):
    def __init__(self,
        window_size,
        n_tokens_per_image,
        n_embed,
        split = "train",
        modality = "both",
    ) -> None:
        super().__init__()
        self.window_size        = window_size
        self.n                  = n_tokens_per_image
        self.n_embed            = n_embed
        self.modality           = modality
        
        self.token_names, self.accu = {0: None}, 0

        dsec_root = "/data/storage/jianwen/DSEC"
        dsec = os.path.join(dsec_root, f"{split}_images")
        for subfolder in sorted(os.listdir(dsec)):
            warped      = sorted(glob.glob(os.path.join(dsec, subfolder, "images", "left", "imageToken", "*.pt")))
            eventImage  = sorted(glob.glob(os.path.join(dsec, subfolder, "images", "left", "eventToken", "*.pt")))
            assert len(warped) == len(eventImage)
            for i in range(len(warped)):
                if self.modality in ["both", "event"]:
                    self.accu += self.n
                    self.token_names[self.accu] = eventImage[i]
                if self.modality in ["both", "image"]:
                    self.accu += self.n
                    self.token_names[self.accu] = warped[i]
            # self.accu += self.n
            # self.token_names[self.accu] = "EOS"
        self.token_names = dict(sorted(self.token_names.items()))
        self.keys = [k for k, _ in self.token_names.items()]
        self.vals = [v for _, v in self.token_names.items()]

        # bddd        = os.path.join("/data/storage/jianwen/bdd100k/paired", f"{split}")
        # warped      = sorted(glob.glob(os.path.join(bddd, "eventToken", "*.pt")))
        # eventImage  = sorted(glob.glob(os.path.join(bddd, "imageToken", "*.pt")))
        # assert len(warped) == len(eventImage)
        # for i in range(len(warped)):
        #     self.accu += self.n
        #     self.token_names[self.accu] = eventImage[i]
        #     self.accu += 1
        #     self.token_names[self.accu] = "SOS"
        #     self.accu += self.n
        #     self.token_names[self.accu] = warped[i]
        #     self.accu += 1
        #     self.token_names[self.accu] = "SOS"

        # if split == "test":
        #     split = "valid"
        # scap = os.path.join("/data/storage/jianwen/EventScape", f"{split}")
        # for cluster in sorted(os.listdir(scap)):
        #     cluster_dir = os.path.join(scap, cluster)
        #     for sequence in sorted(os.listdir(cluster_dir)):
        #         sequence_dir = os.path.join(cluster_dir, sequence)
        #         warped = sorted(glob.glob(os.path.join(sequence_dir, "image_features", "*.pt")))
        #         eventImage = sorted(glob.glob(os.path.join(sequence_dir, "event_features", "*.pt")))
        #         assert len(warped) == len(eventImage)
        #         for i in range(len(warped)):
        #             self.accu += self.n
        #             self.token_names[self.accu] = eventImage[i]
        #             self.accu += 1
        #             self.token_names[self.accu] = "SOS"
        #             self.accu += self.n
        #             self.token_names[self.accu] = warped[i]
        #             self.accu += 1
        #             self.token_names[self.accu] = "SOS"
        #         self.accu += self.n
        #         self.token_names[self.accu] = "EOS"

        # if split == "test" or split == "valid":
        #     split = "val"
        # nima = os.path.join("/data/storage/jianwen/N_ImageNet", f"extracted_{split}")
        # for cls in sorted(os.listdir(nima)):
        #     cls_dir = os.path.join(nima, cls)
        #     warped = sorted(glob.glob(os.path.join(cls_dir, "imageToken", "*.pt")))
        #     eventImage = sorted(glob.glob(os.path.join(cls_dir, "eventToken", "*.pt")))
        #     assert len(warped) == len(eventImage)
        #     for i in range(len(warped)):
        #         self.accu += self.n
        #         self.token_names[self.accu] = eventImage[i]
        #         self.accu += self.n
        #         self.token_names[self.accu] = warped[i]


        # ddd17 = os.path.join("/data/storage/jianwen/ddd17_seg/data")
        # if split == "train":
        #     nums = [0, 3, 4, 5, 6, 7]
        # else:
        #     nums = [1]
        # for num in nums:
        #     event_root = os.path.join(ddd17, f"dir{num}/event_features")
        #     image_root = os.path.join(ddd17, f"dir{num}/image_features")
        #     for name in sorted(os.listdir(event_root)):
        #         eventImage = os.path.join(event_root, name)
        #         warped = os.path.join(image_root, name)
        #         self.accu += self.n
        #         self.token_names[self.accu] = eventImage
        #         self.accu += 1
        #         self.token_names[self.accu] = "SOS"
        #         self.accu += self.n
        #         self.token_names[self.accu] = warped
        #         self.accu += 1
        #         self.token_names[self.accu] = "SOS"
        #     self.accu += self.n
        #     self.token_names[self.accu] = "EOS"

        self.keys = [k for k, _ in self.token_names.items()]
        self.vals = [v for _, v in self.token_names.items()]

    def __len__(self) -> int:
        return self.accu - self.window_size

    def _geturls(self, a: int, b: int):
        start = bisect_right(self.keys, a) - 1
        if start < 0:
            start = 0
        a_offset = a - self.keys[start]

        end = bisect_left(self.keys, b)
        if end == len(self.keys):
            end = len(self.keys) - 1
        b_offset = b - self.keys[start]

        return self.vals[start : end + 1], a_offset, b_offset

    def __getitem__(self, idx: int):
        urls, a_offset, b_offset = self._geturls(idx, idx + self.window_size + 1)
        tokens, ids = [], []
        for url in urls:
            if url is None:
                continue
            elif url == "SOS":
                token = torch.zeros((1, self.n_embed), dtype=torch.float32)
                ids.append(torch.full((1, ), 0, dtype=torch.int64))
            elif url == "EOS":
                token = torch.zeros((self.n, self.n_embed), dtype=torch.float32)
                ids.append(torch.full((self.n, ), 0, dtype=torch.int64))
            if "N_ImageNet" in url or "bdd100k" in url:
                if "imageToken" in url or "image_features" in url:
                    token = torch.load(url, weights_only=True, map_location="cpu").data
                    ids.append(torch.full((token.shape[0],), 3, dtype=torch.int64))
                elif "eventToken" in url or "event_features" in url:
                    token = torch.load(url, weights_only=True, map_location="cpu").data
                    ids.append(torch.full((token.shape[0],), 4, dtype=torch.int64))
            else:
                if "imageToken" in url or "image_features" in url:
                    token = torch.load(url, weights_only=True, map_location="cpu").data
                    ids.append(torch.full((token.shape[0],), 1, dtype=torch.int64))
                elif "eventToken" in url or "event_features" in url:
                    token = torch.load(url, weights_only=True, map_location="cpu").data
                    ids.append(torch.full((token.shape[0],), 2, dtype=torch.int64))
            tokens.append(token)
        tokens = torch.concat(tokens)
        ids = torch.concat(ids)
        slot = tokens[a_offset : b_offset].clone()
        ids = ids[a_offset : b_offset].clone()
        return slot, ids

class SequenceToTensor(nn.Module):
    def __init__(self, type: str, modalities: Sequence[str], origi_H: int | None = None):
        super().__init__()
        assert type in ["EI", "IL", "EL", "EO"], (
            "type must be either 'EI' for event-image, 'IL' for image-label, "
            "'EL' for event-label or 'EO' for event-optical flow"
        )
        if len(modalities) == 0:
            raise ValueError("`modalities` must contain at least one entry")
        self.type = type
        self.modalities = tuple(modalities)
        self.origi_H = origi_H

    def __call__(self, frames: Sequence[Image.Image], label: Image.Image | None):
        if len(frames) != len(self.modalities):
            raise ValueError("Modalities and frames length mismatch")

        tensors: List[torch.Tensor] = []
        for frame, modality in zip(frames, self.modalities):
            array = np.array(frame)
            tensor = torch.tensor(array).permute(2, 0, 1).float() / 255.0
            if self.origi_H is not None:
                tensor = tensor[:, : self.origi_H]
            tensors.append(tensor)
        frames_tensor = torch.stack(tensors, dim=0)

        if label is None:
            label_tensor = None
        else:
            if self.type == "EI":
                label_tensor = torch.tensor(np.array(label)).permute(2, 0, 1).float() / 255.0
            elif self.type == "EO":
                label_tensor = torch.tensor(np.array(label)).permute(2, 0, 1).float()
            elif self.type in ["IL", "EL"]:
                label_tensor = torch.tensor(np.array(label), dtype=torch.int64)
            else:
                raise ValueError(f"Unsupported type {self.type}")
        return frames_tensor, label_tensor


class SequenceNormalize(nn.Module):
    def __init__(self, m_e, s_e, m_i, s_i, type: str, modalities: Sequence[str]):
        super().__init__()
        assert type in ["EI", "IL", "EL", "EO"], (
            "type must be either 'EI' for event-image, 'IL' for image-label, "
            "'EL' for event-label or 'EO' for event-optical flow"
        )
        self.m_e = m_e
        self.s_e = s_e
        self.m_i = m_i
        self.s_i = s_i
        self.type = type
        self.modalities = tuple(modalities)

    def __call__(self, frames: torch.Tensor, label: torch.Tensor | None):
        normalized = []
        for idx, modality in enumerate(self.modalities):
            frame = frames[idx]
            if modality == "event":
                normalized.append(TF.normalize(frame, self.m_e, self.s_e))
            elif modality == "image":
                normalized.append(TF.normalize(frame, self.m_i, self.s_i))
            else:
                raise ValueError(f"Unsupported modality {modality}")
        frames = torch.stack(normalized, dim=0)

        if label is None:
            return frames, None
        if self.type == "EI":
            label = TF.normalize(label, self.m_i, self.s_i)
        return frames, label


class SequenceRandomSwapEventRedBlue:
    def __init__(self, modalities: Sequence[str], p: float = 0.5):
        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` must be in [0, 1]")
        self.p = p
        self.modalities = tuple(modalities)

    def __call__(self, frames: torch.Tensor, label: torch.Tensor | None = None):
        if frames.ndim != 4:
            return frames, label
        if random.random() >= self.p:
            return frames, label

        frames = frames.clone()
        for idx, modality in enumerate(self.modalities):
            if modality == "event" and frames[idx].shape[0] >= 3:
                frames[idx] = frames[idx][[2, 1, 0], ...]
        return frames, label


class SequenceRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` must be in [0, 1]")
        self.p = p

    def __call__(self, frames: torch.Tensor, label: torch.Tensor | None = None):
        if random.random() < self.p:
            frames = torch.flip(frames, dims=(-1,))
            if label is not None:
                label = torch.flip(label, dims=(-1,))
        return frames, label


class SequenceResizeKeepRatio(nn.Module):
    def __init__(self, short_side_target_size, type: str, scale_range=None):
        super().__init__()
        self.short_side_target_size = short_side_target_size
        self.scale_range = scale_range
        assert type in ["EI", "IL", "EL", "EO"], (
            "type must be either 'EI' for event-image, 'IL' for image-label, "
            "'EL' for event-label or 'EO' for event-optical flow"
        )
        self.type = type

    def __call__(self, frames: torch.Tensor, label: torch.Tensor | None):
        current_target = self.short_side_target_size
        if self.scale_range is not None:
            min_scale, max_scale = self.scale_range
            scale = np.random.random_sample() * (max_scale - min_scale) + min_scale
            current_target = int(round(current_target * scale))

        h, w = frames.shape[-2:]
        short_side = min(h, w)
        if short_side == 0:
            raise ValueError("Input height or width must be > 0 for resizing")
        ratio = current_target / short_side
        new_h, new_w = math.ceil(h * ratio), math.ceil(w * ratio)

        resized_frames = [
            TF.resize(frame, (new_h, new_w), interpolation=InterpolationMode.BILINEAR)
            for frame in frames
        ]
        frames = torch.stack(resized_frames, dim=0)

        if label is None:
            return frames, None

        if self.type.endswith("L"):
            label = TF.resize(
                label.unsqueeze(0), (new_h, new_w), interpolation=InterpolationMode.NEAREST
            ).squeeze(0).to(torch.int64)
        else:
            label = TF.resize(label, (new_h, new_w), interpolation=InterpolationMode.BILINEAR)
        return frames, label


class SequencePadToMinSide:
    def __init__(self, target: Union[int, Tuple[int, int]], pad_x1: int = 0, pad_x2: int = 0):
        if isinstance(target, int):
            if target <= 0:
                raise ValueError("`target` must be a positive integer")
            self.target_h = self.target_w = target
        elif (
            isinstance(target, tuple)
            and len(target) == 2
            and all(isinstance(t, int) and t > 0 for t in target)
        ):
            self.target_h, self.target_w = target
        else:
            raise ValueError("`target` must be a positive int or tuple of two positive ints")
        self.pad_x1 = pad_x1
        self.pad_x2 = pad_x2

    def __call__(self, frames: torch.Tensor, label: torch.Tensor | None):
        h, w = frames.shape[-2:]
        pad_bottom = max(0, self.target_h - h)
        pad_right = max(0, self.target_w - w)

        if pad_bottom == 0 and pad_right == 0:
            return frames, label

        frames = F.pad(frames, (0, pad_right, 0, pad_bottom), mode="constant", value=self.pad_x1)
        if label is None:
            return frames, None
        label = F.pad(
            label.unsqueeze(0),
            (0, pad_right, 0, pad_bottom),
            mode="constant",
            value=self.pad_x2,
        ).squeeze(0)
        return frames, label


class SequenceRandomCrop:
    def __init__(
        self,
        crop_size: Union[int, Tuple[int, int]],
        type: str,
        cat_max_ratio: float = 1.0,
        ignore_index: int = 255,
        trials: int = 10,
    ):
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert len(crop_size) == 2 and crop_size[0] > 0 and crop_size[1] > 0
        self.ch, self.cw = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index
        self.trials = trials
        assert type in ["EI", "IL", "EL", "EO"], (
            "type must be either 'EI' for event-image, 'IL' for image-label, "
            "'EL' for event-label or 'EO' for event-optical flow"
        )
        self.type = type

    def _rand_bbox(self, h: int, w: int) -> Tuple[int, int, int, int]:
        off_h = random.randint(0, h - self.ch)
        off_w = random.randint(0, w - self.cw)
        return off_h, off_h + self.ch, off_w, off_w + self.cw

    def _crop_tensor(self, tensor: torch.Tensor, y1: int, y2: int, x1: int, x2: int):
        return tensor[..., y1:y2, x1:x2]

    def __call__(self, frames: torch.Tensor, label: torch.Tensor | None):
        h, w = frames.shape[-2:]
        if h < self.ch or w < self.cw:
            raise ValueError(f"crop_size {self.ch, self.cw} exceeds image size {(h, w)}")

        y1, y2, x1, x2 = self._rand_bbox(h, w)

        if label is not None and self.type.endswith("L"):
            found = False
            if self.cat_max_ratio < 1.0:
                for _ in range(self.trials):
                    y1, y2, x1, x2 = self._rand_bbox(h, w)
                    seg_crop = self._crop_tensor(label, y1, y2, x1, x2)
                    valid = seg_crop[seg_crop != self.ignore_index].flatten()
                    if valid.any():
                        bincount = torch.bincount(valid, minlength=256)
                        ratio = bincount.max().item() / len(valid)
                        if ratio < self.cat_max_ratio:
                            found = True
                            break
            if not found:
                top = (h - self.ch) // 2
                left = (w - self.cw) // 2
                y1, y2, x1, x2 = top, top + self.ch, left, left + self.cw

        frames = self._crop_tensor(frames, y1, y2, x1, x2)
        if label is None:
            return frames, None
        label = self._crop_tensor(label, y1, y2, x1, x2)
        return frames, label


class SequenceCenterCrop(nn.Module):
    def __init__(self, size: Union[int, Tuple[int, int]]):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.th, self.tw = size

    def forward(self, frames: torch.Tensor, label: torch.Tensor | None):
        _, _, H, W = frames.shape
        if self.th > H or self.tw > W:
            raise ValueError(f"Crop size {(self.th, self.tw)} exceeds input size {(H, W)}")

        top = (H - self.th) // 2
        left = (W - self.tw) // 2
        frames = frames[..., top : top + self.th, left : left + self.tw]
        if label is None:
            return frames, None
        label = label[..., top : top + self.th, left : left + self.tw]
        return frames, label


class SequencePairedProcessor(nn.Module):
    def __init__(self, transforms: Sequence):
        super().__init__()
        self.transforms = transforms

    def forward(self, frames, label):
        for tf in self.transforms:
            frames, label = tf(frames, label)
        return frames, label


class ToTensor(nn.Module):
    def __init__(self, type, origi_H=None):
        super().__init__()
        self.origi_H = origi_H
        assert type in ["EI", "IL", "EL", "EO"], "type must be either 'EI' for event-image or 'IL' for image-label or 'EL' for event-label"
        self.type = type

    def __call__(self, x1, x2):
        if isinstance(x1, torch.Tensor):
            pass
        else:
            x1 = torch.tensor(np.array(x1)).permute(2, 0, 1)/255.0
            
        if self.origi_H is None:
            origi_H = x1.shape[1]
        else:
            origi_H = self.origi_H
        x1 = x1[:, :origi_H]
        if x2 is None:
            return x1, None
        else:
            if self.type == "EI":
                x2 = torch.tensor(np.array(x2)).permute(2, 0, 1)/255.0
            elif self.type == "EO":
                x2 = torch.tensor(np.array(x2)).permute(2, 0, 1)
            elif self.type in ["IL", "EL"]:
                x2 = torch.tensor(np.array(x2), dtype=torch.int64)
            return x1, x2
        
class Normalize(nn.Module):
    def __init__(self, m_e, s_e, m_i, s_i, type):
        super().__init__()
        self.m_e = m_e
        self.s_e = s_e
        self.m_i = m_i
        self.s_i = s_i
        assert type in ["EI", "IL", "EL", "EO"], f"{type} must be either 'EI' for event-image or 'IL' for image-label or 'EL' for event-label"
        self.type = type

    def forward(self, x1, x2):
        if self.type.startswith("E"):
            if x1.shape[0] == 3:
                x1 = TF.normalize(x1, self.m_e, self.s_e)
        else:
            x1 = TF.normalize(x1, self.m_i, self.s_i)
        if x2 is None:
            return x1, None
        else:
            x2 = TF.normalize(x2, self.m_i, self.s_i) if self.type == "EI" else x2
        return x1, x2

class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` must be in [0, 1]")
        self.p = p

    def __call__(self, x1, x2):
        if random.random() < self.p:
            x1 = TF.hflip(x1)
            x2 = TF.hflip(x2)
        return x1, x2

class RandomSwapEventRedBlue:
    """随机交换事件图像的红色和蓝色通道，配对图像不变。

    - 仅当 type 为 'EI' 或 'EL' 时对 x1 生效（x1 为 event）。
    - 对 'IL'（无 event）的情形不做任何处理。
    - 若通道数不足 3，则安全跳过。
    """

    def __init__(self, p: float = 0.5, type: str = "EI"):
        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` must be in [0, 1]")
        assert type in ["EI", "IL", "EL"], "type must be either 'EI', 'IL' or 'EL'"
        self.p = p
        self.type = type

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor=None):
        if self.type in ("EI", "EL") and random.random() < self.p:
            if x1.ndim == 3 and x1.shape[0] == 3:
                x1 = x1[[2, 1, 0], ...].contiguous()
        if x2 is None:
            return x1
        else:
            return x1, x2

class EnsureTensorPair(nn.Module):
    """Ensure the event tensor / label types are float32 and long respectively."""

    def __init__(self, dtype: torch.dtype = torch.float32, label_dtype: torch.dtype = torch.long):
        super().__init__()
        self.dtype = dtype
        self.label_dtype = label_dtype

    def forward(self, x1, x2):
        x1 = x1.to(self.dtype).contiguous()
        if x2 is None:
            return x1, None
        x2 = x2.to(self.label_dtype).contiguous()
        return x1, x2

class EventTensorJitter(nn.Module):
    """Lightweight intensity jitter tailored to multi-channel event tensors."""

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        bias_range: Tuple[float, float] = (-0.05, 0.05),
        noise_std: float = 0.0,
        channel_wise: bool = True,
        clamp: Optional[Tuple[float, float]] = (0.0, 1.0),
        p: float = 0.5,
    ):
        super().__init__()
        assert 0.0 <= p <= 1.0
        self.scale_range = scale_range
        self.bias_range = bias_range
        self.noise_std = float(noise_std)
        self.channel_wise = channel_wise
        self.clamp = clamp
        self.p = p

    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor]):
        if random.random() > self.p:
            return x1, x2

        c = x1.shape[0]
        if self.channel_wise:
            scales = torch.empty(c, device=x1.device, dtype=x1.dtype).uniform_(*self.scale_range).view(c, 1, 1)
            biases = torch.empty(c, device=x1.device, dtype=x1.dtype).uniform_(*self.bias_range).view(c, 1, 1)
        else:
            scale = x1.new_empty(1).uniform_(*self.scale_range)
            bias = x1.new_empty(1).uniform_(*self.bias_range)
            scales = scale.view(1, 1, 1)
            biases = bias.view(1, 1, 1)

        x1 = x1 * scales + biases
        if self.noise_std > 0.0:
            x1 = x1 + torch.randn_like(x1) * self.noise_std
        if self.clamp is not None:
            lo, hi = self.clamp
            x1 = x1.clamp(lo, hi)
        return x1, x2

class RandomShiftScaleRotateTensor(nn.Module):
    """Apply a shared random affine transform to an event tensor and its label map.

    This mirrors the augmentation strategy used in the ECDDP segmentation
    training pipeline (Pad -> RandomResizedCrop -> ShiftScaleRotate), but keeps
    the tensor size fixed for simplicity.
    """

    def __init__(
        self,
        shift_limit: float = 0.05,
        scale_limit: float = 0.05,
        rotate_limit: float = 15.0,
        p: float = 0.7,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.shift_limit = max(0.0, float(shift_limit))
        self.scale_limit = max(0.0, float(scale_limit))
        self.rotate_limit = max(0.0, float(rotate_limit))
        self.p = float(p)
        self.ignore_index = ignore_index

    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor]):
        if random.random() > self.p:
            return x1, x2
        _, h, w = x1.shape
        max_dx = self.shift_limit * w
        max_dy = self.shift_limit * h
        translate = (
            int(round(random.uniform(-max_dx, max_dx))),
            int(round(random.uniform(-max_dy, max_dy))),
        )
        angle = random.uniform(-self.rotate_limit, self.rotate_limit)
        scale = random.uniform(max(1.0 - self.scale_limit, 1e-2), 1.0 + self.scale_limit)

        x1 = TF.affine(
            x1,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        if x2 is None:
            return x1, x2
        label_dtype = x2.dtype
        label = x2.unsqueeze(0).float()
        label = TF.affine(
            label,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=self.ignore_index,
        )
        x2 = label.squeeze(0).to(label_dtype)
        return x1, x2

class ResizeKeepRatio(nn.Module):
    """
    Resize so that (h, w) fits inside max_size while preserving aspect ratio.
    Args:
        short_side_target_size: int: the short side will be reszied to this scale, then the long side will be rescaled according to the ratio
        scale_range (optional): tuple (min_scale, max_scale): if not None, short_side_target_size will be first rescale to the random sampled scale, then resize image
    """
    def __init__(self, short_side_target_size, type, scale_range=None):
        self.short_side_target_size = short_side_target_size
        self.scale_range = scale_range
        assert type in ["EI", "IL", "EL", "EO"], "type must be either 'EI' for event-image or 'IL' for image-label or 'EL' for event-label"
        self.type = type

    def __call__(self, x1, x2):
        current_short_side_target_size = self.short_side_target_size
        if self.scale_range is not None:
            min_scale, max_scale = self.scale_range[0], self.scale_range[1]
            scale = np.random.random_sample() * (max_scale - min_scale) + min_scale 
            current_short_side_target_size *= scale

        h, w = x1.shape[-2:]
        short_side = min(h, w)
        ratio = current_short_side_target_size / short_side
        new_h, new_w = math.ceil(h * ratio), math.ceil(w * ratio)

        x1 = TF.resize(x1, (new_h, new_w), interpolation=InterpolationMode.BILINEAR)
        if x2 is None:
            return x1, None
        else:
            if self.type.endswith("L"):
                # labels may be HxW; add a fake channel for resizing with NEAREST
                x2 = TF.resize(x2.unsqueeze(0), (new_h, new_w), interpolation=InterpolationMode.NEAREST).squeeze(0)
            else:
                x2 = TF.resize(x2, (new_h, new_w), interpolation=InterpolationMode.BILINEAR)
            return x1, x2

class ResizeHW(nn.Module):
    """Resize a paired (event, image) or (image, label) to exact (H, W).

    Args:
        size: int or (H, W) target spatial size.
        type: 'EI' | 'IL' | 'EL' indicating the pairing type.
        interpolation: 'bilinear' | 'bicubic' | 'nearest' for image/event paths.

    Notes:
        - For label targets (IL/EL), interpolation is always NEAREST to preserve classes.
        - Works with torch tensors or PIL images.
    """
    def __init__(self, size: Union[int, Tuple[int, int]], type: str, interpolation: str = 'bilinear'):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        assert len(size) == 2 and size[0] > 0 and size[1] > 0
        assert type in ["EI", "IL", "EL"], "type must be either 'EI' for event-image or 'IL' for image-label or 'EL' for event-label"
        self.size = size
        self.type = type

        if interpolation == 'bilinear':
            self.interp_x1 = InterpolationMode.BILINEAR
            self.interp_x2 = InterpolationMode.BILINEAR if type == "EI" else InterpolationMode.NEAREST
        elif interpolation == 'bicubic':
            self.interp_x1 = InterpolationMode.BICUBIC
            self.interp_x2 = InterpolationMode.BICUBIC if type == "EI" else InterpolationMode.NEAREST
        elif interpolation == 'nearest':
            self.interp_x1 = InterpolationMode.NEAREST
            self.interp_x2 = InterpolationMode.NEAREST
        else:
            raise ValueError(f"Unsupported interpolation mode: {interpolation}")

    def __call__(self, x1, x2):
        x1 = TF.resize(x1, self.size, interpolation=self.interp_x1)
        if self.type == "EI":
            x2 = TF.resize(x2, self.size, interpolation=self.interp_x2)
        else:
            # labels may be HxW; add a fake channel for resizing with NEAREST
            x2 = TF.resize(x2.unsqueeze(0), self.size, interpolation=self.interp_x2).squeeze(0)
        return x1, x2

class PadToMinSide:
    def __init__(
        self,
        target: Union[int, Tuple[int, int]],
        pad_x1: int = 0,
        pad_x2: int = 0,
    ):
        if isinstance(target, int):
            if target <= 0:
                raise ValueError("`target` must be a positive integer")
            self.target_h = self.target_w = target
        elif (
            isinstance(target, tuple)
            and len(target) == 2
            and all(isinstance(t, int) and t > 0 for t in target)
        ):
            self.target_h, self.target_w = target
        else:
            raise ValueError(
                "`target` must be a positive int or a tuple of two positive ints"
            )

        self.pad_x1 = pad_x1
        self.pad_x2 = pad_x2

    @staticmethod
    def _get_hw(x: torch.Tensor) -> Tuple[int, int]:
        """Return (H, W) for a 3-D image tensor CxHxW."""
        if x.ndim != 3:
            raise ValueError("Expect image with shape (C, H, W)")
        return x.shape[-2], x.shape[-1]

    @staticmethod
    def _pad_tensor(
        img: torch.Tensor,
        pad_right: int,
        pad_bottom: int,
        value: int,
    ) -> torch.Tensor:
        """Pad on the right / bottom with constant `value`."""
        padding = (0, pad_right, 0, pad_bottom)  # (left, right, top, bottom)
        return F.pad(img, padding, mode="constant", value=value)

    def __call__(self, x1, x2):
        h, w = self._get_hw(x1)
        pad_bottom = max(0, self.target_h - h)
        pad_right = max(0, self.target_w - w)

        if pad_bottom == 0 and pad_right == 0:
            if x2 is None:
                return x1, None
            else:
                return x1, x2

        x1 = self._pad_tensor(x1, pad_right, pad_bottom, self.pad_x1)
        if x2 is None:
            return x1, None
        else:
            x2 = self._pad_tensor(x2, pad_right, pad_bottom, self.pad_x2)
            return x1, x2

class RandomResizedCrop:
    """Paired random resized crop aligned with torchvision implementation.
    
    Args
    ----
    size           : int | (h, w)  - desired output size.
    scale          : tuple         - range of size of the origin size cropped (default: (0.08, 1.0)).
    ratio          : tuple         - range of aspect ratio of the origin aspect ratio cropped (default: (3/4, 4/3)).
    interpolation  : str           - interpolation mode for resizing (default: 'bilinear').
    type           : str           - type of paired data ('EI', 'IL', 'EL').
    """
    
    def __init__(self, 
                 size: Union[int, Tuple[int, int]], 
                 type: str,
                 scale: Tuple[float, float] = (0.08, 1.0),
                 ratio: Tuple[float, float] = (3./4., 4./3.),
                 interpolation: str = 'bilinear'):
        if isinstance(size, int):
            size = (size, size)
        assert len(size) == 2 and size[0] > 0 and size[1] > 0
        self.size = size
        self.scale = scale
        self.ratio = ratio
        assert type in ["EI", "IL", "EL"], "type must be either 'EI' for event-image or 'IL' for image-label or 'EL' for event-label"
        self.type = type
        
        if interpolation == 'bilinear':
            self.interpolation_x1 = InterpolationMode.BILINEAR
            self.interpolation_x2 = InterpolationMode.BILINEAR if type == "EI" else InterpolationMode.NEAREST
        elif interpolation == 'bicubic':
            self.interpolation_x1 = InterpolationMode.BICUBIC
            self.interpolation_x2 = InterpolationMode.BICUBIC if type == "EI" else InterpolationMode.NEAREST
        else:
            raise ValueError(f"Unsupported interpolation mode: {interpolation}")
    
    def get_params(self, img: torch.Tensor) -> Tuple[int, int, int, int]:
        """Get parameters for random crop.
        
        Args:
            img: Input image tensor with shape (C, H, W)
            
        Returns:
            tuple: top, left, height, width for cropping
        """
        height, width = img.shape
        area = height * width
        
        log_ratio = torch.log(torch.tensor(self.ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()
            
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w
                
        # Fallback to center crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w
    
    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random resized crop to both inputs.
        
        Args:
            x1: First input tensor (event or image)
            x2: Second input tensor (image or label)
            
        Returns:
            tuple: Transformed (x1, x2)
        """
        # Get crop parameters using x1 as reference
        i, j, h, w = self.get_params(x1)
        
        # Crop both inputs with same parameters
        x1_cropped = x1[..., i:i+h, j:j+w]
        x2_cropped = x2[..., i:i+h, j:j+w] if x2.ndim == 3 else x2[i:i+h, j:j+w]
        
        # Resize to target size
        x1 = TF.resize(x1_cropped, self.size, interpolation=self.interpolation_x1)
        if self.type == "EI":
            x2 = TF.resize(x2_cropped, self.size, interpolation=self.interpolation_x2)
        else:
            # For labels, need to add batch dimension for resize then remove it
            x2 = TF.resize(x2_cropped.unsqueeze(0), self.size, interpolation=self.interpolation_x2).squeeze(0)
        
        return x1, x2

class RandomCrop:
    """Torchvision re-implementation of MMSeg RandomCrop.

    Args
    ----
    crop_size      : int | (h, w)  - desired crop size.
    cat_max_ratio  : float         - upper bound on dominant-class area.
    ignore_index   : int           - label value to ignore.
    trials         : int           - *extra* retries after the first sample (defaults to 10, so ≤ 11 total tries).
    """

    def __init__(self, crop_size: Union[int, Tuple[int, int]], type, cat_max_ratio: float = 1.0, ignore_index: int = 255, trials: int = 10):
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert len(crop_size) == 2 and crop_size[0] > 0 and crop_size[1] > 0
        self.ch, self.cw = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index
        self.trials = trials
        assert type in ["EI", "IL", "EL", "EO"], "type must be either 'EI' for event-image or 'IL' for image-label or 'EL' for event-label"
        self.type = type
    
    def _rand_bbox(self, h: int, w: int) -> Tuple[int, int, int, int]:
        """Generate a random bbox that *may* extend past image borders."""
        off_h = random.randint(0, h - self.ch)
        off_w = random.randint(0, w - self.cw)
        return off_h, off_h + self.ch, off_w, off_w + self.cw  # y1, y2, x1, x2

    def _crop(self, image: torch.Tensor, y1: int, y2: int, x1: int, x2: int):
        """Tensor slice identical to NumPy slicing."""
        return image[..., y1:y2, x1:x2]
    
    def __call__(self, input1, input2):
        h, w = input1.shape[-2:]
        if h < self.ch or w < self.cw:
            raise ValueError(f"crop_size {self.ch, self.cw} exceeds image size {(h, w)}")

        # pick an initial window --------------------------------
        y1, y2, x1, x2 = self._rand_bbox(h, w)

        if self.type.endswith("L"):
            # retry if dominated by one class ------------------------
            found = False
            if self.cat_max_ratio < 1.0:
                for _ in range(self.trials):
                    y1, y2, x1, x2 = self._rand_bbox(h, w)

                    seg_crop = self._crop(input2, y1, y2, x1, x2)
                    valid    = seg_crop[seg_crop != self.ignore_index].flatten()
                    if valid.any():
                        bincount = torch.bincount(valid, minlength=256)
                        assert bincount.sum() == len(valid)
                        ratio = bincount.max().item() / len(valid)
                        if ratio < self.cat_max_ratio:
                            found=True
                            break
            # final crop (always exactly crop_size) ------------------
            if not found:
                top, left  = (h - self.ch) // 2, (w - self.cw) // 2
                y1, y2, x1, x2 = top, top + self.ch, left, left + self.cw

        input2 = self._crop(input2, y1, y2, x1, x2)
        input1 = self._crop(input1, y1, y2, x1, x2)
        return input1, input2

class CenterCrop(nn.Module):
    def __init__(self, size):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.th, self.tw = size

    def forward(self, x1, x2):
        _, H, W = x1.shape
        if self.th > H or self.tw > W:
            raise ValueError(f"Crop size {(self.th, self.tw)} exceeds input size {(H, W)}")

        top  = (H - self.th) // 2
        left = (W - self.tw) // 2

        x1 = x1[..., top : top + self.th, left : left + self.tw]
        if x2 is None:
            return x1, None
        else:
            x2 = x2[..., top : top + self.th, left : left + self.tw]
            return x1, x2
        
class PhotoMetricDistortion:
    """MMSeg PhotoMetricDistortion for *torch tensors* in [0,1].

    Sequence (each with p=0.5):
        1) random brightness
        2) random contrast  (mode-0 branch)
        3) RGB → HSV
        4) random saturation
        5) random hue
        6) HSV → RGB
        7) random contrast  (mode-1 branch)
    """

    def __init__(
        self,
        brightness_delta: int = 32,                # ±32/255
        contrast_range: Tuple[float, float] = (0.5, 1.5),
        saturation_range: Tuple[float, float] = (0.5, 1.5),
        hue_delta: int = 18,                      # ±18°  (0‑180)
    ):
        self.bd = brightness_delta / 255.0
        self.cl, self.cu = contrast_range
        self.sl, self.su = saturation_range
        self.hd = hue_delta / 180.0               # convert to 0‑1

    # ------------------------------------------------------------
    # helpers ----------------------------------------------------
    # ------------------------------------------------------------
    def _hsv_to_rgb(self, image: torch.Tensor) -> torch.Tensor:
        h, s, v = image.unbind(dim=-3)
        h6 = h.mul(6)
        i = torch.floor(h6)
        f = h6.sub_(i)
        i = i.to(dtype=torch.int32)

        sxf = s * f
        one_minus_s = 1.0 - s
        q = (1.0 - sxf).mul_(v).clamp_(0.0, 1.0)
        t = sxf.add_(one_minus_s).mul_(v).clamp_(0.0, 1.0)
        p = one_minus_s.mul_(v).clamp_(0.0, 1.0)
        i.remainder_(6)

        vpqt = torch.stack((v, p, q, t), dim=-3)

        # vpqt -> rgb mapping based on i
        select = torch.tensor([[0, 2, 1, 1, 3, 0], [3, 0, 0, 2, 1, 1], [1, 1, 3, 0, 0, 2]], dtype=torch.long)
        select = select.to(device=image.device, non_blocking=True)

        select = select[:, i]
        if select.ndim > 3:
            # if input.shape is (B, ..., C, H, W) then
            # select.shape is (C, B, ...,  H, W)
            # thus we move C axis to get (B, ..., C, H, W)
            select = select.moveaxis(0, -3)

        return vpqt.gather(-3, select)
    
    def _rgb_to_hsv(self, image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        r, g, b = image.unbind(dim=-3)

        maxc, _ = torch.max(image, dim=-3)
        minc, _ = torch.min(image, dim=-3)
        v      = maxc

        delta  = maxc - minc
        # Saturation ----------------------------------------------------------------
        s      = delta / (v + eps)                       # v can be zero

        # Hue -----------------------------------------------------------------------
        # denominator is never 0 thanks to eps
        rc = (v - r) / (delta + eps)
        gc = (v - g) / (delta + eps)
        bc = (v - b) / (delta + eps)

        h = torch.zeros_like(v)
        mask = delta > eps

        h[mask & (v == r)] = (bc - gc)[mask & (v == r)]
        h[mask & (v == g)] = (2.0 + rc - bc)[mask & (v == g)]
        h[mask & (v == b)] = (4.0 + gc - rc)[mask & (v == b)]
        h = (h / 6.0) % 1.0                               # put in [0,1)

        return torch.stack((h, s, v), dim=-3)
    
    @staticmethod
    def _convert(image: torch.Tensor, alpha: float = 1.0, beta: float = 0.0):
        """image * alpha + beta with clipping to [0,1]."""
        return (image * alpha + beta).clamp_(0.0, 1.0)

    def _brightness(self, image: torch.Tensor):
        if random.randint(0, 1):
            beta = random.uniform(-self.bd, self.bd)
            image = self._convert(image, beta=beta)
        return image

    def _contrast(self, image: torch.Tensor):
        if random.randint(0, 1):
            alpha = random.uniform(self.cl, self.cu)
            image = self._convert(image, alpha=alpha)
        return image

    def _saturation(self, image: torch.Tensor):
        if random.randint(0, 1):
            hsv = self._rgb_to_hsv(image)
            alpha = random.uniform(self.sl, self.su)
            hsv[1] = self._convert(hsv[1], alpha=alpha)  # channel‑wise op
            image = self._hsv_to_rgb(hsv)
        return image

    def _hue(self, image: torch.Tensor):
        if random.randint(0, 1):
            hsv = self._rgb_to_hsv(image)
            delta = random.uniform(-self.hd, self.hd)
            hsv[0] = (hsv[0] + delta) % 1.0
            image = self._hsv_to_rgb(hsv)
        return image

    def __call__(self, x1, x2):

        # 1) brightness
        x1 = self._brightness(x1)

        # decide contrast position
        mode = random.randint(0, 1)
        if mode == 1:
            x1 = self._contrast(x1)

        # 3‑5) H/S space
        x1 = self._saturation(x1)
        x1 = self._hue(x1)

        # 7) contrast (mode 0)
        if mode == 0:
            x1 = self._contrast(x1)

        return x1, x2

class PairedProcessor(nn.Module):
    def __init__(self, img_tf: list): 
        '''image: PIL, label: PIL'''
        super().__init__() 
        self.img_tf = img_tf
    
    def forward(self, x1, x2):
        for tf in self.img_tf:
            x1, x2 = tf(x1, x2)
        return x1, x2


def _resize_intrinsics_meta(intr: Optional[Dict[str, float]], target_h: int, target_w: int) -> None:
    if not intr:
        return
    width = intr.get("width")
    height = intr.get("height")
    if width is None or height is None:
        return
    if width <= 0 or height <= 0:
        return
    sx = float(target_w) / float(width)
    sy = float(target_h) / float(height)
    intr["fx"] = intr.get("fx", 0.0) * sx
    intr["fy"] = intr.get("fy", 0.0) * sy
    intr["cx"] = (intr.get("cx", 0.0) + 0.5) * sx - 0.5
    intr["cy"] = (intr.get("cy", 0.0) + 0.5) * sy - 0.5
    intr["width"] = float(target_w)
    intr["height"] = float(target_h)


def _shift_intrinsics(intr: Optional[Dict[str, float]], dx: float, dy: float) -> None:
    if not intr:
        return
    intr["cx"] = intr.get("cx", 0.0) - float(dx)
    intr["cy"] = intr.get("cy", 0.0) - float(dy)


def _flip_intrinsics_horizontally(intr: Optional[Dict[str, float]]) -> None:
    if not intr:
        return
    width = intr.get("width")
    if width is None:
        return
    intr["cx"] = (float(width) - 1.0) - intr.get("cx", 0.0)


class DepthTransformCompose(nn.Module):
    """Apply a list of depth transforms to a sample dictionary."""

    def __init__(self, transforms_list: Sequence):
        super().__init__()
        self.transforms = list(transforms_list)

    def forward(self, sample: Dict[str, torch.Tensor | Image.Image | np.ndarray]):
        for tfm in self.transforms:
            sample = tfm(sample)
        return sample


class DepthToTensor(nn.Module):
    """Convert PIL / numpy inputs in the sample dict to torch tensors."""

    def __init__(self):
        super().__init__()

    def forward(self, sample: Dict[str, Union[Image.Image, np.ndarray, torch.Tensor]]):
        event = sample["event"]
        image = sample["image"]
        depth = sample["depth"]
        mask = sample["mask"]

        if isinstance(event, Image.Image):
            event = torch.from_numpy(np.array(event, copy=True)).permute(2, 0, 1).float() / 255.0
        elif isinstance(event, np.ndarray):
            event = torch.from_numpy(event).permute(2, 0, 1).float() / 255.0

        if isinstance(image, Image.Image):
            image = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if isinstance(depth, Image.Image):
            depth = torch.from_numpy(np.array(depth, copy=True)).float()
        elif isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth).float()
        elif isinstance(depth, torch.Tensor):
            depth = depth.float()

        if isinstance(mask, Image.Image):
            mask = torch.from_numpy(np.array(mask, copy=True)).float()
        elif isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
        elif isinstance(mask, torch.Tensor):
            mask = mask.float()

        sample["event"] = event
        sample["image"] = image
        sample["depth"] = depth
        sample["mask"] = mask
        return sample


class DepthNormalize(nn.Module):
    """Normalize event and RGB frames using dataset mean/std statistics."""

    def __init__(
        self,
        event_mean: Sequence[float],
        event_std: Sequence[float],
        image_mean: Sequence[float],
        image_std: Sequence[float],
    ):
        super().__init__()
        self.event_mean = event_mean
        self.event_std = event_std
        self.image_mean = image_mean
        self.image_std = image_std

    def _prepare_stats(self, stats: Sequence[float], channels: int, dtype, device) -> torch.Tensor:
        tensor = torch.as_tensor(stats, dtype=dtype, device=device)
        if tensor.numel() == channels:
            return tensor.view(channels, 1, 1)
        if tensor.numel() == 1:
            tensor = tensor.expand(channels)
        else:
            repeats = math.ceil(channels / tensor.numel())
            tensor = tensor.repeat(repeats)[:channels]
        return tensor.view(channels, 1, 1)

    def forward(self, sample: Dict[str, torch.Tensor]):
        event = sample["event"]
        img = sample["image"]
        event_mean = self._prepare_stats(self.event_mean, event.shape[0], event.dtype, event.device)
        event_std = self._prepare_stats(self.event_std, event.shape[0], event.dtype, event.device)
        sample["event"] = (event - event_mean) / event_std.clamp_min(1e-6)

        img_mean = self._prepare_stats(self.image_mean, img.shape[0], img.dtype, img.device)
        img_std = self._prepare_stats(self.image_std, img.shape[0], img.dtype, img.device)
        sample["image"] = (img - img_mean) / img_std.clamp_min(1e-6)
        return sample


class DepthRandomHorizontalFlip(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError("`p` must be in [0, 1]")
        self.p = p

    def forward(self, sample: Dict[str, torch.Tensor]):
        if random.random() < self.p:
            sample["event"] = TF.hflip(sample["event"])
            sample["image"] = TF.hflip(sample["image"])
            sample["depth"] = torch.flip(sample["depth"], dims=[1])
            sample["mask"] = torch.flip(sample["mask"], dims=[1])
            _flip_intrinsics_horizontally(sample.get("intrinsics"))
        return sample


class DepthRandomResizedCrop(nn.Module):
    """Apply aligned random resized crop for event/image/depth/mask."""

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        scale: Tuple[float, float] = (0.6, 1.0),
        ratio: Tuple[float, float] = (0.9, 1.1),
        antialias: bool = True,
    ):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        if len(size) != 2:
            raise ValueError("`size` must be an int or (h, w)")
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.antialias = antialias

    def _get_params(self, height: int, width: int) -> Tuple[int, int, int, int]:
        area = height * width
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))

        for _ in range(10):
            target_area = area * random.uniform(*self.scale)
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to center crop
        in_ratio = float(width) / float(height)
        if in_ratio < self.ratio[0]:
            w = width
            h = int(round(w / self.ratio[0]))
        elif in_ratio > self.ratio[1]:
            h = height
            w = int(round(h * self.ratio[1]))
        else:
            h = height
            w = width
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, sample: Dict[str, torch.Tensor]):
        _, height, width = sample["event"].shape
        i, j, h, w = self._get_params(height, width)

        sample["event"] = TF.resized_crop(
            sample["event"], i, j, h, w, self.size, interpolation=InterpolationMode.BILINEAR, antialias=self.antialias
        )
        sample["image"] = TF.resized_crop(
            sample["image"], i, j, h, w, self.size, interpolation=InterpolationMode.BILINEAR, antialias=self.antialias
        )

        depth = sample["depth"].unsqueeze(0)
        depth = TF.resized_crop(
            depth, i, j, h, w, self.size, interpolation=InterpolationMode.BILINEAR, antialias=self.antialias
        )
        sample["depth"] = depth.squeeze(0)

        mask = sample["mask"].unsqueeze(0)
        mask = TF.resized_crop(mask, i, j, h, w, self.size, interpolation=InterpolationMode.NEAREST, antialias=False)
        sample["mask"] = mask.squeeze(0)

        intr = sample.get("intrinsics")
        if intr:
            _shift_intrinsics(intr, j, i)
            intr["width"] = float(w)
            intr["height"] = float(h)
            _resize_intrinsics_meta(intr, self.size[0], self.size[1])

        return sample


class DepthResize(nn.Module):
    """Resize all modalities to a target resolution."""

    def __init__(self, size: Union[int, Tuple[int, int]], antialias: bool = True):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        if len(size) != 2:
            raise ValueError("`size` must be an int or (h, w)")
        self.size = size
        self.antialias = antialias

    def forward(self, sample: Dict[str, torch.Tensor]):
        sample["event"] = TF.resize(
            sample["event"], self.size, interpolation=InterpolationMode.BILINEAR, antialias=self.antialias
        )
        sample["image"] = TF.resize(
            sample["image"], self.size, interpolation=InterpolationMode.BILINEAR, antialias=self.antialias
        )

        depth = sample["depth"].unsqueeze(0)
        depth = TF.resize(depth, self.size, interpolation=InterpolationMode.BILINEAR, antialias=self.antialias)
        sample["depth"] = depth.squeeze(0)

        mask = sample["mask"].unsqueeze(0)
        mask = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST, antialias=False)
        sample["mask"] = mask.squeeze(0)
        _resize_intrinsics_meta(sample.get("intrinsics"), self.size[0], self.size[1])
        return sample


class DepthClamp(nn.Module):
    """Clamp depth range and deactivate samples outside the target range."""

    def __init__(self, min_depth: float | None, max_depth: float | None):
        super().__init__()
        self.min_depth = float(min_depth) if min_depth is not None else None
        self.max_depth = float(max_depth) if max_depth is not None else None
        if self.min_depth is not None and self.max_depth is not None:
            if self.max_depth < self.min_depth:
                raise ValueError("max_depth must be >= min_depth.")

    def forward(self, sample: Dict[str, torch.Tensor]):
        depth = sample["depth"]
        mask = sample["mask"]

        if isinstance(mask, torch.Tensor) and mask.dtype != torch.bool:
            valid = mask > 0.5
        else:
            valid = mask.bool()

        if self.min_depth is not None:
            valid = valid & (depth >= self.min_depth)
        if self.max_depth is not None:
            valid = valid & (depth <= self.max_depth)

        depth_clamped = depth
        if self.min_depth is not None:
            depth_clamped = torch.clamp(depth_clamped, min=self.min_depth)
        if self.max_depth is not None:
            depth_clamped = torch.clamp(depth_clamped, max=self.max_depth)

        if mask.dtype == torch.bool:
            mask_out = valid
        else:
            mask_out = valid.to(mask.dtype)

        depth_clamped = depth_clamped * mask_out.to(depth_clamped.dtype)

        sample["depth"] = depth_clamped
        sample["mask"] = mask_out
        return sample


class DepthColorJitter(nn.Module):
    """Apply color jitter augmentation to the RGB frame."""

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p: float = 0.5):
        super().__init__()
        self.jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.p = p

    def forward(self, sample: Dict[str, torch.Tensor]):
        if random.random() < self.p:
            sample["image"] = self.jitter(sample["image"])
        return sample


class DepthRandomGamma(nn.Module):
    """Random gamma augmentation on RGB frames."""

    def __init__(self, gamma_range: Tuple[float, float] = (0.9, 1.1), p: float = 0.3):
        super().__init__()
        self.gamma_range = gamma_range
        self.p = p

    def forward(self, sample: Dict[str, torch.Tensor]):
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            sample["image"] = sample["image"].clamp(0.0, 1.0).pow(gamma)
        return sample


class DepthAdditiveGaussianNoise(nn.Module):
    """Inject Gaussian noise into the event representation."""

    def __init__(self, std: float = 0.02, p: float = 0.5):
        super().__init__()
        self.std = std
        self.p = p

    def forward(self, sample: Dict[str, torch.Tensor]):
        if random.random() < self.p:
            noise = torch.randn_like(sample["event"]) * self.std
            sample["event"] = (sample["event"] + noise).clamp(0.0, 1.0)
        return sample


class DepthEnsureContiguous(nn.Module):
    """Ensure tensors are contiguous in memory for dataloader efficiency."""

    def forward(self, sample: Dict[str, torch.Tensor]):
        for key in ["event", "image", "depth", "mask"]:
            tensor = sample[key]
            if isinstance(tensor, torch.Tensor):
                sample[key] = tensor.contiguous()
        return sample
