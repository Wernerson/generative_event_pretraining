from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Dict, Iterable, Optional, Tuple, Union

import math
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

try:
    from torch.serialization import add_safe_globals
except ImportError:  # pragma: no cover - older torch
    add_safe_globals = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if add_safe_globals is not None:
    add_safe_globals([np.core.multiarray.scalar])

if "torch._six" not in sys.modules:
    stub = ModuleType("torch._six")
    stub.inf = math.inf
    sys.modules["torch._six"] = stub

if "tensorboardX" not in sys.modules:
    tbx = ModuleType("tensorboardX")

    class _SummaryWriter:  # pragma: no cover - debug scaffold
        def __init__(self, *_, **__):
            pass

        def add_scalar(self, *_, **__):
            pass

        def add_image(self, *_, **__):
            pass

        def close(self):
            pass

        def flush(self):
            pass

        def set_step(self, *_, **__):
            pass

        def update(self, *_, **__):
            pass

    tbx.SummaryWriter = _SummaryWriter
    sys.modules["tensorboardX"] = tbx

from mem.mem.modeling_finetune import VisionTransformer
from mem.mem.utils import finetune


DEFAULT_CKPT = Path("/data/storage/jianwen/nimagenet-pt-checkpoint-74.pth")
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
NUMPY_EXTENSIONS = {".npy", ".npz"}
TORCH_EXTENSIONS = {".pt", ".pth"}


@dataclass
class MEMEncoderConfig:
    """Configuration container for the MEM ViT encoder."""

    ckpt_path: Union[str, Path] = DEFAULT_CKPT
    image_size: Tuple[int, int] = (224, 224)
    patch_size: Union[int, Tuple[int, int]] = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    layer_scale_init_value: float = 0.1
    use_abs_pos_emb: bool = False
    use_rel_pos_bias: bool = True
    use_mean_pooling: bool = True
    init_scale: float = 0.001
    in_chans: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    model_key: str = "model|module"
    model_prefix: str = ""
    freeze: bool = True

    def as_patch_tuple(self) -> Tuple[int, int]:
        if isinstance(self.patch_size, tuple):
            return self.patch_size
        return (self.patch_size, self.patch_size)

    def resolved_ckpt(self) -> Path:
        return Path(self.ckpt_path).expanduser().resolve()


class MemEncoder(nn.Module):
    """Wrapper around the MEM ViT backbone that returns pooled and token features."""

    def __init__(self, config: MEMEncoderConfig):
        super().__init__()
        self.config = config
        init_values = None if config.layer_scale_init_value <= 0 else config.layer_scale_init_value
        self.backbone = VisionTransformer(
            img_size=config.image_size,
            patch_size=config.as_patch_tuple(),
            in_chans=config.in_chans,
            num_classes=0,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=True,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            drop_path_rate=config.drop_path_rate,
            init_values=init_values,
            use_abs_pos_emb=config.use_abs_pos_emb,
            use_rel_pos_bias=config.use_rel_pos_bias,
            use_shared_rel_pos_bias=False,
            use_mean_pooling=config.use_mean_pooling,
            init_scale=config.init_scale,
            use_batch_norm=False,
        )
        self._load_checkpoint()
        self.to(device=config.device, dtype=config.dtype)
        if config.freeze:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    def _load_checkpoint(self) -> None:
        ckpt_path = self.config.resolved_ckpt()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"MEM checkpoint not found: {ckpt_path}")
        args = SimpleNamespace(
            finetune=str(ckpt_path),
            model_key=self.config.model_key,
            model_prefix=self.config.model_prefix,
        )
        original_load = torch.load

        def _patched_load(*load_args, **load_kwargs):
            load_kwargs.setdefault("weights_only", False)
            return original_load(*load_args, **load_kwargs)

        torch.load = _patched_load
        try:
            finetune(args, self.backbone)
        finally:
            torch.load = original_load

    def forward(
        self, x: torch.Tensor, return_tokens: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        feats = self._encode_tokens(x)
        if return_tokens:
            return feats
        return feats["pooled"]

    def extract_features(
        self, x: torch.Tensor, return_tokens: bool = True
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            feats = self._encode_tokens(x)
        return feats if return_tokens else {"pooled": feats["pooled"]}

    def _encode_tokens(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = self.backbone.patch_embed(x)
        batch = tokens.shape[0]
        cls_tokens = self.backbone.cls_token.expand(batch, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        if self.backbone.pos_embed is not None:
            tokens = tokens + self.backbone.pos_embed
        tokens = self.backbone.pos_drop(tokens)
        rel_pos_bias = (
            self.backbone.rel_pos_bias() if self.backbone.rel_pos_bias is not None else None
        )
        for block in self.backbone.blocks:
            tokens = block(tokens, rel_pos_bias=rel_pos_bias)
        tokens = self.backbone.norm(tokens)
        patch_tokens = tokens[:, 1:, :]
        cls_token = tokens[:, 0]
        if self.backbone.fc_norm is not None:
            pooled = self.backbone.fc_norm(patch_tokens.mean(1))
        else:
            pooled = cls_token
        return {"cls": cls_token, "pooled": pooled, "patch_tokens": patch_tokens}


def build_mem_transform(
    image_size: Tuple[int, int],
    normalize: bool = True,
    mean: Iterable[float] = DEFAULT_MEAN,
    std: Iterable[float] = DEFAULT_STD,
) -> transforms.Compose:
    tx = [
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
    ]
    if normalize:
        tx.append(
            transforms.Normalize(mean=_broadcast_stats(mean, 3), std=_broadcast_stats(std, 3))
        )
    return transforms.Compose(tx)


def _broadcast_stats(values: Iterable[float], channels: int) -> Tuple[float, ...]:
    seq = tuple(values)
    if len(seq) == channels:
        return seq
    if len(seq) == 1:
        return tuple(seq[0] for _ in range(channels))
    if len(seq) > channels:
        return tuple(seq[:channels])
    last = seq[-1]
    padded = list(seq) + [last] * (channels - len(seq))
    return tuple(padded)


def _ensure_chw(array: np.ndarray) -> torch.Tensor:
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected (H, W, C) or (C, H, W) array, received shape {arr.shape}")
    if arr.shape[0] in {1, 3} and arr.shape[0] < arr.shape[1]:
        tensor = torch.from_numpy(arr)
    else:
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = tensor.float()
    if tensor.max() > 1:
        tensor = tensor / 255.0
    return tensor


def load_event_tensor(
    path: Union[str, Path],
    image_size: Tuple[int, int],
    normalize: bool = True,
    mean: Iterable[float] = DEFAULT_MEAN,
    std: Iterable[float] = DEFAULT_STD,
) -> torch.Tensor:
    """Load an event frame (PNG/JPEG/NPY/PT) and return a BCHW tensor."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")
    suffix = source.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        img = Image.open(source).convert("RGB")
        tensor = build_mem_transform(image_size, normalize, mean, std)(img)
    elif suffix in NUMPY_EXTENSIONS:
        data = np.load(source)
        if isinstance(data, np.lib.npyio.NpzFile):
            if "events" in data:
                data = data["events"]
            else:
                data = data[data.files[0]]
        tensor = _ensure_chw(data)
        resize = transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True)
        tensor = resize(tensor)
        if normalize:
            tensor = transforms.Normalize(
                mean=_broadcast_stats(mean, tensor.shape[0]),
                std=_broadcast_stats(std, tensor.shape[0]),
            )(tensor)
    elif suffix in TORCH_EXTENSIONS:
        loaded = torch.load(source, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict):
            for key in ("image", "event", "events", "tensor"):
                if key in loaded:
                    loaded = loaded[key]
                    break
        tensor = torch.as_tensor(loaded).float()
        if tensor.ndim == 3 and tensor.shape[0] not in {1, 3}:
            tensor = tensor.permute(2, 0, 1)
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        resize = transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True)
        tensor = resize(tensor)
        if normalize:
            tensor = transforms.Normalize(
                mean=_broadcast_stats(mean, tensor.shape[0]),
                std=_broadcast_stats(std, tensor.shape[0]),
            )(tensor)
    else:
        raise ValueError(f"Unsupported input format: {suffix}")
    return tensor.unsqueeze(0)


def parse_dtype(value: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    value = value.lower()
    if value not in table:
        raise ValueError(f"Unsupported dtype '{value}'. Choose from {list(table)}.")
    return table[value]


def build_config_from_args(args: argparse.Namespace) -> MEMEncoderConfig:
    base = MEMEncoderConfig()
    dtype = parse_dtype(args.dtype) if args.dtype else base.dtype
    img_size = tuple(args.image_size) if args.image_size else base.image_size
    patch = args.patch_size if args.patch_size else base.patch_size
    device = args.device or base.device
    cfg = replace(
        base,
        ckpt_path=args.ckpt,
        image_size=img_size,
        patch_size=patch,
        device=device,
        dtype=dtype,
        freeze=not args.trainable,
    )
    return cfg


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MEM encoder feature extractor")
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT), help="Path to MEM checkpoint")
    parser.add_argument("--input", type=str, help="Path to an event frame (.png/.npy/.pt)")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda:0")
    parser.add_argument("--dtype", type=str, default="float32", help="Tensor dtype (float32, float16, bf16)")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("H", "W"), help="Resize height/width")
    parser.add_argument("--patch-size", type=int, help="Patch size (defaults to 16)")
    parser.add_argument("--return-patches", action="store_true", help="Return patch tokens in addition to pooled feature")
    parser.add_argument("--save", type=str, help="Optional .pt output path for extracted features")
    parser.add_argument("--trainable", action="store_true", help="Keep encoder parameters trainable")
    parser.add_argument("--no-normalize", action="store_true", help="Skip mean/std normalization on inputs")
    return parser.parse_args()


def main() -> None:
    args = cli()
    config = build_config_from_args(args)
    encoder = MemEncoder(config).to(config.device, dtype=config.dtype)
    if args.input:
        sample = load_event_tensor(
            args.input, config.image_size, normalize=not args.no_normalize, mean=DEFAULT_MEAN, std=DEFAULT_STD
        ).to(config.device, dtype=config.dtype)
    else:
        sample = torch.rand(
            1, config.in_chans, config.image_size[0], config.image_size[1], device=config.device, dtype=config.dtype
        )
    feats = encoder(sample, return_tokens=args.return_patches)
    if isinstance(feats, dict):
        shapes = {k: tuple(v.shape) for k, v in feats.items()}
        print("Extracted features:", shapes)
    else:
        print("Extracted pooled feature:", tuple(feats.shape))
    if args.save:
        output = {k: v.detach().cpu() for k, v in feats.items()} if isinstance(feats, dict) else feats.detach().cpu()
        torch.save(output, args.save)
        print(f"Saved features to {args.save}")


if __name__ == "__main__":
    main()
