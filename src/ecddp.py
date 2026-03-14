from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
ECDDP_ROOT = REPO_ROOT / "Event-Camera-Data-Dense-Pre-training"
SWIN_PATH = ECDDP_ROOT / "model" / "swin_model.py"

if not SWIN_PATH.exists():
    raise FileNotFoundError(f"Missing SWIN model file at {SWIN_PATH}")

_swin_spec = importlib.util.spec_from_file_location("ecddp_swin_pad", SWIN_PATH)
_swin_module = importlib.util.module_from_spec(_swin_spec)  # type: ignore[arg-type]
sys.modules[_swin_spec.name] = _swin_module  # type: ignore[attr-defined]
assert _swin_spec.loader is not None
_swin_spec.loader.exec_module(_swin_module)  # type: ignore[arg-type]
SWINPad = _swin_module.SWINPad


def _to_tuple(values: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(v) for v in values)


@dataclass
class ECDDPEncoderConfig:
    ckpt_path: Union[str, Path]
    image_size: Tuple[int, int]
    patch_size: int = 4
    in_chans: int = 3
    embed_dim: int = 96
    depths: Sequence[int] = (2, 2, 6, 2)
    num_heads: Sequence[int] = (3, 6, 12, 24)
    window_size: int = 7
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    ape: bool = False
    patch_norm: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    keep_patch_keys: bool = False
    load_teacher: bool = False
    freeze: bool = True

    def resolved_ckpt(self) -> Path:
        return Path(self.ckpt_path).expanduser().resolve()

    @property
    def output_patch_size(self) -> int:
        # patch embedding reduces by patch_size, and there are len(depths)-1 patch-merging stages
        return self.patch_size * (2 ** (len(self.depths) - 1))


class ECDDPEncoder(nn.Module):
    def __init__(self, config: ECDDPEncoderConfig):
        super().__init__()
        self.config = config
        if not config.resolved_ckpt().exists():
            raise FileNotFoundError(f"ECDDP checkpoint not found: {config.ckpt_path}")

        depths = _to_tuple(config.depths)
        num_heads = _to_tuple(config.num_heads)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.backbone = SWINPad(
            pretrain_img_size=max(config.image_size),
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=config.window_size,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            drop_path_rate=config.drop_path_rate,
            norm_layer=norm_layer,
            ape=config.ape,
            patch_norm=config.patch_norm,
            out_indices=tuple(range(len(depths))),
            use_checkpoint=False,
            pretrained_checkpoint=str(config.resolved_ckpt()),
            keep_patch_keys=config.keep_patch_keys,
            load_teacher=config.load_teacher,
            num_classes=0,
        )
        self.backbone.to(device=config.device, dtype=config.dtype)
        if config.freeze:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor, return_tokens: bool = False):
        feats = self.extract_features(x)
        if return_tokens:
            return feats
        return feats["patch_tokens"]

    def extract_features(self, x: torch.Tensor) -> dict:
        outs = self.backbone(x)
        deepest = outs[-1]
        B, C, H, W = deepest.shape
        tokens = deepest.flatten(2).transpose(1, 2)
        return {
            "patch_tokens": tokens,
            "patch_shape": (H, W),
            "feature_map": deepest,
            "pyramid_feats": outs,
        }
