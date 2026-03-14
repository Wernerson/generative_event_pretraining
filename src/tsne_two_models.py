from __future__ import annotations

import sys
sys.path.append("dinov2")

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from config import Config
from dinov2.models.vision_transformer import vit_small, vit_base

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class ScriptConfig:
    data_root: str = "/data/storage/jianwen/N_ImageNet/extracted_val"
    samples_per_class: int = 50
    num_classes: int = 20
    class_seed: int = 0
    batch_size: int = 32
    num_workers: int = 4
    device: str = "cuda:0"
    image_size: int = 224
    figure_path: str = "tsne_two_models.png"
    figure_dpi: int = 300
    show_legend: bool = True
    image_extensions: tuple[str, ...] = (".png",)
    tsne_perplexity: float | None = 80.0
    tsne_early_exaggeration: float = 4.0
    tsne_learning_rate: float | str = 150.0
    tsne_n_iter: int = 1500
    tsne_metric: str = "cosine"
    tsne_standardize: bool = True
    label_mapping_path: str | None = "src/imagenet_mapping.json"
    imagenet_index_path: str | None = "src/imagenet_class_index.json"
    legend_max_items: int | None = None
    legend_columns: int = 5
    legend_loc: str = "upper center"
    legend_bbox: tuple[float, float] = (0.5, -0.015)
    legend_marker_size: float = 6.0
    legend_fontsize: float = 12.5
    axis_limit: float | None = 4.0
    grid_major_step: float | None = 1.0
    grid_minor_step: float | None = None
    axis_fill_ratio: float = 0.97


@dataclass
class ModelSpec:
    name: str
    checkpoint: str
    state_key: str | None = None  # e.g. "event_encoder"


SCRIPT_CFG = ScriptConfig()
MODEL_SPECS: Sequence[ModelSpec] = (
    ModelSpec(
        name="Model-A",
        checkpoint="/data/storage/jianwen/cache/dinov2/dinov2_vitb14_reg4_pretrain.pth",
        state_key=None,
    ),
    ModelSpec(
        name="Model-B",
        checkpoint="/data/storage/jianwen/cache/ckpt_matters/gra_base_16x.pt",
        state_key="event_encoder",
    ),
)


class LimitedImageNet(Dataset):
    def __init__(
        self,
        root: str,
        samples_per_class: int,
        num_classes: int,
        class_seed: int,
        image_size: int,
        extensions: Iterable[str],
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.samples: list[tuple[Path, int]] = []
        self.class_names: list[str] = []
        self.samples_per_class = samples_per_class
        self.num_classes = num_classes
        self.class_seed = class_seed
        self.extensions = tuple(ext.lower() for ext in extensions)

        cfg = Config()
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.NIMA_ME, std=cfg.NIMA_SE),
            ]
        )
        self._gather_samples()

    def _gather_samples(self) -> None:
        assert self.root.is_dir(), f"Dataset root {self.root} not found."
        eligible_classes: list[tuple[str, list[Path]]] = []
        for class_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            image_files = self._list_images(class_dir)
            if len(image_files) < self.samples_per_class:
                continue
            eligible_classes.append((class_dir.name, image_files))

        if not eligible_classes:
            raise RuntimeError(f"No RGB samples were found in {self.root}.")

        rng = random.Random(self.class_seed)
        if self.num_classes > 0:
            if len(eligible_classes) < self.num_classes:
                raise ValueError(
                    f"Requested {self.num_classes} classes but only {len(eligible_classes)} have >= {self.samples_per_class} samples."
                )
            selected = rng.sample(eligible_classes, self.num_classes)
        else:
            selected = eligible_classes

        self.class_names = [name for name, _ in selected]

        for class_id, (name, files) in enumerate(selected):
            take = files[: self.samples_per_class]
            for path in take:
                self.samples.append((path, class_id))

        if not self.samples:
            raise RuntimeError("No samples collected after filtering.")

    def _list_images(self, class_dir: Path) -> list[Path]:
        files: list[Path] = []
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in self.extensions:
                files.append(path)
        return files

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        tensor = self.transform(img)
        return tensor, label


def load_label_mapping(mapping_path: str | None, class_index_path: str | None) -> dict[str, str]:
    """Map WordNet ids (e.g. n01440764) to human-readable names."""

    def _load_json(path: str | None) -> dict[str, object]:
        if not path:
            return {}
        target = Path(path)
        if not target.is_file():
            print(f"[Legend] Mapping file not found at {target}.")
            return {}
        try:
            with target.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[Legend] Failed to read mapping {target}: {exc}")
            return {}
        if not isinstance(payload, dict):
            print(f"[Legend] Mapping at {target} is not a dict.")
            return {}
        return payload

    idx_to_name_raw = _load_json(mapping_path)
    wnid_index_raw = _load_json(class_index_path)

    idx_to_name: dict[int, str] = {}
    for key, value in idx_to_name_raw.items():
        try:
            idx = int(key)
        except Exception:
            continue
        idx_to_name[idx] = str(value)

    wnid_to_idx: dict[str, int] = {}
    for key, value in wnid_index_raw.items():
        try:
            idx = int(key)
        except Exception:
            continue
        if isinstance(value, (list, tuple)) and value:
            wnid = str(value[0])
            wnid_to_idx[wnid] = idx
            if idx not in idx_to_name and len(value) > 1:
                idx_to_name[idx] = str(value[1])

    if not wnid_to_idx:
        return {}

    return {wnid: idx_to_name.get(idx, wnid) for wnid, idx in wnid_to_idx.items()}


def build_backbone(image_size: int) -> torch.nn.Module:
    """Returns a ViT backbone compatible with stored checkpoints."""
    # model = vit_small(
    #     patch_size=14,
    #     img_size=518,
    #     block_chunks=0,
    #     init_values=1e-6,
    # )
    model = vit_base(
        patch_size=14,
        img_size=518,
        block_chunks=0,
        init_values=1e-6,
        num_register_tokens = 4,
    )
    return model


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k.split("module.", 1)[1]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model: torch.nn.Module, spec: ModelSpec) -> None:
    if not os.path.isfile(spec.checkpoint):
        raise FileNotFoundError(f"Checkpoint for {spec.name} not found: {spec.checkpoint}")
    state = torch.load(spec.checkpoint, map_location="cpu")
    if spec.state_key and spec.state_key in state:
        state = state[spec.state_key]
    elif spec.state_key is not None:
        raise KeyError(f"state_key '{spec.state_key}' was not found in checkpoint for {spec.name}.")
    if isinstance(state, dict):
        for candidate in ("state_dict", "model", "net", "encoder", "backbone"):
            if candidate in state and isinstance(state[candidate], dict):
                state = state[candidate]
                break
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint for {spec.name} does not contain a state dict.")
    state = _strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[{spec.name}] Missing keys ({len(missing)}): {missing}")
    if unexpected:
        print(f"[{spec.name}] Unexpected keys ({len(unexpected)}): {unexpected}")


def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    description: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    autocast_device = "cuda" if device.startswith("cuda") else "cpu"
    pbar = tqdm(loader, desc=description, ncols=120)
    with torch.no_grad():
        for images, target in pbar:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=autocast_device, enabled=device.startswith("cuda")):
                feats = model.forward_features(images)["x_norm_clstoken"]
            features.append(feats.cpu())
            labels.append(target)
    features_np = torch.cat(features, dim=0).numpy()
    labels_np = torch.cat(labels, dim=0).numpy()
    return features_np, labels_np


def compute_tsne(features: np.ndarray, cfg: ScriptConfig) -> np.ndarray:
    n_samples = features.shape[0]
    if n_samples < 3:
        raise ValueError("Need at least 3 samples to run t-SNE.")
    # Adaptive perplexity keeps the algorithm stable when the dataset is small,
    # while allowing a user override for denser / looser clusters.
    if cfg.tsne_perplexity is None:
        max_perplexity = max(5, min(30, n_samples // 3))
        perplexity = min(max_perplexity, n_samples - 1)
    else:
        perplexity = max(5.0, min(cfg.tsne_perplexity, float(n_samples - 1)))
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate=cfg.tsne_learning_rate,
        random_state=SEED,
        perplexity=perplexity,
        early_exaggeration=cfg.tsne_early_exaggeration,
        metric=cfg.tsne_metric,
    )
    coords = tsne.fit_transform(features)
    if cfg.tsne_standardize:
        coords = coords - coords.mean(axis=0, keepdims=True)
        std = coords.std(axis=0, keepdims=True)
        std[std == 0.0] = 1.0
        coords = coords / std
    return coords


def plot_embeddings(
    ax: plt.Axes,
    coords: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    show_grid: bool = True,
    show_legend: bool = False,
    legend_labels: Sequence[str] | None = None,
    legend_max_items: int | None = 20,
    legend_columns: int = 1,
    legend_loc: str = "center left",
    legend_bbox: tuple[float, float] = (1.02, 1.0),
    legend_marker_size: float = 6.0,
    legend_fontsize: float | None = None,
    axis_limit: float | None = None,
    grid_major_step: float | None = None,
    grid_minor_step: float | None = None,
    cmap: matplotlib.colors.Colormap | None = None,
) -> None:
    if cmap is None:
        cmap = plt.get_cmap("tab20", len(class_names))
    norm = matplotlib.colors.BoundaryNorm(np.arange(len(class_names) + 1) - 0.5, cmap.N)
    if axis_limit:
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
    ax.set_axisbelow(True)
    ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=10, cmap=cmap, norm=norm, alpha=0.85, linewidths=0.0)
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    if show_grid:
        if grid_major_step:
            major_locator = MultipleLocator(grid_major_step)
            ax.xaxis.set_major_locator(major_locator)
            ax.yaxis.set_major_locator(major_locator)
        if grid_minor_step:
            minor_locator = MultipleLocator(grid_minor_step)
            ax.xaxis.set_minor_locator(minor_locator)
            ax.yaxis.set_minor_locator(minor_locator)
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.9, color="#b0b0b0")
        if grid_minor_step:
            ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.5, color="#d0d0d0")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if show_legend and class_names:
        handles = build_legend_handles(
            cmap=cmap,
            class_names=class_names,
            legend_labels=legend_labels,
            legend_max_items=legend_max_items,
            legend_marker_size=legend_marker_size,
        )
        if handles:
            ax.legend(
                handles=handles,
                loc=legend_loc,
                bbox_to_anchor=legend_bbox,
                frameon=False,
                ncol=legend_columns,
                title="Classes" if legend_columns == 1 else None,
                prop={"size": legend_fontsize} if legend_fontsize else None,
            )


def build_legend_handles(
    cmap: matplotlib.colors.Colormap,
    class_names: Sequence[str],
    legend_labels: Sequence[str] | None,
    legend_max_items: int | None,
    legend_marker_size: float,
) -> list[Line2D]:
    label_source = legend_labels if legend_labels is not None else class_names
    limit = legend_max_items if legend_max_items and legend_max_items > 0 else len(class_names)
    usable = min(len(class_names), len(label_source), limit)
    handles: list[Line2D] = []
    for idx in range(usable):
        color = cmap(idx)
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=legend_marker_size,
                markerfacecolor=color,
                markeredgewidth=0.0,
                label=label_source[idx],
            )
        )
    return handles


def analyze_features(features: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    """Compute class-separation metrics on encoder features."""
    metrics: dict[str, float | None] = {}
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}

    def _safe_metric(func, name: str) -> None:
        try:
            value = float(func(features, labels))
        except Exception:
            metrics[name] = None
        else:
            metrics[name] = value

    _safe_metric(silhouette_score, "silhouette")
    _safe_metric(calinski_harabasz_score, "calinski_harabasz")
    _safe_metric(davies_bouldin_score, "davies_bouldin")
    return metrics


def main() -> None:
    cfg = SCRIPT_CFG
    label_mapping = load_label_mapping(cfg.label_mapping_path, cfg.imagenet_index_path)
    dataset = LimitedImageNet(
        root=cfg.data_root,
        samples_per_class=cfg.samples_per_class,
        num_classes=cfg.num_classes,
        class_seed=cfg.class_seed,
        image_size=cfg.image_size,
        extensions=cfg.image_extensions,
    )
    print(
        f"Loaded {len(dataset)} samples from {len(dataset.class_names)} classes "
        f"(limit {cfg.samples_per_class} per class)."
    )
    legend_labels = [label_mapping.get(name, name) for name in dataset.class_names]
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    metrics_log: list[tuple[str, dict[str, float | None]]] = []
    plot_payload: list[tuple[int, ModelSpec, np.ndarray, np.ndarray]] = []
    for idx, spec in enumerate(MODEL_SPECS):
        print(f"[{spec.name}] Preparing backbone and loading weights from {spec.checkpoint}")
        model = build_backbone(cfg.image_size).to(cfg.device)
        load_checkpoint(model, spec)
        feats, lbls = extract_features(model, loader, cfg.device, description=f"{spec.name} inference")
        feature_metrics = analyze_features(feats, lbls)
        metrics_log.append((spec.name, feature_metrics))
        metric_msg = ", ".join(
            f"{k}={v:.4f}" if v is not None else f"{k}=None" for k, v in feature_metrics.items()
        )
        print(f"[{spec.name}] Feature metrics: {metric_msg}")
        coords = compute_tsne(feats, cfg)
        plot_payload.append((idx, spec, coords, lbls))

    if not plot_payload:
        print("No models were processed; skipping visualization.")
        return

    num_models = len(plot_payload)
    fig, axes = plt.subplots(
        1,
        num_models,
        figsize=(6 * num_models, 6),
        squeeze=False,
    )
    axes = axes.flatten()
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    shared_cmap = plt.get_cmap("tab20", len(dataset.class_names))
    clamped_fill = None
    if cfg.axis_fill_ratio is not None:
        clamped_fill = min(max(cfg.axis_fill_ratio, 0.1), 0.999)
    for axis_idx, (axis, (idx, spec, coords, lbls)) in enumerate(zip(axes, plot_payload)):
        coords_to_plot = coords
        if (
            cfg.axis_limit
            and cfg.axis_limit > 0
            and clamped_fill
            and coords.size
        ):
            max_abs = float(np.abs(coords).max())
            if max_abs > 0:
                coords_to_plot = coords * ((cfg.axis_limit * clamped_fill) / max_abs)
        plot_embeddings(
            ax=axis,
            coords=coords_to_plot,
            labels=lbls,
            class_names=dataset.class_names,
            show_grid=True,
            show_legend=False,
            legend_labels=legend_labels,
            legend_max_items=cfg.legend_max_items,
            legend_columns=cfg.legend_columns,
            legend_loc=cfg.legend_loc,
            legend_bbox=cfg.legend_bbox,
            legend_marker_size=cfg.legend_marker_size,
            legend_fontsize=cfg.legend_fontsize,
            axis_limit=cfg.axis_limit,
            grid_major_step=cfg.grid_major_step,
            grid_minor_step=cfg.grid_minor_step,
            cmap=shared_cmap,
        )
        label_text = panel_labels[axis_idx] if axis_idx < len(panel_labels) else f"({axis_idx + 1})"
        axis.set_title("")
        axis.text(0.5, -0.055, label_text, transform=axis.transAxes, fontsize=13, ha="center", va="top")

    # Provide a tight gap between subplots and extra bottom margin for the shared legend.
    bottom_margin = 0.08 if cfg.show_legend else 0.05
    fig.subplots_adjust(wspace=0.06, bottom=bottom_margin, left=0.06, right=0.98, top=0.98)

    if cfg.show_legend:
        handles = build_legend_handles(
            cmap=shared_cmap,
            class_names=dataset.class_names,
            legend_labels=legend_labels,
            legend_max_items=cfg.legend_max_items,
            legend_marker_size=cfg.legend_marker_size,
        )
        if handles:
            fig.legend(
                handles=handles,
                loc=cfg.legend_loc,
                bbox_to_anchor=cfg.legend_bbox,
                frameon=False,
                ncol=cfg.legend_columns,
                prop={"size": cfg.legend_fontsize},
            )

    out_path = Path(cfg.figure_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg.figure_dpi, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined t-SNE figure saved to {out_path} and {pdf_path}")

    print("=== Feature Quality Summary ===")
    for name, metrics in metrics_log:
        summary = ", ".join(
            f"{k}={v:.4f}" if v is not None else f"{k}=None" for k, v in metrics.items()
        )
        print(f"{name}: {summary}")


if __name__ == "__main__":
    main()
