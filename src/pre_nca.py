from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
import shutil
from typing import Dict, Iterable, List, Tuple

import h5py
import hdf5plugin  # noqa: F401  # registers BLOSC filters for h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


# Dataset location and output configuration.
DATASET_ROOT = Path(os.getenv("PRE_NCA_DATASET_ROOT", "/data/storage/jianwen/ncaltech101"))
OUTPUT_ROOT = Path(os.getenv("PRE_NCA_OUTPUT_ROOT", str(DATASET_ROOT / "event_frames")))
ORIGINAL_ROOT = Path(
    os.getenv("PRE_NCA_ORIGINAL_ROOT", "/data/storage/jianwen/caltech-101/101_ObjectCategories")
)

ORIGINAL_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

OVERWRITE = os.getenv("PRE_NCA_OVERWRITE", "").strip().lower() in {"1", "true", "yes", "y"}

# Mapping from the desired split name to the original dataset split(s).
# - new train := original training + validation
# - new valid := original testing
SPLIT_SOURCES: Dict[str, Tuple[str, ...]] = {
    "train": ("training", "validation"),
    "valid": ("testing",),
}

# Allow overriding worker count through an environment variable, otherwise
# default to a small multiple of available cores to avoid saturating IO.
DEFAULT_WORKERS = max(1, min(16, (os.cpu_count() or 1)))
NUM_WORKERS = int(os.getenv("PRE_NCA_WORKERS", DEFAULT_WORKERS))

# Percentile used to clamp event counts before normalisation.
ACCUM_PERCENTILE = float(os.getenv("PRE_NCA_PERCENTILE", 99.0))


def accumulate_to_rgb(x: np.ndarray, y: np.ndarray, p: np.ndarray, shape, pct: float = 99.0) -> np.ndarray:
    """Convert event streams to a red/blue event image.

    Positive events (p=True) map to red, negative events map to blue, all over a white background.
    The percentile clamp stabilises very active pixels.
    """
    if len(shape) == 3:
        height, width = shape[:2]
    else:
        height, width = shape

    x = np.asarray(x)
    y = np.asarray(y)
    p = np.asarray(p, dtype=bool)

    mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[mask]
    y = y[mask]
    p = p[mask]

    pos = np.zeros((height, width), dtype=np.float32)
    neg = np.zeros((height, width), dtype=np.float32)

    if x.size:
        np.add.at(pos, (y[p], x[p]), 1)
        np.add.at(neg, (y[~p], x[~p]), 1)

    def _normalise(arr: np.ndarray) -> np.ndarray:
        if arr.max() <= 0:
            return arr
        thresh = np.percentile(arr[arr > 0], pct) if np.any(arr > 0) else float(arr.max())
        if thresh <= 0:
            thresh = float(arr.max())
        return np.clip(arr, 0, thresh) / thresh

    pos = _normalise(pos)
    neg = _normalise(neg)

    dominantly_pos = pos >= neg
    inten_pos = pos * dominantly_pos
    inten_neg = neg * (~dominantly_pos)

    red = np.ones((height, width), dtype=np.float32)
    green = np.ones((height, width), dtype=np.float32)
    blue = np.ones((height, width), dtype=np.float32)

    green -= inten_pos
    blue -= inten_pos

    red -= inten_neg
    green -= inten_neg

    rgb = np.stack([np.clip(red, 0, 1), np.clip(green, 0, 1), np.clip(blue, 0, 1)], axis=-1)
    return (rgb * 255).astype(np.uint8)


@lru_cache(maxsize=1)
def _original_lookup() -> Dict[Tuple[str, str], Path]:
    """Build a lookup from (class_name, sample_stem) to the original RGB image path."""
    if not ORIGINAL_ROOT.is_dir():
        raise FileNotFoundError(f"Missing original image root: {ORIGINAL_ROOT}")

    result: Dict[Tuple[str, str], Path] = {}
    for class_dir in sorted(ORIGINAL_ROOT.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.iterdir():
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in ORIGINAL_EXTS:
                continue
            key = (class_dir.name, img_path.stem)
            # Prefer the first occurrence; duplicates would indicate dataset issues.
            result.setdefault(key, img_path)
    return result


def _find_original_image(h5_path: Path) -> Path:
    """Locate the RGB counterpart for a given .h5 sample."""
    class_name = h5_path.parent.name
    lookup = _original_lookup()
    orig_path = lookup.get((class_name, h5_path.stem))
    if orig_path is None:
        raise FileNotFoundError(f"No matching original image for {class_name}/{h5_path.stem}")
    return orig_path


def _collect_jobs() -> List[Tuple[Path, Path]]:
    """Enumerate all (h5_path, out_dir) jobs across splits/classes."""
    jobs: List[Tuple[Path, Path]] = []

    for split, source_splits in SPLIT_SOURCES.items():
        for source in source_splits:
            source_root = DATASET_ROOT / source
            if not source_root.is_dir():
                raise FileNotFoundError(f"Missing source split directory: {source_root}")

            for class_dir in sorted(source_root.iterdir()):
                if not class_dir.is_dir():
                    continue

                target_dir = OUTPUT_ROOT / split / class_dir.name
                target_dir.mkdir(parents=True, exist_ok=True)

                for h5_path in sorted(class_dir.glob("*.h5")):
                    jobs.append((h5_path, target_dir))

    return jobs


def _convert_one(h5_path: Path, out_dir: Path) -> Tuple[str, str]:
    """Convert a single .h5 file into an event image PNG."""
    out_path = out_dir / f"{h5_path.stem}.png"
    orig_path = _find_original_image(h5_path)
    orig_target = out_dir / orig_path.name

    need_event = True
    need_original = True
    # if not need_event and not need_original:
    #     return "skip", str(out_path)

    try:
        if need_event:
            with h5py.File(h5_path, "r") as handle:
                events = handle["events"]

                width = int(events["width"][()])
                height = int(events["height"][()])

                xs = np.asarray(events["x"], dtype=np.int64)
                ys = np.asarray(events["y"], dtype=np.int64)
                ps = np.asarray(events["p"], dtype=np.int8)

            pol = ps > 0

            if xs.size:
                xmin = int(xs.min())
                xmax = int(xs.max())
                ymin = int(ys.min())
                ymax = int(ys.max())

                xs = xs - xmin
                ys = ys - ymin
                active_width = int(xmax - xmin + 1)
                active_height = int(ymax - ymin + 1)
            else:
                active_width = width
                active_height = height

            event_rgb = accumulate_to_rgb(
                xs,
                ys,
                pol,
                (active_height, active_width),
                pct=ACCUM_PERCENTILE,
            )

            with Image.open(orig_path) as orig_img:
                orig_width, orig_height = orig_img.size

            event_img = Image.fromarray(event_rgb, mode="RGB")
            if (orig_width, orig_height) != event_img.size:
                event_img = event_img.resize((orig_width, orig_height), resample=Image.BILINEAR)
            event_img.save(out_path)

        if need_original:
            shutil.copy2(orig_path, orig_target)

        return "ok", str(out_path)
    except Exception as exc:  # pragma: no cover - error path for logging
        return "error", f"{h5_path}: {exc}"


def _run_conversion(jobs: Iterable[Tuple[Path, Path]]) -> Tuple[int, int, List[str]]:
    """Execute conversion jobs in parallel and aggregate statistics."""
    jobs = list(jobs)
    if not jobs:
        return 0, 0, []

    errors: List[str] = []
    processed = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_map = {executor.submit(_convert_one, path, out_dir): path for path, out_dir in jobs}

        for fut in tqdm(
            as_completed(future_map),
            total=len(future_map),
            desc="Building event images",
        ):
            status, info = fut.result()
            if status == "ok":
                processed += 1
            elif status == "skip":
                skipped += 1
            else:
                errors.append(info)

    return processed, skipped, errors


def main() -> None:
    jobs = _collect_jobs()
    print(f"Discovered {len(jobs)} event samples to process.")
    processed, skipped, errors = _run_conversion(jobs)

    print(f"Finished: converted={processed}, skipped={skipped}, errors={len(errors)}.")
    if errors:
        print("Errors (showing up to 20):")
        for message in errors[:20]:
            print(f"  - {message}")


if __name__ == "__main__":
    main()
