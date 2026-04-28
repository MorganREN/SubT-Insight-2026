"""
prepare_roboflow_crack_partial_raw.py

Convert a Roboflow semantic-segmentation crack dataset into this project's raw
segmentation layout.

This converter is intentionally partial-label by default:
    tunnel-crack pixels -> 1 crack
    all non-crack pixels -> 255 ignore

That prevents a crack-only external dataset from incorrectly teaching the model
that unannotated leakage/spalling/segment damage pixels are background.

Expected Roboflow "png-mask-semantic" zip layout:
    train/{image}.jpg
    train/{image}_mask.png
    train/_classes.csv
    valid/{image}.jpg
    valid/{image}_mask.png
    test/{image}.jpg
    test/{image}_mask.png

Recommended for extra training data:
    python data_tools/prepare_roboflow_crack_partial_raw.py \
      --zip_path dataset/tunnel-crack-wxclm.v6i.png-mask-semantic.zip \
      --output_dir dataset/roboflow_tunnel_crack_partial_raw \
      --merge_splits_to_train
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_ZIP = Path("dataset/tunnel-crack-wxclm.v6i.png-mask-semantic.zip")
DEFAULT_SOURCE = Path("dataset/roboflow_tunnel_crack_source")
DEFAULT_OUTPUT = Path("dataset/roboflow_tunnel_crack_partial_raw")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PROJECT_CRACK_ID = 1
IGNORE_ID = 255


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _extract_zip(zip_path: Path, source_dir: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Zip not found: {zip_path}\n"
            "Download the Roboflow png-mask-semantic export into dataset/ first, "
            "or pass --zip_path."
        )
    _clean_dir(source_dir)
    with ZipFile(zip_path) as zf:
        zf.extractall(source_dir)


def _read_classes_csv(split_dir: Path) -> dict[int, str]:
    csv_path = split_dir / "_classes.csv"
    if not csv_path.exists():
        return {}

    classes: dict[int, str] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]

    for row in rows:
        if len(row) < 2:
            continue
        try:
            idx = int(row[0])
        except ValueError:
            continue
        classes[idx] = row[1].strip()
    return classes


def _find_crack_class_id(source_dir: Path, class_name_hint: str) -> int:
    all_classes: dict[int, str] = {}
    for split_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
        all_classes.update(_read_classes_csv(split_dir))

    if not all_classes:
        raise RuntimeError("Could not find any _classes.csv in the extracted Roboflow dataset.")

    hint = class_name_hint.lower()
    for idx, name in sorted(all_classes.items()):
        if name.lower() == hint:
            return idx
    for idx, name in sorted(all_classes.items()):
        if "crack" in name.lower():
            return idx

    raise RuntimeError(f"Could not identify crack class from classes: {all_classes}")


def _image_paths(split_dir: Path) -> list[Path]:
    return sorted(
        p for p in split_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_SUFFIXES
        and not p.name.endswith("_mask.png")
        and "_mask." not in p.name
        and not p.name.startswith("_")
        and ":Zone.Identifier" not in p.name
    )


def _pairs_for_split(source_dir: Path, split: str) -> list[tuple[Path, Path]]:
    split_dir = source_dir / split
    if not split_dir.exists():
        return []

    pairs: list[tuple[Path, Path]] = []
    for image_path in _image_paths(split_dir):
        mask_path = image_path.with_name(f"{image_path.stem}_mask.png")
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


def _convert_mask(mask_path: Path, crack_class_id: int, background_target: str) -> np.ndarray:
    raw = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    out = np.full(raw.shape, IGNORE_ID, dtype=np.uint8)
    out[raw == crack_class_id] = PROJECT_CRACK_ID
    if background_target == "background":
        out[raw == 0] = 0
    return out


def _save_visualization(image_path: Path, mask: np.ndarray, out_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    crack = Image.fromarray((mask == PROJECT_CRACK_ID).astype(np.uint8) * 180, mode="L")
    overlay.paste((255, 0, 0, 180), mask=crack)
    blended = Image.alpha_composite(image.convert("RGBA"), overlay)

    draw = ImageDraw.Draw(blended)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((8, 8, 122, 34), fill=(0, 0, 0, 150))
    draw.text((14, 12), "crack", fill=(255, 255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    blended.convert("RGB").save(out_path, quality=95)


def _prepare_dirs(output_dir: Path, splits: list[str]) -> dict[str, dict[str, Path]]:
    _clean_dir(output_dir)
    dirs: dict[str, dict[str, Path]] = {"img": {}, "ann": {}, "vis": {}}
    for split in splits:
        dirs["img"][split] = output_dir / "img_dir" / split
        dirs["ann"][split] = output_dir / "ann_dir" / split
        dirs["vis"][split] = output_dir / "visualization" / split
        for path in (dirs["img"][split], dirs["ann"][split], dirs["vis"][split]):
            path.mkdir(parents=True, exist_ok=True)
    return dirs


def _convert_pairs(
    pairs: list[tuple[Path, Path]],
    out_split: str,
    dirs: dict[str, dict[str, Path]],
    crack_class_id: int,
    background_target: str,
    prefix: str,
) -> dict:
    split_stats = {
        "images": 0,
        "crack_pixels": 0,
        "ignore_pixels": 0,
        "background_pixels": 0,
    }
    used_stems: set[str] = set()

    for image_path, mask_path in pairs:
        stem = f"{prefix}{image_path.stem}" if prefix else image_path.stem
        while stem in used_stems:
            stem = f"{stem}_dup"
        used_stems.add(stem)

        mask = _convert_mask(mask_path, crack_class_id, background_target)
        image = Image.open(image_path).convert("RGB")
        if image.size != (mask.shape[1], mask.shape[0]):
            image = image.resize((mask.shape[1], mask.shape[0]), Image.Resampling.LANCZOS)

        image.save(dirs["img"][out_split] / f"{stem}.jpg", quality=95)
        Image.fromarray(mask, mode="L").save(dirs["ann"][out_split] / f"{stem}.png")
        _save_visualization(image_path, mask, dirs["vis"][out_split] / f"{stem}.jpg")

        split_stats["images"] += 1
        split_stats["crack_pixels"] += int((mask == PROJECT_CRACK_ID).sum())
        split_stats["ignore_pixels"] += int((mask == IGNORE_ID).sum())
        split_stats["background_pixels"] += int((mask == 0).sum())

    return split_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Roboflow crack-only semantic masks to partial-label raw data.")
    parser.add_argument("--zip_path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument(
        "--merge_splits_to_train",
        action="store_true",
        help="Write all Roboflow splits into img_dir/train and ann_dir/train. Recommended for external partial labels.",
    )
    parser.add_argument(
        "--background_target",
        choices=["ignore", "background"],
        default="ignore",
        help="How to map non-crack/background pixels. Use ignore for crack-only datasets.",
    )
    parser.add_argument("--class_name_hint", default="tunnel-crack")
    parser.add_argument("--summary_json", type=Path, default=None)
    args = parser.parse_args()

    print(f"Extracting {args.zip_path} -> {args.source_dir}")
    _extract_zip(args.zip_path, args.source_dir)

    crack_class_id = _find_crack_class_id(args.source_dir, args.class_name_hint)
    print(f"Roboflow crack class id: {crack_class_id}")

    output_splits = ["train"] if args.merge_splits_to_train else args.splits
    dirs = _prepare_dirs(args.output_dir, output_splits)

    summary = {
        "zip_path": str(args.zip_path),
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "crack_class_id": crack_class_id,
        "project_crack_id": PROJECT_CRACK_ID,
        "background_target": args.background_target,
        "merge_splits_to_train": args.merge_splits_to_train,
        "splits": {},
    }

    for split in args.splits:
        pairs = _pairs_for_split(args.source_dir, split)
        if not pairs:
            print(f"[{split}] no image/mask pairs found, skip")
            continue
        out_split = "train" if args.merge_splits_to_train else split
        prefix = f"{split}_" if args.merge_splits_to_train else ""
        stats = _convert_pairs(
            pairs,
            out_split,
            dirs,
            crack_class_id,
            args.background_target,
            prefix,
        )
        summary["splits"][split] = {"output_split": out_split, **stats}
        print(f"[{split}] {stats['images']} images -> {args.output_dir}/img_dir/{out_split}")

    if args.summary_json is None:
        args.summary_json = args.output_dir / "summary.json"
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary: {args.summary_json}")


if __name__ == "__main__":
    main()
