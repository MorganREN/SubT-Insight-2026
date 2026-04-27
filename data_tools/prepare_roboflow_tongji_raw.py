"""
prepare_roboflow_tongji_raw.py

Convert the Roboflow tongji-tunnel semantic-mask export into this project's
raw segmentation layout, then remove images that duplicate the existing Tongji
data already present in this workspace.

The source zip is expected to be a Roboflow "png-mask-semantic" export:
    train/{image}.jpg
    train/{image}_mask.png
    train/_classes.csv

Output:
    dataset/roboflow_tongji_tunnel_raw_unfiltered/
    dataset/roboflow_tongji_tunnel_raw/
    dataset/roboflow_tongji_tunnel_duplicate_report.csv
    dataset/roboflow_tongji_tunnel_duplicate_summary.json

Default class mapping:
    background  -> 0 background
    cracks-MKjm -> 1 crack
    leakage     -> 3 leakage_w   (可用 --leakage_target ignore 改为 255)
    spalling    -> 5 lining_falling_off (可用 --spalling_target segment_damage 改为 6)

Run (推荐的 final 配置：leakage ignore，spalling as segment_damage):
    conda run -n subt-2026 python data_tools/prepare_roboflow_tongji_raw.py \
      --spalling_target segment_damage \
      --leakage_target ignore \
      --unfiltered_dir dataset/roboflow_tongji_tunnel_raw_final_unfiltered \
      --clean_dir dataset/roboflow_tongji_tunnel_raw_final \
      --report_csv dataset/roboflow_tongji_tunnel_duplicate_report_final.csv \
      --summary_json dataset/roboflow_tongji_tunnel_duplicate_summary_final.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage.metrics import structural_similarity as ssim


ZIP_PATH = Path("dataset/tongji-tunnel.v1i.png-mask-semantic.zip")
SOURCE_DIR = Path("dataset/roboflow_tongji_tunnel_source")
UNFILTERED_DIR = Path("dataset/roboflow_tongji_tunnel_raw_unfiltered")
CLEAN_DIR = Path("dataset/roboflow_tongji_tunnel_raw")
REPORT_CSV = Path("dataset/roboflow_tongji_tunnel_duplicate_report.csv")
SUMMARY_JSON = Path("dataset/roboflow_tongji_tunnel_duplicate_summary.json")

CLASS_DEFECTS = {
    1: "crack",
    2: "leakage_b",
    3: "leakage_w",
    4: "leakage_g",
    5: "lining_falling_off",
    6: "segment_damage",
}

CLASS_COLORS_RGB: dict[int, tuple[int, int, int]] = {
    1: (255, 0, 0),
    2: (255, 128, 0),
    3: (0, 0, 255),
    4: (0, 255, 255),
    5: (255, 255, 0),
    6: (255, 0, 255),
}

DEFAULT_ROBOFLOW_TO_PROJECT = {
    0: 0,
    1: 1,
    2: 3,
    3: 5,
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}


@dataclass
class ImageFingerprint:
    path: str
    root: str
    width: int
    height: int
    sha256: str
    phash: int
    dhash: int


@dataclass
class DuplicateMatch:
    roboflow_image: str
    roboflow_mask: str
    reference_image: str
    reference_root: str
    decision: str
    phash_distance: int
    dhash_distance: int
    ssim: float


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _extract_zip(zip_path: Path, source_dir: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Roboflow zip not found: {zip_path}")
    _clean_dir(source_dir)
    with ZipFile(zip_path) as zf:
        zf.extractall(source_dir)


def _image_paths(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix in IMAGE_SUFFIXES
        and not p.name.endswith("_mask.png")
        and not p.name.endswith(".json:Zone.Identifier")
        and ":Zone.Identifier" not in p.name
    )


def _roboflow_pairs(source_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for image_path in _image_paths(source_dir):
        mask_path = image_path.with_name(f"{image_path.stem}_mask.png")
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


def _read_rgb(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path).convert("RGB"))


def _sha256_image(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(str(rgb.size).encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _phash(image: Image.Image) -> int:
    gray = np.array(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    dct = cv2.dct(gray)
    low = dct[:8, :8].copy()
    flat = low.flatten()
    median = np.median(flat[1:])
    bits = flat > median
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


def _dhash(image: Image.Image) -> int:
    gray = np.array(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.uint8)
    bits = gray[:, :-1] > gray[:, 1:]
    out = 0
    for bit in bits.flatten():
        out = (out << 1) | int(bit)
    return out


def _hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def _fingerprint(path: Path, root: Path) -> ImageFingerprint | None:
    try:
        image = _read_rgb(path)
    except Exception:
        return None
    return ImageFingerprint(
        path=str(path),
        root=str(root),
        width=image.size[0],
        height=image.size[1],
        sha256=_sha256_image(image),
        phash=_phash(image),
        dhash=_dhash(image),
    )


def _build_reference_fingerprints(reference_roots: list[Path]) -> list[ImageFingerprint]:
    refs: list[ImageFingerprint] = []
    for root in reference_roots:
        if not root.exists():
            continue
        for path in _image_paths(root):
            fp = _fingerprint(path, root)
            if fp is not None:
                refs.append(fp)
    return refs


def _ssim_resized(path_a: Path, path_b: Path, size: tuple[int, int]) -> float:
    a = np.array(_read_rgb(path_a).convert("L").resize(size, Image.Resampling.LANCZOS), dtype=np.uint8)
    b = np.array(_read_rgb(path_b).convert("L").resize(size, Image.Resampling.LANCZOS), dtype=np.uint8)
    return float(ssim(a, b, data_range=255))


def _find_duplicate(
    image_path: Path,
    references: list[ImageFingerprint],
    *,
    phash_threshold: int,
    dhash_threshold: int,
    ssim_threshold: float,
    top_k: int,
) -> DuplicateMatch | None:
    fp = _fingerprint(image_path, image_path.parent)
    if fp is None:
        return None

    exact = [ref for ref in references if ref.sha256 == fp.sha256]
    if exact:
        ref = exact[0]
        return DuplicateMatch(
            roboflow_image=str(image_path),
            roboflow_mask=str(image_path.with_name(f"{image_path.stem}_mask.png")),
            reference_image=ref.path,
            reference_root=ref.root,
            decision="exact_sha256",
            phash_distance=0,
            dhash_distance=0,
            ssim=1.0,
        )

    candidates = []
    for ref in references:
        pd = _hamming(fp.phash, ref.phash)
        dd = _hamming(fp.dhash, ref.dhash)
        score = pd + dd
        if pd <= phash_threshold or dd <= dhash_threshold:
            candidates.append((score, pd, dd, ref))
    candidates.sort(key=lambda x: x[0])

    for _score, pd, dd, ref in candidates[:top_k]:
        sim = _ssim_resized(image_path, Path(ref.path), (fp.width, fp.height))
        if sim >= ssim_threshold:
            return DuplicateMatch(
                roboflow_image=str(image_path),
                roboflow_mask=str(image_path.with_name(f"{image_path.stem}_mask.png")),
                reference_image=ref.path,
                reference_root=ref.root,
                decision="perceptual_hash_ssim",
                phash_distance=pd,
                dhash_distance=dd,
                ssim=sim,
            )

    return None


def _convert_mask(mask_path: Path, image_size: tuple[int, int], mapping: dict[int, int]) -> np.ndarray:
    raw = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    if raw.shape[::-1] != image_size:
        raw = np.array(Image.fromarray(raw).resize(image_size, Image.Resampling.NEAREST), dtype=np.uint8)
    out = np.zeros_like(raw, dtype=np.uint8)
    for raw_value, mapped_value in mapping.items():
        out[raw == raw_value] = mapped_value
    return out


def _make_overlay(image_arr: np.ndarray, mask_arr: np.ndarray, intensity: int = 115) -> np.ndarray:
    color_mask = np.zeros_like(image_arr, dtype=np.uint8)
    for cls_id, color in CLASS_COLORS_RGB.items():
        color_mask[mask_arr == cls_id] = color
    overlay = image_arr.astype(np.int16).copy()
    hit = mask_arr > 0
    overlay[hit] = np.clip(
        image_arr[hit].astype(np.int16) - intensity + color_mask[hit].astype(np.int16),
        0,
        255,
    )
    return overlay.astype(np.uint8)


def _add_legend(overlay_arr: np.ndarray, present: set[int]) -> np.ndarray:
    items = [(cid, CLASS_DEFECTS[cid], CLASS_COLORS_RGB[cid]) for cid in sorted(present) if cid in CLASS_DEFECTS]
    if not items:
        return overlay_arr

    img = Image.fromarray(overlay_arr)
    w, h = img.size
    cols = min(len(items), 3)
    rows = int(np.ceil(len(items) / cols))
    pad, row_h, swatch = 8, 24, 14
    canvas = Image.new("RGB", (w, h + pad * 2 + rows * row_h), (245, 245, 245))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line([(0, h), (w, h)], fill=(180, 180, 180), width=1)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    col_w = max(1, w // cols)
    for idx, (cid, name, color) in enumerate(items):
        x0 = (idx % cols) * col_w + pad
        y0 = h + pad + (idx // cols) * row_h
        draw.rectangle([x0, y0, x0 + swatch, y0 + swatch], fill=color, outline=(0, 0, 0))
        draw.text((x0 + swatch + 4, y0 + 1), f"{cid}:{name}", fill=(30, 30, 30), font=font)
    return np.array(canvas, dtype=np.uint8)


def _init_raw_dirs(root: Path) -> dict[str, Path]:
    _clean_dir(root)
    dirs = {
        "img": root / "img_dir" / "train",
        "ann": root / "ann_dir" / "train",
        "overlay": root / "visualization" / "train" / "mask_overlay",
        "segmented": root / "visualization" / "train" / "segmented",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _convert_pairs(
    pairs: list[tuple[Path, Path]],
    output_root: Path,
    mapping: dict[int, int],
    skip_images: set[str] | None = None,
) -> int:
    dirs = _init_raw_dirs(output_root)
    skip_images = skip_images or set()
    converted = 0

    for image_path, mask_path in pairs:
        if str(image_path) in skip_images:
            continue
        stem = image_path.stem
        image = _read_rgb(image_path)
        image_arr = np.array(image, dtype=np.uint8)
        mask_arr = _convert_mask(mask_path, image.size, mapping)

        Image.fromarray(image_arr).save(dirs["img"] / f"{stem}.jpg", quality=95)
        Image.fromarray(mask_arr, mode="L").save(dirs["ann"] / f"{stem}.png")

        present = set(int(v) for v in np.unique(mask_arr) if v > 0)
        Image.fromarray(_add_legend(_make_overlay(image_arr, mask_arr), present)).save(dirs["overlay"] / f"{stem}.png")

        segmented = np.zeros_like(image_arr)
        segmented[mask_arr > 0] = image_arr[mask_arr > 0]
        Image.fromarray(segmented).save(dirs["segmented"] / f"{stem}.png")
        converted += 1

    return converted


def _write_report(matches: list[DuplicateMatch], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(DuplicateMatch.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for match in matches:
            writer.writerow(asdict(match))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Roboflow Tongji raw data and remove duplicates.")
    parser.add_argument("--zip_path", type=Path, default=ZIP_PATH)
    parser.add_argument("--source_dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--unfiltered_dir", type=Path, default=UNFILTERED_DIR)
    parser.add_argument("--clean_dir", type=Path, default=CLEAN_DIR)
    parser.add_argument("--report_csv", type=Path, default=REPORT_CSV)
    parser.add_argument("--summary_json", type=Path, default=SUMMARY_JSON)
    parser.add_argument(
        "--reference_roots",
        nargs="+",
        type=Path,
        default=[
            Path("dataset/tongji"),
            Path("dataset/tongji_data_raw/img_dir"),
            Path("dataset/tongji_data_awesome/img_dir"),
        ],
        help="Image roots used to detect duplicates. Defaults include raw Tongji and derived Tongji splits.",
    )
    parser.add_argument("--phash_threshold", type=int, default=6)
    parser.add_argument("--dhash_threshold", type=int, default=6)
    parser.add_argument("--ssim_threshold", type=float, default=0.97)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument(
        "--spalling_target",
        choices=["lining_falling_off", "segment_damage"],
        default="lining_falling_off",
        help="Map Roboflow spalling to project class 5 (lining_falling_off) or 6 (segment_damage).",
    )
    parser.add_argument(
        "--leakage_target",
        choices=["leakage_w", "leakage_b", "ignore"],
        default="leakage_w",
        help="Map Roboflow leakage to leakage_w(3) / leakage_b(2) / ignore(255).",
    )
    args = parser.parse_args()

    mapping = dict(DEFAULT_ROBOFLOW_TO_PROJECT)
    mapping[3] = 5 if args.spalling_target == "lining_falling_off" else 6
    mapping[2] = {"leakage_w": 3, "leakage_b": 2, "ignore": 255}[args.leakage_target]

    print(f"Extracting {args.zip_path} -> {args.source_dir}")
    _extract_zip(args.zip_path, args.source_dir)

    pairs = _roboflow_pairs(args.source_dir)
    if not pairs:
        raise RuntimeError(f"No Roboflow image/mask pairs found under {args.source_dir}")
    print(f"Found Roboflow pairs: {len(pairs)}")

    unfiltered_count = _convert_pairs(pairs, args.unfiltered_dir, mapping)
    print(f"Unfiltered raw dataset: {unfiltered_count} images -> {args.unfiltered_dir}")

    print("Building reference fingerprints...")
    references = _build_reference_fingerprints(args.reference_roots)
    print(f"Reference images: {len(references)}")

    matches: list[DuplicateMatch] = []
    for idx, (image_path, _mask_path) in enumerate(pairs, 1):
        match = _find_duplicate(
            image_path,
            references,
            phash_threshold=args.phash_threshold,
            dhash_threshold=args.dhash_threshold,
            ssim_threshold=args.ssim_threshold,
            top_k=args.top_k,
        )
        if match is not None:
            matches.append(match)
        if idx % 50 == 0 or idx == len(pairs):
            print(f"  checked {idx}/{len(pairs)} duplicates={len(matches)}")

    duplicate_images = {m.roboflow_image for m in matches}
    clean_count = _convert_pairs(pairs, args.clean_dir, mapping, skip_images=duplicate_images)
    _write_report(matches, args.report_csv)

    by_root: dict[str, int] = {}
    by_decision: dict[str, int] = {}
    for match in matches:
        by_root[match.reference_root] = by_root.get(match.reference_root, 0) + 1
        by_decision[match.decision] = by_decision.get(match.decision, 0) + 1

    summary = {
        "zip_path": str(args.zip_path),
        "source_dir": str(args.source_dir),
        "unfiltered_dir": str(args.unfiltered_dir),
        "clean_dir": str(args.clean_dir),
        "roboflow_pairs": len(pairs),
        "unfiltered_count": unfiltered_count,
        "duplicate_count": len(matches),
        "clean_count": clean_count,
        "reference_roots": [str(p) for p in args.reference_roots],
        "reference_count": len(references),
        "thresholds": {
            "phash_threshold": args.phash_threshold,
            "dhash_threshold": args.dhash_threshold,
            "ssim_threshold": args.ssim_threshold,
            "top_k": args.top_k,
        },
        "class_mapping": mapping,
        "spalling_target": args.spalling_target,
        "duplicates_by_reference_root": by_root,
        "duplicates_by_decision": by_decision,
    }
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Duplicates removed from clean dataset: {len(matches)}")
    print(f"Clean raw dataset: {clean_count} images -> {args.clean_dir}")
    print(f"Duplicate report: {args.report_csv}")
    print(f"Summary: {args.summary_json}")


if __name__ == "__main__":
    main()
