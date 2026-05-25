#!/usr/bin/env python3
"""Compute Tongji instance-level morphology metrics and plot violin charts.

This is a compact standalone version of the morphology/violin logic from
notebooks/tunnel_defect_analysis.ipynb. It reads LabelMe JSON files under
dataset/tongji, computes shape descriptors for each polygon instance, and
plots compactness, solidity, and elongation from the real instance data.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset/tongji"
OUTPUT_DIR = DATASET_ROOT / "analysis_output"

FIGURE_DPI = 300
FIGURE_STYLE = "seaborn-v0_8-whitegrid"
PALETTE = "tab10"
FONT_SETTINGS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

SPLIT_DIRS = {
    "train": "train",
    "val": "val",
    "test": "test",
}

METRICS_TO_PLOT = [
    ("compactness", "Compactness (4πA/P²)", "Circularity → 1 = perfect circle"),
    ("solidity", "Solidity (area / convex hull)", "Higher = more filled/convex"),
    ("elongation", "Elongation (major/minor)", "Higher = more elongated/linear"),
]

OUTPUT_STEM = "05_morphological_violin_polygon"
MORPH_RECORDS_CSV = "morphological_instance_metrics_polygon.csv"


def load_labelme_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("shapes", [])
    return data


def polygon_morphology(points: list[list[float]]) -> dict[str, float] | None:
    contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if len(contour) < 3:
        return None

    area = abs(float(cv2.contourArea(contour)))
    perimeter = float(cv2.arcLength(contour, closed=True))
    if area < 5 or perimeter <= 0:
        return None

    hull = cv2.convexHull(contour)
    hull_area = abs(float(cv2.contourArea(hull)))
    moments = cv2.moments(contour)

    compactness = (4.0 * np.pi * area) / (perimeter**2)
    solidity = area / hull_area if hull_area > 0 else np.nan

    if moments["m00"] > 0:
        cov_xx = moments["mu20"] / moments["m00"]
        cov_yy = moments["mu02"] / moments["m00"]
        cov_xy = moments["mu11"] / moments["m00"]
        eigvals = np.linalg.eigvalsh(np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float))
        minor, major = np.maximum(eigvals, 0.0)
        elongation = float(np.sqrt(major / minor)) if minor > 0 else np.inf
    else:
        elongation = np.nan

    return {
        "pixel_area": area,
        "perimeter": perimeter,
        "compactness": compactness,
        "solidity": solidity,
        "elongation": elongation,
    }


def discover_splits(root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    found_split = False

    for split_name, folder_name in SPLIT_DIRS.items():
        split_dir = root / folder_name
        if not split_dir.exists():
            continue
        json_paths = sorted(split_dir.rglob("*.json"))
        if json_paths:
            result[split_name].extend(json_paths)
            found_split = True

    if not found_split:
        result["all"].extend(
            sorted(path for path in root.rglob("*.json") if OUTPUT_DIR not in path.parents)
        )

    return dict(result)


def scan_annotations(root: Path) -> pd.DataFrame:
    records = []
    label_counter: Counter[str] = Counter()
    split_map = discover_splits(root)

    print("Detected JSON files:", flush=True)
    for split_name, json_paths in split_map.items():
        print(f"  {split_name:<8}: {len(json_paths)}", flush=True)

    for split_name, json_paths in split_map.items():
        for json_path in json_paths:
            data = load_labelme_json(json_path)
            height = int(data.get("imageHeight", 0) or 0)
            width = int(data.get("imageWidth", 0) or 0)
            if height <= 0 or width <= 0:
                continue

            for shape in data["shapes"]:
                if shape.get("shape_type", "polygon") != "polygon":
                    continue

                points = shape.get("points", [])
                if len(points) < 3:
                    continue

                label = str(shape.get("label", "unknown")).strip()
                label_counter[label] += 1
                records.append(
                    {
                        "split": split_name,
                        "image_id": json_path.stem,
                        "label": label,
                        "img_h": height,
                        "img_w": width,
                        "points": points,
                    }
                )

    df_ann = pd.DataFrame(records)
    print(f"Total annotations: {len(df_ann)}", flush=True)
    print("Class counts:", flush=True)
    for label, count in sorted(label_counter.items(), key=lambda item: -item[1]):
        print(f"  {label:<22} {count}", flush=True)

    return df_ann


def compute_morphology(df_ann: pd.DataFrame) -> pd.DataFrame:
    morph_records = []

    for idx, row in df_ann.iterrows():
        if idx and idx % 250 == 0:
            print(f"  processed {idx}/{len(df_ann)} annotations...", flush=True)

        metrics = polygon_morphology(row["points"])
        if metrics is None:
            continue

        morph_records.append(
            {
                "split": row["split"],
                "image_id": row["image_id"],
                "label": row["label"],
                **metrics,
            }
        )

    df_morph = pd.DataFrame(morph_records)
    print(f"Computed morphology for {len(df_morph)} annotations.", flush=True)
    return df_morph


def plot_violins(df_morph: pd.DataFrame, output_dir: Path) -> None:
    plt.style.use(FIGURE_STYLE)
    plt.rcParams.update(FONT_SETTINGS)

    all_classes = sorted(df_morph["label"].unique())
    colors = plt.get_cmap(PALETTE)(np.linspace(0, 1, len(all_classes)))

    output_dir.mkdir(parents=True, exist_ok=True)

    for metric, ylabel, note in METRICS_TO_PLOT:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.suptitle(ylabel, fontweight="bold", fontsize=12, y=0.98)

        data_to_plot = []
        for cls in all_classes:
            vals = df_morph.loc[df_morph["label"] == cls, metric]
            vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) > 0:
                vals = vals.clip(upper=np.percentile(vals, 98))
            data_to_plot.append(vals.values)

        vparts = ax.violinplot(
            data_to_plot,
            positions=range(len(all_classes)),
            showmedians=True,
            showextrema=True,
        )

        for pc, color in zip(vparts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.75)

        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if key in vparts:
                vparts[key].set_color("0.25")
                vparts[key].set_linewidth(1.0)

        ax.set_xticks(range(len(all_classes)))
        ax.set_xticklabels(all_classes, rotation=30, ha="right", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        fig.text(0.14, 0.90, note, ha="left", va="top", fontsize=8.5, color="gray")

        fig.tight_layout(rect=(0, 0, 1, 0.89))
        for suffix in (".png", ".pdf", ".svg"):
            out_path = output_dir / f"{OUTPUT_STEM}_{metric}{suffix}"
            fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
            print(f"Saved: {out_path}", flush=True)
        plt.close(fig)


def main() -> None:
    df_ann = scan_annotations(DATASET_ROOT)
    df_morph = compute_morphology(df_ann)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / MORPH_RECORDS_CSV
    df_morph.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}", flush=True)

    plot_violins(df_morph, OUTPUT_DIR)


if __name__ == "__main__":
    main()
