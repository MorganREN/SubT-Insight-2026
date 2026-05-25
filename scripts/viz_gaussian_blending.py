"""
Visualize Gaussian blending used by tiled inference.

This is an explanatory figure generator rather than a model inference script.
It reuses the production tile layout and Gaussian window, then injects
deterministic simulated uncertainty at internal tile boundaries to illustrate
why weighted reconstruction suppresses stitching seams.

Outputs:
  paper_figures/gaussian_blending_process.{png,pdf,svg}

Usage:
  python scripts/viz_gaussian_blending.py
  python scripts/viz_gaussian_blending.py \
      --image_path dataset/tongji_data_raw/img_dir/test/test5.jpg
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictor.tiling import (  # noqa: E402
    _TILING_PARAMS,
    _assign_group,
    _make_gaussian_weight,
    _reflect_pad,
)


DEFAULT_IMAGE = PROJECT_ROOT / "dataset/tongji_data_raw/img_dir/test/test4.jpg"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "paper_figures"
EXPORTS = (("png", 260), ("pdf", None), ("svg", None))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


@dataclass
class TileLayout:
    image: np.ndarray
    padded: np.ndarray
    group: str
    patch_size: int
    stride: int
    pad_h: int
    pad_w: int
    positions: list[tuple[int, int]]
    gaussian: np.ndarray


def build_layout(image: np.ndarray, sigma_ratio: float) -> TileLayout:
    """Recreate the tile geometry and window used by tiled_predict."""
    height, width = image.shape[:2]
    group = _assign_group(height, width)
    params = _TILING_PARAMS[group]
    if params is None:
        raise ValueError(
            f"Image size {width}x{height} belongs to the '{group}' group, "
            "which is resized directly during inference and does not use "
            "Gaussian tiled blending. Choose an image with at least 200,000 pixels."
        )

    patch_size, stride = params
    pad_h = (
        (stride - (height - patch_size) % stride) % stride
        if height > patch_size
        else max(0, patch_size - height)
    )
    pad_w = (
        (stride - (width - patch_size) % stride) % stride
        if width > patch_size
        else max(0, patch_size - width)
    )
    padded = (
        _reflect_pad(image, pad_h, pad_w)
        if pad_h > 0 or pad_w > 0
        else image
    )
    padded_h, padded_w = padded.shape[:2]
    positions = [
        (y, x)
        for y in range(0, padded_h - patch_size + 1, stride)
        for x in range(0, padded_w - patch_size + 1, stride)
    ]
    return TileLayout(
        image=image,
        padded=padded,
        group=group,
        patch_size=patch_size,
        stride=stride,
        pad_h=pad_h,
        pad_w=pad_w,
        positions=positions,
        gaussian=_make_gaussian_weight(patch_size, sigma_ratio=sigma_ratio),
    )


def _internal_edge_profile(
    y: int,
    x: int,
    patch_size: int,
    padded_h: int,
    padded_w: int,
) -> np.ndarray:
    """Build uncertainty near internal tile boundaries, not image borders."""
    decay = max(1.0, patch_size * 0.085)
    distance = np.arange(patch_size, dtype=np.float32)
    start_ramp = np.exp(-distance / decay)
    end_ramp = start_ramp[::-1]
    profile = np.zeros((patch_size, patch_size), dtype=np.float32)

    if x > 0:
        profile = np.maximum(profile, start_ramp[np.newaxis, :])
    if x + patch_size < padded_w:
        profile = np.maximum(profile, end_ramp[np.newaxis, :])
    if y > 0:
        profile = np.maximum(profile, start_ramp[:, np.newaxis])
    if y + patch_size < padded_h:
        profile = np.maximum(profile, end_ramp[:, np.newaxis])
    return profile


def simulate_tile_output(
    patch: np.ndarray,
    y: int,
    x: int,
    layout: TileLayout,
    artifact_strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a deterministic illustrative color error at internal tile edges."""
    profile = _internal_edge_profile(
        y, x, layout.patch_size, layout.padded.shape[0], layout.padded.shape[1]
    )
    tint = rng.normal(size=3).astype(np.float32)
    tint /= max(float(np.abs(tint).max()), 1e-8)
    tile = patch.astype(np.float32) / 255.0
    tile += artifact_strength * profile[:, :, np.newaxis] * tint[np.newaxis, np.newaxis, :]
    return np.clip(tile, 0.0, 1.0)


def reconstruct_demo(
    layout: TileLayout,
    artifact_strength: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Blend simulated tile outputs with uniform and Gaussian weighting."""
    padded_h, padded_w = layout.padded.shape[:2]
    uniform_sum = np.zeros((padded_h, padded_w, 3), dtype=np.float32)
    gaussian_sum = np.zeros((padded_h, padded_w, 3), dtype=np.float32)
    uniform_weights = np.zeros((padded_h, padded_w), dtype=np.float32)
    gaussian_weights = np.zeros((padded_h, padded_w), dtype=np.float32)
    rng = np.random.default_rng(seed)
    center = (padded_h / 2.0, padded_w / 2.0)
    example_pos = min(
        layout.positions,
        key=lambda pos: (
            (pos[0] + layout.patch_size / 2.0 - center[0]) ** 2
            + (pos[1] + layout.patch_size / 2.0 - center[1]) ** 2
        ),
    )
    example_tile = None

    for y, x in layout.positions:
        patch = layout.padded[y:y + layout.patch_size, x:x + layout.patch_size]
        tile_output = simulate_tile_output(
            patch, y, x, layout, artifact_strength=artifact_strength, rng=rng
        )
        if (y, x) == example_pos:
            example_tile = tile_output.copy()
        uniform_sum[y:y + layout.patch_size, x:x + layout.patch_size] += tile_output
        uniform_weights[y:y + layout.patch_size, x:x + layout.patch_size] += 1.0
        gaussian_sum[y:y + layout.patch_size, x:x + layout.patch_size] += (
            tile_output * layout.gaussian[:, :, np.newaxis]
        )
        gaussian_weights[y:y + layout.patch_size, x:x + layout.patch_size] += layout.gaussian

    uniform_sum /= np.maximum(uniform_weights[:, :, np.newaxis], 1e-8)
    gaussian_sum /= np.maximum(gaussian_weights[:, :, np.newaxis], 1e-8)
    height, width = layout.image.shape[:2]
    assert example_tile is not None
    return (
        uniform_sum[:height, :width],
        gaussian_sum[:height, :width],
        gaussian_weights,
        example_tile,
        example_pos,
    )


def _style_image_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#555555")


def _show_rgb(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    ax.imshow(np.clip(image, 0.0, 1.0) if image.dtype.kind == "f" else image)
    _style_image_axis(ax, title)


def _add_tile_grid(ax: plt.Axes, layout: TileLayout) -> None:
    colors = ("#15AABF", "#F08C00", "#E64980")
    for idx, (y, x) in enumerate(layout.positions):
        color = colors[idx % len(colors)]
        ax.add_patch(
            Rectangle(
                (x, y),
                layout.patch_size,
                layout.patch_size,
                fill=False,
                edgecolor=color,
                linewidth=1.35,
            )
        )
        if len(layout.positions) <= 16:
            ax.text(
                x + 10,
                y + 24,
                f"T{idx + 1}",
                fontsize=7.5,
                color="white",
                bbox={"facecolor": color, "edgecolor": "none", "pad": 1.5, "alpha": 0.9},
            )
    height, width = layout.image.shape[:2]
    if layout.pad_h or layout.pad_w:
        ax.add_patch(
            Rectangle(
                (0, 0),
                width,
                height,
                fill=False,
                edgecolor="white",
                linestyle="--",
                linewidth=1.4,
            )
        )


def _select_zoom_box(layout: TileLayout) -> tuple[int, int, int]:
    """Choose an internal seam intersection for the qualitative close-up."""
    size = max(96, min(220, layout.patch_size // 3))
    candidates = sorted({x for _, x in layout.positions if x > 0})
    row_candidates = sorted({y for y, _ in layout.positions if y > 0})
    seam_x = candidates[len(candidates) // 2] if candidates else layout.image.shape[1] // 2
    seam_y = (
        row_candidates[len(row_candidates) // 2]
        if row_candidates
        else layout.image.shape[0] // 2
    )
    height, width = layout.image.shape[:2]
    left = int(np.clip(seam_x - size // 2, 0, max(0, width - size)))
    top = int(np.clip(seam_y - size // 2, 0, max(0, height - size)))
    return top, left, size


def build_figure(
    layout: TileLayout,
    uniform: np.ndarray,
    gaussian: np.ndarray,
    gaussian_weights: np.ndarray,
    example_tile: np.ndarray,
    example_pos: tuple[int, int],
    sigma_ratio: float,
    artifact_strength: float,
) -> plt.Figure:
    original = layout.image.astype(np.float32) / 255.0
    uniform_error = np.mean(np.abs(uniform - original), axis=2)
    gaussian_error = np.mean(np.abs(gaussian - original), axis=2)
    uniform_mae = float(uniform_error.mean())
    gaussian_mae = float(gaussian_error.mean())
    improvement = 100.0 * (uniform_mae - gaussian_mae) / max(uniform_mae, 1e-8)

    fig, axes = plt.subplots(2, 4, figsize=(18.5, 10.1))
    fig.subplots_adjust(left=0.035, right=0.975, top=0.84, bottom=0.075, wspace=0.20, hspace=0.24)
    fig.suptitle("Gaussian Blending for Tiled Inference", fontsize=20, fontweight="bold", y=0.965)
    fig.text(
        0.5,
        0.921,
        (
            f"Production layout: group={layout.group}  |  patch={layout.patch_size}  |  "
            f"stride={layout.stride}  |  tiles={len(layout.positions)}  |  "
            f"padding=(right {layout.pad_w}, bottom {layout.pad_h})"
        ),
        ha="center",
        fontsize=10,
        color="#444444",
    )

    axes[0, 0].imshow(layout.padded)
    _add_tile_grid(axes[0, 0], layout)
    _style_image_axis(axes[0, 0], "Original image + overlapping tiles")

    _show_rgb(
        axes[0, 1],
        example_tile,
        f"Example tile output T{layout.positions.index(example_pos) + 1}\n(edge artifact simulated)",
    )

    im_window = axes[0, 2].imshow(layout.gaussian, cmap="magma", vmin=0.0, vmax=1.0)
    _style_image_axis(axes[0, 2], f"Gaussian weight window\nsigma ratio = {sigma_ratio:.2f}")
    fig.colorbar(im_window, ax=axes[0, 2], fraction=0.046, pad=0.03)

    im_sum = axes[0, 3].imshow(gaussian_weights, cmap="viridis")
    _style_image_axis(axes[0, 3], "Accumulated Gaussian weights")
    fig.colorbar(im_sum, ax=axes[0, 3], fraction=0.046, pad=0.03)

    top, left, size = _select_zoom_box(layout)
    _show_rgb(
        axes[1, 0],
        uniform,
        f"Uniform blending\nMAE = {uniform_mae:.4f}",
    )
    axes[1, 0].add_patch(
        Rectangle((left, top), size, size, fill=False, edgecolor="#FFD43B", linewidth=2.0)
    )
    _show_rgb(
        axes[1, 1],
        gaussian,
        f"Gaussian blending\nMAE = {gaussian_mae:.4f}",
    )
    axes[1, 1].add_patch(
        Rectangle((left, top), size, size, fill=False, edgecolor="#FFD43B", linewidth=2.0)
    )

    divider = np.ones((size, 8, 3), dtype=np.float32)
    zoom_compare = np.concatenate(
        [
            uniform[top:top + size, left:left + size],
            divider,
            gaussian[top:top + size, left:left + size],
        ],
        axis=1,
    )
    axes[1, 2].imshow(zoom_compare)
    _style_image_axis(axes[1, 2], "Interior seam close-up\nUniform  |  Gaussian")
    axes[1, 2].text(
        0.25,
        0.04,
        "Uniform",
        transform=axes[1, 2].transAxes,
        ha="center",
        color="white",
        fontsize=9,
        bbox={"facecolor": "#222222", "alpha": 0.65, "edgecolor": "none"},
    )
    axes[1, 2].text(
        0.75,
        0.04,
        "Gaussian",
        transform=axes[1, 2].transAxes,
        ha="center",
        color="white",
        fontsize=9,
        bbox={"facecolor": "#222222", "alpha": 0.65, "edgecolor": "none"},
    )

    axes[1, 3].axis("off")
    axes[1, 3].text(
        0.02,
        0.90,
        "Weighted reconstruction",
        transform=axes[1, 3].transAxes,
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 3].text(
        0.02,
        0.69,
        r"$P(x) = \frac{\sum_i w_i(x)\,P_i(x)}{\sum_i w_i(x)}$",
        transform=axes[1, 3].transAxes,
        fontsize=18,
        color="#1C3D5A",
    )
    axes[1, 3].text(
        0.02,
        0.51,
        "High confidence near tile centers;\n"
        "lower influence near overlap boundaries.",
        transform=axes[1, 3].transAxes,
        fontsize=10,
        linespacing=1.5,
        color="#333333",
    )
    axes[1, 3].text(
        0.02,
        0.31,
        f"Simulated seam error reduction: {improvement:.1f}%\n"
        f"Artifact strength: {artifact_strength:.2f}",
        transform=axes[1, 3].transAxes,
        fontsize=10.5,
        color="#087F5B",
        fontweight="bold",
        linespacing=1.5,
    )
    axes[1, 3].text(
        0.02,
        0.07,
        "Illustration only: colored edge artifacts are\n"
        "simulated uncertainty, not model predictions.",
        transform=axes[1, 3].transAxes,
        fontsize=9,
        color="#666666",
        style="italic",
        linespacing=1.4,
    )
    fig.text(
        0.5,
        0.025,
        "The tile layout and Gaussian window match predictor/tiling.py; only the illustrative edge artifacts are synthesized.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a publication-style visualization of tiled Gaussian blending."
    )
    parser.add_argument("--image_path", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sigma_ratio",
        type=float,
        default=0.25,
        help="Gaussian sigma relative to patch size; production default is 0.25.",
    )
    parser.add_argument(
        "--artifact_strength",
        type=float,
        default=0.20,
        help="Magnitude of simulated internal tile-edge color uncertainty.",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sigma_ratio <= 0:
        raise SystemExit("--sigma_ratio must be positive.")
    if args.artifact_strength < 0:
        raise SystemExit("--artifact_strength must be non-negative.")
    if not args.image_path.exists():
        raise SystemExit(f"Input image does not exist: {args.image_path}")

    image = np.asarray(Image.open(args.image_path).convert("RGB"), dtype=np.uint8)
    try:
        layout = build_layout(image, sigma_ratio=args.sigma_ratio)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    uniform, gaussian, weights, example_tile, example_pos = reconstruct_demo(
        layout, artifact_strength=args.artifact_strength, seed=args.seed
    )
    fig = build_figure(
        layout,
        uniform,
        gaussian,
        weights,
        example_tile,
        example_pos,
        sigma_ratio=args.sigma_ratio,
        artifact_strength=args.artifact_strength,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension, dpi in EXPORTS:
        output_path = args.output_dir / f"gaussian_blending_process.{extension}"
        fig.savefig(output_path, bbox_inches="tight", facecolor="white", dpi=dpi)
        print(f"saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
